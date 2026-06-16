"""Adaptive, learning two-wave measured-superpath heuristic core.

Wave 1 works on a distance-aware count graph: vertex weight = number of function
sequences a pass appears in; edge weight ``p -> q`` = number of sequences where
``p`` precedes ``q`` with closer pairs adding more (the ``count_distance`` mode).
A fraction of the smallest edges is pruned. For every vertex we enumerate all
length-3 paths (3 edges = 4 passes), keep the per-vertex top with the budget
allocated proportionally to vertex weight so the totals come to ~``gen_budget``
paths, sort them, and keep the top ``select_count`` to measure on the TU.

Those measured segments become super-vertices. Wave-2 super-edges are *measured
on the TU*: pairs are sampled biased toward the best super-vertices, the
concatenation is run, and the result both defines the edge and updates the
sampling weights (vertices that chain synergistically are sampled more -- a
bandit-style use of every measurement). Finally the best super-paths of lengths
2..5 are searched over the measured super-graph and measured.

This module holds the pure (LLVM-free) pieces; the measurement loops live in the
evaluator since they need real TU runs.
"""

from __future__ import annotations

from random import Random

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult


def node_support(function_results: list[FunctionPassResult]) -> dict[str, int]:
    """Vertex weight: number of function sequences each pass appears in."""
    support: dict[str, int] = {}
    for result in function_results:
        for pass_name in set(result.passes):
            support[pass_name] = support.get(pass_name, 0) + 1
    return support


def prune_small_edges(edge_counts: dict[tuple[str, str], int], prune_percent: float) -> dict[tuple[str, str], int]:
    """Drop the smallest ``prune_percent`` % of edges by weight."""
    if not edge_counts or prune_percent <= 0:
        return dict(edge_counts)
    items = sorted(edge_counts.items(), key=lambda kv: (kv[1], kv[0]))
    cut = int(len(items) * min(prune_percent, 100.0) / 100.0)
    return dict(items[cut:])


def _adjacency(edges: dict[tuple[str, str], int]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for source, target in sorted(edges):
        adjacency.setdefault(source, []).append(target)
    return adjacency


def vertex_budget_paths(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    support: dict[str, int],
    *,
    total_budget: int,
    path_nodes: int = 4,
    per_vertex_cap: int = 6000,
) -> list[tuple[str, ...]]:
    """Per-vertex top length-(path_nodes-1) paths, budgeted by vertex weight.

    Graceful degradation for small graphs: the target path length is lowered
    from ``path_nodes`` down to 1 (single passes) until a non-empty pool is
    produced, so tiny graphs that admit no 4-node path still yield candidates.
    Returns up to ``total_budget`` paths sorted by graph weight (descending).
    """
    adjacency = _adjacency(edges)
    total_support = sum(support.get(v, 0) for v in nodes) or 1

    for target_nodes in range(max(1, path_nodes), 0, -1):
        weight_of: dict[tuple[str, ...], int] = {}
        selected: dict[tuple[str, ...], None] = {}
        for vertex in sorted(nodes):
            found: list[tuple[int, tuple[str, ...]]] = []
            if target_nodes == 1:
                found.append((0, (vertex,)))
            else:
                explored = 0

                def dfs(path: list[str], seen: set[str], score: int) -> None:
                    nonlocal explored
                    if len(path) == target_nodes:
                        found.append((score, tuple(path)))
                        return
                    for nxt in adjacency.get(path[-1], []):
                        if explored >= per_vertex_cap:
                            return
                        if nxt in seen:
                            continue
                        explored += 1
                        dfs(path + [nxt], seen | {nxt}, score + edges[(path[-1], nxt)])

                dfs([vertex], {vertex}, 0)
            if not found:
                continue
            k_v = max(1, round(total_budget * support.get(vertex, 0) / total_support))
            found.sort(key=lambda sp: (-sp[0], sp[1]))
            for score, path in found[:k_v]:
                if path not in weight_of:
                    weight_of[path] = score
                    selected[path] = None
        if selected:
            return sorted(selected, key=lambda p: (-weight_of[p], p))[: max(1, total_budget)]
    return []


def select_diverse_by_start(
    paths: list[tuple[str, ...]],
    count: int,
) -> list[tuple[str, ...]]:
    """Select ``count`` paths balancing start-diversity and graph weight.

    ``paths`` must already be ordered by graph weight (best first). First one path
    per distinct starting pass is taken (coverage -- this is what surfaces
    interprocedural starts like mergefunc that a pure top-by-weight cut discards),
    then the remainder is filled greedily by weight (quality).
    """
    if count <= 0 or not paths:
        return list(paths[: max(0, count)])
    rank = {path: index for index, path in enumerate(paths)}
    by_start: dict[str, list[tuple[str, ...]]] = {}
    for path in paths:
        by_start.setdefault(path[0], []).append(path)
    selected: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    # Phase A: one path per start, strongest starts first.
    for start in sorted(by_start, key=lambda s: rank[by_start[s][0]]):
        candidate = by_start[start][0]
        selected.append(candidate)
        seen.add(candidate)
        if len(selected) >= count:
            return selected
    # Phase B: fill the rest by global weight.
    for path in paths:
        if path not in seen:
            selected.append(path)
            seen.add(path)
            if len(selected) >= count:
                break
    return selected


def concat_segments(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """Concatenate two segments with a one-pass overlap dedup."""
    add = right[1:] if (left and right and left[-1] == right[0]) else right
    return left + tuple(add)


def sample_index(weights: dict[int, float], rng: Random) -> int | None:
    """Weighted sample of an index (deterministic given the RNG state)."""
    items = sorted(weights.items())
    total = sum(w for _i, w in items)
    if total <= 0:
        return items[0][0] if items else None
    cursor = rng.random() * total
    for index, weight in items:
        cursor -= weight
        if cursor <= 0:
            return index
    return items[-1][0]


def superpaths_by_length(
    segments: list[tuple[str, ...]],
    super_edges: dict[tuple[int, int], float],
    *,
    lengths: tuple[int, ...] = (2, 3, 4, 5),
    per_length: int = 62,
    beam: int = 96,
    max_length: int = 20,
) -> list[tuple[str, ...]]:
    """Beam-search the best super-paths of each segment count in ``lengths``.

    The super-graph contains only measured edges (``super_edges``); a super-path
    score is the sum of its measured edge values. Segments are concatenated with
    one-pass overlap dedup. Returns up to ``per_length`` candidates per length.
    """
    count = len(segments)
    if count == 0 or not super_edges:
        return []
    adjacency: dict[int, list[int]] = {i: [] for i in range(count)}
    for (i, j) in super_edges:
        adjacency[i].append(j)

    collected: dict[int, list[tuple[float, tuple[str, ...]]]] = {length: [] for length in lengths}
    states = [(0.0, 1, i, tuple(segments[i])) for i in range(count)]
    states.sort(key=lambda s: -s[0])
    states = states[:beam]
    max_segments = max(lengths)
    for _depth in range(2, max_segments + 1):
        nxt: list[tuple[float, int, int, tuple[str, ...]]] = []
        for score, cnt, last, passes in states:
            for j in adjacency[last]:
                merged = concat_segments(passes, tuple(segments[j]))
                if len(merged) > max_length:
                    continue
                nxt.append((score + super_edges[(last, j)], cnt + 1, j, merged))
        nxt.sort(key=lambda s: -s[0])
        states = nxt[:beam]
        for score, cnt, _last, passes in states:
            if cnt in collected:
                collected[cnt].append((score, passes))

    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for length in lengths:
        ranked = sorted(set(collected[length]), key=lambda sp: (-sp[0], sp[1]))
        for _score, passes in ranked[: max(1, per_length)]:
            if passes not in seen:
                seen.add(passes)
                out.append(passes)
    return out
