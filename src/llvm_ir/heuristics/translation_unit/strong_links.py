"""Strong-link beam segment-tree.

"Strong links" are contiguous pass subsequences (substrings, length >= 1) that
occur in more than one function's best sequence; their weight is the number of
functions containing them. They are the leaves of a balanced binary tree where
each node keeps a beam of the top-K sequences (by measured .text best-prefix).

Merge is graph-guided: from the left/right beams we form concatenations l+r and
r+l only for pairs whose junction last(l)->first(r) is an edge in the pass-order
graph (ranked by edge weight), measure the most promising ones on the TU, and
keep the node's top-K. The root beam's best sequence is the answer.
"""

from __future__ import annotations

from collections.abc import Callable

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult


def mine_strong_links(
    function_results: list[FunctionPassResult],
    *,
    min_support: int = 2,
    max_links: int = 0,
    positives_only: bool = True,
) -> list[tuple[tuple[str, ...], int]]:
    """Contiguous subsequences in >= min_support functions, weighted by function count."""
    support: dict[tuple[str, ...], set[int]] = {}
    for index, result in enumerate(function_results):
        if positives_only and result.delta <= 0:
            continue
        seq = tuple(result.passes)
        substrings: set[tuple[str, ...]] = set()
        for i in range(len(seq)):
            for j in range(i + 1, len(seq) + 1):
                substrings.add(seq[i:j])
        for sub in substrings:
            support.setdefault(sub, set()).add(index)
    links = [
        (sub, len(funcs))
        for sub, funcs in support.items()
        if len(funcs) >= min_support
    ]
    links.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    if max_links and max_links > 0:
        links = links[:max_links]
    return links


# measure(passes) -> (best_size, best_passes); edge_weight(u, v) -> int
MeasureFn = Callable[[tuple[str, ...]], tuple[int, tuple[str, ...]]]
EdgeWeightFn = Callable[[str, str], int]


def enumerate_all_orderings(
    passes: list[str],
    *,
    max_length: int,
    cap: int,
) -> list[tuple[str, ...]] | None:
    """All simple (no-repeat) ordered subsequences of ``passes``, length 1..max_length.

    This is the exact "try every subsequence" set for a tiny pass alphabet. Returns
    ``None`` (bail) if the count would exceed ``cap`` -- the caller then falls back
    to the beam tree, since exhaustive enumeration is only tractable for tiny graphs.
    """
    items = sorted(set(passes))
    limit = min(max_length, len(items))
    result: list[tuple[str, ...]] = []

    def dfs(prefix: list[str], used: set[str]) -> bool:
        for p in items:
            if p in used:
                continue
            seq = prefix + [p]
            result.append(tuple(seq))
            if len(result) > cap:
                return False
            if len(seq) < limit:
                if not dfs(seq, used | {p}):
                    return False
        return True

    if not dfs([], set()):
        return None
    return result


def beam_segment_tree_merge(
    leaves: list[list[tuple[tuple[str, ...], int]]],
    edge_weight: EdgeWeightFn,
    measure: MeasureFn,
    *,
    beam: int = 24,
    concat_cap: int = 48,
    max_length: int = 16,
) -> tuple[tuple[tuple[str, ...], int], int]:
    """Bottom-up beam merge. Returns ((best_seq, best_size), eval_count)."""
    nodes = [list(leaf) for leaf in leaves if leaf]
    if not nodes:
        return ((), None), 0  # no leaves -> caller falls back to baseline
    eval_count = 0

    def top(cands: list[tuple[tuple[str, ...], int]]) -> list[tuple[tuple[str, ...], int]]:
        best: dict[tuple[str, ...], int] = {}
        for seq, size in cands:
            if seq not in best or size < best[seq]:
                best[seq] = size
        return sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:beam]

    while len(nodes) > 1:
        next_level = []
        i = 0
        n = len(nodes)
        while i < n:
            if i + 1 >= n:
                next_level.append(nodes[i])
                break
            left, right = nodes[i], nodes[i + 1]
            candidates = list(left) + list(right)
            seen = {seq for seq, _ in candidates}
            # Graph-guided concatenations first (junction is a real order-graph
            # edge, ranked by edge weight), then fill the budget with the
            # remaining concatenations ranked by combined child size (cheapest
            # first) -- this explores more subsequences when edges are sparse.
            # Edge-backed concatenations ranked by edge weight (desc); the rest
            # ranked by combined child quality (smallest total .text first), so
            # the budget is spent merging the best sub-results.
            edge_pairs: list[tuple[int, tuple[str, ...]]] = []
            extra_pairs: list[tuple[int, tuple[str, ...]]] = []
            for lseq, lsz in left:
                for rseq, rsz in right:
                    if not (lseq and rseq):
                        continue
                    lr, rl = lseq + rseq, rseq + lseq
                    w1 = edge_weight(lseq[-1], rseq[0])
                    if w1 > 0:
                        edge_pairs.append((w1, lr))
                    else:
                        extra_pairs.append((lsz + rsz, lr))
                    w2 = edge_weight(rseq[-1], lseq[0])
                    if w2 > 0:
                        edge_pairs.append((w2, rl))
                    else:
                        extra_pairs.append((lsz + rsz, rl))
            edge_pairs.sort(key=lambda item: -item[0])
            extra_pairs.sort(key=lambda item: item[0])  # best (smallest) children first
            ordered: list[tuple[str, ...]] = []
            queued: set[tuple[str, ...]] = set()
            for _k, combo in edge_pairs:
                if combo not in queued:
                    queued.add(combo)
                    ordered.append(combo)
            for _k, combo in extra_pairs:
                if combo not in queued:
                    queued.add(combo)
                    ordered.append(combo)
            cap = concat_cap if concat_cap > 0 else len(ordered)
            for combo in ordered[:cap]:
                combo = tuple(combo[:max_length])
                if not combo or combo in seen:
                    continue
                seen.add(combo)
                best_size, best_passes = measure(combo)
                eval_count += 1
                candidates.append((tuple(best_passes), int(best_size)))
            next_level.append(top(candidates))
            i += 2
        nodes = next_level
    root = nodes[0]
    return (root[0] if root else ((), 0)), eval_count
