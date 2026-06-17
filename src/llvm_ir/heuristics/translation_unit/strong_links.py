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


def beam_segment_tree_merge(
    leaves: list[list[tuple[tuple[str, ...], int]]],
    edge_weight: EdgeWeightFn,
    measure: MeasureFn,
    *,
    beam: int = 10,
    concat_cap: int = 12,
    max_length: int = 16,
) -> tuple[tuple[tuple[str, ...], int], int]:
    """Bottom-up beam merge. Returns ((best_seq, best_size), eval_count)."""
    nodes = [list(leaf) for leaf in leaves if leaf]
    if not nodes:
        return ((), 0), 0
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
            # graph-guided concatenations
            pairs: list[tuple[int, tuple[str, ...]]] = []
            for lseq, _ls in left:
                for rseq, _rs in right:
                    if lseq and rseq:
                        w1 = edge_weight(lseq[-1], rseq[0])
                        if w1 > 0:
                            pairs.append((w1, lseq + rseq))
                        w2 = edge_weight(rseq[-1], lseq[0])
                        if w2 > 0:
                            pairs.append((w2, rseq + lseq))
            if not pairs and left and right:
                # fallback: best-of-each, both orders, even without a graph edge
                a, b = left[0][0], right[0][0]
                pairs = [(0, a + b), (0, b + a)]
            pairs.sort(key=lambda item: -item[0])
            for _w, combo in pairs[:concat_cap]:
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
