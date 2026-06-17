"""Segment-tree style hierarchical merge of measured super-vertices.

The super-vertices (measured 4-pass segments, as in measured_superpath) are the
leaves of a balanced binary tree. At every internal node the two children's
sequences ``l`` and ``r`` are combined by trying both orders on the real TU
(``l+r`` and ``r+l``) and keeping the best of ``[l, r, l+r, r+l]`` by measured
``.text`` best-prefix. The root's best sequence is the answer.

To keep sequence lengths bounded (a naive "keep the full concatenation" would
double the length at every level), each node stores the *best prefix* of the
winning candidate; harmful tails are dropped and concatenations are capped at
``max_length``.
"""

from __future__ import annotations

from collections.abc import Callable

# measure(passes) -> (best_size, best_passes); reads/writes the shared TU cache.
MeasureFn = Callable[[tuple[str, ...]], tuple[int, tuple[str, ...]]]


def segment_tree_merge(
    leaves: list[tuple[tuple[str, ...], int]],
    measure: MeasureFn,
    *,
    max_length: int = 16,
) -> tuple[tuple[tuple[str, ...], int], int]:
    """Bottom-up pairwise merge of measured leaves.

    ``leaves`` is a list of ``(best_prefix_sequence, best_size)`` already measured
    on the TU. Returns ``((best_sequence, best_size), eval_count)`` where
    ``eval_count`` is the number of new TU measurements performed during merging.
    """
    nodes: list[tuple[tuple[str, ...], int]] = [(tuple(seq), int(size)) for seq, size in leaves]
    if not nodes:
        return ((), None), 0  # no leaves -> caller falls back to baseline
    eval_count = 0
    while len(nodes) > 1:
        next_level: list[tuple[tuple[str, ...], int]] = []
        index = 0
        count = len(nodes)
        while index < count:
            if index + 1 >= count:
                next_level.append(nodes[index])  # odd tail carries up unchanged
                break
            left_seq, left_size = nodes[index]
            right_seq, right_size = nodes[index + 1]
            candidates = [(left_seq, left_size), (right_seq, right_size)]
            seen = {left_seq, right_seq}
            for combo in (left_seq + right_seq, right_seq + left_seq):
                combo = tuple(combo[:max_length])
                if not combo or combo in seen:
                    continue
                seen.add(combo)
                best_size, best_passes = measure(combo)
                eval_count += 1
                candidates.append((tuple(best_passes), int(best_size)))
            best = min(candidates, key=lambda item: (item[1], item[0]))
            next_level.append(best)
            index += 2
        nodes = next_level
    return nodes[0], eval_count
