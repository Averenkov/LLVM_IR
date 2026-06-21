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


# measure_exact(passes) -> (size, ok); size of applying the EXACT tuple on the TU
# (not best-prefix), with ok=False if any pass in the tuple crashed. Reads/writes
# the shared prefix cache, so extending an already-measured path costs one pass.
ExactFn = Callable[[tuple[str, ...]], tuple[object, bool]]


def _beam_dp_over_order(
    order: tuple[str, ...],
    measure_exact: ExactFn,
    *,
    max_length: int,
    node_beam: int,
    edge_weight=None,
    graph_slots: int = 0,
    graph_tol: float = 0.15,
) -> list[tuple[tuple[str, ...], int]]:
    """Beam search with per-pass include/skip over a fixed pass order.

    Generalises "best prefix": the first two passes are always applied (the
    base), then for each subsequent pass every surviving path may either apply
    it (one new TU measurement) or skip it (free, path unchanged). Only the top
    ``node_beam`` paths by measured ``.text`` survive each step. Returns the
    surviving ``(sequence, size)`` paths (subsequences of ``order``).

    Graph-informed survival (``edge_weight`` + ``graph_slots`` > 0): besides the
    top ``node_beam`` by size, up to ``graph_slots`` extra paths are kept even if
    they are currently *worse*, provided their last pass has a strong order-graph
    edge to the NEXT pass and they are within ``graph_tol`` of the best size.
    This lets an "enabling" pass that temporarily increases ``.text`` survive so
    a strongly-linked following pass can pay off (lose now to win later).
    """
    order = tuple(order)
    if len(order) <= 2:
        size, ok = measure_exact(order)
        return [(order, int(size))] if ok else []
    base = order[:2]
    size, ok = measure_exact(base)
    if not ok:
        return []
    dp: dict[tuple[str, ...], int] = {base: int(size)}
    use_graph = edge_weight is not None and graph_slots > 0
    for idx in range(2, len(order)):
        pass_name = order[idx]
        nxt: dict[tuple[str, ...], int] = dict(dp)  # skip branch: paths unchanged
        for seq in list(dp):
            if len(seq) >= max_length:
                continue
            nseq = seq + (pass_name,)
            nsize, nok = measure_exact(nseq)
            if nok and (nseq not in nxt or int(nsize) < nxt[nseq]):
                nxt[nseq] = int(nsize)
        ranked = sorted(nxt.items(), key=lambda kv: (kv[1], kv[0]))
        keep = ranked[:node_beam]
        if use_graph and idx + 1 < len(order) and len(ranked) > node_beam:
            next_pass = order[idx + 1]
            best_size = ranked[0][1]
            kept = {s for s, _ in keep}
            extra = []
            for seq, sz in ranked[node_beam:]:
                if not seq or seq in kept:
                    continue
                w = edge_weight(seq[-1], next_pass)
                if w > 0 and (graph_tol <= 0 or sz <= best_size * (1.0 + graph_tol)):
                    extra.append((w, sz, seq))
            extra.sort(key=lambda t: (-t[0], t[1], t[2]))
            keep = keep + [(seq, sz) for _w, sz, seq in extra[:graph_slots]]
        dp = dict(keep)
    return list(dp.items())


def segment_tree_beam_merge(
    leaves: list[tuple[tuple[str, ...], int]],
    measure: MeasureFn,
    measure_exact: ExactFn,
    *,
    max_length: int = 16,
    node_beam: int = 4,
    edge_weight=None,
    graph_slots: int = 0,
    graph_tol: float = 0.15,
) -> tuple[tuple[str, ...], int]:
    """Segment-tree merge whose node combine is a beam DP (include/skip).

    Like :func:`segment_tree_merge`, but at every internal node, instead of only
    keeping the best prefix of ``l+r`` / ``r+l``, a width-``node_beam`` beam
    search over each order chooses the best *subsequence*. The node result is the
    best of ``[l, r, best-prefix(l+r), best-prefix(r+l), beam paths...]``, so it
    is never worse than :func:`segment_tree_merge`. Returns ``(best_seq, size)``.
    """
    nodes: list[tuple[tuple[str, ...], int]] = [(tuple(seq), int(size)) for seq, size in leaves]
    if not nodes:
        return (), None
    while len(nodes) > 1:
        next_level: list[tuple[tuple[str, ...], int]] = []
        index = 0
        count = len(nodes)
        while index < count:
            if index + 1 >= count:
                next_level.append(nodes[index])
                break
            left, right = nodes[index], nodes[index + 1]
            candidates: list[tuple[tuple[str, ...], int]] = [left, right]
            for order in (left[0] + right[0], right[0] + left[0]):
                order = tuple(order)
                if not order:
                    continue
                # Best-prefix of the plain concatenation: guarantees this node is
                # never worse than the non-beam segment_tree.
                bp_size, bp_passes = measure(order[:max_length])
                candidates.append((tuple(bp_passes), int(bp_size)))
                candidates.extend(
                    _beam_dp_over_order(
                        order, measure_exact, max_length=max_length, node_beam=node_beam,
                        edge_weight=edge_weight, graph_slots=graph_slots, graph_tol=graph_tol,
                    )
                )
            best = min(candidates, key=lambda item: (item[1], item[0]))
            next_level.append(best)
            index += 2
        nodes = next_level
    return nodes[0]
