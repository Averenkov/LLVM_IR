"""Consensus subsequence heuristic: frequent gapped patterns + purposeful noise.

The signal is the *largest common subsequences* of the per-function best pass
sequences -- long, order-preserving but non-contiguous patterns that many
functions agree on (mined as frequent gapped sequential patterns, PrefixSpan
style). These consensus skeletons form a reliable backbone.

Because the per-function signal structurally under-represents interprocedural
passes (they are near no-ops on isolated functions), pure consensus is blind to
exactly the passes that matter most on the whole TU. So the noise is purposeful:
candidates are stochastic realizations of a skeleton where the gaps are filled
with passes drawn from an *exploration distribution* orthogonal to per-function
support (start frequency + catalog floor). Consensus gives the backbone; noise
explores the TU-only opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult


@dataclass(frozen=True)
class ConsensusConfig:
    minsup: int = 2
    max_pattern_length: int = 8
    beam: int = 200
    top_skeletons: int = 60
    max_length: int = 12
    epsilon_insert: float = 0.25
    epsilon_drop: float = 0.1
    swap_prob: float = 0.05
    pool_size: int = 5000
    walk_seed: int = 7
    filler_floor: float = 1.0


@dataclass(frozen=True)
class Skeleton:
    passes: tuple[str, ...]
    support: int
    weight: float


def _positives(function_results: list[FunctionPassResult]):
    seqs = []
    for result in function_results:
        if result.delta > 0 and result.passes:
            seqs.append((tuple(result.passes), float(result.delta)))
    return seqs


def mine_frequent_subsequences(
    function_results: list[FunctionPassResult],
    *,
    config: ConsensusConfig,
) -> list[Skeleton]:
    """Mine frequent gapped subsequences (PrefixSpan-lite) over positive functions.

    Support of a pattern = number of functions containing it as an (ordered,
    gapped) subsequence; weight = sum of those functions' deltas.
    """
    seqs = _positives(function_results)
    if not seqs:
        return []
    deltas = [delta for _seq, delta in seqs]
    sequences = [seq for seq, _delta in seqs]

    skeletons: list[Skeleton] = []
    # A projection entry is (seq_index, suffix_start). One entry per sequence.
    frontier: list[tuple[tuple[str, ...], list[tuple[int, int]]]] = [
        ((), [(i, 0) for i in range(len(sequences))])
    ]
    while frontier:
        next_frontier: list[tuple[tuple[str, ...], list[tuple[int, int]], float]] = []
        for pattern, projection in frontier:
            item_proj: dict[str, list[tuple[int, int]]] = {}
            item_seqs: dict[str, set[int]] = {}
            for seq_index, start in projection:
                seq = sequences[seq_index]
                seen: set[str] = set()
                for position in range(start, len(seq)):
                    item = seq[position]
                    if item in seen:
                        continue
                    seen.add(item)
                    item_proj.setdefault(item, []).append((seq_index, position + 1))
                    item_seqs.setdefault(item, set()).add(seq_index)
            for item, support_seqs in item_seqs.items():
                if len(support_seqs) < config.minsup:
                    continue
                new_pattern = pattern + (item,)
                weight = sum(deltas[i] for i in support_seqs)
                if len(new_pattern) >= 2:
                    skeletons.append(Skeleton(new_pattern, len(support_seqs), weight))
                if len(new_pattern) < config.max_pattern_length:
                    next_frontier.append((new_pattern, item_proj[item], weight))
        next_frontier.sort(key=lambda item: (-item[2], item[0]))
        frontier = [(p, proj) for p, proj, _w in next_frontier[: config.beam]]

    skeletons.sort(key=lambda s: (-len(s.passes), -s.weight, -s.support, s.passes))
    return skeletons[: config.top_skeletons]


def exploration_distribution(
    function_results: list[FunctionPassResult],
    start_counts: dict[str, int],
    *,
    filler_floor: float,
) -> dict[str, float]:
    """Filler weights orthogonal to subsequence support: start frequency + floor.

    Every pass in the catalog is reachable (floor), but passes that often *start*
    good sequences (including interprocedural ones like iroutliner) are favored.
    """
    catalog = {p for result in function_results for p in result.passes}
    return {p: float(start_counts.get(p, 0)) + filler_floor for p in sorted(catalog)}


def _sample_weighted(weights: dict, rng: Random):
    items = sorted(weights.items())
    total = sum(w for _k, w in items)
    if total <= 0:
        return items[0][0] if items else None
    cursor = rng.random() * total
    for key, weight in items:
        cursor -= weight
        if cursor <= 0:
            return key
    return items[-1][0]


def generate_candidate_pool(
    skeletons: list[Skeleton],
    exploration: dict[str, float],
    *,
    config: ConsensusConfig,
    rng: Random,
) -> list[tuple[str, ...]]:
    """Stochastic realizations of skeletons with noise-filled gaps."""
    if not skeletons:
        return []
    skeleton_weights = {index: max(s.weight, 1.0) for index, s in enumerate(skeletons)}
    pool: dict[tuple[str, ...], None] = {}
    max_length = config.max_length
    for _ in range(config.pool_size):
        sk_index = _sample_weighted(skeleton_weights, rng)
        if sk_index is None:
            break
        skeleton = skeletons[sk_index]
        candidate: list[str] = []
        for element in skeleton.passes:
            if len(candidate) >= max_length:
                break
            while rng.random() < config.epsilon_insert and len(candidate) < max_length:
                filler = _sample_weighted(exploration, rng)
                if filler is not None:
                    candidate.append(filler)
            if rng.random() < config.epsilon_drop:
                continue
            candidate.append(element)
        while rng.random() < config.epsilon_insert and len(candidate) < max_length:
            filler = _sample_weighted(exploration, rng)
            if filler is not None:
                candidate.append(filler)
        # optional adjacent swap noise
        if len(candidate) >= 2 and rng.random() < config.swap_prob:
            i = rng.randrange(len(candidate) - 1)
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
        candidate = candidate[:max_length]
        if candidate:
            pool[tuple(candidate)] = None
    return sorted(pool)
