"""Chunk-Forest candidate generation for translation-unit pass ordering."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Literal

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult
from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph
from llvm_ir.stages.translation_unit.path_scoring import node_priority

ChunkKind = Literal["mined", "macro", "single"]


@dataclass(frozen=True)
class Chunk:
    passes: tuple[str, ...]
    weight: float
    kind: ChunkKind
    support: int


@dataclass(frozen=True)
class ChunkRef:
    chunk_index: int
    start: int
    end: int
    partial: bool = False


@dataclass(frozen=True)
class CandidatePath:
    passes: tuple[str, ...]
    chunks: tuple[ChunkRef, ...]
    score: float = 0.0
    new_nodes: int = 0


@dataclass(frozen=True)
class ChunkInventoryConfig:
    ngram_max: int = 4
    closure_theta: float = 0.8
    min_support: int = 2
    top_chunks: int = 30
    macro_top: int = 3
    singles: int = 12
    ngrams_per_bucket: int = 5


@dataclass(frozen=True)
class ChunkWalkConfig:
    pool_size: int = 100_000
    walk_seed: int = 7
    max_length: int = 12
    teleport: float = 0.15
    walk_stall: int = 20_000


@dataclass(frozen=True)
class ChunkSelectionConfig:
    paths: int = 250
    lambda_cache: float = 0.0
    gamma_diversity: float = 0.5
    max_real_evals: int = 0


@dataclass(frozen=True)
class ChunkGraph:
    chunks: tuple[Chunk, ...]
    start_weights: dict[int, float]
    edges: dict[int, dict[int, float]]
    overlaps: dict[tuple[int, int], int]


@dataclass(frozen=True)
class CandidatePoolResult:
    paths: list[CandidatePath]
    walks_executed: int


def positive_function_results(results: list[FunctionPassResult]) -> list[FunctionPassResult]:
    return [result for result in results if result.delta > 0 and result.passes]


def mine_chunks(
    results: list[FunctionPassResult],
    graph: PassOrderGraph | None = None,
    *,
    config: ChunkInventoryConfig = ChunkInventoryConfig(),
) -> list[Chunk]:
    positives = positive_function_results(results)
    core_weights, core_supports = _raw_ngram_stats(positives, config.ngram_max)
    closed_chunks: list[Chunk] = []
    if positives and len(positives) >= config.min_support:
        closed_by_passes: dict[tuple[str, ...], Chunk] = {}
        for core, core_weight in sorted(core_weights.items(), key=lambda item: (-item[1], item[0])):
            if core_supports[core] < config.min_support:
                continue
            closed = _close_chunk(core, positives, core_weight, config.closure_theta)
            weight, support = _chain_weight_and_support(closed, positives)
            if support < config.min_support:
                continue
            chunk = Chunk(closed, weight, "mined", support)
            previous = closed_by_passes.get(closed)
            if previous is None or chunk.weight > previous.weight:
                closed_by_passes[closed] = chunk
        closed_chunks = sorted(
            closed_by_passes.values(),
            key=lambda chunk: (-chunk.weight, -chunk.support, chunk.passes),
        )

    raw_bucket_chunks = _raw_ngram_bucket_chunks(
        core_weights,
        core_supports,
        protected_chunks=closed_chunks,
        config=config,
    )
    mined = sorted(
        _dedupe_chunks(closed_chunks + raw_bucket_chunks),
        key=lambda chunk: (-chunk.weight, -chunk.support, chunk.passes),
    )[: config.top_chunks]
    macros = [
        Chunk(tuple(result.passes), float(result.delta), "macro", 1)
        for result in sorted(positives, key=lambda item: (-item.delta, item.function))[: config.macro_top]
        if result.passes
    ]
    singles = _single_chunks(positives, graph, count=config.singles)
    return _dedupe_chunks(mined + macros + singles)


def _raw_ngram_stats(
    results: list[FunctionPassResult],
    ngram_max: int,
) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], int]]:
    weights: dict[tuple[str, ...], float] = defaultdict(float)
    supports: dict[tuple[str, ...], int] = defaultdict(int)
    for result in results:
        seen: set[tuple[str, ...]] = set()
        max_ngram = min(ngram_max, len(result.passes))
        for length in range(2, max_ngram + 1):
            for start in range(0, len(result.passes) - length + 1):
                seen.add(tuple(result.passes[start : start + length]))
        for core in seen:
            weights[core] += result.delta
            supports[core] += 1
    return dict(weights), dict(supports)


def _raw_ngram_bucket_chunks(
    weights: dict[tuple[str, ...], float],
    supports: dict[tuple[str, ...], int],
    *,
    protected_chunks: list[Chunk],
    config: ChunkInventoryConfig,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for length in range(2, config.ngram_max + 1):
        candidates = [
            Chunk(passes, weights[passes], "mined", supports[passes])
            for passes in weights
            if len(passes) == length and supports[passes] >= config.min_support
        ]
        filtered = [
            chunk
            for chunk in candidates
            if not _is_subsequence_covered(chunk, protected_chunks + chunks)
        ]
        chunks.extend(
            sorted(filtered, key=lambda chunk: (-chunk.weight, -chunk.support, chunk.passes))[
                : config.ngrams_per_bucket
            ]
        )
    return chunks


def _is_subsequence_covered(candidate: Chunk, chunks: list[Chunk]) -> bool:
    for chunk in chunks:
        if len(chunk.passes) < len(candidate.passes) or chunk.weight <= candidate.weight:
            continue
        if _find_occurrences(chunk.passes, candidate.passes):
            return True
    return False


def _close_chunk(
    core: tuple[str, ...],
    results: list[FunctionPassResult],
    current_weight: float,
    theta: float,
) -> tuple[str, ...]:
    chain = core
    while True:
        extensions: dict[tuple[str, ...], None] = {}
        for result in results:
            for start in _find_occurrences(result.passes, chain):
                if start > 0:
                    extensions[(result.passes[start - 1],) + chain] = None
                end = start + len(chain)
                if end < len(result.passes):
                    extensions[chain + (result.passes[end],)] = None
        best: tuple[str, ...] | None = None
        best_weight = float("-inf")
        for candidate in extensions:
            weight, _support = _chain_weight_and_support(candidate, results)
            if weight > best_weight or (weight == best_weight and candidate < (best or candidate)):
                best = candidate
                best_weight = weight
        if best is None or best_weight < current_weight * theta:
            return chain
        chain = best
        current_weight = best_weight


def _chain_weight_and_support(
    chain: tuple[str, ...],
    results: list[FunctionPassResult],
) -> tuple[float, int]:
    weight = 0.0
    support = 0
    for result in results:
        if _find_occurrences(result.passes, chain):
            weight += result.delta
            support += 1
    return weight, support


def _find_occurrences(passes: list[str] | tuple[str, ...], needle: tuple[str, ...]) -> list[int]:
    if not needle or len(needle) > len(passes):
        return []
    width = len(needle)
    return [
        index
        for index in range(0, len(passes) - width + 1)
        if tuple(passes[index : index + width]) == needle
    ]


def _single_chunks(
    results: list[FunctionPassResult],
    graph: PassOrderGraph | None,
    *,
    count: int,
) -> list[Chunk]:
    pass_weight: dict[str, float] = defaultdict(float)
    pass_support: dict[str, int] = defaultdict(int)
    for result in results:
        for pass_name in set(result.passes):
            pass_weight[pass_name] += result.delta
            pass_support[pass_name] += 1
    if graph is not None:
        for node in graph.nodes:
            pass_weight.setdefault(node, max(float(node_priority(graph, node)), 0.0))
            pass_support.setdefault(node, 0)
    ranked = sorted(
        pass_weight,
        key=lambda node: (
            -(pass_weight[node] + max(node_priority(graph, node), 0) if graph else pass_weight[node]),
            node,
        ),
    )
    return [
        Chunk((pass_name,), max(pass_weight[pass_name], 1.0), "single", pass_support[pass_name])
        for pass_name in ranked[: max(0, count)]
    ]


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    rank = {"mined": 0, "macro": 1, "single": 2}
    by_passes: dict[tuple[str, ...], Chunk] = {}
    for chunk in chunks:
        previous = by_passes.get(chunk.passes)
        if previous is None or (chunk.weight, -rank[chunk.kind]) > (previous.weight, -rank[previous.kind]):
            by_passes[chunk.passes] = chunk
    return sorted(by_passes.values(), key=lambda chunk: (rank[chunk.kind], -chunk.weight, chunk.passes))


def build_chunk_graph(
    chunks: list[Chunk],
    results: list[FunctionPassResult],
    graph: PassOrderGraph | None = None,
    *,
    splice_epsilon: float = 0.05,
    start_floor: float = 0.1,
) -> ChunkGraph:
    """Build a chunk graph with observed, splice, and normalized glue edges.

    Observed and splice weights are function-level delta supports. Glue edges use
    pass-level order-graph weights scaled by the ratio of median existing chunk
    edge weight to median used pass-edge weight, so the fallback remains on the
    same functional-weight scale instead of mixing in measured TU bytes.
    """
    positives = positive_function_results(results)
    start_weights: dict[int, float] = defaultdict(float)
    edge_weights: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    overlaps: dict[tuple[int, int], int] = {}
    for result in positives:
        occurrences: dict[int, list[int]] = {}
        for index, chunk in enumerate(chunks):
            starts = _find_occurrences(result.passes, chunk.passes)
            if starts:
                occurrences[index] = starts
                if any(start <= 1 for start in starts):
                    start_weights[index] += result.delta
        for left_index, left_starts in occurrences.items():
            left_len = len(chunks[left_index].passes)
            for right_index, right_starts in occurrences.items():
                if left_index == right_index:
                    continue
                found = False
                for left_start in left_starts:
                    left_end = left_start + left_len
                    if any(0 <= right_start - left_end <= 2 for right_start in right_starts):
                        found = True
                        break
                if found:
                    edge_weights[left_index][right_index] += result.delta
                    overlaps.setdefault((left_index, right_index), 0)

    for left_index, left in enumerate(chunks):
        for right_index, right in enumerate(chunks):
            if left_index == right_index or edge_weights[left_index].get(right_index, 0) > 0:
                continue
            overlap = _max_splice_overlap(left.passes, right.passes)
            if overlap <= 0:
                continue
            merged = left.passes + right.passes[overlap:]
            weight, _support = _chain_weight_and_support(merged, positives)
            if weight <= 0 and splice_epsilon > 0:
                weight = splice_epsilon * min(left.weight, right.weight)
            if weight > 0:
                edge_weights[left_index][right_index] = weight
                overlaps[(left_index, right_index)] = overlap

    if graph is not None:
        _add_order_graph_glue_edges(chunks, graph, edge_weights, overlaps)

    for index, chunk in enumerate(chunks):
        floor = start_floor * chunk.weight
        start_weights[index] = max(start_weights.get(index, 0.0), floor)
    return ChunkGraph(
        chunks=tuple(chunks),
        start_weights=dict(start_weights),
        edges={source: dict(targets) for source, targets in edge_weights.items()},
        overlaps=dict(overlaps),
    )


def _max_splice_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    max_overlap = min(len(left), len(right) - 1)
    for width in range(max_overlap, 0, -1):
        if left[-width:] == right[:width]:
            return width
    return 0


def _add_order_graph_glue_edges(
    chunks: list[Chunk],
    graph: PassOrderGraph,
    edge_weights: dict[int, dict[int, float]],
    overlaps: dict[tuple[int, int], int],
) -> None:
    existing = [weight for targets in edge_weights.values() for weight in targets.values() if weight > 0]
    pass_edges: list[int] = []
    pairs: list[tuple[int, int, int]] = []
    for left_index, left in enumerate(chunks):
        for right_index, right in enumerate(chunks):
            if left_index == right_index or edge_weights[left_index].get(right_index, 0) > 0:
                continue
            weight = graph.edge_counts.get((left.passes[-1], right.passes[0]), 0)
            if weight > 0:
                pass_edges.append(weight)
                pairs.append((left_index, right_index, weight))
    if not pairs:
        return
    glue_scale = (median(existing) / median(pass_edges)) if existing and pass_edges else 1.0
    for left_index, right_index, weight in pairs:
        edge_weights[left_index][right_index] = weight * glue_scale
        overlaps[(left_index, right_index)] = 0


def generate_candidate_pool(
    chunk_graph: ChunkGraph,
    *,
    config: ChunkWalkConfig = ChunkWalkConfig(),
) -> list[CandidatePath]:
    return generate_candidate_pool_with_stats(chunk_graph, config=config).paths


def generate_candidate_pool_with_stats(
    chunk_graph: ChunkGraph,
    *,
    config: ChunkWalkConfig = ChunkWalkConfig(),
) -> CandidatePoolResult:
    if not chunk_graph.chunks or config.pool_size <= 0 or config.max_length <= 0:
        return CandidatePoolResult(paths=[], walks_executed=0)
    rng = random.Random(config.walk_seed)
    unique: dict[tuple[str, ...], CandidatePath] = {}
    stalls = 0
    walks_executed = 0
    for walk_index in range(config.pool_size):
        walks_executed = walk_index + 1
        start_weights = _teleport_weights(chunk_graph, set()) if config.teleport > 0 else chunk_graph.start_weights
        start = _weighted_choice(start_weights, rng)
        if start is None:
            break
        used = {start}
        chunk_indexes = [start]
        candidate = _candidate_from_chunks(chunk_graph, chunk_indexes, config.max_length)
        current = start
        while len(candidate.passes) < config.max_length and len(used) < len(chunk_graph.chunks):
            outgoing = {
                index: weight
                for index, weight in chunk_graph.edges.get(current, {}).items()
                if index not in used and weight > 0
            }
            if not outgoing and config.teleport <= 0:
                break
            use_teleport = not outgoing or rng.random() < config.teleport
            if use_teleport:
                next_index = _weighted_choice(_teleport_weights(chunk_graph, used), rng)
            else:
                next_index = _weighted_choice(outgoing, rng)
            if next_index is None:
                break
            chunk_indexes.append(next_index)
            used.add(next_index)
            current = next_index
            candidate = _candidate_from_chunks(chunk_graph, chunk_indexes, config.max_length)
        if candidate.passes and candidate.passes not in unique:
            unique[candidate.passes] = candidate
            stalls = 0
        else:
            stalls += 1
        if config.walk_stall > 0 and stalls >= config.walk_stall:
            break
    paths = sorted(unique.values(), key=lambda candidate: (candidate.passes, len(candidate.chunks)))
    return CandidatePoolResult(paths=paths, walks_executed=walks_executed)


def _teleport_weights(chunk_graph: ChunkGraph, used: set[int]) -> dict[int, float]:
    return {
        index: max(chunk_graph.start_weights.get(index, 0.0), chunk.weight)
        for index, chunk in enumerate(chunk_graph.chunks)
        if index not in used
    }


def _candidate_from_chunks(
    chunk_graph: ChunkGraph,
    chunk_indexes: list[int],
    max_length: int,
) -> CandidatePath:
    passes: list[str] = []
    refs: list[ChunkRef] = []
    previous_index: int | None = None
    for chunk_index in chunk_indexes:
        chunk_passes = chunk_graph.chunks[chunk_index].passes
        overlap = chunk_graph.overlaps.get((previous_index, chunk_index), 0) if previous_index is not None else 0
        append_passes = chunk_passes[overlap:]
        if len(passes) >= max_length or not append_passes:
            break
        start = len(passes)
        remaining = max_length - len(passes)
        taken = tuple(append_passes[:remaining])
        passes.extend(taken)
        end = len(passes)
        refs.append(ChunkRef(chunk_index, start, end, len(taken) < len(append_passes)))
        previous_index = chunk_index
        if len(taken) < len(append_passes):
            break
    return CandidatePath(tuple(passes), tuple(refs))


def _weighted_choice(weights: dict[int, float], rng: random.Random) -> int | None:
    positive = [(index, weight) for index, weight in sorted(weights.items()) if weight > 0]
    if not positive:
        return None
    total = sum(weight for _index, weight in positive)
    cursor = rng.random() * total
    for index, weight in positive:
        cursor -= weight
        if cursor <= 0:
            return index
    return positive[-1][0]


def select_paths(
    pool: list[CandidatePath],
    chunks: tuple[Chunk, ...],
    values: dict[int, float],
    *,
    config: ChunkSelectionConfig,
    initial_trie: set[tuple[str, ...]] | None = None,
    initial_picked: Counter[int] | None = None,
    excluded_paths: set[tuple[str, ...]] | None = None,
) -> tuple[list[CandidatePath], set[tuple[str, ...]], Counter[int]]:
    selected: list[CandidatePath] = []
    trie = set(initial_trie or {()})
    picked = Counter(initial_picked or {})
    excluded = excluded_paths or set()
    lambda_cache = config.lambda_cache
    if lambda_cache == 0.0:
        lambda_cache = auto_lambda(chunks, values)
    remaining = [candidate for candidate in pool if candidate.passes not in excluded]
    while remaining and len(selected) < config.paths:
        best_index = -1
        best_gain = float("-inf")
        best_raw_gain = float("-inf")
        best_path: tuple[str, ...] | None = None
        best_nodes = 0
        for index, candidate in enumerate(remaining):
            new_nodes = count_new_prefix_nodes(candidate.passes, trie)
            if config.max_real_evals and len(trie) - 1 + new_nodes > config.max_real_evals:
                continue
            raw_gain = 0.0
            for ref in candidate.chunks:
                if ref.partial:
                    continue
                raw_gain += values.get(ref.chunk_index, 0.0) * (config.gamma_diversity ** picked[ref.chunk_index])
            gain = raw_gain - lambda_cache * new_nodes
            if (
                best_path is None
                or gain > best_gain
                or (gain == best_gain and raw_gain > best_raw_gain)
                or (gain == best_gain and raw_gain == best_raw_gain and candidate.passes < best_path)
            ):
                best_gain = gain
                best_raw_gain = raw_gain
                best_path = candidate.passes
                best_index = index
                best_nodes = new_nodes
        if best_index < 0:
            break
        candidate = remaining.pop(best_index)
        selected_candidate = CandidatePath(candidate.passes, candidate.chunks, best_gain, best_nodes)
        selected.append(selected_candidate)
        for ref in candidate.chunks:
            if not ref.partial:
                picked[ref.chunk_index] += 1
        add_prefixes(candidate.passes, trie)
    return selected, trie, picked


def auto_lambda(chunks: tuple[Chunk, ...], values: dict[int, float]) -> float:
    """Calibrate cache penalty as median chunk value divided by average chunk length.

    This keeps one new prefix-node penalty comparable to one pass worth of chunk
    value. It is the only automatic scale used during selection; measured TU-byte
    values and mined function-level weights are normalized by the evaluator before
    they are mixed in wave two.
    """
    if not chunks:
        return 0.0
    chunk_values = [max(values.get(index, chunk.weight), 0.0) for index, chunk in enumerate(chunks)]
    avg_len = sum(len(chunk.passes) for chunk in chunks) / len(chunks)
    return median(chunk_values) / avg_len if avg_len else 0.0


def count_new_prefix_nodes(path: tuple[str, ...], trie: set[tuple[str, ...]]) -> int:
    return sum(1 for index in range(1, len(path) + 1) if path[:index] not in trie)


def add_prefixes(path: tuple[str, ...], trie: set[tuple[str, ...]]) -> None:
    trie.add(())
    for index in range(1, len(path) + 1):
        trie.add(path[:index])


def chunk_kind_counts(chunks: list[Chunk] | tuple[Chunk, ...]) -> dict[str, int]:
    counts = Counter(chunk.kind for chunk in chunks)
    return {
        "chunks_mined": counts.get("mined", 0),
        "chunks_macro": counts.get("macro", 0),
        "chunks_single": counts.get("single", 0),
    }
