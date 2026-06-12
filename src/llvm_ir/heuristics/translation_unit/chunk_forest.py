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
    min_inventory: int = 10


@dataclass(frozen=True)
class ChunkWalkConfig:
    pool_size: int = 100_000
    walk_seed: int = 7
    max_length: int = 12


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


def positive_function_results(results: list[FunctionPassResult]) -> list[FunctionPassResult]:
    return [result for result in results if result.delta > 0 and result.passes]


def mine_chunks(
    results: list[FunctionPassResult],
    graph: PassOrderGraph | None = None,
    *,
    config: ChunkInventoryConfig = ChunkInventoryConfig(),
) -> list[Chunk]:
    positives = positive_function_results(results)
    mined: list[Chunk] = []
    if positives and len(positives) >= config.min_support:
        core_weights: dict[tuple[str, ...], float] = defaultdict(float)
        core_supports: dict[tuple[str, ...], int] = defaultdict(int)
        for result in positives:
            seen: set[tuple[str, ...]] = set()
            max_ngram = min(config.ngram_max, len(result.passes))
            for length in range(2, max_ngram + 1):
                for start in range(0, len(result.passes) - length + 1):
                    seen.add(tuple(result.passes[start : start + length]))
            for core in seen:
                core_weights[core] += result.delta
                core_supports[core] += 1
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
        mined = sorted(
            closed_by_passes.values(),
            key=lambda chunk: (-chunk.weight, -chunk.support, chunk.passes),
        )[: config.top_chunks]

    macros = [
        Chunk(tuple(result.passes), float(result.delta), "macro", 1)
        for result in sorted(positives, key=lambda item: (-item.delta, item.function))[: config.macro_top]
        if result.passes
    ]
    inventory = _dedupe_chunks(mined + macros)
    if len(inventory) < config.min_inventory:
        inventory = _dedupe_chunks(
            inventory
            + _single_chunks(positives, graph, needed=config.min_inventory - len(inventory))
        )
    return inventory


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
    needed: int,
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
        key=lambda node: (-(pass_weight[node] + max(node_priority(graph, node), 0) if graph else pass_weight[node]), node),
    )
    return [
        Chunk((pass_name,), max(pass_weight[pass_name], 1.0), "single", pass_support[pass_name])
        for pass_name in ranked[: max(0, needed)]
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
) -> ChunkGraph:
    positives = positive_function_results(results)
    start_weights: dict[int, float] = defaultdict(float)
    edge_weights: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
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
    if not start_weights:
        for index, chunk in enumerate(chunks):
            start_weights[index] = max(chunk.weight, 1.0)
    return ChunkGraph(
        chunks=tuple(chunks),
        start_weights=dict(start_weights),
        edges={source: dict(targets) for source, targets in edge_weights.items()},
    )


def generate_candidate_pool(
    chunk_graph: ChunkGraph,
    *,
    config: ChunkWalkConfig = ChunkWalkConfig(),
) -> list[CandidatePath]:
    if not chunk_graph.chunks or config.pool_size <= 0 or config.max_length <= 0:
        return []
    rng = random.Random(config.walk_seed)
    unique: dict[tuple[str, ...], CandidatePath] = {}
    for _ in range(config.pool_size):
        start = _weighted_choice(chunk_graph.start_weights, rng)
        if start is None:
            break
        used = {start}
        chunk_indexes = [start]
        current = start
        while True:
            outgoing = {
                index: weight
                for index, weight in chunk_graph.edges.get(current, {}).items()
                if index not in used and weight > 0
            }
            next_index = _weighted_choice(outgoing, rng)
            if next_index is None:
                break
            chunk_indexes.append(next_index)
            used.add(next_index)
            current = next_index
            if sum(len(chunk_graph.chunks[index].passes) for index in chunk_indexes) >= config.max_length:
                break
        candidate = _candidate_from_chunks(chunk_graph.chunks, chunk_indexes, config.max_length)
        if candidate.passes and candidate.passes not in unique:
            unique[candidate.passes] = candidate
    return sorted(unique.values(), key=lambda candidate: (candidate.passes, len(candidate.chunks)))


def _candidate_from_chunks(
    chunks: tuple[Chunk, ...],
    chunk_indexes: list[int],
    max_length: int,
) -> CandidatePath:
    passes: list[str] = []
    refs: list[ChunkRef] = []
    for chunk_index in chunk_indexes:
        chunk_passes = chunks[chunk_index].passes
        if len(passes) >= max_length:
            break
        start = len(passes)
        remaining = max_length - len(passes)
        taken = tuple(chunk_passes[:remaining])
        passes.extend(taken)
        end = len(passes)
        refs.append(ChunkRef(chunk_index, start, end, len(taken) < len(chunk_passes)))
        if len(taken) < len(chunk_passes):
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
        score = best_gain
        selected_candidate = CandidatePath(candidate.passes, candidate.chunks, score, best_nodes)
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
