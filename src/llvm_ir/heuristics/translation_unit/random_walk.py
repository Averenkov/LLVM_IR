"""Random-walk path heuristic for pass-order graphs."""

from __future__ import annotations

import random
from dataclasses import dataclass

from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph


@dataclass(frozen=True)
class RandomWalkPathConfig:
    max_length: int = 12
    walks: int = 256
    seed: int = 0
    min_edge_weight: int = 1


def random_walk_path(
    graph: PassOrderGraph,
    *,
    config: RandomWalkPathConfig | None = None,
) -> list[str]:
    """Sample weighted walks and return the best-scoring path found."""
    paths = random_walk_top_paths(graph, config=config, top_k=1)
    return paths[0] if paths else []


def random_walk_top_paths(
    graph: PassOrderGraph,
    *,
    config: RandomWalkPathConfig | None = None,
    top_k: int = 10,
) -> list[list[str]]:
    """Sample weighted walks and return top unique paths by graph score."""
    if config is None:
        config = RandomWalkPathConfig()
    nodes = sorted(graph.nodes)
    if not nodes:
        return []

    max_length = len(nodes) if config.max_length <= 0 else max(1, min(config.max_length, len(nodes)))
    walks = max(1, config.walks)
    adjacent = _adjacency(graph, nodes, min_edge_weight=config.min_edge_weight)
    start_weights = [_start_weight(graph, node, adjacent) for node in nodes]
    rng = random.Random(config.seed)

    candidates: dict[tuple[str, ...], tuple[int, int, int, int, tuple[str, ...]]] = {}
    for _ in range(walks):
        path = _sample_walk(
            nodes,
            adjacent,
            start_weights=start_weights,
            max_length=max_length,
            rng=rng,
        )
        if not path:
            continue
        key = tuple(path)
        candidates[key] = _score_key(graph, path)

    ranked = sorted(
        candidates,
        key=lambda path: (candidates[path], path),
        reverse=True,
    )
    return [list(path) for path in ranked[: max(1, top_k)]]


def _sample_walk(
    nodes: list[str],
    adjacent: dict[str, list[tuple[str, int]]],
    *,
    start_weights: list[int],
    max_length: int,
    rng: random.Random,
) -> list[str]:
    current = _weighted_choice(nodes, start_weights, rng)
    path = [current]
    visited = {current}

    while len(path) < max_length:
        candidates = [
            (target, weight)
            for target, weight in adjacent.get(current, [])
            if target not in visited
        ]
        if not candidates:
            break
        targets = [target for target, _ in candidates]
        weights = [weight for _, weight in candidates]
        current = _weighted_choice(targets, weights, rng)
        path.append(current)
        visited.add(current)

    return path


def _adjacency(
    graph: PassOrderGraph,
    nodes: list[str],
    *,
    min_edge_weight: int,
) -> dict[str, list[tuple[str, int]]]:
    node_set = set(nodes)
    adjacent = {node: [] for node in nodes}
    for (source, target), weight in sorted(graph.edge_counts.items()):
        if (
            source in node_set
            and target in node_set
            and source != target
            and weight >= min_edge_weight
        ):
            adjacent[source].append((target, weight))
    return adjacent


def _start_weight(
    graph: PassOrderGraph,
    node: str,
    adjacent: dict[str, list[tuple[str, int]]],
) -> int:
    outgoing = sum(weight for _, weight in adjacent.get(node, []))
    incoming = sum(
        weight
        for (source, target), weight in graph.edge_counts.items()
        if source != target and target == node
    )
    return max(1, outgoing - incoming + 1)


def _weighted_choice(
    items: list[str],
    weights: list[int],
    rng: random.Random,
) -> str:
    total = sum(max(weight, 0) for weight in weights)
    if total <= 0:
        return items[rng.randrange(len(items))]
    point = rng.uniform(0, total)
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(weight, 0)
        if point <= cumulative:
            return item
    return items[-1]


def _score_key(
    graph: PassOrderGraph,
    path: list[str],
) -> tuple[int, int, int, int, tuple[str, ...]]:
    order_score = 0
    conflict_score = 0
    adjacent_score = 0
    for left_index, source in enumerate(path):
        for target in path[left_index + 1 :]:
            order_score += graph.edge_weight(source, target)
            conflict_score += graph.edge_weight(target, source)
    for source, target in zip(path, path[1:]):
        adjacent_score += graph.edge_weight(source, target)
    return (
        order_score - conflict_score,
        order_score,
        adjacent_score,
        len(path),
        tuple(reversed(path)),
    )
