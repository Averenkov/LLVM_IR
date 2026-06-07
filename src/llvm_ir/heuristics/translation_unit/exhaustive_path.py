"""Exhaustive fixed-length path heuristic for pass-order graphs."""

from __future__ import annotations

from dataclasses import dataclass

from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph


@dataclass(frozen=True)
class ExhaustivePathConfig:
    path_length: int = 6
    min_edge_weight: int = 1
    fallback_to_shorter: bool = True


def exhaustive_fixed_length_path(
    graph: PassOrderGraph,
    *,
    config: ExhaustivePathConfig | None = None,
) -> list[str]:
    """Enumerate simple directed paths and return the best fixed-length path."""
    paths = exhaustive_fixed_length_top_paths(graph, config=config, top_k=1)
    return paths[0] if paths else []


def exhaustive_fixed_length_top_paths(
    graph: PassOrderGraph,
    *,
    config: ExhaustivePathConfig | None = None,
    top_k: int = 10,
) -> list[list[str]]:
    """Enumerate simple directed paths and return top paths by graph score."""
    if config is None:
        config = ExhaustivePathConfig()
    nodes = sorted(graph.nodes)
    if not nodes:
        return []

    target_length = max(1, min(config.path_length, len(nodes)))
    adjacent = _adjacency(graph, nodes, min_edge_weight=config.min_edge_weight)
    lengths = [target_length]
    if config.fallback_to_shorter:
        lengths.extend(range(target_length - 1, 0, -1))

    for length in lengths:
        paths = _top_paths_of_length(graph, nodes, adjacent, length, top_k=max(1, top_k))
        if paths:
            return paths
    return []


def _top_paths_of_length(
    graph: PassOrderGraph,
    nodes: list[str],
    adjacent: dict[str, list[str]],
    length: int,
    *,
    top_k: int,
) -> list[list[str]]:
    candidates: dict[tuple[str, ...], tuple[int, int, int, int, tuple[str, ...]]] = {}

    def visit(path: list[str], seen: set[str]) -> None:
        if len(path) == length:
            key = tuple(path)
            candidates[key] = _score_key(graph, path)
            return
        for target in adjacent.get(path[-1], []):
            if target in seen:
                continue
            seen.add(target)
            path.append(target)
            visit(path, seen)
            path.pop()
            seen.remove(target)

    for node in nodes:
        visit([node], {node})
    ranked = sorted(
        candidates,
        key=lambda path: (candidates[path], path),
        reverse=True,
    )
    return [list(path) for path in ranked[:top_k]]


def _adjacency(
    graph: PassOrderGraph,
    nodes: list[str],
    *,
    min_edge_weight: int,
) -> dict[str, list[str]]:
    node_set = set(nodes)
    adjacent = {node: [] for node in nodes}
    for (source, target), weight in sorted(graph.edge_counts.items()):
        if (
            source in node_set
            and target in node_set
            and source != target
            and weight >= min_edge_weight
        ):
            adjacent[source].append(target)
    return adjacent


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
