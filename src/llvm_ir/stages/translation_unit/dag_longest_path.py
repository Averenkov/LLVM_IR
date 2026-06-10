"""Cycle-breaking DAG longest-path heuristic for pass-order graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .graph.order_graph import PassOrderGraph
from .path_scoring import sorted_nodes_by_priority


@dataclass(frozen=True)
class DAGLongestPathConfig:
    min_net_weight: int = 1
    max_length: int = 0


def dag_longest_path(
    graph: PassOrderGraph,
    *,
    config: DAGLongestPathConfig | None = None,
) -> list[str]:
    """Find a high-support path after pairwise cycle breaking."""
    if config is None:
        config = DAGLongestPathConfig()
    order = sorted_nodes_by_priority(graph)
    dag_edges: dict[str, list[tuple[str, int]]] = {node: [] for node in order}

    for left_index, source in enumerate(order):
        for target in order[left_index + 1 :]:
            forward = graph.edge_weight(source, target)
            backward = graph.edge_weight(target, source)
            net = forward - backward
            if net >= config.min_net_weight:
                dag_edges[source].append((target, net))
            # Reverse preferences are intentionally ignored here: the heuristic
            # keeps edges aligned with the priority order so the graph remains a DAG.

    best_score: dict[str, int] = {node: 0 for node in order}
    best_path: dict[str, list[str]] = {node: [node] for node in order}

    for source in order:
        for target, weight in dag_edges[source]:
            candidate_score = best_score[source] + weight
            candidate_path = best_path[source] + [target]
            if config.max_length > 0 and len(candidate_path) > config.max_length:
                continue
            if (
                candidate_score > best_score[target]
                or (
                    candidate_score == best_score[target]
                    and len(candidate_path) > len(best_path[target])
                )
            ):
                best_score[target] = candidate_score
                best_path[target] = candidate_path

    if not best_path:
        return []
    return max(
        best_path.values(),
        key=lambda path: (path_weight(graph, path), len(path), tuple(reversed(path))),
    )


def path_weight(graph: PassOrderGraph, path: list[str]) -> int:
    return sum(
        graph.edge_weight(source, target)
        for source, target in zip(path, path[1:])
    )
