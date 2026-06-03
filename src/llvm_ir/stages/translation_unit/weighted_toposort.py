"""Weighted topological-sort heuristic for pass-order graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .graph.order_graph import PassOrderGraph
from .path_scoring import node_priority


@dataclass(frozen=True)
class WeightedToposortConfig:
    max_length: int = 0
    min_edge_weight: int = 1


def weighted_toposort_path(
    graph: PassOrderGraph,
    *,
    config: WeightedToposortConfig | None = None,
) -> list[str]:
    """Build an order by repeatedly selecting weighted zero-in-degree nodes.

    Cycles are handled by selecting the remaining node with the best outgoing
    minus incoming support, which is equivalent to breaking the weakest local
    opposition at the current frontier.
    """
    if config is None:
        config = WeightedToposortConfig()
    remaining = set(graph.nodes)
    path: list[str] = []

    while remaining:
        if config.max_length > 0 and len(path) >= config.max_length:
            break
        incoming = {
            node: _incoming_weight(graph, node, remaining, config.min_edge_weight)
            for node in remaining
        }
        zero_incoming = [node for node, weight in incoming.items() if weight == 0]
        if zero_incoming:
            selected = max(
                zero_incoming,
                key=lambda node: (
                    _outgoing_weight(
                        graph,
                        node,
                        remaining,
                        config.min_edge_weight,
                    ),
                    node_priority(graph, node),
                    _reverse_lex_key(node),
                ),
            )
        else:
            selected = max(
                remaining,
                key=lambda node: (
                    _outgoing_weight(
                        graph,
                        node,
                        remaining,
                        config.min_edge_weight,
                    )
                    - incoming[node],
                    node_priority(graph, node),
                    _reverse_lex_key(node),
                ),
            )
        path.append(selected)
        remaining.remove(selected)
    return path


def _incoming_weight(
    graph: PassOrderGraph,
    node: str,
    remaining: set[str],
    min_edge_weight: int,
) -> int:
    return sum(
        weight
        for (source, target), weight in graph.edge_counts.items()
        if target == node
        and source in remaining
        and source != node
        and weight >= min_edge_weight
    )


def _outgoing_weight(
    graph: PassOrderGraph,
    node: str,
    remaining: set[str],
    min_edge_weight: int,
) -> int:
    return sum(
        weight
        for (source, target), weight in graph.edge_counts.items()
        if source == node
        and target in remaining
        and target != node
        and weight >= min_edge_weight
    )


def _reverse_lex_key(value: str) -> tuple[int, ...]:
    return tuple(-ord(char) for char in value)
