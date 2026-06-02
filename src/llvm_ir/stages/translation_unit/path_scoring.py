"""Scoring helpers for pass-order graph heuristics."""

from __future__ import annotations

from dataclasses import dataclass

from .order_graph import PassOrderGraph


@dataclass(frozen=True)
class PathScore:
    order_score: int
    conflict_score: int
    net_score: int
    adjacent_score: int
    length: int
    node_coverage: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "order_score": self.order_score,
            "conflict_score": self.conflict_score,
            "net_score": self.net_score,
            "adjacent_score": self.adjacent_score,
            "length": self.length,
            "node_coverage": self.node_coverage,
        }


def score_path(graph: PassOrderGraph, path: list[str]) -> PathScore:
    """Score a candidate path against all pairwise graph order constraints."""
    order_score = 0
    conflict_score = 0
    adjacent_score = 0
    for left_index, source in enumerate(path):
        for target in path[left_index + 1 :]:
            order_score += graph.edge_weight(source, target)
            conflict_score += graph.edge_weight(target, source)
    for source, target in zip(path, path[1:]):
        adjacent_score += graph.edge_weight(source, target)
    node_count = len(graph.nodes)
    coverage = len(set(path)) / node_count if node_count else 0.0
    return PathScore(
        order_score=order_score,
        conflict_score=conflict_score,
        net_score=order_score - conflict_score,
        adjacent_score=adjacent_score,
        length=len(path),
        node_coverage=coverage,
    )


def node_priority(graph: PassOrderGraph, node: str) -> int:
    """Positive priority means the graph tends to place the node earlier."""
    outgoing = 0
    incoming = 0
    for edge in graph.edges:
        if edge.source == node:
            outgoing += edge.weight
        if edge.target == node:
            incoming += edge.weight
    return outgoing - incoming


def sorted_nodes_by_priority(graph: PassOrderGraph) -> list[str]:
    return sorted(
        graph.nodes,
        key=lambda node: (-node_priority(graph, node), node),
    )
