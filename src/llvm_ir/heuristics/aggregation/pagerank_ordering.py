"""PageRank-based aggregation heuristic."""

from __future__ import annotations

from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .stats import compute_node_stats


class PageRankOrdering(AggregationHeuristic):
    name = "pagerank"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        nodes = sorted(graph.nodes)
        stats = compute_node_stats(dataset, graph)
        damping = float(config.get("damping", 0.85))
        max_iter = int(config.get("max_iter", 50))
        eps = float(config.get("eps", 1e-6))
        personalization = _personalization(nodes, {node: stats[node].gain for node in nodes})
        incoming_rank = _pagerank(nodes, graph.delta_weight, personalization, damping, max_iter, eps)
        reverse_weights = {(target, source): weight for (source, target), weight in graph.delta_weight.items()}
        outgoing_rank = _pagerank(nodes, reverse_weights, personalization, damping, max_iter, eps)
        sequence = sorted(
            nodes,
            key=lambda node: (-(incoming_rank[node] - outgoing_rank[node]), node),
        )
        return AggregationResult([sequence], sequence, len(sequence), {"rank": incoming_rank})


def _personalization(nodes: list[str], gain: dict[str, float]) -> dict[str, float]:
    total = sum(max(gain.get(node, 0.0), 0.0) for node in nodes)
    if total <= 0 and nodes:
        return {node: 1.0 / len(nodes) for node in nodes}
    return {node: max(gain.get(node, 0.0), 0.0) / total for node in nodes}


def _pagerank(
    nodes: list[str],
    weights: dict[tuple[str, str], float],
    personalization: dict[str, float],
    damping: float,
    max_iter: int,
    eps: float,
) -> dict[str, float]:
    if not nodes:
        return {}
    rank = {node: 1.0 / len(nodes) for node in nodes}
    outgoing = {node: 0.0 for node in nodes}
    for (source, target), weight in weights.items():
        if source in outgoing and target in outgoing and source != target:
            outgoing[source] += max(weight, 0.0)
    for _ in range(max_iter):
        next_rank = {
            node: (1.0 - damping) * personalization.get(node, 0.0)
            for node in nodes
        }
        dangling = sum(rank[node] for node in nodes if outgoing[node] <= 0)
        for node in nodes:
            next_rank[node] += damping * dangling * personalization.get(node, 0.0)
        for (source, target), weight in weights.items():
            if source == target or source not in rank or target not in rank:
                continue
            if outgoing[source] > 0:
                next_rank[target] += damping * rank[source] * max(weight, 0.0) / outgoing[source]
        diff = sum(abs(next_rank[node] - rank[node]) for node in nodes)
        rank = next_rank
        if diff <= eps:
            break
    return rank

