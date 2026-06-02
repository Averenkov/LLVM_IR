"""Markov hitting-time aggregation heuristic."""

from __future__ import annotations

from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph


class MarkovHittingOrdering(AggregationHeuristic):
    name = "markov_hitting"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        del dataset
        nodes = sorted(graph.nodes)
        if not nodes:
            return AggregationResult([[]], [], 0, {})
        max_iter = int(config.get("max_iter", 200))
        eps = float(config.get("eps", 1e-6))
        stationary = _stationary_distribution(nodes, graph.delta_weight, max_iter, eps)
        sequence = sorted(nodes, key=lambda node: (-stationary[node], node))
        return AggregationResult([sequence], sequence, len(sequence), {"stationary": stationary})


def _stationary_distribution(
    nodes: list[str],
    weights: dict[tuple[str, str], float],
    max_iter: int,
    eps: float,
) -> dict[str, float]:
    rank = {node: 1.0 / len(nodes) for node in nodes}
    outgoing = {node: 0.0 for node in nodes}
    for (source, target), weight in weights.items():
        if source in outgoing and target in outgoing and source != target:
            outgoing[source] += max(weight, 0.0)
    for _ in range(max_iter):
        next_rank = {node: 0.0 for node in nodes}
        for source in nodes:
            if outgoing[source] <= 0:
                share = rank[source] / len(nodes)
                for target in nodes:
                    next_rank[target] += share
            else:
                for target in nodes:
                    weight = max(weights.get((source, target), 0.0), 0.0)
                    if weight:
                        next_rank[target] += rank[source] * weight / outgoing[source]
        diff = sum(abs(next_rank[node] - rank[node]) for node in nodes)
        rank = next_rank
        if diff <= eps:
            break
    return rank
