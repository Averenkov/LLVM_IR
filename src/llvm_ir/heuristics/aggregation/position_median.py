"""Position-median aggregation heuristic."""

from __future__ import annotations

from collections import defaultdict
from statistics import pstdev
from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .stats import compute_node_stats


class PositionMedianOrdering(AggregationHeuristic):
    name = "position_median"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        tau = float(config.get("tau", 0.35))
        use_filter = bool(config.get("use_filter", True))
        positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
        raw_positions: dict[str, list[float]] = defaultdict(list)
        for result in dataset.results:
            length = max(len(result.sequence), 1)
            weight = float(max(result.delta, 0))
            for index, node in enumerate(result.sequence):
                if node not in graph.nodes:
                    continue
                normalized = index / length
                positions[node].append((normalized, weight))
                raw_positions[node].append(normalized)
        stats = compute_node_stats(dataset, graph)
        keep = []
        medians = {}
        for node in graph.nodes:
            values = positions.get(node, [])
            if not values:
                continue
            spread = pstdev(raw_positions[node]) if len(raw_positions[node]) > 1 else 0.0
            if use_filter and spread > tau:
                continue
            medians[node] = _weighted_median(values)
            keep.append(node)
        sequence = sorted(keep, key=lambda node: (medians[node], -stats[node].gain, node))
        return AggregationResult([sequence], sequence, len(sequence), {"medians": medians})


def _weighted_median(values: list[tuple[float, float]]) -> float:
    total = sum(max(weight, 0.0) for _value, weight in values)
    if total <= 0:
        sorted_values = sorted(value for value, _weight in values)
        return sorted_values[len(sorted_values) // 2]
    cumulative = 0.0
    for value, weight in sorted(values):
        cumulative += max(weight, 0.0)
        if cumulative >= total / 2.0:
            return value
    return sorted(values)[-1][0]

