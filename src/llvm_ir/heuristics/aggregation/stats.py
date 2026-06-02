"""Node-level statistics for aggregation heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import Dataset, PassGraph


@dataclass(frozen=True)
class NodeStats:
    freq: int
    gain: float
    harmful_loss: float
    harm: float
    net_delta: float
    out_delta: float
    in_delta: float
    out_count: float
    in_count: float


def compute_node_stats(dataset: Dataset, graph: PassGraph, *, lambda_: float = 0.1) -> dict[str, NodeStats]:
    freq = {node: 0 for node in graph.nodes}
    gain = {node: 0.0 for node in graph.nodes}
    harmful_loss = {node: 0.0 for node in graph.nodes}

    for result in dataset.results:
        seen = set(result.sequence)
        for node in seen:
            if node not in freq:
                continue
            freq[node] += 1
            if result.delta >= 0:
                gain[node] += float(result.delta)
            else:
                harmful_loss[node] += float(abs(result.delta))

    stats = {}
    for node in graph.nodes:
        out_delta = sum(
            weight
            for (source, _target), weight in graph.delta_weight.items()
            if source == node
        )
        in_delta = sum(
            weight
            for (_source, target), weight in graph.delta_weight.items()
            if target == node
        )
        out_count = sum(
            weight
            for (source, _target), weight in graph.count_weight.items()
            if source == node
        )
        in_count = sum(
            weight
            for (_source, target), weight in graph.count_weight.items()
            if target == node
        )
        harm = harmful_loss[node] - lambda_ * gain[node]
        stats[node] = NodeStats(
            freq=freq[node],
            gain=gain[node],
            harmful_loss=harmful_loss[node],
            harm=harm,
            net_delta=out_delta - in_delta,
            out_delta=out_delta,
            in_delta=in_delta,
            out_count=out_count,
            in_count=in_count,
        )
    return stats


def normalize_values(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi):
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def node_gain_priority(stats: dict[str, NodeStats], node: str) -> tuple[float, float, str]:
    item = stats.get(node)
    if item is None:
        return (0.0, 0.0, node)
    return (item.gain, item.net_delta, node)

