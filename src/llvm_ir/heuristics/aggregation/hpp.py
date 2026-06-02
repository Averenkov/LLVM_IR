"""Harmful-pass penalized aggregation heuristic."""

from __future__ import annotations

import math
from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .graph_utils import best_prefix_by_graph_score, blended_weights, eades_order
from .stats import compute_node_stats, normalize_values


class HPPHeuristic(AggregationHeuristic):
    name = "hpp"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        alpha = float(config.get("alpha", 0.5))
        beta = float(config.get("beta", 1.0))
        lambda_ = float(config.get("lambda_", 0.1))
        theta = float(config.get("theta", -math.inf))
        weights = blended_weights(graph, alpha=alpha)
        keep, scores = harmful_pass_filter(
            dataset,
            graph,
            weights,
            beta=beta,
            lambda_=lambda_,
            theta=theta,
        )
        subgraph = graph.induced(keep)
        subweights = {
            edge: weight
            for edge, weight in weights.items()
            if edge[0] in keep and edge[1] in keep
        }
        sequence = eades_order(set(subgraph.nodes), subweights)
        prefix_len, prefix_score = best_prefix_by_graph_score(sequence, subweights)
        return AggregationResult(
            sequences=[sequence],
            chosen_sequence=sequence,
            chosen_prefix_length=prefix_len,
            extra={"scores": scores, "kept": sorted(keep), "prefix_graph_score": prefix_score},
        )


def harmful_pass_filter(
    dataset: Dataset,
    graph: PassGraph,
    weights: dict[tuple[str, str], float],
    *,
    beta: float,
    lambda_: float,
    theta: float,
) -> tuple[set[str], dict[str, float]]:
    stats = compute_node_stats(dataset, graph, lambda_=lambda_)
    harm_norm = normalize_values({node: item.harm for node, item in stats.items()})
    scores = {}
    keep = set()
    for node in graph.nodes:
        outgoing = sum(
            weight for (source, _target), weight in weights.items() if source == node
        )
        incoming = sum(
            weight for (_source, target), weight in weights.items() if target == node
        )
        score = outgoing - incoming - beta * harm_norm.get(node, 0.0)
        scores[node] = score
        if score >= theta:
            keep.add(node)
    return keep, scores

