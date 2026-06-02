"""WFAS-Eades aggregation heuristic."""

from __future__ import annotations

from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .graph_utils import best_prefix_by_graph_score, blended_weights, eades_order


class WFASEades(AggregationHeuristic):
    name = "wfas_eades"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        del dataset
        alpha = float(config.get("alpha", 0.5))
        weights = blended_weights(graph, alpha=alpha)
        sequence = eades_order(set(graph.nodes), weights)
        prefix_len, prefix_score = best_prefix_by_graph_score(sequence, weights)
        return AggregationResult(
            sequences=[sequence],
            chosen_sequence=sequence,
            chosen_prefix_length=prefix_len,
            extra={"prefix_graph_score": prefix_score, "alpha": alpha},
        )

