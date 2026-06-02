"""Main HPP-Eades-TopK aggregation heuristic."""

from __future__ import annotations

import math
from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .beam_diversity import beam_search_diverse
from .graph_utils import blended_weights, eades_order, unique_sequences
from .hpp import harmful_pass_filter


class HPPEadesTopK(AggregationHeuristic):
    name = "hpp_eades_topk"
    supports_topk = True

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
        beam_width = int(config.get("beam_width", 16))
        top_k = int(config.get("top_k", 4))
        gamma = float(config.get("gamma", 0.5))
        prefix_grid = list(config.get("prefix_grid", [0.3, 0.5, 0.7, 1.0]))

        weights = blended_weights(graph, alpha=alpha)
        keep, scores = harmful_pass_filter(
            dataset,
            graph,
            weights,
            beta=beta,
            lambda_=lambda_,
            theta=theta,
        )
        subweights = {
            edge: weight
            for edge, weight in weights.items()
            if edge[0] in keep and edge[1] in keep
        }
        eades = eades_order(set(keep), subweights)
        beam_sequences = beam_search_diverse(
            sorted(keep),
            subweights,
            beam_width=beam_width,
            top_k=top_k,
            gamma=gamma,
            seed_order=eades,
        )
        sequences = unique_sequences(beam_sequences + [eades], limit=top_k + 1)
        chosen_sequence, chosen_prefix, best_score = _choose_by_prefix_grid(
            sequences,
            subweights,
            prefix_grid,
        )
        return AggregationResult(
            sequences=sequences,
            chosen_sequence=chosen_sequence,
            chosen_prefix_length=chosen_prefix,
            extra={
                "hpp_scores": scores,
                "kept": sorted(keep),
                "eades_order": eades,
                "prefix_grid": prefix_grid,
                "prefix_graph_score": best_score,
            },
        )


def _choose_by_prefix_grid(
    sequences: list[list[str]],
    weights: dict[tuple[str, str], float],
    prefix_grid: list[float],
) -> tuple[list[str], int, float]:
    from .graph_utils import path_graph_score

    best_sequence: list[str] = []
    best_length = 0
    best_score = float("-inf")
    for sequence in sequences:
        if not sequence:
            continue
        for fraction in prefix_grid:
            length = max(0, min(len(sequence), math.ceil(float(fraction) * len(sequence))))
            score = path_graph_score(sequence[:length], weights)
            if score > best_score:
                best_sequence = sequence
                best_length = length
                best_score = score
    return best_sequence, best_length, best_score
