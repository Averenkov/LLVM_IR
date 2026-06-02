"""Beam search with diversity penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .graph_utils import blended_weights, unique_sequences


@dataclass(frozen=True)
class BeamState:
    prefix: list[str]
    score: float

    @property
    def used(self) -> set[str]:
        return set(self.prefix)


class BeamSearchDiversity(AggregationHeuristic):
    name = "beam_diversity"
    supports_topk = True

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        del dataset
        weights = blended_weights(graph, alpha=float(config.get("alpha", 0.5)))
        sequences = beam_search_diverse(
            sorted(graph.nodes),
            weights,
            beam_width=int(config.get("beam_width", 16)),
            top_k=int(config.get("top_k", 4)),
            gamma=float(config.get("gamma", 0.5)),
        )
        chosen = sequences[0] if sequences else []
        return AggregationResult(
            sequences=sequences,
            chosen_sequence=chosen,
            chosen_prefix_length=len(chosen),
            extra={"top_k": len(sequences)},
        )


def beam_search_diverse(
    nodes: list[str],
    weights: dict[tuple[str, str], float],
    *,
    beam_width: int,
    top_k: int,
    gamma: float,
    seed_order: list[str] | None = None,
) -> list[list[str]]:
    if not nodes:
        return []
    order_index = {node: index for index, node in enumerate(seed_order or nodes)}
    beams = [BeamState([], 0.0)]
    for _step in range(len(nodes)):
        expansions: list[BeamState] = []
        for state in beams:
            unused = [node for node in nodes if node not in state.used]
            for node in unused:
                gain = _marginal_gain(state.prefix, node, weights)
                diversity = _diversity_penalty(state.used | {node}, beams)
                expansions.append(
                    BeamState(state.prefix + [node], state.score + gain - gamma * diversity)
                )
        expansions.sort(
            key=lambda state: (
                -state.score,
                [order_index.get(node, len(nodes)) for node in state.prefix],
                state.prefix,
            )
        )
        beams = expansions[: max(1, beam_width)]
    beams.sort(key=lambda state: (-state.score, state.prefix))
    return unique_sequences([state.prefix for state in beams], limit=top_k)


def _marginal_gain(prefix: list[str], node: str, weights: dict[tuple[str, str], float]) -> float:
    return sum(weights.get((prev, node), 0.0) - weights.get((node, prev), 0.0) for prev in prefix)


def _diversity_penalty(candidate: set[str], beams: list[BeamState]) -> float:
    if not beams:
        return 0.0
    values = []
    for state in beams:
        other = state.used
        union = candidate | other
        values.append(len(candidate & other) / len(union) if union else 0.0)
    return sum(values) / len(values)

