"""Voting ensemble aggregation heuristic."""

from __future__ import annotations

from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph


class VotingEnsemble(AggregationHeuristic):
    name = "voting_ensemble"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        from .registry import build_heuristic

        voter_names = list(
            config.get(
                "voters",
                ["wfas_eades", "scc_ordering", "pagerank", "position_median", "hpp"],
            )
        )
        rankings: list[list[str]] = []
        for name in voter_names:
            if name == self.name:
                continue
            heuristic = build_heuristic(str(name))
            rankings.append(heuristic.aggregate(dataset, graph, config).chosen_sequence)
        scores = {node: 0.0 for node in graph.nodes}
        for ranking in rankings:
            n = len(ranking)
            for index, node in enumerate(ranking):
                scores[node] += n - index
        sequence = sorted(graph.nodes, key=lambda node: (-scores[node], node))
        return AggregationResult(
            sequences=[sequence],
            chosen_sequence=sequence,
            chosen_prefix_length=len(sequence),
            extra={"voters": voter_names, "scores": scores},
        )

