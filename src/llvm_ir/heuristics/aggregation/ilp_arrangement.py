"""ILP/LP-relaxed linear arrangement heuristic."""

from __future__ import annotations

from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .graph_utils import blended_weights, eades_order


class ILPLinearArrangement(AggregationHeuristic):
    name = "ilp_arrangement"
    optional = True

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        del dataset
        weights = blended_weights(graph, alpha=float(config.get("alpha", 0.5)))
        try:
            sequence = _solve_with_pulp(set(graph.nodes), weights)
            solver = "pulp"
        except ImportError as exc:
            sequence = eades_order(set(graph.nodes), weights)
            solver = "fallback_eades"
            return AggregationResult(
                [sequence],
                sequence,
                len(sequence),
                {"unavailable": True, "solver": solver, "error": str(exc)},
            )
        return AggregationResult([sequence], sequence, len(sequence), {"solver": solver})


def _solve_with_pulp(nodes: set[str], weights: dict[tuple[str, str], float]) -> list[str]:
    try:
        import pulp
    except ImportError as exc:
        raise ImportError("PuLP is not installed") from exc
    ordered = sorted(nodes)
    problem = pulp.LpProblem("llvm_ir_linear_arrangement", pulp.LpMaximize)
    x: dict[tuple[str, str], object] = {}
    for left in ordered:
        for right in ordered:
            if left == right:
                continue
            x[(left, right)] = pulp.LpVariable(f"x_{left}_{right}", 0, 1)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            problem += x[(left, right)] + x[(right, left)] == 1
    problem += pulp.lpSum(
        weight * x[(source, target)]
        for (source, target), weight in weights.items()
        if source in nodes and target in nodes and source != target
    )
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    scores = {
        node: sum(float(pulp.value(x[(node, other)]) or 0.0) for other in ordered if other != node)
        for node in ordered
    }
    return sorted(ordered, key=lambda node: (-scores[node], node))
