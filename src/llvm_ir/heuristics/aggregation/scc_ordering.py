"""SCC-aware aggregation heuristic."""

from __future__ import annotations

from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph
from .graph_utils import blended_weights, strongly_connected_components, topological_order
from .stats import compute_node_stats


class SCCOrdering(AggregationHeuristic):
    name = "scc_ordering"

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        weights = blended_weights(graph, alpha=float(config.get("alpha", 0.5)))
        stats = compute_node_stats(dataset, graph)
        components = strongly_connected_components(set(graph.nodes), weights)
        comp_id = {
            node: index
            for index, component in enumerate(components)
            for node in component
        }
        comp_nodes = {f"c{index}" for index in range(len(components))}
        comp_weights: dict[tuple[str, str], float] = {}
        for (source, target), weight in weights.items():
            left = comp_id[source]
            right = comp_id[target]
            if left == right:
                continue
            pair = (f"c{left}", f"c{right}")
            comp_weights[pair] = comp_weights.get(pair, 0.0) + weight
        ordered_components = topological_order(comp_nodes, comp_weights)
        sequence: list[str] = []
        for component_name in ordered_components:
            index = int(component_name[1:])
            component = components[index]
            sequence.extend(
                sorted(
                    component,
                    key=lambda node: (
                        -(stats[node].gain / max(stats[node].freq, 1)),
                        -stats[node].net_delta,
                        node,
                    ),
                )
            )
        return AggregationResult(
            sequences=[sequence],
            chosen_sequence=sequence,
            chosen_prefix_length=len(sequence),
            extra={"scc_count": len(components)},
        )

