"""Tests for the network-max-flow path heuristic (no LLVM required)."""

import unittest

from llvm_ir.heuristics.translation_unit.max_flow_paths import (
    MaxFlowConfig,
    scale_flow_length,
    top_flow_paths,
)
from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph


def _graph(edges, starts=None):
    graph = PassOrderGraph(benchmark="t")
    for (s, t), w in edges.items():
        graph.nodes.update((s, t))
        graph.edge_counts[(s, t)] = w
    if starts:
        graph.start_counts = dict(starts)
    return graph


class MaxFlowPathsTests(unittest.TestCase):
    def test_length3_flows_ordered_by_flow_value(self) -> None:
        # a->b(5)->d(4) bottleneck 4 ; a->c(3)->d(2) bottleneck 2 ; source a=10.
        graph = _graph(
            {("a", "b"): 5, ("a", "c"): 3, ("b", "d"): 4, ("c", "d"): 2},
            starts={"a": 10},
        )
        flows = top_flow_paths(graph, config=MaxFlowConfig(min_length=3, max_length=3, per_length_top_k=10))
        self.assertEqual(flows[0], ["a", "b", "d"])  # highest-flow path first
        self.assertIn(["a", "c", "d"], flows)

    def test_length2_flows(self) -> None:
        graph = _graph({("a", "b"): 5, ("a", "c"): 3}, starts={"a": 10})
        flows = top_flow_paths(graph, config=MaxFlowConfig(min_length=2, max_length=2, per_length_top_k=10))
        self.assertEqual(flows[0], ["a", "b"])
        self.assertIn(["a", "c"], flows)

    def test_multiple_lengths_combined_and_deduped(self) -> None:
        graph = _graph(
            {("a", "b"): 5, ("a", "c"): 3, ("b", "d"): 4, ("c", "d"): 2},
            starts={"a": 10},
        )
        flows = top_flow_paths(graph, config=MaxFlowConfig(min_length=2, max_length=3, per_length_top_k=10))
        lengths = {len(p) for p in flows}
        self.assertEqual(lengths, {2, 3})
        # no duplicates
        self.assertEqual(len(flows), len({tuple(p) for p in flows}))

    def test_bottleneck_limits_flow(self) -> None:
        # Narrow middle edge caps the flow regardless of fat outer edges.
        graph = _graph(
            {("a", "b"): 100, ("b", "c"): 1, ("c", "d"): 100},
            starts={"a": 100},
        )
        flows = top_flow_paths(graph, config=MaxFlowConfig(min_length=4, max_length=4, per_length_top_k=5))
        self.assertEqual(flows[0], ["a", "b", "c", "d"])

    def test_scale_flow_length(self) -> None:
        self.assertEqual(scale_flow_length(2904, base=4, cap=8), 4)
        self.assertEqual(scale_flow_length(60_000, base=4, cap=8), 5)
        self.assertEqual(scale_flow_length(200_000, base=4, cap=8), 6)
        self.assertEqual(scale_flow_length(1_191_776, base=4, cap=8), 7)
        self.assertEqual(scale_flow_length(1_191_776, base=4, cap=6), 6)  # capped

    def test_empty_graph(self) -> None:
        self.assertEqual(top_flow_paths(PassOrderGraph(benchmark="x")), [])


if __name__ == "__main__":
    unittest.main()
