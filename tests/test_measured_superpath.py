"""Tests for the adaptive measured-superpath heuristic core (no LLVM required)."""

import unittest
from random import Random

from llvm_ir.heuristics.translation_unit.measured_superpath import (
    concat_segments,
    node_support,
    prune_small_edges,
    sample_index,
    superpaths_by_length,
    vertex_budget_paths,
)
from llvm_ir.stages.translation_unit.contracts import FunctionPassResult


def _fr(passes):
    return FunctionPassResult(function="f", baseline_size=100, best_size=80, passes=passes)


class CoreTests(unittest.TestCase):
    def test_node_support_counts_sequences(self) -> None:
        results = [_fr(["a", "b", "a"]), _fr(["a", "c"]), _fr(["d"])]
        support = node_support(results)
        self.assertEqual(support["a"], 2)
        self.assertEqual(support["b"], 1)

    def test_prune_small_edges_drops_bottom_fraction(self) -> None:
        edges = {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 3, ("c", "d"): 4, ("d", "e"): 5}
        kept = prune_small_edges(edges, 40.0)
        self.assertEqual(len(kept), 3)
        self.assertNotIn(("a", "b"), kept)

    def test_vertex_budget_paths_length3(self) -> None:
        edges = {("a", "b"): 5, ("b", "c"): 4, ("c", "d"): 3, ("d", "e"): 2}
        nodes = ["a", "b", "c", "d", "e"]
        support = {n: 1 for n in nodes}
        paths = vertex_budget_paths(nodes, edges, support, total_budget=10, path_nodes=4)
        self.assertTrue(all(len(p) == 4 for p in paths))
        self.assertIn(("a", "b", "c", "d"), paths)

    def test_vertex_budget_degrades_on_small_graphs(self) -> None:
        # 3-node chain admits no 4-node path -> must fall back to 3-node paths.
        edges = {("a", "b"): 5, ("b", "c"): 4}
        support = {"a": 1, "b": 1, "c": 1}
        paths = vertex_budget_paths(["a", "b", "c"], edges, support, total_budget=10, path_nodes=4)
        self.assertTrue(paths)
        self.assertIn(("a", "b", "c"), paths)
        self.assertTrue(all(len(p) <= 3 for p in paths))

    def test_vertex_budget_single_node_graph(self) -> None:
        # 1 node, no edges -> the single pass itself is the only candidate.
        paths = vertex_budget_paths(["a"], {}, {"a": 1}, total_budget=10, path_nodes=4)
        self.assertEqual(paths, [("a",)])

    def test_vertex_budget_respects_total_cap(self) -> None:
        nodes = ["a", "b", "c", "d"]
        edges = {(s, t): 1 for s in nodes for t in nodes if s != t}
        support = {n: 1 for n in nodes}
        self.assertLessEqual(len(vertex_budget_paths(nodes, edges, support, total_budget=3, path_nodes=4)), 3)

    def test_concat_overlap_dedup(self) -> None:
        self.assertEqual(concat_segments(("a", "b"), ("b", "c")), ("a", "b", "c"))
        self.assertEqual(concat_segments(("a", "b"), ("c", "d")), ("a", "b", "c", "d"))

    def test_sample_index_deterministic_and_biased(self) -> None:
        weights = {0: 0.0, 1: 1000.0, 2: 0.0}
        # index 1 dominates -> should be sampled
        self.assertEqual(sample_index(weights, Random(1)), 1)

    def test_superpaths_only_via_measured_edges_and_lengths(self) -> None:
        segments = [("a", "b"), ("b", "c"), ("c", "d"), ("x", "y")]
        super_edges = {(0, 1): 5.0, (1, 2): 4.0}  # measured chain 0->1->2; x,y isolated
        paths = superpaths_by_length(segments, super_edges, lengths=(2, 3), per_length=10, beam=32, max_length=10)
        self.assertIn(("a", "b", "c"), paths)        # 0->1 (len 2)
        self.assertIn(("a", "b", "c", "d"), paths)    # 0->1->2 (len 3)
        self.assertFalse(any("x" in p for p in paths))  # isolated segment unused

    def test_superpaths_respects_max_length(self) -> None:
        segments = [("a", "b", "c", "d"), ("e", "f", "g", "h")]
        super_edges = {(0, 1): 3.0}
        self.assertEqual(
            superpaths_by_length(segments, super_edges, lengths=(2,), per_length=5, beam=8, max_length=6),
            [],
        )


if __name__ == "__main__":
    unittest.main()
