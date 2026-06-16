"""Tests for the adaptive measured-superpath heuristic core (no LLVM required)."""

import unittest
from random import Random

from llvm_ir.heuristics.translation_unit.measured_superpath import (
    concat_segments,
    node_support,
    prune_small_edges,
    sample_index,
    select_diverse_by_start,
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

    def test_select_diverse_covers_each_start(self) -> None:
        # weight-sorted: heaviest 3 all start with 'a'; 'b'/'c' starts rank lower.
        paths = [
            ("a", "1"), ("a", "2"), ("a", "3"),  # heavy, same start
            ("b", "1"),                            # lighter, distinct start
            ("c", "1"),                            # lighter, distinct start
        ]
        sel = select_diverse_by_start(paths, count=3)
        starts = {p[0] for p in sel}
        # pure top-3-by-weight would be {a} only; diversity must include b and c
        self.assertEqual(starts, {"a", "b", "c"})

    def test_select_diverse_fills_by_weight_after_coverage(self) -> None:
        paths = [("a", "1"), ("a", "2"), ("b", "1"), ("c", "1")]
        sel = select_diverse_by_start(paths, count=4)
        # coverage first (a,b,c best), then fill with next-best by weight (a,2)
        self.assertEqual(set(sel), {("a", "1"), ("b", "1"), ("c", "1"), ("a", "2")})

    def test_select_diverse_single_start(self) -> None:
        paths = [("a", "1"), ("a", "2"), ("a", "3")]
        self.assertEqual(select_diverse_by_start(paths, count=2), [("a", "1"), ("a", "2")])

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

    def test_superpaths_vertex_weight_prefers_high_profit_segments(self) -> None:
        # Two length-2 paths with equal edge weight; profits favor the (a,b)->(b,c) route.
        segments = [("a", "b"), ("b", "c"), ("d", "e"), ("e", "f")]
        super_edges = {(0, 1): 1.0, (2, 3): 1.0}
        profits = [100.0, 100.0, 0.0, 0.0]
        ranked = superpaths_by_length(
            segments, super_edges, profits, lengths=(2,), per_length=2, beam=8, max_length=8, vertex_weight=1.0
        )
        # High-profit path (a,b,c) must rank before the zero-profit (d,e,f).
        self.assertEqual(ranked[0], ("a", "b", "c"))

    def test_superpaths_respects_max_length(self) -> None:
        segments = [("a", "b", "c", "d"), ("e", "f", "g", "h")]
        super_edges = {(0, 1): 3.0}
        self.assertEqual(
            superpaths_by_length(segments, super_edges, lengths=(2,), per_length=5, beam=8, max_length=6),
            [],
        )


if __name__ == "__main__":
    unittest.main()
