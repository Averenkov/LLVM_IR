"""Tests for the strong-link beam segment-tree (mock measure, no LLVM)."""

import unittest

from llvm_ir.heuristics.translation_unit.strong_links import (
    beam_segment_tree_merge,
    mine_strong_links,
)
from llvm_ir.stages.translation_unit.contracts import FunctionPassResult


def _fr(passes, delta=10):
    return FunctionPassResult(function="f", baseline_size=100, best_size=100 - delta, passes=passes)


class MineTests(unittest.TestCase):
    def test_contiguous_subseq_shared_by_two_functions(self) -> None:
        results = [_fr(["a", "b", "c"]), _fr(["x", "b", "c", "y"]), _fr(["z"])]
        links = dict(mine_strong_links(results, min_support=2))
        # "b","c","b,c" appear in functions 0 and 1 -> support 2
        self.assertEqual(links[("b", "c")], 2)
        self.assertEqual(links[("b",)], 2)
        self.assertNotIn(("a",), links)  # only in 1 function

    def test_max_links_caps_by_weight(self) -> None:
        results = [_fr(["a", "b"]), _fr(["a", "b"]), _fr(["a", "c"])]
        links = mine_strong_links(results, min_support=2, max_links=2)
        self.assertEqual(len(links), 2)
        # 'a' (support 3) must be among the top
        self.assertIn(("a",), dict(links))

    def test_positives_only(self) -> None:
        results = [_fr(["a", "b"], delta=10), _fr(["a", "b"], delta=0)]  # 2nd not positive
        links = dict(mine_strong_links(results, min_support=2, positives_only=True))
        self.assertEqual(links, {})  # only 1 positive function -> nothing shared by >=2


class BeamMergeTests(unittest.TestCase):
    def test_graph_guided_concat_and_beam(self) -> None:
        # mock TU: a+b -> 700 (good), others weaker
        def measure(passes):
            p = list(passes)
            if p[:2] == ["a", "b"]:
                return 700, ("a", "b")
            if p == ["a"]:
                return 900, ("a",)
            if p == ["b"]:
                return 950, ("b",)
            return 1000, tuple(passes)

        # edge a->b exists (weight 5), b->a doesn't
        def edge_weight(u, v):
            return 5 if (u, v) == ("a", "b") else 0

        leaves = [[(("a",), 900)], [(("b",), 950)]]
        (seq, size), evals = beam_segment_tree_merge(
            leaves, edge_weight, measure, beam=10, concat_cap=12, max_length=8
        )
        self.assertEqual(seq, ("a", "b"))  # graph-guided a+b found
        self.assertEqual(size, 700)
        self.assertGreater(evals, 0)

    def test_beam_size_capped(self) -> None:
        def measure(passes):
            return 1000 - len(passes), tuple(passes)

        def edge_weight(u, v):
            return 1  # everything connectable

        leaves = [[((c,), 999)] for c in "abcdefgh"]
        # run one merge level manually via full merge; just check it returns a valid best
        (seq, size), _ = beam_segment_tree_merge(leaves, edge_weight, measure, beam=3, concat_cap=4, max_length=8)
        self.assertTrue(seq)
        self.assertLessEqual(size, 999)

    def test_single_leaf(self) -> None:
        (seq, size), evals = beam_segment_tree_merge(
            [[(("a",), 900)]], lambda u, v: 0, lambda p: (900, p), beam=10, concat_cap=4, max_length=8
        )
        self.assertEqual((seq, size), (("a",), 900))
        self.assertEqual(evals, 0)


if __name__ == "__main__":
    unittest.main()
