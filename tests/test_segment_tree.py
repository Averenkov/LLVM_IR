"""Tests for the segment-tree hierarchical merge (mock measure, no LLVM)."""

import unittest

from llvm_ir.heuristics.translation_unit.segment_tree import segment_tree_merge


# Mock TU: baseline 1000. Each pass cuts a fixed amount the first time it appears;
# the concatenation a+b+c+d is the global optimum.
CUT = {"a": 100, "b": 80, "c": 60, "d": 40, "x": 5}


def _measure(passes):
    size = 1000
    seen = set()
    for p in passes:
        if p in CUT and p not in seen:
            size -= CUT[p]
            seen.add(p)
    # best-prefix == full here (monotone improvement), return full sequence
    return size, tuple(passes)


class SegmentTreeTests(unittest.TestCase):
    def test_merge_finds_combined_optimum(self) -> None:
        leaves = [(("a",), 900), (("b",), 920), (("c",), 940), (("d",), 960)]
        (seq, size), evals = segment_tree_merge(leaves, _measure, max_length=8)
        # best reachable combines all four -> 1000-100-80-60-40 = 720
        self.assertEqual(size, 720)
        self.assertEqual(set(seq), {"a", "b", "c", "d"})
        self.assertGreater(evals, 0)

    def test_tries_both_orders(self) -> None:
        # order matters: only b-before-a improves in this mock
        def measure(passes):
            size = 1000
            if list(passes)[:2] == ["b", "a"]:
                size = 700
            elif "a" in passes:
                size = 900
            return size, tuple(passes)

        leaves = [(("a",), 900), (("b",), 1000)]
        (seq, size), _ = segment_tree_merge(leaves, measure, max_length=8)
        self.assertEqual(size, 700)
        self.assertEqual(seq, ("b", "a"))

    def test_respects_max_length(self) -> None:
        captured = []

        def measure(passes):
            captured.append(len(passes))
            return 1000 - len(passes), tuple(passes)

        leaves = [(("a", "b", "c"), 997), (("d", "e", "f"), 997)]
        segment_tree_merge(leaves, measure, max_length=4)
        self.assertTrue(all(n <= 4 for n in captured))

    def test_single_leaf(self) -> None:
        (seq, size), evals = segment_tree_merge([(("a",), 900)], _measure, max_length=8)
        self.assertEqual((seq, size), (("a",), 900))
        self.assertEqual(evals, 0)

    def test_empty(self) -> None:
        (seq, size), evals = segment_tree_merge([], _measure, max_length=8)
        self.assertEqual(seq, ())
        self.assertEqual(evals, 0)


if __name__ == "__main__":
    unittest.main()
