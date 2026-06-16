"""Tests for the function-sequences baseline heuristic (no LLVM required)."""

import unittest

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult
from llvm_ir.stages.translation_unit.evaluate_function_sequences import (
    unique_function_sequences,
)


def _fr(function, passes):
    return FunctionPassResult(function=function, baseline_size=100, best_size=80, passes=passes)


class FunctionSequencesTests(unittest.TestCase):
    def test_dedupes_and_drops_empty(self) -> None:
        results = [
            _fr("f1", ["a", "b"]),
            _fr("f2", ["a", "b"]),  # duplicate of f1
            _fr("f3", ["c"]),
            _fr("f4", []),  # empty -> dropped
        ]
        seqs = unique_function_sequences(results)
        self.assertEqual(seqs, [["a", "b"], ["c"]])  # deduped, sorted, no empty

    def test_deterministic_order(self) -> None:
        results = [_fr("f1", ["z"]), _fr("f2", ["a", "b"]), _fr("f3", ["a"])]
        self.assertEqual(unique_function_sequences(results), [["a"], ["a", "b"], ["z"]])


if __name__ == "__main__":
    unittest.main()
