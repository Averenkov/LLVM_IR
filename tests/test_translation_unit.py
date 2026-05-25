from __future__ import annotations

import unittest
from pathlib import Path

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult, TranslationUnitPlan


class TranslationUnitContractTests(unittest.TestCase):
    def test_translation_unit_plan_keeps_function_level_results(self) -> None:
        function_result = FunctionPassResult(
            function="foo.bc",
            baseline_size=10,
            best_size=7,
            passes=["instcombine"],
        )
        plan = TranslationUnitPlan(
            bitcode_path=Path("module.bc"),
            passes=["instcombine"],
            source_results=[function_result],
        )

        self.assertEqual(function_result.delta, 3)
        self.assertEqual(plan.source_results, [function_result])


if __name__ == "__main__":
    unittest.main()
