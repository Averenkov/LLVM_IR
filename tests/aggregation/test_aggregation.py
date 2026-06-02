from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from llvm_ir.heuristics.aggregation.base import (
    Dataset,
    PerFunctionResult,
    build_pass_graph,
)
from llvm_ir.heuristics.aggregation.export_paths import export_aggregation_paths
from llvm_ir.heuristics.aggregation.graph_utils import eades_order
from llvm_ir.heuristics.aggregation.registry import available_heuristics, build_heuristic
from llvm_ir.heuristics.aggregation.tu_eval import TUEvaluator
from llvm_ir.heuristics.aggregation.voting_ensemble import VotingEnsemble
from llvm_ir.heuristics.aggregation.wfas_eades import WFASEades


def demo_dataset() -> Dataset:
    return Dataset(
        results=[
            PerFunctionResult("f1", ["a", "b", "c"], 100, 80),
            PerFunctionResult("f2", ["a", "c", "d"], 120, 90),
            PerFunctionResult("f3", ["b", "c"], 90, 70),
            PerFunctionResult("f4", ["d", "a"], 50, 55),
        ],
        name="demo",
    )


class AggregationHeuristicTests(unittest.TestCase):
    def test_all_registered_heuristics_return_valid_sequences(self) -> None:
        dataset = demo_dataset()
        graph = build_pass_graph(dataset.results)

        for name in available_heuristics():
            with self.subTest(name=name):
                result = build_heuristic(name).aggregate(dataset, graph, {})
                for sequence in result.sequences:
                    self.assertEqual(len(sequence), len(set(sequence)))
                    self.assertTrue(set(sequence).issubset(graph.nodes))
                self.assertTrue(set(result.chosen_sequence).issubset(graph.nodes))
                self.assertLessEqual(
                    result.chosen_prefix_length,
                    len(result.chosen_sequence),
                )

    def test_wfas_eades_is_topological_on_dag(self) -> None:
        nodes = {"a", "b", "c"}
        weights = {("a", "b"): 1.0, ("b", "c"): 1.0, ("a", "c"): 1.0}

        self.assertEqual(eades_order(nodes, weights), ["a", "b", "c"])

    def test_voting_preserves_order_for_identical_voters(self) -> None:
        dataset = demo_dataset()
        graph = build_pass_graph(dataset.results)
        expected = WFASEades().aggregate(dataset, graph, {}).chosen_sequence

        result = VotingEnsemble().aggregate(
            dataset,
            graph,
            {"voters": ["wfas_eades", "wfas_eades"]},
        )

        self.assertEqual(result.chosen_sequence, expected)

    def test_hpp_eades_topk_has_no_duplicates_and_respects_k(self) -> None:
        dataset = demo_dataset()
        graph = build_pass_graph(dataset.results)
        result = build_heuristic("hpp_eades_topk").aggregate(
            dataset,
            graph,
            {"top_k": 3, "beam_width": 4},
        )

        self.assertLessEqual(len(result.sequences), 4)
        self.assertEqual(
            len({tuple(sequence) for sequence in result.sequences}),
            len(result.sequences),
        )

    def test_export_aggregation_paths_uses_evaluator_report_shape(self) -> None:
        dataset = Dataset(
            results=[
                PerFunctionResult("suite-v0_1_f1.bc", ["a", "b"], 10, 7),
                PerFunctionResult("suite-v0_1_f2.bc", ["a", "c"], 10, 8),
                PerFunctionResult("suite-v0_2_f1.bc", ["d"], 10, 9),
            ],
            name="demo",
        )

        report = export_aggregation_paths(
            dataset,
            ["wfas_eades", "hpp_eades_topk"],
            max_length=2,
        )

        self.assertEqual(len(report["results"]), 4)
        self.assertEqual(
            {row["benchmark"] for row in report["results"]},
            {"suite-v0_1", "suite-v0_2"},
        )
        self.assertTrue(
            all({"benchmark", "heuristic", "path"} <= set(row) for row in report["results"])
        )
        self.assertTrue(all(len(row["path"]) <= 2 for row in report["results"]))

    def test_tu_evaluator_uses_disk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            tu = root / "input.bc"
            tu.write_bytes(b"bitcode")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            counter = root / "counter.txt"
            counter.write_text("0", encoding="utf-8")
            opt = _write_script(
                bin_dir / "opt",
                f"""#!/usr/bin/env python3
from pathlib import Path
counter = Path({str(counter)!r})
counter.write_text(str(int(counter.read_text()) + 1))
args = __import__('sys').argv
Path(args[args.index('-o') + 1]).write_bytes(Path(args[-3]).read_bytes())
""",
            )
            llc = _write_script(
                bin_dir / "llc",
                """#!/usr/bin/env python3
from pathlib import Path
args = __import__('sys').argv
Path(args[args.index('-o') + 1]).write_text('obj')
""",
            )
            size = _write_script(
                bin_dir / "llvm-size",
                """#!/usr/bin/env python3
print('text data bss dec hex filename')
print('42 0 0 42 2a file.o')
""",
            )
            evaluator = TUEvaluator(
                tu,
                opt_bin=str(opt),
                llc_bin=str(llc),
                size_bin=str(size),
                cache_dir=root / "cache",
            )

            first = evaluator.evaluate(["a"])
            second = evaluator.evaluate(["a"])

            self.assertTrue(first.success)
            self.assertEqual(first.text_size, 42)
            self.assertEqual(second.text_size, 42)
            self.assertEqual(counter.read_text(encoding="utf-8"), "1")


def _write_script(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | os.stat(path).st_mode | 0o111)
    return path


if __name__ == "__main__":
    unittest.main()
