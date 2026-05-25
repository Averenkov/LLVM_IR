from __future__ import annotations

import argparse
import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llvm_ir.stages.dataset import builder as dataset_builder
from llvm_ir.stages.function_search import pass_search
from llvm_ir.stages.function_search.algorithms import CEMPassSearch
from llvm_ir.stages.function_search.cem import CEMConfig


class PipelineIntegrationTests(unittest.TestCase):
    def test_dataset_output_feeds_function_search_stage(self) -> None:
        class FakeBenchmark:
            uri = "benchmark://demo-v0/unit"

        class FakeDataset:
            def __iter__(self):
                return iter(["benchmark://demo-v0/unit"])

            def benchmark(self, uri: str) -> FakeBenchmark:
                if uri != "benchmark://demo-v0/unit":
                    raise AssertionError(uri)
                return FakeBenchmark()

        class FakeEnv:
            datasets = {"demo-v0": FakeDataset()}

            def reset(self, benchmark: object) -> None:
                self.benchmark = benchmark

            def write_bitcode(self, path: str) -> None:
                Path(path).write_bytes(b"module bc")

            def close(self) -> None:
                self.closed = True

        def fake_extract(input_bc: Path, out_ll_dir: Path, runner=dataset_builder.run_tool):
            self.assertEqual(input_bc.name, "unit.bc")
            (out_ll_dir / "large.ll").write_text(
                "define i32 @large() {\n  ret i32 0\n}\n",
                encoding="utf-8",
            )
            (out_ll_dir / "small.ll").write_text(
                "define i32 @small() {\n  ret i32 0\n}\n",
                encoding="utf-8",
            )
            return {"large.ll": 100, "small.ll": 1}

        def fake_assembler(cmd: list[str]) -> None:
            self.assertEqual(cmd[0], "llvm-as")
            Path(cmd[-1]).write_bytes(b"function bc")

        with tempfile.TemporaryDirectory() as tmp_str, mock.patch.object(
            dataset_builder,
            "extract_functions_from_bc",
            fake_extract,
        ):
            tmp_path = Path(tmp_str)
            dataset_dir = tmp_path / "dataset"
            args = argparse.Namespace(
                dataset="demo-v0",
                output_dir=str(dataset_dir),
                work_dir=str(tmp_path / "work"),
                top_percent=50.0,
                max_benchmarks=None,
                benchmark_file=None,
                no_function_selection=False,
                overwrite=False,
                keep_intermediate=False,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                rc = dataset_builder.build_dataset(
                    args,
                    make_env=FakeEnv,
                    runner=fake_assembler,
                    tool_checker=lambda _name: "/usr/bin/tool",
                )

            self.assertEqual(rc, 0)
            bitcode_files = pass_search.select_bitcode_files(dataset_dir, limit=0, seed=1)
            self.assertEqual([path.name for path in bitcode_files], ["unit_large.bc"])

            def fake_apply(_input_bc: Path, _passes: list[str], output_bc: Path) -> None:
                output_bc.write_bytes(b"optimized bc")

            def fake_optimize(_input_bc: Path, output_bc: Path) -> None:
                output_bc.write_bytes(b"oz bc")

            def fake_measure(bitcode_path: Path, _workdir: Path) -> int:
                if bitcode_path.name == "oz.bc":
                    return 9
                if bitcode_path.name.endswith("prefix1.bc"):
                    return 7
                return 10

            algorithm = CEMPassSearch(
                CEMConfig(
                    steps=1,
                    iterations=1,
                    candidates=1,
                    elite_size=1,
                    allow_stop=False,
                    epsilon=0.0,
                )
            )
            with (
                mock.patch.object(pass_search, "apply_pass_sequence", fake_apply),
                mock.patch.object(pass_search, "measure_text_size", fake_measure),
                mock.patch.object(pass_search, "optimize_oz", fake_optimize),
            ):
                row = pass_search.run_search_for_function(
                    bitcode_files[0],
                    ["good-pass"],
                    algorithm=algorithm,
                    rng=random.Random(1),
                )

            self.assertEqual(row["function"], "unit_large.bc")
            self.assertEqual(row["search_algorithm"], "cem")
            self.assertEqual(row["cem_delta"], 3)
            self.assertEqual(row["cem_best_passes"], ["good-pass"])


if __name__ == "__main__":
    unittest.main()
