from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from llvm_ir.stages.dataset import builder


class DatasetBuilderTests(unittest.TestCase):
    def test_sanitize_filename_replaces_unsafe_characters(self) -> None:
        self.assertEqual(
            builder.sanitize_filename("cbench-v1/qsort::main$1"),
            "cbench-v1_qsort__main_1",
        )

    def test_parse_function_names_preserves_order_and_skips_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            ll_path = Path(tmp_str) / "module.ll"
            ll_path.write_text(
                """
declare i32 @external(i32)
define dso_local i32 @main(i32 %argc) {
  ret i32 %argc
}
define internal void @"quoted.name"() {
  ret void
}
define dso_local i32 @main(i32 %argc) {
  ret i32 %argc
}
""",
                encoding="utf-8",
            )

            self.assertEqual(builder.parse_function_names(ll_path), ["main", "quoted.name"])

    def test_safe_function_stem_shortens_long_names_with_hash(self) -> None:
        long_name = "_ZN" + "VeryLongTemplateName" * 20

        stem = builder.safe_function_stem(long_name, max_len=64)

        self.assertLessEqual(len(stem), 64)
        self.assertRegex(stem, r"_[0-9a-f]{12}$")
        self.assertEqual(builder.parse_llvm_symbol_name('"name$with$dollars"'), "name$with$dollars")


    def test_count_llvm_ir_instructions_uses_line_based_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            ll_path = Path(tmp_str) / "func.ll"
            ll_path.write_text(
                """
; comment
target triple = "x86_64"
define i32 @foo() {
entry:
  %x = add i32 1, 2
  br label %exit
exit:
  ret i32 %x
}
attributes #0 = { nounwind }
!0 = !{}
""",
                encoding="utf-8",
            )

            self.assertEqual(builder.count_llvm_ir_instructions(ll_path), 4)


    def test_select_top_functions_keeps_at_least_one_and_sorts_by_size_then_name(self) -> None:
        functions = {"b.ll": 10, "a.ll": 10, "c.ll": 1, "d.ll": 9}

        self.assertEqual(builder.select_top_functions(functions, 50.0), [("a.ll", 10), ("b.ll", 10)])
        self.assertEqual(builder.select_top_functions(functions, 1.0), [("a.ll", 10)])


    def test_select_top_functions_rejects_non_positive_percent(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_percent"):
            builder.select_top_functions({"a.ll": 1}, 0)


    def test_select_functions_can_keep_all_functions(self) -> None:
        functions = {"b.ll": 10, "a.ll": 1}

        self.assertEqual(
            builder.select_functions(functions, None),
            [("a.ll", 1), ("b.ll", 10)],
        )


    def test_iter_benchmarks_is_deterministic_and_respects_limit(self) -> None:
        class FakeBenchmark:
            def __init__(self, uri: str) -> None:
                self.uri = uri

        class FakeDataset:
            def __iter__(self):
                return iter(["benchmark://z", "benchmark://a", "benchmark://m"])

            def benchmark(self, uri: str) -> FakeBenchmark:
                return FakeBenchmark(uri)

        class FakeEnv:
            datasets = {"demo": FakeDataset()}

        uris = [benchmark.uri for benchmark in builder.iter_benchmarks(FakeEnv(), "demo", 2)]

        self.assertEqual(uris, ["benchmark://a", "benchmark://m"])

    def test_read_benchmark_uris_reads_benchmark_set_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            csv_path = Path(tmp_str) / "benchmarks.csv"
            csv_path.write_text(
                "suite,benchmark_uri\n"
                "chstone,benchmark://chstone-v0/aes\n"
                "mibench,benchmark://mibench-v1/qsort\n",
                encoding="utf-8",
            )

            self.assertEqual(
                builder.read_benchmark_uris(csv_path),
                ["benchmark://chstone-v0/aes", "benchmark://mibench-v1/qsort"],
            )

    def test_iter_benchmark_uris_uses_dataset_from_each_uri(self) -> None:
        class FakeBenchmark:
            def __init__(self, uri: str) -> None:
                self.uri = uri

        class FakeDataset:
            def __init__(self, key: str) -> None:
                self.key = key

            def benchmark(self, uri: str) -> FakeBenchmark:
                if not uri.startswith(f"benchmark://{self.key}/"):
                    raise AssertionError(uri)
                return FakeBenchmark(uri)

        class FakeEnv:
            datasets = {
                "chstone-v0": FakeDataset("chstone-v0"),
                "mibench-v1": FakeDataset("mibench-v1"),
            }

        uris = [
            benchmark.uri
            for benchmark in builder.iter_benchmark_uris(
                FakeEnv(),
                ["benchmark://chstone-v0/aes", "benchmark://mibench-v1/qsort"],
                None,
            )
        ]

        self.assertEqual(uris, ["benchmark://chstone-v0/aes", "benchmark://mibench-v1/qsort"])
        self.assertEqual(
            builder.benchmark_short_name(FakeBenchmark("benchmark://npb-v0/1"), include_dataset=True),
            "npb-v0__1",
        )


    def test_benchmark_short_name_preserves_benchmark_underscores(self) -> None:
        class FakeBenchmark:
            uri = "benchmark://cbench-v1/qsort_main"

        self.assertEqual(
            builder.benchmark_short_name(FakeBenchmark(), include_dataset=True),
            "cbench-v1__qsort_main",
        )

    def test_build_dataset_orchestrates_compiler_gym_and_llvm_tools(self) -> None:
        class FakeBenchmark:
            uri = "benchmark://cbench-v1/qsort_main"

        class FakeDataset:
            def __iter__(self):
                return iter(["benchmark://cbench-v1/qsort_main"])

            def benchmark(self, uri: str) -> FakeBenchmark:
                self_uri = "benchmark://cbench-v1/qsort_main"
                if uri != self_uri:
                    raise AssertionError(uri)
                return FakeBenchmark()

        class FakeEnv:
            datasets = {"cbench-v1": FakeDataset()}

            def __init__(self) -> None:
                self.closed = False
                self.resets: list[object] = []

            def reset(self, benchmark: object) -> None:
                self.resets.append(benchmark)

            def write_bitcode(self, path: str) -> None:
                Path(path).write_bytes(b"fake bc")

            def close(self) -> None:
                self.closed = True

        fake_env = FakeEnv()

        def fake_extract(input_bc: Path, out_ll_dir: Path, runner=builder.run_tool):
            self.assertEqual(input_bc.name, "cbench-v1__qsort_main.bc")
            (out_ll_dir / "large.ll").write_text("define i32 @large() {\n  ret i32 0\n}\n")
            (out_ll_dir / "small.ll").write_text("define i32 @small() {\n  ret i32 0\n}\n")
            return {"large.ll": 100, "small.ll": 1}

        assembled: list[Path] = []

        def fake_runner(cmd: list[str]) -> None:
            self.assertEqual(cmd[0], "llvm-as")
            output = Path(cmd[-1])
            output.write_bytes(b"assembled")
            assembled.append(output)

        with tempfile.TemporaryDirectory() as tmp_str, mock.patch.object(
            builder, "extract_functions_from_bc", fake_extract
        ):
            tmp_path = Path(tmp_str)
            args = argparse.Namespace(
                dataset="cbench-v1",
                output_dir=str(tmp_path / "out"),
                work_dir=str(tmp_path / "work"),
                top_percent=50.0,
                max_benchmarks=None,
                benchmark_file=None,
                no_function_selection=False,
                overwrite=False,
                keep_intermediate=False,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                rc = builder.build_dataset(
                    args,
                    make_env=lambda: fake_env,
                    runner=fake_runner,
                    tool_checker=lambda _name: "/usr/bin/tool",
                )

            self.assertEqual(rc, 0)
            self.assertTrue(fake_env.closed)
            self.assertEqual(len(fake_env.resets), 1)
            self.assertEqual([path.name for path in assembled], ["cbench-v1__qsort_main__large.bc"])
            self.assertEqual(
                (tmp_path / "out" / "cbench-v1__qsort_main__large.bc").read_bytes(),
                b"assembled",
            )
            self.assertFalse((tmp_path / "work").exists())


if __name__ == "__main__":
    unittest.main()
