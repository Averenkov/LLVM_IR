from __future__ import annotations

import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult, TranslationUnitPlan
from llvm_ir.stages.translation_unit.order_graph import (
    PassOrderGraph,
    benchmark_id_from_function_name,
    build_pass_order_graph,
    build_pass_order_graphs_by_benchmark,
    load_function_pass_results_from_report,
    load_graphs_from_report,
    main as order_graph_main,
    support_weight_for_result,
    write_graph_report,
)
from llvm_ir.stages.translation_unit.beam_search import beam_search_path
from llvm_ir.stages.translation_unit.cycle_breaking_max_path import (
    CycleBreakingMaxPathConfig,
    cycle_breaking_max_path,
)
from llvm_ir.stages.translation_unit.dag_longest_path import dag_longest_path
from llvm_ir.stages.translation_unit.greedy_consensus import greedy_consensus_path
from llvm_ir.stages.translation_unit.path_heuristics import (
    HeuristicRunConfig,
    compare_heuristics,
    main as path_heuristics_main,
    parse_heuristics,
)
from llvm_ir.stages.translation_unit.path_scoring import score_path
from llvm_ir.stages.translation_unit.weighted_toposort import weighted_toposort_path
from llvm_ir.stages.translation_unit.evaluate import (
    TranslationUnitSequence,
    benchmark_uri_from_id,
    load_heuristic_sequences,
    summarize_evaluations,
    write_translation_unit_bitcodes,
)


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

    def test_pass_order_graph_adds_pairwise_edges_from_sequences(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="demo-v0_unit_f1.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b", "c"],
                ),
                FunctionPassResult(
                    function="demo-v0_unit_f2.bc",
                    baseline_size=10,
                    best_size=8,
                    passes=["c", "a"],
                ),
            ],
            benchmark="demo-v0_unit",
        )

        self.assertTrue(graph.has_edge("a", "b"))
        self.assertTrue(graph.has_edge("a", "c"))
        self.assertTrue(graph.has_edge("b", "c"))
        self.assertTrue(graph.has_edge("c", "a"))
        self.assertEqual(graph.edge_weight("a", "c"), 1)
        self.assertEqual(graph.function_count, 2)
        self.assertEqual(graph.sequence_count, 2)

    def test_pass_order_graph_can_use_delta_weights(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="demo-v0_unit_f1.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b"],
                ),
                FunctionPassResult(
                    function="demo-v0_unit_f2.bc",
                    baseline_size=10,
                    best_size=8,
                    passes=["a", "b"],
                ),
                FunctionPassResult(
                    function="demo-v0_unit_f3.bc",
                    baseline_size=10,
                    best_size=10,
                    passes=["a", "b"],
                ),
            ],
            benchmark="demo-v0_unit",
            weight_mode="delta",
        )

        self.assertEqual(graph.weight_mode, "delta")
        self.assertEqual(graph.edge_weight("a", "b"), 5)
        self.assertEqual(graph.function_count, 3)
        self.assertEqual(graph.sequence_count, 3)

    def test_support_weight_for_result_clamps_negative_delta(self) -> None:
        improved = FunctionPassResult(
            function="improved.bc",
            baseline_size=10,
            best_size=7,
            passes=["a"],
        )
        regressed = FunctionPassResult(
            function="regressed.bc",
            baseline_size=10,
            best_size=12,
            passes=["a"],
        )

        self.assertEqual(support_weight_for_result(improved, "count"), 1)
        self.assertEqual(support_weight_for_result(improved, "delta"), 3)
        self.assertEqual(support_weight_for_result(regressed, "delta"), 0)

    def test_pass_order_graph_to_dict_is_stable(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="demo-v0_unit_f1.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["b", "a", "c"],
                ),
            ],
            benchmark="demo-v0_unit",
            weight_mode="delta",
        )

        rendered = graph.to_dict()

        self.assertEqual(rendered["benchmark"], "demo-v0_unit")
        self.assertEqual(rendered["weight_mode"], "delta")
        self.assertEqual(rendered["nodes"], ["a", "b", "c"])
        self.assertEqual(
            rendered["edges"],
            [
                {"source": "a", "target": "c", "weight": 3},
                {"source": "b", "target": "a", "weight": 3},
                {"source": "b", "target": "c", "weight": 3},
            ],
        )

    def test_pass_order_graph_ignores_duplicate_pair_in_one_sequence(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="demo-v0_unit_f1.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b", "a", "b"],
                ),
            ],
            benchmark="demo-v0_unit",
        )

        self.assertEqual(graph.edge_weight("a", "b"), 1)
        self.assertEqual(graph.edge_weight("b", "a"), 1)
        self.assertFalse(graph.has_edge("a", "a"))

    def test_build_graphs_by_benchmark_groups_extracted_functions(self) -> None:
        graphs = build_pass_order_graphs_by_benchmark(
            [
                FunctionPassResult(
                    function="suite-v0_1_foo.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b"],
                ),
                FunctionPassResult(
                    function="suite-v0_1_bar.bc",
                    baseline_size=10,
                    best_size=8,
                    passes=["b", "c"],
                ),
                FunctionPassResult(
                    function="suite-v0_2_baz.bc",
                    baseline_size=10,
                    best_size=9,
                    passes=["x", "y"],
                ),
            ]
        )

        self.assertEqual(set(graphs), {"suite-v0_1", "suite-v0_2"})
        self.assertEqual(graphs["suite-v0_1"].function_count, 2)
        self.assertTrue(graphs["suite-v0_1"].has_edge("a", "b"))
        self.assertTrue(graphs["suite-v0_1"].has_edge("b", "c"))
        self.assertTrue(graphs["suite-v0_2"].has_edge("x", "y"))

    def test_build_graphs_by_benchmark_propagates_delta_weight_mode(self) -> None:
        graphs = build_pass_order_graphs_by_benchmark(
            [
                FunctionPassResult(
                    function="suite-v0_1_foo.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b"],
                ),
                FunctionPassResult(
                    function="suite-v0_1_bar.bc",
                    baseline_size=10,
                    best_size=8,
                    passes=["a", "b"],
                ),
            ],
            weight_mode="delta",
        )

        graph = graphs["suite-v0_1"]
        self.assertEqual(graph.weight_mode, "delta")
        self.assertEqual(graph.edge_weight("a", "b"), 5)

    def test_benchmark_id_from_function_name_uses_suite_and_benchmark(self) -> None:
        self.assertEqual(
            benchmark_id_from_function_name("tensorflow-v0_1985_mangled.bc"),
            "tensorflow-v0_1985",
        )
        self.assertEqual(
            benchmark_id_from_function_name("chstone-v0_gsm_Autocorrelation.bc"),
            "chstone-v0_gsm",
        )

    def test_load_function_pass_results_from_report(self) -> None:
        report = {
            "config": {"algorithm": "random"},
            "rows": [
                {
                    "function": "suite-v0_1_foo.bc",
                    "search_algorithm": "random",
                    "baseline_size": 10,
                    "random_best_size": 7,
                    "random_best_passes": ["a", "b"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_str:
            path = Path(tmp_str) / "comparison.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            results = load_function_pass_results_from_report(path)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].function, "suite-v0_1_foo.bc")
        self.assertEqual(results[0].passes, ["a", "b"])

    def test_load_function_pass_results_can_override_algorithm(self) -> None:
        report = {
            "config": {"algorithm": "cem"},
            "rows": [
                {
                    "function": "suite-v0_1_foo.bc",
                    "search_algorithm": "cem",
                    "baseline_size": 10,
                    "cem_best_size": 8,
                    "cem_best_passes": ["cem-pass"],
                    "random_best_size": 7,
                    "random_best_passes": ["random-pass"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_str:
            path = Path(tmp_str) / "comparison.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            results = load_function_pass_results_from_report(path, algorithm="random")

        self.assertEqual(results[0].best_size, 7)
        self.assertEqual(results[0].passes, ["random-pass"])

    def test_write_graph_report_persists_weight_mode_and_edges(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="suite-v0_1_foo.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b"],
                ),
            ],
            benchmark="suite-v0_1",
            weight_mode="delta",
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            output = Path(tmp_str) / "graphs" / "order_graphs.json"
            write_graph_report(
                {"suite-v0_1": graph},
                output,
                input_path=Path("comparison.json"),
                algorithm="cem",
                weight_mode="delta",
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["algorithm"], "cem")
        self.assertEqual(payload["weight_mode"], "delta")
        self.assertEqual(payload["benchmark_count"], 1)
        self.assertEqual(payload["graphs"][0]["edges"][0]["weight"], 3)

    def test_load_graphs_from_report_round_trips_graphs(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="suite-v0_1_foo.bc",
                    baseline_size=10,
                    best_size=7,
                    passes=["a", "b"],
                ),
            ],
            benchmark="suite-v0_1",
            weight_mode="delta",
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            output = Path(tmp_str) / "order_graphs.json"
            write_graph_report(
                {"suite-v0_1": graph},
                output,
                input_path=Path("comparison.json"),
                algorithm="cem",
                weight_mode="delta",
            )
            loaded = load_graphs_from_report(output)

        self.assertEqual(set(loaded), {"suite-v0_1"})
        self.assertEqual(loaded["suite-v0_1"].weight_mode, "delta")
        self.assertEqual(loaded["suite-v0_1"].edge_weight("a", "b"), 3)

    def test_order_graph_cli_builds_delta_graph_report(self) -> None:
        report = {
            "config": {"algorithm": "cem"},
            "rows": [
                {
                    "function": "suite-v0_1_foo.bc",
                    "search_algorithm": "cem",
                    "baseline_size": 10,
                    "cem_best_size": 7,
                    "cem_best_passes": ["a", "b"],
                },
                {
                    "function": "suite-v0_1_bar.bc",
                    "search_algorithm": "cem",
                    "baseline_size": 10,
                    "cem_best_size": 8,
                    "cem_best_passes": ["a", "b"],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            input_path = root / "comparison.json"
            output_path = root / "order_graphs.json"
            input_path.write_text(json.dumps(report), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                rc = order_graph_main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--algorithm",
                        "cem",
                        "--weight-mode",
                        "delta",
                    ]
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(payload["weight_mode"], "delta")
        self.assertEqual(payload["graphs"][0]["edges"][0]["weight"], 5)

    def test_greedy_consensus_orders_by_graph_priority(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("f1.bc", 10, 7, ["a", "b", "c"]),
                FunctionPassResult("f2.bc", 10, 8, ["a", "c"]),
            ],
            benchmark="demo",
        )

        self.assertEqual(greedy_consensus_path(graph), ["a", "b", "c"])

    def test_weighted_toposort_respects_dag_order(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("f1.bc", 10, 7, ["a", "b", "d"]),
                FunctionPassResult("f2.bc", 10, 8, ["a", "c", "d"]),
            ],
            benchmark="demo",
        )

        path = weighted_toposort_path(graph)

        self.assertLess(path.index("a"), path.index("b"))
        self.assertLess(path.index("a"), path.index("c"))
        self.assertLess(path.index("b"), path.index("d"))
        self.assertLess(path.index("c"), path.index("d"))

    def test_weighted_toposort_breaks_cycles_by_net_support(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("f1.bc", 10, 7, ["a", "b"]),
                FunctionPassResult("f2.bc", 10, 8, ["a", "b"]),
                FunctionPassResult("f3.bc", 10, 9, ["b", "a"]),
            ],
            benchmark="demo",
        )

        self.assertEqual(weighted_toposort_path(graph), ["a", "b"])

    def test_dag_longest_path_uses_pairwise_net_support(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("f1.bc", 10, 7, ["a", "b", "c"]),
                FunctionPassResult("f2.bc", 10, 8, ["a", "b", "c"]),
                FunctionPassResult("f3.bc", 10, 9, ["c", "b"]),
            ],
            benchmark="demo",
        )

        self.assertEqual(dag_longest_path(graph), ["a", "b", "c"])

    def test_cycle_breaking_max_path_removes_weakest_cycle_edge(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b"], support_weight=5)
        graph.add_sequence(["b", "c"], support_weight=4)
        graph.add_sequence(["c", "a"], support_weight=1)
        graph.add_sequence(["c", "d"], support_weight=6)

        self.assertEqual(cycle_breaking_max_path(graph), ["a", "b", "c", "d"])

    def test_cycle_breaking_max_path_respects_max_length(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b"], support_weight=5)
        graph.add_sequence(["b", "c"], support_weight=4)
        graph.add_sequence(["c", "a"], support_weight=1)
        graph.add_sequence(["c", "d"], support_weight=6)

        path = cycle_breaking_max_path(
            graph,
            config=CycleBreakingMaxPathConfig(max_length=3),
        )

        self.assertEqual(path, ["b", "c", "d"])

    def test_beam_search_finds_high_net_score_path(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("f1.bc", 10, 7, ["a", "b", "c"]),
                FunctionPassResult("f2.bc", 10, 8, ["a", "b", "c"]),
                FunctionPassResult("f3.bc", 10, 9, ["c", "b"]),
            ],
            benchmark="demo",
        )

        path = beam_search_path(graph)
        self.assertEqual(path[:3], ["a", "b", "c"])
        self.assertGreater(score_path(graph, path).net_score, 0)

    def test_compare_heuristics_scores_all_requested_methods(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("suite-v0_1_foo.bc", 10, 7, ["a", "b", "c"]),
                FunctionPassResult("suite-v0_1_bar.bc", 10, 8, ["a", "c"]),
            ],
            benchmark="suite-v0_1",
        )

        report = compare_heuristics(
            {"suite-v0_1": graph},
            [
                "greedy_consensus",
                "dag_longest_path",
                "cycle_breaking_max_path",
                "beam_search",
                "weighted_toposort",
            ],
            config=HeuristicRunConfig(max_length=3),
        )

        self.assertEqual(
            set(report["summary"]),
            {
                "greedy_consensus",
                "dag_longest_path",
                "cycle_breaking_max_path",
                "beam_search",
                "weighted_toposort",
            },
        )
        self.assertEqual(len(report["results"]), 5)
        self.assertTrue(all(item["path"] for item in report["results"]))

    def test_parse_heuristics_expands_all_and_csv(self) -> None:
        self.assertEqual(
            parse_heuristics("all"),
            [
                "greedy_consensus",
                "dag_longest_path",
                "cycle_breaking_max_path",
                "beam_search",
                "weighted_toposort",
            ],
        )
        self.assertEqual(parse_heuristics("beam_search,greedy_consensus"), [
            "beam_search",
            "greedy_consensus",
        ])

    def test_path_heuristics_cli_writes_report(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult("suite-v0_1_foo.bc", 10, 7, ["a", "b", "c"]),
                FunctionPassResult("suite-v0_1_bar.bc", 10, 8, ["a", "c"]),
            ],
            benchmark="suite-v0_1",
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            input_path = root / "order_graphs.json"
            output_path = root / "heuristics.json"
            write_graph_report(
                {"suite-v0_1": graph},
                input_path,
                input_path=Path("comparison.json"),
                algorithm="cem",
                weight_mode="count",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = path_heuristics_main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--heuristics",
                        "greedy_consensus",
                        "--max-length",
                        "3",
                    ]
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(set(payload["summary"]), {"greedy_consensus"})
        self.assertEqual(payload["results"][0]["path"], ["a", "b", "c"])

    def test_benchmark_uri_from_id_reconstructs_compiler_gym_uri(self) -> None:
        self.assertEqual(
            benchmark_uri_from_id("tensorflow-v0_1985"),
            "benchmark://tensorflow-v0/1985",
        )
        self.assertEqual(
            benchmark_uri_from_id("mibench-v1_lame-newmdct-1"),
            "benchmark://mibench-v1/lame-newmdct-1",
        )

    def test_load_heuristic_sequences_reads_paths(self) -> None:
        report = {
            "results": [
                {
                    "benchmark": "suite-v0_1",
                    "heuristic": "beam_search",
                    "path": ["a", "b"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_str:
            path = Path(tmp_str) / "heuristics.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            sequences = load_heuristic_sequences(path)

        self.assertEqual(
            sequences,
            [
                TranslationUnitSequence(
                    benchmark="suite-v0_1",
                    heuristic="beam_search",
                    passes=["a", "b"],
                )
            ],
        )

    def test_summarize_evaluations_groups_by_heuristic(self) -> None:
        summary = summarize_evaluations(
            [
                {
                    "heuristic": "h1",
                    "baseline_size": 100,
                    "oz_size": 80,
                    "final_size": 90,
                    "final_delta": 10,
                    "best_size": 70,
                    "best_delta": 30,
                    "best_prefix_len": 2,
                    "error": "",
                },
                {
                    "heuristic": "h1",
                    "baseline_size": 100,
                    "oz_size": 80,
                    "final_size": 105,
                    "final_delta": -5,
                    "best_size": 95,
                    "best_delta": 5,
                    "best_prefix_len": 1,
                    "error": "failed",
                },
            ]
        )

        self.assertEqual(summary["h1"]["benchmarks"], 2)
        self.assertEqual(summary["h1"]["failed"], 1)
        self.assertEqual(summary["h1"]["improved_final"], 1)
        self.assertEqual(summary["h1"]["improved_best"], 2)
        self.assertEqual(summary["h1"]["total_best_delta"], 35)
        self.assertEqual(summary["h1"]["beats_oz_best"], 1)

    def test_write_translation_unit_bitcodes_uses_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            existing = root / "suite-v0_1.bc"
            existing.write_bytes(b"bc")

            def fail_make_env():
                raise AssertionError("env should not be created for existing bc")

            paths = write_translation_unit_bitcodes(
                ["suite-v0_1"],
                root,
                make_env=fail_make_env,
            )

        self.assertEqual(paths["suite-v0_1"], existing)

    def test_write_translation_unit_bitcodes_copies_direct_site_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            site_data = root / "site_data"
            source = (
                site_data
                / "llvm-v0"
                / "benchmark"
                / "suite-v0"
                / "contents"
                / "suite-v0"
                / "1.bc"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"direct")
            output_dir = root / "bitcodes"

            def fail_make_env():
                raise AssertionError("env should not be created for local site-data")

            paths = write_translation_unit_bitcodes(
                ["suite-v0_1"],
                output_dir,
                site_data_paths=[site_data],
                make_env=fail_make_env,
            )

            self.assertEqual(paths["suite-v0_1"].read_bytes(), b"direct")

    def test_write_translation_unit_bitcodes_copies_recursive_site_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            site_data = root / "site_data"
            source = (
                site_data
                / "llvm-v0"
                / "benchmark"
                / "chstone-v0"
                / "contents"
                / "nested"
                / "benchmarks"
                / "CHStone"
                / "aes.bc"
            )
            source.parent.mkdir(parents=True)
            source.write_bytes(b"recursive")
            output_dir = root / "bitcodes"

            def fail_make_env():
                raise AssertionError("env should not be created for local site-data")

            paths = write_translation_unit_bitcodes(
                ["chstone-v0_aes"],
                output_dir,
                site_data_paths=[site_data],
                make_env=fail_make_env,
            )

            self.assertEqual(paths["chstone-v0_aes"].read_bytes(), b"recursive")


if __name__ == "__main__":
    unittest.main()
