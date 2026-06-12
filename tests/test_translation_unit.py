from __future__ import annotations

import unittest
import contextlib
import io
import json
import tempfile
import time
from pathlib import Path

from llvm_ir.stages.translation_unit.contracts import FunctionPassResult, TranslationUnitPlan
from llvm_ir.stages.translation_unit.graph.order_graph import (
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
from llvm_ir.heuristics.translation_unit.cycle_breaking_max_path import (
    CycleBreakingMaxPathConfig,
    cycle_breaking_diverse_start_paths,
    cycle_breaking_max_path,
    cycle_breaking_max_paths,
    cycle_breaking_top_start_paths,
)
from llvm_ir.heuristics.translation_unit.exhaustive_path import (
    ExhaustivePathConfig,
    exhaustive_fixed_length_path,
)
from llvm_ir.heuristics.translation_unit.random_walk import (
    RandomWalkPathConfig,
    random_walk_path,
)
from llvm_ir.heuristics.translation_unit.chunk_forest import (
    CandidatePath,
    Chunk,
    ChunkInventoryConfig,
    ChunkRef,
    ChunkSelectionConfig,
    ChunkWalkConfig,
    build_chunk_graph,
    generate_candidate_pool,
    mine_chunks,
    select_paths,
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
from llvm_ir.stages.translation_unit import evaluate_topk_paths
from llvm_ir.stages.translation_unit import evaluate_chunk_forest
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
        self.assertEqual(graph.start_counts["a"], 5)
        self.assertEqual(graph.function_count, 3)
        self.assertEqual(graph.sequence_count, 3)

    def test_pass_order_graph_can_make_nearby_pairs_stronger(self) -> None:
        graph = build_pass_order_graph(
            [
                FunctionPassResult(
                    function="demo-v0_unit_f1.bc",
                    baseline_size=10,
                    best_size=8,
                    passes=["a", "b", "c", "d"],
                ),
            ],
            benchmark="demo-v0_unit",
            weight_mode="delta_distance",
        )

        self.assertEqual(graph.weight_mode, "delta_distance")
        self.assertEqual(graph.start_counts["a"], 2)
        self.assertEqual(graph.edge_weight("a", "b"), 24)
        self.assertEqual(graph.edge_weight("a", "c"), 12)
        self.assertEqual(graph.edge_weight("a", "d"), 8)
        self.assertEqual(graph.edge_weight("b", "c"), 24)
        self.assertEqual(graph.edge_weight("b", "d"), 12)
        self.assertEqual(graph.edge_weight("c", "d"), 24)

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
        self.assertEqual(support_weight_for_result(improved, "count_distance"), 1)
        self.assertEqual(support_weight_for_result(improved, "delta"), 3)
        self.assertEqual(support_weight_for_result(improved, "delta_distance"), 3)
        self.assertEqual(support_weight_for_result(regressed, "delta"), 0)
        self.assertEqual(support_weight_for_result(regressed, "delta_distance"), 0)

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
        self.assertEqual(rendered["start_counts"], [{"pass": "b", "weight": 3}])
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
        self.assertEqual(
            benchmark_id_from_function_name("chstone-v0_adpcm_decode.bc"),
            "chstone-v0_adpcm",
        )
        self.assertEqual(
            benchmark_id_from_function_name("mibench-v1_lame-newmdct-1_mdct.bc"),
            "mibench-v1_lame-newmdct-1",
        )
        self.assertEqual(
            benchmark_id_from_function_name("mibench-v1_jpeg-c_astex_codelet__2.bc"),
            "mibench-v1_jpeg-c",
        )
        self.assertEqual(
            benchmark_id_from_function_name("opencv-v0_4_ZNSt3__116__pad_and_output.bc"),
            "opencv-v0_4",
        )
        self.assertEqual(
            benchmark_id_from_function_name("tensorflow-v0_1985_ZN5Eigen__internal.bc"),
            "tensorflow-v0_1985",
        )
        self.assertEqual(
            benchmark_id_from_function_name("cbench-v1__qsort_main__main.bc"),
            "cbench-v1_qsort_main",
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

    def test_dag_longest_path_ignores_reverse_priority_edges(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update({"a", "b", "c"})
        graph.edge_counts[("a", "c")] = 100
        graph.edge_counts[("b", "a")] = 10

        self.assertEqual(dag_longest_path(graph), ["a", "c"])

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

    def test_cycle_breaking_max_paths_returns_ranked_unique_paths(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "e"])
        graph.edge_counts.update(
            {
                ("a", "b"): 10,
                ("b", "d"): 10,
                ("a", "c"): 8,
                ("c", "d"): 8,
                ("a", "e"): 6,
                ("e", "d"): 6,
            }
        )

        paths = cycle_breaking_max_paths(
            graph,
            config=CycleBreakingMaxPathConfig(max_length=3),
            top_k=2,
        )

        self.assertEqual(paths, [["a", "b", "d"], ["a", "c", "d"]])
        self.assertEqual(cycle_breaking_max_path(graph), paths[0])

    def test_cycle_breaking_diverse_start_paths_uses_start_counts(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "x", "y", "z"])
        graph.start_counts.update({"x": 30, "a": 20, "z": 10})
        graph.edge_counts.update(
            {
                ("a", "b"): 10,
                ("b", "c"): 10,
                ("x", "y"): 3,
                ("y", "c"): 3,
                ("z", "d"): 5,
            }
        )

        paths = cycle_breaking_diverse_start_paths(
            graph,
            config=CycleBreakingMaxPathConfig(max_length=3),
            top_k=3,
        )

        self.assertEqual([path[0] for path in paths], ["x", "a", "z"])
        self.assertEqual(paths[1], ["a", "b", "c"])

    def test_cycle_breaking_top_start_paths_returns_top_paths_per_start(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "x", "y", "e"])
        graph.start_counts.update({"a": 30, "x": 20})
        graph.edge_counts.update(
            {
                ("a", "b"): 10,
                ("b", "d"): 10,
                ("a", "c"): 8,
                ("c", "d"): 8,
                ("x", "y"): 7,
                ("y", "d"): 7,
                ("x", "e"): 6,
                ("e", "d"): 6,
            }
        )

        paths = cycle_breaking_top_start_paths(
            graph,
            config=CycleBreakingMaxPathConfig(max_length=3),
            top_starts=2,
            paths_per_start=2,
        )

        self.assertEqual(
            paths,
            [
                ["a", "b", "d"],
                ["a", "c", "d"],
                ["x", "y", "d"],
                ["x", "e", "d"],
            ],
        )

    def test_exhaustive_fixed_length_path_selects_best_length_six_path(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b", "c", "d", "e", "f"], support_weight=10)
        graph.add_sequence(["a", "x", "y", "z", "q", "r"], support_weight=1)

        path = exhaustive_fixed_length_path(
            graph,
            config=ExhaustivePathConfig(path_length=6),
        )

        self.assertEqual(path, ["a", "b", "c", "d", "e", "f"])

    def test_exhaustive_fixed_length_path_falls_back_when_no_length_six_path(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b", "c"], support_weight=10)

        path = exhaustive_fixed_length_path(
            graph,
            config=ExhaustivePathConfig(path_length=6),
        )

        self.assertEqual(path, ["a", "b", "c"])

    def test_random_walk_path_is_reproducible_and_uses_graph_edges(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b"], support_weight=10)
        graph.add_sequence(["b", "c"], support_weight=10)
        graph.add_sequence(["a", "d"], support_weight=1)

        config = RandomWalkPathConfig(max_length=3, walks=64, seed=7)
        first = random_walk_path(graph, config=config)
        second = random_walk_path(graph, config=config)

        self.assertEqual(first, second)
        self.assertEqual(first, ["a", "b", "c"])

    def test_random_walk_path_respects_max_length(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b", "c", "d"], support_weight=10)

        path = random_walk_path(
            graph,
            config=RandomWalkPathConfig(max_length=2, walks=32, seed=3),
        )

        self.assertLessEqual(len(path), 2)
        self.assertTrue(path)

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
                "exhaustive_len6",
                "random_walk",
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
                "exhaustive_len6",
                "random_walk",
                "beam_search",
                "weighted_toposort",
            },
        )
        self.assertEqual(len(report["results"]), 7)
        self.assertTrue(all(item["path"] for item in report["results"]))

    def test_parse_heuristics_expands_all_and_csv(self) -> None:
        self.assertEqual(
            parse_heuristics("all"),
            [
                "greedy_consensus",
                "dag_longest_path",
                "cycle_breaking_max_path",
                "exhaustive_len6",
                "random_walk",
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
        self.assertEqual(
            benchmark_uri_from_id("cbench-v1_qsort_main"),
            "benchmark://cbench-v1/qsort_main",
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

    def test_generate_topk_paths_accepts_random_walk_topk_alias(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.add_sequence(["a", "b", "c"], support_weight=10)
        args = type(
            "Args",
            (),
            {
                "max_length": 3,
                "random_walks": 16,
                "random_seed": 7,
                "min_edge_weight": 1,
                "top_k": 5,
            },
        )

        paths = evaluate_topk_paths.generate_topk_paths(
            graph,
            "random_walk_topk",
            args,
        )

        self.assertGreaterEqual(len(paths), 1)
        self.assertLessEqual(len(paths), 5)

    def test_superpath_candidates_combine_segment_delta_and_edge_weight(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.edge_counts[("d", "e")] = 20
        segments = [
            evaluate_topk_paths.SuperSegmentCandidate(
                index=1,
                passes=("a", "b", "c", "d"),
                vertex_delta=10,
            ),
            evaluate_topk_paths.SuperSegmentCandidate(
                index=2,
                passes=("e", "f", "g", "h"),
                vertex_delta=5,
            ),
            evaluate_topk_paths.SuperSegmentCandidate(
                index=3,
                passes=("x", "y", "z", "q"),
                vertex_delta=12,
            ),
        ]

        candidates = evaluate_topk_paths._build_superpath_candidates(
            graph,
            segments,
            top_k=2,
            max_pass_length=12,
            min_edge_weight=1,
        )

        self.assertEqual(candidates[0].passes, ("a", "b", "c", "d", "e", "f", "g", "h"))
        self.assertEqual(candidates[0].score, 15)
        self.assertEqual(candidates[0].vertex_delta, 15)
        self.assertEqual(candidates[0].edge_score, 20)
        self.assertEqual(candidates[1].passes, ("x", "y", "z", "q"))

    def test_superpath_score_uses_edge_weight_only_as_tiebreaker(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.edge_counts[("b", "c")] = 10_000
        graph.edge_counts[("y", "z")] = 1
        segments = [
            evaluate_topk_paths.SuperSegmentCandidate(
                index=1,
                passes=("a", "b"),
                vertex_delta=1,
            ),
            evaluate_topk_paths.SuperSegmentCandidate(
                index=2,
                passes=("c", "d"),
                vertex_delta=1,
            ),
            evaluate_topk_paths.SuperSegmentCandidate(
                index=3,
                passes=("x", "y"),
                vertex_delta=10,
            ),
            evaluate_topk_paths.SuperSegmentCandidate(
                index=4,
                passes=("z", "q"),
                vertex_delta=10,
            ),
        ]

        candidates = evaluate_topk_paths._build_superpath_candidates(
            graph,
            segments,
            top_k=4,
            max_pass_length=4,
            min_edge_weight=1,
        )

        self.assertEqual(candidates[0].passes, ("x", "y", "z", "q"))
        self.assertEqual(candidates[0].score, 20)
        self.assertEqual(candidates[0].edge_score, 1)
        by_passes = {candidate.passes: candidate for candidate in candidates}
        weak_glue = by_passes[("a", "b", "c", "d")]
        self.assertEqual(weak_glue.score, 2)
        self.assertEqual(weak_glue.edge_score, 10_000)
        self.assertNotEqual(candidates[0].passes, weak_glue.passes)

    def test_superpath_candidate_generation_rejects_high_overlap_segments(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.edge_counts[("c", "b")] = 5
        segments = [
            evaluate_topk_paths.SuperSegmentCandidate(
                index=1,
                passes=("a", "b", "c"),
                vertex_delta=10,
            ),
            evaluate_topk_paths.SuperSegmentCandidate(
                index=2,
                passes=("b", "c", "d"),
                vertex_delta=9,
            ),
        ]

        candidates = evaluate_topk_paths._build_superpath_candidates(
            graph,
            segments,
            top_k=5,
            max_pass_length=6,
            min_edge_weight=1,
            max_overlap=1,
        )

        self.assertNotIn(
            ("a", "b", "c", "b", "c", "d"),
            {candidate.passes for candidate in candidates},
        )

    def test_generate_superpath_segments_falls_back_to_single_nodes(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "e"])
        args = type(
            "Args",
            (),
            {
                "segment_min_length": 4,
                "segment_max_length": 6,
                "segment_top_k": 100,
                "segment_max_jaccard": 0.75,
                "tiny_graph_threshold": 4,
                "min_edge_weight": 1,
                "top_starts": 20,
                "paths_per_start": 20,
            },
        )
        original = evaluate_topk_paths.cycle_breaking_top_start_paths
        try:
            evaluate_topk_paths.cycle_breaking_top_start_paths = (
                lambda *_args, **_kwargs: []
            )
            result = evaluate_topk_paths._generate_superpath_segments_with_stats(graph, args)
        finally:
            evaluate_topk_paths.cycle_breaking_top_start_paths = original

        self.assertEqual(result.segment_length_floor, 1)
        self.assertEqual(
            {tuple(path) for path in result.paths},
            {(node,) for node in graph.nodes},
        )

    def test_generate_superpath_segments_falls_back_to_length_two(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "e"])
        raw_paths = [["a", "b"], ["c"]]
        args = type(
            "Args",
            (),
            {
                "segment_min_length": 4,
                "segment_max_length": 6,
                "segment_top_k": 100,
                "segment_max_jaccard": 0.75,
                "tiny_graph_threshold": 4,
                "min_edge_weight": 1,
                "top_starts": 20,
                "paths_per_start": 20,
            },
        )
        original = evaluate_topk_paths.cycle_breaking_top_start_paths
        try:
            evaluate_topk_paths.cycle_breaking_top_start_paths = (
                lambda *_args, **_kwargs: raw_paths
            )
            result = evaluate_topk_paths._generate_superpath_segments_with_stats(graph, args)
        finally:
            evaluate_topk_paths.cycle_breaking_top_start_paths = original

        self.assertFalse(result.tiny_graph_mode)
        self.assertEqual(result.segment_length_floor, 2)
        self.assertEqual(result.paths, [["a", "b"]])

    def test_generate_superpath_segments_enumerates_all_tiny_graph_paths(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c"])
        graph.edge_counts[("a", "b")] = 3
        graph.edge_counts[("b", "c")] = 2
        args = type(
            "Args",
            (),
            {
                "segment_min_length": 4,
                "segment_max_length": 6,
                "segment_top_k": 100,
                "segment_max_jaccard": 0.75,
                "tiny_graph_threshold": 4,
                "min_edge_weight": 1,
                "top_starts": 20,
                "paths_per_start": 20,
            },
        )

        result = evaluate_topk_paths._generate_superpath_segments_with_stats(graph, args)

        self.assertTrue(result.tiny_graph_mode)
        self.assertEqual(result.segment_length_floor, 1)
        self.assertEqual(
            {tuple(path) for path in result.paths},
            {
                ("a",),
                ("b",),
                ("c",),
                ("a", "b"),
                ("b", "c"),
                ("a", "b", "c"),
            },
        )

    def test_evaluate_superpath_finds_best_paths_on_two_and_three_node_graphs(self) -> None:
        scenarios = [
            (["a", "b"], [("a", "b")], 10),
            (["a", "b", "c"], [("a", "b"), ("b", "c")], 15),
        ]
        for nodes, edges, expected_delta in scenarios:
            with self.subTest(nodes=nodes):
                graph = PassOrderGraph(benchmark="demo")
                graph.nodes.update(nodes)
                for source, target in edges:
                    graph.edge_counts[(source, target)] = 5
                sizes: dict[str, int] = {}

                def fake_measure(bitcode_path: Path, workdir: Path) -> int:
                    return sizes.get(str(bitcode_path), 100)

                def fake_optimize(input_bc: Path, output_bc: Path) -> None:
                    sizes[str(output_bc)] = 90

                def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
                    sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - 5

                args = type(
                    "Args",
                    (),
                    {
                        "heuristic": "cycle_breaking_superpath_topk",
                        "top_k": 20,
                        "superpath_eval_top_k": 20,
                        "top_starts": 20,
                        "paths_per_start": 20,
                        "max_length": 20,
                        "min_edge_weight": 1,
                        "segment_top_k": 100,
                        "segment_min_length": 4,
                        "segment_max_length": 6,
                        "segment_max_jaccard": 0.75,
                        "tiny_graph_threshold": 4,
                        "superpath_beam_factor": 5,
                        "superpath_max_candidates": 100,
                        "superpath_min_segment_delta": 1,
                        "superpath_max_overlap": 1,
                    },
                )
                original_measure = evaluate_topk_paths.measure_text_size
                original_optimize = evaluate_topk_paths.optimize_oz
                original_apply = evaluate_topk_paths.apply_pass_sequence
                try:
                    evaluate_topk_paths.measure_text_size = fake_measure
                    evaluate_topk_paths.optimize_oz = fake_optimize
                    evaluate_topk_paths.apply_pass_sequence = fake_apply
                    selected, _, _ = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                        graph,
                        Path("demo.bc"),
                        args=args,
                    )
                finally:
                    evaluate_topk_paths.measure_text_size = original_measure
                    evaluate_topk_paths.optimize_oz = original_optimize
                    evaluate_topk_paths.apply_pass_sequence = original_apply

                self.assertEqual(selected["error"], "")
                self.assertEqual(selected["best_delta"], expected_delta)
                self.assertEqual(selected["best_passes"], nodes)
                self.assertTrue(selected["tiny_graph_mode"])

    def test_evaluate_superpath_uses_single_node_tiny_graph_candidate(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.add("sroa")
        graph.start_counts["sroa"] = 10
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 90

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - 20

        args = type(
            "Args",
            (),
            {
                "heuristic": "cycle_breaking_superpath_topk",
                "top_k": 10,
                "superpath_eval_top_k": 10,
                "top_starts": 20,
                "paths_per_start": 20,
                "max_length": 20,
                "min_edge_weight": 1,
                "segment_top_k": 100,
                "segment_min_length": 4,
                "segment_max_length": 6,
                "segment_max_jaccard": 0.75,
                "tiny_graph_threshold": 4,
                "superpath_beam_factor": 5,
                "superpath_max_candidates": 100,
                "superpath_min_segment_delta": 1,
                "superpath_max_overlap": 1,
            },
        )
        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            selected, candidates, _ = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                graph,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply

        self.assertEqual(selected["error"], "")
        self.assertEqual(selected["best_delta"], 20)
        self.assertEqual(selected["segment_length_floor"], 1)
        self.assertTrue(selected["tiny_graph_mode"])
        self.assertEqual(candidates[0]["passes"], ["sroa"])

    def test_generate_superpath_segments_applies_jaccard_diversity(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "e", "x", "y", "z", "q"])
        raw_paths = [
            ["a", "b", "c", "d", "e"],
            ["a", "b", "c", "d"],
            ["x", "y", "z", "q"],
        ]
        args = type(
            "Args",
            (),
            {
                "segment_min_length": 2,
                "segment_max_length": 5,
                "segment_top_k": 3,
                "segment_max_jaccard": 0.75,
                "min_edge_weight": 1,
                "top_starts": 10,
                "paths_per_start": 10,
            },
        )
        original = evaluate_topk_paths.cycle_breaking_top_start_paths

        try:
            evaluate_topk_paths.cycle_breaking_top_start_paths = (
                lambda *_args, **_kwargs: raw_paths
            )
            segments = evaluate_topk_paths._generate_superpath_segments(graph, args)
        finally:
            evaluate_topk_paths.cycle_breaking_top_start_paths = original

        self.assertEqual(len(segments), 2)
        self.assertIn(["x", "y", "z", "q"], segments)
        self.assertEqual(
            sum(1 for segment in segments if set(segment) & {"a", "b", "c", "d", "e"}),
            1,
        )

    def test_superpath_candidate_generation_is_bounded_for_dense_segments(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        segments = []
        for index in range(250):
            passes = tuple(f"p{index}_{offset}" for offset in range(4))
            segments.append(
                evaluate_topk_paths.SuperSegmentCandidate(
                    index=index,
                    passes=passes,
                    vertex_delta=250 - index,
                )
            )
        for left in segments:
            for right in segments:
                if left.index != right.index:
                    graph.edge_counts[(left.passes[-1], right.passes[0])] = 1

        started = time.monotonic()
        candidates = evaluate_topk_paths._build_superpath_candidates(
            graph,
            segments,
            top_k=10,
            max_pass_length=12,
            min_edge_weight=1,
            beam_factor=5,
            max_candidates=100_000,
        )
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5.0)
        self.assertLessEqual(len(candidates), 10)

    def test_superpath_candidate_generation_reports_truncation(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        segments = []
        for index in range(20):
            passes = (f"s{index}", f"e{index}")
            segments.append(
                evaluate_topk_paths.SuperSegmentCandidate(
                    index=index,
                    passes=passes,
                    vertex_delta=20 - index,
                )
            )
        for left in segments:
            for right in segments:
                if left.index != right.index:
                    graph.edge_counts[(left.passes[-1], right.passes[0])] = 1

        result = evaluate_topk_paths._build_superpath_candidates_with_stats(
            graph,
            segments,
            top_k=10,
            max_pass_length=8,
            min_edge_weight=1,
            beam_factor=5,
            max_candidates=30,
        )

        self.assertTrue(result.truncated)
        self.assertLessEqual(result.generated_count, 30)
        self.assertLessEqual(len(result.candidates), 10)

    def test_validate_args_rejects_segment_min_length_one(self) -> None:
        args = evaluate_topk_paths.parse_args(
            [
                "--graph",
                "graph.json",
                "--bitcode-dir",
                "bitcode",
                "--output",
                "out.json",
                "--heuristic",
                "cycle_breaking_superpath_topk",
                "--segment-min-length",
                "1",
            ]
        )

        with self.assertRaisesRegex(ValueError, "segment-min-length"):
            evaluate_topk_paths.validate_args(args)

    def test_evaluate_superpath_reports_cache_costs(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d"])
        graph.edge_counts[("b", "c")] = 5
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 95

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - 1

        args = type(
            "Args",
            (),
            {
                "heuristic": "cycle_breaking_superpath_topk",
                "top_k": 1,
                "superpath_eval_top_k": 1,
                "max_length": 4,
                "min_edge_weight": 1,
                "superpath_beam_factor": 5,
                "superpath_max_candidates": 100,
                "superpath_min_segment_delta": 1,
                "superpath_max_overlap": 1,
            },
        )
        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        original_segments = evaluate_topk_paths._generate_superpath_segments
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            evaluate_topk_paths._generate_superpath_segments = (
                lambda *_args, **_kwargs: [["a", "b"], ["c", "d"]]
            )
            selected, candidates, cache_count = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                graph,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply
            evaluate_topk_paths._generate_superpath_segments = original_segments

        self.assertEqual(selected["segment_eval_cost"], 4)
        self.assertEqual(selected["superpath_eval_cost"], 2)
        self.assertEqual(selected["segment_eval_cost"] + selected["superpath_eval_cost"] + 1, cache_count)
        self.assertEqual(candidates[0]["passes"], ["a", "b", "c", "d"])

    def test_evaluate_superpath_recovers_tail_failed_segment_prefix(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "bad"])
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 95

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            if passes[0] == "bad":
                raise RuntimeError("boom")
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - 5

        args = type(
            "Args",
            (),
            {
                "heuristic": "cycle_breaking_superpath_topk",
                "top_k": 1,
                "superpath_eval_top_k": 1,
                "max_length": 4,
                "min_edge_weight": 1,
                "segment_min_length": 2,
                "superpath_beam_factor": 5,
                "superpath_max_candidates": 100,
                "superpath_min_segment_delta": 1,
                "superpath_max_overlap": 1,
            },
        )
        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        original_segments = evaluate_topk_paths._generate_superpath_segments
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            evaluate_topk_paths._generate_superpath_segments = (
                lambda *_args, **_kwargs: [["a", "b", "bad"]]
            )
            selected, candidates, _ = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                graph,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply
            evaluate_topk_paths._generate_superpath_segments = original_segments

        self.assertEqual(selected["segment_failures"], 1)
        self.assertEqual(selected["segments_recovered_from_tail"], 1)
        self.assertEqual(selected["segment_valid_count"], 1)
        self.assertEqual(candidates[0]["passes"], ["a", "b"])
        self.assertEqual(candidates[0]["best_delta"], 10)

    def test_evaluate_superpath_truncates_segment_to_best_prefix_and_glues_from_prefix_tail(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "bad1", "bad2", "bad3", "c", "d"])
        graph.edge_counts[("b", "c")] = 9
        graph.edge_counts[("bad3", "c")] = 100
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 95

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            delta_by_pass = {
                "a": 5,
                "b": 5,
                "bad1": -3,
                "bad2": -3,
                "bad3": -3,
                "c": 3,
                "d": 3,
            }
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - delta_by_pass[passes[0]]

        args = type(
            "Args",
            (),
            {
                "heuristic": "cycle_breaking_superpath_topk",
                "top_k": 1,
                "superpath_eval_top_k": 1,
                "max_length": 8,
                "min_edge_weight": 1,
                "segment_min_length": 4,
                "superpath_beam_factor": 5,
                "superpath_max_candidates": 100,
                "superpath_min_segment_delta": 1,
                "superpath_max_overlap": 1,
            },
        )
        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        original_segments = evaluate_topk_paths._generate_superpath_segments
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            evaluate_topk_paths._generate_superpath_segments = (
                lambda *_args, **_kwargs: [["a", "b", "bad1", "bad2", "bad3"], ["c", "d"]]
            )
            selected, candidates, _ = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                graph,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply
            evaluate_topk_paths._generate_superpath_segments = original_segments

        self.assertEqual(selected["segments_truncated_to_best_prefix"], 1)
        self.assertEqual(selected["segment_valid_count"], 2)
        self.assertEqual(candidates[0]["superpath_edge_score"], 9)
        self.assertEqual(candidates[0]["passes"], ["a", "b", "c", "d"])
        self.assertEqual(candidates[0]["best_delta"], 16)

    def test_evaluate_superpath_keeps_default_behavior_on_large_graph(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        graph.edge_counts[("d", "e")] = 7
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 90

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - 1

        args = type(
            "Args",
            (),
            {
                "heuristic": "cycle_breaking_superpath_topk",
                "top_k": 10,
                "superpath_eval_top_k": 10,
                "max_length": 8,
                "min_edge_weight": 1,
                "segment_min_length": 4,
                "tiny_graph_threshold": 4,
                "superpath_beam_factor": 5,
                "superpath_max_candidates": 100,
                "superpath_min_segment_delta": 1,
                "superpath_max_overlap": 1,
            },
        )
        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        original_segments = evaluate_topk_paths._generate_superpath_segments
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            evaluate_topk_paths._generate_superpath_segments = (
                lambda *_args, **_kwargs: [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
            )
            selected, candidates, _ = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                graph,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply
            evaluate_topk_paths._generate_superpath_segments = original_segments

        self.assertEqual(selected["passes"], ["a", "b", "c", "d", "e", "f", "g", "h"])
        self.assertEqual(selected["best_delta"], 8)
        self.assertEqual(selected["segment_length_floor"], 4)
        self.assertFalse(selected["tiny_graph_mode"])
        self.assertFalse(selected["segment_delta_filter_bypassed"])
        self.assertEqual(selected["segments_truncated_to_best_prefix"], 0)
        self.assertEqual(candidates[0]["superpath_edge_score"], 7)

    def test_evaluate_superpath_bypasses_delta_filter_for_nonimproving_segments(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b", "c", "d", "e"])
        graph.edge_counts[("b", "c")] = 5
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 90

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100)

        args = type(
            "Args",
            (),
            {
                "heuristic": "cycle_breaking_superpath_topk",
                "top_k": 10,
                "superpath_eval_top_k": 10,
                "max_length": 8,
                "min_edge_weight": 1,
                "segment_min_length": 2,
                "tiny_graph_threshold": 4,
                "superpath_beam_factor": 5,
                "superpath_max_candidates": 100,
                "superpath_min_segment_delta": 1,
                "superpath_max_overlap": 1,
            },
        )
        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        original_segments = evaluate_topk_paths._generate_superpath_segments
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            evaluate_topk_paths._generate_superpath_segments = (
                lambda *_args, **_kwargs: [["a", "b"], ["c", "d"], ["e"]]
            )
            selected, candidates, _ = evaluate_topk_paths.evaluate_superpath_for_benchmark(
                graph,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply
            evaluate_topk_paths._generate_superpath_segments = original_segments

        self.assertEqual(selected["error"], "")
        self.assertEqual(selected["best_delta"], 0)
        self.assertTrue(selected["segment_delta_filter_bypassed"])
        self.assertEqual(selected["segment_valid_count"], 3)
        self.assertGreaterEqual(len(candidates), 1)

    def test_evaluate_topk_for_benchmark_selects_best_real_candidate(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b"])
        graph.start_counts.update({"a": 5, "b": 3})
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 90

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            delta = {"a": 5, "b": 20}[passes[0]]
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - delta

        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            selected, candidates, cache_count = evaluate_topk_paths.evaluate_topk_for_benchmark(
                graph,
                Path("demo.bc"),
                [["a"], ["b"]],
                heuristic="cycle_breaking_diverse_starts_top10",
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply

        self.assertEqual(selected["candidate_index"], 2)
        self.assertEqual(selected["best_delta"], 20)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(cache_count, 3)

    def test_evaluate_topk_for_benchmark_can_measure_instructions(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["a", "b"])
        sizes: dict[str, int] = {}
        instructions: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        instruction_measurements: list[str] = []

        def fake_instruction_count(bitcode_path: Path, workdir: Path) -> int:
            instruction_measurements.append(str(bitcode_path))
            return instructions.get(str(bitcode_path), 50)

        def fake_combined_measure(bitcode_path: Path, workdir: Path) -> tuple[int, int]:
            return (
                sizes.get(str(bitcode_path), 100),
                instructions.get(str(bitcode_path), 50),
            )

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 90
            instructions[str(output_bc)] = 45

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            size_delta = {"a": 5, "b": 20}[passes[0]]
            instr_delta = {"a": 8, "b": 12}[passes[0]]
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - size_delta
            instructions[str(output_bc)] = instructions.get(str(input_bc), 50) - instr_delta

        original_measure = evaluate_topk_paths.measure_text_size
        original_instruction_count = evaluate_topk_paths.measure_machine_instruction_count
        original_combined_measure = evaluate_topk_paths.measure_text_and_instruction_count
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.measure_machine_instruction_count = fake_instruction_count
            evaluate_topk_paths.measure_text_and_instruction_count = fake_combined_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            selected, candidates, _ = evaluate_topk_paths.evaluate_topk_for_benchmark(
                graph,
                Path("demo.bc"),
                [["a"], ["b"]],
                heuristic="cycle_breaking_diverse_starts_top10",
                measure_instructions=True,
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.measure_machine_instruction_count = original_instruction_count
            evaluate_topk_paths.measure_text_and_instruction_count = original_combined_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply

        self.assertEqual(selected["candidate_index"], 2)
        self.assertEqual(selected["baseline_instruction_count"], 50)
        self.assertEqual(selected["best_instruction_delta"], 12)
        self.assertEqual(selected["instruction_measurement"], "deferred")
        self.assertEqual(selected["instruction_eval_cost"], 1)
        self.assertEqual(len(instruction_measurements), 1)
        self.assertIsNone(candidates[0]["best_instruction_count"])
        summary = evaluate_topk_paths.make_report_payload(
            type(
                "Args",
                (),
                {
                    "graph": Path("graph.json"),
                    "bitcode_dir": Path("bc"),
                    "heuristic": "cycle_breaking_diverse_starts_top10",
                    "top_k": 2,
                    "top_starts": 10,
                    "paths_per_start": 10,
                    "max_length": 12,
                    "min_edge_weight": 1,
                    "random_walks": 0,
                    "random_seed": 0,
                    "exhaustive_length": 6,
                    "measure_instructions": True,
                },
            )(),
            [selected],
            candidates,
            {},
        )["summary"]["cycle_breaking_diverse_starts_top10"]
        self.assertEqual(summary["total_best_instruction_delta"], 12)
        self.assertEqual(summary["weighted_best_instruction_percent"], 24.0)

    def test_measure_text_and_instruction_count_uses_one_object_compilation(self) -> None:
        original_run_cmd = evaluate_topk_paths.run_cmd
        commands: list[str] = []

        class Result:
            def __init__(self, stdout: str = "") -> None:
                self.stdout = stdout

        def fake_run_cmd(cmd):
            commands.append(cmd[0])
            if cmd[0] == "llc":
                obj_path = Path(cmd[cmd.index("-o") + 1])
                obj_path.write_bytes(b"obj")
                return Result()
            if cmd[0] == "llvm-size":
                return Result("text data bss dec hex filename\n123 0 0 123 7b demo.o\n")
            if cmd[0] == "llvm-objdump":
                return Result("   0:\t90\n   1:\t90\nlabel:\n")
            raise AssertionError(f"unexpected command: {cmd}")

        try:
            evaluate_topk_paths.run_cmd = fake_run_cmd
            with tempfile.TemporaryDirectory() as tmp_str:
                workdir = Path(tmp_str)
                size, count = evaluate_topk_paths.measure_text_and_instruction_count(
                    Path("demo.bc"),
                    workdir,
                )
                object_files = list(workdir.glob("*.o"))
        finally:
            evaluate_topk_paths.run_cmd = original_run_cmd

        self.assertEqual(size, 123)
        self.assertEqual(count, 2)
        self.assertEqual(commands.count("llc"), 1)
        self.assertEqual(commands, ["llc", "llvm-size", "llvm-objdump"])
        self.assertEqual(object_files, [])

    def test_measure_machine_instruction_count_removes_temporary_object(self) -> None:
        original_run_cmd = evaluate_topk_paths.run_cmd

        class Result:
            def __init__(self, stdout: str = "") -> None:
                self.stdout = stdout

        def fake_run_cmd(cmd):
            if cmd[0] == "llc":
                obj_path = Path(cmd[cmd.index("-o") + 1])
                obj_path.write_bytes(b"obj")
                return Result()
            if cmd[0] == "llvm-objdump":
                return Result("   0:\t90\n   1:\t90\nlabel:\n")
            raise AssertionError(f"unexpected command: {cmd}")

        try:
            evaluate_topk_paths.run_cmd = fake_run_cmd
            with tempfile.TemporaryDirectory() as tmp_str:
                workdir = Path(tmp_str)
                count = evaluate_topk_paths.measure_machine_instruction_count(
                    Path("demo.bc"),
                    workdir,
                )
                object_files = list(workdir.glob("*.o"))
        finally:
            evaluate_topk_paths.run_cmd = original_run_cmd

        self.assertEqual(count, 2)
        self.assertEqual(object_files, [])

    def test_evaluate_topk_reports_tail_failures_and_prefix_failures(self) -> None:
        graph = PassOrderGraph(benchmark="demo")
        graph.nodes.update(["good", "bad"])
        graph.start_counts.update({"good": 5})
        graph.edge_counts[("good", "bad")] = 5
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 95

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            if passes == ["bad"]:
                raise RuntimeError("pass failed")
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - 20

        original_measure = evaluate_topk_paths.measure_text_size
        original_optimize = evaluate_topk_paths.optimize_oz
        original_apply = evaluate_topk_paths.apply_pass_sequence
        try:
            evaluate_topk_paths.measure_text_size = fake_measure
            evaluate_topk_paths.optimize_oz = fake_optimize
            evaluate_topk_paths.apply_pass_sequence = fake_apply
            selected, candidates, cache_count = evaluate_topk_paths.evaluate_topk_for_benchmark(
                graph,
                Path("demo.bc"),
                [["good", "bad"]],
                heuristic="cycle_breaking_diverse_starts_top10",
            )
        finally:
            evaluate_topk_paths.measure_text_size = original_measure
            evaluate_topk_paths.optimize_oz = original_optimize
            evaluate_topk_paths.apply_pass_sequence = original_apply

        self.assertEqual(selected["best_prefix_len"], 1)
        self.assertEqual(selected["best_delta"], 20)
        self.assertEqual(selected["error_kind"], "tail_failure")
        self.assertEqual(selected["prefix_failures"], 1)
        self.assertEqual(candidates[0]["error_kind"], "tail_failure")
        self.assertEqual(cache_count, 3)

        summary = evaluate_topk_paths.make_report_payload(
            type(
                "Args",
                (),
                {
                    "graph": Path("graph.json"),
                    "bitcode_dir": Path("bc"),
                    "heuristic": "cycle_breaking_diverse_starts_top10",
                    "top_k": 1,
                    "top_starts": 10,
                    "paths_per_start": 10,
                    "max_length": 12,
                    "min_edge_weight": 1,
                    "random_walks": 0,
                    "random_seed": 0,
                    "exhaustive_length": 6,
                    "measure_instructions": False,
                },
            )(),
            [selected],
            candidates,
            {"demo": cache_count},
        )["summary"]["cycle_breaking_diverse_starts_top10"]

        self.assertEqual(summary["tail_failures"], 1)
        self.assertEqual(summary["full_failures"], 0)
        self.assertEqual(summary["prefix_failures"], 1)


    def test_chunk_forest_mining_anchor_closes_core(self) -> None:
        results = [
            FunctionPassResult(
                function="demo-v0_encode.bc",
                baseline_size=2000,
                best_size=800,
                passes=["sroa", "instcombine", "simplifycfg", "gvn", "dse", "adce"],
            ),
            FunctionPassResult(
                function="demo-v0_decode.bc",
                baseline_size=1800,
                best_size=1000,
                passes=["sroa", "instcombine", "simplifycfg", "gvn", "licm", "instcombine"],
            ),
            FunctionPassResult(
                function="demo-v0_init.bc",
                baseline_size=700,
                best_size=400,
                passes=["mem2reg", "instcombine", "simplifycfg", "dse"],
            ),
            FunctionPassResult(
                function="demo-v0_main.bc",
                baseline_size=100,
                best_size=100,
                passes=["instcombine", "simplifycfg"],
            ),
        ]

        chunks = mine_chunks(
            results,
            config=ChunkInventoryConfig(
                ngram_max=4,
                closure_theta=0.8,
                min_support=2,
                top_chunks=5,
                macro_top=0,
            ),
        )
        mined = {chunk.passes: chunk for chunk in chunks if chunk.kind == "mined"}
        chunk = mined[("sroa", "instcombine", "simplifycfg", "gvn")]

        self.assertEqual(chunk.weight, 2000)
        self.assertEqual(chunk.support, 2)
        self.assertNotIn(("instcombine", "simplifycfg"), mined)

    def test_chunk_forest_mining_uses_support_semantics_and_macro_fallback(self) -> None:
        results = [
            FunctionPassResult(
                function="demo-v0_repeat.bc",
                baseline_size=1000,
                best_size=900,
                passes=["a", "b", "a", "b", "c"],
            ),
            FunctionPassResult(
                function="demo-v0_unique.bc",
                baseline_size=1000,
                best_size=850,
                passes=["x", "y", "z"],
            ),
        ]

        chunks = mine_chunks(
            results,
            config=ChunkInventoryConfig(
                ngram_max=2,
                closure_theta=1.0,
                min_support=2,
                top_chunks=5,
                macro_top=1,
                min_inventory=1,
            ),
        )

        self.assertEqual([chunk.kind for chunk in chunks], ["macro"])
        self.assertEqual(chunks[0].passes, ("x", "y", "z"))
        self.assertEqual(chunks[0].weight, 150)

    def test_chunk_forest_pool_is_deterministic_and_marks_partial_chunks(self) -> None:
        chunks = [
            Chunk(("a", "b"), 10, "mined", 2),
            Chunk(("c", "d", "e"), 8, "mined", 2),
        ]
        graph = build_chunk_graph(
            chunks,
            [
                FunctionPassResult(
                    function="demo-v0_f.bc",
                    baseline_size=100,
                    best_size=80,
                    passes=["a", "b", "c", "d", "e"],
                )
            ],
        )

        left = generate_candidate_pool(
            graph,
            config=ChunkWalkConfig(pool_size=20, walk_seed=3, max_length=4),
        )
        right = generate_candidate_pool(
            graph,
            config=ChunkWalkConfig(pool_size=20, walk_seed=3, max_length=4),
        )

        self.assertEqual(left, right)
        candidate = next(item for item in left if item.passes == ("a", "b", "c", "d"))
        self.assertEqual([ref.chunk_index for ref in candidate.chunks], [0, 1])
        self.assertFalse(candidate.chunks[0].partial)
        self.assertTrue(candidate.chunks[1].partial)

    def test_chunk_forest_selection_prefers_shared_prefix_and_diversity(self) -> None:
        chunks = (
            Chunk(("a", "b"), 10, "mined", 2),
            Chunk(("a", "c"), 10, "mined", 2),
            Chunk(("d", "e"), 9, "mined", 2),
        )
        pool = [
            CandidatePath(("a", "b"), (ChunkRef(0, 0, 2),)),
            CandidatePath(("a", "c"), (ChunkRef(1, 0, 2),)),
            CandidatePath(("d", "e"), (ChunkRef(2, 0, 2),)),
        ]

        selected, _, _ = select_paths(
            pool,
            chunks,
            {0: 10, 1: 10, 2: 9},
            config=ChunkSelectionConfig(paths=2, lambda_cache=2.0, gamma_diversity=0.5),
        )

        self.assertEqual([candidate.passes for candidate in selected], [("a", "b"), ("a", "c")])

        duplicate_pool = [
            CandidatePath(("a", "b"), (ChunkRef(0, 0, 2),)),
            CandidatePath(("a", "b", "x"), (ChunkRef(0, 0, 2),)),
            CandidatePath(("d", "e"), (ChunkRef(2, 0, 2),)),
        ]
        selected, _, _ = select_paths(
            duplicate_pool,
            chunks,
            {0: 10, 2: 9},
            config=ChunkSelectionConfig(paths=2, lambda_cache=0.0, gamma_diversity=0.1),
        )
        self.assertEqual(selected[1].passes, ("d", "e"))

    def test_chunk_forest_credit_assignment_and_rescale_use_tu_bytes(self) -> None:
        chunks = (
            Chunk(("bad",), 1000, "mined", 2),
            Chunk(("good",), 10, "mined", 2),
        )
        candidates = [
            CandidatePath(("bad", "good"), (ChunkRef(0, 0, 1), ChunkRef(1, 1, 2))),
            CandidatePath(("bad",), (ChunkRef(0, 0, 1),)),
            CandidatePath(("good",), (ChunkRef(1, 0, 1),)),
            CandidatePath(("good", "bad"), (ChunkRef(1, 0, 1), ChunkRef(0, 1, 2))),
        ]
        prefix_cache = {
            (): {"size": 100, "error": ""},
            ("bad",): {"size": 105, "error": ""},
            ("bad", "good"): {"size": 95, "error": ""},
            ("good",): {"size": 90, "error": ""},
            ("good", "bad"): {"size": 91, "error": ""},
        }

        observations = evaluate_chunk_forest.assign_chunk_credit(candidates, prefix_cache)
        values, measured_count, scale = evaluate_chunk_forest.rescore_chunk_values(chunks, observations)

        self.assertEqual(observations[0], [-5, -5, -1])
        self.assertEqual(observations[1], [10, 10, 10])
        self.assertEqual(measured_count, 2)
        self.assertLess(values[0], 0)
        self.assertEqual(values[1], 10)
        self.assertLess(scale, 0.01)

    def test_chunk_forest_evaluator_works_on_single_function_mock(self) -> None:
        results = [
            FunctionPassResult(
                function="demo-v0_one.bc",
                baseline_size=100,
                best_size=70,
                passes=["a", "b"],
            )
        ]
        sizes: dict[str, int] = {}

        def fake_measure(bitcode_path: Path, workdir: Path) -> int:
            return sizes.get(str(bitcode_path), 100)

        def fake_optimize(input_bc: Path, output_bc: Path) -> None:
            sizes[str(output_bc)] = 95

        def fake_apply(input_bc: Path, passes: list[str], output_bc: Path) -> None:
            sizes[str(output_bc)] = sizes.get(str(input_bc), 100) - {"a": 10, "b": 15}[passes[0]]

        args = type(
            "Args",
            (),
            {
                "ngram_max": 4,
                "closure_theta": 0.8,
                "min_support": 2,
                "top_chunks": 30,
                "macro_top": 3,
                "pool_size": 50,
                "walk_seed": 7,
                "max_length": 12,
                "waves": 2,
                "paths": 4,
                "lambda_cache": 0.0,
                "gamma_diversity": 0.5,
                "max_real_evals_per_benchmark": 0,
            },
        )()
        original_measure = evaluate_chunk_forest.measure_text_size
        original_optimize = evaluate_chunk_forest.optimize_oz
        cache_globals = evaluate_chunk_forest.evaluate_candidate_with_prefix_cache.__globals__
        original_cache_measure = cache_globals["measure_text_size"]
        original_apply = cache_globals["apply_pass_sequence"]
        try:
            evaluate_chunk_forest.measure_text_size = fake_measure
            evaluate_chunk_forest.optimize_oz = fake_optimize
            cache_globals["measure_text_size"] = fake_measure
            cache_globals["apply_pass_sequence"] = fake_apply
            selected, candidates, cache_count = evaluate_chunk_forest.evaluate_chunk_forest_for_benchmark(
                "demo-v0",
                results,
                Path("demo.bc"),
                args=args,
            )
        finally:
            evaluate_chunk_forest.measure_text_size = original_measure
            evaluate_chunk_forest.optimize_oz = original_optimize
            cache_globals["measure_text_size"] = original_cache_measure
            cache_globals["apply_pass_sequence"] = original_apply

        self.assertEqual(selected["heuristic"], "chunk_forest")
        self.assertEqual(selected["error"], "")
        self.assertEqual(selected["best_delta"], 25)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreaterEqual(cache_count, 2)
        self.assertGreaterEqual(selected["chunks_macro"] + selected["chunks_single"], 1)

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
