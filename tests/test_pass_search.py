from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from llvm_ir.stages.function_search.cem import (
    CEMConfig,
    CEMSearch,
    CandidateResult,
    actions_to_passes,
    cyclic_shifts,
    search_pass_sequence_for_function,
)
from llvm_ir.stages.function_search.algorithms import CEMPassSearch, FunctionSearchContext
from llvm_ir.stages.function_search.algorithms import RandomPassSearch
from llvm_ir.stages.function_search import pass_search
from llvm_ir.stages.function_search.random_search import (
    RandomSearchConfig,
    search_pass_sequence_randomly,
)
from llvm_ir.stages.function_search.pass_search import select_bitcode_files, summarize_rows


class PassSearchTests(unittest.TestCase):
    def test_cem_update_increases_elite_action_probability(self) -> None:
        search = CEMSearch(
            passes=["a", "b", "c"],
            steps=2,
            rng=random.Random(1),
            candidates=4,
            elite_size=2,
            smoothing=1.0,
            min_prob=0.0,
            epsilon=0.0,
        )

        class Result:
            def __init__(self, actions, size, reward):
                self.actions = actions
                self.size = size
                self.reward = reward

        search.update(
            [
                Result([2, 1], 10, 5.0),
                Result([2, 0], 11, 4.0),
                Result([0, 0], 12, 1.0),
            ]
        )

        self.assertGreater(search.probabilities[0][2], search.probabilities[0][0])
        self.assertGreater(search.probabilities[1][0], 0.0)
        self.assertGreater(search.probabilities[1][1], 0.0)

    def test_stop_action_truncates_pass_sequence(self) -> None:
        passes = ["a", "b"]
        stop_action = len(passes)

        self.assertEqual(
            actions_to_passes([0, stop_action, 1], passes, stop_action=stop_action),
            ["a"],
        )

    def test_empty_elite_actions_reinforce_stop(self) -> None:
        search = CEMSearch(
            passes=["a", "b"],
            steps=2,
            rng=random.Random(1),
            candidates=2,
            elite_size=1,
            smoothing=1.0,
            min_prob=0.0,
            epsilon=0.0,
            allow_stop=True,
        )

        search.update(
            [
                CandidateResult(actions=[], passes=[], size=10, reward=0.0),
            ]
        )

        assert search.stop_action is not None
        self.assertGreater(
            search.probabilities[0][search.stop_action],
            search.probabilities[0][0],
        )

    def test_cem_search_uses_candidate_evaluator_for_one_function(self) -> None:
        calls = []

        def evaluate(actions, selected_passes, candidate_id):
            calls.append((actions, selected_passes, candidate_id))
            size = 10 - selected_passes.count("good")
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=size,
                reward=10 - size,
            )

        result = search_pass_sequence_for_function(
            ["bad", "good"],
            baseline_size=10,
            config=CEMConfig(
                steps=2,
                iterations=2,
                candidates=3,
                elite_size=1,
                smoothing=0.5,
                min_prob=0.0,
                epsilon=0.5,
                evaluate_shifts=False,
            ),
            rng=random.Random(3),
            evaluate_candidate=evaluate,
        )

        self.assertEqual(result.total_evaluated, 6)
        self.assertEqual(result.failed, 0)
        self.assertGreaterEqual(result.delta, 0)
        self.assertEqual(len(calls), 6)

    def test_cem_search_preserves_zero_size_best_candidate(self) -> None:
        sizes = iter([0, 5])

        def evaluate(actions, selected_passes, candidate_id):
            size = next(sizes)
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=size,
                reward=10 - size,
            )

        result = search_pass_sequence_for_function(
            ["a"],
            baseline_size=10,
            config=CEMConfig(
                steps=1,
                iterations=1,
                candidates=2,
                elite_size=1,
                allow_stop=False,
            ),
            rng=random.Random(1),
            evaluate_candidate=evaluate,
        )

        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.size, 0)

    def test_cem_search_does_not_evaluate_shifts_by_default(self) -> None:
        calls = []

        def evaluate(actions, selected_passes, candidate_id):
            calls.append((actions, selected_passes, candidate_id))
            return CandidateResult(actions, selected_passes, 10, 0.0)

        result = search_pass_sequence_for_function(
            ["a", "b", "c"],
            baseline_size=10,
            config=CEMConfig(
                steps=3,
                iterations=2,
                candidates=3,
                elite_size=1,
                allow_stop=False,
            ),
            rng=random.Random(1),
            evaluate_candidate=evaluate,
        )

        self.assertEqual(result.total_evaluated, 6)
        self.assertEqual(len(calls), 6)

    def test_cem_search_evaluates_unique_cyclic_shifts(self) -> None:
        self.assertEqual(
            cyclic_shifts([0, 1, 2]),
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        )
        self.assertEqual(cyclic_shifts([1, 1, 1]), [[1, 1, 1]])
        calls = []

        def evaluate(actions, selected_passes, candidate_id):
            calls.append((actions, selected_passes, candidate_id))
            size = 10 - int(actions == [1, 2, 0])
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=size,
                reward=10 - size,
            )

        search = CEMSearch(
            passes=["a", "b", "c"],
            steps=3,
            rng=random.Random(1),
            candidates=1,
            elite_size=1,
            smoothing=0.5,
            min_prob=0.0,
            epsilon=0.0,
            allow_stop=False,
        )
        search.sample = lambda: [0, 1, 2]  # type: ignore[method-assign]

        original_search = CEMSearch
        try:
            import llvm_ir.stages.function_search.cem as cem_module

            cem_module.CEMSearch = lambda **_kwargs: search  # type: ignore[assignment]
            result = search_pass_sequence_for_function(
                ["a", "b", "c"],
                baseline_size=10,
                config=CEMConfig(
                    steps=3,
                    iterations=1,
                    candidates=1,
                    elite_size=1,
                    allow_stop=False,
                    evaluate_shifts=True,
                ),
                rng=random.Random(3),
                evaluate_candidate=evaluate,
            )
        finally:
            cem_module.CEMSearch = original_search  # type: ignore[assignment]

        self.assertEqual(result.total_evaluated, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual([call[0] for call in calls], cyclic_shifts([0, 1, 2]))
        self.assertEqual(result.best.actions, [1, 2, 0])

    def test_cem_algorithm_implements_function_search_interface(self) -> None:
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

        def evaluate(actions, selected_passes, candidate_id):
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=7,
                reward=3,
            )

        result = algorithm.search(
            FunctionSearchContext(
                bitcode_path=Path("function.bc"),
                passes=["good"],
                baseline_size=10,
                rng=random.Random(1),
                evaluate_candidate=evaluate,
            )
        )

        self.assertEqual(algorithm.name, "cem")
        self.assertEqual(result.best_size, 7)

    def test_random_search_preserves_zero_size_best_candidate(self) -> None:
        sizes = iter([0, 5])

        def evaluate(actions, selected_passes, candidate_id):
            size = next(sizes)
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=size,
                reward=10 - size,
            )

        result = search_pass_sequence_randomly(
            ["a"],
            baseline_size=10,
            config=RandomSearchConfig(
                steps=1,
                iterations=1,
                candidates=2,
                allow_stop=False,
            ),
            rng=random.Random(1),
            evaluate_candidate=evaluate,
        )

        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.size, 0)

    def test_random_search_uses_candidate_evaluator_for_one_function(self) -> None:
        calls = []

        def evaluate(actions, selected_passes, candidate_id):
            calls.append((actions, selected_passes, candidate_id))
            size = 10 - selected_passes.count("good")
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=size,
                reward=10 - size,
            )

        result = search_pass_sequence_randomly(
            ["bad", "good"],
            baseline_size=10,
            config=RandomSearchConfig(
                steps=2,
                iterations=2,
                candidates=3,
                allow_stop=False,
                evaluate_shifts=False,
            ),
            rng=random.Random(3),
            evaluate_candidate=evaluate,
        )

        self.assertEqual(result.total_evaluated, 6)
        self.assertEqual(result.failed, 0)
        self.assertGreaterEqual(result.delta, 0)
        self.assertEqual(len(calls), 6)

    def test_random_search_does_not_evaluate_shifts_by_default(self) -> None:
        calls = []

        def evaluate(actions, selected_passes, candidate_id):
            calls.append((actions, selected_passes, candidate_id))
            return CandidateResult(actions, selected_passes, 10, 0.0)

        result = search_pass_sequence_randomly(
            ["a", "b", "c"],
            baseline_size=10,
            config=RandomSearchConfig(
                steps=3,
                iterations=2,
                candidates=3,
                allow_stop=False,
            ),
            rng=random.Random(1),
            evaluate_candidate=evaluate,
        )

        self.assertEqual(result.total_evaluated, 6)
        self.assertEqual(len(calls), 6)

    def test_random_algorithm_implements_function_search_interface(self) -> None:
        algorithm = RandomPassSearch(
            RandomSearchConfig(
                steps=1,
                iterations=1,
                candidates=1,
                allow_stop=False,
                evaluate_shifts=False,
            )
        )

        def evaluate(actions, selected_passes, candidate_id):
            return CandidateResult(
                actions=actions,
                passes=selected_passes,
                size=6,
                reward=4,
            )

        result = algorithm.search(
            FunctionSearchContext(
                bitcode_path=Path("function.bc"),
                passes=["good"],
                baseline_size=10,
                rng=random.Random(1),
                evaluate_candidate=evaluate,
            )
        )

        self.assertEqual(algorithm.name, "random")
        self.assertEqual(result.best_size, 6)

    def test_evaluate_sequence_returns_best_prefix(self) -> None:
        original_apply = pass_search.apply_pass_sequence
        original_measure = pass_search.measure_text_size

        def fake_apply(_input_bc, _passes, output_bc):
            Path(output_bc).write_bytes(b"bc")

        def fake_measure(bitcode_path, _workdir):
            if bitcode_path.name.endswith("prefix1.bc"):
                return 8
            if bitcode_path.name.endswith("prefix2.bc"):
                return 12
            return 9

        try:
            pass_search.apply_pass_sequence = fake_apply
            pass_search.measure_text_size = fake_measure
            with tempfile.TemporaryDirectory() as tmp_str:
                result = pass_search.evaluate_sequence(
                    Path("input.bc"),
                    [0, 1, 2],
                    ["good", "bad", "neutral"],
                    baseline_size=10,
                    workdir=Path(tmp_str),
                    candidate_id="cand",
                )
        finally:
            pass_search.apply_pass_sequence = original_apply
            pass_search.measure_text_size = original_measure

        self.assertEqual(result.size, 8)
        self.assertEqual(result.passes, ["good"])
        self.assertEqual(result.actions, [0])

    def test_measure_text_size_removes_temporary_object(self) -> None:
        original_run_cmd = pass_search.run_cmd

        class Result:
            stdout = "text data bss dec hex filename\n42 0 0 42 2a demo.o\n"

        def fake_run_cmd(cmd):
            if cmd[0] == "llc":
                obj_path = Path(cmd[cmd.index("-o") + 1])
                obj_path.write_bytes(b"obj")
                return Result()
            if cmd[0] == "llvm-size":
                return Result()
            raise AssertionError(f"unexpected command: {cmd}")

        try:
            pass_search.run_cmd = fake_run_cmd
            with tempfile.TemporaryDirectory() as tmp_str:
                workdir = Path(tmp_str)
                size = pass_search.measure_text_size(Path("demo.bc"), workdir)
                object_files = list(workdir.glob("*.o"))
        finally:
            pass_search.run_cmd = original_run_cmd

        self.assertEqual(size, 42)
        self.assertEqual(object_files, [])

    def test_filter_valid_passes_can_validate_on_multiple_files(self) -> None:
        original_run_cmd = pass_search.run_cmd

        def fake_run_cmd(cmd):
            pass_name = cmd[1].removeprefix("-passes=")
            bitcode_name = Path(cmd[2]).name
            if pass_name == "late-valid" and bitcode_name == "second.bc":
                return None
            if pass_name == "always-valid":
                return None
            raise pass_search.LLVMCommandError("invalid")

        try:
            pass_search.run_cmd = fake_run_cmd
            valid_one, invalid_one = pass_search.filter_valid_passes(
                [Path("first.bc")],
                ["late-valid", "always-valid", "never-valid"],
            )
            valid_two, invalid_two = pass_search.filter_valid_passes(
                [Path("first.bc"), Path("second.bc")],
                ["late-valid", "always-valid", "never-valid"],
            )
        finally:
            pass_search.run_cmd = original_run_cmd

        self.assertEqual(valid_one, ["always-valid"])
        self.assertEqual(invalid_one, ["late-valid", "never-valid"])
        self.assertEqual(valid_two, ["late-valid", "always-valid"])
        self.assertEqual(invalid_two, ["never-valid"])

    def test_select_bitcode_files_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            for name in ("c.bc", "a.bc", "b.bc"):
                (root / name).write_bytes(b"bc")

            first = select_bitcode_files(root, limit=2, seed=7)
            second = select_bitcode_files(root, limit=2, seed=7)

            self.assertEqual([p.name for p in first], [p.name for p in second])
            self.assertEqual(len(first), 2)

    def test_run_pass_search_jobs_serial_preserves_file_order(self) -> None:
        original_run = pass_search.run_search_for_function
        calls = []

        def fake_run(bitcode_path, _passes, *, algorithm, rng):
            calls.append((bitcode_path.name, algorithm.name, rng.random()))
            return {
                "function": bitcode_path.name,
                "search_algorithm": algorithm.name,
            }

        try:
            pass_search.run_search_for_function = fake_run
            rows = pass_search.run_pass_search_jobs(
                [Path("b.bc"), Path("a.bc")],
                ["pass"],
                algorithm_name="random",
                cem_config=CEMConfig(
                    steps=1,
                    iterations=1,
                    candidates=1,
                    elite_size=1,
                ),
                seed=7,
                jobs=1,
            )
        finally:
            pass_search.run_search_for_function = original_run

        expected_rng_values = [
            random.Random(pass_search.function_seed(7, 1)).random(),
            random.Random(pass_search.function_seed(7, 2)).random(),
        ]

        self.assertEqual([row["function"] for row in rows], ["b.bc", "a.bc"])
        self.assertEqual([call[1] for call in calls], ["random", "random"])
        self.assertEqual([call[2] for call in calls], expected_rng_values)

    def test_summarize_rows_counts_improvements_and_oz_wins(self) -> None:
        summary = summarize_rows(
            [
                {
                    "baseline_size": 10,
                    "oz_size": 8,
                    "oz_delta": 2,
                    "cem_best_size": 7,
                    "cem_delta": 3,
                },
                {
                    "baseline_size": 10,
                    "oz_size": 6,
                    "oz_delta": 4,
                    "cem_best_size": 9,
                    "cem_delta": 1,
                },
            ]
        )

        self.assertEqual(summary["functions"], 2)
        self.assertEqual(summary["cem_improved"], 2)
        self.assertEqual(summary["cem_beats_oz"], 1)

    def test_summarize_rows_counts_ppo_when_available(self) -> None:
        summary = summarize_rows(
            [
                {
                    "baseline_size": 10,
                    "oz_size": 8,
                    "oz_delta": 2,
                    "cem_best_size": 7,
                    "cem_delta": 3,
                    "ppo_best_size": 6,
                    "ppo_best_delta": 4,
                },
                {
                    "baseline_size": 10,
                    "oz_size": 6,
                    "oz_delta": 4,
                    "cem_best_size": 5,
                    "cem_delta": 5,
                    "ppo_best_size": 9,
                    "ppo_best_delta": 1,
                },
            ]
        )

        self.assertEqual(summary["ppo_available"], 2)
        self.assertEqual(summary["ppo_improved"], 2)
        self.assertEqual(summary["ppo_beats_cem"], 1)
        self.assertEqual(summary["cem_beats_ppo"], 1)

    def test_summarize_rows_counts_random_search(self) -> None:
        summary = summarize_rows(
            [
                {
                    "search_algorithm": "random",
                    "baseline_size": 10,
                    "oz_size": 8,
                    "oz_delta": 2,
                    "random_best_size": 7,
                    "random_delta": 3,
                },
                {
                    "search_algorithm": "random",
                    "baseline_size": 10,
                    "oz_size": 6,
                    "oz_delta": 4,
                    "random_best_size": 9,
                    "random_delta": 1,
                },
            ]
        )

        self.assertEqual(summary["functions"], 2)
        self.assertEqual(summary["random_improved"], 2)
        self.assertEqual(summary["random_beats_oz"], 1)


if __name__ == "__main__":
    unittest.main()
