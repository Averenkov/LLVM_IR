from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from llvm_ir.cem import (
    CEMConfig,
    CEMSearch,
    CandidateResult,
    actions_to_passes,
    search_pass_sequence_for_function,
)
from llvm_ir import pass_search
from llvm_ir.pass_search import select_bitcode_files, summarize_rows


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
            ),
            rng=random.Random(3),
            evaluate_candidate=evaluate,
        )

        self.assertEqual(result.total_evaluated, 6)
        self.assertEqual(result.failed, 0)
        self.assertGreaterEqual(result.delta, 0)
        self.assertEqual(len(calls), 6)

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

    def test_select_bitcode_files_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            root = Path(tmp_str)
            for name in ("c.bc", "a.bc", "b.bc"):
                (root / name).write_bytes(b"bc")

            first = select_bitcode_files(root, limit=2, seed=7)
            second = select_bitcode_files(root, limit=2, seed=7)

            self.assertEqual([p.name for p in first], [p.name for p in second])
            self.assertEqual(len(first), 2)

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


if __name__ == "__main__":
    unittest.main()
