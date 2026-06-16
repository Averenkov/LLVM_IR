"""Tests for the TU-MCTS search core (no LLVM required)."""

import unittest

from llvm_ir.heuristics.translation_unit.tu_mcts import (
    BudgetExhausted,
    MCTSConfig,
    MeasureResult,
    run_mcts,
)


GOOD = {"g1": 120, "g2": 90, "g3": 70}
ACTIONS = ["g1", "g2", "g3", "bad1", "bad2", "x1", "x2"]


def _measure(key):
    size = 1000
    seen = set()
    for pass_name in key:
        if pass_name in GOOD and pass_name not in seen:
            size -= GOOD[pass_name]
            seen.add(pass_name)
        elif pass_name.startswith("bad"):
            size += 50
    return MeasureResult(size, "")


def _prior(prefix):
    return {a: (2.0 if a == "g1" and not prefix else 1.0) for a in ACTIONS}


class TUMCTSCoreTests(unittest.TestCase):
    def test_finds_improving_sequence(self) -> None:
        result = run_mcts(
            baseline_size=1000,
            prior_fn=_prior,
            measure=_measure,
            config=MCTSConfig(max_length=6, rollout_length=3, branching=7, seed=7, max_iterations=3000),
        )
        # Optimal reachable is g1+g2+g3 -> 720; allow the search to land near it.
        self.assertLessEqual(result.best_size, 720)
        self.assertGreater(1000 - result.best_size, 0)
        self.assertTrue(set(result.best_passes) <= set(ACTIONS))

    def test_respects_eval_budget(self) -> None:
        calls = {"n": 0}
        cache = {(): MeasureResult(1000, "")}

        def measure(key):
            if key in cache:
                return cache[key]
            if calls["n"] >= 25:
                raise BudgetExhausted()
            calls["n"] += 1
            cache[key] = _measure(key)
            return cache[key]

        result = run_mcts(
            baseline_size=1000,
            prior_fn=_prior,
            measure=measure,
            config=MCTSConfig(max_length=6, branching=7, seed=7),
        )
        self.assertLessEqual(calls["n"], 25)
        self.assertTrue(result.budget_exhausted)
        self.assertLessEqual(result.best_size, 1000)

    def test_handles_crashing_pass_as_terminal(self) -> None:
        def measure(key):
            if key and key[-1] == "g2":  # pretend g2 crashes opt
                return MeasureResult(None, "LLVMCommandError: boom")
            return _measure(key)

        result = run_mcts(
            baseline_size=1000,
            prior_fn=_prior,
            measure=measure,
            config=MCTSConfig(max_length=6, branching=7, seed=7, max_iterations=2000),
        )
        # g2 is unusable, but g1+g3 still improves and must not crash the search.
        self.assertLessEqual(result.best_size, 810)
        self.assertNotIn("g2", result.best_passes)


if __name__ == "__main__":
    unittest.main()
