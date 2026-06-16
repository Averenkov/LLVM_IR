"""Tests for the consensus-subsequence heuristic (no LLVM required)."""

import unittest
from random import Random

from llvm_ir.heuristics.translation_unit.consensus_subseq import (
    ConsensusConfig,
    exploration_distribution,
    generate_candidate_pool,
    mine_frequent_subsequences,
)
from llvm_ir.stages.translation_unit.contracts import FunctionPassResult
from llvm_ir.stages.translation_unit.evaluate_consensus_subseq import (
    calibrate_exploration,
    pass_marginals_from_cache,
)


def _fr(passes, delta=10):
    return FunctionPassResult(function="f", baseline_size=100, best_size=100 - delta, passes=passes)


class MiningTests(unittest.TestCase):
    def test_mines_common_gapped_subsequence(self) -> None:
        # a..b..c in order across all three, with gaps -> longest common subseq a,b,c.
        results = [
            _fr(["a", "x", "b", "y", "c"]),
            _fr(["a", "b", "z", "c"]),
            _fr(["q", "a", "b", "c"]),
        ]
        skeletons = mine_frequent_subsequences(results, config=ConsensusConfig(minsup=3, max_pattern_length=6))
        patterns = {s.passes for s in skeletons}
        self.assertIn(("a", "b", "c"), patterns)  # gapped consensus found
        # support of the full pattern is all 3 functions
        full = next(s for s in skeletons if s.passes == ("a", "b", "c"))
        self.assertEqual(full.support, 3)

    def test_minsup_filters_rare_patterns(self) -> None:
        results = [_fr(["a", "b"]), _fr(["a", "b"]), _fr(["c", "d"])]
        skeletons = mine_frequent_subsequences(results, config=ConsensusConfig(minsup=2, max_pattern_length=4))
        patterns = {s.passes for s in skeletons}
        self.assertIn(("a", "b"), patterns)
        self.assertNotIn(("c", "d"), patterns)  # support 1 < minsup


class GenerationTests(unittest.TestCase):
    def test_pool_is_deterministic_and_bounded(self) -> None:
        results = [_fr(["a", "b", "c"]), _fr(["a", "b", "c"])]
        skeletons = mine_frequent_subsequences(results, config=ConsensusConfig(minsup=2))
        explore = exploration_distribution(results, {"a": 5}, filler_floor=1.0)
        cfg = ConsensusConfig(minsup=2, max_length=8, pool_size=200, walk_seed=7)
        pool_a = generate_candidate_pool(skeletons, explore, config=cfg, rng=Random(7))
        pool_b = generate_candidate_pool(skeletons, explore, config=cfg, rng=Random(7))
        self.assertEqual(pool_a, pool_b)  # deterministic by seed
        self.assertTrue(pool_a)
        for cand in pool_a:
            self.assertLessEqual(len(cand), 8)

    def test_noise_can_inject_non_consensus_passes(self) -> None:
        results = [_fr(["a", "b"]), _fr(["a", "b"])]
        skeletons = mine_frequent_subsequences(results, config=ConsensusConfig(minsup=2))
        # exploration favors 'iro' (a stand-in for an interprocedural pass not in consensus)
        explore = {"a": 1.0, "b": 1.0, "iro": 100.0}
        cfg = ConsensusConfig(minsup=2, epsilon_insert=0.6, pool_size=500, walk_seed=1)
        pool = generate_candidate_pool(skeletons, explore, config=cfg, rng=Random(1))
        self.assertTrue(any("iro" in cand for cand in pool))  # noise injected the explorer


class TeacherTests(unittest.TestCase):
    def test_pass_marginals_from_cache(self) -> None:
        prefix_cache = {
            (): {"size": 1000, "error": ""},
            ("a",): {"size": 940, "error": ""},          # a -> 60
            ("a", "b"): {"size": 900, "error": ""},      # b -> 40
        }
        means = pass_marginals_from_cache([["a", "b"]], prefix_cache)
        self.assertEqual(means["a"], 60)
        self.assertEqual(means["b"], 40)

    def test_calibrate_exploration_prefers_measured_winners(self) -> None:
        base = {"a": 1.0, "b": 1.0, "c": 1.0}
        measured = {"a": 500.0, "b": -10.0}  # a strong, b harmful, c unmeasured
        out = calibrate_exploration(base, measured)
        self.assertGreater(out["a"], out["b"])  # measured winner outranks harmful
        self.assertGreater(out["a"], out["c"])  # and outranks unmeasured
        self.assertTrue(all(v > 0 for v in out.values()))  # floor keeps reachable


if __name__ == "__main__":
    unittest.main()
