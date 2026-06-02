"""Random search for one function's LLVM pass sequence."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .cem import (
    CandidateEvaluator,
    CandidateResult,
    actions_to_passes,
    cyclic_shifts,
)


@dataclass
class RandomSearchConfig:
    steps: int = 6
    iterations: int = 3
    candidates: int = 8
    allow_stop: bool = True
    evaluate_shifts: bool = False


@dataclass
class RandomSearchResult:
    baseline_size: int
    best: CandidateResult | None
    total_evaluated: int
    failed: int

    @property
    def best_size(self) -> int:
        if self.best is None or self.best.size is None:
            return self.baseline_size
        return self.best.size

    @property
    def delta(self) -> int:
        return self.baseline_size - self.best_size


def search_pass_sequence_randomly(
    passes: list[str],
    baseline_size: int,
    *,
    config: RandomSearchConfig,
    rng: random.Random,
    evaluate_candidate: CandidateEvaluator,
) -> RandomSearchResult:
    """Find a pass sequence by uniformly sampling candidates."""
    if not passes:
        raise ValueError("Random search requires at least one pass")

    steps = max(1, config.steps)
    iterations = max(1, config.iterations)
    candidates = max(1, config.candidates)
    action_count = len(passes) + int(config.allow_stop)
    stop_action = len(passes) if config.allow_stop else None
    best: CandidateResult | None = None
    total_evaluated = 0
    failed = 0

    for iteration in range(iterations):
        for candidate in range(candidates):
            actions = [rng.randrange(action_count) for _ in range(steps)]
            candidates_to_evaluate = (
                cyclic_shifts(actions) if config.evaluate_shifts else [actions]
            )
            for shift_index, shifted_actions in enumerate(candidates_to_evaluate):
                selected_passes = actions_to_passes(
                    shifted_actions,
                    passes,
                    stop_action=stop_action,
                )
                result = evaluate_candidate(
                    shifted_actions,
                    selected_passes,
                    f"iter{iteration}_cand{candidate}_shift{shift_index}",
                )
                total_evaluated += 1
                if result.size is None:
                    failed += 1
                elif best is None or result.size < (best.size or baseline_size + 1):
                    best = result

    return RandomSearchResult(
        baseline_size=baseline_size,
        best=best,
        total_evaluated=total_evaluated,
        failed=failed,
    )
