"""Cross-Entropy Method search for one function's LLVM pass sequence."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CandidateResult:
    actions: list[int]
    passes: list[str]
    size: int | None
    reward: float
    error: str = ""


@dataclass
class CEMConfig:
    steps: int = 6
    iterations: int = 3
    candidates: int = 8
    elite_size: int = 3
    smoothing: float = 0.65
    min_prob: float = 0.001
    epsilon: float = 0.05
    allow_stop: bool = True
    evaluate_shifts: bool = False


@dataclass
class CEMResult:
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


CandidateEvaluator = Callable[[list[int], list[str], str], CandidateResult]


@dataclass
class CEMSearch:
    passes: list[str]
    steps: int
    rng: random.Random
    candidates: int
    elite_size: int
    smoothing: float
    min_prob: float
    epsilon: float
    allow_stop: bool = True
    probabilities: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("CEMSearch requires at least one pass")
        action_count = self.action_count
        self.steps = max(1, self.steps)
        self.candidates = max(1, self.candidates)
        self.elite_size = max(1, self.elite_size)
        self.smoothing = min(1.0, max(0.0, self.smoothing))
        self.epsilon = min(1.0, max(0.0, self.epsilon))
        self.min_prob = min(1.0 / action_count, max(0.0, self.min_prob))
        uniform = 1.0 / action_count
        self.probabilities = [
            [uniform for _ in range(action_count)] for _ in range(self.steps)
        ]

    def sample(self) -> list[int]:
        actions = []
        for position in range(self.steps):
            if self.rng.random() < self.epsilon:
                actions.append(self.rng.randrange(self.action_count))
            else:
                actions.append(sample_categorical(self.rng, self.probabilities[position]))
        return actions

    def update(self, evaluated: list[CandidateResult]) -> None:
        successful = [item for item in evaluated if item.size is not None]
        if not successful:
            return
        elite = sorted(successful, key=lambda item: item.reward, reverse=True)[: self.elite_size]
        action_count = self.action_count
        floor_mass = self.min_prob * action_count
        remaining_mass = max(0.0, 1.0 - floor_mass)

        for position in range(self.steps):
            counts = [1.0 for _ in range(action_count)]
            for item in elite:
                action = self._action_for_update(item.actions, position)
                if action is not None:
                    counts[action] += 1.0
            total = sum(counts)
            target = [
                self.min_prob + remaining_mass * (count / total)
                for count in counts
            ]
            blended = [
                (1.0 - self.smoothing) * old + self.smoothing * new
                for old, new in zip(self.probabilities[position], target)
            ]
            self.probabilities[position] = normalize(blended)

    @property
    def action_count(self) -> int:
        return len(self.passes) + int(self.allow_stop)

    @property
    def stop_action(self) -> int | None:
        return len(self.passes) if self.allow_stop else None

    def _action_for_update(self, actions: list[int], position: int) -> int | None:
        if position < len(actions):
            return actions[position]
        return self.stop_action


def search_pass_sequence_for_function(
    passes: list[str],
    baseline_size: int,
    *,
    config: CEMConfig,
    rng: random.Random,
    evaluate_candidate: CandidateEvaluator,
) -> CEMResult:
    """Find a pass sequence for one function using CEM.

    The evaluator owns the domain-specific work: applying the selected passes,
    measuring the resulting function, and returning a CandidateResult.
    """
    search = CEMSearch(
        passes=passes,
        steps=config.steps,
        rng=rng,
        candidates=config.candidates,
        elite_size=config.elite_size,
        smoothing=config.smoothing,
        min_prob=config.min_prob,
        epsilon=config.epsilon,
        allow_stop=config.allow_stop,
    )
    best: CandidateResult | None = None
    total_evaluated = 0
    failed = 0

    for iteration in range(config.iterations):
        evaluated = []
        for candidate in range(config.candidates):
            actions = search.sample()
            candidates_to_evaluate = (
                cyclic_shifts(actions) if config.evaluate_shifts else [actions]
            )
            for shift_index, shifted_actions in enumerate(candidates_to_evaluate):
                selected_passes = actions_to_passes(
                    shifted_actions,
                    passes,
                    stop_action=search.stop_action,
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
                evaluated.append(result)
        search.update(evaluated)

    return CEMResult(
        baseline_size=baseline_size,
        best=best,
        total_evaluated=total_evaluated,
        failed=failed,
    )


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in values]


def actions_to_passes(
    actions: list[int],
    passes: list[str],
    *,
    stop_action: int | None = None,
) -> list[str]:
    selected = []
    for action in actions:
        if stop_action is not None and action == stop_action:
            break
        selected.append(passes[action])
    return selected


def cyclic_shifts(actions: list[int]) -> list[list[int]]:
    """Return unique cyclic shifts of an action sequence, preserving order."""
    if not actions:
        return [[]]
    shifts = []
    seen = set()
    for offset in range(len(actions)):
        shifted = actions[offset:] + actions[:offset]
        key = tuple(shifted)
        if key in seen:
            continue
        seen.add(key)
        shifts.append(shifted)
    return shifts


def sample_categorical(rng: random.Random, probs: list[float]) -> int:
    needle = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probs):
        cumulative += prob
        if needle <= cumulative:
            return index
    return len(probs) - 1
