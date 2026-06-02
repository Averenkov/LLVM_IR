"""Interfaces for per-function LLVM pass-sequence search algorithms."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .cem import (
    CandidateEvaluator,
    CandidateResult,
    CEMConfig,
    search_pass_sequence_for_function,
)
from .random_search import RandomSearchConfig, search_pass_sequence_randomly


@dataclass
class FunctionSearchContext:
    """Data and callbacks needed to optimize one function-level bitcode file."""

    bitcode_path: Path
    passes: list[str]
    baseline_size: int
    rng: random.Random
    evaluate_candidate: CandidateEvaluator


@dataclass
class FunctionSearchResult:
    """Generic result of a per-function pass-search algorithm."""

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


class FunctionPassSearchAlgorithm(Protocol):
    """Common interface for algorithms that search passes for one function."""

    name: str

    def search(self, context: FunctionSearchContext) -> FunctionSearchResult:
        """Run the algorithm for one function and return the best candidate."""


@dataclass
class CEMPassSearch:
    """CEM implementation of the per-function pass-search interface."""

    config: CEMConfig
    name: str = "cem"

    def search(self, context: FunctionSearchContext) -> FunctionSearchResult:
        result = search_pass_sequence_for_function(
            context.passes,
            context.baseline_size,
            config=self.config,
            rng=context.rng,
            evaluate_candidate=context.evaluate_candidate,
        )
        return FunctionSearchResult(
            baseline_size=result.baseline_size,
            best=result.best,
            total_evaluated=result.total_evaluated,
            failed=result.failed,
        )


@dataclass
class RandomPassSearch:
    """Uniform random implementation of the per-function pass-search interface."""

    config: RandomSearchConfig
    name: str = "random"

    def search(self, context: FunctionSearchContext) -> FunctionSearchResult:
        result = search_pass_sequence_randomly(
            context.passes,
            context.baseline_size,
            config=self.config,
            rng=context.rng,
            evaluate_candidate=context.evaluate_candidate,
        )
        return FunctionSearchResult(
            baseline_size=result.baseline_size,
            best=result.best,
            total_evaluated=result.total_evaluated,
            failed=result.failed,
        )


def build_function_search_algorithm(
    name: str,
    *,
    cem_config: CEMConfig,
    random_config: RandomSearchConfig | None = None,
) -> FunctionPassSearchAlgorithm:
    """Build a per-function search algorithm by name."""
    normalized = name.lower()
    if normalized == "cem":
        return CEMPassSearch(cem_config)
    if normalized == "random":
        if random_config is None:
            random_config = RandomSearchConfig(
                steps=cem_config.steps,
                iterations=cem_config.iterations,
                candidates=cem_config.candidates,
                allow_stop=cem_config.allow_stop,
                evaluate_shifts=cem_config.evaluate_shifts,
            )
        return RandomPassSearch(random_config)
    raise ValueError(f"Unknown function pass-search algorithm: {name}")


def candidate_to_row_prefix(
    prefix: str,
    best: CandidateResult | None,
    baseline_size: int,
    best_size: int,
    delta: int,
    total_evaluated: int,
    failed: int,
) -> dict[str, object]:
    """Render algorithm result fields with a stable prefix for reports."""
    return {
        f"{prefix}_best_size": best_size,
        f"{prefix}_delta": delta,
        f"{prefix}_best_passes": best.passes if best else [],
        f"{prefix}_best_actions": best.actions if best else [],
        f"{prefix}_total_evaluated": total_evaluated,
        f"{prefix}_failed": failed,
        f"{prefix}_baseline_size": baseline_size,
    }
