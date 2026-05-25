"""Contracts for future translation-unit pass-sequence heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FunctionPassResult:
    """Best pass sequence found for one extracted function."""

    function: str
    baseline_size: int
    best_size: int
    passes: list[str]

    @property
    def delta(self) -> int:
        return self.baseline_size - self.best_size


@dataclass(frozen=True)
class TranslationUnitPlan:
    """A pass sequence selected for a whole translation unit."""

    bitcode_path: Path
    passes: list[str]
    source_results: list[FunctionPassResult]


class TranslationUnitHeuristic(Protocol):
    """Interface for stage 3 algorithms that aggregate function-level results."""

    name: str

    def build_plan(
        self,
        bitcode_path: Path,
        function_results: list[FunctionPassResult],
    ) -> TranslationUnitPlan:
        """Build a translation-unit pass plan from per-function search results."""
