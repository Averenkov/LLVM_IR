"""Metrics for aggregation heuristic comparisons."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HeuristicMetrics:
    name: str
    graph_score_delta: float
    fas_weight: float
    coverage: float
    final_size: int | None
    best_prefix_size: int | None
    delta_vs_oz: int | None
    norm_best: float | None
    fail_rate: float
    beat_oz: bool
    tu_evals: int
    wallclock_s: float

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def wilcoxon_signed_rank(_left: list[float], _right: list[float]) -> float | None:
    """Return None when scipy is unavailable; kept as graceful optional metric."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if not _left or len(_left) != len(_right):
        return None
    try:
        return float(wilcoxon(_left, _right).pvalue)
    except ValueError:
        return None

