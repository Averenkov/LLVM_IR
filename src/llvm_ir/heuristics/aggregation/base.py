"""Base data structures for aggregation heuristics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerFunctionResult:
    function_id: str
    sequence: list[str]
    baseline_size: int
    best_size: int

    @property
    def delta(self) -> int:
        return self.baseline_size - self.best_size


@dataclass
class Dataset:
    results: list[PerFunctionResult]
    translation_unit_path: str = ""
    baseline_text_size: int = 0
    oz_text_size: int = 0
    name: str = "dataset"


@dataclass
class PassGraph:
    nodes: set[str] = field(default_factory=set)
    count_weight: dict[tuple[str, str], float] = field(default_factory=dict)
    delta_weight: dict[tuple[str, str], float] = field(default_factory=dict)

    def add_sequence(self, sequence: list[str], *, delta: int) -> None:
        self.nodes.update(sequence)
        support_delta = max(delta, 0)
        seen_pairs: set[tuple[str, str]] = set()
        for left_index, source in enumerate(sequence):
            for target in sequence[left_index + 1 :]:
                if source == target:
                    continue
                pair = (source, target)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                self.count_weight[pair] = self.count_weight.get(pair, 0.0) + 1.0
                self.delta_weight[pair] = (
                    self.delta_weight.get(pair, 0.0) + float(support_delta)
                )

    def edge_weight(self, source: str, target: str, *, mode: str = "delta") -> float:
        if source == target:
            return 0.0
        if mode == "count":
            return self.count_weight.get((source, target), 0.0)
        if mode == "delta":
            return self.delta_weight.get((source, target), 0.0)
        raise ValueError(f"Unknown edge weight mode: {mode}")

    def weight_map(self, *, mode: str = "delta") -> dict[tuple[str, str], float]:
        if mode == "count":
            return dict(self.count_weight)
        if mode == "delta":
            return dict(self.delta_weight)
        raise ValueError(f"Unknown edge weight mode: {mode}")

    def induced(self, keep: set[str]) -> "PassGraph":
        graph = PassGraph(nodes=set(self.nodes) & keep)
        for pair, weight in self.count_weight.items():
            if pair[0] in keep and pair[1] in keep and pair[0] != pair[1]:
                graph.count_weight[pair] = weight
        for pair, weight in self.delta_weight.items():
            if pair[0] in keep and pair[1] in keep and pair[0] != pair[1]:
                graph.delta_weight[pair] = weight
        return graph


@dataclass
class AggregationResult:
    sequences: list[list[str]]
    chosen_sequence: list[str]
    chosen_prefix_length: int
    extra: dict[str, Any] = field(default_factory=dict)


class AggregationHeuristic(ABC):
    name: str
    supports_topk: bool = False
    optional: bool = False

    @abstractmethod
    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        """Aggregate per-function pass sequences into TU-level candidates."""


def build_pass_graph(results: list[PerFunctionResult]) -> PassGraph:
    graph = PassGraph()
    for result in results:
        graph.add_sequence(result.sequence, delta=result.delta)
    return graph


def result_from_sequence(sequence: list[str], *, extra: dict[str, Any] | None = None) -> AggregationResult:
    return AggregationResult(
        sequences=[sequence],
        chosen_sequence=sequence,
        chosen_prefix_length=len(sequence),
        extra=extra or {},
    )


def load_dataset_from_pass_search_report(
    report_path: Path,
    *,
    algorithm: str | None = None,
    name: str | None = None,
) -> Dataset:
    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if algorithm is None:
        algorithm = str(payload.get("config", {}).get("algorithm") or "")
    results: list[PerFunctionResult] = []
    for row in payload.get("rows", []):
        row_algorithm = str(row.get("search_algorithm") or algorithm)
        prefix = algorithm or row_algorithm
        passes_key = f"{prefix}_best_passes"
        best_size_key = f"{prefix}_best_size"
        if passes_key not in row:
            prefix = row_algorithm
            passes_key = f"{prefix}_best_passes"
            best_size_key = f"{prefix}_best_size"
        if passes_key not in row or best_size_key not in row:
            continue
        results.append(
            PerFunctionResult(
                function_id=str(row["function"]),
                sequence=list(row.get(passes_key) or []),
                baseline_size=int(row.get("baseline_size") or 0),
                best_size=int(row.get(best_size_key) or row.get("baseline_size") or 0),
            )
        )
    return Dataset(results=results, name=name or report_path.stem)
