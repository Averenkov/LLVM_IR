"""Comparison runner for aggregation heuristics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable

from .base import AggregationHeuristic, Dataset, build_pass_graph
from .graph_utils import blended_weights, path_graph_score
from .metrics import HeuristicMetrics


def run_comparison(
    datasets: list[Dataset],
    heuristics: list[AggregationHeuristic],
    evaluator_factory: Callable[[Dataset], object | None],
    output_path: str | Path,
    *,
    config: dict[str, object] | None = None,
    no_tu_eval: bool = False,
    top_k_validate: bool = False,
) -> dict[str, object]:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    config = config or {}
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        graph = build_pass_graph(dataset.results)
        weights = blended_weights(graph, alpha=float(config.get("alpha", 0.5)))
        evaluator = None if no_tu_eval else evaluator_factory(dataset)
        for heuristic in heuristics:
            start = time.monotonic()
            result = heuristic.aggregate(dataset, graph, dict(config))
            candidates = result.sequences if top_k_validate else [result.chosen_sequence]
            evals = []
            for sequence in candidates:
                if evaluator is None:
                    continue
                evaluated = evaluator.evaluate(sequence)
                evals.append(evaluated)
            successful = [item for item in evals if item.success and item.text_size is not None]
            final_size = successful[0].text_size if successful else None
            best_prefix_size = min((item.text_size for item in successful), default=None)
            baseline = dataset.baseline_text_size or None
            oz = dataset.oz_text_size or None
            delta_vs_oz = (oz - best_prefix_size) if oz and best_prefix_size is not None else None
            norm_best = (
                best_prefix_size / baseline
                if baseline and best_prefix_size is not None
                else None
            )
            sequence = result.chosen_sequence
            metrics = HeuristicMetrics(
                name=heuristic.name,
                graph_score_delta=path_graph_score(sequence, weights),
                fas_weight=max(-path_graph_score(sequence, weights), 0.0),
                coverage=len(set(sequence)) / len(graph.nodes) if graph.nodes else 0.0,
                final_size=final_size,
                best_prefix_size=best_prefix_size,
                delta_vs_oz=delta_vs_oz,
                norm_best=norm_best,
                fail_rate=(
                    sum(1 for item in evals if not item.success) / len(evals)
                    if evals
                    else 0.0
                ),
                beat_oz=bool(oz and best_prefix_size is not None and best_prefix_size < oz),
                tu_evals=len(evals),
                wallclock_s=time.monotonic() - start,
            )
            row = {"dataset": dataset.name, **metrics.to_dict()}
            rows.append(row)
    _write_csv(output / "metrics.csv", rows)
    report = {"rows": rows, "config": config}
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
