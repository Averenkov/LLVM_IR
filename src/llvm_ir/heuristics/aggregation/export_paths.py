"""Export aggregation heuristic paths for whole-TU evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from llvm_ir.stages.translation_unit.order_graph import benchmark_id_from_function_name

from .base import Dataset, PerFunctionResult, build_pass_graph, load_dataset_from_pass_search_report
from .graph_utils import blended_weights, path_graph_score
from .registry import available_heuristics, build_heuristic


def group_dataset_by_benchmark(dataset: Dataset) -> dict[str, Dataset]:
    grouped: dict[str, list[PerFunctionResult]] = defaultdict(list)
    for result in dataset.results:
        grouped[benchmark_id_from_function_name(result.function_id)].append(result)
    return {
        benchmark: Dataset(results=results, name=benchmark)
        for benchmark, results in sorted(grouped.items())
    }


def export_aggregation_paths(
    dataset: Dataset,
    heuristic_names: list[str],
    *,
    config: dict[str, Any] | None = None,
    max_length: int = 0,
    include_topk: bool = False,
) -> dict[str, Any]:
    config = config or {}
    results = []
    for benchmark, benchmark_dataset in group_dataset_by_benchmark(dataset).items():
        graph = build_pass_graph(benchmark_dataset.results)
        weights = blended_weights(graph, alpha=float(config.get("alpha", 0.5)))
        for heuristic_name in heuristic_names:
            heuristic = build_heuristic(heuristic_name)
            aggregation_result = heuristic.aggregate(benchmark_dataset, graph, dict(config))
            candidate_sequences = (
                aggregation_result.sequences
                if include_topk and heuristic.supports_topk
                else [aggregation_result.chosen_sequence]
            )
            for rank, sequence in enumerate(candidate_sequences, start=1):
                path = _truncate(sequence, max_length)
                exported_name = (
                    f"{heuristic.name}_top{rank}"
                    if include_topk and heuristic.supports_topk
                    else heuristic.name
                )
                results.append(
                    {
                        "benchmark": benchmark,
                        "heuristic": exported_name,
                        "source_heuristic": heuristic.name,
                        "topk_rank": rank if include_topk and heuristic.supports_topk else None,
                        "path": path,
                        "score": {
                            "graph_score_delta": path_graph_score(path, weights),
                            "length": len(path),
                            "node_coverage": (
                                len(set(path)) / len(graph.nodes)
                                if graph.nodes
                                else 0.0
                            ),
                        },
                        "chosen_prefix_length": min(
                            aggregation_result.chosen_prefix_length,
                            len(path),
                        ),
                    }
                )
    return {
        "config": {
            "heuristics": heuristic_names,
            "max_length": max_length,
            "include_topk": include_topk,
            **config,
        },
        "summary": summarize_exported_paths(results),
        "results": results,
    }


def summarize_exported_paths(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[str(item["heuristic"])].append(item)
    summary = {}
    for heuristic, items in sorted(grouped.items()):
        summary[heuristic] = {
            "benchmarks": len(items),
            "total_graph_score_delta": sum(
                float(item["score"]["graph_score_delta"]) for item in items
            ),
            "mean_graph_score_delta": (
                sum(float(item["score"]["graph_score_delta"]) for item in items)
                / len(items)
            ),
            "mean_length": sum(int(item["score"]["length"]) for item in items)
            / len(items),
            "mean_node_coverage": sum(
                float(item["score"]["node_coverage"]) for item in items
            )
            / len(items),
        }
    return summary


def write_export_report(report: dict[str, Any], output_path: Path, *, input_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["input"] = str(input_path)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_heuristics(value: str) -> list[str]:
    if value == "all":
        return available_heuristics()
    return [item.strip() for item in value.split(",") if item.strip()]


def _truncate(sequence: list[str], max_length: int) -> list[str]:
    if max_length > 0:
        return list(sequence[:max_length])
    return list(sequence)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aggregation heuristic paths for whole-TU evaluation."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--heuristics", default="all")
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--include-topk", action="store_true")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset_from_pass_search_report(args.input, algorithm=args.algorithm)
    heuristics = parse_heuristics(args.heuristics)
    report = export_aggregation_paths(
        dataset,
        heuristics,
        config={
            "alpha": args.alpha,
            "top_k": args.top_k,
            "beam_width": args.beam_width,
        },
        max_length=args.max_length,
        include_topk=args.include_topk,
    )
    write_export_report(report, args.output, input_path=args.input)
    print(
        json.dumps(
            {
                "benchmarks": len(group_dataset_by_benchmark(dataset)),
                "heuristics": heuristics,
                "output": str(args.output),
                "summary": report["summary"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
