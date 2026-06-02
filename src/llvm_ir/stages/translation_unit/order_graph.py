"""Build pass-order graphs from per-function pass-search results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .contracts import FunctionPassResult

WeightMode = Literal["count", "delta"]


@dataclass(frozen=True)
class PassOrderEdge:
    """Directed evidence that one pass was observed before another pass."""

    source: str
    target: str
    weight: int


@dataclass
class PassOrderGraph:
    """Directed pass-order graph for one benchmark or translation unit."""

    benchmark: str
    weight_mode: WeightMode = "count"
    function_count: int = 0
    sequence_count: int = 0
    nodes: set[str] = field(default_factory=set)
    edge_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def add_sequence(self, passes: list[str], *, support_weight: int = 1) -> None:
        """Add all pairwise order constraints from one pass sequence."""
        self.function_count += 1
        if not passes:
            return
        self.sequence_count += 1
        self.nodes.update(passes)
        if support_weight <= 0:
            return
        seen_pairs: set[tuple[str, str]] = set()
        for left_index, source in enumerate(passes):
            for target in passes[left_index + 1 :]:
                if source == target:
                    continue
                pair = (source, target)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                self.edge_counts[pair] = self.edge_counts.get(pair, 0) + support_weight

    @property
    def edges(self) -> list[PassOrderEdge]:
        return [
            PassOrderEdge(source, target, weight)
            for (source, target), weight in sorted(self.edge_counts.items())
        ]

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self.edge_counts

    def edge_weight(self, source: str, target: str) -> int:
        return self.edge_counts.get((source, target), 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "weight_mode": self.weight_mode,
            "function_count": self.function_count,
            "sequence_count": self.sequence_count,
            "nodes": sorted(self.nodes),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                }
                for edge in self.edges
            ],
        }


def benchmark_id_from_function_name(function_name: str) -> str:
    """Infer the benchmark id from a per-function bitcode file name.

    Dataset files are named as `<suite>_<benchmark>_<function>.bc`, for example
    `tensorflow-v0_1985_<mangled-function>.bc`.
    """
    stem = Path(function_name).stem
    parts = stem.split("_", 2)
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def build_pass_order_graph(
    function_results: Iterable[FunctionPassResult],
    *,
    benchmark: str = "",
    weight_mode: WeightMode = "count",
) -> PassOrderGraph:
    graph = PassOrderGraph(benchmark=benchmark, weight_mode=weight_mode)
    for result in function_results:
        graph.add_sequence(
            list(result.passes),
            support_weight=support_weight_for_result(result, weight_mode),
        )
    return graph


def build_pass_order_graphs_by_benchmark(
    function_results: Iterable[FunctionPassResult],
    *,
    benchmark_key: Callable[[str], str] = benchmark_id_from_function_name,
    weight_mode: WeightMode = "count",
) -> dict[str, PassOrderGraph]:
    groups: dict[str, list[FunctionPassResult]] = defaultdict(list)
    for result in function_results:
        groups[benchmark_key(result.function)].append(result)
    return {
        benchmark: build_pass_order_graph(
            results,
            benchmark=benchmark,
            weight_mode=weight_mode,
        )
        for benchmark, results in sorted(groups.items())
    }


def support_weight_for_result(
    result: FunctionPassResult,
    weight_mode: WeightMode,
) -> int:
    if weight_mode == "count":
        return 1
    if weight_mode == "delta":
        return max(result.delta, 0)
    raise ValueError(f"Unknown graph weight mode: {weight_mode}")


def load_function_pass_results_from_report(
    report_path: Path,
    *,
    algorithm: str | None = None,
) -> list[FunctionPassResult]:
    """Load best per-function pass sequences from pass_search comparison JSON."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("rows", [])
    if algorithm is None:
        algorithm = str(report.get("config", {}).get("algorithm") or "")
    results = []
    for row in rows:
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
            FunctionPassResult(
                function=str(row["function"]),
                baseline_size=int(row["baseline_size"]),
                best_size=int(row[best_size_key]),
                passes=list(row.get(passes_key) or []),
            )
        )
    return results


def write_graph_report(
    graphs: dict[str, PassOrderGraph],
    output_path: Path,
    *,
    input_path: Path,
    algorithm: str | None,
    weight_mode: WeightMode,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(input_path),
        "algorithm": algorithm,
        "weight_mode": weight_mode,
        "benchmark_count": len(graphs),
        "graphs": [graph.to_dict() for graph in graphs.values()],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def pass_order_graph_from_dict(payload: dict[str, Any]) -> PassOrderGraph:
    graph = PassOrderGraph(
        benchmark=str(payload["benchmark"]),
        weight_mode=payload.get("weight_mode", "count"),
        function_count=int(payload.get("function_count", 0)),
        sequence_count=int(payload.get("sequence_count", 0)),
        nodes=set(payload.get("nodes") or []),
    )
    for edge in payload.get("edges") or []:
        graph.edge_counts[(str(edge["source"]), str(edge["target"]))] = int(
            edge["weight"]
        )
    return graph


def load_graphs_from_report(report_path: Path) -> dict[str, PassOrderGraph]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    graphs = {}
    for graph_payload in report.get("graphs") or []:
        graph = pass_order_graph_from_dict(graph_payload)
        graphs[graph.benchmark] = graph
    return graphs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-benchmark pass-order graphs from pass-search results."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to pass_search comparison.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write graph JSON.",
    )
    parser.add_argument(
        "--algorithm",
        default="",
        help="Algorithm prefix to read, for example cem or random. "
        "Defaults to comparison config algorithm.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["count", "delta"],
        default="count",
        help="Edge weight mode: count adds 1 per supporting function sequence; "
        "delta adds the function size improvement.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    algorithm = args.algorithm or None
    weight_mode: WeightMode = args.weight_mode
    results = load_function_pass_results_from_report(args.input, algorithm=algorithm)
    graphs = build_pass_order_graphs_by_benchmark(
        results,
        weight_mode=weight_mode,
    )
    write_graph_report(
        graphs,
        args.output,
        input_path=args.input,
        algorithm=algorithm,
        weight_mode=weight_mode,
    )
    print(
        json.dumps(
            {
                "functions": len(results),
                "benchmarks": len(graphs),
                "weight_mode": weight_mode,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
