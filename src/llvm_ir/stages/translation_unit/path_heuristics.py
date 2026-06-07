"""Run translation-unit path heuristics on pass-order graphs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .beam_search import BeamSearchConfig, beam_search_path
from llvm_ir.heuristics.translation_unit.cycle_breaking_max_path import (
    CycleBreakingMaxPathConfig,
    cycle_breaking_max_path,
)
from llvm_ir.heuristics.translation_unit.exhaustive_path import (
    ExhaustivePathConfig,
    exhaustive_fixed_length_path,
)
from llvm_ir.heuristics.translation_unit.random_walk import (
    RandomWalkPathConfig,
    random_walk_path,
)
from .dag_longest_path import DAGLongestPathConfig, dag_longest_path
from .greedy_consensus import GreedyConsensusConfig, greedy_consensus_path
from .graph.order_graph import PassOrderGraph, load_graphs_from_report
from .path_scoring import score_path
from .weighted_toposort import WeightedToposortConfig, weighted_toposort_path


@dataclass(frozen=True)
class HeuristicRunConfig:
    max_length: int = 12
    beam_width: int = 16
    min_net_weight: int = 1
    min_edge_weight: int = 1
    conflict_penalty: float = 1.0
    random_walks: int = 256
    random_seed: int = 0
    exhaustive_length: int = 6


def run_heuristic(
    graph: PassOrderGraph,
    heuristic: str,
    *,
    config: HeuristicRunConfig,
) -> list[str]:
    if heuristic == "greedy_consensus":
        return greedy_consensus_path(
            graph,
            config=GreedyConsensusConfig(max_length=config.max_length),
        )
    if heuristic == "dag_longest_path":
        return dag_longest_path(
            graph,
            config=DAGLongestPathConfig(
                min_net_weight=config.min_net_weight,
                max_length=config.max_length,
            ),
        )
    if heuristic == "cycle_breaking_max_path":
        return cycle_breaking_max_path(
            graph,
            config=CycleBreakingMaxPathConfig(
                max_length=config.max_length,
                min_edge_weight=config.min_edge_weight,
            ),
        )
    if heuristic == "exhaustive_len6":
        return exhaustive_fixed_length_path(
            graph,
            config=ExhaustivePathConfig(
                path_length=config.exhaustive_length,
                min_edge_weight=config.min_edge_weight,
            ),
        )
    if heuristic == "random_walk":
        return random_walk_path(
            graph,
            config=RandomWalkPathConfig(
                max_length=config.max_length,
                walks=config.random_walks,
                seed=config.random_seed,
                min_edge_weight=config.min_edge_weight,
            ),
        )
    if heuristic == "beam_search":
        return beam_search_path(
            graph,
            config=BeamSearchConfig(
                beam_width=config.beam_width,
                max_length=config.max_length,
                conflict_penalty=config.conflict_penalty,
            ),
        )
    if heuristic == "weighted_toposort":
        return weighted_toposort_path(
            graph,
            config=WeightedToposortConfig(
                max_length=config.max_length,
                min_edge_weight=config.min_edge_weight,
            ),
        )
    raise ValueError(f"Unknown translation-unit heuristic: {heuristic}")


def compare_heuristics(
    graphs: dict[str, PassOrderGraph],
    heuristics: list[str],
    *,
    config: HeuristicRunConfig,
) -> dict[str, Any]:
    results = []
    for benchmark, graph in sorted(graphs.items()):
        for heuristic in heuristics:
            path = run_heuristic(graph, heuristic, config=config)
            score = score_path(graph, path)
            results.append(
                {
                    "benchmark": benchmark,
                    "heuristic": heuristic,
                    "weight_mode": graph.weight_mode,
                    "function_count": graph.function_count,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edge_counts),
                    "path": path,
                    "score": score.to_dict(),
                }
            )
    return {
        "config": {
            "heuristics": heuristics,
            "max_length": config.max_length,
            "beam_width": config.beam_width,
            "min_net_weight": config.min_net_weight,
            "min_edge_weight": config.min_edge_weight,
            "conflict_penalty": config.conflict_penalty,
            "random_walks": config.random_walks,
            "random_seed": config.random_seed,
            "exhaustive_length": config.exhaustive_length,
        },
        "summary": summarize_heuristic_results(results),
        "results": results,
    }


def summarize_heuristic_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(str(item["heuristic"]), []).append(item)
    summary = {}
    for heuristic, items in sorted(grouped.items()):
        if not items:
            continue
        total_order = sum(int(item["score"]["order_score"]) for item in items)
        total_conflict = sum(int(item["score"]["conflict_score"]) for item in items)
        total_net = sum(int(item["score"]["net_score"]) for item in items)
        summary[heuristic] = {
            "benchmarks": len(items),
            "total_order_score": total_order,
            "total_conflict_score": total_conflict,
            "total_net_score": total_net,
            "mean_net_score": total_net / len(items),
            "mean_length": sum(int(item["score"]["length"]) for item in items)
            / len(items),
            "mean_node_coverage": sum(
                float(item["score"]["node_coverage"]) for item in items
            )
            / len(items),
        }
    return summary


def write_heuristic_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_heuristics(value: str) -> list[str]:
    if value == "all":
        return [
            "greedy_consensus",
            "dag_longest_path",
            "cycle_breaking_max_path",
            "exhaustive_len6",
            "random_walk",
            "beam_search",
            "weighted_toposort",
        ]
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run path heuristics on translation-unit pass-order graphs."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--heuristics",
        default="all",
        help="Comma-separated heuristics or all. Available: "
        "greedy_consensus,dag_longest_path,cycle_breaking_max_path,"
        "exhaustive_len6,random_walk,beam_search,weighted_toposort.",
    )
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--min-net-weight", type=int, default=1)
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--conflict-penalty", type=float, default=1.0)
    parser.add_argument("--random-walks", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--exhaustive-length", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = HeuristicRunConfig(
        max_length=args.max_length,
        beam_width=args.beam_width,
        min_net_weight=args.min_net_weight,
        min_edge_weight=args.min_edge_weight,
        conflict_penalty=args.conflict_penalty,
        random_walks=args.random_walks,
        random_seed=args.random_seed,
        exhaustive_length=args.exhaustive_length,
    )
    graphs = load_graphs_from_report(args.input)
    heuristics = parse_heuristics(args.heuristics)
    report = compare_heuristics(graphs, heuristics, config=config)
    report["input"] = str(args.input)
    write_heuristic_report(report, args.output)
    print(
        json.dumps(
            {
                "benchmarks": len(graphs),
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
