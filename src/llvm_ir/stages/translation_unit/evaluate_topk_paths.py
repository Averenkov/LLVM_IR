"""Evaluate top-k translation-unit paths generated directly from pass-order graphs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.translation_unit.cycle_breaking_max_path import (
    CycleBreakingMaxPathConfig,
    cycle_breaking_diverse_start_paths,
    cycle_breaking_max_paths,
    cycle_breaking_top_start_paths,
)
from llvm_ir.heuristics.translation_unit.exhaustive_path import (
    ExhaustivePathConfig,
    exhaustive_fixed_length_top_paths,
)
from llvm_ir.heuristics.translation_unit.random_walk import (
    RandomWalkPathConfig,
    random_walk_top_paths,
)
from llvm_ir.stages.function_search.pass_search import (
    apply_pass_sequence,
    measure_text_size,
    optimize_oz,
    require_tools,
    run_cmd,
)
from llvm_ir.stages.translation_unit.evaluate import (
    summarize_evaluations,
    write_translation_unit_bitcodes,
)
from llvm_ir.stages.translation_unit.graph.order_graph import (
    PassOrderGraph,
    load_graphs_from_report,
)
from llvm_ir.stages.translation_unit.path_scoring import score_path

PathGenerator = Callable[[PassOrderGraph, argparse.Namespace], list[list[str]]]
INSTRUCTION_LINE_RE = re.compile(r"^\s*[0-9a-fA-F]+:\s")


def measure_machine_instruction_count(bitcode_path: Path, workdir: Path) -> int:
    obj_path = workdir / f"{bitcode_path.stem}.instr.o"
    try:
        run_cmd(["llc", "-filetype=obj", str(bitcode_path), "-o", str(obj_path)])
        out = run_cmd(["llvm-objdump", "-d", str(obj_path)]).stdout.splitlines()
        return sum(1 for line in out if INSTRUCTION_LINE_RE.match(line))
    finally:
        obj_path.unlink(missing_ok=True)


def generate_topk_paths(
    graph: PassOrderGraph,
    heuristic: str,
    args: argparse.Namespace,
) -> list[list[str]]:
    if heuristic in {"cycle_breaking_max_path_top10", "cycle_breaking_max_path_topk"}:
        return cycle_breaking_max_paths(
            graph,
            config=CycleBreakingMaxPathConfig(
                max_length=args.max_length,
                min_edge_weight=args.min_edge_weight,
            ),
            top_k=args.top_k,
        )
    if heuristic in {"cycle_breaking_diverse_starts_top10", "cycle_breaking_diverse_starts_topk"}:
        return cycle_breaking_diverse_start_paths(
            graph,
            config=CycleBreakingMaxPathConfig(
                max_length=args.max_length,
                min_edge_weight=args.min_edge_weight,
            ),
            top_k=args.top_k,
        )
    if heuristic == "cycle_breaking_top_starts_top_paths":
        return cycle_breaking_top_start_paths(
            graph,
            config=CycleBreakingMaxPathConfig(
                max_length=args.max_length,
                min_edge_weight=args.min_edge_weight,
            ),
            top_starts=args.top_starts,
            paths_per_start=args.paths_per_start,
        )
    if heuristic in {"random_walk_top10", "random_walk_topk"}:
        return random_walk_top_paths(
            graph,
            config=RandomWalkPathConfig(
                max_length=args.max_length,
                walks=args.random_walks,
                seed=args.random_seed,
                min_edge_weight=args.min_edge_weight,
            ),
            top_k=args.top_k,
        )
    if heuristic == "exhaustive_len6_top10":
        return exhaustive_fixed_length_top_paths(
            graph,
            config=ExhaustivePathConfig(
                path_length=args.exhaustive_length,
                min_edge_weight=args.min_edge_weight,
            ),
            top_k=args.top_k,
        )
    raise ValueError(f"Unknown top-k translation-unit heuristic: {heuristic}")


def evaluate_candidate_with_prefix_cache(
    passes: list[str],
    workdir: Path,
    prefix_cache: dict[tuple[str, ...], dict[str, Any]],
    *,
    measure_instructions: bool = False,
) -> dict[str, Any]:
    baseline = prefix_cache[()]
    current_key: tuple[str, ...] = ()
    best_size = int(baseline["size"])
    best_prefix_len = 0
    best_passes: list[str] = []
    baseline_instruction_count = baseline.get("instruction_count")
    best_instruction_count = (
        int(baseline_instruction_count)
        if baseline_instruction_count is not None
        else None
    )
    best_instruction_prefix_len = 0
    best_instruction_passes: list[str] = []
    error = ""

    for index, pass_name in enumerate(passes, start=1):
        next_key = current_key + (pass_name,)
        if next_key not in prefix_cache:
            parent = prefix_cache[current_key]
            output_bc = workdir / f"prefix_{len(prefix_cache):05d}.bc"
            try:
                apply_pass_sequence(Path(parent["bc"]), [pass_name], output_bc)
                size = measure_text_size(output_bc, workdir)
                entry = {
                    "bc": str(output_bc),
                    "size": size,
                    "error": "",
                }
                if measure_instructions:
                    entry["instruction_count"] = measure_machine_instruction_count(
                        output_bc,
                        workdir,
                    )
                prefix_cache[next_key] = entry
            except Exception as exc:  # noqa: BLE001
                entry = {
                    "bc": str(parent["bc"]),
                    "size": int(parent["size"]),
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_pass": pass_name,
                }
                if measure_instructions and "instruction_count" in parent:
                    entry["instruction_count"] = int(parent["instruction_count"])
                prefix_cache[next_key] = entry
        entry = prefix_cache[next_key]
        current_key = next_key
        if entry.get("error"):
            error = str(entry["error"])
            break
        current_size = int(entry["size"])
        if current_size < best_size:
            best_size = current_size
            best_prefix_len = index
            best_passes = passes[:index]
        current_instruction_count = entry.get("instruction_count")
        if (
            current_instruction_count is not None
            and (
                best_instruction_count is None
                or int(current_instruction_count) < best_instruction_count
            )
        ):
            best_instruction_count = int(current_instruction_count)
            best_instruction_prefix_len = index
            best_instruction_passes = passes[:index]

    final_entry = prefix_cache[current_key]
    error = error or str(final_entry.get("error") or "")
    error_kind = ""
    if error:
        error_kind = "tail_failure" if best_prefix_len > 0 else "full_failure"
    return {
        "final_size": int(final_entry["size"]),
        "best_size": best_size,
        "best_prefix_len": best_prefix_len,
        "best_passes": best_passes,
        "final_instruction_count": final_entry.get("instruction_count"),
        "best_instruction_count": best_instruction_count,
        "best_instruction_prefix_len": best_instruction_prefix_len,
        "best_instruction_passes": best_instruction_passes,
        "error": error,
        "error_kind": error_kind,
    }


def evaluate_topk_for_benchmark(
    graph: PassOrderGraph,
    bitcode_path: Path,
    paths: list[list[str]],
    *,
    heuristic: str,
    measure_instructions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-tu-topk-eval-") as tmp_str:
        workdir = Path(tmp_str)
        baseline_size = measure_text_size(bitcode_path, workdir)
        baseline_instruction_count = (
            measure_machine_instruction_count(bitcode_path, workdir)
            if measure_instructions
            else None
        )
        oz_size = None
        oz_instruction_count = None
        try:
            oz_bc = workdir / "oz.bc"
            optimize_oz(bitcode_path, oz_bc)
            oz_size = measure_text_size(oz_bc, workdir)
            if measure_instructions:
                oz_instruction_count = measure_machine_instruction_count(oz_bc, workdir)
        except Exception:
            oz_size = None

        prefix_cache: dict[tuple[str, ...], dict[str, Any]] = {
            (): {"bc": str(bitcode_path), "size": baseline_size, "error": ""}
        }
        if measure_instructions:
            prefix_cache[()]["instruction_count"] = baseline_instruction_count
        candidate_rows = []
        best_row: dict[str, Any] | None = None
        for candidate_index, passes in enumerate(paths, start=1):
            result = evaluate_candidate_with_prefix_cache(
                passes,
                workdir,
                prefix_cache,
                measure_instructions=measure_instructions,
            )
            row = {
                "benchmark": graph.benchmark,
                "heuristic": heuristic,
                "candidate_index": candidate_index,
                "start_pass": passes[0] if passes else "",
                "start_weight": graph.start_counts.get(passes[0], 0) if passes else 0,
                "top_k": len(paths),
                "bitcode_path": str(bitcode_path),
                "baseline_size": baseline_size,
                "oz_size": oz_size,
                "oz_delta": baseline_size - oz_size if oz_size is not None else None,
                "baseline_instruction_count": baseline_instruction_count,
                "oz_instruction_count": oz_instruction_count,
                "oz_instruction_delta": (
                    baseline_instruction_count - oz_instruction_count
                    if baseline_instruction_count is not None
                    and oz_instruction_count is not None
                    else None
                ),
                "final_size": result["final_size"],
                "final_delta": baseline_size - int(result["final_size"]),
                "best_size": result["best_size"],
                "best_delta": baseline_size - int(result["best_size"]),
                "best_prefix_len": result["best_prefix_len"],
                "final_instruction_count": result["final_instruction_count"],
                "final_instruction_delta": (
                    baseline_instruction_count - result["final_instruction_count"]
                    if baseline_instruction_count is not None
                    and result["final_instruction_count"] is not None
                    else None
                ),
                "best_instruction_count": result["best_instruction_count"],
                "best_instruction_delta": (
                    baseline_instruction_count - result["best_instruction_count"]
                    if baseline_instruction_count is not None
                    and result["best_instruction_count"] is not None
                    else None
                ),
                "best_instruction_prefix_len": result["best_instruction_prefix_len"],
                "passes": passes,
                "best_passes": result["best_passes"],
                "best_instruction_passes": result["best_instruction_passes"],
                "graph_score": score_path(graph, passes).to_dict(),
                "error": result["error"],
                "error_kind": result["error_kind"],
            }
            candidate_rows.append(row)
            if best_row is None or _selected_key(row) > _selected_key(best_row):
                best_row = row

        if best_row is None:
            best_row = {
                "benchmark": graph.benchmark,
                "heuristic": heuristic,
                "candidate_index": 0,
                "start_pass": "",
                "start_weight": 0,
                "top_k": len(paths),
                "bitcode_path": str(bitcode_path),
                "baseline_size": baseline_size,
                "oz_size": oz_size,
                "oz_delta": baseline_size - oz_size if oz_size is not None else None,
                "baseline_instruction_count": baseline_instruction_count,
                "oz_instruction_count": oz_instruction_count,
                "oz_instruction_delta": (
                    baseline_instruction_count - oz_instruction_count
                    if baseline_instruction_count is not None
                    and oz_instruction_count is not None
                    else None
                ),
                "final_size": baseline_size,
                "final_delta": 0,
                "best_size": baseline_size,
                "best_delta": 0,
                "best_prefix_len": 0,
                "final_instruction_count": baseline_instruction_count,
                "final_instruction_delta": 0 if baseline_instruction_count is not None else None,
                "best_instruction_count": baseline_instruction_count,
                "best_instruction_delta": 0 if baseline_instruction_count is not None else None,
                "best_instruction_prefix_len": 0,
                "passes": [],
                "best_passes": [],
                "best_instruction_passes": [],
                "graph_score": {},
                "error": "no candidate paths",
                "error_kind": "full_failure",
            }
        prefix_failures = sum(1 for entry in prefix_cache.values() if entry.get("error"))
        best_row["prefix_failures"] = prefix_failures
        return dict(best_row), candidate_rows, len(prefix_cache)


def _selected_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["best_delta"]),
        int(row["final_delta"]),
        -int(row["candidate_index"]),
    )


def build_report(
    graphs: dict[str, PassOrderGraph],
    bitcode_paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prefix_cache_counts: dict[str, int] = {}
    benchmarks = sorted(graphs)
    if args.limit:
        benchmarks = benchmarks[: args.limit]

    for index, benchmark in enumerate(benchmarks, start=1):
        print(
            f"[{index}/{len(benchmarks)}] {benchmark}: {args.heuristic} top-{args.top_k}",
            flush=True,
        )
        graph = graphs[benchmark]
        paths = generate_topk_paths(graph, args.heuristic, args)
        selected, candidates, cache_count = evaluate_topk_for_benchmark(
            graph,
            bitcode_paths[benchmark],
            paths,
            heuristic=args.heuristic,
            measure_instructions=args.measure_instructions,
        )
        selected_rows.append(selected)
        candidate_rows.extend(candidates)
        prefix_cache_counts[benchmark] = cache_count
        write_report(
            make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts),
            args.output,
        )

    return make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts)


def make_report_payload(
    args: argparse.Namespace,
    selected_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    prefix_cache_counts: dict[str, int],
) -> dict[str, Any]:
    summary = summarize_evaluations(selected_rows)
    add_instruction_summary(summary, selected_rows)
    add_failure_summary(summary, selected_rows)
    if selected_rows and args.heuristic in summary:
        summary[args.heuristic]["mean_selected_from_top_k"] = sum(
            int(row["candidate_index"]) for row in selected_rows
        ) / len(selected_rows)
        summary[args.heuristic]["mean_start_weight"] = sum(
            int(row.get("start_weight") or 0) for row in selected_rows
        ) / len(selected_rows)
    return {
        "graph_input": str(args.graph),
        "bitcode_dir": str(args.bitcode_dir),
        "config": {
            "heuristic": args.heuristic,
            "top_k": args.top_k,
            "top_starts": args.top_starts,
            "paths_per_start": args.paths_per_start,
            "max_length": args.max_length,
            "min_edge_weight": args.min_edge_weight,
            "random_walks": args.random_walks,
            "random_seed": args.random_seed,
            "exhaustive_length": args.exhaustive_length,
            "measure_instructions": args.measure_instructions,
        },
        "summary": summary,
        "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows),
        "candidate_rows": list(candidate_rows),
    }


def add_failure_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["heuristic"]), []).append(row)
    for heuristic, items in grouped.items():
        summary.setdefault(heuristic, {})
        summary[heuristic].update(
            {
                "tail_failures": sum(
                    1 for row in items if row.get("error_kind") == "tail_failure"
                ),
                "full_failures": sum(
                    1 for row in items if row.get("error_kind") == "full_failure"
                ),
                "prefix_failures": sum(
                    int(row.get("prefix_failures") or 0) for row in items
                ),
            }
        )


def add_instruction_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("baseline_instruction_count") is None:
            continue
        grouped.setdefault(str(row["heuristic"]), []).append(row)
    for heuristic, items in grouped.items():
        total_baseline = sum(int(row["baseline_instruction_count"]) for row in items)
        total_final_delta = sum(int(row.get("final_instruction_delta") or 0) for row in items)
        total_best_delta = sum(int(row.get("best_instruction_delta") or 0) for row in items)
        oz_rows = [row for row in items if row.get("oz_instruction_count") is not None]
        beats_oz = [
            row
            for row in oz_rows
            if row.get("best_instruction_count") is not None
            and int(row["best_instruction_count"]) < int(row["oz_instruction_count"])
        ]
        summary.setdefault(heuristic, {})
        summary[heuristic].update(
            {
                "total_baseline_instruction_count": total_baseline,
                "total_final_instruction_delta": total_final_delta,
                "total_best_instruction_delta": total_best_delta,
                "weighted_final_instruction_percent": (
                    100.0 * total_final_delta / total_baseline
                    if total_baseline
                    else 0.0
                ),
                "weighted_best_instruction_percent": (
                    100.0 * total_best_delta / total_baseline
                    if total_baseline
                    else 0.0
                ),
                "oz_instruction_available": len(oz_rows),
                "beats_oz_best_instruction": len(beats_oz),
                "beats_oz_best_instruction_percent": (
                    100.0 * len(beats_oz) / len(oz_rows)
                    if oz_rows
                    else None
                ),
                "mean_best_instruction_prefix_len": (
                    sum(int(row.get("best_instruction_prefix_len") or 0) for row in items)
                    / len(items)
                    if items
                    else 0.0
                ),
            }
        )


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate top-k graph-generated pass paths on whole translation units."
    )
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--heuristic",
        default="cycle_breaking_diverse_starts_top10",
        choices=[
            "cycle_breaking_diverse_starts_top10",
            "cycle_breaking_diverse_starts_topk",
            "cycle_breaking_max_path_top10",
            "cycle_breaking_max_path_topk",
            "cycle_breaking_top_starts_top_paths",
            "random_walk_top10",
            "random_walk_topk",
            "exhaustive_len6_top10",
        ],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-starts", type=int, default=10)
    parser.add_argument("--paths-per-start", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--random-walks", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--exhaustive-length", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--measure-instructions",
        action="store_true",
        help="Also measure machine instruction counts with llvm-objdump -d.",
    )
    parser.add_argument(
        "--site-data",
        action="append",
        default=[],
        type=Path,
        help="CompilerGym site-data root to copy whole benchmark bitcode from.",
    )
    parser.add_argument(
        "--overwrite-bitcode",
        action="store_true",
        help="Delete and restore bitcode-dir before evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_tools("opt", "llc", "llvm-size")
    if args.measure_instructions:
        require_tools("llvm-objdump")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.top_starts <= 0:
        raise ValueError("--top-starts must be positive")
    if args.paths_per_start <= 0:
        raise ValueError("--paths-per-start must be positive")
    if args.overwrite_bitcode and args.bitcode_dir.exists():
        shutil.rmtree(args.bitcode_dir)
    graphs = load_graphs_from_report(args.graph)
    benchmarks = sorted(graphs)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    bitcode_paths = write_translation_unit_bitcodes(
        benchmarks,
        args.bitcode_dir,
        site_data_paths=args.site_data,
    )
    report = build_report(
        {benchmark: graphs[benchmark] for benchmark in benchmarks},
        bitcode_paths,
        args,
    )
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
