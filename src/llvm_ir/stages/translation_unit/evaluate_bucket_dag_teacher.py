"""Two-wave bucket-DAG heuristic that uses the whole-TU run as a teacher.

Wave 1 derives top paths from the function-level pass-order graph (the same
bucket-layered DAG construction as ``bucket_dag``) and measures them on the real
translation unit. From those measurements it attributes, to every order-graph
edge ``u -> v``, the average measured ``.text`` byte gain seen when ``v`` was
applied right after ``u``. These measured marginals re-weight the graph, which
re-ranks the buckets and changes which forward edges survive. Wave 2 derives a
fresh batch of top paths from this teacher-calibrated graph and measures them.
The best measured prefix across both waves is reported.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.translation_unit.bucket_dag import (
    BucketDagConfig,
    bucket_layer_top_paths,
    build_teacher_graph,
)
from llvm_ir.stages.function_search.pass_search import (
    LLVMCommandError,
    apply_pass_sequence,
    optimize_oz,
    require_tools,
)
from llvm_ir.stages.translation_unit.evaluate import (
    summarize_evaluations,
    write_translation_unit_bitcodes,
)
from llvm_ir.stages.translation_unit.evaluate_chunk_forest import (
    filter_passes_from_results,
    find_crashing_passes,
    group_results_by_benchmark,
)
from llvm_ir.stages.translation_unit.evaluate_topk_paths import (
    add_failure_summary,
    add_instruction_summary,
    evaluate_candidate_with_prefix_cache,
    measure_deferred_candidate_instructions,
    measure_text_and_instruction_count,
    measure_text_size,
    write_report,
)
from llvm_ir.stages.translation_unit.graph.order_graph import (
    build_pass_order_graph,
    load_function_pass_results_from_report,
)

HEURISTIC = "bucket_dag_teacher"


def _budget_left(prefix_cache: dict, budget: int) -> bool:
    return budget <= 0 or (len(prefix_cache) - 1) < budget


def edge_marginals_from_cache(
    paths: list[list[str]],
    prefix_cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[tuple[str, str], float]:
    """Mean measured ``.text`` byte gain per edge ``u -> v`` across measured paths.

    For each step ``v`` applied after ``u`` in a path, the marginal is
    ``size(prefix up to u) - size(prefix up to v)`` read from the prefix cache.
    Steps past a crashed prefix are skipped.
    """
    observations: dict[tuple[str, str], list[int]] = {}
    for path in paths:
        prev_entry = prefix_cache.get(())
        for index, pass_name in enumerate(path):
            entry = prefix_cache.get(tuple(path[: index + 1]))
            if entry is None or entry.get("error"):
                break
            if index >= 1 and prev_entry is not None and not prev_entry.get("error"):
                observations.setdefault((path[index - 1], pass_name), []).append(
                    int(prev_entry["size"]) - int(entry["size"])
                )
            prev_entry = entry
    return {edge: sum(values) / len(values) for edge, values in observations.items()}


def _make_row(
    benchmark: str,
    bitcode_path: Path,
    candidate_index: int,
    passes: list[str],
    result: dict[str, Any],
    baseline_size: int,
    oz_size: int | None,
    baseline_instruction_count: int | None,
    oz_instruction_count: int | None,
    wave: int,
) -> dict[str, Any]:
    final_size = int(result["final_size"])
    best_size = int(result["best_size"])
    return {
        "benchmark": benchmark,
        "heuristic": HEURISTIC,
        "candidate_index": candidate_index,
        "wave": wave,
        "bitcode_path": str(bitcode_path),
        "baseline_size": baseline_size,
        "oz_size": oz_size,
        "oz_delta": (baseline_size - oz_size) if oz_size is not None else None,
        "final_size": final_size,
        "final_delta": baseline_size - final_size,
        "best_size": best_size,
        "best_delta": baseline_size - best_size,
        "best_prefix_len": int(result["best_prefix_len"]),
        "passes": list(passes),
        "best_passes": list(result["best_passes"]),
        "error": result["error"],
        "error_kind": result["error_kind"],
        "baseline_instruction_count": baseline_instruction_count,
        "oz_instruction_count": oz_instruction_count,
        "oz_instruction_delta": (
            baseline_instruction_count - oz_instruction_count
            if baseline_instruction_count is not None and oz_instruction_count is not None
            else None
        ),
        "final_instruction_count": result.get("final_instruction_count"),
        "best_instruction_count": result.get("best_instruction_count"),
        "best_instruction_delta": (
            baseline_instruction_count - int(result["best_instruction_count"])
            if baseline_instruction_count is not None
            and result.get("best_instruction_count") is not None
            else None
        ),
        "best_instruction_prefix_len": result.get("best_instruction_prefix_len", 0),
        "best_instruction_passes": result.get("best_instruction_passes", []),
    }


def _evaluate_paths(
    paths, *, benchmark, bitcode_path, workdir, prefix_cache, wave, offset, budget,
    baseline_size, oz_size, baseline_instruction_count, oz_instruction_count, excluded,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    rows: list[dict[str, Any]] = []
    evaluated: list[list[str]] = []
    for local_index, path in enumerate(paths):
        if tuple(path) in excluded:
            continue
        if not _budget_left(prefix_cache, budget):
            break
        result = evaluate_candidate_with_prefix_cache(
            list(path), workdir, prefix_cache, measure_instructions=False
        )
        rows.append(
            _make_row(
                benchmark, bitcode_path, offset + local_index, list(path), result,
                baseline_size, oz_size, baseline_instruction_count, oz_instruction_count, wave,
            )
        )
        evaluated.append(list(path))
        excluded.add(tuple(path))
    return rows, evaluated


def evaluate_bucket_dag_teacher_for_benchmark(
    benchmark: str,
    function_results,
    bitcode_path: Path,
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-bucket-teacher-") as tmp_str:
        workdir = Path(tmp_str)
        measure_instructions = args.measure_instructions

        crashing: set[str] = set()
        if getattr(args, "prevalidate_passes", True):
            crashing = find_crashing_passes(bitcode_path, function_results, workdir)
            if crashing:
                function_results = filter_passes_from_results(function_results, crashing)

        graph = build_pass_order_graph(function_results, benchmark=benchmark, weight_mode="delta")

        if measure_instructions:
            baseline_size, baseline_instruction_count = measure_text_and_instruction_count(bitcode_path, workdir)
        else:
            baseline_size = measure_text_size(bitcode_path, workdir)
            baseline_instruction_count = None
        oz_size = None
        oz_instruction_count = None
        try:
            oz_bc = workdir / "oz.bc"
            optimize_oz(bitcode_path, oz_bc)
            if measure_instructions:
                oz_size, oz_instruction_count = measure_text_and_instruction_count(oz_bc, workdir)
            else:
                oz_size = measure_text_size(oz_bc, workdir)
        except Exception:
            oz_size = None

        prefix_cache: dict[tuple[str, ...], dict[str, Any]] = {
            (): {"bc": str(bitcode_path), "size": baseline_size, "error": ""}
        }
        if measure_instructions:
            prefix_cache[()]["instruction_count"] = baseline_instruction_count

        config = BucketDagConfig(
            chunk_size=args.chunk_size,
            max_length=args.max_length,
            min_edge_weight=args.min_edge_weight,
        )
        waves = max(1, args.waves)
        wave1_target = args.top_k if waves == 1 else max(1, args.top_k // waves)
        wave2_target = max(0, args.top_k - wave1_target)
        budget = args.max_real_evals_per_benchmark
        excluded: set[tuple[str, ...]] = set()

        # ---- Wave 1: function-level bucket DAG ----
        paths1 = bucket_layer_top_paths(graph, config=config, top_k=wave1_target)
        rows1, evaluated1 = _evaluate_paths(
            paths1, benchmark=benchmark, bitcode_path=bitcode_path, workdir=workdir,
            prefix_cache=prefix_cache, wave=1, offset=0, budget=budget,
            baseline_size=baseline_size, oz_size=oz_size,
            baseline_instruction_count=baseline_instruction_count,
            oz_instruction_count=oz_instruction_count, excluded=excluded,
        )
        wave1_evals = len(prefix_cache) - 1

        # ---- Teacher calibration from measured TU edge marginals ----
        edge_means = edge_marginals_from_cache(evaluated1, prefix_cache)

        rows2: list[dict[str, Any]] = []
        teacher_edges = 0
        if waves > 1 and wave2_target and edge_means and _budget_left(prefix_cache, budget):
            teacher = build_teacher_graph(graph, edge_means)
            teacher_edges = sum(1 for w in teacher.edge_counts.values() if w > 0)
            paths2 = bucket_layer_top_paths(teacher, config=config, top_k=wave2_target)
            rows2, _ = _evaluate_paths(
                paths2, benchmark=benchmark, bitcode_path=bitcode_path, workdir=workdir,
                prefix_cache=prefix_cache, wave=2, offset=len(rows1), budget=budget,
                baseline_size=baseline_size, oz_size=oz_size,
                baseline_instruction_count=baseline_instruction_count,
                oz_instruction_count=oz_instruction_count, excluded=excluded,
            )
        wave2_evals = (len(prefix_cache) - 1) - wave1_evals

        candidate_rows = rows1 + rows2
        common = {
            "chunk_size": args.chunk_size,
            "waves": waves,
            "crashing_passes": sorted(crashing),
            "edges_measured": len(edge_means),
            "teacher_edges": teacher_edges,
            "wave1_real_evals": wave1_evals,
            "wave2_real_evals": wave2_evals,
            "real_evals": len(prefix_cache) - 1,
        }

        if candidate_rows:
            best_row = dict(max(candidate_rows, key=lambda r: (r["best_delta"], r["final_delta"], -r["candidate_index"])))
        else:
            best_row = _empty_row(benchmark, bitcode_path, baseline_size, oz_size,
                                  baseline_instruction_count, oz_instruction_count)

        instruction_eval_cost = 0
        if measure_instructions and baseline_instruction_count is not None and best_row["best_passes"]:
            instr_result, instruction_eval_cost = measure_deferred_candidate_instructions(
                list(best_row["best_passes"]), workdir, prefix_cache
            )
            best_row["best_instruction_count"] = instr_result["best_instruction_count"]
            best_row["best_instruction_prefix_len"] = instr_result["best_instruction_prefix_len"]
            best_row["best_instruction_passes"] = instr_result["best_instruction_passes"]
            best_row["final_instruction_count"] = instr_result["final_instruction_count"]
            best_row["best_instruction_delta"] = baseline_instruction_count - instr_result["best_instruction_count"]
            best_row["final_instruction_delta"] = baseline_instruction_count - instr_result["final_instruction_count"]

        best_row.update(common)
        best_row["instruction_eval_cost"] = instruction_eval_cost
        for row in candidate_rows:
            row.update(common)
        return best_row, candidate_rows, len(prefix_cache)


def _empty_row(benchmark, bitcode_path, baseline_size, oz_size, baseline_instr, oz_instr):
    return {
        "benchmark": benchmark, "heuristic": HEURISTIC, "candidate_index": 0,
        "bitcode_path": str(bitcode_path), "baseline_size": baseline_size, "oz_size": oz_size,
        "oz_delta": (baseline_size - oz_size) if oz_size is not None else None,
        "final_size": baseline_size, "final_delta": 0, "best_size": baseline_size,
        "best_delta": 0, "best_prefix_len": 0, "passes": [], "best_passes": [],
        "error": "no bucket-dag candidates", "error_kind": "full_failure",
        "baseline_instruction_count": baseline_instr, "oz_instruction_count": oz_instr,
        "oz_instruction_delta": (baseline_instr - oz_instr) if baseline_instr is not None and oz_instr is not None else None,
        "final_instruction_count": baseline_instr, "final_instruction_delta": 0 if baseline_instr is not None else None,
        "best_instruction_count": baseline_instr, "best_instruction_delta": 0 if baseline_instr is not None else None,
        "best_instruction_prefix_len": 0, "best_instruction_passes": [],
    }


def make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts):
    summary = summarize_evaluations(selected_rows)
    add_instruction_summary(summary, selected_rows)
    add_failure_summary(summary, selected_rows)
    if selected_rows and HEURISTIC in summary:
        summary[HEURISTIC]["mean_real_evals"] = sum(int(r.get("real_evals") or 0) for r in selected_rows) / len(selected_rows)
        summary[HEURISTIC]["total_real_evals"] = sum(int(r.get("real_evals") or 0) for r in selected_rows)
    return {
        "comparison_input": str(args.comparison),
        "bitcode_dir": str(args.bitcode_dir),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary,
        "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows),
        "candidate_rows": list(candidate_rows),
    }


def _run_one(benchmark, function_results, bitcode_path, args):
    """Worker entry point: evaluate one benchmark (picklable for ProcessPool)."""
    return benchmark, evaluate_bucket_dag_teacher_for_benchmark(
        benchmark, function_results, bitcode_path, args=args
    )


def build_report(groups, bitcode_paths, args) -> dict[str, Any]:
    benchmarks = sorted(groups)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    jobs = max(1, getattr(args, "jobs", 1))
    results: dict[str, tuple] = {}

    def assemble() -> dict[str, Any]:
        ordered = [b for b in benchmarks if b in results]
        selected_rows = [results[b][0] for b in ordered]
        candidate_rows = [row for b in ordered for row in results[b][1]]
        prefix_cache_counts = {b: results[b][2] for b in ordered}
        payload = make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts)
        write_report(payload, args.output)
        return payload

    if jobs == 1:
        for index, benchmark in enumerate(benchmarks, start=1):
            print(f"[{index}/{len(benchmarks)}] {benchmark}: bucket_dag_teacher waves={args.waves} top_k={args.top_k}", flush=True)
            results[benchmark] = evaluate_bucket_dag_teacher_for_benchmark(
                benchmark, groups[benchmark], bitcode_paths[benchmark], args=args
            )
            assemble()
        return assemble()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"running {len(benchmarks)} benchmarks with --jobs {jobs}", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one, benchmark, groups[benchmark], bitcode_paths[benchmark], args): benchmark
            for benchmark in benchmarks
        }
        done = 0
        for future in as_completed(futures):
            benchmark, result = future.result()
            results[benchmark] = result
            done += 1
            print(f"[{done}/{len(benchmarks)}] {benchmark} done", flush=True)
            assemble()
    return assemble()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-wave bucket-DAG heuristic with whole-TU teacher.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--max-real-evals-per-benchmark", type=int, default=0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Process benchmarks in parallel across this many worker processes.",
    )
    parser.add_argument("--prevalidate-passes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--measure-instructions", action="store_true")
    parser.add_argument("--site-data", action="append", default=[], type=Path)
    parser.add_argument("--overwrite-bitcode", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_tools("opt", "llc", "llvm-size")
    if args.measure_instructions:
        require_tools("llvm-objdump")
    if args.overwrite_bitcode and args.bitcode_dir.exists():
        shutil.rmtree(args.bitcode_dir)
    results = load_function_pass_results_from_report(args.comparison, algorithm=args.algorithm or None)
    groups = group_results_by_benchmark(results)
    benchmarks = sorted(groups)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    bitcode_paths = write_translation_unit_bitcodes(benchmarks, args.bitcode_dir, site_data_paths=args.site_data)
    report = build_report({b: groups[b] for b in benchmarks}, bitcode_paths, args)
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
