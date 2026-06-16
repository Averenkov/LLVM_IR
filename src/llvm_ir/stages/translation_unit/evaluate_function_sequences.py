"""Baseline heuristic: apply each function's best pass sequence to the whole TU.

The per-function search already found a good pass sequence for every function in
a benchmark. This heuristic does the simplest possible thing: take each unique
per-function sequence as a whole-TU candidate, apply it to the whole-TU bitcode,
measure ``.text`` (best prefix), and keep the best. No graph, no waves -- a
direct test of how well locally-found sequences transfer to the full TU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from llvm_ir.stages.function_search.pass_search import optimize_oz, require_tools
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
    load_function_pass_results_from_report,
)

HEURISTIC = "function_sequences"


def _make_row(benchmark, bitcode_path, candidate_index, passes, result,
              baseline_size, oz_size, baseline_instr, oz_instr) -> dict[str, Any]:
    final_size = int(result["final_size"])
    best_size = int(result["best_size"])
    return {
        "benchmark": benchmark, "heuristic": HEURISTIC, "candidate_index": candidate_index,
        "bitcode_path": str(bitcode_path), "baseline_size": baseline_size, "oz_size": oz_size,
        "oz_delta": (baseline_size - oz_size) if oz_size is not None else None,
        "final_size": final_size, "final_delta": baseline_size - final_size,
        "best_size": best_size, "best_delta": baseline_size - best_size,
        "best_prefix_len": int(result["best_prefix_len"]),
        "passes": list(passes), "best_passes": list(result["best_passes"]),
        "error": result["error"], "error_kind": result["error_kind"],
        "baseline_instruction_count": baseline_instr, "oz_instruction_count": oz_instr,
        "oz_instruction_delta": (baseline_instr - oz_instr) if baseline_instr is not None and oz_instr is not None else None,
        "final_instruction_count": result.get("final_instruction_count"),
        "best_instruction_count": result.get("best_instruction_count"),
        "best_instruction_delta": (
            baseline_instr - int(result["best_instruction_count"])
            if baseline_instr is not None and result.get("best_instruction_count") is not None else None
        ),
        "best_instruction_prefix_len": result.get("best_instruction_prefix_len", 0),
        "best_instruction_passes": result.get("best_instruction_passes", []),
    }


def _empty_row(benchmark, bitcode_path, baseline_size, oz_size, baseline_instr, oz_instr):
    return {
        "benchmark": benchmark, "heuristic": HEURISTIC, "candidate_index": 0,
        "bitcode_path": str(bitcode_path), "baseline_size": baseline_size, "oz_size": oz_size,
        "oz_delta": (baseline_size - oz_size) if oz_size is not None else None,
        "final_size": baseline_size, "final_delta": 0, "best_size": baseline_size, "best_delta": 0,
        "best_prefix_len": 0, "passes": [], "best_passes": [],
        "error": "no function sequences", "error_kind": "full_failure",
        "baseline_instruction_count": baseline_instr, "oz_instruction_count": oz_instr,
        "oz_instruction_delta": (baseline_instr - oz_instr) if baseline_instr is not None and oz_instr is not None else None,
        "final_instruction_count": baseline_instr, "final_instruction_delta": 0 if baseline_instr is not None else None,
        "best_instruction_count": baseline_instr, "best_instruction_delta": 0 if baseline_instr is not None else None,
        "best_instruction_prefix_len": 0, "best_instruction_passes": [],
    }


def unique_function_sequences(function_results) -> list[list[str]]:
    """Distinct non-empty per-function best sequences, deterministically ordered."""
    seen: set[tuple[str, ...]] = set()
    sequences: list[list[str]] = []
    for result in function_results:
        passes = [p for p in result.passes]
        key = tuple(passes)
        if not passes or key in seen:
            continue
        seen.add(key)
        sequences.append(passes)
    sequences.sort()
    return sequences


def evaluate_function_sequences_for_benchmark(benchmark, function_results, bitcode_path, *, args):
    with tempfile.TemporaryDirectory(prefix="llvm-ir-fnseq-") as tmp_str:
        workdir = Path(tmp_str)
        measure_instructions = args.measure_instructions

        crashing: set[str] = set()
        if getattr(args, "prevalidate_passes", True):
            crashing = find_crashing_passes(bitcode_path, function_results, workdir)
            if crashing:
                function_results = filter_passes_from_results(function_results, crashing)

        if measure_instructions:
            baseline_size, baseline_instr = measure_text_and_instruction_count(bitcode_path, workdir)
        else:
            baseline_size = measure_text_size(bitcode_path, workdir)
            baseline_instr = None
        oz_size = None
        oz_instr = None
        try:
            oz_bc = workdir / "oz.bc"
            optimize_oz(bitcode_path, oz_bc)
            if measure_instructions:
                oz_size, oz_instr = measure_text_and_instruction_count(oz_bc, workdir)
            else:
                oz_size = measure_text_size(oz_bc, workdir)
        except Exception:
            oz_size = None

        prefix_cache: dict[tuple[str, ...], dict[str, Any]] = {
            (): {"bc": str(bitcode_path), "size": baseline_size, "error": ""}
        }
        if measure_instructions:
            prefix_cache[()]["instruction_count"] = baseline_instr

        candidates = unique_function_sequences(function_results)
        candidate_rows: list[dict[str, Any]] = []
        for index, passes in enumerate(candidates):
            result = evaluate_candidate_with_prefix_cache(list(passes), workdir, prefix_cache, measure_instructions=False)
            candidate_rows.append(_make_row(benchmark, bitcode_path, index, list(passes), result,
                                            baseline_size, oz_size, baseline_instr, oz_instr))

        common = {
            "candidate_count": len(candidates),
            "crashing_passes": sorted(crashing),
            "real_evals": len(prefix_cache) - 1,
        }
        if candidate_rows:
            best_row = dict(max(candidate_rows, key=lambda r: (r["best_delta"], r["final_delta"], -r["candidate_index"])))
        else:
            best_row = _empty_row(benchmark, bitcode_path, baseline_size, oz_size, baseline_instr, oz_instr)

        instruction_eval_cost = 0
        if measure_instructions and baseline_instr is not None and best_row["best_passes"]:
            instr_result, instruction_eval_cost = measure_deferred_candidate_instructions(
                list(best_row["best_passes"]), workdir, prefix_cache
            )
            best_row["best_instruction_count"] = instr_result["best_instruction_count"]
            best_row["best_instruction_prefix_len"] = instr_result["best_instruction_prefix_len"]
            best_row["best_instruction_passes"] = instr_result["best_instruction_passes"]
            best_row["final_instruction_count"] = instr_result["final_instruction_count"]
            best_row["best_instruction_delta"] = baseline_instr - instr_result["best_instruction_count"]
            best_row["final_instruction_delta"] = baseline_instr - instr_result["final_instruction_count"]

        best_row.update(common)
        best_row["instruction_eval_cost"] = instruction_eval_cost
        for row in candidate_rows:
            row.update(common)
        return best_row, candidate_rows, len(prefix_cache)


def make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts):
    summary = summarize_evaluations(selected_rows)
    add_instruction_summary(summary, selected_rows)
    add_failure_summary(summary, selected_rows)
    if selected_rows and HEURISTIC in summary:
        summary[HEURISTIC]["mean_real_evals"] = sum(int(r.get("real_evals") or 0) for r in selected_rows) / len(selected_rows)
        summary[HEURISTIC]["total_real_evals"] = sum(int(r.get("real_evals") or 0) for r in selected_rows)
        summary[HEURISTIC]["mean_candidate_count"] = sum(int(r.get("candidate_count") or 0) for r in selected_rows) / len(selected_rows)
    return {
        "comparison_input": str(args.comparison), "bitcode_dir": str(args.bitcode_dir),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary, "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows), "candidate_rows": list(candidate_rows),
    }


def _run_one(benchmark, function_results, bitcode_path, args):
    return benchmark, evaluate_function_sequences_for_benchmark(benchmark, function_results, bitcode_path, args=args)


def build_report(groups, bitcode_paths, args) -> dict[str, Any]:
    benchmarks = sorted(groups)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    jobs = max(1, getattr(args, "jobs", 1))
    results: dict[str, tuple] = {}

    def assemble():
        ordered = [b for b in benchmarks if b in results]
        selected_rows = [results[b][0] for b in ordered]
        candidate_rows = [row for b in ordered for row in results[b][1]]
        prefix_cache_counts = {b: results[b][2] for b in ordered}
        payload = make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts)
        write_report(payload, args.output)
        return payload

    if jobs == 1:
        for index, benchmark in enumerate(benchmarks, start=1):
            print(f"[{index}/{len(benchmarks)}] {benchmark}: function_sequences", flush=True)
            results[benchmark] = evaluate_function_sequences_for_benchmark(
                benchmark, groups[benchmark], bitcode_paths[benchmark], args=args
            )
            assemble()
        return assemble()

    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"running {len(benchmarks)} benchmarks with --jobs {jobs}", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_run_one, b, groups[b], bitcode_paths[b], args): b for b in benchmarks}
        done = 0
        for future in as_completed(futures):
            benchmark, result = future.result()
            results[benchmark] = result
            done += 1
            print(f"[{done}/{len(benchmarks)}] {benchmark} done", flush=True)
            assemble()
    return assemble()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply each function's best sequence to the whole TU (baseline).")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
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
