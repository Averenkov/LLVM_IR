"""Segment-tree hierarchical-merge heuristic on whole translation units.

Phase 1 builds measured 4-pass super-vertices (same as measured_superpath:
distance-aware count graph, vertex-weight-budgeted length-3 paths, start-diverse
selection, measured on the TU). The selected segments are sorted and become the
leaves of a balanced binary tree; a bottom-up merge tries both concatenation
orders on the TU at each node and keeps the best (see segment_tree). The root's
best sequence is reported.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.translation_unit.measured_superpath import (
    node_support,
    prune_small_edges,
    select_diverse_by_start,
    vertex_budget_paths,
)
from llvm_ir.heuristics.translation_unit.segment_tree import segment_tree_merge
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
    build_pass_order_graph,
    load_function_pass_results_from_report,
)
from llvm_ir.stages.translation_unit.measure_cache import MeasureCache

HEURISTIC = "segment_tree"


def _row(benchmark, bitcode_path, passes, best_size, baseline_size, oz_size, baseline_instr, oz_instr, extra):
    return {
        "benchmark": benchmark, "heuristic": HEURISTIC, "candidate_index": 0,
        "bitcode_path": str(bitcode_path), "baseline_size": baseline_size, "oz_size": oz_size,
        "oz_delta": (baseline_size - oz_size) if oz_size is not None else None,
        "final_size": best_size, "final_delta": baseline_size - best_size,
        "best_size": best_size, "best_delta": baseline_size - best_size,
        "best_prefix_len": len(passes),
        "passes": list(passes), "best_passes": list(passes),
        "error": "" if passes else "no segments", "error_kind": "" if passes else "full_failure",
        "baseline_instruction_count": baseline_instr, "oz_instruction_count": oz_instr,
        "oz_instruction_delta": (baseline_instr - oz_instr) if baseline_instr is not None and oz_instr is not None else None,
        "final_instruction_count": None, "best_instruction_count": None,
        "final_instruction_delta": None, "best_instruction_delta": None,
        "best_instruction_prefix_len": 0, "best_instruction_passes": [],
        **extra,
    }


def evaluate_segment_tree_for_benchmark(benchmark, function_results, bitcode_path, *, args):
    with tempfile.TemporaryDirectory(prefix="llvm-ir-segtree-") as tmp_str:
        workdir = Path(tmp_str)
        measure_instructions = bool(args.measure_instructions)

        crashing: set[str] = set()
        if getattr(args, "prevalidate_passes", True):
            crashing = find_crashing_passes(bitcode_path, function_results, workdir)
            if crashing:
                function_results = filter_passes_from_results(function_results, crashing)

        graph = build_pass_order_graph(function_results, benchmark=benchmark, weight_mode="count_distance")
        support = node_support(function_results)
        edges = prune_small_edges(graph.edge_counts, args.prune_percent)
        nodes = sorted(graph.nodes)
        size_scale = min(args.max_size_scale, len(nodes) / max(1, args.size_ref))
        eff_select = max(args.min_budget, round(args.select_count * size_scale))

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
        cache = MeasureCache(getattr(args, "measure_cache_dir", None), bitcode_path)

        def measure(passes):
            res = evaluate_candidate_with_prefix_cache(
                list(passes), workdir, prefix_cache, measure_instructions=False, size_cache=cache
            )
            return int(res["best_size"]), tuple(res["best_passes"])

        # ---- Phase 1: measured 4-pass super-vertices (leaves) ----
        generated = vertex_budget_paths(
            nodes, edges, support, total_budget=args.gen_budget, path_nodes=args.segment_nodes
        )
        selected = (
            select_diverse_by_start(generated, eff_select)
            if getattr(args, "diverse_select", True)
            else generated[:eff_select]
        )
        leaves: list[tuple[tuple[str, ...], int]] = []
        for seg in selected:
            best_size, best_passes = measure(seg)
            leaves.append((best_passes, best_size))
        leaf_evals = len(prefix_cache) - 1

        # ---- Phase 2: segment-tree merge ----
        leaves.sort(key=lambda item: (item[1], item[0]))  # best (smallest .text) first
        (best_passes, best_size), merge_evals = segment_tree_merge(
            leaves, measure, max_length=args.max_length
        )
        if not best_passes or best_size is None:
            best_passes, best_size = (), baseline_size

        extra = {
            "kept_edges": len(edges), "graph_nodes": len(nodes), "size_scale": round(size_scale, 3),
            "leaves": len(leaves), "eff_select": eff_select, "crashing_passes": sorted(crashing),
            "leaf_real_evals": leaf_evals, "merge_real_evals": merge_evals,
            "real_evals": len(prefix_cache) - 1,
        }
        row = _row(benchmark, bitcode_path, best_passes, int(best_size),
                   baseline_size, oz_size, baseline_instr, oz_instr, extra)

        instruction_eval_cost = 0
        if measure_instructions and baseline_instr is not None and best_passes:
            instr_result, instruction_eval_cost = measure_deferred_candidate_instructions(
                list(best_passes), workdir, prefix_cache, size_cache=cache
            )
            row["best_instruction_count"] = instr_result["best_instruction_count"]
            row["best_instruction_prefix_len"] = instr_result["best_instruction_prefix_len"]
            row["best_instruction_passes"] = instr_result["best_instruction_passes"]
            row["final_instruction_count"] = instr_result["final_instruction_count"]
            row["best_instruction_delta"] = baseline_instr - instr_result["best_instruction_count"]
            row["final_instruction_delta"] = baseline_instr - instr_result["final_instruction_count"]
        row["instruction_eval_cost"] = instruction_eval_cost

        cache.flush()
        return row, [row], len(prefix_cache)


def make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts):
    summary = summarize_evaluations(selected_rows)
    add_instruction_summary(summary, selected_rows)
    add_failure_summary(summary, selected_rows)
    if selected_rows and HEURISTIC in summary:
        summary[HEURISTIC]["mean_real_evals"] = sum(int(r.get("real_evals") or 0) for r in selected_rows) / len(selected_rows)
        summary[HEURISTIC]["total_real_evals"] = sum(int(r.get("real_evals") or 0) for r in selected_rows)
    return {
        "comparison_input": str(args.comparison), "bitcode_dir": str(args.bitcode_dir),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary, "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows), "candidate_rows": list(candidate_rows),
    }


def _run_one(benchmark, function_results, bitcode_path, args):
    return benchmark, evaluate_segment_tree_for_benchmark(benchmark, function_results, bitcode_path, args=args)


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
            print(f"[{index}/{len(benchmarks)}] {benchmark}: segment_tree", flush=True)
            results[benchmark] = evaluate_segment_tree_for_benchmark(
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
            benchmark = futures[future]
            try:
                _b, result = future.result()
                results[benchmark] = result
            except Exception as exc:  # noqa: BLE001
                print(f"[{done + 1}/{len(benchmarks)}] {benchmark} FAILED, skipped: {type(exc).__name__}: {exc}", flush=True)
            done += 1
            if benchmark in results:
                print(f"[{done}/{len(benchmarks)}] {benchmark} done", flush=True)
            assemble()
    return assemble()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment-tree hierarchical-merge heuristic on whole TUs.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--select-count", type=int, default=500, help="Super-vertices (leaves) selected and measured.")
    parser.add_argument("--gen-budget", type=int, default=4000, help="Paths generated in phase 1 (n*k).")
    parser.add_argument("--segment-nodes", type=int, default=4, help="Passes per super-vertex.")
    parser.add_argument("--prune-percent", type=float, default=20.0)
    parser.add_argument("--max-length", type=int, default=16, help="Cap on merged-sequence length.")
    parser.add_argument("--size-ref", type=int, default=40)
    parser.add_argument("--max-size-scale", type=float, default=3.0)
    parser.add_argument("--min-budget", type=int, default=16)
    parser.add_argument("--diverse-select", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--prevalidate-passes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measure-cache-dir", default=".measure_cache")
    parser.add_argument("--no-measure-cache", dest="measure_cache_dir", action="store_const", const=None)
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
