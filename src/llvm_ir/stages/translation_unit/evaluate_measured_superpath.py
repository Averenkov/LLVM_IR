"""Two-wave measured-superpath evaluator (see measured_superpath core)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from random import Random

from llvm_ir.heuristics.translation_unit.measured_superpath import (
    concat_segments,
    node_support,
    prune_small_edges,
    sample_index,
    select_diverse_by_start,
    superpaths_by_length,
    vertex_budget_paths,
)
from llvm_ir.stages.function_search.pass_search import optimize_oz, require_tools
from llvm_ir.stages.translation_unit.evaluate import (
    summarize_evaluations,
    write_translation_unit_bitcodes,
)
from llvm_ir.stages.translation_unit.evaluate_bucket_dag_teacher import (
    edge_marginals_from_cache,
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

HEURISTIC = "measured_superpath"


def _make_row(benchmark, bitcode_path, candidate_index, passes, result,
              baseline_size, oz_size, baseline_instr, oz_instr, wave) -> dict[str, Any]:
    final_size = int(result["final_size"])
    best_size = int(result["best_size"])
    return {
        "benchmark": benchmark, "heuristic": HEURISTIC, "candidate_index": candidate_index, "wave": wave,
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
        "error": "no superpath candidates", "error_kind": "full_failure",
        "baseline_instruction_count": baseline_instr, "oz_instruction_count": oz_instr,
        "oz_instruction_delta": (baseline_instr - oz_instr) if baseline_instr is not None and oz_instr is not None else None,
        "final_instruction_count": baseline_instr, "final_instruction_delta": 0 if baseline_instr is not None else None,
        "best_instruction_count": baseline_instr, "best_instruction_delta": 0 if baseline_instr is not None else None,
        "best_instruction_prefix_len": 0, "best_instruction_passes": [],
    }


def _evaluate(candidates, *, benchmark, bitcode_path, workdir, prefix_cache, wave, offset,
              budget, baseline_size, oz_size, baseline_instr, oz_instr, excluded):
    rows: list[dict[str, Any]] = []
    evaluated: list[list[str]] = []
    for local_index, passes in enumerate(candidates):
        if tuple(passes) in excluded:
            continue
        if budget > 0 and (len(prefix_cache) - 1) >= budget:
            break
        result = evaluate_candidate_with_prefix_cache(list(passes), workdir, prefix_cache, measure_instructions=False)
        rows.append(_make_row(benchmark, bitcode_path, offset + local_index, list(passes), result,
                              baseline_size, oz_size, baseline_instr, oz_instr, wave))
        evaluated.append(list(passes))
        excluded.add(tuple(passes))
    return rows, evaluated


def evaluate_measured_superpath_for_benchmark(benchmark, function_results, bitcode_path, *, args):
    with tempfile.TemporaryDirectory(prefix="llvm-ir-msp-") as tmp_str:
        workdir = Path(tmp_str)
        measure_instructions = args.measure_instructions

        crashing: set[str] = set()
        if getattr(args, "prevalidate_passes", True):
            crashing = find_crashing_passes(bitcode_path, function_results, workdir)
            if crashing:
                function_results = filter_passes_from_results(function_results, crashing)

        graph = build_pass_order_graph(function_results, benchmark=benchmark, weight_mode="count_distance")
        support = node_support(function_results)
        edges = prune_small_edges(graph.edge_counts, args.prune_percent)
        nodes = sorted(graph.nodes)

        # Scale the TU-eval budgets with graph size: down for small graphs, and
        # *up* (to --max-size-scale) for large graphs, which are sparse at the
        # base budget and hold the most weighted improvement (e.g. tensorflow-v0_2,
        # 81 nodes). Graphs at exactly size_ref nodes keep the base budget.
        size_scale = min(args.max_size_scale, len(nodes) / max(1, args.size_ref))
        eff_select = max(args.min_budget, round(args.select_count * size_scale))
        eff_edges = max(args.min_budget, round(args.edge_samples * size_scale))
        eff_paths = max(args.min_budget, round(args.path_budget * size_scale))

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

        budget = args.max_real_evals_per_benchmark
        excluded: set[tuple[str, ...]] = set()

        def budget_ok() -> bool:
            return budget <= 0 or (len(prefix_cache) - 1) < budget

        # ---- Phase 1: generate gen_budget paths, keep top select_count, measure ----
        generated = vertex_budget_paths(
            nodes, edges, support, total_budget=args.gen_budget, path_nodes=args.segment_nodes
        )
        if getattr(args, "diverse_select", True):
            selected = select_diverse_by_start(generated, eff_select)
        else:
            selected = generated[:eff_select]
        rows1, _ = _evaluate(
            selected, benchmark=benchmark, bitcode_path=bitcode_path, workdir=workdir,
            prefix_cache=prefix_cache, wave=1, offset=0, budget=budget,
            baseline_size=baseline_size, oz_size=oz_size, baseline_instr=baseline_instr,
            oz_instr=oz_instr, excluded=excluded,
        )
        phase1_evals = len(prefix_cache) - 1

        # Super-vertices = measured segments; bias sampling by measured profit.
        segments = [tuple(r["passes"]) for r in rows1]
        profits = [float(r["best_delta"]) for r in rows1]
        floor = 1.0
        sample_w: dict[int, float] = {i: max(profits[i], 0.0) + floor for i in range(len(segments))}

        # ---- Phase 2: adaptive, learning measured super-edges ----
        super_edges: dict[tuple[int, int], float] = {}
        edge_rows: list[dict[str, Any]] = []
        tried: set[tuple[int, int]] = set()
        rng = Random(args.seed)
        attempts = 0
        explore_edges = 0
        attempt_cap = eff_edges * 6
        n_seg = len(segments)
        if args.waves > 1 and n_seg >= 2:
            while len(super_edges) < eff_edges and attempts < attempt_cap and budget_ok():
                attempts += 1
                # epsilon-greedy: explore uniformly, else exploit profit-weighted "best-to-best".
                if rng.random() < args.epsilon_edge:
                    i = rng.randrange(n_seg)
                    j = rng.randrange(n_seg)
                    explore_edges += 1
                else:
                    i = sample_index(sample_w, rng)
                    j = sample_index(sample_w, rng)
                if i is None or j is None or i == j or (i, j) in tried:
                    continue
                tried.add((i, j))
                cand = concat_segments(segments[i], segments[j])[: args.super_max_length]
                if not cand or tuple(cand) in excluded:
                    continue
                result = evaluate_candidate_with_prefix_cache(list(cand), workdir, prefix_cache, measure_instructions=False)
                edge_rows.append(_make_row(benchmark, bitcode_path, len(rows1) + len(edge_rows), list(cand),
                                           result, baseline_size, oz_size, baseline_instr, oz_instr, 2))
                excluded.add(tuple(cand))
                edge_profit = float(baseline_size - int(result["best_size"]))
                super_edges[(i, j)] = edge_profit
                # Learn: reward vertices that chain synergistically beyond their best single.
                bonus = max(edge_profit - max(profits[i], profits[j]), 0.0)
                sample_w[i] += args.learn_alpha * bonus
                sample_w[j] += args.learn_alpha * bonus
        phase2_evals = (len(prefix_cache) - 1) - phase1_evals

        # ---- Phase 3: top super-paths of lengths 2..5, split evenly, measured ----
        rows3: list[dict[str, Any]] = []
        lengths = tuple(int(x) for x in str(args.super_lengths).split(",") if x.strip())
        if super_edges and budget_ok():
            per_length = max(1, eff_paths // max(1, len(lengths)))
            supercands = superpaths_by_length(
                segments, super_edges, profits, lengths=lengths, per_length=per_length,
                beam=args.super_beam, max_length=args.super_max_length,
                vertex_weight=args.vertex_weight,
            )
            rows3, _ = _evaluate(
                supercands, benchmark=benchmark, bitcode_path=bitcode_path, workdir=workdir,
                prefix_cache=prefix_cache, wave=3, offset=len(rows1) + len(edge_rows), budget=budget,
                baseline_size=baseline_size, oz_size=oz_size, baseline_instr=baseline_instr,
                oz_instr=oz_instr, excluded=excluded,
            )
        phase3_evals = (len(prefix_cache) - 1) - phase1_evals - phase2_evals

        candidate_rows = rows1 + edge_rows + rows3
        common = {
            "kept_edges": len(edges), "pruned_percent": args.prune_percent,
            "graph_nodes": len(nodes), "size_scale": round(size_scale, 3),
            "segment_floor": min((len(p) for p in selected), default=0),
            "eff_select": eff_select, "eff_edges": eff_edges, "eff_paths": eff_paths,
            "gen_budget": args.gen_budget, "select_count": len(rows1),
            "measured_edges": len(super_edges), "edge_attempts": attempts,
            "explore_edges": explore_edges,
            "distinct_starts": len({s[0] for s in segments if s}),
            "crashing_passes": sorted(crashing),
            "phase1_real_evals": phase1_evals, "phase2_real_evals": phase2_evals,
            "phase3_real_evals": phase3_evals, "real_evals": len(prefix_cache) - 1,
            "waves": args.waves,
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
    return {
        "comparison_input": str(args.comparison), "bitcode_dir": str(args.bitcode_dir),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary, "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows), "candidate_rows": list(candidate_rows),
    }


def _run_one(benchmark, function_results, bitcode_path, args):
    return benchmark, evaluate_measured_superpath_for_benchmark(benchmark, function_results, bitcode_path, args=args)


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
            print(f"[{index}/{len(benchmarks)}] {benchmark}: measured_superpath waves={args.waves}", flush=True)
            results[benchmark] = evaluate_measured_superpath_for_benchmark(
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
    parser = argparse.ArgumentParser(description="Two-wave measured-superpath heuristic.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--gen-budget", type=int, default=2000, help="Paths generated in phase 1 (n*k).")
    parser.add_argument("--select-count", type=int, default=250, help="Top generated paths measured as super-vertices.")
    parser.add_argument("--segment-nodes", type=int, default=4, help="Passes per phase-1 path (length+1).")
    parser.add_argument("--prune-percent", type=float, default=20.0, help="Drop this %% of smallest edges.")
    parser.add_argument("--edge-samples", type=int, default=250, help="Measured super-edges sampled (best-to-best).")
    parser.add_argument("--path-budget", type=int, default=250, help="Super-paths measured, split across lengths.")
    parser.add_argument("--learn-alpha", type=float, default=1.0, help="Online weight boost per unit edge synergy.")
    parser.add_argument(
        "--diverse-select",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cover each starting pass before filling by weight when selecting segments.",
    )
    parser.add_argument("--epsilon-edge", type=float, default=0.2, help="Fraction of phase-2 edges sampled uniformly (explore).")
    parser.add_argument("--size-ref", type=int, default=40, help="Graph nodes at which budgets reach base size.")
    parser.add_argument("--max-size-scale", type=float, default=3.0, help="Max budget multiplier for large graphs.")
    parser.add_argument("--min-budget", type=int, default=16, help="Floor for scaled per-phase budgets on small graphs.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vertex-weight", type=float, default=1.0, help="Weight of segment profits in super-path score (0 = edges only).")
    parser.add_argument("--super-beam", type=int, default=96)
    parser.add_argument("--super-max-length", type=int, default=20)
    parser.add_argument("--super-lengths", default="2,3,4,5", help="Super-path segment counts to emit.")
    parser.add_argument("--max-real-evals-per-benchmark", type=int, default=0)
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
