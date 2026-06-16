"""Evaluate the TU-MCTS heuristic on whole translation units.

TU-MCTS searches the pass-sequence prefix space online, using the real measured
``.text`` size on the whole-TU bitcode as the reward and the per-function
pass-order graph only as a soft PUCT prior. See
``llvm_ir.heuristics.translation_unit.tu_mcts`` for the search itself; this
module wires it to real LLVM measurement, a prefix cache, pass pre-validation,
and the shared report format.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.translation_unit.tu_mcts import (
    BudgetExhausted,
    MCTSConfig,
    MeasureResult,
    run_mcts,
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
    measure_text_and_instruction_count,
    measure_text_size,
    write_report,
)
from llvm_ir.stages.translation_unit.graph.order_graph import (
    build_pass_order_graph,
    load_function_pass_results_from_report,
)

HEURISTIC = "tu_mcts"


def _make_prior_fn(graph, actions):
    """Soft prior over candidate next passes from the delta-weighted order graph."""

    def prior_fn(prefix: tuple[str, ...]) -> dict[str, float]:
        if not prefix:
            return {a: float(graph.start_counts.get(a, 0)) for a in actions}
        last = prefix[-1]
        return {a: float(graph.edge_weight(last, a)) for a in actions}

    return prior_fn


def _make_measure_fn(bitcode_path, workdir, prefix_cache, budget, counter):
    """Incremental, budget-bounded ``.text`` measurement backed by a prefix cache."""

    def measure(key: tuple[str, ...]) -> MeasureResult:
        cached = prefix_cache.get(key)
        if cached is not None:
            return MeasureResult(cached["size"], cached["error"])
        if budget and counter["evals"] >= budget:
            raise BudgetExhausted()
        parent = prefix_cache[key[:-1]]
        if parent["error"]:
            entry = {"bc": parent["bc"], "size": parent["size"], "error": parent["error"]}
            prefix_cache[key] = entry
            return MeasureResult(None, parent["error"])
        out_bc = workdir / f"mcts_{len(prefix_cache):05d}.bc"
        counter["evals"] += 1
        try:
            apply_pass_sequence(Path(parent["bc"]), [key[-1]], out_bc)
            size = measure_text_size(out_bc, workdir)
        except LLVMCommandError as exc:
            entry = {"bc": parent["bc"], "size": parent["size"], "error": f"{type(exc).__name__}: {exc}"}
            prefix_cache[key] = entry
            return MeasureResult(None, entry["error"])
        entry = {"bc": str(out_bc), "size": size, "error": ""}
        prefix_cache[key] = entry
        return MeasureResult(size, "")

    return measure


def _best_prefix_instructions(
    bitcode_path: Path,
    workdir: Path,
    best_passes: tuple[str, ...],
    baseline_instruction_count: int,
) -> dict[str, Any]:
    """Measure machine-instruction counts along the chosen best sequence."""
    current_bc = bitcode_path
    best_instr = baseline_instruction_count
    best_len = 0
    best_size = None
    final_instr = baseline_instruction_count
    for index, pass_name in enumerate(best_passes, start=1):
        out_bc = workdir / f"mcts_instr_{index:03d}.bc"
        try:
            apply_pass_sequence(current_bc, [pass_name], out_bc)
            size, instr = measure_text_and_instruction_count(out_bc, workdir)
        except LLVMCommandError:
            break
        current_bc = out_bc
        final_instr = instr
        if best_size is None or size < best_size:
            best_size, best_instr, best_len = size, instr, index
    return {
        "final_instruction_count": final_instr,
        "best_instruction_count": best_instr,
        "best_instruction_prefix_len": best_len,
        "best_instruction_passes": list(best_passes[:best_len]),
    }


def evaluate_tu_mcts_for_benchmark(
    benchmark: str,
    function_results,
    bitcode_path: Path,
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-tu-mcts-") as tmp_str:
        workdir = Path(tmp_str)
        measure_instructions = args.measure_instructions

        crashing: set[str] = set()
        if getattr(args, "prevalidate_passes", True):
            crashing = find_crashing_passes(bitcode_path, function_results, workdir)
            if crashing:
                function_results = filter_passes_from_results(function_results, crashing)

        graph = build_pass_order_graph(function_results, benchmark=benchmark, weight_mode="delta")
        actions = sorted(graph.nodes)

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
        counter = {"evals": 0}
        config = MCTSConfig(
            max_length=args.max_length,
            c_puct=args.c_puct,
            rollout_length=args.rollout_length,
            branching=args.branching,
            prior_floor=args.prior_floor,
            seed=args.seed,
        )
        result = run_mcts(
            baseline_size=baseline_size,
            prior_fn=_make_prior_fn(graph, actions),
            measure=_make_measure_fn(
                bitcode_path, workdir, prefix_cache, args.max_real_evals_per_benchmark, counter
            ),
            config=config,
        )

        best_passes = result.best_passes
        best_size = result.best_size
        # The final size is the size after applying the whole best sequence; for
        # MCTS the reported best is itself a measured prefix, so final == best.
        final_size = best_size
        diagnostics = {
            "real_evals": counter["evals"],
            "nodes_expanded": result.nodes_expanded,
            "iterations": result.iterations,
            "max_depth": result.max_depth,
            "budget_exhausted": result.budget_exhausted,
            "action_count": len(actions),
            "crashing_passes": sorted(crashing),
            "budget": args.max_real_evals_per_benchmark,
        }

        row: dict[str, Any] = {
            "benchmark": benchmark,
            "heuristic": HEURISTIC,
            "candidate_index": 0,
            "bitcode_path": str(bitcode_path),
            "baseline_size": baseline_size,
            "oz_size": oz_size,
            "oz_delta": (baseline_size - oz_size) if oz_size is not None else None,
            "final_size": final_size,
            "final_delta": baseline_size - final_size,
            "best_size": best_size,
            "best_delta": baseline_size - best_size,
            "best_prefix_len": result.best_prefix_len,
            "passes": list(best_passes),
            "best_passes": list(best_passes),
            "error": "",
            "error_kind": "",
            "baseline_instruction_count": baseline_instruction_count,
            "oz_instruction_count": oz_instruction_count,
            "oz_instruction_delta": (
                baseline_instruction_count - oz_instruction_count
                if baseline_instruction_count is not None and oz_instruction_count is not None
                else None
            ),
            "final_instruction_count": None,
            "final_instruction_delta": None,
            "best_instruction_count": None,
            "best_instruction_delta": None,
            "best_instruction_prefix_len": 0,
            "best_instruction_passes": [],
            **diagnostics,
        }

        if measure_instructions and baseline_instruction_count is not None and best_passes:
            instr = _best_prefix_instructions(
                bitcode_path, workdir, best_passes, baseline_instruction_count
            )
            row.update(instr)
            row["final_instruction_delta"] = baseline_instruction_count - instr["final_instruction_count"]
            row["best_instruction_delta"] = baseline_instruction_count - instr["best_instruction_count"]
        elif measure_instructions and baseline_instruction_count is not None:
            row["final_instruction_count"] = baseline_instruction_count
            row["final_instruction_delta"] = 0
            row["best_instruction_count"] = baseline_instruction_count
            row["best_instruction_delta"] = 0

        return row, [row], len(prefix_cache)


def make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts):
    summary = summarize_evaluations(selected_rows)
    add_instruction_summary(summary, selected_rows)
    add_failure_summary(summary, selected_rows)
    if selected_rows and HEURISTIC in summary:
        summary[HEURISTIC]["mean_real_evals"] = sum(
            int(row.get("real_evals") or 0) for row in selected_rows
        ) / len(selected_rows)
        summary[HEURISTIC]["total_real_evals"] = sum(
            int(row.get("real_evals") or 0) for row in selected_rows
        )
        summary[HEURISTIC]["mean_nodes_expanded"] = sum(
            int(row.get("nodes_expanded") or 0) for row in selected_rows
        ) / len(selected_rows)
    return {
        "comparison_input": str(args.comparison),
        "bitcode_dir": str(args.bitcode_dir),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "summary": summary,
        "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows),
        "candidate_rows": list(candidate_rows),
    }


def build_report(groups, bitcode_paths, args) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prefix_cache_counts: dict[str, int] = {}
    benchmarks = sorted(groups)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    for index, benchmark in enumerate(benchmarks, start=1):
        print(
            f"[{index}/{len(benchmarks)}] {benchmark}: tu_mcts budget={args.max_real_evals_per_benchmark}",
            flush=True,
        )
        selected, candidates, cache_count = evaluate_tu_mcts_for_benchmark(
            benchmark, groups[benchmark], bitcode_paths[benchmark], args=args
        )
        selected_rows.append(selected)
        candidate_rows.extend(candidates)
        prefix_cache_counts[benchmark] = cache_count
        write_report(
            make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts),
            args.output,
        )
    return make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TU-MCTS on whole translation units.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--algorithm", default="", help="Pass-search prefix to read from comparison.json.")
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-real-evals-per-benchmark", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--rollout-length", type=int, default=3)
    parser.add_argument("--branching", type=int, default=8)
    parser.add_argument("--prior-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--prevalidate-passes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop passes that crash opt on the benchmark before searching.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--measure-instructions", action="store_true")
    parser.add_argument("--site-data", action="append", default=[], type=Path)
    parser.add_argument("--overwrite-bitcode", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_real_evals_per_benchmark < 0:
        raise ValueError("--max-real-evals-per-benchmark must be non-negative")
    if args.branching <= 0:
        raise ValueError("--branching must be positive")
    if args.rollout_length < 0:
        raise ValueError("--rollout-length must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
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
    bitcode_paths = write_translation_unit_bitcodes(
        benchmarks, args.bitcode_dir, site_data_paths=args.site_data
    )
    report = build_report({b: groups[b] for b in benchmarks}, bitcode_paths, args)
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
