"""Two-wave Chunk-Forest evaluation on whole translation units."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.translation_unit.chunk_forest import (
    CandidatePath,
    Chunk,
    ChunkRef,
    ChunkInventoryConfig,
    ChunkSelectionConfig,
    ChunkWalkConfig,
    build_chunk_graph,
    chunk_kind_counts,
    generate_candidate_pool,
    mine_chunks,
    select_paths,
)
from llvm_ir.stages.function_search.pass_search import require_tools
from llvm_ir.stages.translation_unit.evaluate import (
    summarize_evaluations,
    write_translation_unit_bitcodes,
)
from llvm_ir.stages.translation_unit.evaluate_topk_paths import (
    _make_candidate_row,
    _selected_key,
    add_failure_summary,
    add_instruction_summary,
    apply_instruction_result_to_row,
    evaluate_candidate_with_prefix_cache,
    measure_deferred_candidate_instructions,
    measure_text_and_instruction_count,
    measure_text_size,
    optimize_oz,
    write_report,
)
from llvm_ir.stages.translation_unit.graph.order_graph import (
    PassOrderGraph,
    benchmark_id_from_function_name,
    build_pass_order_graph,
    load_function_pass_results_from_report,
)
from llvm_ir.stages.translation_unit.contracts import FunctionPassResult

HEURISTIC = "chunk_forest"


def group_results_by_benchmark(
    results: list[FunctionPassResult],
) -> dict[str, list[FunctionPassResult]]:
    grouped: dict[str, list[FunctionPassResult]] = defaultdict(list)
    for result in results:
        grouped[benchmark_id_from_function_name(result.function)].append(result)
    return dict(sorted(grouped.items()))


def _single_chunk_pool(chunks: tuple[Chunk, ...], max_length: int) -> list[CandidatePath]:
    pool = []
    for index, chunk in enumerate(chunks):
        passes = chunk.passes[:max_length]
        if not passes:
            continue
        partial = len(passes) < len(chunk.passes)
        pool.append(CandidatePath(passes, (ChunkRef(index, 0, len(passes), partial),)))
    return pool


def _candidate_key(candidate: CandidatePath) -> tuple[str, ...]:
    return candidate.passes


def _evaluate_wave(
    *,
    graph: PassOrderGraph,
    bitcode_path: Path,
    workdir: Path,
    prefix_cache: dict[tuple[str, ...], dict[str, Any]],
    baseline_size: int,
    oz_size: int | None,
    baseline_instruction_count: int | None,
    oz_instruction_count: int | None,
    candidates: list[CandidatePath],
    wave: int,
    candidate_offset: int,
    measure_instructions: bool,
    deferred_instructions: bool,
    extra_common: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    start_count = len(prefix_cache)
    for local_index, candidate in enumerate(candidates, start=1):
        passes = list(candidate.passes)
        result = evaluate_candidate_with_prefix_cache(
            passes,
            workdir,
            prefix_cache,
            measure_instructions=measure_instructions and not deferred_instructions,
        )
        row = _make_candidate_row(
            graph=graph,
            heuristic=HEURISTIC,
            candidate_index=candidate_offset + local_index,
            passes=passes,
            result=result,
            bitcode_path=bitcode_path,
            baseline_size=baseline_size,
            oz_size=oz_size,
            baseline_instruction_count=baseline_instruction_count,
            oz_instruction_count=oz_instruction_count,
            top_k=extra_common["paths"],
            extra={
                **extra_common,
                "wave": wave,
                "wave_candidate_index": local_index,
                "chunk_forest_score": candidate.score,
                "chunk_forest_new_nodes": candidate.new_nodes,
                "chunk_indexes": [ref.chunk_index for ref in candidate.chunks],
            },
        )
        rows.append(row)
    return rows, len(prefix_cache) - start_count


def assign_chunk_credit(
    evaluated: list[CandidatePath],
    prefix_cache: dict[tuple[str, ...], dict[str, Any]],
) -> dict[int, list[int]]:
    observations: dict[int, list[int]] = defaultdict(list)
    for candidate in evaluated:
        failed = False
        for ref in candidate.chunks:
            if ref.partial or failed:
                continue
            before_key = candidate.passes[: ref.start]
            after_key = candidate.passes[: ref.end]
            before = prefix_cache.get(before_key)
            after = prefix_cache.get(after_key)
            if before is None or after is None:
                continue
            if before.get("error"):
                failed = True
                continue
            if after.get("error"):
                failed = True
                continue
            observations[ref.chunk_index].append(int(before["size"]) - int(after["size"]))
    return observations


def rescore_chunk_values(
    chunks: tuple[Chunk, ...],
    observations: dict[int, list[int]],
) -> tuple[dict[int, float], int, float]:
    measured: dict[int, float] = {
        index: sum(values) / len(values)
        for index, values in observations.items()
        if len(values) >= 2
    }
    if measured:
        positive_mean = sum(max(value, 0.0) for value in measured.values()) / len(measured)
        mined_mean = sum(chunks[index].weight for index in measured) / len(measured)
        scale = positive_mean / mined_mean if mined_mean > 0 else 0.0
    else:
        scale = 0.0
    values = {
        index: measured.get(index, chunk.weight * scale)
        for index, chunk in enumerate(chunks)
    }
    return values, len(measured), scale


def evaluate_chunk_forest_for_benchmark(
    benchmark: str,
    function_results: list[FunctionPassResult],
    bitcode_path: Path,
    *,
    args: argparse.Namespace,
    measure_instructions: bool = False,
    instruction_measurement: str = "deferred",
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    graph = build_pass_order_graph(function_results, benchmark=benchmark, weight_mode="delta")
    with tempfile.TemporaryDirectory(prefix="llvm-ir-tu-chunk-forest-") as tmp_str:
        workdir = Path(tmp_str)
        deferred_instructions = measure_instructions and instruction_measurement == "deferred"
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

        inventory_config = ChunkInventoryConfig(
            ngram_max=args.ngram_max,
            closure_theta=args.closure_theta,
            min_support=args.min_support,
            top_chunks=args.top_chunks,
            macro_top=args.macro_top,
        )
        chunks = tuple(mine_chunks(function_results, graph, config=inventory_config))
        chunk_graph = build_chunk_graph(list(chunks), function_results)
        pool = generate_candidate_pool(
            chunk_graph,
            config=ChunkWalkConfig(
                pool_size=args.pool_size,
                walk_seed=args.walk_seed,
                max_length=args.max_length,
            ),
        )
        if not pool and chunks:
            pool = _single_chunk_pool(chunks, args.max_length)
        pool_by_passes = {_candidate_key(candidate): candidate for candidate in pool}
        pool = sorted(pool_by_passes.values(), key=lambda candidate: candidate.passes)

        wave_count = max(1, args.waves)
        first_wave_target = args.paths if wave_count == 1 else max(1, args.paths // wave_count)
        second_wave_target = max(0, args.paths - first_wave_target)
        wave_eval_limit = args.max_real_evals_per_benchmark // wave_count if args.max_real_evals_per_benchmark else 0
        common = {
            **chunk_kind_counts(chunks),
            "paths": args.paths,
            "pool_unique_paths": len(pool),
            "wave1_real_evals": 0,
            "wave2_real_evals": 0,
            "chunks_remeasured": 0,
            "rescoring_scale": 0.0,
            "top_chunks": [],
        }
        mined_values = {index: chunk.weight for index, chunk in enumerate(chunks)}
        wave1, trie, picked = select_paths(
            pool,
            chunks,
            mined_values,
            config=ChunkSelectionConfig(
                paths=first_wave_target,
                lambda_cache=args.lambda_cache,
                gamma_diversity=args.gamma_diversity,
                max_real_evals=wave_eval_limit,
            ),
        )
        rows1, wave1_cost = _evaluate_wave(
            graph=graph,
            bitcode_path=bitcode_path,
            workdir=workdir,
            prefix_cache=prefix_cache,
            baseline_size=baseline_size,
            oz_size=oz_size,
            baseline_instruction_count=baseline_instruction_count,
            oz_instruction_count=oz_instruction_count,
            candidates=wave1,
            wave=1,
            candidate_offset=0,
            measure_instructions=measure_instructions,
            deferred_instructions=deferred_instructions,
            extra_common=common,
        )
        common["wave1_real_evals"] = wave1_cost

        rows2: list[dict[str, Any]] = []
        wave2: list[CandidatePath] = []
        wave2_cost = 0
        observations = assign_chunk_credit(wave1, prefix_cache)
        rescored_values, chunks_remeasured, scale = rescore_chunk_values(chunks, observations)
        common["chunks_remeasured"] = chunks_remeasured
        common["rescoring_scale"] = scale
        common["top_chunks"] = [
            {"passes": list(chunks[index].passes), "value": rescored_values[index]}
            for index in sorted(rescored_values, key=lambda item: (-rescored_values[item], chunks[item].passes))[:5]
        ]
        if wave_count > 1 and second_wave_target:
            wave2_limit = 0
            if args.max_real_evals_per_benchmark:
                wave2_limit = max(0, args.max_real_evals_per_benchmark - (len(trie) - 1))
            wave2, _trie2, _picked2 = select_paths(
                pool,
                chunks,
                rescored_values,
                config=ChunkSelectionConfig(
                    paths=second_wave_target,
                    lambda_cache=args.lambda_cache,
                    gamma_diversity=args.gamma_diversity,
                    max_real_evals=wave2_limit,
                ),
                initial_trie=trie,
                initial_picked=picked,
                excluded_paths={candidate.passes for candidate in wave1},
            )
            rows2, wave2_cost = _evaluate_wave(
                graph=graph,
                bitcode_path=bitcode_path,
                workdir=workdir,
                prefix_cache=prefix_cache,
                baseline_size=baseline_size,
                oz_size=oz_size,
                baseline_instruction_count=baseline_instruction_count,
                oz_instruction_count=oz_instruction_count,
                candidates=wave2,
                wave=2,
                candidate_offset=len(rows1),
                measure_instructions=measure_instructions,
                deferred_instructions=deferred_instructions,
                extra_common=common,
            )
            common["wave2_real_evals"] = wave2_cost

        candidate_rows = rows1 + rows2
        for row in candidate_rows:
            row.update(common)
        best_row = max(candidate_rows, key=_selected_key) if candidate_rows else None
        instruction_eval_cost = 0
        if deferred_instructions and best_row is not None and baseline_instruction_count is not None:
            instruction_result, instruction_eval_cost = measure_deferred_candidate_instructions(
                list(best_row["passes"]),
                workdir,
                prefix_cache,
            )
            apply_instruction_result_to_row(best_row, instruction_result, baseline_instruction_count)
        if best_row is None:
            best_row = _empty_row(
                graph,
                bitcode_path,
                baseline_size,
                oz_size,
                baseline_instruction_count,
                oz_instruction_count,
                common,
            )
        best_row.update(common)
        best_row["prefix_failures"] = sum(1 for entry in prefix_cache.values() if entry.get("error"))
        best_row["instruction_measurement"] = instruction_measurement
        best_row["instruction_eval_cost"] = instruction_eval_cost
        best_row["real_evals"] = len(prefix_cache) - 1
        return dict(best_row), candidate_rows, len(prefix_cache)


def _empty_row(
    graph: PassOrderGraph,
    bitcode_path: Path,
    baseline_size: int,
    oz_size: int | None,
    baseline_instruction_count: int | None,
    oz_instruction_count: int | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": graph.benchmark,
        "heuristic": HEURISTIC,
        "candidate_index": 0,
        "start_pass": "",
        "start_weight": 0,
        "top_k": extra["paths"],
        "bitcode_path": str(bitcode_path),
        "baseline_size": baseline_size,
        "oz_size": oz_size,
        "oz_delta": baseline_size - oz_size if oz_size is not None else None,
        "baseline_instruction_count": baseline_instruction_count,
        "oz_instruction_count": oz_instruction_count,
        "oz_instruction_delta": (
            baseline_instruction_count - oz_instruction_count
            if baseline_instruction_count is not None and oz_instruction_count is not None
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
        "error": "no chunk-forest candidates",
        "error_kind": "full_failure",
        **extra,
    }



def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value

def make_report_payload(
    args: argparse.Namespace,
    selected_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    prefix_cache_counts: dict[str, int],
) -> dict[str, Any]:
    summary = summarize_evaluations(selected_rows)
    add_instruction_summary(summary, selected_rows)
    add_failure_summary(summary, selected_rows)
    if selected_rows and HEURISTIC in summary:
        summary[HEURISTIC]["mean_real_evals"] = sum(int(row.get("real_evals") or 0) for row in selected_rows) / len(selected_rows)
        summary[HEURISTIC]["total_real_evals"] = sum(int(row.get("real_evals") or 0) for row in selected_rows)
    return {
        "comparison_input": str(args.comparison),
        "bitcode_dir": str(args.bitcode_dir),
        "config": {key: _jsonable(value) for key, value in vars(args).items()},
        "summary": summary,
        "prefix_cache_counts": dict(prefix_cache_counts),
        "selected_rows": list(selected_rows),
        "candidate_rows": list(candidate_rows),
    }


def build_report(
    groups: dict[str, list[FunctionPassResult]],
    bitcode_paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prefix_cache_counts: dict[str, int] = {}
    benchmarks = sorted(groups)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    for index, benchmark in enumerate(benchmarks, start=1):
        print(f"[{index}/{len(benchmarks)}] {benchmark}: chunk_forest paths={args.paths} waves={args.waves}", flush=True)
        selected, candidates, cache_count = evaluate_chunk_forest_for_benchmark(
            benchmark,
            groups[benchmark],
            bitcode_paths[benchmark],
            args=args,
            measure_instructions=args.measure_instructions,
            instruction_measurement=args.instruction_measurement,
        )
        selected_rows.append(selected)
        candidate_rows.extend(candidates)
        prefix_cache_counts[benchmark] = cache_count
        write_report(make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts), args.output)
    return make_report_payload(args, selected_rows, candidate_rows, prefix_cache_counts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chunk-Forest on whole translation units.")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--algorithm", default="", help="Pass-search prefix to read from comparison.json.")
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paths", "--top-k", dest="paths", type=int, default=500)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=100_000)
    parser.add_argument("--walk-seed", type=int, default=7)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--closure-theta", type=float, default=0.8)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--top-chunks", type=int, default=30)
    parser.add_argument("--macro-top", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--lambda-cache", type=float, default=0.0)
    parser.add_argument("--gamma-diversity", type=float, default=0.5)
    parser.add_argument("--max-real-evals-per-benchmark", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--measure-instructions", action="store_true")
    parser.add_argument("--instruction-measurement", choices=["deferred", "eager"], default="deferred")
    parser.add_argument("--site-data", action="append", default=[], type=Path)
    parser.add_argument("--overwrite-bitcode", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.waves <= 0:
        raise ValueError("--waves must be positive")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.ngram_max < 2:
        raise ValueError("--ngram-max must be >= 2")
    if not 0.0 <= args.closure_theta <= 1.0:
        raise ValueError("--closure-theta must be between 0 and 1")
    if args.min_support <= 0:
        raise ValueError("--min-support must be positive")
    if args.top_chunks < 0 or args.macro_top < 0:
        raise ValueError("--top-chunks and --macro-top must be non-negative")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_real_evals_per_benchmark < 0:
        raise ValueError("--max-real-evals-per-benchmark must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    require_tools("opt", "llc", "llvm-size")
    if args.measure_instructions:
        require_tools("llvm-objdump")
    if args.overwrite_bitcode and args.bitcode_dir.exists():
        shutil.rmtree(args.bitcode_dir)
    results = load_function_pass_results_from_report(
        args.comparison,
        algorithm=args.algorithm or None,
    )
    groups = group_results_by_benchmark(results)
    benchmarks = sorted(groups)
    if args.limit:
        benchmarks = benchmarks[: args.limit]
    bitcode_paths = write_translation_unit_bitcodes(
        benchmarks,
        args.bitcode_dir,
        site_data_paths=args.site_data,
    )
    report = build_report({benchmark: groups[benchmark] for benchmark in benchmarks}, bitcode_paths, args)
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
