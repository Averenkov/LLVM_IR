"""Evaluate top-k translation-unit paths generated directly from pass-order graphs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.translation_unit.bucket_dag import (
    BucketDagConfig,
    bucket_layer_top_paths,
)
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


@dataclass(frozen=True)
class SuperSegmentCandidate:
    index: int
    passes: tuple[str, ...]
    vertex_delta: int


@dataclass(frozen=True)
class SuperPathCandidate:
    segment_indexes: tuple[int, ...]
    passes: tuple[str, ...]
    score: int
    vertex_delta: int
    edge_score: int


@dataclass(frozen=True)
class SuperPathBuildResult:
    candidates: list[SuperPathCandidate]
    truncated: bool
    generated_count: int


@dataclass(frozen=True)
class SuperSegmentGenerationResult:
    paths: list[list[str]]
    segment_length_floor: int
    tiny_graph_mode: bool


@dataclass(frozen=True)
class MeasuredSuperSegment:
    passes: tuple[str, ...]
    vertex_delta: int
    recovered_from_tail: bool
    truncated_to_best_prefix: bool


def measure_machine_instruction_count(bitcode_path: Path, workdir: Path) -> int:
    obj_path = workdir / f"{bitcode_path.stem}.instr.o"
    try:
        run_cmd(["llc", "-filetype=obj", str(bitcode_path), "-o", str(obj_path)])
        out = run_cmd(["llvm-objdump", "-d", str(obj_path)]).stdout.splitlines()
        return sum(1 for line in out if INSTRUCTION_LINE_RE.match(line))
    finally:
        obj_path.unlink(missing_ok=True)


def measure_text_and_instruction_count(
    bitcode_path: Path,
    workdir: Path,
) -> tuple[int, int]:
    obj_path = workdir / f"{bitcode_path.stem}.metrics.o"
    try:
        run_cmd(["llc", "-filetype=obj", str(bitcode_path), "-o", str(obj_path)])
        size_output = run_cmd(["llvm-size", str(obj_path)]).stdout.strip().splitlines()
        if len(size_output) < 2:
            raise RuntimeError(f"unexpected llvm-size output: {size_output!r}")
        text_size = int(size_output[1].split()[0])
        disassembly = run_cmd(["llvm-objdump", "-d", str(obj_path)]).stdout.splitlines()
        instruction_count = sum(
            1 for line in disassembly if INSTRUCTION_LINE_RE.match(line)
        )
        return text_size, instruction_count
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
    if heuristic == "cycle_breaking_superpath_topk":
        raise ValueError("use evaluate_superpath_for_benchmark for superpath heuristic")
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
    if heuristic in {"bucket_dag_topk", "bucket_dag_top250"}:
        return bucket_layer_top_paths(
            graph,
            config=BucketDagConfig(
                chunk_size=args.chunk_size,
                max_length=args.max_length,
                min_edge_weight=args.min_edge_weight,
            ),
            top_k=args.top_k,
        )
    raise ValueError(f"Unknown top-k translation-unit heuristic: {heuristic}")


def _rank_path_for_segments(graph: PassOrderGraph, passes: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
    return (
        score_path(graph, list(passes)).net_score,
        len(passes),
        tuple(reversed(passes)),
    )


def _generate_superpath_segments(
    graph: PassOrderGraph,
    args: argparse.Namespace,
) -> list[list[str]]:
    return _generate_superpath_segments_with_stats(graph, args).paths


def _generate_superpath_segments_with_stats(
    graph: PassOrderGraph,
    args: argparse.Namespace,
) -> SuperSegmentGenerationResult:
    segment_min_length = int(getattr(args, "segment_min_length", 4))
    segment_max_length = max(segment_min_length, int(getattr(args, "segment_max_length", 6)))
    segment_top_k = max(1, int(getattr(args, "segment_top_k", 100)))
    segment_max_jaccard = float(getattr(args, "segment_max_jaccard", 0.75))
    tiny_graph_threshold = int(getattr(args, "tiny_graph_threshold", 4))
    tiny_graph_mode = len(graph.nodes) <= tiny_graph_threshold

    if tiny_graph_mode:
        raw_paths = _enumerate_tiny_graph_paths(
            graph,
            max_length=min(len(graph.nodes), segment_max_length),
            min_edge_weight=args.min_edge_weight,
        )
        floors = [1]
    else:
        raw_paths = cycle_breaking_top_start_paths(
            graph,
            config=CycleBreakingMaxPathConfig(
                max_length=segment_max_length,
                min_edge_weight=args.min_edge_weight,
            ),
            top_starts=args.top_starts,
            paths_per_start=args.paths_per_start,
        )
        raw_paths.extend([[node] for node in sorted(graph.nodes)])
        floors = [segment_min_length]
        if segment_min_length > 2:
            floors.append(2)
        floors.append(1)

    selected_floor = floors[-1]
    ranked: list[tuple[str, ...]] = []
    for floor in floors:
        unique = {
            tuple(path)
            for path in raw_paths
            if floor <= len(path) <= segment_max_length
        }
        ranked = sorted(
            unique,
            key=lambda path: _rank_path_for_segments(graph, path),
            reverse=True,
        )
        selected_floor = floor
        if ranked:
            break

    if tiny_graph_mode:
        selected = ranked
    else:
        selected: list[tuple[str, ...]] = []
        for path in ranked:
            if all(_jaccard_similarity(path, existing) <= segment_max_jaccard for existing in selected):
                selected.append(path)
                if len(selected) >= segment_top_k:
                    break
    return SuperSegmentGenerationResult(
        paths=[list(path) for path in selected],
        segment_length_floor=selected_floor,
        tiny_graph_mode=tiny_graph_mode,
    )


def _enumerate_tiny_graph_paths(
    graph: PassOrderGraph,
    *,
    max_length: int,
    min_edge_weight: int,
) -> list[list[str]]:
    if max_length <= 0:
        return []
    adjacency: dict[str, list[str]] = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.weight >= min_edge_weight:
            adjacency.setdefault(edge.source, []).append(edge.target)
    for targets in adjacency.values():
        targets.sort()

    paths: set[tuple[str, ...]] = set()

    def visit(path: tuple[str, ...]) -> None:
        paths.add(path)
        if len(path) >= max_length:
            return
        for next_node in adjacency.get(path[-1], []):
            if next_node not in path:
                visit(path + (next_node,))

    for node in sorted(graph.nodes):
        visit((node,))
    return [list(path) for path in sorted(paths)]

def _jaccard_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _superpath_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum((left_counts & right_counts).values())


def _superpath_rank_key(candidate: SuperPathCandidate) -> tuple[int, int, int, int, tuple[str, ...]]:
    return (
        candidate.score,
        candidate.vertex_delta,
        candidate.edge_score,
        -len(candidate.passes),
        tuple(reversed(candidate.passes)),
    )


def _measured_super_segment_from_result(
    raw_path: list[str],
    result: dict[str, Any],
    baseline_size: int,
) -> MeasuredSuperSegment | None:
    if result.get("error") and result.get("error_kind") != "tail_failure":
        return None
    recovered_from_tail = bool(result.get("error"))
    best_prefix_len = int(result.get("best_prefix_len") or 0)
    if best_prefix_len >= 1:
        passes = tuple(result.get("best_passes") or raw_path[:best_prefix_len])
        size = int(result["best_size"])
    else:
        passes = tuple(raw_path)
        size = int(result["final_size"])
    if not passes:
        return None
    return MeasuredSuperSegment(
        passes=passes,
        vertex_delta=baseline_size - size,
        recovered_from_tail=recovered_from_tail,
        truncated_to_best_prefix=best_prefix_len >= 1 and best_prefix_len < len(raw_path),
    )


def _build_superpath_candidates(
    graph: PassOrderGraph,
    segments: list[SuperSegmentCandidate],
    *,
    top_k: int,
    max_pass_length: int,
    min_edge_weight: int,
    beam_factor: int = 5,
    max_candidates: int = 100_000,
    max_overlap: int = 1,
) -> list[SuperPathCandidate]:
    return _build_superpath_candidates_with_stats(
        graph,
        segments,
        top_k=top_k,
        max_pass_length=max_pass_length,
        min_edge_weight=min_edge_weight,
        beam_factor=beam_factor,
        max_candidates=max_candidates,
        max_overlap=max_overlap,
    ).candidates


def _build_superpath_candidates_with_stats(
    graph: PassOrderGraph,
    segments: list[SuperSegmentCandidate],
    *,
    top_k: int,
    max_pass_length: int,
    min_edge_weight: int,
    beam_factor: int,
    max_candidates: int,
    max_overlap: int = 1,
) -> SuperPathBuildResult:
    if not segments:
        return SuperPathBuildResult(candidates=[], truncated=False, generated_count=0)
    top_k = max(1, top_k)
    beam_width = max(1, beam_factor) * top_k
    max_candidates = max(1, max_candidates)
    if max_pass_length <= 0:
        max_pass_length = max(len(segment.passes) for segment in segments)
    min_segment_length = max(1, min(len(segment.passes) for segment in segments))
    max_super_segments = max(1, max_pass_length // min_segment_length)

    outgoing: dict[int, list[tuple[int, int]]] = {segment.index: [] for segment in segments}
    by_index = {segment.index: segment for segment in segments}
    for left in segments:
        left_last = left.passes[-1]
        for right in segments:
            if left.index == right.index:
                continue
            edge_weight = graph.edge_counts.get((left_last, right.passes[0]), 0)
            if edge_weight >= min_edge_weight:
                outgoing[left.index].append((right.index, edge_weight))
        outgoing[left.index].sort(key=lambda item: (-item[1], item[0]))

    candidates: list[SuperPathCandidate] = []
    truncated = False

    def add_candidate(candidate: SuperPathCandidate) -> bool:
        nonlocal truncated
        if len(candidates) >= max_candidates:
            truncated = True
            return False
        candidates.append(candidate)
        if len(candidates) >= max_candidates:
            truncated = True
        return True

    layer: list[SuperPathCandidate] = []
    for segment in segments:
        if len(segment.passes) > max_pass_length:
            continue
        layer.append(
            SuperPathCandidate(
                segment_indexes=(segment.index,),
                passes=segment.passes,
                score=segment.vertex_delta,
                vertex_delta=segment.vertex_delta,
                edge_score=0,
            )
        )
    layer = sorted(layer, key=_superpath_rank_key, reverse=True)[:beam_width]
    for candidate in layer:
        if not add_candidate(candidate):
            break

    while layer and not truncated:
        next_layer: list[SuperPathCandidate] = []
        for current in layer:
            if len(current.segment_indexes) >= max_super_segments:
                continue
            used = set(current.segment_indexes)
            last_index = current.segment_indexes[-1]
            for next_index, edge_weight in outgoing.get(last_index, []):
                if next_index in used:
                    continue
                next_segment = by_index[next_index]
                if _superpath_overlap(current.passes, next_segment.passes) > max_overlap:
                    continue
                next_passes = current.passes + next_segment.passes
                if len(next_passes) > max_pass_length:
                    continue
                next_layer.append(
                    SuperPathCandidate(
                        segment_indexes=current.segment_indexes + (next_index,),
                        passes=next_passes,
                        score=current.score + next_segment.vertex_delta,
                        vertex_delta=current.vertex_delta + next_segment.vertex_delta,
                        edge_score=current.edge_score + edge_weight,
                    )
                )
        if not next_layer:
            break
        layer = sorted(next_layer, key=_superpath_rank_key, reverse=True)[:beam_width]
        for candidate in layer:
            if not add_candidate(candidate):
                break

    unique: dict[tuple[str, ...], SuperPathCandidate] = {}
    for candidate in candidates:
        current = unique.get(candidate.passes)
        if current is None or _superpath_rank_key(candidate) > _superpath_rank_key(current):
            unique[candidate.passes] = candidate
    return SuperPathBuildResult(
        candidates=sorted(unique.values(), key=_superpath_rank_key, reverse=True)[:top_k],
        truncated=truncated,
        generated_count=len(candidates),
    )

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
    baseline_instruction_count = (
        baseline.get("instruction_count") if measure_instructions else None
    )
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
                if measure_instructions:
                    size, instruction_count = measure_text_and_instruction_count(
                        output_bc,
                        workdir,
                    )
                else:
                    size = measure_text_size(output_bc, workdir)
                    instruction_count = None
                entry = {
                    "bc": str(output_bc),
                    "size": size,
                    "error": "",
                }
                if instruction_count is not None:
                    entry["instruction_count"] = instruction_count
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


def measure_deferred_candidate_instructions(
    passes: list[str],
    workdir: Path,
    prefix_cache: dict[tuple[str, ...], dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    baseline_instruction_count = prefix_cache[()].get("instruction_count")
    if baseline_instruction_count is None:
        raise ValueError("baseline instruction count is required for deferred measurement")

    best_instruction_count = int(baseline_instruction_count)
    best_instruction_prefix_len = 0
    best_instruction_passes: list[str] = []
    final_instruction_count = int(baseline_instruction_count)
    current_key: tuple[str, ...] = ()
    instruction_eval_cost = 0

    for index, pass_name in enumerate(passes, start=1):
        next_key = current_key + (pass_name,)
        entry = prefix_cache.get(next_key)
        if entry is None or entry.get("error"):
            break
        if entry.get("instruction_count") is None:
            entry["instruction_count"] = measure_machine_instruction_count(
                Path(entry["bc"]),
                workdir,
            )
            instruction_eval_cost += 1
        current_key = next_key
        final_instruction_count = int(entry["instruction_count"])
        if final_instruction_count < best_instruction_count:
            best_instruction_count = final_instruction_count
            best_instruction_prefix_len = index
            best_instruction_passes = passes[:index]

    return (
        {
            "final_instruction_count": final_instruction_count,
            "best_instruction_count": best_instruction_count,
            "best_instruction_prefix_len": best_instruction_prefix_len,
            "best_instruction_passes": best_instruction_passes,
        },
        instruction_eval_cost,
    )


def apply_instruction_result_to_row(
    row: dict[str, Any],
    instruction_result: dict[str, Any],
    baseline_instruction_count: int,
) -> None:
    final_instruction_count = int(instruction_result["final_instruction_count"])
    best_instruction_count = int(instruction_result["best_instruction_count"])
    row.update(
        {
            "final_instruction_count": final_instruction_count,
            "final_instruction_delta": baseline_instruction_count - final_instruction_count,
            "best_instruction_count": best_instruction_count,
            "best_instruction_delta": baseline_instruction_count - best_instruction_count,
            "best_instruction_prefix_len": instruction_result[
                "best_instruction_prefix_len"
            ],
            "best_instruction_passes": instruction_result["best_instruction_passes"],
        }
    )


def evaluate_topk_for_benchmark(
    graph: PassOrderGraph,
    bitcode_path: Path,
    paths: list[list[str]],
    *,
    heuristic: str,
    measure_instructions: bool = False,
    instruction_measurement: str = "deferred",
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-tu-topk-eval-") as tmp_str:
        workdir = Path(tmp_str)
        deferred_instructions = (
            measure_instructions and instruction_measurement == "deferred"
        )
        if measure_instructions:
            baseline_size, baseline_instruction_count = measure_text_and_instruction_count(
                bitcode_path,
                workdir,
            )
        else:
            baseline_size = measure_text_size(bitcode_path, workdir)
            baseline_instruction_count = None
        oz_size = None
        oz_instruction_count = None
        try:
            oz_bc = workdir / "oz.bc"
            optimize_oz(bitcode_path, oz_bc)
            if measure_instructions:
                oz_size, oz_instruction_count = measure_text_and_instruction_count(
                    oz_bc,
                    workdir,
                )
            else:
                oz_size = measure_text_size(oz_bc, workdir)
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
                measure_instructions=measure_instructions and not deferred_instructions,
            )
            row = _make_candidate_row(
                graph=graph,
                heuristic=heuristic,
                candidate_index=candidate_index,
                passes=passes,
                result=result,
                bitcode_path=bitcode_path,
                baseline_size=baseline_size,
                oz_size=oz_size,
                baseline_instruction_count=baseline_instruction_count,
                oz_instruction_count=oz_instruction_count,
                top_k=len(paths),
            )
            candidate_rows.append(row)
            if best_row is None or _selected_key(row) > _selected_key(best_row):
                best_row = row

        instruction_eval_cost = 0
        if (
            deferred_instructions
            and best_row is not None
            and baseline_instruction_count is not None
        ):
            instruction_result, instruction_eval_cost = (
                measure_deferred_candidate_instructions(
                    list(best_row["passes"]),
                    workdir,
                    prefix_cache,
                )
            )
            apply_instruction_result_to_row(
                best_row,
                instruction_result,
                baseline_instruction_count,
            )

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
        best_row["instruction_measurement"] = instruction_measurement
        best_row["instruction_eval_cost"] = instruction_eval_cost
        return dict(best_row), candidate_rows, len(prefix_cache)


def _make_candidate_row(
    *,
    graph: PassOrderGraph,
    heuristic: str,
    candidate_index: int,
    passes: list[str],
    result: dict[str, Any],
    bitcode_path: Path,
    baseline_size: int,
    oz_size: int | None,
    baseline_instruction_count: int | None,
    oz_instruction_count: int | None,
    top_k: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "benchmark": graph.benchmark,
        "heuristic": heuristic,
        "candidate_index": candidate_index,
        "start_pass": passes[0] if passes else "",
        "start_weight": graph.start_counts.get(passes[0], 0) if passes else 0,
        "top_k": top_k,
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
    if extra:
        row.update(extra)
    return row


def evaluate_superpath_for_benchmark(
    graph: PassOrderGraph,
    bitcode_path: Path,
    *,
    args: argparse.Namespace,
    measure_instructions: bool = False,
    instruction_measurement: str = "deferred",
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-tu-superpath-eval-") as tmp_str:
        workdir = Path(tmp_str)
        deferred_instructions = (
            measure_instructions and instruction_measurement == "deferred"
        )
        if measure_instructions:
            baseline_size, baseline_instruction_count = measure_text_and_instruction_count(
                bitcode_path,
                workdir,
            )
        else:
            baseline_size = measure_text_size(bitcode_path, workdir)
            baseline_instruction_count = None
        oz_size = None
        oz_instruction_count = None
        try:
            oz_bc = workdir / "oz.bc"
            optimize_oz(bitcode_path, oz_bc)
            if measure_instructions:
                oz_size, oz_instruction_count = measure_text_and_instruction_count(
                    oz_bc,
                    workdir,
                )
            else:
                oz_size = measure_text_size(oz_bc, workdir)
        except Exception:
            oz_size = None

        prefix_cache: dict[tuple[str, ...], dict[str, Any]] = {
            (): {"bc": str(bitcode_path), "size": baseline_size, "error": ""}
        }
        if measure_instructions:
            prefix_cache[()]["instruction_count"] = baseline_instruction_count

        segment_paths = _generate_superpath_segments(graph, args)
        tiny_graph_mode = len(graph.nodes) <= int(getattr(args, "tiny_graph_threshold", 4))
        configured_segment_floor = int(getattr(args, "segment_min_length", 4))
        if tiny_graph_mode:
            segment_length_floor = 1
        elif any(len(path) >= configured_segment_floor for path in segment_paths):
            segment_length_floor = configured_segment_floor
        elif any(len(path) >= 2 for path in segment_paths):
            segment_length_floor = 2
        else:
            segment_length_floor = 1
        segment_eval_start_count = len(prefix_cache)
        segment_candidates: list[SuperSegmentCandidate] = []
        measured_segments: dict[tuple[str, ...], tuple[int, MeasuredSuperSegment]] = {}
        segment_failures = 0
        segments_filtered_nonpositive = 0
        segments_recovered_from_tail = 0
        segments_truncated_to_best_prefix = 0
        for segment_index, segment_path in enumerate(segment_paths, start=1):
            result = evaluate_candidate_with_prefix_cache(
                segment_path,
                workdir,
                prefix_cache,
                measure_instructions=measure_instructions and not deferred_instructions,
            )
            if result["error"]:
                segment_failures += 1
            measured_segment = _measured_super_segment_from_result(
                segment_path,
                result,
                baseline_size,
            )
            if measured_segment is None:
                continue
            if measured_segment.passes in measured_segments:
                continue
            measured_segments[measured_segment.passes] = (segment_index, measured_segment)
            if measured_segment.recovered_from_tail:
                segments_recovered_from_tail += 1
            if measured_segment.truncated_to_best_prefix and not measured_segment.recovered_from_tail:
                segments_truncated_to_best_prefix += 1
            if measured_segment.vertex_delta < args.superpath_min_segment_delta:
                segments_filtered_nonpositive += 1
                continue
            segment_candidates.append(
                SuperSegmentCandidate(
                    index=segment_index,
                    passes=measured_segment.passes,
                    vertex_delta=measured_segment.vertex_delta,
                )
            )

        segment_delta_filter_bypassed = False
        if not segment_candidates and measured_segments:
            segment_delta_filter_bypassed = True
            fallback_segments = sorted(
                measured_segments.values(),
                key=lambda item: (
                    item[1].vertex_delta,
                    -len(item[1].passes),
                    tuple(reversed(item[1].passes)),
                ),
                reverse=True,
            )[:3]
            segment_candidates = [
                SuperSegmentCandidate(
                    index=index,
                    passes=segment.passes,
                    vertex_delta=segment.vertex_delta,
                )
                for index, segment in fallback_segments
            ]

        segment_eval_end_count = len(prefix_cache)
        segment_eval_cost = segment_eval_end_count - segment_eval_start_count
        superpath_eval_top_k = args.superpath_eval_top_k or args.top_k
        superpath_result = _build_superpath_candidates_with_stats(
            graph,
            segment_candidates,
            top_k=superpath_eval_top_k,
            max_pass_length=args.max_length,
            min_edge_weight=args.min_edge_weight,
            beam_factor=args.superpath_beam_factor,
            max_candidates=args.superpath_max_candidates,
            max_overlap=args.superpath_max_overlap,
        )
        superpath_candidates = superpath_result.candidates

        candidate_rows = []
        best_row: dict[str, Any] | None = None
        superpath_eval_start_count = len(prefix_cache)
        for candidate_index, candidate in enumerate(superpath_candidates, start=1):
            passes = list(candidate.passes)
            result = evaluate_candidate_with_prefix_cache(
                passes,
                workdir,
                prefix_cache,
                measure_instructions=measure_instructions and not deferred_instructions,
            )
            row = _make_candidate_row(
                graph=graph,
                heuristic=args.heuristic,
                candidate_index=candidate_index,
                passes=passes,
                result=result,
                bitcode_path=bitcode_path,
                baseline_size=baseline_size,
                oz_size=oz_size,
                baseline_instruction_count=baseline_instruction_count,
                oz_instruction_count=oz_instruction_count,
                top_k=len(superpath_candidates),
                extra={
                    "superpath_score": candidate.score,
                    "superpath_vertex_delta": candidate.vertex_delta,
                    "superpath_edge_score": candidate.edge_score,
                    "superpath_segments": list(candidate.segment_indexes),
                    "segment_candidate_count": len(segment_paths),
                    "segment_valid_count": len(segment_candidates),
                    "segment_length_floor": segment_length_floor,
                    "tiny_graph_mode": tiny_graph_mode,
                    "segment_failures": segment_failures,
                    "segments_filtered_nonpositive": segments_filtered_nonpositive,
                    "segments_recovered_from_tail": segments_recovered_from_tail,
                    "segments_truncated_to_best_prefix": segments_truncated_to_best_prefix,
                    "segment_delta_filter_bypassed": segment_delta_filter_bypassed,
                    "superpath_truncated": superpath_result.truncated,
                    "segment_eval_cost": segment_eval_cost,
                    "superpath_eval_cost": 0,
                },
            )
            candidate_rows.append(row)
            if best_row is None or _selected_key(row) > _selected_key(best_row):
                best_row = row

        superpath_eval_cost = len(prefix_cache) - superpath_eval_start_count
        for row in candidate_rows:
            row["superpath_eval_cost"] = superpath_eval_cost

        instruction_eval_cost = 0
        if (
            deferred_instructions
            and best_row is not None
            and baseline_instruction_count is not None
        ):
            instruction_result, instruction_eval_cost = (
                measure_deferred_candidate_instructions(
                    list(best_row["passes"]),
                    workdir,
                    prefix_cache,
                )
            )
            apply_instruction_result_to_row(
                best_row,
                instruction_result,
                baseline_instruction_count,
            )

        if best_row is None:
            best_row = {
                "benchmark": graph.benchmark,
                "heuristic": args.heuristic,
                "candidate_index": 0,
                "start_pass": "",
                "start_weight": 0,
                "top_k": len(superpath_candidates),
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
                "error": "no superpath candidates",
                "error_kind": "full_failure",
                "segment_candidate_count": len(segment_paths),
                "segment_valid_count": len(segment_candidates),
                "segment_length_floor": segment_length_floor,
                "tiny_graph_mode": tiny_graph_mode,
                "segment_failures": segment_failures,
                "segments_filtered_nonpositive": segments_filtered_nonpositive,
                "segments_recovered_from_tail": segments_recovered_from_tail,
                "segments_truncated_to_best_prefix": segments_truncated_to_best_prefix,
                "segment_delta_filter_bypassed": segment_delta_filter_bypassed,
                "superpath_truncated": superpath_result.truncated,
            }
        prefix_failures = sum(1 for entry in prefix_cache.values() if entry.get("error"))
        best_row["prefix_failures"] = prefix_failures
        best_row["superpath_eval_cost"] = superpath_eval_cost
        best_row["instruction_measurement"] = instruction_measurement
        best_row["instruction_eval_cost"] = instruction_eval_cost
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
        if args.heuristic == "cycle_breaking_superpath_topk":
            selected, candidates, cache_count = evaluate_superpath_for_benchmark(
                graph,
                bitcode_paths[benchmark],
                args=args,
                measure_instructions=args.measure_instructions,
                instruction_measurement=args.instruction_measurement,
            )
        else:
            paths = generate_topk_paths(graph, args.heuristic, args)
            selected, candidates, cache_count = evaluate_topk_for_benchmark(
                graph,
                bitcode_paths[benchmark],
                paths,
                heuristic=args.heuristic,
                measure_instructions=args.measure_instructions,
                instruction_measurement=args.instruction_measurement,
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
            "segment_top_k": getattr(args, "segment_top_k", 100),
            "segment_min_length": getattr(args, "segment_min_length", 4),
            "segment_max_length": getattr(args, "segment_max_length", 6),
            "superpath_beam_factor": getattr(args, "superpath_beam_factor", 5),
            "superpath_max_candidates": getattr(args, "superpath_max_candidates", 100_000),
            "superpath_min_segment_delta": getattr(args, "superpath_min_segment_delta", 1),
            "superpath_max_overlap": getattr(args, "superpath_max_overlap", 1),
            "segment_max_jaccard": getattr(args, "segment_max_jaccard", 0.75),
            "tiny_graph_threshold": getattr(args, "tiny_graph_threshold", 4),
            "superpath_eval_top_k": getattr(args, "superpath_eval_top_k", 0),
            "measure_instructions": args.measure_instructions,
            "instruction_measurement": getattr(args, "instruction_measurement", "deferred"),
            "chunk_size": getattr(args, "chunk_size", 4),
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
            "cycle_breaking_superpath_topk",
            "random_walk_top10",
            "random_walk_topk",
            "exhaustive_len6_top10",
            "bucket_dag_topk",
            "bucket_dag_top250",
        ],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Bucket size for the bucket_dag heuristic (nodes per layer).",
    )
    parser.add_argument("--top-starts", type=int, default=10)
    parser.add_argument("--paths-per-start", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--min-edge-weight", type=int, default=1)
    parser.add_argument("--random-walks", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--exhaustive-length", type=int, default=6)
    parser.add_argument("--segment-top-k", type=int, default=100)
    parser.add_argument("--segment-min-length", type=int, default=4)
    parser.add_argument("--segment-max-length", type=int, default=6)
    parser.add_argument("--superpath-beam-factor", type=int, default=5)
    parser.add_argument("--superpath-max-candidates", type=int, default=100_000)
    parser.add_argument(
        "--superpath-min-segment-delta",
        type=int,
        default=1,
        help="Minimum measured best-prefix delta for a segment before fallback.",
    )
    parser.add_argument("--superpath-max-overlap", type=int, default=1)
    parser.add_argument("--segment-max-jaccard", type=float, default=0.75)
    parser.add_argument(
        "--tiny-graph-threshold",
        type=int,
        default=4,
        help="Enumerate all simple paths when the pass graph has at most this many nodes.",
    )
    parser.add_argument("--superpath-eval-top-k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--measure-instructions",
        action="store_true",
        help="Also measure machine instruction counts with llvm-objdump -d.",
    )
    parser.add_argument(
        "--instruction-measurement",
        choices=["deferred", "eager"],
        default="deferred",
        help=(
            "Measure instructions only for the size-selected candidate (deferred) "
            "or for every prefix (eager)."
        ),
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


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.top_starts <= 0:
        raise ValueError("--top-starts must be positive")
    if args.paths_per_start <= 0:
        raise ValueError("--paths-per-start must be positive")
    if args.segment_top_k <= 0:
        raise ValueError("--segment-top-k must be positive")
    if args.segment_min_length < 2:
        raise ValueError("--segment-min-length must be >= 2")
    if args.segment_max_length < args.segment_min_length:
        raise ValueError("--segment-max-length must be >= --segment-min-length")
    if args.superpath_beam_factor <= 0:
        raise ValueError("--superpath-beam-factor must be positive")
    if args.superpath_max_candidates <= 0:
        raise ValueError("--superpath-max-candidates must be positive")
    if args.superpath_max_overlap < 0:
        raise ValueError("--superpath-max-overlap must be non-negative")
    if not 0.0 <= args.segment_max_jaccard <= 1.0:
        raise ValueError("--segment-max-jaccard must be between 0 and 1")
    if args.tiny_graph_threshold < 0:
        raise ValueError("--tiny-graph-threshold must be non-negative")
    if args.superpath_eval_top_k < 0:
        raise ValueError("--superpath-eval-top-k must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_tools("opt", "llc", "llvm-size")
    if args.measure_instructions:
        require_tools("llvm-objdump")
    validate_args(args)
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
