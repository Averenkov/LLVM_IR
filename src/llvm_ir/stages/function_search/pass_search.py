"""Compare pass-search methods on per-function LLVM bitcode."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cem import CandidateResult, CEMConfig
from .algorithms import (
    FunctionPassSearchAlgorithm,
    FunctionSearchContext,
    build_function_search_algorithm,
    candidate_to_row_prefix,
)
from .passes import DEFAULT_PASSES


class LLVMCommandError(RuntimeError):
    """Raised when an LLVM command fails."""


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        raise LLVMCommandError(
            f"{' '.join(cmd)} failed with rc={exc.returncode}: {exc.stderr.strip()}"
        ) from exc


def require_tools(*tools: str) -> None:
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        raise LLVMCommandError("missing LLVM tools in PATH: " + ", ".join(missing))


def measure_text_size(bitcode_path: Path, workdir: Path) -> int:
    obj_path = workdir / f"{bitcode_path.stem}.o"
    run_cmd(["llc", "-filetype=obj", str(bitcode_path), "-o", str(obj_path)])
    out = run_cmd(["llvm-size", str(obj_path)]).stdout.strip().splitlines()
    if len(out) < 2:
        raise LLVMCommandError(f"unexpected llvm-size output: {out!r}")
    return int(out[1].split()[0])


def apply_pass_sequence(
    input_bc: Path,
    passes: list[str],
    output_bc: Path,
) -> None:
    if not passes:
        shutil.copyfile(input_bc, output_bc)
        return
    run_cmd(["opt", f"-passes={','.join(passes)}", str(input_bc), "-o", str(output_bc)])


def optimize_oz(input_bc: Path, output_bc: Path) -> None:
    run_cmd(["opt", "-passes=default<Oz>", str(input_bc), "-o", str(output_bc)])


def filter_valid_passes(
    bitcode_paths: Path | list[Path],
    passes: list[str],
) -> tuple[list[str], list[str]]:
    """Keep passes accepted by local opt on at least one validation bitcode."""
    paths = [bitcode_paths] if isinstance(bitcode_paths, Path) else list(bitcode_paths)
    if not paths:
        return [], list(passes)

    valid = []
    invalid = []
    with tempfile.TemporaryDirectory(prefix="llvm-ir-pass-check-") as tmp:
        out = Path(tmp) / "check.bc"
        for pass_name in passes:
            pass_valid = False
            for bitcode_path in paths:
                try:
                    run_cmd(["opt", f"-passes={pass_name}", str(bitcode_path), "-o", str(out)])
                    pass_valid = True
                    break
                except LLVMCommandError:
                    continue
            if pass_valid:
                valid.append(pass_name)
            else:
                invalid.append(pass_name)
    return valid, invalid


def evaluate_sequence(
    input_bc: Path,
    actions: list[int],
    selected_passes: list[str],
    baseline_size: int,
    workdir: Path,
    candidate_id: str,
) -> CandidateResult:
    if not selected_passes:
        return CandidateResult(
            actions=[],
            passes=[],
            size=baseline_size,
            reward=0.0,
        )

    current_bc = input_bc
    best_size = baseline_size
    best_prefix_len = 0
    evaluated_prefixes = 0

    for index, pass_name in enumerate(selected_passes, start=1):
        output_bc = workdir / f"{candidate_id}_prefix{index}.bc"
        try:
            apply_pass_sequence(current_bc, [pass_name], output_bc)
            size = measure_text_size(output_bc, workdir)
        except Exception as exc:  # noqa: BLE001
            if evaluated_prefixes:
                return CandidateResult(
                    actions=list(actions[:best_prefix_len]),
                    passes=list(selected_passes[:best_prefix_len]),
                    size=best_size,
                    reward=baseline_size - best_size,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return CandidateResult(
                actions=list(actions),
                passes=selected_passes,
                size=None,
                reward=float("-inf"),
                error=f"{type(exc).__name__}: {exc}",
            )
        evaluated_prefixes += 1
        current_bc = output_bc
        if size < best_size:
            best_size = size
            best_prefix_len = index

    return CandidateResult(
        actions=list(actions[:best_prefix_len]),
        passes=list(selected_passes[:best_prefix_len]),
        size=best_size,
        reward=baseline_size - best_size,
    )


def run_search_for_function(
    bitcode_path: Path,
    passes: list[str],
    *,
    algorithm: FunctionPassSearchAlgorithm,
    rng: random.Random,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"llvm-ir-{algorithm.name}-") as tmp:
        workdir = Path(tmp)
        baseline_size = measure_text_size(bitcode_path, workdir)
        oz_bc = workdir / "oz.bc"
        oz_error = ""
        oz_size = None
        try:
            optimize_oz(bitcode_path, oz_bc)
            oz_size = measure_text_size(oz_bc, workdir)
        except Exception as exc:  # noqa: BLE001
            oz_error = f"{type(exc).__name__}: {exc}"

        def evaluate_candidate(
            actions: list[int],
            selected_passes: list[str],
            candidate_id: str,
        ) -> CandidateResult:
            return evaluate_sequence(
                bitcode_path,
                actions,
                selected_passes,
                baseline_size,
                workdir,
                candidate_id,
            )

        result = algorithm.search(
            FunctionSearchContext(
                bitcode_path=bitcode_path,
                passes=passes,
                baseline_size=baseline_size,
                rng=rng,
                evaluate_candidate=evaluate_candidate,
            )
        )
        best = result.best
        row = {
            "function": bitcode_path.name,
            "search_algorithm": algorithm.name,
            "baseline_size": baseline_size,
            "oz_size": oz_size,
            "oz_delta": baseline_size - oz_size if oz_size is not None else None,
            "oz_error": oz_error,
        }
        row.update(
            candidate_to_row_prefix(
                algorithm.name,
                best,
                baseline_size,
                result.best_size,
                result.delta,
                result.total_evaluated,
                result.failed,
            )
        )
        if algorithm.name == "cem":
            row.update({
                "cem_best_size": result.best_size,
                "cem_delta": result.delta,
                "cem_best_passes": best.passes if best else [],
                "cem_best_actions": best.actions if best else [],
                "cem_total_evaluated": result.total_evaluated,
                "cem_failed": result.failed,
            })
        elif algorithm.name == "random":
            row.update({
                "random_best_size": result.best_size,
                "random_delta": result.delta,
                "random_best_passes": best.passes if best else [],
                "random_best_actions": best.actions if best else [],
                "random_total_evaluated": result.total_evaluated,
                "random_failed": result.failed,
            })
        return row


def run_cem_for_function(
    bitcode_path: Path,
    passes: list[str],
    *,
    steps: int,
    iterations: int,
    candidates: int,
    elite_size: int,
    smoothing: float,
    min_prob: float,
    epsilon: float,
    allow_stop: bool,
    evaluate_shifts: bool,
    rng: random.Random,
) -> dict[str, Any]:
    config = CEMConfig(
        steps=steps,
        iterations=iterations,
        candidates=candidates,
        elite_size=elite_size,
        smoothing=smoothing,
        min_prob=min_prob,
        epsilon=epsilon,
        allow_stop=allow_stop,
        evaluate_shifts=evaluate_shifts,
    )
    algorithm = build_function_search_algorithm("cem", cem_config=config)
    return run_search_for_function(
        bitcode_path,
        passes,
        algorithm=algorithm,
        rng=rng,
    )


def function_seed(seed: int, index: int) -> int:
    """Return the deterministic per-function seed used by serial and parallel jobs."""
    return seed + index * 1_000_003


def _run_search_job(
    job: tuple[int, str, list[str], str, CEMConfig, int],
) -> tuple[int, str, dict[str, Any]]:
    index, bitcode_path_str, passes, algorithm_name, config, seed = job
    algorithm = build_function_search_algorithm(algorithm_name, cem_config=config)
    row = run_search_for_function(
        Path(bitcode_path_str),
        passes,
        algorithm=algorithm,
        rng=random.Random(seed),
    )
    return index, Path(bitcode_path_str).name, row


def run_pass_search_jobs(
    files: list[Path],
    passes: list[str],
    *,
    algorithm_name: str,
    cem_config: CEMConfig,
    seed: int,
    jobs: int,
) -> list[dict[str, Any]]:
    """Run per-function pass search, optionally in parallel."""
    if jobs <= 1:
        algorithm = build_function_search_algorithm(
            algorithm_name,
            cem_config=cem_config,
        )
        rows = []
        for index, bitcode_path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {bitcode_path.name}", flush=True)
            rows.append(
                run_search_for_function(
                    bitcode_path,
                    passes,
                    algorithm=algorithm,
                    rng=random.Random(function_seed(seed, index)),
                )
            )
        return rows

    rows_by_index: dict[int, dict[str, Any]] = {}
    job_args = [
        (
            index,
            str(bitcode_path),
            passes,
            algorithm_name,
            cem_config,
            function_seed(seed, index),
        )
        for index, bitcode_path in enumerate(files, start=1)
    ]
    completed = 0
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(_run_search_job, job) for job in job_args]
        for future in as_completed(futures):
            index, function_name, row = future.result()
            rows_by_index[index] = row
            completed += 1
            print(
                f"[{completed}/{len(files)}] done {function_name}",
                flush=True,
            )
    return [rows_by_index[index] for index in range(1, len(files) + 1)]


def run_ppo_for_function(
    bitcode_path: Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    try:
        from llvm_minimizer.config import load_config
        from llvm_minimizer.ppo.infer import rollout
    except ImportError as exc:
        raise RuntimeError(
            "PPO evaluation requires llvm-minimizer and its dependencies. "
            "Run with llvm-minimizer's virtualenv or add it to PYTHONPATH."
        ) from exc

    cfg = load_config(config_path)
    result = rollout(cfg, checkpoint_path, bitcode_path)
    return {
        "ppo_final_size": result.final_size,
        "ppo_final_delta": result.baseline_size - result.final_size,
        "ppo_best_size": result.best_size,
        "ppo_best_delta": result.baseline_size - result.best_size,
        "ppo_best_step": result.best_step,
        "ppo_best_passes": result.best_passes,
        "ppo_passes": result.passes,
        "ppo_error": "",
    }


def select_bitcode_files(dataset_dir: Path, limit: int, seed: int) -> list[Path]:
    files = sorted(dataset_dir.glob("*.bc"))
    if limit and limit < len(files):
        rng = random.Random(seed)
        files = sorted(rng.sample(files, limit))
    return files


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": utc_timestamp(),
        "config": config,
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "function",
        "search_algorithm",
        "baseline_size",
        "oz_size",
        "oz_delta",
        "cem_best_size",
        "cem_delta",
        "random_best_size",
        "random_delta",
        "ppo_best_size",
        "ppo_best_delta",
        "ppo_final_size",
        "ppo_final_delta",
        "ppo_best_step",
        "cem_total_evaluated",
        "cem_failed",
        "random_total_evaluated",
        "random_failed",
        "cem_best_passes",
        "random_best_passes",
        "ppo_best_passes",
        "ppo_passes",
        "oz_error",
        "ppo_error",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["cem_best_passes"] = ",".join(row.get("cem_best_passes") or [])
            rendered["random_best_passes"] = ",".join(
                row.get("random_best_passes") or []
            )
            rendered["ppo_best_passes"] = ",".join(row.get("ppo_best_passes") or [])
            rendered["ppo_passes"] = ",".join(row.get("ppo_passes") or [])
            writer.writerow({name: rendered.get(name) for name in fieldnames})


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    algorithm = str(rows[0].get("search_algorithm") or "cem")
    delta_key = f"{algorithm}_delta"
    best_size_key = f"{algorithm}_best_size"
    if delta_key not in rows[0]:
        delta_key = "cem_delta"
        best_size_key = "cem_best_size"
        algorithm = "cem"
    improved = [row for row in rows if row[delta_key] > 0]
    oz_available = [row for row in rows if row["oz_delta"] is not None]
    beats_oz = [
        row for row in oz_available
        if row[best_size_key] < row["oz_size"]
    ]
    summary = {
        "functions": len(rows),
        f"{algorithm}_improved": len(improved),
        f"{algorithm}_improved_percent": 100.0 * len(improved) / len(rows),
        f"{algorithm}_total_delta": sum(int(row[delta_key]) for row in rows),
        f"{algorithm}_mean_delta": (
            sum(float(row[delta_key]) for row in rows) / len(rows)
        ),
        "oz_available": len(oz_available),
        f"{algorithm}_beats_oz": len(beats_oz),
        f"{algorithm}_beats_oz_percent": 100.0 * len(beats_oz) / len(oz_available)
        if oz_available
        else None,
        "oz_total_delta": sum(int(row["oz_delta"] or 0) for row in rows),
    }
    ppo_rows = [row for row in rows if row.get("ppo_best_delta") is not None]
    if ppo_rows:
        ppo_improved = [row for row in ppo_rows if int(row["ppo_best_delta"]) > 0]
        ppo_beats_oz = [
            row for row in ppo_rows
            if row.get("oz_size") is not None and row["ppo_best_size"] < row["oz_size"]
        ]
        ppo_beats_algorithm = [
            row for row in ppo_rows
            if row["ppo_best_size"] < row[best_size_key]
        ]
        algorithm_beats_ppo = [
            row for row in ppo_rows
            if row[best_size_key] < row["ppo_best_size"]
        ]
        summary.update({
            "ppo_available": len(ppo_rows),
            "ppo_improved": len(ppo_improved),
            "ppo_improved_percent": 100.0 * len(ppo_improved) / len(ppo_rows),
            "ppo_total_best_delta": sum(int(row["ppo_best_delta"]) for row in ppo_rows),
            "ppo_mean_best_delta": (
                sum(float(row["ppo_best_delta"]) for row in ppo_rows) / len(ppo_rows)
            ),
            "ppo_beats_oz": len(ppo_beats_oz),
            "ppo_beats_oz_percent": 100.0 * len(ppo_beats_oz) / len(ppo_rows),
            f"ppo_beats_{algorithm}": len(ppo_beats_algorithm),
            f"{algorithm}_beats_ppo": len(algorithm_beats_ppo),
            f"{algorithm}_ties_ppo": (
                len(ppo_rows) - len(ppo_beats_algorithm) - len(algorithm_beats_ppo)
            ),
        })
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pass-sequence search on per-function LLVM bitcode."
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/autotune_stratified_30_functions_bc",
        help="Directory with per-function .bc files.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--algorithm",
        choices=["cem", "random"],
        default="cem",
        help="Per-function pass-search algorithm.",
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for per-function search.",
    )
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.65)
    parser.add_argument("--min-prob", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument(
        "--no-stop-action",
        action="store_true",
        help="Disable STOP action and force fixed-length pass sequences.",
    )
    parser.add_argument(
        "--sequence-shifts",
        action="store_true",
        help="Evaluate unique cyclic shifts of every sampled sequence.",
    )
    parser.add_argument(
        "--no-sequence-shifts",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Sequence shifts are disabled by "
            "default; use --sequence-shifts to enable them."
        ),
    )
    parser.add_argument(
        "--ppo-config",
        default="",
        help="Optional llvm-minimizer YAML config for PPO evaluation.",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        default="",
        help="Optional trained llvm-minimizer PPO checkpoint.",
    )
    parser.add_argument(
        "--no-filter-invalid-passes",
        action="store_true",
        help="Use the raw pass list without checking pass availability in local opt.",
    )
    parser.add_argument(
        "--validate-passes-on",
        type=int,
        default=1,
        help="Validate passes on the first N selected bitcode files; valid if any succeeds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_tools("opt", "llc", "llvm-size")
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiments") / "pass_search_compare" / utc_timestamp()
    )
    files = select_bitcode_files(dataset_dir, args.limit, args.seed)
    passes = list(DEFAULT_PASSES)
    invalid_passes: list[str] = []
    validate_passes_on = max(1, args.validate_passes_on)
    if files and not args.no_filter_invalid_passes:
        passes, invalid_passes = filter_valid_passes(
            files[:validate_passes_on],
            passes,
        )
        if invalid_passes:
            print(
                "Filtered invalid passes: " + ", ".join(invalid_passes),
                flush=True,
            )
    cem_config = CEMConfig(
        steps=args.steps,
        iterations=args.iterations,
        candidates=args.candidates,
        elite_size=args.elite_size,
        smoothing=args.smoothing,
        min_prob=args.min_prob,
        epsilon=args.epsilon,
        allow_stop=not args.no_stop_action,
        evaluate_shifts=args.sequence_shifts and not args.no_sequence_shifts,
    )
    ppo_config = Path(args.ppo_config).resolve() if args.ppo_config else None
    ppo_checkpoint = Path(args.ppo_checkpoint).resolve() if args.ppo_checkpoint else None
    rows = run_pass_search_jobs(
        files,
        passes,
        algorithm_name=args.algorithm,
        cem_config=cem_config,
        seed=args.seed,
        jobs=args.jobs,
    )
    if ppo_config and ppo_checkpoint:
        for index, (bitcode_path, row) in enumerate(zip(files, rows), start=1):
            print(f"[ppo {index}/{len(files)}] {bitcode_path.name}", flush=True)
            try:
                row.update(
                    run_ppo_for_function(
                        bitcode_path,
                        config_path=ppo_config,
                        checkpoint_path=ppo_checkpoint,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                row.update({
                    "ppo_final_size": None,
                    "ppo_final_delta": None,
                    "ppo_best_size": None,
                    "ppo_best_delta": None,
                    "ppo_best_step": None,
                    "ppo_best_passes": [],
                    "ppo_passes": [],
                    "ppo_error": f"{type(exc).__name__}: {exc}",
                })
    config = {
        "dataset_dir": str(dataset_dir),
        "limit": args.limit,
        "seed": args.seed,
        "algorithm": args.algorithm,
        "steps": args.steps,
        "iterations": args.iterations,
        "candidates": args.candidates,
        "elite_size": args.elite_size,
        "smoothing": args.smoothing,
        "min_prob": args.min_prob,
        "epsilon": args.epsilon,
        "jobs": args.jobs,
        "allow_stop": not args.no_stop_action,
        "evaluate_shifts": args.sequence_shifts and not args.no_sequence_shifts,
        "pass_count": len(passes),
        "invalid_passes_filtered": invalid_passes,
        "validate_passes_on": validate_passes_on,
        "ppo_config": str(ppo_config) if ppo_config else None,
        "ppo_checkpoint": str(ppo_checkpoint) if ppo_checkpoint else None,
    }
    write_outputs(rows, output_dir, config)
    print(json.dumps(summarize_rows(rows), indent=2), flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
