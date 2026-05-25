"""Compare pass-search methods on per-function LLVM bitcode."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cem import CandidateResult, CEMConfig, search_pass_sequence_for_function
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


def filter_valid_passes(bitcode_path: Path, passes: list[str]) -> tuple[list[str], list[str]]:
    """Keep passes accepted by the local opt binary for the given bitcode."""
    valid = []
    invalid = []
    with tempfile.TemporaryDirectory(prefix="llvm-ir-pass-check-") as tmp:
        out = Path(tmp) / "check.bc"
        for pass_name in passes:
            try:
                run_cmd(["opt", f"-passes={pass_name}", str(bitcode_path), "-o", str(out)])
                valid.append(pass_name)
            except LLVMCommandError:
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
    rng: random.Random,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-cem-") as tmp:
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

        config = CEMConfig(
            steps=steps,
            iterations=iterations,
            candidates=candidates,
            elite_size=elite_size,
            smoothing=smoothing,
            min_prob=min_prob,
            epsilon=epsilon,
            allow_stop=allow_stop,
        )

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

        result = search_pass_sequence_for_function(
            passes,
            baseline_size,
            config=config,
            rng=rng,
            evaluate_candidate=evaluate_candidate,
        )
        best = result.best
        return {
            "function": bitcode_path.name,
            "baseline_size": baseline_size,
            "oz_size": oz_size,
            "oz_delta": baseline_size - oz_size if oz_size is not None else None,
            "oz_error": oz_error,
            "cem_best_size": result.best_size,
            "cem_delta": result.delta,
            "cem_best_passes": best.passes if best else [],
            "cem_best_actions": best.actions if best else [],
            "cem_total_evaluated": result.total_evaluated,
            "cem_failed": result.failed,
        }


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
        "baseline_size",
        "oz_size",
        "oz_delta",
        "cem_best_size",
        "cem_delta",
        "ppo_best_size",
        "ppo_best_delta",
        "ppo_final_size",
        "ppo_final_delta",
        "ppo_best_step",
        "cem_total_evaluated",
        "cem_failed",
        "cem_best_passes",
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
            rendered["cem_best_passes"] = ",".join(row["cem_best_passes"])
            rendered["ppo_best_passes"] = ",".join(row.get("ppo_best_passes") or [])
            rendered["ppo_passes"] = ",".join(row.get("ppo_passes") or [])
            writer.writerow({name: rendered.get(name) for name in fieldnames})


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    improved = [row for row in rows if row["cem_delta"] > 0]
    oz_available = [row for row in rows if row["oz_delta"] is not None]
    beats_oz = [
        row for row in oz_available
        if row["cem_best_size"] < row["oz_size"]
    ]
    summary = {
        "functions": len(rows),
        "cem_improved": len(improved),
        "cem_improved_percent": 100.0 * len(improved) / len(rows),
        "cem_total_delta": sum(int(row["cem_delta"]) for row in rows),
        "cem_mean_delta": sum(float(row["cem_delta"]) for row in rows) / len(rows),
        "oz_available": len(oz_available),
        "cem_beats_oz": len(beats_oz),
        "cem_beats_oz_percent": 100.0 * len(beats_oz) / len(oz_available)
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
        ppo_beats_cem = [
            row for row in ppo_rows
            if row["ppo_best_size"] < row["cem_best_size"]
        ]
        cem_beats_ppo = [
            row for row in ppo_rows
            if row["cem_best_size"] < row["ppo_best_size"]
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
            "ppo_beats_cem": len(ppo_beats_cem),
            "cem_beats_ppo": len(cem_beats_ppo),
            "cem_ties_ppo": len(ppo_rows) - len(ppo_beats_cem) - len(cem_beats_ppo),
        })
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CEM pass search on per-function LLVM bitcode."
    )
    parser.add_argument(
        "--dataset-dir",
        default="datasets/autotune_stratified_30_functions_bc",
        help="Directory with per-function .bc files.",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.65)
    parser.add_argument("--min-prob", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument(
        "--no-stop-action",
        action="store_true",
        help="Disable CEM STOP action and force fixed-length pass sequences.",
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
    if files and not args.no_filter_invalid_passes:
        passes, invalid_passes = filter_valid_passes(files[0], passes)
        if invalid_passes:
            print(
                "Filtered invalid passes: " + ", ".join(invalid_passes),
                flush=True,
            )
    rng = random.Random(args.seed)
    rows = []
    ppo_config = Path(args.ppo_config).resolve() if args.ppo_config else None
    ppo_checkpoint = Path(args.ppo_checkpoint).resolve() if args.ppo_checkpoint else None
    for index, bitcode_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {bitcode_path.name}", flush=True)
        row = run_cem_for_function(
            bitcode_path,
            passes,
            steps=args.steps,
            iterations=args.iterations,
            candidates=args.candidates,
            elite_size=args.elite_size,
            smoothing=args.smoothing,
            min_prob=args.min_prob,
            epsilon=args.epsilon,
            allow_stop=not args.no_stop_action,
            rng=rng,
        )
        if ppo_config and ppo_checkpoint:
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
        rows.append(row)
    config = {
        "dataset_dir": str(dataset_dir),
        "limit": args.limit,
        "seed": args.seed,
        "steps": args.steps,
        "iterations": args.iterations,
        "candidates": args.candidates,
        "elite_size": args.elite_size,
        "smoothing": args.smoothing,
        "min_prob": args.min_prob,
        "epsilon": args.epsilon,
        "allow_stop": not args.no_stop_action,
        "pass_count": len(passes),
        "invalid_passes_filtered": invalid_passes,
        "ppo_config": str(ppo_config) if ppo_config else None,
        "ppo_checkpoint": str(ppo_checkpoint) if ppo_checkpoint else None,
    }
    write_outputs(rows, output_dir, config)
    print(json.dumps(summarize_rows(rows), indent=2), flush=True)
    print(f"Wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
