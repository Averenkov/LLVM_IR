"""Evaluate aggregated pass sequences on whole translation-unit bitcode."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..function_search.pass_search import (
    LLVMCommandError,
    apply_pass_sequence,
    measure_text_size,
    optimize_oz,
    require_tools,
)


@dataclass(frozen=True)
class TranslationUnitSequence:
    benchmark: str
    heuristic: str
    passes: list[str]


@dataclass(frozen=True)
class TranslationUnitEvaluation:
    benchmark: str
    heuristic: str
    bitcode_path: str
    baseline_size: int
    oz_size: int | None
    final_size: int
    best_size: int
    best_prefix_len: int
    passes: list[str]
    best_passes: list[str]
    error: str = ""

    @property
    def final_delta(self) -> int:
        return self.baseline_size - self.final_size

    @property
    def best_delta(self) -> int:
        return self.baseline_size - self.best_size


def benchmark_uri_from_id(benchmark: str) -> str:
    suite, name = benchmark.split("_", 1)
    return f"benchmark://{suite}/{name}"


def load_heuristic_sequences(report_path: Path) -> list[TranslationUnitSequence]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [
        TranslationUnitSequence(
            benchmark=str(row["benchmark"]),
            heuristic=str(row["heuristic"]),
            passes=list(row.get("path") or []),
        )
        for row in report.get("results", [])
    ]


def _make_compiler_gym_env() -> Any:
    try:
        import compiler_gym
    except ImportError as exc:
        raise RuntimeError(
            "compiler_gym is required to restore whole benchmark bitcode. "
            "Run this command from an environment with compiler_gym installed."
        ) from exc
    try:
        return compiler_gym.make("llvm-v0", disable_env_checker=True)
    except TypeError:
        return compiler_gym.make("llvm-v0")


def _lookup_dataset(env: Any, dataset_key: str) -> Any:
    try:
        return env.datasets[dataset_key]
    except KeyError:
        return env.datasets[f"benchmark://{dataset_key}"]


def _candidate_benchmark_roots(site_data_path: Path) -> list[Path]:
    return [
        site_data_path / "llvm-v0" / "benchmark",
        site_data_path / "benchmark",
        site_data_path,
    ]


def _find_site_data_bitcode(benchmark: str, site_data_paths: list[Path]) -> Path | None:
    suite, name = benchmark.split("_", 1)
    direct_suffixes = [
        Path(suite) / "contents" / suite / f"{name}.bc",
    ]
    for site_data_path in site_data_paths:
        for benchmark_root in _candidate_benchmark_roots(site_data_path):
            for suffix in direct_suffixes:
                candidate = benchmark_root / suffix
                if candidate.exists():
                    return candidate
            contents = benchmark_root / suite / "contents"
            if not contents.exists():
                continue
            matches = sorted(contents.rglob(f"{name}.bc"))
            if matches:
                return matches[0]
    return None


def _copy_local_site_data_bitcodes(
    benchmarks: list[str],
    output_dir: Path,
    site_data_paths: list[Path],
) -> dict[str, Path]:
    written = {}
    for benchmark in benchmarks:
        output_path = output_dir / f"{benchmark}.bc"
        if output_path.exists():
            written[benchmark] = output_path
            continue
        source_path = _find_site_data_bitcode(benchmark, site_data_paths)
        if source_path is None:
            continue
        shutil.copyfile(source_path, output_path)
        written[benchmark] = output_path
    return written


def write_translation_unit_bitcodes(
    benchmarks: list[str],
    output_dir: Path,
    *,
    site_data_paths: list[Path] | None = None,
    make_env: Callable[[], Any] = _make_compiler_gym_env,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    site_data_paths = site_data_paths or []
    local = _copy_local_site_data_bitcodes(
        benchmarks,
        output_dir,
        site_data_paths,
    )
    existing = {
        benchmark: output_dir / f"{benchmark}.bc"
        for benchmark in benchmarks
        if (output_dir / f"{benchmark}.bc").exists()
    }
    existing.update(local)
    missing = [benchmark for benchmark in benchmarks if benchmark not in existing]
    if not missing:
        return existing

    env = make_env()
    written = dict(existing)
    try:
        for benchmark in missing:
            uri = benchmark_uri_from_id(benchmark)
            dataset_key = uri.removeprefix("benchmark://").rsplit("/", 1)[0]
            dataset = _lookup_dataset(env, dataset_key)
            compiler_gym_benchmark = dataset.benchmark(uri)
            output_path = output_dir / f"{benchmark}.bc"
            env.reset(benchmark=compiler_gym_benchmark)
            env.write_bitcode(str(output_path))
            written[benchmark] = output_path
    finally:
        env.close()
    return written


def evaluate_sequence_on_bitcode(
    bitcode_path: Path,
    sequence: TranslationUnitSequence,
) -> TranslationUnitEvaluation:
    with tempfile.TemporaryDirectory(prefix="llvm-ir-tu-eval-") as tmp_str:
        workdir = Path(tmp_str)
        baseline_size = measure_text_size(bitcode_path, workdir)
        oz_size = None
        try:
            oz_bc = workdir / "oz.bc"
            optimize_oz(bitcode_path, oz_bc)
            oz_size = measure_text_size(oz_bc, workdir)
        except Exception:
            oz_size = None

        current_bc = bitcode_path
        current_size = baseline_size
        final_size = baseline_size
        best_size = baseline_size
        best_prefix_len = 0
        best_passes: list[str] = []
        error = ""

        for index, pass_name in enumerate(sequence.passes, start=1):
            output_bc = workdir / f"{sequence.heuristic}_{index}.bc"
            try:
                apply_pass_sequence(current_bc, [pass_name], output_bc)
                current_size = measure_text_size(output_bc, workdir)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                break
            current_bc = output_bc
            final_size = current_size
            if current_size < best_size:
                best_size = current_size
                best_prefix_len = index
                best_passes = sequence.passes[:index]

        return TranslationUnitEvaluation(
            benchmark=sequence.benchmark,
            heuristic=sequence.heuristic,
            bitcode_path=str(bitcode_path),
            baseline_size=baseline_size,
            oz_size=oz_size,
            final_size=final_size,
            best_size=best_size,
            best_prefix_len=best_prefix_len,
            passes=list(sequence.passes),
            best_passes=best_passes,
            error=error,
        )


def summarize_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["heuristic"])].append(row)
    summary = {}
    for heuristic, items in sorted(grouped.items()):
        total_baseline = sum(int(row["baseline_size"]) for row in items)
        total_final_delta = sum(int(row["final_delta"]) for row in items)
        total_best_delta = sum(int(row["best_delta"]) for row in items)
        oz_rows = [row for row in items if row.get("oz_size") is not None]
        beats_oz = [
            row for row in oz_rows
            if int(row["best_size"]) < int(row["oz_size"])
        ]
        failed = [row for row in items if row.get("error")]
        # Per-benchmark percentages (each benchmark weighted equally) for the
        # macro / mean-of-percent average, alongside the size-weighted micro one.
        best_percents = [
            100.0 * int(row["best_delta"]) / int(row["baseline_size"])
            for row in items
            if int(row["baseline_size"]) > 0
        ]
        final_percents = [
            100.0 * int(row["final_delta"]) / int(row["baseline_size"])
            for row in items
            if int(row["baseline_size"]) > 0
        ]
        summary[heuristic] = {
            "benchmarks": len(items),
            "failed": len(failed),
            "improved_final": sum(1 for row in items if int(row["final_delta"]) > 0),
            "improved_best": sum(1 for row in items if int(row["best_delta"]) > 0),
            "total_final_delta": total_final_delta,
            "total_best_delta": total_best_delta,
            "weighted_final_percent": (
                100.0 * total_final_delta / total_baseline
                if total_baseline
                else 0.0
            ),
            "weighted_best_percent": (
                100.0 * total_best_delta / total_baseline
                if total_baseline
                else 0.0
            ),
            "macro_final_percent": (sum(final_percents) / len(final_percents) if final_percents else 0.0),
            "macro_best_percent": (sum(best_percents) / len(best_percents) if best_percents else 0.0),
            "min_best_percent": (min(best_percents) if best_percents else 0.0),
            "max_best_percent": (max(best_percents) if best_percents else 0.0),
            "oz_available": len(oz_rows),
            "beats_oz_best": len(beats_oz),
            "beats_oz_best_percent": (
                100.0 * len(beats_oz) / len(oz_rows)
                if oz_rows
                else None
            ),
            "mean_best_prefix_len": (
                sum(int(row["best_prefix_len"]) for row in items) / len(items)
                if items
                else 0.0
            ),
        }
    return summary


def evaluation_to_row(result: TranslationUnitEvaluation) -> dict[str, Any]:
    return {
        "benchmark": result.benchmark,
        "heuristic": result.heuristic,
        "bitcode_path": result.bitcode_path,
        "baseline_size": result.baseline_size,
        "oz_size": result.oz_size,
        "oz_delta": (
            result.baseline_size - result.oz_size
            if result.oz_size is not None
            else None
        ),
        "final_size": result.final_size,
        "final_delta": result.final_delta,
        "best_size": result.best_size,
        "best_delta": result.best_delta,
        "best_prefix_len": result.best_prefix_len,
        "passes": result.passes,
        "best_passes": result.best_passes,
        "error": result.error,
    }


def evaluate_translation_unit_sequences(
    sequences: list[TranslationUnitSequence],
    bitcode_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    rows = []
    for index, sequence in enumerate(sequences, start=1):
        print(
            f"[{index}/{len(sequences)}] {sequence.benchmark} {sequence.heuristic}",
            flush=True,
        )
        result = evaluate_sequence_on_bitcode(
            bitcode_paths[sequence.benchmark],
            sequence,
        )
        rows.append(evaluation_to_row(result))
    return rows


def write_evaluation_report(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    heuristics_input: Path,
    bitcode_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "heuristics_input": str(heuristics_input),
        "bitcode_dir": str(bitcode_dir),
        "summary": summarize_evaluations(rows),
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic pass sequences on whole translation units."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--bitcode-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--site-data",
        action="append",
        default=[],
        type=Path,
        help=(
            "CompilerGym site-data root to copy whole benchmark bitcode from "
            "before falling back to the CompilerGym service."
        ),
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
    if args.overwrite_bitcode and args.bitcode_dir.exists():
        shutil.rmtree(args.bitcode_dir)
    sequences = load_heuristic_sequences(args.input)
    if args.limit and args.limit < len(sequences):
        sequences = sequences[: args.limit]
    benchmarks = sorted({sequence.benchmark for sequence in sequences})
    bitcode_paths = write_translation_unit_bitcodes(
        benchmarks,
        args.bitcode_dir,
        site_data_paths=args.site_data,
    )
    rows = evaluate_translation_unit_sequences(sequences, bitcode_paths)
    write_evaluation_report(
        rows,
        args.output,
        heuristics_input=args.input,
        bitcode_dir=args.bitcode_dir,
    )
    print(json.dumps(summarize_evaluations(rows), indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
