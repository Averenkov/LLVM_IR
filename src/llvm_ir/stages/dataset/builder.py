"""Build function-level LLVM bitcode datasets from CompilerGym benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


DEFINE_RE = re.compile(r"^\s*define\b.*?@([^(]+)\(", re.ASCII)
ToolRunner = Callable[[list[str]], None]
ToolChecker = Callable[[str], str | None]


def run_tool(cmd: list[str]) -> None:
    """Run an LLVM command and raise a readable error on failure."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )


def which_or_die(name: str, tool_checker: ToolChecker = shutil.which) -> str:
    """Return the path to a required executable or fail with a helpful message."""
    path = tool_checker(name)
    if not path:
        raise RuntimeError(
            f"Required tool '{name}' not found in PATH. Install LLVM tools or adjust PATH."
        )
    return path


def sanitize_filename(name: str) -> str:
    """Make a benchmark or function name safe for filesystem paths."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def parse_llvm_symbol_name(raw_name: str) -> str:
    """Normalize the function name captured from LLVM IR for LLVM CLI tools."""
    if len(raw_name) >= 2 and raw_name[0] == '"' and raw_name[-1] == '"':
        return raw_name[1:-1]
    return raw_name


def safe_function_stem(function_name: str, max_len: int = 96) -> str:
    """Create a stable, filesystem-safe stem for a function name."""
    safe = sanitize_filename(function_name).strip("._-") or "function"
    if len(safe) <= max_len:
        return safe

    digest = hashlib.sha1(function_name.encode("utf-8", errors="replace")).hexdigest()[:12]
    prefix_len = max_len - len(digest) - 1
    return f"{safe[:prefix_len]}_{digest}"


def count_llvm_ir_instructions(ll_file_path: Path) -> int:
    """Count LLVM IR instructions using the line-based heuristic from the prototype."""
    content = ll_file_path.read_text(encoding="utf-8", errors="replace")

    count = 0
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.endswith(":"):
            continue
        if line.startswith(("define ", "declare ", "!", "attributes ", "target ")):
            continue
        count += 1
    return count


def parse_function_names(ll_path: Path) -> list[str]:
    """Parse defined function names from an LLVM IR file, preserving first-seen order."""
    funcs: list[str] = []
    seen: set[str] = set()
    with ll_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = DEFINE_RE.match(line)
            if not match:
                continue
            name = parse_llvm_symbol_name(match.group(1))
            if name not in seen:
                seen.add(name)
                funcs.append(name)
    return funcs


def extract_functions_from_bc(
    input_bc: Path,
    out_ll_dir: Path,
    runner: ToolRunner = run_tool,
) -> dict[str, int]:
    """Extract all defined functions from one bitcode file into per-function .ll files."""
    out_ll_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, int] = {}

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        full_ll = tmp / "full.ll"
        runner(["llvm-dis", str(input_bc), "-o", str(full_ll)])

        for name in parse_function_names(full_ll):
            safe = safe_function_stem(name)
            tmp_bc = tmp / f"{safe}.bc"
            out_ll = out_ll_dir / f"{safe}.ll"
            try:
                runner(["llvm-extract", f"-func={name}", str(input_bc), "-o", str(tmp_bc)])
                runner(["llvm-dis", str(tmp_bc), "-o", str(out_ll)])
            except RuntimeError as exc:
                print(f"  Skipping function {name} in {input_bc.name}: {exc}", file=sys.stderr)
                continue
            result[out_ll.name] = count_llvm_ir_instructions(out_ll)

    return result


def select_top_functions(
    functions: dict[str, int],
    top_percent: float,
) -> list[tuple[str, int]]:
    """Select the largest top-percent functions, keeping at least one per benchmark."""
    if not functions:
        return []
    if top_percent <= 0:
        raise ValueError("top_percent must be positive")

    top_k = max(1, round(len(functions) * top_percent / 100.0))
    return sorted(functions.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def select_functions(
    functions: dict[str, int],
    top_percent: float | None,
) -> list[tuple[str, int]]:
    """Select functions for the dataset, or keep all when selection is disabled."""
    if top_percent is None:
        return sorted(functions.items(), key=lambda item: item[0])
    return select_top_functions(functions, top_percent)


def iter_benchmarks(env: Any, dataset_name: str, max_benchmarks: int | None) -> Iterable[Any]:
    """Yield CompilerGym benchmark objects in deterministic URI order."""
    dataset = env.datasets[dataset_name]
    uris = sorted(str(benchmark) for benchmark in dataset)
    if max_benchmarks is not None:
        uris = uris[:max_benchmarks]
    for uri in uris:
        yield dataset.benchmark(uri)


def read_benchmark_uris(csv_path: Path) -> list[str]:
    """Read benchmark URIs from a benchmark-set CSV file."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []
    if "benchmark_uri" not in rows[0]:
        raise ValueError(f"{csv_path} must contain a benchmark_uri column")
    return [row["benchmark_uri"] for row in rows if row.get("benchmark_uri")]


def _dataset_key_from_benchmark_uri(benchmark_uri: str) -> str:
    """Return the CompilerGym dataset key for a benchmark URI."""
    without_scheme = benchmark_uri.removeprefix("benchmark://")
    return without_scheme.rsplit("/", 1)[0]


def _lookup_dataset(env: Any, dataset_key: str) -> Any:
    """Look up a CompilerGym dataset by short key, falling back to URI form."""
    try:
        return env.datasets[dataset_key]
    except KeyError:
        return env.datasets[f"benchmark://{dataset_key}"]


def iter_benchmark_uris(env: Any, benchmark_uris: list[str], max_benchmarks: int | None) -> Iterable[Any]:
    """Yield CompilerGym benchmark objects for explicit benchmark URIs."""
    selected = benchmark_uris[:max_benchmarks] if max_benchmarks is not None else benchmark_uris
    for uri in selected:
        dataset = _lookup_dataset(env, _dataset_key_from_benchmark_uri(uri))
        yield dataset.benchmark(uri)


def benchmark_short_name(benchmark: Any, *, include_dataset: bool = False) -> str:
    """Convert a CompilerGym benchmark URI to a safe dataset file prefix."""
    uri = str(benchmark.uri)
    if include_dataset:
        return sanitize_filename(uri.removeprefix("benchmark://"))
    return sanitize_filename(uri.rsplit("/", 1)[-1])


def _make_compiler_gym_env() -> Any:
    try:
        import compiler_gym
    except ImportError as exc:
        raise RuntimeError(
            "compiler_gym is required to build datasets. Install the optional "
            "dependency with: pip install '.[compiler-gym]'"
        ) from exc
    try:
        return compiler_gym.make("llvm-v0", disable_env_checker=True)
    except TypeError:
        return compiler_gym.make("llvm-v0")


def build_dataset(
    args: argparse.Namespace,
    *,
    make_env: Callable[[], Any] = _make_compiler_gym_env,
    runner: ToolRunner = run_tool,
    tool_checker: ToolChecker = shutil.which,
) -> int:
    """Build the selected per-function .bc dataset."""
    for tool in ("llvm-dis", "llvm-as", "llvm-extract"):
        which_or_die(tool, tool_checker)

    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_bcs_dir = work_dir / "input_bcs"
    input_functions_dir = work_dir / "input_functions"
    input_bcs_dir.mkdir(parents=True, exist_ok=True)
    input_functions_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Opening CompilerGym dataset: {args.dataset}")
    env = make_env()
    benchmarks_processed = 0
    functions_kept = 0
    try:
        benchmark_file = getattr(args, "benchmark_file", None)
        if benchmark_file:
            benchmark_uris = read_benchmark_uris(Path(benchmark_file))
            benchmarks = iter_benchmark_uris(env, benchmark_uris, args.max_benchmarks)
            include_dataset_in_name = True
        else:
            benchmarks = iter_benchmarks(env, args.dataset, args.max_benchmarks)
            include_dataset_in_name = False

        for benchmark in benchmarks:
            name = benchmark_short_name(benchmark, include_dataset=include_dataset_in_name)
            bc_path = input_bcs_dir / f"{name}.bc"

            print(f"[2/4] Benchmark {name}: write_bitcode -> {bc_path.name}")
            env.reset(benchmark=benchmark)
            env.write_bitcode(str(bc_path))

            with tempfile.TemporaryDirectory(prefix=f"{name}_funcs_", dir=str(work_dir)) as tmp_str:
                tmp_funcs = Path(tmp_str)
                try:
                    functions = extract_functions_from_bc(bc_path, tmp_funcs, runner=runner)
                except RuntimeError as exc:
                    print(f"  Skipping {name}: {exc}", file=sys.stderr)
                    continue

                selection_percent = (
                    None
                    if getattr(args, "no_function_selection", False)
                    else args.top_percent
                )
                selected = select_functions(functions, selection_percent)
                if not selected:
                    print(f"  No defined functions found in {name}; skipping")
                    continue

                if selection_percent is None:
                    print(f"  Found {len(functions)} functions, keeping all = {len(selected)}")
                else:
                    print(
                        f"  Found {len(functions)} functions, "
                        f"keeping top {selection_percent}% = {len(selected)}"
                    )
                for ll_fname, _count in selected:
                    src = tmp_funcs / ll_fname
                    func_stem = Path(ll_fname).stem
                    dst = input_functions_dir / f"{name}_{func_stem}.ll"
                    shutil.copy(src, dst)
                    functions_kept += 1

            benchmarks_processed += 1
    finally:
        env.close()

    print(f"[3/4] Assembling selected functions to .bc -> {output_dir}")
    assembled = 0
    for ll_path in sorted(input_functions_dir.glob("*.ll")):
        bc_path = output_dir / f"{ll_path.stem}.bc"
        try:
            runner(["llvm-as", str(ll_path), "-o", str(bc_path)])
            assembled += 1
        except RuntimeError as exc:
            print(f"  llvm-as failed on {ll_path.name}: {exc}", file=sys.stderr)

    if not args.keep_intermediate:
        print(f"[4/4] Cleaning intermediate files in {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"[4/4] Intermediate files kept in {work_dir}")

    print("")
    print(f"Done. Benchmarks processed: {benchmarks_processed}")
    print(f"      Functions selected:   {functions_kept}")
    print(f"      .bc files assembled:  {assembled}")
    print(f"      Output directory:     {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a function-level .bc dataset from CompilerGym by splitting "
            "benchmarks into functions and keeping the largest top N percent."
        )
    )
    parser.add_argument("--dataset", default="cbench-v1", help="CompilerGym dataset name.")
    parser.add_argument(
        "--benchmark-file",
        default=None,
        help="CSV file with a benchmark_uri column. If set, --dataset is ignored.",
    )
    parser.add_argument(
        "--output-dir",
        default="./input_functions_bc",
        help="Final directory with per-function .bc files.",
    )
    parser.add_argument(
        "--work-dir",
        default="./build_workspace",
        help="Directory for intermediate .bc/.ll files.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=20.0,
        help="Percent of largest functions to keep.",
    )
    parser.add_argument(
        "--no-function-selection",
        action="store_true",
        help="Keep all extracted functions instead of selecting the largest top percent.",
    )
    parser.add_argument(
        "--max-benchmarks",
        type=int,
        default=None,
        help="Limit benchmark count for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --output-dir before building.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep --work-dir after the run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return build_dataset(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
