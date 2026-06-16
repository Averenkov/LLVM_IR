"""Run Chunk-Forest on a single benchmark (debug/repro driver)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llvm_ir.stages.translation_unit.evaluate_chunk_forest import (
    build_report,
    group_results_by_benchmark,
    load_function_pass_results_from_report,
    parse_args,
    write_report,
)
from llvm_ir.stages.translation_unit.evaluate import write_translation_unit_bitcodes


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--benchmark", required=True)
    known, rest = pre.parse_known_args()
    args = parse_args(rest)

    results = load_function_pass_results_from_report(
        args.comparison, algorithm=args.algorithm or None
    )
    groups = group_results_by_benchmark(results)
    benchmark = known.benchmark
    if benchmark not in groups:
        raise SystemExit(f"benchmark {benchmark} not in comparison results")
    groups = {benchmark: groups[benchmark]}
    bitcode_paths = write_translation_unit_bitcodes(
        [benchmark], args.bitcode_dir, site_data_paths=args.site_data
    )
    report = build_report(groups, bitcode_paths, args)
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2), flush=True)
    row = report["selected_rows"][0]
    print(
        json.dumps(
            {
                "benchmark": row["benchmark"],
                "best_delta": row["best_delta"],
                "best_passes": row["best_passes"],
                "real_evals": row["real_evals"],
                "crashing_passes": row.get("crashing_passes"),
                "wave1_real_evals": row.get("wave1_real_evals"),
                "wave2_real_evals": row.get("wave2_real_evals"),
                "pool_unique_paths": row.get("pool_unique_paths"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
