"""CLI for aggregation heuristic experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llvm_ir.heuristics.aggregation.base import load_dataset_from_pass_search_report
from llvm_ir.heuristics.aggregation.compare import run_comparison
from llvm_ir.heuristics.aggregation.registry import available_heuristics, build_heuristic
from llvm_ir.heuristics.aggregation.tu_eval import TUEvaluator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregation heuristics.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--heuristic", default="hpp_eades_topk")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--tu-path", default="")
    parser.add_argument("--baseline-text-size", type=int, default=0)
    parser.add_argument("--oz-text-size", type=int, default=0)
    parser.add_argument("--no-tu-eval", action="store_true")
    parser.add_argument("--top-k-validate", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".aggregation_cache"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    dataset = load_dataset_from_pass_search_report(
        args.dataset,
        algorithm=args.algorithm,
        name=args.dataset.stem,
    )
    dataset.translation_unit_path = args.tu_path
    dataset.baseline_text_size = args.baseline_text_size
    dataset.oz_text_size = args.oz_text_size
    names = available_heuristics() if args.all else [args.heuristic]
    heuristics = [build_heuristic(name) for name in names]

    def evaluator_factory(current_dataset):
        if not current_dataset.translation_unit_path:
            return None
        return TUEvaluator(current_dataset.translation_unit_path, cache_dir=args.cache_dir)

    report = run_comparison(
        [dataset],
        heuristics,
        evaluator_factory,
        args.output,
        config=config,
        no_tu_eval=args.no_tu_eval or not args.tu_path,
        top_k_validate=args.top_k_validate,
    )
    print(
        json.dumps(
            {
                "heuristics": names,
                "rows": len(report["rows"]),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return dict(json.loads(text))
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text) or {}
    return dict(loaded)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = _parse_scalar(value.strip())
    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return [item.strip() for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


if __name__ == "__main__":
    raise SystemExit(main())

