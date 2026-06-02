"""Compatibility wrapper for `python -m scripts.run_aggregation`."""

from llvm_ir.scripts.run_aggregation import main


if __name__ == "__main__":
    raise SystemExit(main())
