"""Backward-compatible entrypoint for dataset stage."""

from .stages.dataset.builder import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())  # type: ignore[name-defined]

