"""Backward-compatible entrypoint for function pass-search CLI."""

from .stages.function_search.pass_search import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())  # type: ignore[name-defined]

