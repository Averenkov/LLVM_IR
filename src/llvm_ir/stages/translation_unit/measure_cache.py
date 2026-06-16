"""Persistent cross-run cache of TU measurements (`.text` size, instruction count).

Repeated runs of the TU heuristics measure heavily overlapping pass-sequence
prefixes on the same translation units. Codegen (``llc``) dominates each
measurement, so caching ``(TU, pass sequence) -> (size, instructions)`` across
runs lets a later run skip ``llc`` for any prefix already measured.

One JSON file per translation unit (keyed by a content hash of its bitcode) maps
a comma-joined pass sequence to its measured values. Because each benchmark runs
in its own worker process and owns a distinct TU file, parallel ``--jobs`` runs
never write the same cache file, so no locking is needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def tu_key(bitcode_path: Path) -> str:
    digest = hashlib.sha1()
    with open(bitcode_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class MeasureCache:
    """Per-TU persistent map: pass-sequence -> {"size", "instr"}."""

    def __init__(self, cache_dir: Path | str | None, bitcode_path: Path) -> None:
        self.enabled = cache_dir is not None
        self.path: Path | None = None
        self.data: dict[str, dict[str, int]] = {}
        self.dirty = False
        if self.enabled:
            self.path = Path(cache_dir) / f"{tu_key(bitcode_path)}.json"
            if self.path.exists():
                try:
                    self.data = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - a corrupt cache is just ignored
                    self.data = {}

    @staticmethod
    def _key(passes: tuple[str, ...] | list[str]) -> str:
        return ",".join(passes)

    def get_size(self, passes) -> int | None:
        entry = self.data.get(self._key(passes))
        return int(entry["size"]) if entry and "size" in entry else None

    def get_instr(self, passes) -> int | None:
        entry = self.data.get(self._key(passes))
        if entry and entry.get("instr") is not None:
            return int(entry["instr"])
        return None

    def put(self, passes, size: int, instr: int | None = None) -> None:
        if not self.enabled:
            return
        key = self._key(passes)
        entry = self.data.get(key) or {}
        entry["size"] = int(size)
        if instr is not None:
            entry["instr"] = int(instr)
        self.data[key] = entry
        self.dirty = True

    def flush(self) -> None:
        if not (self.enabled and self.dirty and self.path is not None):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)
        self.dirty = False
