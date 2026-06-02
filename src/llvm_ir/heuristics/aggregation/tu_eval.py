"""Translation-unit evaluation with a disk cache."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalResult:
    text_size: int | None
    success: bool
    error: str | None
    elapsed_s: float


class TUEvaluator:
    def __init__(
        self,
        tu_path: str | Path,
        opt_bin: str = "opt",
        llc_bin: str = "llc",
        size_bin: str = "llvm-size",
        cache_dir: str | Path = ".aggregation_cache",
    ) -> None:
        self.tu_path = Path(tu_path)
        self.opt_bin = opt_bin
        self.llc_bin = llc_bin
        self.size_bin = size_bin
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tu_hash = self._file_hash(self.tu_path) if self.tu_path.exists() else "missing"

    def evaluate(self, passes: list[str], timeout: float = 60.0) -> EvalResult:
        key = self._cache_key(passes)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return EvalResult(**payload)

        start = time.monotonic()
        if not self.tu_path.exists():
            result = EvalResult(None, False, f"TU path does not exist: {self.tu_path}", 0.0)
            self._write_cache(cache_path, result)
            return result

        with tempfile.TemporaryDirectory(prefix="llvm-ir-agg-eval-") as tmp_str:
            tmp = Path(tmp_str)
            optimized = tmp / "optimized.bc"
            obj_path = tmp / "optimized.o"
            try:
                if passes:
                    opt_cmd = [
                        self.opt_bin,
                        f"-passes={','.join(passes)}",
                        str(self.tu_path),
                        "-o",
                        str(optimized),
                    ]
                    self._run(opt_cmd, timeout)
                else:
                    shutil.copyfile(self.tu_path, optimized)
                self._run(
                    [self.llc_bin, "-filetype=obj", str(optimized), "-o", str(obj_path)],
                    timeout,
                )
                size = self._run([self.size_bin, str(obj_path)], timeout)
                text_size = self._parse_text_size(size.stdout)
                result = EvalResult(text_size, True, None, time.monotonic() - start)
            except Exception as exc:  # noqa: BLE001
                result = EvalResult(None, False, str(exc), time.monotonic() - start)
        self._write_cache(cache_path, result)
        return result

    def _run(self, cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    def _cache_key(self, passes: list[str]) -> str:
        payload = self.tu_hash + "\n" + ",".join(passes)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _write_cache(self, cache_path: Path, result: EvalResult) -> None:
        cache_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _parse_text_size(output: str) -> int:
        lines = output.strip().splitlines()
        if len(lines) < 2:
            raise ValueError(f"unexpected llvm-size output: {output!r}")
        return int(lines[1].split()[0])

