"""Stage the dataset-builder cbench per-function .bc into LLVM_IR and reconstruct
whole-TU bitcode via llvm-link (Variant B: no compiler_gym needed).

- per-function .bc -> datasets/cbench_v1_functions_bc/cbench-v1_<orig>.bc
  (suite prefix so benchmark_id_from_function_name groups correctly)
- per-benchmark TU -> experiments/translation_unit_bitcode/cbench_v1/cbench-v1_<bench>.bc
  (llvm-link of that benchmark's per-function .bc; defines top-20% funcs, others
  remain external declarations)
"""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

SRC = Path.home() / "diplom/dataset-builder/input_functions_bc"
DST_FUNCS = Path("datasets/cbench_v1_functions_bc")
DST_TU = Path("experiments/translation_unit_bitcode/cbench_v1")
# Full per-benchmark bitcode as served by CompilerGym (preferred TU source).
SITE_DATA = (
    Path.home()
    / ".local/share/compiler_gym/llvm-v0/benchmark/cbench-v1/contents/cBench-v1"
)


def main() -> int:
    DST_FUNCS.mkdir(parents=True, exist_ok=True)
    DST_TU.mkdir(parents=True, exist_ok=True)
    by_bench: dict[str, list[Path]] = defaultdict(list)

    n = 0
    for bc in sorted(SRC.glob("*.bc")):
        bench = bc.stem.split("_", 1)[0]
        dst = DST_FUNCS / f"cbench-v1_{bc.name}"
        shutil.copyfile(bc, dst)
        by_bench[bench].append(bc)
        n += 1
    print(f"staged {n} per-function .bc into {DST_FUNCS} ({len(by_bench)} benchmarks)")

    for bench, files in sorted(by_bench.items()):
        out = DST_TU / f"cbench-v1_{bench}.bc"
        site = SITE_DATA / f"{bench}.bc"
        if site.exists():
            # Preferred: the real full-benchmark bitcode (all functions), matching
            # the autotune setup (per-function dataset = top-20%, TU = full module).
            shutil.copyfile(site, out)
            print(f"  copied cbench-v1_{bench}.bc  (full TU from site-data, {out.stat().st_size} bytes)")
            continue
        # Fallback: reconstruct from top-20% per-function .bc via llvm-link.
        cmd = ["llvm-link", "-o", str(out), *[str(f) for f in files]]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  [FAIL] {bench} ({len(files)} funcs): {proc.stderr.strip()[:160]}")
            continue
        print(f"  linked cbench-v1_{bench}.bc  (llvm-link top-20%, {out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
