"""Exhaustive-ish single-benchmark search for tensorflow-v0_1985.

Combines: pass pre-validation, seeded known-good sequences, large-budget MCTS,
and local search (extension + ablation hill-climbing), all sharing one prefix
cache. Reports the best measured pass sequence (smallest .text best-prefix).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from llvm_ir.heuristics.translation_unit.tu_mcts import (
    BudgetExhausted,
    MCTSConfig,
    MeasureResult,
    run_mcts,
)
from llvm_ir.stages.function_search.pass_search import (
    LLVMCommandError,
    apply_pass_sequence,
    optimize_oz,
)
from llvm_ir.stages.translation_unit.evaluate_chunk_forest import (
    filter_passes_from_results,
    find_crashing_passes,
    group_results_by_benchmark,
)
from llvm_ir.stages.translation_unit.evaluate_topk_paths import (
    evaluate_candidate_with_prefix_cache,
    measure_text_and_instruction_count,
    measure_text_size,
)
from llvm_ir.stages.translation_unit.graph.order_graph import (
    build_pass_order_graph,
    load_function_pass_results_from_report,
)

BENCH = "tensorflow-v0_1985"
COMP = Path("runs/post_bugfix_random_20260611_120635/random/pass_search/comparison.json")
BITCODE = Path("experiments/translation_unit_bitcode/autotune_stratified_30/tensorflow-v0_1985.bc")

import os

MCTS_BUDGET = int(os.environ.get("MCTS_BUDGET", "1500"))
LOCAL_EVAL_CAP = int(os.environ.get("LOCAL_EVAL_CAP", "2500"))

# Known-good sequences observed across earlier runs (seeds).
SEEDS = [
    ["iroutliner", "separate-const-offset-from-gep", "early-cse", "ipsccp", "mergefunc"],
    ["memcpyopt", "deadargelim", "newgvn", "iroutliner", "separate-const-offset-from-gep",
     "early-cse", "ipsccp", "mergereturn", "mergefunc", "gvn", "sink"],
    ["gvn", "iroutliner", "memcpyopt", "memcpyopt", "iroutliner",
     "separate-const-offset-from-gep", "early-cse", "ipsccp", "mergefunc", "mergereturn", "gvn-hoist"],
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="study-tf1985-") as tmp:
        workdir = Path(tmp)
        results = load_function_pass_results_from_report(COMP, algorithm="random")
        fr = group_results_by_benchmark(results)[BENCH]

        log(f"prevalidating passes on TU ({len(fr)} functions)...")
        crashing = find_crashing_passes(BITCODE, fr, workdir)
        log(f"crashing passes dropped: {sorted(crashing)}")
        fr = filter_passes_from_results(fr, crashing)
        graph = build_pass_order_graph(fr, benchmark=BENCH, weight_mode="delta")
        actions = sorted(graph.nodes)

        baseline, baseline_instr = measure_text_and_instruction_count(BITCODE, workdir)
        oz_bc = workdir / "oz.bc"
        optimize_oz(BITCODE, oz_bc)
        oz_size, oz_instr = measure_text_and_instruction_count(oz_bc, workdir)
        log(f"baseline .text={baseline}  -Oz .text={oz_size}  (-Oz delta={baseline-oz_size})")

        prefix_cache = {(): {"bc": str(BITCODE), "size": baseline, "error": ""}}

        def best_in_cache():
            best_key, best_size = (), baseline
            for key, entry in prefix_cache.items():
                if entry.get("error"):
                    continue
                if entry["size"] < best_size:
                    best_size, best_key = entry["size"], key
            return best_key, best_size

        # ---- Seeds ----
        log("measuring seeds...")
        for seed in SEEDS:
            evaluate_candidate_with_prefix_cache(list(seed), workdir, prefix_cache)
        bk, bs = best_in_cache()
        log(f"after seeds: best .text={bs} (delta={baseline-bs})  passes={list(bk)}")

        # ---- MCTS with bounded budget (shares prefix cache) ----
        mcts_start = len(prefix_cache) - 1

        def measure(key):
            cached = prefix_cache.get(key)
            if cached is not None:
                return MeasureResult(cached["size"], cached["error"])
            # Budget = real misses spent *during the MCTS phase only*.
            if (len(prefix_cache) - 1) - mcts_start >= MCTS_BUDGET:
                raise BudgetExhausted()
            parent = prefix_cache[key[:-1]]
            if parent["error"]:
                prefix_cache[key] = {"bc": parent["bc"], "size": parent["size"], "error": parent["error"]}
                return MeasureResult(None, parent["error"])
            out = workdir / f"m_{len(prefix_cache):06d}.bc"
            try:
                apply_pass_sequence(Path(parent["bc"]), [key[-1]], out)
                size = measure_text_size(out, workdir)
            except LLVMCommandError as exc:
                prefix_cache[key] = {"bc": parent["bc"], "size": parent["size"], "error": str(exc)}
                return MeasureResult(None, str(exc))
            prefix_cache[key] = {"bc": str(out), "size": size, "error": ""}
            return MeasureResult(size, "")

        def prior_fn(prefix):
            if not prefix:
                return {a: float(graph.start_counts.get(a, 0)) for a in actions}
            return {a: float(graph.edge_weight(prefix[-1], a)) for a in actions}

        log(f"running MCTS (budget {MCTS_BUDGET})...")
        run_mcts(baseline_size=baseline, prior_fn=prior_fn, measure=measure,
                 config=MCTSConfig(max_length=14, c_puct=2.0, rollout_length=4, branching=10, seed=7))
        bk, bs = best_in_cache()
        log(f"after MCTS ({len(prefix_cache)-1-mcts_start} evals): best .text={bs} (delta={baseline-bs})  passes={list(bk)}")

        # ---- Local search: extension + ablation hill-climbing ----
        log("local search (extension + ablation)...")
        start_evals = len(prefix_cache) - 1
        while (len(prefix_cache) - 1) - start_evals < LOCAL_EVAL_CAP:
            best_key, best_size = best_in_cache()
            improved = False
            # Extension: append each candidate pass.
            for p in actions:
                if (len(prefix_cache) - 1) - start_evals >= LOCAL_EVAL_CAP:
                    break
                evaluate_candidate_with_prefix_cache(list(best_key) + [p], workdir, prefix_cache)
            nk, ns = best_in_cache()
            if ns < best_size:
                improved = True
            # Ablation: drop each position from the current best.
            best_key, best_size = best_in_cache()
            for i in range(len(best_key)):
                if (len(prefix_cache) - 1) - start_evals >= LOCAL_EVAL_CAP:
                    break
                cand = list(best_key[:i]) + list(best_key[i + 1:])
                if cand:
                    evaluate_candidate_with_prefix_cache(cand, workdir, prefix_cache)
            nk, ns = best_in_cache()
            if ns < best_size:
                improved = True
            log(f"  local round: best .text={ns} (delta={baseline-ns})  evals={len(prefix_cache)-1}")
            if not improved:
                break

        # ---- Final report ----
        best_key, best_size = best_in_cache()
        # Instruction count for the winning sequence.
        cur = BITCODE
        best_instr = baseline_instr
        for i, p in enumerate(best_key, 1):
            out = workdir / f"fin_{i}.bc"
            apply_pass_sequence(cur, [p], out)
            _, instr = measure_text_and_instruction_count(out, workdir)
            cur = out
            best_instr = instr
        report = {
            "benchmark": BENCH,
            "baseline_text": baseline,
            "oz_text": oz_size,
            "oz_delta": baseline - oz_size,
            "best_text": best_size,
            "best_delta": baseline - best_size,
            "improvement_vs_oz_bytes": (baseline - best_size) - (baseline - oz_size),
            "best_passes": list(best_key),
            "best_text_pct": 100.0 * (baseline - best_size) / baseline,
            "baseline_instr": baseline_instr,
            "oz_instr": oz_instr,
            "best_instr": best_instr,
            "total_real_evals": len(prefix_cache) - 1,
            "crashing_passes": sorted(crashing),
        }
        Path("runs/study_tf1985.json").parent.mkdir(parents=True, exist_ok=True)
        Path("runs/study_tf1985.json").write_text(json.dumps(report, indent=2) + "\n")
        log("DONE")
        print(json.dumps(report, indent=2), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
