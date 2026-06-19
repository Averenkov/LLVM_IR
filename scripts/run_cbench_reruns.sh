#!/usr/bin/env bash
# Finish the cbench reruns SEQUENTIALLY. ghostscript (4.7MB, ~31s/eval) is
# isolated into its own run with a TRIMMED budget (max-size-scale 1.0 + fewer
# candidates) so it does not dominate wall-clock or fill /tmp; everything else
# keeps the full budget so it merges fairly with the original runs.
# cycle_breaking + flow_paths are already complete (23/23) and skipped.
set -u

P=runs/cbench_random_20260617_225213/pass_search
BITS=experiments/translation_unit_bitcode/cbench_v1
ALGO=random
JOBS=8
LOG=runs/cbench_reruns_driver.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ts()  { date +%Y%m%d_%H%M%S; }
disk_free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
guard() { local f; f=$(disk_free_gb); if [ "$f" -lt 8 ]; then log "FATAL: only ${f}G free — abort."; exit 1; fi; }

run() {  # run <name> <module> <args...>
  guard
  local name="$1"; shift
  local mod="$1"; shift
  local run="runs/cbench_rerun_${name}_$(ts)"; mkdir -p "$run"
  log ">>> $name -> $run (free $(disk_free_gb)G)"
  PYTHONPATH=src python3 -m "$mod" "$@" \
    --bitcode-dir "$BITS" --output "$run/tu_eval.json" \
    --jobs "$JOBS" --measure-instructions > "$run/run.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$run/tu_eval.json" ]; then
    log "<<< $name DONE ($run), free $(disk_free_gb)G"
  else
    log "<<< $name FAILED rc=$rc (see $run/run.log)"
  fi
  rm -rf /tmp/llvm-ir-* 2>/dev/null
}

# ---------- ghostscript-only, TRIMMED budget (cheap, runs first) ----------
run measured_superpath_gs llvm_ir.stages.translation_unit.evaluate_measured_superpath \
  --comparison "$P/cmp_gs_measured_superpath.json" --algorithm "$ALGO" \
  --gen-budget 400 --select-count 60 --segment-nodes 4 --prune-percent 20 \
  --edge-samples 80 --path-budget 80 --super-lengths 2,3,4,5 \
  --learn-alpha 1.0 --super-beam 48 --super-max-length 20 --seed 7 \
  --size-ref 40 --max-size-scale 1.0 --min-budget 16 --diverse-select \
  --epsilon-edge 0.2 --vertex-weight 1.0 --prevalidate-passes

run strong_links_gs llvm_ir.stages.translation_unit.evaluate_strong_links \
  --comparison "$P/cmp_gs_strong_links.json" --algorithm "$ALGO" \
  --min-support 2 --max-links 60 --add-singles --beam 12 --concat-cap 16 \
  --small-threshold 20 --small-beam 64 --small-concat-cap 0 --tiny-threshold 6 \
  --exhaustive-cap 5000 --max-length 16 --prevalidate-passes

run bucket_dag_teacher_gs llvm_ir.stages.translation_unit.evaluate_bucket_dag_teacher \
  --comparison "$P/cmp_gs_bucket_dag_teacher.json" --algorithm "$ALGO" \
  --top-k 120 --waves 2 --chunk-size 4 --max-length 12 --min-edge-weight 1 \
  --prevalidate-passes

run random_walk_1000_gs llvm_ir.stages.translation_unit.evaluate_topk_paths \
  --graph "$P/graph_gs_random_walk_1000.json" --heuristic random_walk_topk \
  --top-k 200 --top-starts 10 --paths-per-start 10 --max-length 12 \
  --min-edge-weight 1 --random-walks 10000 --random-seed 7 --exhaustive-length 6 \
  --segment-top-k 40 --segment-min-length 4 --segment-max-length 6 \
  --superpath-beam-factor 5 --superpath-max-candidates 20000 \
  --superpath-min-segment-delta 1 --superpath-max-overlap 1 \
  --segment-max-jaccard 0.75 --tiny-graph-threshold 4 \
  --instruction-measurement deferred --chunk-size 4

run segment_tree_gs llvm_ir.stages.translation_unit.evaluate_segment_tree \
  --comparison "$P/cmp_gs_segment_tree.json" --algorithm "$ALGO" \
  --select-count 80 --gen-budget 500 --segment-nodes 4 --prune-percent 20 \
  --max-length 16 --size-ref 40 --max-size-scale 1.0 --min-budget 16 \
  --diverse-select --prevalidate-passes

# ---------- medium modules (jpeg/tiff/lame), FULL budget ----------
run random_walk_1000_others llvm_ir.stages.translation_unit.evaluate_topk_paths \
  --graph "$P/graph_others_random_walk_1000.json" --heuristic random_walk_topk \
  --top-k 1000 --top-starts 10 --paths-per-start 10 --max-length 12 \
  --min-edge-weight 1 --random-walks 50000 --random-seed 7 --exhaustive-length 6 \
  --segment-top-k 100 --segment-min-length 4 --segment-max-length 6 \
  --superpath-beam-factor 5 --superpath-max-candidates 100000 \
  --superpath-min-segment-delta 1 --superpath-max-overlap 1 \
  --segment-max-jaccard 0.75 --tiny-graph-threshold 4 \
  --instruction-measurement deferred --chunk-size 4

run segment_tree_others llvm_ir.stages.translation_unit.evaluate_segment_tree \
  --comparison "$P/cmp_others_segment_tree.json" --algorithm "$ALGO" \
  --select-count 500 --gen-budget 4000 --segment-nodes 4 --prune-percent 20 \
  --max-length 16 --size-ref 40 --max-size-scale 3.0 --min-budget 16 \
  --diverse-select --prevalidate-passes

log "ALL CBENCH RERUNS COMPLETE"
