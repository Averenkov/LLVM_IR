#!/usr/bin/env bash
# Wait for the cbench stage-2 function search to finish, then run all stage-3
# TU heuristics on the full cbench TU bitcodes, sequentially (jobs=8 each so the
# shared MeasureCache warms up and later heuristics reuse measurements).
#
# Usage: scripts/run_cbench_stage3.sh <function_search_PID> <run_root>
set -u

PID="${1:?need function_search PID}"
ROOT="${2:?need run root, e.g. runs/cbench_random_20260617_225213}"
CMP="$ROOT/pass_search/comparison.json"
GRAPH="$ROOT/pass_search/order_graphs_delta.json"
BITS="experiments/translation_unit_bitcode/cbench_v1"
ALGO="random"
JOBS=8
DRIVER_LOG="$ROOT/stage3_driver.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }
ts()  { date +%Y%m%d_%H%M%S; }

log "waiting for function search PID $PID ..."
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
log "PID $PID gone. checking comparison.json"

if [ ! -s "$CMP" ]; then
  log "FATAL: $CMP missing/empty — stage 2 did not finish cleanly. abort."
  exit 1
fi
log "comparison.json OK ($(python3 -c "import json;print(len(json.load(open('$CMP')).get('rows',[])))" 2>/dev/null) rows)"

# Build the order graph (delta weights) for the graph-driven heuristics.
log "building order graph (delta) -> $GRAPH"
PYTHONPATH=src python3 -m llvm_ir.stages.translation_unit.graph.order_graph \
  --input "$CMP" --output "$GRAPH" --algorithm "$ALGO" --weight-mode delta \
  >> "$DRIVER_LOG" 2>&1
if [ ! -s "$GRAPH" ]; then
  log "WARN: graph build failed — cycle_breaking/random_walk will be skipped."
fi

run() {  # run <name> <module> <args...>
  local name="$1"; shift
  local mod="$1"; shift
  local run="runs/cbench_${name}_$(ts)"; mkdir -p "$run"
  log ">>> $name -> $run"
  PYTHONPATH=src python3 -m "$mod" "$@" \
    --output "$run/tu_eval.json" --jobs "$JOBS" --measure-instructions \
    > "$run/run.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -s "$run/tu_eval.json" ]; then
    log "<<< $name DONE ($run)"
  else
    log "<<< $name FAILED rc=$rc (see $run/run.log)"
  fi
}

# --- comparison-driven heuristics (build their own graph internally) ---
run measured_superpath llvm_ir.stages.translation_unit.evaluate_measured_superpath \
  --comparison "$CMP" --algorithm "$ALGO" --bitcode-dir "$BITS" \
  --gen-budget 2000 --select-count 250 --segment-nodes 4 --prune-percent 20 \
  --edge-samples 250 --path-budget 250 --super-lengths 2,3,4,5 \
  --learn-alpha 1.0 --super-beam 96 --super-max-length 20 --seed 7 \
  --size-ref 40 --max-size-scale 3.0 --min-budget 16 --diverse-select \
  --epsilon-edge 0.2 --vertex-weight 1.0 --prevalidate-passes

run strong_links llvm_ir.stages.translation_unit.evaluate_strong_links \
  --comparison "$CMP" --algorithm "$ALGO" --bitcode-dir "$BITS" \
  --min-support 2 --max-links 200 --add-singles --beam 24 --concat-cap 48 \
  --small-threshold 20 --small-beam 64 --small-concat-cap 0 --tiny-threshold 6 \
  --exhaustive-cap 5000 --max-length 16 --prevalidate-passes

run segment_tree llvm_ir.stages.translation_unit.evaluate_segment_tree \
  --comparison "$CMP" --algorithm "$ALGO" --bitcode-dir "$BITS" \
  --select-count 500 --gen-budget 4000 --segment-nodes 4 --prune-percent 20 \
  --max-length 16 --size-ref 40 --max-size-scale 3.0 --min-budget 16 \
  --diverse-select --prevalidate-passes

run flow_paths llvm_ir.stages.translation_unit.evaluate_flow_paths \
  --comparison "$CMP" --algorithm "$ALGO" --bitcode-dir "$BITS" \
  --waves 2 --min-flow-length 2 --base-flow-length 4 --max-flow-length 20 \
  --per-length-top-k 20 --min-edge-weight 1 --prevalidate-passes

run bucket_dag_teacher llvm_ir.stages.translation_unit.evaluate_bucket_dag_teacher \
  --comparison "$CMP" --algorithm "$ALGO" --bitcode-dir "$BITS" \
  --top-k 500 --waves 2 --chunk-size 4 --max-length 12 --min-edge-weight 1 \
  --prevalidate-passes

# --- graph-driven heuristics (need the prebuilt --graph report) ---
if [ -s "$GRAPH" ]; then
  run cycle_breaking llvm_ir.stages.translation_unit.evaluate_topk_paths \
    --graph "$GRAPH" --bitcode-dir "$BITS" \
    --heuristic cycle_breaking_top_starts_top_paths \
    --top-k 10 --top-starts 10 --paths-per-start 10 --max-length 12 \
    --min-edge-weight 1 --random-walks 2048 --random-seed 7 --exhaustive-length 6

  run random_walk_1000 llvm_ir.stages.translation_unit.evaluate_topk_paths \
    --graph "$GRAPH" --bitcode-dir "$BITS" \
    --heuristic random_walk_topk \
    --top-k 1000 --top-starts 10 --paths-per-start 10 --max-length 12 \
    --min-edge-weight 1 --random-walks 50000 --random-seed 7 --exhaustive-length 6 \
    --segment-top-k 100 --segment-min-length 4 --segment-max-length 6 \
    --superpath-beam-factor 5 --superpath-max-candidates 100000 \
    --superpath-min-segment-delta 1 --superpath-max-overlap 1 \
    --segment-max-jaccard 0.75 --tiny-graph-threshold 4 \
    --instruction-measurement deferred --chunk-size 4
else
  log "skipped cycle_breaking + random_walk (no graph)"
fi

log "ALL STAGE-3 cbench RUNS COMPLETE"
