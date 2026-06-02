#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-nightly_big_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/$RUN_ID}"
LOG="$RUN_DIR/nightly.log"

DATASET_DIR="${DATASET_DIR:-datasets/autotune_stratified_30_functions_bc}"
BITCODE_DIR="${BITCODE_DIR:-experiments/translation_unit_bitcode/autotune_stratified_30}"
SITE_DATA="${SITE_DATA:-/home/vladimir/diplom/diplom_LLVM_IR/.compiler_gym/site_data}"

SEED="${SEED:-7}"
LIMIT="${LIMIT:-0}"
STEPS="${STEPS:-8}"
ITERATIONS="${ITERATIONS:-6}"
CANDIDATES="${CANDIDATES:-32}"
JOBS="${JOBS:-8}"

MAX_LENGTH="${MAX_LENGTH:-12}"
BEAM_WIDTH="${BEAM_WIDTH:-16}"

RUN_TU_EVAL="${RUN_TU_EVAL:-1}"
RUN_AGGREGATION_GRAPH_ONLY="${RUN_AGGREGATION_GRAPH_ONLY:-1}"
RUN_AGGREGATION_TU_EVAL="${RUN_AGGREGATION_TU_EVAL:-1}"
AGG_TOP_K="${AGG_TOP_K:-4}"
AGG_INCLUDE_TOPK="${AGG_INCLUDE_TOPK:-0}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export COMPILER_GYM_CACHE="${COMPILER_GYM_CACHE:-$ROOT_DIR/.cache/compiler_gym/cache}"
export COMPILER_GYM_TRANSIENT_CACHE="${COMPILER_GYM_TRANSIENT_CACHE:-$ROOT_DIR/.cache/compiler_gym/transient}"

mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

run_cmd() {
  log "RUN: $*"
  set +e
  "$@" 2>&1 | tee -a "$LOG"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAILED with status $status: $*"
    exit "$status"
  fi
}

write_run_config() {
  cat > "$RUN_DIR/config.txt" <<EOF
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
DATASET_DIR=$DATASET_DIR
BITCODE_DIR=$BITCODE_DIR
SITE_DATA=$SITE_DATA
SEED=$SEED
LIMIT=$LIMIT
STEPS=$STEPS
ITERATIONS=$ITERATIONS
CANDIDATES=$CANDIDATES
JOBS=$JOBS
MAX_LENGTH=$MAX_LENGTH
BEAM_WIDTH=$BEAM_WIDTH
RUN_TU_EVAL=$RUN_TU_EVAL
RUN_AGGREGATION_GRAPH_ONLY=$RUN_AGGREGATION_GRAPH_ONLY
RUN_AGGREGATION_TU_EVAL=$RUN_AGGREGATION_TU_EVAL
AGG_TOP_K=$AGG_TOP_K
AGG_INCLUDE_TOPK=$AGG_INCLUDE_TOPK
PYTHONPATH=$PYTHONPATH
EOF
}

run_algorithm() {
  local algorithm="$1"
  local algorithm_dir="$RUN_DIR/$algorithm"
  local pass_dir="$algorithm_dir/pass_search"
  local graph_dir="$algorithm_dir/translation_unit_graphs"
  local heur_dir="$algorithm_dir/translation_unit_heuristics/delta"
  local eval_dir="$algorithm_dir/translation_unit_eval/delta"
  local aggregation_dir="$algorithm_dir/aggregation_graph_only"
  local aggregation_paths_dir="$algorithm_dir/aggregation_heuristics"
  local aggregation_eval_dir="$algorithm_dir/aggregation_tu_eval"

  mkdir -p \
    "$pass_dir" \
    "$graph_dir" \
    "$heur_dir" \
    "$eval_dir" \
    "$aggregation_dir" \
    "$aggregation_paths_dir" \
    "$aggregation_eval_dir"

  log "=== $algorithm: per-function pass search ==="
  run_cmd python3 -m llvm_ir.stages.function_search.pass_search \
    --dataset-dir "$DATASET_DIR" \
    --algorithm "$algorithm" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --iterations "$ITERATIONS" \
    --candidates "$CANDIDATES" \
    --jobs "$JOBS" \
    --output-dir "$pass_dir"

  log "=== $algorithm: build count graph ==="
  run_cmd python3 -m llvm_ir.stages.translation_unit.order_graph \
    --input "$pass_dir/comparison.json" \
    --algorithm "$algorithm" \
    --weight-mode count \
    --output "$graph_dir/order_graphs_count.json"

  log "=== $algorithm: build delta graph ==="
  run_cmd python3 -m llvm_ir.stages.translation_unit.order_graph \
    --input "$pass_dir/comparison.json" \
    --algorithm "$algorithm" \
    --weight-mode delta \
    --output "$graph_dir/order_graphs_delta.json"

  log "=== $algorithm: translation-unit graph heuristics ==="
  run_cmd python3 -m llvm_ir.stages.translation_unit.path_heuristics \
    --input "$graph_dir/order_graphs_delta.json" \
    --heuristics all \
    --max-length "$MAX_LENGTH" \
    --beam-width "$BEAM_WIDTH" \
    --output "$heur_dir/all_heuristics.json"

  if [[ "$RUN_TU_EVAL" == "1" ]]; then
    log "=== $algorithm: whole translation-unit evaluation ==="
    run_cmd python3 -m llvm_ir.stages.translation_unit.evaluate \
      --input "$heur_dir/all_heuristics.json" \
      --site-data "$SITE_DATA" \
      --bitcode-dir "$BITCODE_DIR" \
      --output "$eval_dir/tu_eval_all_heuristics.json"
  else
    log "=== $algorithm: skip whole TU evaluation (RUN_TU_EVAL=$RUN_TU_EVAL) ==="
  fi

  if [[ "$RUN_AGGREGATION_GRAPH_ONLY" == "1" ]]; then
    log "=== $algorithm: new aggregation package graph-only comparison ==="
    run_cmd python3 -m scripts.run_aggregation \
      --dataset "$pass_dir/comparison.json" \
      --algorithm "$algorithm" \
      --all \
      --compare \
      --no-tu-eval \
      --output "$aggregation_dir"
  else
    log "=== $algorithm: skip aggregation graph-only comparison (RUN_AGGREGATION_GRAPH_ONLY=$RUN_AGGREGATION_GRAPH_ONLY) ==="
  fi

  log "=== $algorithm: export all 11 aggregation heuristic paths ==="
  local include_topk_args=()
  if [[ "$AGG_INCLUDE_TOPK" == "1" ]]; then
    include_topk_args+=(--include-topk)
  fi
  run_cmd python3 -m llvm_ir.heuristics.aggregation.export_paths \
    --input "$pass_dir/comparison.json" \
    --algorithm "$algorithm" \
    --heuristics all \
    --max-length "$MAX_LENGTH" \
    --beam-width "$BEAM_WIDTH" \
    --top-k "$AGG_TOP_K" \
    "${include_topk_args[@]}" \
    --output "$aggregation_paths_dir/all_aggregation_heuristics.json"

  if [[ "$RUN_AGGREGATION_TU_EVAL" == "1" ]]; then
    log "=== $algorithm: whole TU evaluation for all 11 aggregation heuristics ==="
    run_cmd python3 -m llvm_ir.stages.translation_unit.evaluate \
      --input "$aggregation_paths_dir/all_aggregation_heuristics.json" \
      --site-data "$SITE_DATA" \
      --bitcode-dir "$BITCODE_DIR" \
      --output "$aggregation_eval_dir/tu_eval_all_aggregation_heuristics.json"
  else
    log "=== $algorithm: skip aggregation whole TU evaluation (RUN_AGGREGATION_TU_EVAL=$RUN_AGGREGATION_TU_EVAL) ==="
  fi
}

write_final_summary() {
  log "=== write final summary ==="
  python3 - "$RUN_DIR" <<'PY' | tee -a "$LOG"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary = {"run_dir": str(run_dir), "algorithms": {}}
for algorithm in ("cem", "random"):
    root = run_dir / algorithm
    data = {}
    pass_report = root / "pass_search" / "comparison.json"
    heur_report = root / "translation_unit_heuristics" / "delta" / "all_heuristics.json"
    eval_report = root / "translation_unit_eval" / "delta" / "tu_eval_all_heuristics.json"
    aggregation_report = root / "aggregation_graph_only" / "metrics.json"
    aggregation_paths_report = root / "aggregation_heuristics" / "all_aggregation_heuristics.json"
    aggregation_eval_report = root / "aggregation_tu_eval" / "tu_eval_all_aggregation_heuristics.json"
    if pass_report.exists():
        payload = json.loads(pass_report.read_text())
        data["pass_search_summary"] = payload.get("summary", {})
        data["pass_search_config"] = payload.get("config", {})
    if heur_report.exists():
        payload = json.loads(heur_report.read_text())
        data["graph_heuristics_summary"] = payload.get("summary", {})
    if eval_report.exists():
        payload = json.loads(eval_report.read_text())
        data["tu_eval_summary"] = payload.get("summary", {})
    if aggregation_report.exists():
        payload = json.loads(aggregation_report.read_text())
        data["aggregation_graph_only_rows"] = payload.get("rows", [])
    if aggregation_paths_report.exists():
        payload = json.loads(aggregation_paths_report.read_text())
        data["aggregation_paths_summary"] = payload.get("summary", {})
    if aggregation_eval_report.exists():
        payload = json.loads(aggregation_eval_report.read_text())
        data["aggregation_tu_eval_summary"] = payload.get("summary", {})
    summary["algorithms"][algorithm] = data

out = run_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
print(json.dumps({"summary": str(out)}, indent=2))
PY
}

log "Nightly big run started"
write_run_config

run_algorithm cem
run_algorithm random
write_final_summary

log "Nightly big run finished successfully"
log "Run directory: $RUN_DIR"
