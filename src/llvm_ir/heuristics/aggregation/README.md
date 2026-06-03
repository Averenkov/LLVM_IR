# Aggregation Heuristics

This package aggregates per-function LLVM pass sequences into translation-unit
pass sequences.

## Heuristics

| Name | Summary | Key Defaults |
|---|---|---|
| `wfas_eades` | Weighted feedback-arc Eades ordering | `alpha=0.5` |
| `scc_ordering` | SCC condensation plus intra-SCC gain ordering | `alpha=0.5` |
| `pagerank` | Personalized PageRank on direct and reverse graphs | `damping=0.85` |
| `position_median` | Weighted median normalized pass position | `tau=0.35` |
| `hpp` | Harmful-pass pruning plus Eades ordering | `beta=1.0`, `lambda_=0.1` |
| `beam_diversity` | Top-K beam search with Jaccard diversity | `beam_width=16`, `top_k=4` |
| `cluster_aware` | Cluster functions, order each cluster, merge ranks | `n_clusters=4` |
| `ilp_arrangement` | Optional PuLP LP/ILP arrangement, Eades fallback | optional dependency |
| `markov_hitting` | Markov stationary/hitting-style ordering | `max_iter=200` |
| `voting_ensemble` | Borda aggregation of other heuristics | 5 default voters |
| `hpp_eades_topk` | Main composite: HPP + Eades + diverse Top-K | `top_k=4` |

## CLI

Graph-only comparison:

```bash
PYTHONPATH=src python3 -m llvm_ir.scripts.run_aggregation \
  --dataset experiments/pass_search_compare/cem_shifts_all_seed7/comparison.json \
  --all \
  --compare \
  --no-tu-eval \
  --output runs/aggregation_demo
```

Single heuristic:

```bash
PYTHONPATH=src python3 -m llvm_ir.scripts.run_aggregation \
  --dataset experiments/pass_search_compare/cem_shifts_all_seed7/comparison.json \
  --heuristic hpp_eades_topk \
  --no-tu-eval \
  --output runs/hpp_eades_topk_demo
```

The output directory contains `metrics.json` and `metrics.csv` with columns:

`heuristic, graph_score_delta, fas_weight, coverage, final_size,
best_prefix_size, delta_vs_oz, norm_best, fail_rate, beat_oz, tu_evals,
wallclock_s`.

TU validation is enabled by passing `--tu-path`, `--baseline-text-size`, and
`--oz-text-size`. Results are cached by SHA-1 of `tu_hash + passes`.
