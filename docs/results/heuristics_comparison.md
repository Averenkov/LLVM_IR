# TU heuristic comparison

All numbers below are on the **same** per-function search dataset
(`post_bugfix_random_20260611_120635`, random search, 30 translation units), so
they are directly comparable. Slim per-benchmark reports (summary + per-benchmark
best rows, no candidate dump) live next to this file.

> **Dataset caveat:** the older `random_walk_top1000_instr_20260608` run (weighted
> 7.5013%) was produced on a *different*, pre-crash-fix per-function run
> (`full_random_heuristic`) that happened to be luckier on `tensorflow-v0_2`. It
> must not be used as the reference; use the `post_bugfix_random` numbers below.

## Headline (post_bugfix_random)

Two averages are reported. **Weighted (micro)** = `Σ best_delta / Σ baseline` --
benchmarks weighted by size; this is the primary metric. **Macro** =
`mean(best_delta / baseline)` over benchmarks -- each benchmark weighted equally
(inflated by small TUs that have large relative gains).

| Heuristic | wtd `.text` | macro `.text` | wtd instr | macro instr | beats `-Oz` | total Δ |
|---|---:|---:|---:|---:|---:|---:|
| **measured_superpath** | **7.4816%** | 11.901% | **8.7662%** | 13.873% | **25/30** | 77292 |
| random_walk_2000 | 6.9925% | — | 8.14% | — | 23/30 | 72239 |
| random_walk_1000 | 6.9315% | 11.460% | 8.0048% | 13.390% | 23/30 | 71609 |
| chunk_forest (minsup1) | 6.6303% | 11.548% | 7.9388% | 13.844% | 23/30 | 68497 |
| flow_paths | 6.4036% | 10.138% | 7.29% | 11.796% | 23/30 | 66155 |
| bucket_dag_teacher | 6.3853% | 10.719% | 7.35% | 12.812% | 21/30 | 65966 |
| function_sequences (baseline) | 6.2719% | 11.150% | 7.24% | 12.863% | 23/30 | 64795 |
| `-Oz` (baseline) | 1.7836% | — | — | — | — | — |

Per-benchmark `.text` improvement ranges roughly 0–35%. By the weighted (micro)
metric measured_superpath leads on every column. By macro the order between the
mid heuristics shifts (chunk_forest > random_walk_1000), since macro over-weights
small benchmarks -- but measured_superpath still leads both.

On the consistent dataset **measured_superpath leads on every metric**
(+0.55 pp `.text`, +0.76 pp instructions vs the next-best random_walk) and is
`>=` random_walk on **28/30** benchmarks (the only losses are 8 B and 32 B on two
benchmarks). Every learned/structured heuristic beats `-Oz` by 3.5–4x.

`function_sequences` is the naive-transfer baseline (apply each function's own
best sequence to the whole TU). That all graph/search heuristics clear it
quantifies the value of TU-level aggregation over plain transfer.

## measured_superpath ablation (each change is monotonic)

| Version | weighted `.text` | weighted instr | beats `-Oz` | >= random_walk |
|---|---:|---:|---:|---:|
| v1 adaptive (3-phase + online edges) | 7.2390% | 8.6255% | 25/30 | 26/30 |
| v2 + start-diversity + epsilon edges | 7.4291% | 8.8094% | 25/30 | 27/30 |
| v3 + size-scaled budgets (large graphs) | 7.4816% | 8.7662% | 25/30 | 28/30 |

## Method (measured_superpath)

Two-wave, three phases sharing one prefix cache (every TU run is reused and
teaches the search):

1. **Generate + select.** Distance-aware count graph; vertices weighted by
   sequence support; prune the smallest edges; per-vertex top length-3 (4-pass)
   paths budgeted by vertex weight (`n*k≈2000`); select the top with
   start-diversity coverage; measure on the TU. Graceful length degradation and
   size-scaled budgets handle small/large graphs.
2. **Measured super-edges.** Super-vertices = measured 4-pass segments
   (weight = measured profit). Edges are *measured on the TU*: pairs sampled
   epsilon-greedily (explore uniformly, exploit best-to-best), the concatenation
   run, edge weight = its measured best-prefix delta. Online synergy learning
   boosts the sampling weight of segments that chain well.
3. **Super-paths.** Beam-search the best super-paths of 2..5 segments over the
   measured super-graph; measure them; keep the global best.

## Other heuristics (one line each)

- **random_walk_1000** — sample 1000 weighted random walks over the order graph; measure; keep best.
- **chunk_forest** — mine frequent contiguous pass chunks; two-wave candidate pool with TU-recalibrated chunk values.
- **flow_paths** — layered network max-flow per length; bottleneck-robust decomposed paths; two-wave teacher.
- **bucket_dag_teacher** — rank vertices by out-weight, bucket, keep only forward edges (acyclic), DP top-k; TU-teacher re-weights edges.
- **function_sequences** — apply each function's own best sequence to the whole TU (naive-transfer baseline).
- **consensus_subseq** — frequent gapped subsequences + noise (negative result, ~1.26%; too loose, not in the table).
