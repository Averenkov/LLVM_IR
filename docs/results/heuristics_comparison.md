# TU heuristic comparison

All numbers below are on the **same** per-function search dataset
(`post_bugfix_random_20260611_120635`, random search, 30 translation units) and
the same ~1000 real-eval budget, so they are directly comparable. The slim
per-benchmark reports live next to this file.

> **Dataset caveat:** the older `random_walk_top1000_instr_20260608` run (weighted
> 7.5013%) was produced on a *different* per-function run (`full_random_heuristic`,
> pre-crash-fix) that happened to be luckier on `tensorflow-v0_2`. It must not be
> used as the reference; use the `post_bugfix_random` numbers below.

## Headline (post_bugfix_random, ~1000 evals)

| Heuristic | weighted best `.text` | weighted best instr | beats `-Oz` | total best Δ |
|---|---:|---:|---:|---:|
| **measured_superpath** | **7.4816%** | **8.7662%** | **25/30** | **77292** |
| random_walk_1000 | 6.9315% | 8.0048% | 23/30 | 71609 |
| chunk_forest | 6.6303% | 7.21% | 23/30 | 68497 |

On the consistent dataset, **measured_superpath leads on every metric**
(+0.55 pp `.text`, +0.76 pp instructions vs random_walk) and is `>=` random_walk
on **28/30** benchmarks; the only losses are 8 B and 32 B on two benchmarks.

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
