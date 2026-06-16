"""Bucket-layered DAG heuristic for pass-order graphs.

The construction eliminates cycles *structurally* instead of by removing edges:

1. each node is weighted by the sum of its outgoing edge weights;
2. nodes are sorted by that weight, descending;
3. the sorted nodes are split into equal-size buckets, numbered 0, 1, 2, ...;
4. only edges going from a smaller-numbered bucket to a larger-numbered bucket
   are kept (within-bucket and backward edges are dropped) -- this is acyclic by
   construction, since every kept edge moves strictly forward in bucket index;
5. a dynamic program enumerates the top-K maximum-weight paths in the resulting
   DAG, which are then measured on the whole TU.

The bucket order itself is a valid topological order (lower bucket index = earlier
position), so the existing max-weight-path DP is reused directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from llvm_ir.heuristics.translation_unit.cycle_breaking_max_path import _max_weight_paths
from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph


@dataclass(frozen=True)
class BucketDagConfig:
    chunk_size: int = 4
    max_length: int = 12
    min_edge_weight: int = 1


def build_teacher_graph(
    graph: PassOrderGraph,
    measured_edge_means: dict[tuple[str, str], float],
) -> PassOrderGraph:
    """Re-weight a pass-order graph using measured whole-TU edge marginals.

    ``measured_edge_means[(u, v)]`` is the average measured ``.text`` byte gain
    observed when pass ``v`` was applied right after pass ``u`` during a real TU
    run (the "teacher" signal). Measured edges take that value directly; edges
    that were never exercised keep their function-level weight rescaled onto the
    measured byte scale, so both live on a comparable scale. Non-positive teacher
    weights are clamped to ``0`` (a forward transition that did not help is not
    worth ranking on), which the bucket DAG's ``min_edge_weight`` filter prunes.
    """
    positive = [value for value in measured_edge_means.values() if value > 0]
    measured_fn_weights = [
        graph.edge_weight(source, target)
        for (source, target) in measured_edge_means
        if measured_edge_means[(source, target)] > 0
    ]
    denom = sum(measured_fn_weights) / len(measured_fn_weights) if measured_fn_weights else 0.0
    scale = (sum(positive) / len(positive)) / denom if positive and denom > 0 else 1.0

    teacher = PassOrderGraph(benchmark=graph.benchmark, weight_mode="delta")
    teacher.nodes = set(graph.nodes)
    teacher.start_counts = dict(graph.start_counts)
    for (source, target), weight in graph.edge_counts.items():
        if source == target:
            continue
        if (source, target) in measured_edge_means:
            value = measured_edge_means[(source, target)]
        else:
            value = weight * scale
        teacher.edge_counts[(source, target)] = max(0, round(value))
    return teacher


def node_out_weights(graph: PassOrderGraph) -> dict[str, int]:
    """Sum of outgoing edge weights per node (self-loops excluded)."""
    weights = {node: 0 for node in graph.nodes}
    for (source, target), weight in graph.edge_counts.items():
        if source != target and source in weights:
            weights[source] += weight
    return weights


def bucket_layer_top_paths(
    graph: PassOrderGraph,
    *,
    config: BucketDagConfig | None = None,
    top_k: int = 250,
) -> list[list[str]]:
    if config is None:
        config = BucketDagConfig()
    nodes = sorted(graph.nodes)
    if not nodes:
        return []

    out_weight = node_out_weights(graph)
    # Step 2: rank nodes by out-weight (descending), lexicographic tie-break.
    order = sorted(nodes, key=lambda node: (-out_weight[node], node))

    # Step 3: equal-size buckets, numbered by position in the ranking.
    chunk_size = max(1, config.chunk_size)
    bucket_of = {node: index // chunk_size for index, node in enumerate(order)}

    # Step 4: keep only edges from a smaller bucket to a larger bucket.
    edges = {
        (source, target): weight
        for (source, target), weight in graph.edge_counts.items()
        if source != target
        and weight >= config.min_edge_weight
        and source in bucket_of
        and target in bucket_of
        and bucket_of[source] < bucket_of[target]
    }

    # Step 5: top-K max-weight paths via DP (order is a valid topological order).
    return _max_weight_paths(
        order,
        edges,
        order,
        max_length=config.max_length,
        top_k=max(1, top_k),
    )
