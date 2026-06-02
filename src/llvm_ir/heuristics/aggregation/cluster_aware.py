"""Cluster-aware aggregation heuristic."""

from __future__ import annotations

import math
import random
from typing import Any

from .base import AggregationHeuristic, AggregationResult, Dataset, PassGraph, build_pass_graph
from .graph_utils import blended_weights, eades_order, unique_sequences


class ClusterAwareAggregation(AggregationHeuristic):
    name = "cluster_aware"
    supports_topk = True

    def aggregate(
        self,
        dataset: Dataset,
        graph: PassGraph,
        config: dict[str, Any],
    ) -> AggregationResult:
        nodes = sorted(graph.nodes)
        if not nodes:
            return AggregationResult([[]], [], 0, {})
        n_clusters = max(1, int(config.get("n_clusters", 4)))
        labels = _cluster_sequences(dataset, nodes, n_clusters, int(config.get("seed", 7)))
        cluster_sequences: list[list[str]] = []
        cluster_weights: list[float] = []
        for cluster_id in sorted(set(labels)):
            results = [
                result
                for result, label in zip(dataset.results, labels)
                if label == cluster_id
            ]
            cluster_graph = build_pass_graph(results)
            weights = blended_weights(cluster_graph, alpha=float(config.get("alpha", 0.5)))
            cluster_sequences.append(eades_order(set(cluster_graph.nodes), weights))
            cluster_weights.append(sum(max(result.delta, 0) for result in results) or 1.0)
        ranks: dict[str, float] = {node: 0.0 for node in nodes}
        for sequence, weight in zip(cluster_sequences, cluster_weights):
            rank_map = {node: index for index, node in enumerate(sequence)}
            missing_rank = len(sequence)
            for node in nodes:
                ranks[node] += weight * rank_map.get(node, missing_rank)
        sequence = sorted(nodes, key=lambda node: (ranks[node], node))
        sequences = unique_sequences([sequence] + cluster_sequences, limit=n_clusters + 1)
        return AggregationResult(
            sequences=sequences,
            chosen_sequence=sequence,
            chosen_prefix_length=len(sequence),
            extra={"clusters": len(set(labels)), "cluster_weights": cluster_weights},
        )


def _cluster_sequences(dataset: Dataset, nodes: list[str], n_clusters: int, seed: int) -> list[int]:
    vectors = [_multi_hot(result.sequence, nodes) for result in dataset.results]
    if not vectors:
        return []
    n_clusters = min(n_clusters, len(vectors))
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return _fallback_kmeans(vectors, n_clusters, seed)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(vectors)
    return [int(label) for label in labels]


def _multi_hot(sequence: list[str], nodes: list[str]) -> list[float]:
    present = set(sequence)
    return [1.0 if node in present else 0.0 for node in nodes]


def _fallback_kmeans(vectors: list[list[float]], n_clusters: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    centroids = [list(vector) for vector in rng.sample(vectors, n_clusters)]
    labels = [0 for _ in vectors]
    for _ in range(20):
        changed = False
        for index, vector in enumerate(vectors):
            label = min(
                range(n_clusters),
                key=lambda cluster: _cosine_distance(vector, centroids[cluster]),
            )
            if labels[index] != label:
                labels[index] = label
                changed = True
        if not changed:
            break
        for cluster in range(n_clusters):
            members = [
                vector for vector, label in zip(vectors, labels) if label == cluster
            ]
            if members:
                centroids[cluster] = [
                    sum(vector[pos] for vector in members) / len(members)
                    for pos in range(len(vectors[0]))
                ]
    return labels


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)

