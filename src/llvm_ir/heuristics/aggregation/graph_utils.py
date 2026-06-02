"""Graph utilities shared by aggregation heuristics."""

from __future__ import annotations

import heapq
from collections import defaultdict

from .base import PassGraph


def normalize_edge_weights(weights: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    if not weights:
        return {}
    max_weight = max(weights.values())
    if max_weight <= 0:
        return {edge: 0.0 for edge in weights}
    return {edge: weight / max_weight for edge, weight in weights.items()}


def blended_weights(graph: PassGraph, *, alpha: float = 0.5) -> dict[tuple[str, str], float]:
    count = normalize_edge_weights(graph.count_weight)
    delta = normalize_edge_weights(graph.delta_weight)
    edges = set(count) | set(delta)
    return {
        edge: alpha * delta.get(edge, 0.0) + (1.0 - alpha) * count.get(edge, 0.0)
        for edge in edges
        if edge[0] != edge[1]
    }


def outgoing_weight(node: str, weights: dict[tuple[str, str], float], nodes: set[str]) -> float:
    return sum(
        weight
        for (source, target), weight in weights.items()
        if source == node and target in nodes and source != target
    )


def incoming_weight(node: str, weights: dict[tuple[str, str], float], nodes: set[str]) -> float:
    return sum(
        weight
        for (source, target), weight in weights.items()
        if target == node and source in nodes and source != target
    )


def eades_order(nodes: set[str], weights: dict[tuple[str, str], float]) -> list[str]:
    remaining = set(nodes)
    left: list[str] = []
    right: list[str] = []
    while remaining:
        changed = True
        while changed:
            changed = False
            sinks = sorted(
                node for node in remaining if outgoing_weight(node, weights, remaining) == 0
            )
            for node in sinks:
                if node in remaining:
                    remaining.remove(node)
                    right.append(node)
                    changed = True
            sources = sorted(
                node for node in remaining if incoming_weight(node, weights, remaining) == 0
            )
            for node in sources:
                if node in remaining:
                    remaining.remove(node)
                    left.append(node)
                    changed = True
        if remaining:
            selected = max(
                remaining,
                key=lambda node: (
                    outgoing_weight(node, weights, remaining)
                    - incoming_weight(node, weights, remaining),
                    node,
                ),
            )
            remaining.remove(selected)
            left.append(selected)
    return left + list(reversed(right))


def topological_order(nodes: set[str], weights: dict[tuple[str, str], float]) -> list[str]:
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node: 0 for node in nodes}
    priority = {node: outgoing_weight(node, weights, nodes) for node in nodes}
    for (source, target), weight in weights.items():
        if weight <= 0 or source == target or source not in nodes or target not in nodes:
            continue
        outgoing[source].append(target)
        indegree[target] += 1
    heap = [(-priority[node], node) for node, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _score, node = heapq.heappop(heap)
        order.append(node)
        for target in outgoing[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(heap, (-priority[target], target))
    if len(order) < len(nodes):
        missing = set(nodes) - set(order)
        order.extend(eades_order(missing, weights))
    return order


def strongly_connected_components(nodes: set[str], weights: dict[tuple[str, str], float]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []
    adjacency: dict[str, list[str]] = defaultdict(list)
    for (source, target), weight in weights.items():
        if weight > 0 and source in nodes and target in nodes and source != target:
            adjacency[source].append(target)

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adjacency[node]:
            if target not in indices:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return components


def path_graph_score(path: list[str], weights: dict[tuple[str, str], float]) -> float:
    score = 0.0
    for left_index, source in enumerate(path):
        for target in path[left_index + 1 :]:
            score += weights.get((source, target), 0.0)
            score -= weights.get((target, source), 0.0)
    return score


def best_prefix_by_graph_score(path: list[str], weights: dict[tuple[str, str], float]) -> tuple[int, float]:
    best_len = len(path)
    best_score = float("-inf")
    for length in range(0, len(path) + 1):
        score = path_graph_score(path[:length], weights)
        if score > best_score:
            best_score = score
            best_len = length
    return best_len, best_score


def unique_sequences(sequences: list[list[str]], *, limit: int | None = None) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    result = []
    for sequence in sequences:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        result.append(sequence)
        if limit is not None and len(result) >= limit:
            break
    return result
