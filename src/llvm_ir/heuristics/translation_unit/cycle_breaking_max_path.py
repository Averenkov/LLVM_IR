"""Cycle-breaking maximum-weight path heuristic for pass-order graphs."""

from __future__ import annotations

from dataclasses import dataclass

from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph


@dataclass(frozen=True)
class CycleBreakingMaxPathConfig:
    max_length: int = 0
    min_edge_weight: int = 1


def cycle_breaking_max_path(
    graph: PassOrderGraph,
    *,
    config: CycleBreakingMaxPathConfig | None = None,
) -> list[str]:
    """Remove minimum-weight cycle edges, then find a max-weight DAG path."""
    if config is None:
        config = CycleBreakingMaxPathConfig()
    nodes = sorted(graph.nodes)
    edges = {
        (source, target): weight
        for (source, target), weight in graph.edge_counts.items()
        if source != target and weight >= config.min_edge_weight
    }

    while True:
        cycle = _find_cycle(nodes, edges)
        if not cycle:
            break
        weakest = min(
            cycle,
            key=lambda pair: (edges.get(pair, 0), pair[0], pair[1]),
        )
        del edges[weakest]

    order = _topological_order(nodes, edges)
    return _max_weight_path(nodes, edges, order, max_length=config.max_length)


def _find_cycle(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
) -> list[tuple[str, str]]:
    adjacent = _adjacency(nodes, edges)
    state = {node: 0 for node in nodes}
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def visit(node: str) -> list[tuple[str, str]]:
        state[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)
        for target in adjacent.get(node, []):
            if state[target] == 0:
                cycle = visit(target)
                if cycle:
                    return cycle
            elif state[target] == 1:
                cycle_nodes = stack[stack_index[target] :] + [target]
                return list(zip(cycle_nodes, cycle_nodes[1:]))
        stack.pop()
        del stack_index[node]
        state[node] = 2
        return []

    for node in nodes:
        if state[node] == 0:
            cycle = visit(node)
            if cycle:
                return cycle
    return []


def _topological_order(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
) -> list[str]:
    adjacent = _adjacency(nodes, edges)
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        if source in indegree and target in indegree:
            indegree[target] += 1

    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in adjacent.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("cycle breaking did not produce a DAG")
    return order


def _max_weight_path(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    order: list[str],
    *,
    max_length: int,
) -> list[str]:
    if not nodes:
        return []
    max_allowed = len(nodes) if max_length <= 0 else max(1, min(max_length, len(nodes)))
    adjacent = _adjacency(nodes, edges)
    best: dict[str, dict[int, tuple[int, list[str]]]] = {
        node: {1: (0, [node])} for node in nodes
    }

    for source in order:
        source_states = list(best[source].items())
        for target in adjacent.get(source, []):
            weight = edges[(source, target)]
            for length, (score, path) in source_states:
                next_length = length + 1
                if next_length > max_allowed:
                    continue
                candidate = (score + weight, path + [target])
                current = best[target].get(next_length)
                if current is None or _path_key(candidate) > _path_key(current):
                    best[target][next_length] = candidate

    candidates = [candidate for by_length in best.values() for candidate in by_length.values()]
    return max(candidates, key=_path_key)[1]


def _adjacency(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
) -> dict[str, list[str]]:
    node_set = set(nodes)
    adjacent = {node: [] for node in nodes}
    for source, target in sorted(edges):
        if source in node_set and target in node_set:
            adjacent[source].append(target)
    return adjacent


def _path_key(candidate: tuple[int, list[str]]) -> tuple[int, int, tuple[str, ...]]:
    score, path = candidate
    return score, len(path), tuple(reversed(path))
