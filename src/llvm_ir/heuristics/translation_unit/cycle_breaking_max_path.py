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
    paths = cycle_breaking_max_paths(graph, config=config, top_k=1)
    return paths[0] if paths else []


def cycle_breaking_max_paths(
    graph: PassOrderGraph,
    *,
    config: CycleBreakingMaxPathConfig | None = None,
    top_k: int = 10,
) -> list[list[str]]:
    """Remove minimum-weight cycle edges, then return top DAG paths by weight."""
    nodes, edges, order, config = _prepare_cycle_broken_dag(graph, config=config)
    return _max_weight_paths(
        nodes,
        edges,
        order,
        max_length=config.max_length,
        top_k=top_k,
    )


def cycle_breaking_diverse_start_paths(
    graph: PassOrderGraph,
    *,
    config: CycleBreakingMaxPathConfig | None = None,
    top_k: int = 10,
) -> list[list[str]]:
    """Return one max-weight DAG path for each top random-search start pass."""
    nodes, edges, order, config = _prepare_cycle_broken_dag(graph, config=config)
    start_nodes = _ranked_start_nodes(graph, nodes, edges, top_k=max(1, top_k))
    paths = []
    seen_starts: set[str] = set()
    for start in start_nodes:
        path = _max_weight_path_from_start(
            nodes,
            edges,
            order,
            start=start,
            max_length=config.max_length,
        )
        if not path or path[0] in seen_starts:
            continue
        paths.append(path)
        seen_starts.add(path[0])
        if len(paths) >= max(1, top_k):
            break
    return paths


def cycle_breaking_top_start_paths(
    graph: PassOrderGraph,
    *,
    config: CycleBreakingMaxPathConfig | None = None,
    top_starts: int = 10,
    paths_per_start: int = 10,
) -> list[list[str]]:
    """Return top paths for each top random-search start pass."""
    nodes, edges, order, config = _prepare_cycle_broken_dag(graph, config=config)
    paths: list[list[str]] = []
    for start in _ranked_start_nodes(graph, nodes, edges, top_k=max(1, top_starts)):
        paths.extend(
            _max_weight_paths_from_start(
                nodes,
                edges,
                order,
                start=start,
                max_length=config.max_length,
                top_k=paths_per_start,
            )
        )
    return paths


def _prepare_cycle_broken_dag(
    graph: PassOrderGraph,
    *,
    config: CycleBreakingMaxPathConfig | None,
) -> tuple[list[str], dict[tuple[str, str], int], list[str], CycleBreakingMaxPathConfig]:
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
    return nodes, edges, order, config


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
    paths = _max_weight_paths(nodes, edges, order, max_length=max_length, top_k=1)
    return paths[0] if paths else []


def _max_weight_path_from_start(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    order: list[str],
    *,
    start: str,
    max_length: int,
) -> list[str]:
    paths = _max_weight_paths_from_start(
        nodes,
        edges,
        order,
        start=start,
        max_length=max_length,
        top_k=1,
    )
    return paths[0] if paths else []


def _max_weight_paths_from_start(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    order: list[str],
    *,
    start: str,
    max_length: int,
    top_k: int,
) -> list[list[str]]:
    if start not in nodes:
        return []
    max_allowed = len(nodes) if max_length <= 0 else max(1, min(max_length, len(nodes)))
    adjacent = _adjacency(nodes, edges)
    best: dict[str, dict[int, list[tuple[int, list[str]]]]] = {
        start: {1: [(0, [start])]}
    }

    for source in order:
        source_states = list(best.get(source, {}).items())
        for target in adjacent.get(source, []):
            weight = edges[(source, target)]
            for length, candidates in source_states:
                for score, path in candidates:
                    next_length = length + 1
                    if next_length > max_allowed:
                        continue
                    candidate = (score + weight, path + [target])
                    current = best.setdefault(target, {}).setdefault(next_length, [])
                    current.append(candidate)
                    current[:] = _dedupe_ranked_paths(current, top_k=top_k)

    candidates = [
        candidate
        for by_length in best.values()
        for by_length_candidates in by_length.values()
        for candidate in by_length_candidates
    ]
    return [
        path
        for _, path in _dedupe_ranked_paths(candidates, top_k=max(1, top_k))
    ]


def _max_weight_paths(
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    order: list[str],
    *,
    max_length: int,
    top_k: int,
) -> list[list[str]]:
    if not nodes:
        return []
    max_allowed = len(nodes) if max_length <= 0 else max(1, min(max_length, len(nodes)))
    adjacent = _adjacency(nodes, edges)
    best: dict[str, dict[int, list[tuple[int, list[str]]]]] = {
        node: {1: [(0, [node])]} for node in nodes
    }

    for source in order:
        source_states = list(best[source].items())
        for target in adjacent.get(source, []):
            weight = edges[(source, target)]
            for length, candidates in source_states:
                for score, path in candidates:
                    next_length = length + 1
                    if next_length > max_allowed:
                        continue
                    candidate = (score + weight, path + [target])
                    current = best[target].setdefault(next_length, [])
                    current.append(candidate)
                    current[:] = _dedupe_ranked_paths(current, top_k=top_k)

    candidates = [
        candidate
        for by_length in best.values()
        for by_length_candidates in by_length.values()
        for candidate in by_length_candidates
    ]
    return [
        path
        for _, path in _dedupe_ranked_paths(candidates, top_k=max(1, top_k))
    ]


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


def _ranked_start_nodes(
    graph: PassOrderGraph,
    nodes: list[str],
    edges: dict[tuple[str, str], int],
    *,
    top_k: int,
) -> list[str]:
    node_set = set(nodes)
    if graph.start_counts:
        weights = {
            node: weight
            for node, weight in graph.start_counts.items()
            if node in node_set and weight > 0
        }
    else:
        weights = {}
    if not weights:
        adjacent = _adjacency(nodes, edges)
        for node in nodes:
            outgoing = sum(edges[(node, target)] for target in adjacent.get(node, []))
            incoming = sum(
                weight
                for (source, target), weight in edges.items()
                if source != target and target == node
            )
            weights[node] = max(1, outgoing - incoming + 1)
    ranked = sorted(weights, key=lambda node: (-weights[node], node))
    return ranked[: max(1, top_k)]


def _path_key(candidate: tuple[int, list[str]]) -> tuple[int, int, tuple[str, ...]]:
    score, path = candidate
    return score, len(path), tuple(reversed(path))


def _dedupe_ranked_paths(
    candidates: list[tuple[int, list[str]]],
    *,
    top_k: int,
) -> list[tuple[int, list[str]]]:
    unique: dict[tuple[str, ...], tuple[int, list[str]]] = {}
    for score, path in candidates:
        key = tuple(path)
        candidate = (score, path)
        current = unique.get(key)
        if current is None or _path_key(candidate) > _path_key(current):
            unique[key] = candidate
    return sorted(unique.values(), key=_path_key, reverse=True)[: max(1, top_k)]
