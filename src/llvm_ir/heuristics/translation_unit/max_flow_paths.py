"""Top network max-flow "flows" of each length in a pass-order graph.

For a given flow length ``L`` we build a layered (time-expanded) flow network:
``L`` layers of pass-vertices, an edge ``(u, i) -> (v, i+1)`` with capacity equal
to the order-graph weight ``w(u, v)`` for every consecutive layer pair, a source
feeding layer 0 (capacity = how often the pass starts a good sequence), and a
sink drained from layer ``L-1``. The maximum ``source -> sink`` flow is computed
with Dinic's algorithm and decomposed into source-to-sink paths; each decomposed
path is a length-``L`` pass sequence carrying some integer flow. The top paths by
flow value become candidates.

Unlike the max-weight path heuristics (which maximize the *sum* of edge weights),
a flow path is limited by its *bottleneck* (minimum capacity) and the residual
network forces successive paths to diverge, so the top flows are bottleneck-robust
and diverse. The flow length ``L`` grows with benchmark size via
``scale_flow_length`` -- larger translation units explore longer flows.
"""

from __future__ import annotations

from dataclasses import dataclass

from llvm_ir.stages.translation_unit.graph.order_graph import PassOrderGraph

_INF = 10**9


@dataclass(frozen=True)
class MaxFlowConfig:
    min_length: int = 2
    max_length: int = 6
    per_length_top_k: int = 40
    min_edge_weight: int = 1


def scale_flow_length(bitcode_bytes: int, *, base: int, cap: int) -> int:
    """Grow the maximum flow length with benchmark size (capped at ``cap``)."""
    thresholds = (50_000, 150_000, 500_000)
    extra = sum(1 for threshold in thresholds if bitcode_bytes >= threshold)
    return max(base, min(cap, base + extra))


class _Dinic:
    """Minimal Dinic max-flow with explicit per-edge residual flow readout."""

    def __init__(self, num_nodes: int) -> None:
        self.size = num_nodes
        self.graph: list[list[int]] = [[] for _ in range(num_nodes)]
        # edges stored as [to, capacity, flow]
        self.edges: list[list[int]] = []

    def add_edge(self, u: int, v: int, capacity: int) -> int:
        self.graph[u].append(len(self.edges))
        self.edges.append([v, capacity, 0])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, 0])  # reverse edge
        return len(self.edges) - 2

    def _bfs(self, source: int, sink: int, level: list[int]) -> bool:
        for index in range(self.size):
            level[index] = -1
        level[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for edge_id in self.graph[node]:
                to, capacity, flow = self.edges[edge_id]
                if level[to] < 0 and capacity - flow > 0:
                    level[to] = level[node] + 1
                    queue.append(to)
        return level[sink] >= 0

    def _dfs(self, node: int, sink: int, pushed: int, level: list[int], it: list[int]) -> int:
        if node == sink:
            return pushed
        while it[node] < len(self.graph[node]):
            edge_id = self.graph[node][it[node]]
            to, capacity, flow = self.edges[edge_id]
            if level[to] == level[node] + 1 and capacity - flow > 0:
                tr = self._dfs(to, sink, min(pushed, capacity - flow), level, it)
                if tr > 0:
                    self.edges[edge_id][2] += tr
                    self.edges[edge_id ^ 1][2] -= tr
                    return tr
            it[node] += 1
        return 0

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        level = [-1] * self.size
        while self._bfs(source, sink, level):
            it = [0] * self.size
            while True:
                pushed = self._dfs(source, sink, _INF, level, it)
                if pushed == 0:
                    break
                flow += pushed
        return flow


def _start_weight(graph: PassOrderGraph, node: str) -> int:
    return max(int(graph.start_counts.get(node, 0)), 1)


def _flows_of_length(
    graph: PassOrderGraph,
    nodes: list[str],
    forward_edges: dict[tuple[str, str], int],
    length: int,
    per_length_top_k: int,
) -> list[list[str]]:
    if length < 2 or not forward_edges:
        # Length-1 "flows" are just single passes ranked by start weight.
        if length == 1:
            ranked = sorted(nodes, key=lambda n: (-_start_weight(graph, n), n))
            return [[n] for n in ranked[: max(1, per_length_top_k)]]
        return []

    node_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    # Node id layout: source=0, sink=1, then (layer*n + pass_index) + 2.
    source, sink = 0, 1

    def vid(layer: int, pass_index: int) -> int:
        return 2 + layer * n + pass_index

    dinic = _Dinic(2 + length * n)
    # Track which layer-transition edge ids correspond to which (u, v) so we can
    # rebuild paths from the residual flow afterwards.
    layer_edge: dict[int, tuple[int, str, str]] = {}
    for node in nodes:
        dinic.add_edge(source, vid(0, node_index[node]), _start_weight(graph, node))
    for layer in range(length - 1):
        for (u, v), weight in forward_edges.items():
            edge_id = dinic.add_edge(vid(layer, node_index[u]), vid(layer + 1, node_index[v]), weight)
            layer_edge[edge_id] = (layer, u, v)
    for node in nodes:
        dinic.add_edge(vid(length - 1, node_index[node]), sink, _INF)

    dinic.max_flow(source, sink)

    # Decompose the realized flow into source->sink paths.
    paths: list[tuple[int, list[str]]] = []
    # Build mutable residual-flow adjacency: for each node, outgoing edges with flow>0.
    flow_left = {edge_id: max(0, dinic.edges[edge_id][2]) for edge_id in layer_edge}
    # source edges flow:
    src_flow: dict[int, int] = {}
    for edge_id in dinic.graph[source]:
        to, capacity, flow = dinic.edges[edge_id]
        if flow > 0:
            src_flow[edge_id] = flow

    def out_edges_with_flow(node_id: int):
        for edge_id in dinic.graph[node_id]:
            if edge_id in flow_left and flow_left[edge_id] > 0:
                yield edge_id

    guard = 0
    while src_flow and guard < 100000:
        guard += 1
        # pick the source edge with the most remaining flow (greedy "top" first)
        start_edge = max(src_flow, key=lambda e: src_flow[e])
        if src_flow[start_edge] <= 0:
            del src_flow[start_edge]
            continue
        first_node = dinic.edges[start_edge][0]
        passes = [nodes[(first_node - 2) % n]]
        chosen_edges: list[int] = []
        bottleneck = src_flow[start_edge]
        node_id = first_node
        ok = True
        for _ in range(length - 1):
            nxt = next(out_edges_with_flow(node_id), None)
            if nxt is None:
                ok = False
                break
            chosen_edges.append(nxt)
            bottleneck = min(bottleneck, flow_left[nxt])
            to = dinic.edges[nxt][0]
            passes.append(nodes[(to - 2) % n])
            node_id = to
        if not ok or bottleneck <= 0:
            # cannot complete a full-length path from this source unit; drop a unit
            src_flow[start_edge] -= 1
            if src_flow[start_edge] <= 0:
                del src_flow[start_edge]
            continue
        # subtract bottleneck along the chosen path
        src_flow[start_edge] -= bottleneck
        if src_flow[start_edge] <= 0:
            del src_flow[start_edge]
        for edge_id in chosen_edges:
            flow_left[edge_id] -= bottleneck
        paths.append((bottleneck, passes))

    paths.sort(key=lambda item: (-item[0], item[1]))
    return [passes for _flow, passes in paths[: max(1, per_length_top_k)]]


def top_flow_paths(
    graph: PassOrderGraph,
    *,
    config: MaxFlowConfig | None = None,
) -> list[list[str]]:
    if config is None:
        config = MaxFlowConfig()
    nodes = sorted(graph.nodes)
    if not nodes:
        return []
    forward_edges = {
        (source, target): weight
        for (source, target), weight in graph.edge_counts.items()
        if source != target and weight >= config.min_edge_weight
    }

    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for length in range(max(1, config.min_length), max(config.min_length, config.max_length) + 1):
        for path in _flows_of_length(graph, nodes, forward_edges, length, config.per_length_top_k):
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                result.append(path)
    return result
