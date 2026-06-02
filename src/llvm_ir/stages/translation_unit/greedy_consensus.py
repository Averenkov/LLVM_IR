"""Greedy consensus ordering heuristic for pass-order graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .order_graph import PassOrderGraph
from .path_scoring import sorted_nodes_by_priority


@dataclass(frozen=True)
class GreedyConsensusConfig:
    max_length: int = 0


def greedy_consensus_path(
    graph: PassOrderGraph,
    *,
    config: GreedyConsensusConfig | None = None,
) -> list[str]:
    """Sort passes by outgoing-vs-incoming weighted support."""
    if config is None:
        config = GreedyConsensusConfig()
    path = sorted_nodes_by_priority(graph)
    if config.max_length > 0:
        return path[: config.max_length]
    return path
