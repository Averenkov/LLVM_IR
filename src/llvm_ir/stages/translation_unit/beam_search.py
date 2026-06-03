"""Beam-search path heuristic for pass-order graphs."""

from __future__ import annotations

from dataclasses import dataclass

from .graph.order_graph import PassOrderGraph
from .path_scoring import node_priority, score_path


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_width: int = 16
    max_length: int = 12
    conflict_penalty: float = 1.0


def beam_search_path(
    graph: PassOrderGraph,
    *,
    config: BeamSearchConfig | None = None,
) -> list[str]:
    """Build a high-scoring path by expanding the best partial paths."""
    if config is None:
        config = BeamSearchConfig()
    nodes = sorted(graph.nodes)
    if not nodes:
        return []
    max_length = max(1, min(config.max_length, len(nodes)))
    beam_width = max(1, config.beam_width)
    priorities = {node: node_priority(graph, node) for node in nodes}
    beam: list[list[str]] = [[node] for node in nodes]
    beam = top_paths(graph, beam, beam_width, config.conflict_penalty, priorities)
    best = beam[0]

    for _ in range(1, max_length):
        expanded = []
        for path in beam:
            used = set(path)
            for candidate in nodes:
                if candidate in used:
                    continue
                expanded.append(path + [candidate])
        if not expanded:
            break
        beam = top_paths(
            graph,
            expanded,
            beam_width,
            config.conflict_penalty,
            priorities,
        )
        if beam_score(graph, beam[0], config.conflict_penalty, priorities) > beam_score(
            graph,
            best,
            config.conflict_penalty,
            priorities,
        ):
            best = beam[0]
    return best


def top_paths(
    graph: PassOrderGraph,
    paths: list[list[str]],
    beam_width: int,
    conflict_penalty: float,
    priorities: dict[str, int] | None = None,
) -> list[list[str]]:
    if priorities is None:
        priorities = {node: node_priority(graph, node) for node in graph.nodes}
    scored = []
    for path in paths:
        path_score = score_path(graph, path)
        total = (
            path_score.order_score
            - conflict_penalty * path_score.conflict_score
            + sum(priorities[node] for node in path) * 0.001
        )
        scored.append((
            (-total, -path_score.order_score, len(path), path),
            path,
        ))
    return [path for _, path in sorted(scored)[:beam_width]]


def beam_score(
    graph: PassOrderGraph,
    path: list[str],
    conflict_penalty: float,
    priorities: dict[str, int] | None = None,
) -> float:
    if priorities is None:
        priorities = {node: node_priority(graph, node) for node in graph.nodes}
    score = score_path(graph, path)
    priority_bonus = sum(priorities[node] for node in path) * 0.001
    return score.order_score - conflict_penalty * score.conflict_score + priority_bonus
