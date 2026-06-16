"""Monte-Carlo Tree Search directly over whole-TU pass sequences.

Unlike the graph-constructive heuristics (cycle-breaking, beam, random walk,
chunk-forest), this heuristic does not enumerate candidates from a static
per-function structure and then measure them. It searches the pass-sequence
prefix space *online*, using the real measured ``.text`` size after each pass as
the reward signal. The per-function pass-order graph is used only as a soft
PUCT prior, so the search is guided by prior knowledge but not bounded by it.

The search tree is isomorphic to the prefix cache: a node is a pass-sequence
prefix applied to the baseline bitcode, so revisited prefixes cost no extra real
evaluations. Reward is the normalized best-prefix improvement
``max(0, (baseline - size) / baseline)`` and is propagated with **max-backup**
(this is optimization for the single best sequence, not expected-value play).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random


class BudgetExhausted(Exception):
    """Raised by the measure callback when no real-evaluation budget remains."""


@dataclass(frozen=True)
class MeasureResult:
    size: int | None
    error: str = ""


@dataclass(frozen=True)
class MCTSConfig:
    max_length: int = 12
    c_puct: float = 1.5
    rollout_length: int = 3
    branching: int = 8
    prior_floor: float = 0.05
    seed: int = 7
    max_iterations: int = 1_000_000


@dataclass
class MCTSResult:
    best_passes: tuple[str, ...]
    best_size: int
    best_prefix_len: int
    nodes_expanded: int
    iterations: int
    max_depth: int
    budget_exhausted: bool


@dataclass
class _Node:
    key: tuple[str, ...]
    size: int = 0
    parent: "_Node | None" = None
    children: dict[str, "_Node"] = field(default_factory=dict)
    untried: list[str] = field(default_factory=list)
    priors: dict[str, float] = field(default_factory=dict)
    terminal: bool = False
    visits: int = 0
    value: float = 0.0  # max normalized reward seen in this subtree


# A prior provider maps (prefix, candidate_actions) -> {action: weight}.
PriorFn = Callable[[tuple[str, ...]], dict[str, float]]
MeasureFn = Callable[[tuple[str, ...]], MeasureResult]


def run_mcts(
    *,
    baseline_size: int,
    prior_fn: PriorFn,
    measure: MeasureFn,
    config: MCTSConfig,
) -> MCTSResult:
    """Run budget-bounded PUCT search; ``measure`` raises BudgetExhausted to stop.

    ``prior_fn(prefix)`` returns unnormalized action weights for the candidate
    next passes after ``prefix`` (already pruned to a sensible action set). The
    search keeps the top ``branching`` actions per node by prior weight.
    """
    rng = Random(config.seed)
    root = _Node(key=(), size=baseline_size)
    _init_actions(root, prior_fn, config)

    best_size = baseline_size
    best_passes: tuple[str, ...] = ()
    nodes_expanded = 0
    iterations = 0
    max_depth = 0
    budget_exhausted = False

    def normalized(size: int) -> float:
        if baseline_size <= 0:
            return 0.0
        return max(0.0, (baseline_size - size) / baseline_size)

    try:
        while (root.untried or root.children) and iterations < config.max_iterations:
            # --- Selection: descend fully expanded interior nodes via PUCT. ---
            node = root
            while (
                not node.untried
                and node.children
                and not node.terminal
                and len(node.key) < config.max_length
            ):
                node = _select_puct(node, config.c_puct)

            # --- Expansion ---
            if (
                node.untried
                and not node.terminal
                and len(node.key) < config.max_length
            ):
                action = node.untried.pop(0)  # highest-prior untried action
                child_key = node.key + (action,)
                result = measure(child_key)
                child = _Node(key=child_key, parent=node)
                node.children[action] = child
                nodes_expanded += 1
                max_depth = max(max_depth, len(child_key))
                if result.error or result.size is None:
                    child.terminal = True
                    child.size = node.size
                    reward = normalized(node.size)
                else:
                    child.size = result.size
                    _init_actions(child, prior_fn, config)
                    if child.size < best_size:
                        best_size, best_passes = child.size, child_key
                    reward, best_size, best_passes = _rollout(
                        child,
                        measure,
                        prior_fn,
                        config,
                        rng,
                        normalized,
                        best_size,
                        best_passes,
                    )
                _backup(child, reward)
            else:
                # Selection bottomed out at a terminal / max-length node.
                _backup(node, normalized(node.size))

            iterations += 1
    except BudgetExhausted:
        budget_exhausted = True

    return MCTSResult(
        best_passes=best_passes,
        best_size=best_size,
        best_prefix_len=len(best_passes),
        nodes_expanded=nodes_expanded,
        iterations=iterations,
        max_depth=max_depth,
        budget_exhausted=budget_exhausted,
    )


def _init_actions(node: _Node, prior_fn: PriorFn, config: MCTSConfig) -> None:
    if len(node.key) >= config.max_length:
        node.terminal = True
        return
    weights = prior_fn(node.key)
    if not weights:
        node.terminal = True
        return
    floor = config.prior_floor
    adjusted = {action: max(weight, 0.0) + floor for action, weight in weights.items()}
    ranked = sorted(adjusted, key=lambda a: (-adjusted[a], a))[: config.branching]
    total = sum(adjusted[a] for a in ranked)
    node.priors = {a: adjusted[a] / total for a in ranked} if total > 0 else {
        a: 1.0 / len(ranked) for a in ranked
    }
    node.untried = list(ranked)


def _select_puct(node: _Node, c_puct: float) -> _Node:
    parent_visits = max(1, node.visits)
    best_child: _Node | None = None
    best_score = float("-inf")
    for action, child in sorted(node.children.items()):
        prior = node.priors.get(action, 0.0)
        exploit = child.value
        explore = c_puct * prior * math.sqrt(parent_visits) / (1 + child.visits)
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None
    return best_child


def _rollout(
    node: _Node,
    measure: MeasureFn,
    prior_fn: PriorFn,
    config: MCTSConfig,
    rng: Random,
    normalized: Callable[[int], float],
    best_size: int,
    best_passes: tuple[str, ...],
) -> tuple[float, int, tuple[str, ...]]:
    reward = normalized(node.size)
    key = node.key
    steps = min(config.rollout_length, config.max_length - len(node.key))
    for _ in range(steps):
        weights = prior_fn(key)
        if not weights:
            break
        action = _sample_by_weight(weights, rng, config.prior_floor)
        key = key + (action,)
        result = measure(key)
        if result.error or result.size is None:
            break
        reward = max(reward, normalized(result.size))
        if result.size < best_size:
            best_size, best_passes = result.size, key
    return reward, best_size, best_passes


def _sample_by_weight(weights: dict[str, float], rng: Random, floor: float) -> str:
    items = sorted(weights.items())
    adjusted = [(a, max(w, 0.0) + floor) for a, w in items]
    total = sum(w for _a, w in adjusted)
    cursor = rng.random() * total
    for action, weight in adjusted:
        cursor -= weight
        if cursor <= 0:
            return action
    return adjusted[-1][0]


def _backup(node: _Node, reward: float) -> None:
    current: _Node | None = node
    while current is not None:
        current.visits += 1
        if reward > current.value:
            current.value = reward
        current = current.parent
