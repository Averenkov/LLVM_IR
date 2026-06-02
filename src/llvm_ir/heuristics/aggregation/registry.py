"""Registry of aggregation heuristics."""

from __future__ import annotations

from .base import AggregationHeuristic
from .beam_diversity import BeamSearchDiversity
from .cluster_aware import ClusterAwareAggregation
from .hpp import HPPHeuristic
from .hpp_eades_topk import HPPEadesTopK
from .ilp_arrangement import ILPLinearArrangement
from .markov_hitting import MarkovHittingOrdering
from .pagerank_ordering import PageRankOrdering
from .position_median import PositionMedianOrdering
from .scc_ordering import SCCOrdering
from .voting_ensemble import VotingEnsemble
from .wfas_eades import WFASEades


_REGISTRY: dict[str, type[AggregationHeuristic]] = {
    "wfas_eades": WFASEades,
    "scc_ordering": SCCOrdering,
    "pagerank": PageRankOrdering,
    "position_median": PositionMedianOrdering,
    "hpp": HPPHeuristic,
    "beam_diversity": BeamSearchDiversity,
    "cluster_aware": ClusterAwareAggregation,
    "ilp_arrangement": ILPLinearArrangement,
    "markov_hitting": MarkovHittingOrdering,
    "voting_ensemble": VotingEnsemble,
    "hpp_eades_topk": HPPEadesTopK,
}


def registered_heuristics() -> dict[str, type[AggregationHeuristic]]:
    return dict(_REGISTRY)


def available_heuristics() -> list[str]:
    return sorted(_REGISTRY)


def build_heuristic(name: str) -> AggregationHeuristic:
    normalized = name.strip().lower()
    try:
        return _REGISTRY[normalized]()
    except KeyError as exc:
        raise ValueError(f"Unknown aggregation heuristic: {name}") from exc
