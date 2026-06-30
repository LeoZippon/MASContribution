"""Myerson value helpers for graph-constrained games."""

from __future__ import annotations

import networkx as nx

from mas_contribution_bench.attribution.shapley import estimate_shapley_exact, estimate_shapley_sampled


def connected_closure(coalition: frozenset[str], graph: nx.Graph) -> list[frozenset[str]]:
    subgraph = graph.subgraph(coalition)
    return [frozenset(component) for component in nx.connected_components(subgraph.to_undirected())]


def estimate_myerson(
    utility,
    agents: list[str],
    graph: nx.Graph,
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    def graph_restricted_utility(coalition: frozenset[str]) -> float:
        return sum(utility(component) for component in connected_closure(coalition, graph))

    if num_samples is None and len(agents) <= 8:
        return estimate_shapley_exact(graph_restricted_utility, agents)
    return estimate_shapley_sampled(graph_restricted_utility, agents, num_samples=num_samples or 64, seed=seed)
