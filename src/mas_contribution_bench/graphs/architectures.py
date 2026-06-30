"""Architecture helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from mas_contribution_bench.config.loaders import LoadedArchitectureSpec
from mas_contribution_bench.data.schemas import GraphSpec


def architecture_to_graph_spec(architecture: LoadedArchitectureSpec) -> GraphSpec:
    nodes = sorted(set(architecture.roles) | {"final_answer"})
    return GraphSpec(
        nodes=nodes,
        edges=architecture.edges,
        edge_type="communication",
        graph_metadata={
            "architecture_id": architecture.architecture_id,
            "family": architecture.family,
            "orchestration": architecture.orchestration,
        },
    )


def architecture_to_networkx(architecture: LoadedArchitectureSpec) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(architecture.roles)
    graph.add_node("final_answer")
    graph.add_edges_from(architecture.edges)
    return graph


def execution_order(architecture: LoadedArchitectureSpec) -> list[str]:
    mode = architecture.orchestration.get("execution_mode", "")
    if mode in {"sequential", "pipeline", "dag", "sequential_with_supervisor"}:
        graph = architecture_to_networkx(architecture)
        role_graph = graph.subgraph(architecture.roles).copy()
        if nx.is_directed_acyclic_graph(role_graph):
            return list(nx.topological_sort(role_graph))
    return list(architecture.roles)


def bypass_edges(edges: list[tuple[str, str]], removed: set[str]) -> list[tuple[str, str]]:
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    kept: list[tuple[str, str]] = []
    for src, dst in edges:
        if dst in removed:
            incoming[dst].append(src)
        if src in removed:
            outgoing[src].append(dst)
        if src not in removed and dst not in removed:
            kept.append((src, dst))
    for node in removed:
        for src in incoming[node]:
            for dst in outgoing[node]:
                if src not in removed and dst not in removed and src != dst:
                    kept.append((src, dst))
    return sorted(set(kept))


def topology_features(architecture: LoadedArchitectureSpec) -> dict[str, dict[str, float]]:
    graph = architecture_to_networkx(architecture)
    roles = architecture.roles
    undirected = graph.to_undirected()
    degree = dict(graph.degree())
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() else {}
    closeness = nx.closeness_centrality(graph) if graph.number_of_nodes() else {}
    pagerank = nx.pagerank(graph) if graph.number_of_edges() else {n: 0.0 for n in graph.nodes}
    articulation = set(nx.articulation_points(undirected)) if undirected.number_of_nodes() else set()
    depths: dict[str, float] = {role: 0.0 for role in roles}
    if nx.is_directed_acyclic_graph(graph):
        for role in roles:
            ancestors = nx.ancestors(graph, role)
            depths[role] = float(max((len(nx.shortest_path(graph, a, role)) - 1 for a in ancestors), default=0))
    return {
        role: {
            "degree": float(degree.get(role, 0)),
            "in_degree": float(in_degree.get(role, 0)),
            "out_degree": float(out_degree.get(role, 0)),
            "betweenness": float(betweenness.get(role, 0.0)),
            "closeness": float(closeness.get(role, 0.0)),
            "pagerank": float(pagerank.get(role, 0.0)),
            "dag_depth": float(depths.get(role, 0.0)),
            "fan_in": float(in_degree.get(role, 0)),
            "fan_out": float(out_degree.get(role, 0)),
            "is_articulation_point": float(role in articulation),
        }
        for role in roles
    }


def controlled_architecture(raw: dict[str, Any], architecture_id: str = "controlled") -> LoadedArchitectureSpec:
    roles = list(raw.get("roles") or raw.get("controlled_role_set") or [])
    edges = [tuple(edge) for edge in raw.get("edges", [])]
    return LoadedArchitectureSpec(
        architecture_id=architecture_id,
        name=raw.get("name", architecture_id),
        family=raw.get("family", raw.get("template", "controlled")),
        roles=roles,
        canonical_roles={role: role for role in roles},
        entrypoint=raw.get("entrypoint", roles[0] if roles else ""),
        terminal_nodes=list(raw.get("terminal_nodes", ["final_answer"])),
        edges=edges,
        orchestration=dict(raw.get("orchestration", {})),
        default_permissions=dict(raw.get("default_permissions", {})),
        raw=raw,
    )
