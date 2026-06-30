"""Owen value approximation for grouped agents."""

from __future__ import annotations

from collections import defaultdict

from mas_contribution_bench.attribution.shapley import estimate_shapley_exact


def estimate_owen(utility, groups: dict[str, list[str]]) -> dict[str, float]:
    group_names = list(groups)

    def group_utility(active_groups: frozenset[str]) -> float:
        agents = set()
        for group in active_groups:
            agents.update(groups[group])
        return utility(frozenset(agents))

    group_values = estimate_shapley_exact(group_utility, group_names)
    values = defaultdict(float)
    for group, members in groups.items():
        if not members:
            continue

        def within_utility(active_members: frozenset[str]) -> float:
            base_agents = set()
            for other_group, other_members in groups.items():
                if other_group != group:
                    base_agents.update(other_members)
            return utility(frozenset(base_agents | set(active_members)))

        within = estimate_shapley_exact(within_utility, members)
        total_within = sum(abs(v) for v in within.values()) or 1.0
        for agent, value in within.items():
            values[agent] = group_values[group] * abs(value) / total_within
    return dict(values)
