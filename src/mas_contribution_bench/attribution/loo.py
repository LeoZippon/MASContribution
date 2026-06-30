"""Leave-one-out attribution."""

from __future__ import annotations

from typing import Callable


UtilityFn = Callable[[frozenset[str]], float]


def estimate_loo(utility: UtilityFn, agents: list[str]) -> dict[str, float]:
    full = frozenset(agents)
    full_score = utility(full)
    return {
        agent: full_score - utility(frozenset(a for a in agents if a != agent))
        for agent in agents
    }


def estimate_loo_from_scores(coalition_scores: dict[frozenset[str], float], agents: list[str]) -> dict[str, float]:
    return estimate_loo(lambda coalition: coalition_scores.get(coalition, 0.0), agents)
