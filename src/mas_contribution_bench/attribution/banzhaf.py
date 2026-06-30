"""Banzhaf index estimation."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Callable


UtilityFn = Callable[[frozenset[str]], float]


def estimate_banzhaf_exact(utility: UtilityFn, agents: list[str]) -> dict[str, float]:
    totals = {agent: 0.0 for agent in agents}
    for agent in agents:
        others = [a for a in agents if a != agent]
        count = 0
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                coalition = frozenset(subset)
                totals[agent] += utility(coalition | {agent}) - utility(coalition)
                count += 1
        totals[agent] = totals[agent] / max(count, 1)
    return totals


def estimate_banzhaf_sampled(
    utility: UtilityFn,
    agents: list[str],
    num_samples: int = 128,
    seed: int | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    totals = defaultdict(float)
    counts = defaultdict(int)
    for agent in agents:
        others = [a for a in agents if a != agent]
        for _ in range(num_samples):
            coalition = frozenset(a for a in others if rng.random() < 0.5)
            totals[agent] += utility(coalition | {agent}) - utility(coalition)
            counts[agent] += 1
    return {agent: totals[agent] / max(counts[agent], 1) for agent in agents}


def estimate_banzhaf(
    coalition_scores: dict[frozenset[str], float],
    agents: list[str],
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    utility = lambda coalition: coalition_scores.get(coalition, 0.0)
    if num_samples is None and len(agents) <= 10:
        return estimate_banzhaf_exact(utility, agents)
    return estimate_banzhaf_sampled(utility, agents, num_samples=num_samples or 128, seed=seed)
