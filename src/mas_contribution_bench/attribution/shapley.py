"""Exact and sampled Shapley value estimation."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Callable


UtilityFn = Callable[[frozenset[str]], float]


def estimate_shapley_exact(utility: UtilityFn, agents: list[str]) -> dict[str, float]:
    n = len(agents)
    values = {agent: 0.0 for agent in agents}
    for agent in agents:
        others = [a for a in agents if a != agent]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                coalition = frozenset(subset)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                values[agent] += weight * (utility(coalition | {agent}) - utility(coalition))
    return values


def estimate_shapley_sampled(
    utility: UtilityFn,
    agents: list[str],
    num_samples: int = 64,
    seed: int | None = None,
) -> dict[str, float]:
    rng = random.Random(seed)
    totals = defaultdict(float)
    for _ in range(num_samples):
        order = list(agents)
        rng.shuffle(order)
        coalition: frozenset[str] = frozenset()
        previous = utility(coalition)
        for agent in order:
            updated = coalition | {agent}
            current = utility(updated)
            totals[agent] += current - previous
            coalition = updated
            previous = current
    return {agent: totals[agent] / float(num_samples) for agent in agents}


def estimate_shapley(
    coalition_scores: dict[frozenset[str], float],
    agents: list[str],
    num_samples: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    utility = lambda coalition: coalition_scores.get(coalition, 0.0)
    if num_samples is None and len(agents) <= 8:
        return estimate_shapley_exact(utility, agents)
    return estimate_shapley_sampled(utility, agents, num_samples=num_samples or 64, seed=seed)
