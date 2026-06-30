"""Statistical summaries for benchmark outputs."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any

import numpy as np
from scipy import stats


def summarize_scores(records: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        score = record.get("score")
        if score is None:
            continue
        key = tuple(record.get(k) for k in group_keys)
        groups[key].append(float(score))
    rows = []
    for key, values in groups.items():
        rows.append(
            {
                **dict(zip(group_keys, key)),
                "n": len(values),
                "mean": mean(values),
                "median": median(values),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    return rows


def rank_correlation(x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) < 2 or len(y) < 2:
        return {"spearman": None, "kendall_tau": None}
    spearman = stats.spearmanr(x, y, nan_policy="omit")
    kendall = stats.kendalltau(x, y, nan_policy="omit")
    return {
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
    }


def paired_difference(before: list[float], after: list[float]) -> dict[str, float | None]:
    if len(before) != len(after) or not before:
        return {"mean_delta": None, "p_value": None}
    delta = [a - b for b, a in zip(before, after)]
    test = stats.ttest_rel(after, before) if len(before) > 1 else None
    return {
        "mean_delta": mean(delta),
        "median_delta": median(delta),
        "p_value": float(test.pvalue) if test is not None else None,
    }
