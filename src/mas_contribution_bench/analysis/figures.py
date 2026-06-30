"""Figure-data preparation."""

from __future__ import annotations

from typing import Any


def contribution_bar_data(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "architecture_id": record.get("architecture_id"),
            "dataset": record.get("dataset"),
            "agent_id": record.get("agent_id"),
            "role": record.get("role"),
            "method": record.get("method"),
            "score": record.get("score"),
        }
        for record in records
    ]


def topology_scatter_data(feature_records: list[dict[str, Any]], attribution_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_index = {
        (r.get("task_id"), r.get("architecture_id"), r.get("agent_id")): r
        for r in feature_records
    }
    rows = []
    for attr in attribution_records:
        key = (attr.get("task_id"), attr.get("architecture_id"), attr.get("agent_id"))
        feat = feature_index.get(key, {})
        topo = feat.get("topology_features", {}) or {}
        rows.append({**attr, **{f"topology_{k}": v for k, v in topo.items()}})
    return rows
