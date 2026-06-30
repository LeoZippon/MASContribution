"""Feature extraction for contribution analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mas_contribution_bench.data.schemas import TraceFeatureRecord
from mas_contribution_bench.graphs.architectures import topology_features


def extract_trace_features(
    run_id: str,
    task_id: str,
    architecture,
    trace_records: list[Any],
) -> list[TraceFeatureRecord]:
    sent = defaultdict(int)
    received = defaultdict(int)
    sent_tokens = defaultdict(int)
    received_tokens = defaultdict(int)
    lengths = defaultdict(list)
    tool_calls = defaultdict(int)
    tool_success = defaultdict(list)
    topo = topology_features(architecture)
    for record in trace_records:
        sender = getattr(record, "sender", None) or (record.get("sender") if isinstance(record, dict) else None)
        receiver = getattr(record, "receiver", None) or (record.get("receiver") if isinstance(record, dict) else None)
        content = getattr(record, "content", None) or (record.get("content") if isinstance(record, dict) else "")
        input_tokens = getattr(record, "input_tokens", 0) if not isinstance(record, dict) else record.get("input_tokens", 0)
        output_tokens = getattr(record, "output_tokens", 0) if not isinstance(record, dict) else record.get("output_tokens", 0)
        if sender in architecture.roles:
            sent[sender] += 1
            sent_tokens[sender] += int(output_tokens or 0)
            lengths[sender].append(len(content or ""))
        if receiver in architecture.roles:
            received[receiver] += 1
            received_tokens[receiver] += int(input_tokens or 0)
    records = []
    for role in architecture.roles:
        values = lengths.get(role, [])
        records.append(
            TraceFeatureRecord(
                run_id=run_id,
                task_id=task_id,
                architecture_id=architecture.architecture_id,
                agent_id=role,
                role=role,
                messages_sent=sent[role],
                messages_received=received[role],
                tokens_sent=sent_tokens[role],
                tokens_received=received_tokens[role],
                avg_message_length=(sum(values) / len(values)) if values else None,
                tool_call_count=tool_calls[role],
                tool_success_rate=(sum(tool_success[role]) / len(tool_success[role])) if tool_success[role] else None,
                topology_features=topo.get(role, {}),
            )
        )
    return records
