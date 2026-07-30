"""Hub-scoped workflow execution trace reader module (S6-07c).

Provides loaders for loading and indexing intermediate node trace execution records
for workflow runs, ensuring tenant isolation by filtering strictly on hub_id.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List
from sqlalchemy import select

from common.models.database import EvalFlowTrace

logger = logging.getLogger("evalops.runner.trace_reader")


@dataclass(frozen=True)
class TraceRecord:
    """Frozen dataclass representing an intermediate workflow node trace execution step."""

    node_id: str
    node_type: str
    sequence: int
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    latency_ms: float
    timestamp: Any


async def load_run_traces(session, *, hub_id: str, run_id: str) -> List[TraceRecord]:
    """Loads all workflow flow traces for a given run, strictly scoped by hub_id."""
    if not hub_id or not run_id:
        return []

    stmt = (
        select(EvalFlowTrace)
        .where(EvalFlowTrace.hub_id == hub_id, EvalFlowTrace.run_id == run_id)
        .order_by(EvalFlowTrace.sequence.asc())
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()

    return [
        TraceRecord(
            node_id=r.node_id,
            node_type=r.node_type or "unknown",
            sequence=r.sequence or 0,
            input_state=r.input_state or {},
            output_state=r.output_state or {},
            latency_ms=r.latency_ms or 0.0,
            timestamp=r.timestamp,
        )
        for r in rows
    ]


def index_by_node(traces: List[TraceRecord]) -> Dict[str, List[TraceRecord]]:
    """Indexes trace records by node_id, preserving execution order."""
    indexed: Dict[str, List[TraceRecord]] = {}
    for trace in traces:
        if not trace.node_id:
            continue
        if trace.node_id not in indexed:
            indexed[trace.node_id] = []
        indexed[trace.node_id].append(trace)
    return indexed
