"""Unit test for S5-10b: DB Schema Expansion for Flow Tracing (EvalFlowTrace model)."""

import uuid
from datetime import datetime
from common.models.database import EvalFlowTrace
from projects.evalops.src.models.flow_trace import EvalFlowTrace as ExportedEvalFlowTrace


def test_eval_flow_trace_model_instantiation():
    """Verify EvalFlowTrace model attributes, re-exporting, and serialization structures."""
    trace_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    now = datetime.utcnow()

    trace = EvalFlowTrace(
        id=trace_id,
        run_id=run_id,
        workflow_id="wf_customer_support_v1",
        node_id="node_rag_retrieval",
        node_type="retrieval",
        input_state={"query": "How do I reset password?"},
        output_state={"retrieved_chunks": ["Doc 1", "Doc 2"], "chunk_count": 2},
        latency_ms=124.5,
        timestamp=now
    )

    assert trace.id == trace_id
    assert trace.run_id == run_id
    assert trace.workflow_id == "wf_customer_support_v1"
    assert trace.node_id == "node_rag_retrieval"
    assert trace.node_type == "retrieval"
    assert trace.input_state["query"] == "How do I reset password?"
    assert trace.output_state["chunk_count"] == 2
    assert trace.latency_ms == 124.5
    assert trace.timestamp == now
    assert ExportedEvalFlowTrace is EvalFlowTrace
