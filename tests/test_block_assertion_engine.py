"""Unit test for S5-10c: Intermediate Block Assertion Engine."""

import uuid
from projects.evalops.src.runner.block_assertion_engine import BlockAssertionEngine


def test_block_assertion_engine_operators():
    """Test equals, contains, regex, threshold, and json_schema operators."""
    engine = BlockAssertionEngine()
    run_id = str(uuid.uuid4())

    trace = {
        "node_id": "node_router",
        "output_state": {
            "target_agent": "support_agent",
            "confidence": 0.95,
            "tags": ["urgent", "account"],
            "response_text": "Redirecting user to Support Agent."
        }
    }

    # 1. Equals
    res_eq = engine.evaluate_assertion(
        trace,
        {"node_id": "node_router", "field_path": "output_state.target_agent", "operator": "equals", "expected_value": "support_agent", "metric_name": "router_agent_equals"},
        run_id
    )
    assert res_eq.passed is True
    assert res_eq.metric_score == 1.0

    # 2. Contains
    res_cnt = engine.evaluate_assertion(
        trace,
        {"node_id": "node_router", "field_path": "output_state.tags", "operator": "contains", "expected_value": "urgent", "metric_name": "router_tag_contains"},
        run_id
    )
    assert res_cnt.passed is True

    # 3. Regex
    res_reg = engine.evaluate_assertion(
        trace,
        {"node_id": "node_router", "field_path": "output_state.response_text", "operator": "regex", "expected_value": r"Redirecting.*Support", "metric_name": "router_regex"},
        run_id
    )
    assert res_reg.passed is True

    # 4. Threshold
    res_thresh = engine.evaluate_assertion(
        trace,
        {"node_id": "node_router", "field_path": "output_state.confidence", "operator": "threshold", "min_threshold": 0.90, "max_threshold": 1.0, "metric_name": "router_confidence_threshold"},
        run_id
    )
    assert res_thresh.passed is True

    # 5. Threshold fail
    res_thresh_fail = engine.evaluate_assertion(
        trace,
        {"node_id": "node_router", "field_path": "output_state.confidence", "operator": "threshold", "min_threshold": 0.98, "metric_name": "router_high_thresh"},
        run_id
    )
    assert res_thresh_fail.passed is False

    # 6. JSON Schema check
    res_schema = engine.evaluate_assertion(
        trace,
        {"node_id": "node_router", "field_path": "output_state", "operator": "json_schema", "expected_value": {"target_agent": None, "confidence": None}, "metric_name": "router_schema"},
        run_id
    )
    assert res_schema.passed is True
