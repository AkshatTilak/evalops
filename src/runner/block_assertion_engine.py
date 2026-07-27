"""BlockAssertionEngine for evaluating assertions on intermediate LangGraph workflow node traces.
S5-10c: Evaluates block outputs using operators: equals, contains, regex, threshold, json_schema.
"""

import re
import uuid
import logging
from typing import Dict, Any, List, Optional
from common.models.database import EvalMetricResult

logger = logging.getLogger("evalops.block_assertion_engine")


class BlockAssertionEngine:
    """Evaluates intermediate node traces against block assertions and outputs metric result models."""

    def __init__(self):
        pass

    def evaluate_assertion(
        self,
        trace: Dict[str, Any],
        assertion_config: Dict[str, Any],
        run_id: str
    ) -> EvalMetricResult:
        """Evaluates a single block assertion against a trace node state dictionary.
        
        `assertion_config` structure:
        {
            "node_id": "node_router",
            "field_path": "output_state.target_agent", # dot notation into trace dict
            "operator": "equals" | "contains" | "regex" | "threshold" | "json_schema",
            "expected_value": Any,
            "min_threshold": float (optional for threshold),
            "max_threshold": float (optional for threshold),
            "metric_name": "router_agent_selection"
        }
        """
        metric_name = assertion_config.get("metric_name", f"block_assertion_{assertion_config.get('node_id', 'unknown')}")
        operator = assertion_config.get("operator", "equals").lower()
        field_path = assertion_config.get("field_path", "")
        expected_value = assertion_config.get("expected_value")

        actual_value = self._extract_field_value(trace, field_path)

        passed = False
        reason = ""

        try:
            if operator == "equals":
                passed = actual_value == expected_value
                reason = f"Actual '{actual_value}' equals expected '{expected_value}'" if passed else f"Expected '{expected_value}', but got '{actual_value}'"

            elif operator == "contains":
                if isinstance(actual_value, (list, set, tuple)):
                    passed = expected_value in actual_value
                elif isinstance(actual_value, str):
                    passed = str(expected_value) in actual_value
                else:
                    passed = False
                reason = f"Actual value contains '{expected_value}'" if passed else f"Actual '{actual_value}' does not contain '{expected_value}'"

            elif operator == "regex":
                if actual_value is None:
                    passed = False
                    reason = "Actual value is None, failed regex match."
                else:
                    pattern = str(expected_value)
                    match = re.search(pattern, str(actual_value))
                    passed = match is not None
                    reason = f"Pattern r'{pattern}' matched actual value" if passed else f"Pattern r'{pattern}' failed to match '{actual_value}'"

            elif operator == "threshold":
                num_val = float(actual_value) if actual_value is not None else 0.0
                min_val = assertion_config.get("min_threshold")
                max_val = assertion_config.get("max_threshold")

                min_ok = True if min_val is None else num_val >= float(min_val)
                max_ok = True if max_val is None else num_val <= float(max_val)

                passed = min_ok and max_ok
                reason = f"Value {num_val} satisfies bounds [{min_val}, {max_val}]" if passed else f"Value {num_val} violates bounds [{min_val}, {max_val}]"

            elif operator == "json_schema":
                # Check key presence and types
                if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                    missing_keys = [k for k in expected_value.keys() if k not in actual_value]
                    passed = len(missing_keys) == 0
                    reason = f"Schema keys satisfied" if passed else f"Missing required schema keys: {missing_keys}"
                else:
                    passed = isinstance(actual_value, dict)
                    reason = "Output is valid JSON object" if passed else f"Expected JSON object, got {type(actual_value)}"

            else:
                passed = False
                reason = f"Unsupported operator '{operator}'"

        except Exception as e:
            passed = False
            reason = f"Exception evaluating assertion: {str(e)}"

        score = 1.0 if passed else 0.0

        return EvalMetricResult(
            id=str(uuid.uuid4()),
            run_id=run_id,
            metric_name=metric_name,
            metric_score=score,
            metric_reason=reason,
            framework="intermediate_block",
            threshold=1.0,
            passed=passed
        )

    def evaluate_batch(
        self,
        traces: List[Dict[str, Any]],
        assertion_configs: List[Dict[str, Any]],
        run_id: str
    ) -> List[EvalMetricResult]:
        """Evaluates a batch of assertions across multiple node trace steps."""
        results: List[EvalMetricResult] = []
        trace_by_node = {t.get("node_id"): t for t in traces if t.get("node_id")}

        for config in assertion_configs:
            target_node_id = config.get("node_id")
            trace = trace_by_node.get(target_node_id, {})
            result = self.evaluate_assertion(trace, config, run_id)
            results.append(result)

        return results

    def _extract_field_value(self, data: Dict[str, Any], path: str) -> Any:
        """Navigates dot-notated field path in trace dict (e.g. 'output_state.chunk_count')."""
        if not path:
            return None
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
