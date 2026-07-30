"""BlockAssertionEngine for evaluating intermediate workflow node trace assertions (S6-07c).

Supports V6 operators: equals, contains, not_contains, regex, json_path, latency_under,
occurrence selection (first, last, all), bounded regex matching, and hub-scoped trace isolation.
"""

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select

from common.models.database import EvalFlowTrace, EvalMetricResult, EvalTestCase
from projects.evalops.src.runner.trace_reader import TraceRecord, index_by_node, load_run_traces

logger = logging.getLogger("evalops.block_assertion_engine")

MAX_REGEX_PATTERN_LENGTH = 256
REGEX_TIMEOUT_SECONDS = 1.0


class BlockAssertionEngine:
    """Evaluates intermediate node traces against assertions and outputs EvalMetricResult records."""

    def _extract_field_value(self, data: Dict[str, Any], path: Optional[str]) -> Any:
        """Navigates dot-notated field path in state dict (defaulting to output_state)."""
        if not path:
            return data.get("output_state", data) if isinstance(data, dict) and "output_state" in data else data

        # Strip output_state prefix if provided
        if path.startswith("output_state."):
            path = path[len("output_state."):]
        elif path == "output_state":
            return data.get("output_state", data) if isinstance(data, dict) and "output_state" in data else data

        current = data.get("output_state", data) if isinstance(data, dict) and "output_state" in data else data

        parts = path.split(".")
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def evaluate_single_trace(
        self,
        trace: Optional[TraceRecord],
        assertion_type: str,
        assertion_config: Dict[str, Any],
        expected_value: Optional[str],
        run_id: str,
        case_id: Optional[str] = None,
        node_id: Optional[str] = None,
        is_agent_target: bool = False,
        run_failed: bool = False,
    ) -> EvalMetricResult:
        """Evaluates a single node trace against an assertion operator."""
        m_name = f"node_assertion.{node_id or 'unknown'}.{assertion_type}"

        if is_agent_target:
            return EvalMetricResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                test_case_id=case_id,
                node_id=node_id,
                assertion_type=assertion_type,
                metric_name=m_name,
                metric_score=0.0,
                metric_reason="NODE_ASSERTION_ON_AGENT_TARGET",
                framework="node_assertion",
                threshold=1.0,
                passed=False,
            )

        if run_failed:
            return EvalMetricResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                test_case_id=case_id,
                node_id=node_id,
                assertion_type=assertion_type,
                metric_name=m_name,
                metric_score=0.0,
                metric_reason="RUN_FAILED_BEFORE_NODE",
                framework="node_assertion",
                threshold=1.0,
                passed=False,
            )

        if not trace:
            return EvalMetricResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                test_case_id=case_id,
                node_id=node_id,
                assertion_type=assertion_type,
                metric_name=m_name,
                metric_score=0.0,
                metric_reason=f"NODE_NOT_EXECUTED: node '{node_id}' did not run",
                framework="node_assertion",
                threshold=1.0,
                passed=False,
            )

        field_path = assertion_config.get("field_path")
        case_sensitive = assertion_config.get("case_sensitive", True)
        op = (assertion_type or "equals").lower()

        # Handle latency_under operator directly from trace.latency_ms
        if op == "latency_under":
            try:
                target_lat = float(expected_value) if expected_value is not None else 0.0
                actual_lat = float(trace.latency_ms)
                passed = actual_lat < target_lat
                reason = (
                    f"Latency {actual_lat}ms is under threshold {target_lat}ms"
                    if passed
                    else f"Latency {actual_lat}ms exceeded threshold {target_lat}ms"
                )
            except ValueError as ve:
                passed = False
                reason = f"Invalid threshold value for latency_under: {expected_value}"

            return EvalMetricResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                test_case_id=case_id,
                node_id=node_id,
                assertion_type=assertion_type,
                metric_name=m_name,
                metric_score=1.0 if passed else 0.0,
                metric_reason=reason[:500],
                framework="node_assertion",
                threshold=1.0,
                passed=passed,
            )

        # Extract field value
        actual_val = self._extract_field_value(trace.output_state, field_path)

        if str(actual_val) == "[REDACTED]":
            return EvalMetricResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                test_case_id=case_id,
                node_id=node_id,
                assertion_type=assertion_type,
                metric_name=m_name,
                metric_score=0.0,
                metric_reason="FIELD_REDACTED",
                framework="node_assertion",
                threshold=1.0,
                passed=False,
            )

        passed = False
        reason = ""

        if op == "equals":
            if not case_sensitive and isinstance(actual_val, str) and isinstance(expected_value, str):
                passed = actual_val.lower() == expected_value.lower()
            else:
                passed = str(actual_val) == str(expected_value) if actual_val is not None else False
            reason = (
                f"Actual '{actual_val}' equals expected '{expected_value}'"
                if passed
                else f"Expected '{expected_value}', but got '{actual_val}'"
            )

        elif op == "contains":
            if actual_val is None:
                passed = False
                reason = f"Field is missing, cannot contain '{expected_value}'"
            elif isinstance(actual_val, (list, set, tuple)):
                passed = expected_value in actual_val or str(expected_value) in [str(x) for x in actual_val]
                reason = f"List contains '{expected_value}'" if passed else f"List does not contain '{expected_value}'"
            else:
                act_str = str(actual_val)
                exp_str = str(expected_value) if expected_value is not None else ""
                if not case_sensitive:
                    act_str, exp_str = act_str.lower(), exp_str.lower()
                passed = exp_str in act_str
                reason = f"Text contains '{expected_value}'" if passed else f"Text does not contain '{expected_value}'"

        elif op == "not_contains":
            if actual_val is None:
                passed = True
                reason = "Field is absent; not_contains passed by default."
            elif isinstance(actual_val, (list, set, tuple)):
                passed = expected_value not in actual_val and str(expected_value) not in [str(x) for x in actual_val]
                reason = f"List does not contain '{expected_value}'" if passed else f"List contains '{expected_value}'"
            else:
                act_str = str(actual_val)
                exp_str = str(expected_value) if expected_value is not None else ""
                if not case_sensitive:
                    act_str, exp_str = act_str.lower(), exp_str.lower()
                passed = exp_str not in act_str
                reason = f"Text does not contain '{expected_value}'" if passed else f"Text contains '{expected_value}'"

        elif op == "regex":
            pattern = str(expected_value) if expected_value is not None else ""
            if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
                passed = False
                reason = f"REGEX_PATTERN_TOO_LONG: Pattern length {len(pattern)} exceeds max {MAX_REGEX_PATTERN_LENGTH}"
            elif actual_val is None:
                passed = False
                reason = "Actual value is None, failed regex match."
            else:
                try:
                    match = re.search(pattern, str(actual_val), flags=re.DOTALL)
                    passed = match is not None
                    reason = f"Regex r'{pattern}' matched actual value" if passed else f"Regex r'{pattern}' failed to match"
                except Exception as re_err:
                    passed = False
                    reason = f"REGEX_ERROR: {re_err}"

        elif op == "json_path":
            jpath = assertion_config.get("json_path", "")
            if not jpath:
                passed = False
                reason = "Missing json_path configuration"
            else:
                # Basic json path lookup fallback
                field_val = self._extract_field_value(trace.output_state, jpath)
                if field_val is not None:
                    if expected_value is not None:
                        passed = str(field_val) == str(expected_value)
                        reason = f"JSONPath '{jpath}' matched expected value" if passed else f"JSONPath '{jpath}' got '{field_val}'"
                    else:
                        passed = True
                        reason = f"JSONPath '{jpath}' evaluated to non-empty result"
                else:
                    passed = False
                    reason = f"JSONPath '{jpath}' evaluated to empty result"

        else:
            passed = False
            reason = f"Unsupported assertion_type '{assertion_type}'"

        return EvalMetricResult(
            id=str(uuid.uuid4()),
            run_id=run_id,
            test_case_id=case_id,
            node_id=node_id,
            assertion_type=assertion_type,
            metric_name=m_name,
            metric_score=1.0 if passed else 0.0,
            metric_reason=reason[:500],
            framework="node_assertion",
            threshold=1.0,
            passed=passed,
        )


async def evaluate_node_assertions(
    session,
    *,
    hub_id: str,
    run_id: str,
    eval_run_id: str,
    cases: List[EvalTestCase],
    outcome_by_case: Optional[Dict[str, Any]] = None,
    is_agent_target: bool = False,
) -> List[EvalMetricResult]:
    """Evaluates node assertions for a set of test cases against workflow trace records."""
    node_cases = [c for c in cases if getattr(c, "node_id", None)]
    if not node_cases:
        return []

    engine = BlockAssertionEngine()
    results: List[EvalMetricResult] = []

    if is_agent_target:
        for c in node_cases:
            res = engine.evaluate_single_trace(
                trace=None,
                assertion_type=c.assertion_type or "equals",
                assertion_config=c.assertion_config or {},
                expected_value=c.expected_value,
                run_id=eval_run_id,
                case_id=c.id,
                node_id=c.node_id,
                is_agent_target=True,
            )
            results.append(res)
        return results

    traces = await load_run_traces(session, hub_id=hub_id, run_id=run_id)
    indexed_traces = index_by_node(traces)

    for case in node_cases:
        node_id = case.node_id
        matching_traces = indexed_traces.get(node_id, [])

        outcome = (outcome_by_case or {}).get(case.id)
        run_failed = outcome is not None and outcome.error is not None

        config = case.assertion_config or {}
        occurrence = config.get("occurrence", "last")

        if occurrence == "all":
            if not matching_traces:
                res = engine.evaluate_single_trace(
                    trace=None,
                    assertion_type=case.assertion_type or "equals",
                    assertion_config=config,
                    expected_value=case.expected_value,
                    run_id=eval_run_id,
                    case_id=case.id,
                    node_id=node_id,
                    run_failed=run_failed,
                )
                results.append(res)
            else:
                all_passed = True
                sub_reasons = []
                for tr in matching_traces:
                    sub_res = engine.evaluate_single_trace(
                        trace=tr,
                        assertion_type=case.assertion_type or "equals",
                        assertion_config=config,
                        expected_value=case.expected_value,
                        run_id=eval_run_id,
                        case_id=case.id,
                        node_id=node_id,
                    )
                    if not sub_res.passed:
                        all_passed = False
                        sub_reasons.append(sub_res.metric_reason)

                m_name = f"node_assertion.{node_id}.{case.assertion_type}"
                results.append(
                    EvalMetricResult(
                        id=str(uuid.uuid4()),
                        run_id=eval_run_id,
                        test_case_id=case.id,
                        node_id=node_id,
                        assertion_type=case.assertion_type,
                        metric_name=m_name,
                        metric_score=1.0 if all_passed else 0.0,
                        metric_reason=f"All {len(matching_traces)} occurrences passed" if all_passed else f"Occurrence failure: {'; '.join(sub_reasons)}",
                        framework="node_assertion",
                        threshold=1.0,
                        passed=all_passed,
                    )
                )
        else:
            selected_trace = None
            if matching_traces:
                selected_trace = matching_traces[0] if occurrence == "first" else matching_traces[-1]

            res = engine.evaluate_single_trace(
                trace=selected_trace,
                assertion_type=case.assertion_type or "equals",
                assertion_config=config,
                expected_value=case.expected_value,
                run_id=eval_run_id,
                case_id=case.id,
                node_id=node_id,
                run_failed=run_failed,
            )
            results.append(res)

    return results
