"""Unit tests for S5-01b: DeepEval Evaluation Runner."""

import pytest
from projects.evalops.src.metrics.deepeval_runner import run_deepeval_evaluation
from projects.evalops.src.metrics.schemas import DeepEvalResult, DeepEvalCaseResult, DeepEvalMetricScore


@pytest.mark.asyncio
async def test_run_deepeval_evaluation_metrics():
    """Verify run_deepeval_evaluation computes requested metrics and thresholds."""
    test_cases = [
        {
            "id": "case-de-1",
            "input_query": "Summarize user access security",
            "expected_output": "Access control uses OAuth2 and JWT bearer tokens.",
        }
    ]

    agent_responses = ["Access control relies on OAuth2 and JWT authentication tokens."]
    retrieved_contexts = [["Access control uses OAuth2 and JWT bearer tokens."]]

    result: DeepEvalResult = await run_deepeval_evaluation(
        test_cases=test_cases,
        agent_responses=agent_responses,
        retrieved_contexts=retrieved_contexts,
        requested_metrics=["hallucination", "toxicity", "faithfulness"],
        thresholds={"faithfulness": 0.8},
    )

    assert isinstance(result, DeepEvalResult)
    assert len(result.case_results) == 1

    c0: DeepEvalCaseResult = result.case_results[0]
    assert c0.case_id == "case-de-1"
    assert len(c0.metrics) == 3

    m_names = [m.metric_name for m in c0.metrics]
    assert "hallucination" in m_names
    assert "toxicity" in m_names
    assert "faithfulness" in m_names

    faith_metric = next(m for m in c0.metrics if m.metric_name == "faithfulness")
    assert faith_metric.threshold == 0.8
    assert faith_metric.passed is True
