"""Unit tests for S5-01a: RAGAS Evaluation Runner."""

import pytest
from projects.evalops.src.metrics.ragas_runner import run_ragas_evaluation
from projects.evalops.src.metrics.schemas import RagasEvalResult, RagasCaseResult


@pytest.mark.asyncio
async def test_run_ragas_evaluation_basic():
    """Verify run_ragas_evaluation generates structured scores and aggregate means."""
    test_cases = [
        {
            "id": "case-1",
            "input_query": "What is SyntraFlow?",
            "expected_output": "SyntraFlow is a hybrid retrieval engine.",
            "expected_context": "SyntraFlow provides vector and sparse keyword retrieval.",
        },
        {
            "id": "case-2",
            "input_query": "Explain GuardRoute",
            "expected_output": "GuardRoute is an AI security gateway.",
            "expected_context": "GuardRoute filters prompt injections and scrubs PII.",
        },
    ]

    agent_responses = [
        "SyntraFlow is a hybrid retrieval system for vector and sparse search.",
        "GuardRoute is an AI gateway that filters prompt injection attacks and scrubs PII.",
    ]

    retrieved_contexts = [
        ["SyntraFlow provides vector and sparse keyword retrieval."],
        ["GuardRoute filters prompt injections and scrubs PII."],
    ]

    result: RagasEvalResult = await run_ragas_evaluation(
        test_cases, agent_responses, retrieved_contexts
    )

    assert isinstance(result, RagasEvalResult)
    assert len(result.case_results) == 2
    assert result.mean_faithfulness > 0.0
    assert result.mean_answer_relevancy > 0.0

    c0 = result.case_results[0]
    assert isinstance(c0, RagasCaseResult)
    assert c0.case_id == "case-1"
    assert c0.faithfulness is not None
    assert c0.answer_relevancy is not None
