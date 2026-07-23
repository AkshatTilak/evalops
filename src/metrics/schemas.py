"""Pydantic schemas for RAGAS and DeepEval metrics results."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RagasCaseResult(BaseModel):
    """Result of RAGAS metrics for a single test case."""

    case_id: Optional[str] = None
    input_query: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None


class RagasEvalResult(BaseModel):
    """Aggregated result of RAGAS evaluation run."""

    case_results: List[RagasCaseResult] = Field(default_factory=list)
    mean_faithfulness: Optional[float] = None
    mean_answer_relevancy: Optional[float] = None
    mean_context_recall: Optional[float] = None
    mean_context_precision: Optional[float] = None


class DeepEvalMetricScore(BaseModel):
    """Score for a single DeepEval metric."""

    metric_name: str
    score: Optional[float] = None
    threshold: Optional[float] = 0.7
    passed: Optional[bool] = None
    reason: Optional[str] = None


class DeepEvalCaseResult(BaseModel):
    """Result of DeepEval metrics for a single test case."""

    case_id: Optional[str] = None
    input_query: str
    metrics: List[DeepEvalMetricScore] = Field(default_factory=list)


class DeepEvalResult(BaseModel):
    """Aggregated result of DeepEval evaluation run."""

    case_results: List[DeepEvalCaseResult] = Field(default_factory=list)
    mean_scores: Dict[str, float] = Field(default_factory=dict)
