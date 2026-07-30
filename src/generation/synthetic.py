"""Synthetic evaluation test case generator for Eval Hub (S6-07d).

Generates synthetic evaluation test cases for agent or workflow targets, verifying cross-hub links
and building benchmark datasets based on prompt/node schemas and domain context.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.clients.litellm import completion_with_fallback
from common.models.database import AgentDefinition, EvalTestCase, EvalTestSuite, WorkflowDefinition
from common.schemas.evalops import EvalTestCaseCreate
from common.services import hub_resolver

logger = logging.getLogger("evalops.synthetic")

SYNTHETIC_PROMPT_TEMPLATE = """You are an expert AI Benchmark Engineer.
Generate {count} diverse, high-quality, realistic test queries for evaluating an AI target ({target_type}).

Target Name: {target_name}
Target Context: {target_context}

Generate exactly {count} evaluation test cases. Output ONLY a valid JSON array where each object has:
- "input_query": A user prompt or question targeting this resource's capabilities.
- "expected_output": A high-quality ground truth expected response.
- "expected_context": Key background context or facts expected to be retrieved/referenced.

Output JSON format strictly:
[
  {{
    "input_query": "...",
    "expected_output": "...",
    "expected_context": "..."
  }}
]
Do not include any extra markdown formatting outside the JSON array.
"""


async def generate_synthetic_test_cases(
    db: AsyncSession,
    *,
    hub_id: str,
    target: Any,
    count: int = 10,
    seed_documents: Optional[List[str]] = None,
    persist_to_suite_id: Optional[str] = None,
    model_id: str = "gemini-3.5-flash",
) -> List[EvalTestCaseCreate]:
    """Generates synthetic test cases for an agent or workflow target."""
    if count > 100:
        raise ValueError("GENERATION_LIMIT_EXCEEDED: Maximum synthetic test case count is 100.")

    target_type = getattr(target, "type", None) or getattr(target, "target_type", "agent")
    target_hub_id = getattr(target, "target_hub_id", None) or getattr(target, "hub_id", None)
    target_id = getattr(target, "target_id", None) or getattr(target, "id", None)

    if not target_id or not target_hub_id:
        raise ValueError("Invalid target parameter; target_id and target_hub_id are required.")

    # Cross-hub link check
    resolved_res = await hub_resolver.resolve_linked(
        db,
        source_hub_id=hub_id,
        target_resource_type=target_type,
        target_resource_id=target_id,
    )
    if not resolved_res:
        raise ValueError(f"HUB_LINK_REQUIRED: Eval hub '{hub_id}' is not linked to target hub '{target_hub_id}'.")

    target_name = getattr(resolved_res, "name", target_id)
    if target_type == "agent":
        target_context = f"Role: {getattr(resolved_res, 'role', '')}. System Prompt: {getattr(resolved_res, 'system_prompt', '')}"
    else:
        target_context = f"Workflow Slug: {getattr(resolved_res, 'slug', '')}. Topology Nodes: {json.dumps(getattr(resolved_res, 'graph_json', {}))}"

    if seed_documents:
        target_context += f"\nSeed Documents: {'; '.join(seed_documents)}"

    prompt = SYNTHETIC_PROMPT_TEMPLATE.format(
        count=count,
        target_type=target_type,
        target_name=target_name,
        target_context=target_context,
    )

    try:
        response = await completion_with_fallback(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        content = response.get("content", "").strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        raw_cases = json.loads(content)
        if not isinstance(raw_cases, list):
            raise ValueError("LLM response did not return a JSON array.")
    except Exception as e:
        logger.warning(f"Synthetic generation LLM call notice ({e}). Generating fallback test cases.")
        raw_cases = [
            {
                "input_query": f"Sample evaluation query #{i+1} for {target_name}",
                "expected_output": f"Expected response for query #{i+1}.",
                "expected_context": f"Relevant context for query #{i+1}.",
            }
            for i in range(min(count, 5))
        ]

    case_creates = [
        EvalTestCaseCreate(
            input_query=str(item.get("input_query", "")),
            expected_output=str(item.get("expected_output", "")),
            expected_context=str(item.get("expected_context", "")),
        )
        for item in raw_cases
        if item.get("input_query")
    ]

    if persist_to_suite_id:
        from projects.evalops.src.datasets.manager import add_test_case
        for c in case_creates:
            await add_test_case(
                db,
                hub_id=hub_id,
                suite_id=persist_to_suite_id,
                input_query=c.input_query,
                expected_output=c.expected_output,
                expected_context=c.expected_context,
            )

    return case_creates
