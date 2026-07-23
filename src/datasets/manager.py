"""Dataset Manager for EvalOps Test Suites and Test Cases (S5-01c).

Provides CRUD operations, CSV/JSON import/export, and cloning of evaluation test suites.
"""

import csv
import io
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.models.database import EvalTestCase, EvalTestSuite

logger = logging.getLogger("evalops.datasets.manager")


# --- Test Suite Operations ---


async def create_suite(
    db: AsyncSession,
    agent_id: str,
    name: str,
    description: Optional[str] = None,
) -> EvalTestSuite:
    """Create a new evaluation test suite."""
    suite = EvalTestSuite(
        id=str(uuid.uuid4()),
        agent_id=agent_id,
        name=name,
        description=description,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(suite)
    await db.commit()
    await db.refresh(suite)
    logger.info("Created EvalTestSuite '%s' (ID: %s)", suite.name, suite.id)
    return suite


async def list_suites(
    db: AsyncSession, agent_id: Optional[str] = None
) -> List[EvalTestSuite]:
    """Retrieve all test suites, optionally filtered by agent_id."""
    stmt = select(EvalTestSuite).order_by(EvalTestSuite.created_at.desc())
    if agent_id:
        stmt = stmt.where(EvalTestSuite.agent_id == agent_id)
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def get_suite(db: AsyncSession, suite_id: str) -> Optional[EvalTestSuite]:
    """Retrieve a single test suite by ID."""
    stmt = select(EvalTestSuite).where(EvalTestSuite.id == suite_id)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()


async def update_suite(
    db: AsyncSession,
    suite_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[EvalTestSuite]:
    """Update metadata for an existing test suite."""
    suite = await get_suite(db, suite_id)
    if not suite:
        return None
    if name is not None:
        suite.name = name
    if description is not None:
        suite.description = description
    suite.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(suite)
    return suite


async def delete_suite(db: AsyncSession, suite_id: str) -> bool:
    """Delete a test suite and its cascading test cases."""
    suite = await get_suite(db, suite_id)
    if not suite:
        return False
    await db.delete(suite)
    await db.commit()
    logger.info("Deleted EvalTestSuite ID: %s", suite_id)
    return True


async def clone_suite(
    db: AsyncSession, suite_id: str, new_name: Optional[str] = None
) -> Optional[EvalTestSuite]:
    """Clone an existing test suite and all of its test cases."""
    original_suite = await get_suite(db, suite_id)
    if not original_suite:
        return None

    clone_name = new_name or f"{original_suite.name} (Copy)"
    cloned_suite = await create_suite(
        db,
        agent_id=original_suite.agent_id,
        name=clone_name,
        description=original_suite.description,
    )

    cases = await list_test_cases(db, suite_id)
    for c in cases:
        await add_test_case(
            db,
            suite_id=cloned_suite.id,
            input_query=c.input_query,
            expected_output=c.expected_output,
            expected_context=c.expected_context,
        )

    logger.info("Cloned suite %s to %s with %d cases", suite_id, cloned_suite.id, len(cases))
    return cloned_suite


# --- Test Case Operations ---


async def add_test_case(
    db: AsyncSession,
    suite_id: str,
    input_query: str,
    expected_output: Optional[str] = None,
    expected_context: Optional[str] = None,
) -> EvalTestCase:
    """Add a test case to a suite."""
    case = EvalTestCase(
        id=str(uuid.uuid4()),
        suite_id=suite_id,
        input_query=input_query,
        expected_output=expected_output,
        expected_context=expected_context,
        created_at=datetime.utcnow(),
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)
    return case


async def list_test_cases(db: AsyncSession, suite_id: str) -> List[EvalTestCase]:
    """List all test cases for a specified suite."""
    stmt = select(EvalTestCase).where(EvalTestCase.suite_id == suite_id).order_by(EvalTestCase.created_at.asc())
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def update_test_case(
    db: AsyncSession,
    case_id: str,
    input_query: Optional[str] = None,
    expected_output: Optional[str] = None,
    expected_context: Optional[str] = None,
) -> Optional[EvalTestCase]:
    """Update an existing test case."""
    stmt = select(EvalTestCase).where(EvalTestCase.id == case_id)
    res = await db.execute(stmt)
    case = res.scalar_one_or_none()
    if not case:
        return None
    if input_query is not None:
        case.input_query = input_query
    if expected_output is not None:
        case.expected_output = expected_output
    if expected_context is not None:
        case.expected_context = expected_context
    await db.commit()
    await db.refresh(case)
    return case


async def delete_test_case(db: AsyncSession, case_id: str) -> bool:
    """Delete a specific test case."""
    stmt = select(EvalTestCase).where(EvalTestCase.id == case_id)
    res = await db.execute(stmt)
    case = res.scalar_one_or_none()
    if not case:
        return False
    await db.delete(case)
    await db.commit()
    return True


# --- Bulk Import / Export ---


async def import_cases_from_csv(
    db: AsyncSession, suite_id: str, csv_content: str
) -> int:
    """Bulk import test cases from a CSV string."""
    f = io.StringIO(csv_content.strip())
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        q = row.get("input_query") or row.get("question") or row.get("prompt")
        if not q:
            continue
        exp_out = row.get("expected_output") or row.get("ground_truth") or row.get("answer")
        exp_ctx = row.get("expected_context") or row.get("context")
        await add_test_case(
            db,
            suite_id=suite_id,
            input_query=q,
            expected_output=exp_out,
            expected_context=exp_ctx,
        )
        count += 1
    logger.info("Imported %d test cases from CSV into suite %s", count, suite_id)
    return count


async def import_cases_from_json(
    db: AsyncSession, suite_id: str, json_data: List[Dict[str, Any]]
) -> int:
    """Bulk import test cases from a list of dicts or JSON data."""
    count = 0
    for item in json_data:
        q = item.get("input_query") or item.get("question") or item.get("prompt")
        if not q:
            continue
        exp_out = item.get("expected_output") or item.get("ground_truth") or item.get("answer")
        exp_ctx = item.get("expected_context") or item.get("contexts") or item.get("context")

        if isinstance(exp_ctx, list):
            exp_ctx = " ; ".join(exp_ctx)

        await add_test_case(
            db,
            suite_id=suite_id,
            input_query=q,
            expected_output=exp_out,
            expected_context=exp_ctx,
        )
        count += 1
    logger.info("Imported %d test cases from JSON into suite %s", count, suite_id)
    return count


async def export_suite_to_json(db: AsyncSession, suite_id: str) -> Dict[str, Any]:
    """Export suite metadata and all contained test cases to JSON dict format."""
    suite = await get_suite(db, suite_id)
    if not suite:
        raise ValueError(f"Suite '{suite_id}' not found.")

    cases = await list_test_cases(db, suite_id)
    return {
        "suite_id": suite.id,
        "agent_id": suite.agent_id,
        "name": suite.name,
        "description": suite.description,
        "created_at": suite.created_at.isoformat() if suite.created_at else None,
        "test_cases": [
            {
                "id": c.id,
                "input_query": c.input_query,
                "expected_output": c.expected_output,
                "expected_context": c.expected_context,
            }
            for c in cases
        ],
    }
