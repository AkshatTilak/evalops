"""Hub-Scoped Dataset Manager for Eval Hub Test Suites and Test Cases (S6-07d).

Provides hub-isolated CRUD operations, polymorphic target resolution, CSV/JSON import/export,
suite retarget protection, and suite health diagnostics.
"""

import csv
import io
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from sqlalchemy import select

from common.models.database import (
    EVAL_TARGET_TYPES,
    NODE_ASSERTION_TYPES,
    EvalRunHistory,
    EvalTestCase,
    EvalTestSuite,
    Hub,
)
from common.services import hub_resolver

logger = logging.getLogger("evalops.datasets.manager")


# --- Helper to extract target attributes ---
def _parse_target(target: Any) -> tuple[str, str, str]:
    if hasattr(target, "type"):
        t_type = getattr(target, "type")
        t_hub_id = getattr(target, "target_hub_id")
        t_id = getattr(target, "target_id")
    elif isinstance(target, dict):
        t_type = target.get("type") or target.get("target_type")
        t_hub_id = target.get("target_hub_id") or target.get("hub_id")
        t_id = target.get("target_id") or target.get("id")
    else:
        raise ValueError("Invalid target format.")

    if t_type not in EVAL_TARGET_TYPES:
        raise ValueError(f"Invalid target_type: '{t_type}'. Must be one of {EVAL_TARGET_TYPES}")
    return t_type, t_hub_id, t_id


# --- Test Suite Operations ---


async def create_suite(
    db,
    *,
    hub_id: str,
    name: str,
    description: Optional[str] = None,
    target: Any = None,
    agent_id: Optional[str] = None,
) -> EvalTestSuite:
    """Creates a new evaluation test suite with cross-hub target security validation."""
    # Check suite name uniqueness in hub
    existing_stmt = select(EvalTestSuite).where(EvalTestSuite.hub_id == hub_id, EvalTestSuite.name == name)
    existing_res = await db.execute(existing_stmt)
    if existing_res.scalar_one_or_none():
        raise ValueError(f"SUITE_NAME_TAKEN: A suite named '{name}' already exists in this hub.")

    if target:
        target_type, target_hub_id, target_id = _parse_target(target)
    elif agent_id:
        target_type = "agent"
        target_id = agent_id
        # Resolve target hub_id from agent
        resolved = await hub_resolver.resolve_resource(db, resource_type="agent", resource_id=agent_id)
        if not resolved:
            raise ValueError(f"EVAL_TARGET_MISSING: Agent '{agent_id}' not found.")
        target_hub_id = resolved.hub_id
    else:
        raise ValueError("Either target or agent_id must be provided.")

    # Cross-hub link check
    resolved_res = await hub_resolver.resolve_linked(
        db,
        source_hub_id=hub_id,
        target_resource_type=target_type,
        target_resource_id=target_id,
    )
    if not resolved_res:
        raise ValueError(f"HUB_LINK_REQUIRED: Hub '{hub_id}' is not linked to target hub '{target_hub_id}'.")

    res_hub_id = getattr(resolved_res, "hub_id", None)
    if res_hub_id != target_hub_id:
        raise ValueError(f"CROSS_HUB_REFERENCE_MISMATCH: Target hub '{res_hub_id}' does not match expected '{target_hub_id}'.")

    # Check target hub active
    hub_res = await db.execute(select(Hub).where(Hub.id == res_hub_id))
    hub_obj = hub_res.scalar_one_or_none()
    if hub_obj and hub_obj.is_archived:
        raise ValueError(f"HUB_ARCHIVED: Target hub '{res_hub_id}' is archived.")

    suite = EvalTestSuite(
        id=str(uuid.uuid4()),
        hub_id=hub_id,
        name=name,
        description=description,
        target_type=target_type,
        target_hub_id=target_hub_id,
        target_id=target_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(suite)
    await db.commit()
    await db.refresh(suite)
    logger.info("Created EvalTestSuite '%s' (ID: %s, Hub: %s)", suite.name, suite.id, hub_id)
    return suite


async def list_suites(
    db,
    *,
    hub_id: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
) -> List[EvalTestSuite]:
    """Retrieves test suites strictly scoped to hub_id."""
    stmt = select(EvalTestSuite).where(EvalTestSuite.hub_id == hub_id).order_by(EvalTestSuite.created_at.desc())
    if target_type:
        stmt = stmt.where(EvalTestSuite.target_type == target_type)
    if target_id:
        stmt = stmt.where(EvalTestSuite.target_id == target_id)
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def get_suite(db, *, hub_id: str, suite_id: str) -> Optional[EvalTestSuite]:
    """Retrieves a single test suite strictly scoped by hub_id."""
    stmt = select(EvalTestSuite).where(EvalTestSuite.id == suite_id, EvalTestSuite.hub_id == hub_id)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()


async def update_suite(
    db,
    *,
    hub_id: str,
    suite_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    target: Optional[Any] = None,
) -> Optional[EvalTestSuite]:
    """Updates metadata or target for an existing hub-scoped test suite."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        return None

    if name is not None and name != suite.name:
        existing_stmt = select(EvalTestSuite).where(
            EvalTestSuite.hub_id == hub_id, EvalTestSuite.name == name, EvalTestSuite.id != suite_id
        )
        existing_res = await db.execute(existing_stmt)
        if existing_res.scalar_one_or_none():
            raise ValueError(f"SUITE_NAME_TAKEN: A suite named '{name}' already exists in this hub.")
        suite.name = name

    if description is not None:
        suite.description = description

    if target is not None:
        t_type, t_hub_id, t_id = _parse_target(target)
        if (t_type, t_hub_id, t_id) != (suite.target_type, suite.target_hub_id, suite.target_id):
            # Check if suite has completed runs
            runs_stmt = select(EvalRunHistory).where(
                EvalRunHistory.hub_id == hub_id,
                EvalRunHistory.suite_id == suite_id,
                EvalRunHistory.run_status == "completed",
            )
            runs_res = await db.execute(runs_stmt)
            if runs_res.scalars().first():
                raise ValueError("SUITE_HAS_RUNS_RETARGET_BLOCKED: Cannot retarget suite with completed run history. Clone the suite to set a new target.")

            # Cross-hub link check
            resolved_res = await hub_resolver.resolve_linked(
                db,
                source_hub_id=hub_id,
                target_resource_type=t_type,
                target_resource_id=t_id,
            )
            if not resolved_res:
                raise ValueError(f"HUB_LINK_REQUIRED: Hub '{hub_id}' is not linked to target hub '{t_hub_id}'.")

            suite.target_type = t_type
            suite.target_hub_id = t_hub_id
            suite.target_id = t_id

    suite.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(suite)
    return suite


async def delete_suite(db, *, hub_id: str, suite_id: str) -> bool:
    """Deletes a hub-scoped test suite and its test cases."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        return False
    await db.delete(suite)
    await db.commit()
    logger.info("Deleted EvalTestSuite ID: %s in Hub: %s", suite_id, hub_id)
    return True


async def clone_suite(
    db,
    *,
    hub_id: str,
    suite_id: str,
    new_name: Optional[str] = None,
    target: Optional[Any] = None,
) -> Optional[EvalTestSuite]:
    """Clones an existing suite and test cases into a new suite."""
    original_suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not original_suite:
        return None

    clone_target = target or {
        "type": original_suite.target_type,
        "target_hub_id": original_suite.target_hub_id,
        "target_id": original_suite.target_id,
    }

    base_name = new_name or f"{original_suite.name} (Copy)"
    candidate_name = base_name
    counter = 1
    while True:
        check_stmt = select(EvalTestSuite).where(EvalTestSuite.hub_id == hub_id, EvalTestSuite.name == candidate_name)
        check_res = await db.execute(check_stmt)
        if not check_res.scalar_one_or_none():
            break
        counter += 1
        candidate_name = f"{base_name} ({counter})"

    cloned_suite = await create_suite(
        db,
        hub_id=hub_id,
        name=candidate_name,
        description=original_suite.description,
        target=clone_target,
    )

    cases = await list_test_cases(db, hub_id=hub_id, suite_id=suite_id)
    for c in cases:
        await add_test_case(
            db,
            hub_id=hub_id,
            suite_id=cloned_suite.id,
            input_query=c.input_query,
            expected_output=c.expected_output,
            expected_context=c.expected_context,
            node_id=c.node_id,
            assertion_type=c.assertion_type,
            assertion_config=c.assertion_config,
            expected_value=c.expected_value,
        )

    return cloned_suite


# --- Test Case Operations ---


async def add_test_case(
    db,
    *,
    hub_id: str,
    suite_id: str,
    input_query: str,
    expected_output: Optional[str] = None,
    expected_context: Optional[str] = None,
    node_id: Optional[str] = None,
    assertion_type: Optional[str] = None,
    assertion_config: Optional[Dict[str, Any]] = None,
    expected_value: Optional[str] = None,
) -> EvalTestCase:
    """Adds a test case to a hub-scoped suite."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        raise ValueError(f"Suite '{suite_id}' not found in hub '{hub_id}'.")

    if assertion_type and assertion_type not in NODE_ASSERTION_TYPES:
        raise ValueError(f"INVALID_ASSERTION_TYPE: '{assertion_type}' is not supported.")

    case = EvalTestCase(
        id=str(uuid.uuid4()),
        suite_id=suite_id,
        input_query=input_query,
        expected_output=expected_output,
        expected_context=expected_context,
        node_id=node_id,
        assertion_type=assertion_type,
        assertion_config=assertion_config or {},
        expected_value=expected_value,
        created_at=datetime.utcnow(),
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)
    return case


async def list_test_cases(db, *, hub_id: str, suite_id: str) -> List[EvalTestCase]:
    """Lists test cases for a hub-scoped suite."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        return []
    stmt = select(EvalTestCase).where(EvalTestCase.suite_id == suite_id).order_by(EvalTestCase.created_at.asc())
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def update_test_case(
    db,
    *,
    hub_id: str,
    case_id: str,
    input_query: Optional[str] = None,
    expected_output: Optional[str] = None,
    expected_context: Optional[str] = None,
    node_id: Optional[str] = None,
    assertion_type: Optional[str] = None,
    assertion_config: Optional[Dict[str, Any]] = None,
    expected_value: Optional[str] = None,
) -> Optional[EvalTestCase]:
    """Updates a test case after verifying suite ownership in hub."""
    stmt = select(EvalTestCase).where(EvalTestCase.id == case_id)
    res = await db.execute(stmt)
    case = res.scalar_one_or_none()
    if not case:
        return None

    suite = await get_suite(db, hub_id=hub_id, suite_id=case.suite_id)
    if not suite:
        return None

    if input_query is not None:
        case.input_query = input_query
    if expected_output is not None:
        case.expected_output = expected_output
    if expected_context is not None:
        case.expected_context = expected_context
    if node_id is not None:
        case.node_id = node_id
    if assertion_type is not None:
        if assertion_type not in NODE_ASSERTION_TYPES:
            raise ValueError(f"INVALID_ASSERTION_TYPE: '{assertion_type}' is not supported.")
        case.assertion_type = assertion_type
    if assertion_config is not None:
        case.assertion_config = assertion_config
    if expected_value is not None:
        case.expected_value = expected_value

    await db.commit()
    await db.refresh(case)
    return case


async def delete_test_case(db, *, hub_id: str, case_id: str) -> bool:
    """Deletes a test case after verifying suite ownership in hub."""
    stmt = select(EvalTestCase).where(EvalTestCase.id == case_id)
    res = await db.execute(stmt)
    case = res.scalar_one_or_none()
    if not case:
        return False

    suite = await get_suite(db, hub_id=hub_id, suite_id=case.suite_id)
    if not suite:
        return False

    await db.delete(case)
    await db.commit()
    return True


# --- Bulk Import / Export ---


async def import_cases_from_csv(db, *, hub_id: str, suite_id: str, csv_content: str) -> int:
    """Imports test cases from CSV string into a hub-scoped suite."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        raise ValueError(f"SUITE_NOT_FOUND: Suite '{suite_id}' not found in hub '{hub_id}'.")

    lines = [line for line in csv_content.strip().splitlines() if line.strip()]
    if len(lines) > 5000:
        raise ValueError("IMPORT_TOO_LARGE: CSV import exceeds 5000 row limit.")

    f = io.StringIO("\n".join(lines))
    reader = csv.DictReader(f)
    count = 0

    for idx, row in enumerate(reader, start=1):
        row_suite = row.get("suite_id")
        if row_suite and row_suite != suite_id:
            raise ValueError(f"CROSS_HUB_SUITE_ID: Row {idx} specifies foreign suite_id '{row_suite}'.")

        q = row.get("input_query") or row.get("question") or row.get("prompt")
        if not q:
            continue

        a_type = row.get("assertion_type")
        if a_type and a_type not in NODE_ASSERTION_TYPES:
            raise ValueError(f"INVALID_ASSERTION_TYPE: Row {idx} specifies unsupported assertion_type '{a_type}'.")

        a_config = None
        if row.get("assertion_config"):
            try:
                a_config = json.loads(row["assertion_config"])
            except Exception:
                a_config = {}

        await add_test_case(
            db,
            hub_id=hub_id,
            suite_id=suite_id,
            input_query=q,
            expected_output=row.get("expected_output") or row.get("ground_truth"),
            expected_context=row.get("expected_context") or row.get("context"),
            node_id=row.get("node_id"),
            assertion_type=a_type,
            assertion_config=a_config,
            expected_value=row.get("expected_value"),
        )
        count += 1

    return count


async def import_cases_from_json(db, *, hub_id: str, suite_id: str, json_data: Any) -> int:
    """Imports test cases from JSON string or list/dict into a hub-scoped suite."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        raise ValueError(f"SUITE_NOT_FOUND: Suite '{suite_id}' not found in hub '{hub_id}'.")

    items = json_data
    if isinstance(json_data, dict):
        if "suite" in json_data and isinstance(json_data["suite"], dict):
            suite_block = json_data["suite"]
            if suite_block.get("hub_id") and suite_block.get("hub_id") != hub_id:
                raise ValueError("CROSS_HUB_SUITE_ID: Suite block hub_id does not match destination hub.")
        items = json_data.get("cases", json_data.get("test_cases", []))

    if not isinstance(items, list):
        raise ValueError("Invalid JSON data format; expected list of case dicts.")

    if len(items) > 5000:
        raise ValueError("IMPORT_TOO_LARGE: JSON import exceeds 5000 item limit.")

    count = 0
    for idx, item in enumerate(items, start=1):
        if item.get("suite_id") and item.get("suite_id") != suite_id:
            raise ValueError(f"CROSS_HUB_SUITE_ID: Case {idx} specifies foreign suite_id '{item.get('suite_id')}'.")

        q = item.get("input_query") or item.get("question") or item.get("prompt")
        if not q:
            continue

        a_type = item.get("assertion_type")
        if a_type and a_type not in NODE_ASSERTION_TYPES:
            raise ValueError(f"INVALID_ASSERTION_TYPE: Case {idx} specifies unsupported assertion_type '{a_type}'.")

        await add_test_case(
            db,
            hub_id=hub_id,
            suite_id=suite_id,
            input_query=q,
            expected_output=item.get("expected_output") or item.get("ground_truth"),
            expected_context=item.get("expected_context") or item.get("context"),
            node_id=item.get("node_id"),
            assertion_type=a_type,
            assertion_config=item.get("assertion_config"),
            expected_value=item.get("expected_value"),
        )
        count += 1

    return count


async def export_suite(db, *, hub_id: str, suite_id: str, fmt: Literal["csv", "json"] = "json") -> bytes:
    """Exports suite and test cases to CSV or JSON bytes."""
    suite = await get_suite(db, hub_id=hub_id, suite_id=suite_id)
    if not suite:
        raise ValueError(f"SUITE_NOT_FOUND: Suite '{suite_id}' not found in hub '{hub_id}'.")

    cases = await list_test_cases(db, hub_id=hub_id, suite_id=suite_id)

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "suite_id",
                "input_query",
                "expected_output",
                "expected_context",
                "node_id",
                "assertion_type",
                "assertion_config",
                "expected_value",
            ],
        )
        writer.writeheader()
        for c in cases:
            writer.writerow(
                {
                    "suite_id": suite.id,
                    "input_query": c.input_query,
                    "expected_output": c.expected_output or "",
                    "expected_context": c.expected_context or "",
                    "node_id": c.node_id or "",
                    "assertion_type": c.assertion_type or "",
                    "assertion_config": json.dumps(c.assertion_config) if c.assertion_config else "",
                    "expected_value": c.expected_value or "",
                }
            )
        return output.getvalue().encode("utf-8")

    else:
        payload = {
            "suite": {
                "id": suite.id,
                "hub_id": suite.hub_id,
                "name": suite.name,
                "description": suite.description,
                "target": {
                    "type": suite.target_type,
                    "target_hub_id": suite.target_hub_id,
                    "target_id": suite.target_id,
                },
                "created_at": suite.created_at.isoformat() if suite.created_at else None,
            },
            "cases": [
                {
                    "id": c.id,
                    "suite_id": c.suite_id,
                    "input_query": c.input_query,
                    "expected_output": c.expected_output,
                    "expected_context": c.expected_context,
                    "node_id": c.node_id,
                    "assertion_type": c.assertion_type,
                    "assertion_config": c.assertion_config,
                    "expected_value": c.expected_value,
                }
                for c in cases
            ],
        }
        return json.dumps(payload, indent=2).encode("utf-8")


async def suite_health(db, *, hub_id: str) -> List[Dict[str, Any]]:
    """Inspects suites in hub_id and identifies missing targets or broken links."""
    suites = await list_suites(db, hub_id=hub_id)
    health_reports = []

    for s in suites:
        status = "ok"
        reason = None

        try:
            resolved = await hub_resolver.resolve_linked(
                db,
                source_hub_id=hub_id,
                target_resource_type=s.target_type,
                target_resource_id=s.target_id,
            )
            if not resolved:
                target_exists = await hub_resolver.resolve_resource(
                    db, resource_type=s.target_type, resource_id=s.target_id
                )
                if target_exists:
                    status = "link_revoked"
                    reason = f"Hub link to '{s.target_hub_id}' missing or revoked."
                else:
                    status = "missing"
                    reason = f"Target {s.target_type} '{s.target_id}' deleted."
        except Exception as e:
            status = "missing"
            reason = str(e)

        health_reports.append(
            {
                "suite_id": s.id,
                "name": s.name,
                "target": {
                    "type": s.target_type,
                    "target_hub_id": s.target_hub_id,
                    "target_id": s.target_id,
                },
                "status": status,
                "reason": reason,
            }
        )

# Legacy export alias
export_suite_to_json = export_suite

