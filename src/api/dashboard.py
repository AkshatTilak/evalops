"""Hub-scoped Eval Hub Dashboard Aggregation Service (S6-07e).

Computes SQL-driven metrics, trends, comparisons, and target rollups strictly
filtered by hub_id and filter criteria (target_type, target_id, framework, date_from, date_to).
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy import func, select

from common.models.database import EvalMetricResult, EvalRunHistory, EvalTestSuite, HubLink, AgentDefinition, WorkflowDefinition

logger = logging.getLogger("evalops.api.dashboard")


def _parse_dates(date_from: Optional[str], date_to: Optional[str]) -> tuple[datetime, datetime]:
    now = datetime.utcnow()
    d_to = datetime.fromisoformat(date_to) if date_to else now
    d_from = datetime.fromisoformat(date_from) if date_from else (now - timedelta(days=30))

    if (d_to - d_from).days > 365:
        raise ValueError("DATE_RANGE_TOO_LARGE: Date range cannot exceed 365 days.")
    return d_from, d_to


async def get_dashboard_stats(
    db,
    *,
    hub_id: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    framework: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Computes aggregate dashboard stats for an eval hub."""
    d_from, d_to = _parse_dates(date_from, date_to)

    stmt = select(EvalRunHistory).where(
        EvalRunHistory.hub_id == hub_id,
        EvalRunHistory.created_at >= d_from,
        EvalRunHistory.created_at <= d_to,
    )
    if target_type:
        stmt = stmt.where(EvalRunHistory.target_type == target_type)
    if target_id:
        stmt = stmt.where(EvalRunHistory.target_id == target_id)
    if framework and framework != "both":
        stmt = stmt.where(EvalRunHistory.framework_used == framework)

    res = await db.execute(stmt)
    runs = res.scalars().all()

    total_runs = len(runs)
    if total_runs == 0:
        return {
            "total_runs": 0,
            "completed_runs": 0,
            "pass_rate": 0.0,
            "average_duration_sec": 0.0,
            "metrics": {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "recall": 0.0,
                "precision": 0.0,
            },
            "node_assertions": {"total": 0, "passed": 0, "pass_rate": 0.0},
        }

    completed_runs = sum(1 for r in runs if r.run_status == "completed")
    avg_duration = round(sum(r.duration_sec or 0.0 for r in runs) / total_runs, 2)

    faith_avg = round(sum(r.faithfulness_score or 0.0 for r in runs if r.faithfulness_score is not None) / max(1, sum(1 for r in runs if r.faithfulness_score is not None)), 4)
    rel_avg = round(sum(r.relevance_score or 0.0 for r in runs if r.relevance_score is not None) / max(1, sum(1 for r in runs if r.relevance_score is not None)), 4)
    recall_avg = round(sum(r.recall_score or 0.0 for r in runs if r.recall_score is not None) / max(1, sum(1 for r in runs if r.recall_score is not None)), 4)
    prec_avg = round(sum(r.precision_score or 0.0 for r in runs if r.precision_score is not None) / max(1, sum(1 for r in runs if r.precision_score is not None)), 4)

    total_cases = sum(r.total_test_cases or 0 for r in runs)
    passed_cases = sum(r.passed_count or 0 for r in runs)
    pass_rate = round(passed_cases / total_cases, 4) if total_cases > 0 else 0.0

    node_total = 0
    node_passed = 0
    for r in runs:
        if r.details_json and isinstance(r.details_json, dict) and "node_assertions" in r.details_json:
            na = r.details_json["node_assertions"]
            node_total += na.get("total", 0)
            node_passed += na.get("passed", 0)

    node_pass_rate = round(node_passed / node_total, 4) if node_total > 0 else 0.0

    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "pass_rate": pass_rate,
        "average_duration_sec": avg_duration,
        "metrics": {
            "faithfulness": faith_avg,
            "relevance": rel_avg,
            "recall": recall_avg,
            "precision": prec_avg,
        },
        "node_assertions": {
            "total": node_total,
            "passed": node_passed,
            "pass_rate": node_pass_rate,
        },
    }


async def get_dashboard_trends(
    db,
    *,
    hub_id: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    framework: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Computes daily bucketed metric trends for an eval hub."""
    d_from, d_to = _parse_dates(date_from, date_to)

    stmt = select(EvalRunHistory).where(
        EvalRunHistory.hub_id == hub_id,
        EvalRunHistory.created_at >= d_from,
        EvalRunHistory.created_at <= d_to,
    )
    if target_type:
        stmt = stmt.where(EvalRunHistory.target_type == target_type)
    if target_id:
        stmt = stmt.where(EvalRunHistory.target_id == target_id)
    if framework and framework != "both":
        stmt = stmt.where(EvalRunHistory.framework_used == framework)

    res = await db.execute(stmt)
    runs = res.scalars().all()

    daily_buckets: Dict[str, List[EvalRunHistory]] = {}
    for r in runs:
        day_str = r.created_at.strftime("%Y-%m-%d") if r.created_at else "unknown"
        if day_str not in daily_buckets:
            daily_buckets[day_str] = []
        daily_buckets[day_str].append(r)

    trends = []
    for day in sorted(daily_buckets.keys()):
        day_runs = daily_buckets[day]
        faith = round(sum(r.faithfulness_score or 0.0 for r in day_runs if r.faithfulness_score) / max(1, sum(1 for r in day_runs if r.faithfulness_score)), 4)
        rel = round(sum(r.relevance_score or 0.0 for r in day_runs if r.relevance_score) / max(1, sum(1 for r in day_runs if r.relevance_score)), 4)
        trends.append(
            {
                "date": day,
                "runs_count": len(day_runs),
                "faithfulness": faith,
                "relevance": rel,
            }
        )

    return trends


async def get_dashboard_comparison(
    db,
    *,
    hub_id: str,
    target_ids: List[str],
    framework: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Side-by-side metric comparison across up to 5 target resources."""
    if len(target_ids) > 5:
        raise ValueError("COMPARISON_LIMIT_EXCEEDED: Cannot compare more than 5 targets.")

    results = []
    for t_id in target_ids:
        stmt = select(EvalRunHistory).where(
            EvalRunHistory.hub_id == hub_id,
            EvalRunHistory.target_id == t_id,
        ).order_by(EvalRunHistory.created_at.desc()).limit(10)
        res = await db.execute(stmt)
        runs = res.scalars().all()

        if not runs:
            results.append(
                {
                    "target_id": t_id,
                    "runs_count": 0,
                    "faithfulness": 0.0,
                    "relevance": 0.0,
                }
            )
        else:
            faith = round(sum(r.faithfulness_score or 0.0 for r in runs if r.faithfulness_score) / max(1, sum(1 for r in runs if r.faithfulness_score)), 4)
            rel = round(sum(r.relevance_score or 0.0 for r in runs if r.relevance_score) / max(1, sum(1 for r in runs if r.relevance_score)), 4)
            results.append(
                {
                    "target_id": t_id,
                    "target_type": runs[0].target_type,
                    "runs_count": len(runs),
                    "faithfulness": faith,
                    "relevance": rel,
                }
            )

    return results


async def get_dashboard_targets(
    db,
    *,
    hub_id: str,
    target_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lists per-target rollups for an eval hub."""
    stmt = select(EvalTestSuite).where(EvalTestSuite.hub_id == hub_id)
    if target_type:
        stmt = stmt.where(EvalTestSuite.target_type == target_type)
    res = await db.execute(stmt)
    suites = res.scalars().all()

    target_map: Dict[str, Dict[str, Any]] = {}
    for s in suites:
        key = f"{s.target_type}:{s.target_id}"
        if key not in target_map:
            target_map[key] = {
                "target_type": s.target_type,
                "target_hub_id": s.target_hub_id,
                "target_id": s.target_id,
                "suites_count": 0,
            }
        target_map[key]["suites_count"] += 1

    rollups = []
    for key, data in target_map.items():
        run_stmt = select(EvalRunHistory).where(
            EvalRunHistory.hub_id == hub_id,
            EvalRunHistory.target_id == data["target_id"],
        ).order_by(EvalRunHistory.created_at.desc()).limit(1)
        run_res = await db.execute(run_stmt)
        latest_run = run_res.scalar_one_or_none()

        data["last_run_at"] = latest_run.created_at.isoformat() if latest_run and latest_run.created_at else None
        data["latest_faithfulness"] = latest_run.faithfulness_score if latest_run else None
        data["latest_relevance"] = latest_run.relevance_score if latest_run else None
        rollups.append(data)

    return rollups
