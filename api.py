"""EvalOps service status API.

All evaluation suite, test case, run, and dashboard routes have migrated to
the tenant-isolated Eval Hub surface under /api/hubs/{hub_id}/eval/* (S6-07e/f).
"""

import logging
from fastapi import APIRouter

router = APIRouter(tags=["evalops"])
logger = logging.getLogger("evalops.api")


@router.get("/status")
async def evalops_status() -> dict:
    """EvalOps service status."""
    return {
        "project": "evalops",
        "status": "active",
        "version": "v6",
        "notice": "All evaluation endpoints have migrated to /api/hubs/{hub_id}/eval/*",
    }
