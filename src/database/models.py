"""SQLAlchemy database models for EvalOps benchmark and quality reports.

All models use the shared Base from common.models.database for unified
Alembic migration support across the monorepo.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from common.models.database import Base, EvalTestSuite, EvalTestCase, EvalRunHistory


class EvalOpsReport(Base):
    """Stores evaluation and benchmark execution metrics and quality logs."""

    __tablename__ = "evalops_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String(50), nullable=False)  # 'retrieval', 'routing', 'safety'
    metrics_json = Column(Text, nullable=False)       # JSON string containing evaluation metrics
    created_at = Column(DateTime, default=datetime.utcnow)


__all__ = ["Base", "EvalOpsReport", "EvalTestSuite", "EvalTestCase", "EvalRunHistory"]
