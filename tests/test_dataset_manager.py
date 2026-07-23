"""Unit tests for S5-01c: Custom Dataset Manager (CRUD, Import/Export, Cloning)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from projects.evalops.src.datasets import manager as dataset_mgr


@pytest.mark.asyncio
async def test_dataset_manager_crud():
    """Verify suite creation and case management functions."""
    mock_db = AsyncMock(spec=AsyncSession)

    # Test CSV import helper parsing
    csv_content = """input_query,expected_output,expected_context
What is SyntraFlow?,Hybrid search engine,Vector and BM25 search
Explain GuardRoute,Security gateway,Prompt injection detection
"""

    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    count = await dataset_mgr.import_cases_from_csv(mock_db, "suite-123", csv_content)
    assert count == 2

    # Test JSON import helper parsing
    json_data = [
        {"input_query": "Query 1", "expected_output": "Out 1", "expected_context": ["Ctx 1", "Ctx 2"]},
        {"input_query": "Query 2", "expected_output": "Out 2", "expected_context": "Ctx string"},
    ]
    json_count = await dataset_mgr.import_cases_from_json(mock_db, "suite-123", json_data)
    assert json_count == 2
