"""Catalog 运行血缘（factor_run）测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from factor_engine.storage.catalog import FactorCatalog

pd = pytest.importorskip("pandas")


def test_record_and_list_runs():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.sqlite"
        cat = FactorCatalog(db)
        cat.register(
            "test_factor_v1",
            author="tester",
            frequency="1d",
            ast_hash="abc123",
            expression="rank(close)",
        )
        cat.record_run(
            {
                "run_id": "run001",
                "factor_id": "test_factor_v1",
                "factor_name": "test_factor",
                "ast_hash": "abc123",
                "operator_catalog_hash": "ophash",
                "field_catalog_hash": "fieldhash",
                "expression": "rank(close)",
                "lookback": 20,
                "referenced_columns": ["close"],
                "dq_passed": True,
                "row_count": 100,
                "non_null_count": 95,
                "extra": {"note": "unit test"},
            }
        )
        runs = cat.list_runs("test_factor_v1")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run001"
        assert runs[0]["dq_passed"] == 1
        assert runs[0]["field_catalog_hash"] == "fieldhash"
        cat.close()
