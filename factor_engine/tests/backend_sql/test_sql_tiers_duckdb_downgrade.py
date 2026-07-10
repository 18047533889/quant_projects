# -*- coding: utf-8
"""DuckDB 能力降级与 effective SQL production safe。"""
from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from backend.sql_tiers import (
    SQL_PRODUCTION_SAFE_CANONICALS,
    duckdb_downgraded_canonicals,
    effective_sql_production_safe,
)


def test_effective_sql_production_safe_subset():
    for canon in ("ts_mean", "rank", "add"):
        if canon in SQL_PRODUCTION_SAFE_CANONICALS:
            assert effective_sql_production_safe(canon)


def test_duckdb_downgrade_does_not_remove_core_unconditionally():
    downgraded = duckdb_downgraded_canonicals(refresh=True)
    assert isinstance(downgraded, frozenset)
    # ts_mean 不依赖 corr window，通常不应被降级
    if "ts_mean" in SQL_PRODUCTION_SAFE_CANONICALS:
        assert effective_sql_production_safe("ts_mean")
