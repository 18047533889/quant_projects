# -*- coding: utf-8
"""Backend coverage contract for the static production surface."""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends
from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    load_all()
    register_sql_backends()


def test_daily_has_certified_polars_coverage():
    from factor_engine.backend.fastpath_evidence import polars_executed_parity_canonicals

    missing = DAILY_CANONICALS - polars_executed_parity_canonicals()
    assert not missing, f"缺少纯 Polars 认证: {sorted(missing)}"


def test_daily_has_sql_registry_backends():
    register_sql_backends()
    for name in DAILY_CANONICALS:
        assert "sql" in OperatorRegistry.backends_for(name), name


def test_daily_is_within_certified_duckdb_fastpath():
    from factor_engine.backend.sql_tiers import SQL_PRODUCTION_SAFE_CANONICALS

    assert DAILY_CANONICALS <= SQL_PRODUCTION_SAFE_CANONICALS
