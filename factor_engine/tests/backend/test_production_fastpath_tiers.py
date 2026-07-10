# -*- coding: utf-8
"""Production fast path 分层与 P0/P1 准入测试。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_p0_all_polars_native_production_safe(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.production_fastpath_tiers import P0_PRODUCTION_FASTPATH_CANONICALS

    missing = sorted(c for c in P0_PRODUCTION_FASTPATH_CANONICALS if not is_polars_long_native_production_safe(c))
    assert not missing, f"P0 未进 POLARS_LONG_NATIVE_PRODUCTION_SAFE: {missing}"


def test_p0_all_duckdb_production_safe(_loaded):
    from backend.production_fastpath_tiers import P0_PRODUCTION_FASTPATH_CANONICALS
    from backend.sql_tiers import effective_sql_production_safe

    missing = sorted(c for c in P0_PRODUCTION_FASTPATH_CANONICALS if not effective_sql_production_safe(c))
    assert not missing, f"P0 未进 DUCKDB_SQL_PRODUCTION_SAFE: {missing}"


def test_p1_group_polars_production_safe(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.production_fastpath_tiers import P1_GROUP_CANONICALS

    missing = sorted(c for c in P1_GROUP_CANONICALS if not is_polars_long_native_production_safe(c))
    assert not missing, f"P1 group 未进 PolarsLong production safe: {missing}"


def test_p1_robust_duckdb_production_safe(_loaded):
    from backend.production_fastpath_tiers import P1_DUCKDB_PARITY_PENDING
    from backend.sql_tiers import effective_sql_production_safe

    assert not P1_DUCKDB_PARITY_PENDING
    for canon in ("cs_mad", "cs_mad_zscore", "winsorize", "group_winsorize"):
        assert effective_sql_production_safe(canon), canon


def test_rolling_beta_native_not_map_groups(_loaded):
    from backend.polars_long_policy import POLARS_LONG_MAP_GROUPS, infer_polars_long_tier

    assert infer_polars_long_tier("rolling_beta") == "native"
    assert "rolling_beta" not in POLARS_LONG_MAP_GROUPS


def test_map_groups_production_allowed_empty(_loaded):
    from backend.polars_long_production import (
        APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION,
        POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED,
    )

    assert not POLARS_LONG_MAP_GROUPS_PRODUCTION_ALLOWED
    assert not APPROVED_POLARS_LONG_MAP_GROUPS_PRODUCTION


def test_p2_and_forbidden_not_production_safe(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.production_fastpath_tiers import (
        FORBIDDEN_PRODUCTION_FASTPATH,
        P2_RESEARCH_ONLY,
        resolve_polars_native_canonical,
    )
    from backend.production_fastpath_gate import check_production_fastpath_plan_ops
    from backend.sql_pushdown.plan_fixtures import minimal_plan
    from backend.sql_tiers import effective_sql_production_safe

    sample = sorted(P2_RESEARCH_ONLY | FORBIDDEN_PRODUCTION_FASTPATH)[:12]
    for canon in sample:
        resolved = resolve_polars_native_canonical(canon)
        if resolved != canon:
            continue
        assert not is_polars_long_native_production_safe(canon), canon
        assert not effective_sql_production_safe(canon), canon
        try:
            plan = minimal_plan(canon)
        except Exception:
            continue
        result = check_production_fastpath_plan_ops(plan, strict=True)
        assert not result.ok, f"{canon} 不应通过 strict fastpath gate: {result.violations}"


def test_winsorize_production_safe(_loaded):
    from backend.polars_long_production import (
        is_polars_long_native_production_safe,
        polars_long_production_tier,
    )
    from backend.sql_tiers import effective_sql_production_safe

    assert polars_long_production_tier("winsorize") == "production_safe"
    assert is_polars_long_native_production_safe("winsorize")
    assert is_polars_long_native_production_safe("group_winsorize")
    assert effective_sql_production_safe("winsorize")
    assert effective_sql_production_safe("group_winsorize")


def test_p2_map_groups_not_production_safe(_loaded):
    from backend.polars_long_policy import infer_polars_long_tier
    from backend.production_fastpath_tiers import P2_MAP_GROUPS_CANONICALS, resolve_polars_native_canonical
    from backend.polars_long_production import is_polars_long_native_production_safe

    for canon in P2_MAP_GROUPS_CANONICALS:
        if resolve_polars_native_canonical(canon) != canon:
            continue
        assert infer_polars_long_tier(canon) == "map_groups"
        assert not is_polars_long_native_production_safe(canon)


def test_sql_tier_aliases(_loaded):
    from backend.sql_tiers import (
        CLICKHOUSE_SQL_PARITY_VERIFIED,
        CLICKHOUSE_SQL_PRODUCTION_SAFE,
        DUCKDB_SQL_PARITY_VERIFIED,
        DUCKDB_SQL_PRODUCTION_SAFE,
        SQL_PARITY_VERIFIED_CANONICALS,
        SQL_PRODUCTION_SAFE_CANONICALS,
    )

    assert DUCKDB_SQL_PARITY_VERIFIED == SQL_PARITY_VERIFIED_CANONICALS
    assert DUCKDB_SQL_PRODUCTION_SAFE == SQL_PRODUCTION_SAFE_CANONICALS
    assert CLICKHOUSE_SQL_PRODUCTION_SAFE <= CLICKHOUSE_SQL_PARITY_VERIFIED
