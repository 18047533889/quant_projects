# -*- coding: utf-8
"""Production fast path 分层与 evidence 准入测试。"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def test_primitive_dual_backend_evidence_is_production_safe(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.primitive_evidence import operational_production_certified_set
    from backend.sql_tiers import effective_sql_production_safe

    for canon in sorted(operational_production_certified_set()):
        assert is_polars_long_native_production_safe(canon), canon
        assert effective_sql_production_safe(canon), canon


def test_p0_not_auto_production_without_evidence(_loaded):
    """P0 候选池不再自动等于 production-safe（须 primitive evidence）。"""
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.production_fastpath_tiers import P0_PRODUCTION_FASTPATH_CANONICALS
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    p0_only = P0_PRODUCTION_FASTPATH_CANONICALS - PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert p0_only
    assert not is_polars_long_native_production_safe("fillna")


def test_p1_group_dual_backend_subset(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE

    assert "group_zscore" in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert is_polars_long_native_production_safe("group_zscore")
    assert "group_mean" in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert is_polars_long_native_production_safe("group_mean")
    assert "group_rank" in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert is_polars_long_native_production_safe("group_rank")


def test_p1_robust_winsorize_dual_backend(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.primitive_evidence import (
        DUCKDB_REAL_SQL_VERIFIED,
        PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE,
    )
    from backend.sql_tiers import effective_sql_production_safe

    assert "group_winsorize" in DUCKDB_REAL_SQL_VERIFIED
    assert "group_winsorize" in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    assert is_polars_long_native_production_safe("group_winsorize")
    assert effective_sql_production_safe("winsorize")
    assert is_polars_long_native_production_safe("winsorize")


def test_p1_regression_not_production_safe(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.production_fastpath_tiers import P1_REGRESSION_PARITY_PENDING
    from backend.sql_tiers import effective_sql_production_safe

    for canon in P1_REGRESSION_PARITY_PENDING:
        assert not is_polars_long_native_production_safe(canon), canon
        assert not effective_sql_production_safe(canon), canon


def test_p1_golden_ts_requires_evidence(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.primitive_evidence import PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE
    from backend.production_fastpath_tiers import P1_GOLDEN_VERIFIED_TS

    for canon in P1_GOLDEN_VERIFIED_TS:
        if canon in PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE:
            assert is_polars_long_native_production_safe(canon), canon
        else:
            assert not is_polars_long_native_production_safe(canon), canon


def test_p1_ts_complex_not_production_safe(_loaded):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.production_fastpath_tiers import P1_TS_COMPLEX_PARITY_PENDING
    from backend.sql_tiers import effective_sql_production_safe

    for canon in P1_TS_COMPLEX_PARITY_PENDING:
        assert not is_polars_long_native_production_safe(canon), canon
        assert not effective_sql_production_safe(canon), canon


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
    assert not CLICKHOUSE_SQL_PRODUCTION_SAFE
    assert not CLICKHOUSE_SQL_PARITY_VERIFIED
