# -*- coding: utf-8
"""Backend capability 注册表与 cost routing 测试。"""

from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE
from cleaned_operators.registry import OperatorRegistry
from backend.operator_capability import (
    build_capability_matrix,
    get_best_backend,
    resolve_canonical,
    summarize_operator,
)


@pytest.fixture(scope="module")
def loaded():
    load_all()
    from backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()
    yield


def test_ts_mean_production_capabilities(loaded):
    s = summarize_operator("ts_mean")
    assert s.pandas_numpy == "implemented"
    assert s.polars == "production_safe"
    assert s.duckdb_sql in {"parity_verified", "production_safe"}
    assert s.allow_in_production


def test_bfill_not_polars_production_safe(loaded):
    s = summarize_operator("bfill")
    assert s.polars != "production_safe"
    _, backend = get_best_backend("bfill", mode="production", prefer="auto")
    assert backend == "pandas_numpy"


def test_if_else_polars_long_tier(loaded):
    s = summarize_operator("if_else")
    assert s.polars_long_tier == "native"
    assert summarize_operator("where").polars == "production_safe"


def test_bfill_polars_long_tier(loaded):
    s = summarize_operator("bfill")
    assert s.polars_long_tier == "blocked_causal"
    assert s.polars != "production_safe"


def test_get_best_backend_respects_production_safe(loaded):
    _, backend = get_best_backend("ts_mean", mode="production", prefer="auto")
    assert backend == "polars"
    canon = resolve_canonical("rank")
    assert canon in POLARS_PRODUCTION_SAFE


def test_capability_matrix_covers_implemented(loaded):
    matrix = build_capability_matrix()
    assert len(matrix) >= 300
    ts = next(r for r in matrix if r.canonical == "ts_mean")
    assert ts.polars == "production_safe"


def test_production_fast_path_whitelist(loaded):
    from backend.production_fast_path import (
        is_production_fast_path,
        summarize_production_fast_path,
    )

    assert is_production_fast_path("ts_mean")
    assert is_production_fast_path("group_mean")
    assert is_production_fast_path("group_winsorize")
    assert is_production_fast_path("ts_rank")
    assert is_production_fast_path("ts_sharpe")
    assert is_production_fast_path("cs_resid")
    assert not is_production_fast_path("RSI_WILDER")
    assert is_production_fast_path("ffill")
    summary = summarize_production_fast_path()
    assert summary["production_fast_path_count"] >= 10
    assert "ts_beta" in summary["production_fast_path"]


def test_get_best_backend_matches_registry_preferred(loaded):
    for name in ("ts_mean", "rank", "bfill", "MACD"):
        op_a, b_a = OperatorRegistry.get_preferred(name, prefer="auto")
        op_b, b_b = get_best_backend(name, mode="production", prefer="auto")
        assert b_a == b_b
        assert (op_a is None) == (op_b is None)
