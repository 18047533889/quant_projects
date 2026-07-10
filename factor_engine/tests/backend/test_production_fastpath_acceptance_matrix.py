# -*- coding: utf-8
"""Production fast path 验收矩阵（P0 / P1 关键算子）。"""
from __future__ import annotations

import pytest

pytest.importorskip("polars")


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _row(canon: str):
    from backend.fastpath_coverage import build_fastpath_coverage_row

    return build_fastpath_coverage_row(canon)


ACCEPTANCE_P0 = [
    "add",
    "ts_mean",
    "rank",
    "protected_div",
    "where",
    "vwap",
    "c_mean",
    "cs_pct_rank",
    "power",
    "is_nan",
]

ACCEPTANCE_P1 = [
    "group_mean",
    "group_percentile",
    "ts_corr",
    "rolling_beta",
    "cs_mad",
    "cum_sum",
    "ts_rank",
    "ts_sharpe",
    "ts_autocorr",
    "cs_resid",
    "cs_regression",
]

ACCEPTANCE_P1_IMPLEMENTED_ONLY = [
    "RSI_WILDER",
    "ts_decay_linear",
    "ts_ema",
]

ACCEPTANCE_P2 = [
    "ewm_corr",
    "ts_kurt",
    "fillna_interpolate",
    "RSI_WILDER",
    "ts_decay_linear",
]


@pytest.mark.parametrize("canon", ACCEPTANCE_P0)
def test_acceptance_p0_production_fastpath(_loaded, canon):
    from backend.polars_long_production import is_polars_long_native_production_safe
    from backend.sql_tiers import effective_sql_production_safe
    from backend.operator_capability import _sql_emitter_ok

    row = _row(canon)
    assert row.polars_long_native, canon
    assert is_polars_long_native_production_safe(canon), canon
    assert effective_sql_production_safe(canon) and _sql_emitter_ok(canon), canon
    assert row.polars_long_native_production_safe
    assert row.sql_production_safe and row.sql_emitter_ok


@pytest.mark.parametrize("canon", ACCEPTANCE_P1)
def test_acceptance_p1_polars_or_duckdb(_loaded, canon):
    row = _row(canon)
    assert row.polars_long_native or row.sql_implemented, canon
    assert row.polars_long_native_production_safe or (
        row.sql_production_safe and row.sql_emitter_ok
    ), f"{canon}: {row.fastpath_block_reason}"
    if canon in {"cs_mad", "cs_mad_zscore"}:
        assert row.polars_long_native_production_safe
        assert row.duckdb_sql_production_safe and row.sql_emitter_ok


@pytest.mark.parametrize("canon", ACCEPTANCE_P1_IMPLEMENTED_ONLY)
def test_acceptance_p1_implemented_not_production_safe(_loaded, canon):
    row = _row(canon)
    assert row.polars_long_native or row.polars_long_tier in {"native", "python_rolling"}, canon
    assert not row.production_fast_path, f"{canon} 已实现但不应 production fast path"
    assert row.fastpath_block_reason


@pytest.mark.parametrize("canon", ACCEPTANCE_P2)
def test_acceptance_p2_not_production_fastpath(_loaded, canon):
    row = _row(canon)
    assert not row.production_fast_path, f"{canon} 不应 production fast path"
    assert row.fastpath_block_reason
