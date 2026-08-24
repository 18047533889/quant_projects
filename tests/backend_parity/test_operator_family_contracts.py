# -*- coding: utf-8
"""算子族语义契约测试：Rank / 截面 / Fill / 金融 / Robust / Alias / Plan variant。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy parameter family contract contains operators outside daily production")

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.cross_section_spec import (
    normalize_constant_cross_section_fill,
    normalize_single_valid_is_null,
    zscore_zero_std_fill,
)
from factor_engine.backend.cum_semantics import count_is_expanding_non_null, cum_delta_from_first_valid
from factor_engine.backend.factory import build_backend
from factor_engine.backend.fill_semantics import (
    coalesce_null_only,
    coalesce_phase1_pandas_nan_coalesce,
    nan_to_num_unified_fill,
)
from factor_engine.backend.financial_semantics import (
    log_returns_non_positive_is_null,
    vwap_is_rolling,
    volatility_annualization_factor,
)
from factor_engine.backend.math_domain_semantics import signed_log_formula_name
from factor_engine.backend.output_dtype_semantics import logical_output_representation
from factor_engine.backend.quantile_spec import group_percentile_is_indicator
from factor_engine.backend.rank_spec import RANK_SPECS, rank_spec_for
from factor_engine.backend.robust_spec import cs_mad_zscore_no_gaussian_scaling
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.backend_parity.test_p0_edge_cases_triple_parity import duckdb_source, edge_source
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


def _col(name: str):
    return col(
        {
            "close": "Close",
            "volume": "Volume",
            "grp": "Grp",
            "flag": "Flag",
        }.get(name, name)
    )


# --- 契约标志 ---


def test_rank_spec_contracts():
    rank = rank_spec_for("rank")
    pct = rank_spec_for("rank_pct")
    assert rank.pct_formula == "rank_minus1_over_nminus1"
    assert rank.singleton_value == 0.5
    assert rank.method == "average"
    assert rank.null_policy == "exclude"
    assert pct.pct_formula == "rank_over_n"
    assert rank_spec_for("ts_rank").method == "average"
    assert set(RANK_SPECS) >= {"rank", "rank_pct", "cs_pct_rank", "group_rank", "ts_rank"}


def test_cross_section_contract_flags():
    assert normalize_single_valid_is_null()
    assert normalize_constant_cross_section_fill() == 0.5
    assert zscore_zero_std_fill("zscore") == 0.0
    assert group_percentile_is_indicator()
    assert cs_mad_zscore_no_gaussian_scaling()
    assert count_is_expanding_non_null()
    assert cum_delta_from_first_valid()
    assert coalesce_null_only()
    assert coalesce_phase1_pandas_nan_coalesce()
    assert nan_to_num_unified_fill()
    assert vwap_is_rolling()
    assert volatility_annualization_factor() == pytest.approx(252**0.5)
    assert log_returns_non_positive_is_null()
    assert signed_log_formula_name() == "sign_times_log_abs_plus_eps"
    assert logical_output_representation() == "float64_0_1_null"


# --- Rank 族 triple parity ---


def test_rank_vs_rank_pct_formula(_loaded, edge_source, duckdb_source):
    """rank 用 (r-1)/(n-1)；rank_pct 用 rank/n。"""
    ts = pd.Timestamp("2024-01-02")
    r01 = _run(edge_source, F("rank")(col("close")), "pandas").loc[ts]
    rpct = _run(edge_source, F("rank_pct")(col("close")), "pandas").loc[ts]
    assert r01.loc["A"] == pytest.approx(0.0)
    assert rpct.loc["A"] == pytest.approx(0.5)
    assert r01.loc["B"] == pytest.approx(1.0)
    assert rpct.loc["B"] == pytest.approx(1.0)
    long_r01 = _run(edge_source, F("rank")(col("close")), "polars_long").loc[ts]
    sql_r01 = _run(duckdb_source, F("rank")(_col("close")), "duckdb_sql").loc[ts]
    pd.testing.assert_series_equal(r01, long_r01, check_names=False, rtol=1e-6, atol=1e-6)
    pd.testing.assert_series_equal(r01, sql_r01, check_names=False, rtol=1e-6, atol=1e-6)


def test_rank_tie_average_triple(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
            (pd.Timestamp("2024-01-02"), "D"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([1.0, 1.0, 2.0, np.nan], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    ts = pd.Timestamp("2024-01-02")
    pd_out = _run(src, F("rank_pct")(col("close")), "pandas").loc[ts]
    long_out = _run(src, F("rank_pct")(col("close")), "polars_long").loc[ts]
    assert pd_out.loc["A"] == pytest.approx(0.5)
    assert pd_out.loc["B"] == pytest.approx(0.5)
    assert pd_out.loc["C"] == pytest.approx(1.0)
    assert math.isnan(pd_out.loc["D"])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_group_rank_triple(_loaded, edge_source, duckdb_source):
    pd_out = _run(edge_source, F("group_rank")(col("close"), col("grp")), "pandas")
    long_out = _run(edge_source, F("group_rank")(col("close"), col("grp")), "polars_long")
    sql_out = _run(
        duckdb_source, F("group_rank")(_col("close"), _col("grp")), "duckdb_sql"
    )
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


# --- 截面常数语义 ---


def test_zscore_constant_cross_section_zero(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([5.0, 5.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    for backend in ("pandas", "polars_long"):
        out = _run(src, F("zscore")(col("close")), backend).loc[pd.Timestamp("2024-01-02")]
        assert out.iloc[0] == pytest.approx(0.0)
        assert out.iloc[1] == pytest.approx(0.0)


def test_normalize_constant_and_single(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
        names=["timestamp", "instrument"],
    )
    const = pd.Series([3.0, 3.0], index=idx)
    src_const = InMemorySeriesSource(data={"close": const})
    out_const = _run(src_const, F("normalize")(col("close")), "polars_long").iloc[0]
    assert out_const == pytest.approx(0.5)

    single_idx = idx[:1]
    single = pd.Series([42.0], index=single_idx)
    src_single = InMemorySeriesSource(data={"close": single})
    out_single = _run(src_single, F("normalize")(col("close")), "polars_long").iloc[0]
    assert math.isnan(out_single)


def test_scale_to_triple(_loaded, edge_source, duckdb_source):
    expr_pd = F("scale")(col("close"), 2.0)
    expr_sql = F("scale")(_col("close"), 2.0)
    pd_out = _run(edge_source, expr_pd, "pandas")
    long_out = _run(edge_source, expr_pd, "polars_long")
    sql_out = _run(duckdb_source, expr_sql, "duckdb_sql")
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


# --- Fill 族 ---


def test_coalesce_nan_parity_phase1(_loaded):
    """Phase-1：NaN 与 Pandas coalesce_ 一致（长期拆 ``coalesce_nan`` / SQL null-only）。"""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
        names=["timestamp", "instrument"],
    )
    left = pd.Series([np.nan, np.nan], index=idx)
    right = pd.Series([99.0, 88.0], index=idx)
    src = InMemorySeriesSource(data={"left": left, "right": right})
    pd_out = _run(src, F("coalesce")(col("left"), col("right")), "pandas")
    long_out = _run(src, F("coalesce")(col("left"), col("right")), "polars_long")
    assert pd_out.iloc[0] == pytest.approx(99.0)
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_nan_to_num_unified(_loaded, edge_source):
    pd_out = _run(edge_source, F("nan_to_num")(col("close"), 0.0), "pandas")
    long_out = _run(edge_source, F("nan_to_num")(col("close"), 0.0), "polars_long")
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


# --- 金融定义 ---


def test_log_returns_non_positive_null(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 0.0, 5.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    out = _run(src, F("log_returns")(col("close")), "polars_long")
    assert math.isnan(out.iloc[1])


def test_vwap_zero_volume_null(_loaded):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 12.0], index=idx)
    volume = pd.Series([0.0, 0.0, 50.0], index=idx)
    src = InMemorySeriesSource(data={"close": close, "volume": volume})
    out = _run(src, F("vwap")(col("close"), col("volume"), 2), "polars_long")
    assert math.isnan(out.iloc[1])


# --- group_percentile indicator ---


def test_group_percentile_null_not_zero(_loaded, edge_source):
    out = _run(
        edge_source,
        F("group_percentile")(col("close"), col("grp"), 0.5),
        "polars_long",
    )
    null_rows = out[edge_source.data["close"].isna()]
    assert null_rows.isna().all()
