# -*- coding: utf-8
"""Fusion WindowSpec 与 ts_beta pairwise 修复验证。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="legacy price-volume rollout contract superseded by certified primitives")

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


@pytest.fixture(scope="module")
def _loaded():
    load_all()


@pytest.fixture(scope="module")
def ts_source():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=12, freq="B"), ["A"]],
        names=["timestamp", "instrument"],
    )
    left = pd.Series(np.arange(12, dtype=float), index=idx)
    right = pd.Series(np.linspace(1, 2, 12), index=idx)
    right.iloc[3] = np.nan
    right.iloc[7] = np.nan
    left.iloc[5] = np.nan
    return InMemorySeriesSource(data={"left": left, "right": right, "close": left})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()


def test_fusion_ts_mean_respects_min_periods(_loaded, ts_source):
    w, mp = 5, 4
    direct = F("ts_mean")(col("close"), w, min_periods=mp)
    pl_direct = _run(ts_source, direct, "polars_long")
    # min_periods=4：前 3 行应为 NULL
    assert pd.isna(pl_direct.iloc[0])
    assert pd.isna(pl_direct.iloc[1])
    assert pd.isna(pl_direct.iloc[2])
    assert not pd.isna(pl_direct.iloc[3])
    # fusion unary：rank(ts_mean) — ts_mean 为 NULL 时 rank 亦 NULL
    rank_fused = _run(ts_source, F("rank")(direct), "polars_long")
    assert pd.isna(rank_fused.iloc[0])
    assert pd.isna(rank_fused.iloc[1])
    assert pd.isna(rank_fused.iloc[2])
    assert not pd.isna(rank_fused.iloc[3])


def test_fused_add_ts_mean_matches_generic(_loaded, ts_source):
    w, mp = 6, 5
    expr = F("add")(
        F("ts_mean")(col("close"), w, min_periods=mp),
        F("ts_std")(col("close"), w, min_periods=mp),
    )
    generic_a = _run(ts_source, F("ts_mean")(col("close"), w, min_periods=mp), "polars_long")
    generic_b = _run(ts_source, F("ts_std")(col("close"), w, min_periods=mp), "polars_long")
    expected = generic_a + generic_b
    fused = _run(ts_source, expr, "polars_long")
    pd.testing.assert_series_equal(expected, fused, check_names=False, rtol=1e-6, atol=1e-6)


def test_ts_beta_invalid_rows_null(_loaded):
    """pairwise beta：当前行任一侧 NULL → 输出 NULL（两后端一致）。"""
    from tests.backend_parity.test_production_core_triple_parity import _memory_source

    src = _memory_source()
    expr = F("ts_beta")(col("ret"), col("close"), 3)
    pd_out = _run(src, expr, "pandas")
    pl_out = _run(src, expr, "polars_long")
    invalid = src.data["close"].isna() | src.data["ret"].isna()
    for (ts, inst), is_invalid in invalid.items():
        if is_invalid:
            assert pd.isna(pd_out.loc[(ts, inst)])
            assert pd.isna(pl_out.loc[(ts, inst)])


def test_vwap_pairwise_null_price(_loaded):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=6, freq="B"), ["A"]],
        names=["timestamp", "instrument"],
    )
    price = pd.Series([10.0, 11.0, np.nan, 13.0, 14.0, 15.0], index=idx)
    volume = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    src = InMemorySeriesSource(data={"close": price, "volume": volume})
    expr = F("vwap")(col("close"), col("volume"), 3)
    pd_out = _run(src, expr, "pandas")
    pl_out = _run(src, expr, "polars_long")
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_group_normalize_all_null_group(_loaded):
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([np.nan, np.nan], index=idx)
    grp = pd.Series([1.0, 1.0], index=idx)
    src = InMemorySeriesSource(data={"close": close, "grp": grp})
    expr = F("group_normalize")(col("close"), col("grp"))
    pl_out = _run(src, expr, "polars_long")
    assert pl_out.isna().all()
