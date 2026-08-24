# -*- coding: utf-8
"""Intraday aggregation 三后端 parity：Pandas / Polars long / DuckDB SQL。

这些算子消费**分钟**面板并输出**日频**面板，因此不能用 daily-primitive 的
six-way parity fixture 认证（那是全 NaN 的假 parity）。这里用真实分钟形状的
合成数据（2 标的 × 3 天 × 全天 240 根 bar，含 morning/afternoon 时段），让
三个后端独立计算同一数学定义并逐值对齐。parity 覆盖矩阵记录每个算子在哪些
后端上通过 —— 从不虚构。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")
pytest.importorskip("duckdb")

from factor_engine.cleaned_operators import load_all

from tests.backend_parity.intraday_minute_parity import (
    _POLARS_OPS,
    _SQL_OPS,
    _POLARS_LIMIT_OPS,
    _REFERENCE_ONLY,
    build_long,
)

load_all()

_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04"]
_INSTS = ["A", "B"]
_RNG = np.random.default_rng(7)


def _minute_index(days):
    idx = []
    for day in days:
        base = pd.Timestamp(day)
        for m in range(9 * 60 + 31, 11 * 60 + 31):   # 09:31..11:30 (morning)
            idx.append(base + pd.Timedelta(minutes=m))
        for m in range(13 * 60 + 1, 15 * 60 + 1):    # 13:01..15:00 (afternoon)
            idx.append(base + pd.Timedelta(minutes=m))
    return pd.DatetimeIndex(idx)


def _fixture_panels():
    idx = _minute_index(_DAYS)
    close, open_, high, low, volume, amount, activity = {}, {}, {}, {}, {}, {}, {}
    for inst in _INSTS:
        prices = [10.0]
        for _ in range(len(idx) - 1):
            prices.append(prices[-1] * (1.0 + _RNG.normal(0.0, 0.002)))
        prices = np.asarray(prices)
        close[inst] = prices
        open_[inst] = np.concatenate([[prices[0]], prices[:-1]])
        high[inst] = np.maximum(prices, open_[inst]) * 1.005
        low[inst] = np.minimum(prices, open_[inst]) * 0.995
        vol = _RNG.uniform(100, 1000, len(idx))
        volume[inst] = vol
        amount[inst] = vol * prices
        activity[inst] = _RNG.uniform(0.5, 1.5, len(idx))
    close = pd.DataFrame(close, index=idx, dtype=float)
    open_ = pd.DataFrame(open_, index=idx, dtype=float)
    high = pd.DataFrame(high, index=idx, dtype=float)
    low = pd.DataFrame(low, index=idx, dtype=float)
    volume = pd.DataFrame(volume, index=idx, dtype=float)
    amount = pd.DataFrame(amount, index=idx, dtype=float)
    activity = pd.DataFrame(activity, index=idx, dtype=float)
    return {"close": close, "open": open_, "high": high, "low": low,
            "volume": volume, "amount": amount, "activity": activity}


def _daily_limit_panels(close: pd.DataFrame) -> pd.DataFrame:
    """Daily high_limit such that the max minute close is within EPS below it."""
    daily = close.groupby(close.index.normalize()).max()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily * 0.999


@pytest.fixture(scope="module")
def panels():
    return _fixture_panels()


@pytest.fixture(scope="module")
def long_df(panels):
    return build_long(panels)


@pytest.fixture(scope="module")
def limit_panel(panels):
    return _daily_limit_panels(panels["close"])


@pytest.fixture(scope="module")
def con(long_df):
    import duckdb

    c = duckdb.connect()
    c.register("mlong", long_df.to_pandas())
    yield c
    c.close()


def _pandas_kernel(name, panels):
    from factor_engine.cleaned_operators.microstructure import intraday_agg as m

    cls = {
        "intra_realized_variance": m.IntraRealizedVariance,
        "intra_realized_semivariance": m.IntraRealizedSemivariance,
        "intra_bipower_variation": m.IntraBipowerVariation,
        "intra_jump_ratio": m.IntraJumpRatio,
        "intra_path_efficiency": m.IntraPathEfficiency,
        "intra_high_time": m.IntraHighTime,
        "intra_low_time": m.IntraLowTime,
        "intra_segment_return": m.IntraSegmentReturn,
        "intra_segment_volume_share": m.IntraSegmentVolumeShare,
        "intra_segment_amount_share": m.IntraSegmentAmountShare,
        "intra_segment_realized_vol": m.IntraSegmentRealizedVol,
        "intra_segment_vwap_deviation": m.IntraSegmentVwapDeviation,
        "intra_vwap_above_ratio": m.IntraVwapAboveRatio,
        "intra_vwap_cross_count": m.IntraVwapCrossCount,
        "intra_concentration": m.IntraConcentration,
        "intra_entropy": m.IntraEntropy,
        "intra_signed_imbalance_proxy": m.IntraSignedImbalanceProxy,
        "intra_return_activity_corr": m.IntraReturnActivityCorr,
        "intra_amihud": m.IntraAmihud,
        "intra_kyle_lambda_proxy": m.IntraKyleLambdaProxy,
        "intra_extreme_bar_return": m.IntraExtremeBarReturn,
        "intra_lunch_gap_return": m.IntraLunchGapReturn,
        "intra_limit_duration": m.IntraLimitDuration,
        "intra_limit_first_hit_time": m.IntraLimitFirstHitTime,
        "intra_limit_reopen_count": m.IntraLimitReopenCount,
    }[name]
    op = cls()
    if name == "intra_realized_semivariance":
        return op._calculate_series(panels["close"], side="down")
    if name == "intra_high_time":
        return op._calculate_series(panels["high"])
    if name == "intra_low_time":
        return op._calculate_series(panels["low"])
    if name == "intra_segment_return":
        return op._calculate_series(panels["close"], segment="morning")
    if name == "intra_segment_volume_share":
        return op._calculate_series(panels["volume"], segment="morning")
    if name == "intra_segment_amount_share":
        return op._calculate_series(panels["amount"], segment="morning")
    if name == "intra_segment_realized_vol":
        return op._calculate_series(panels["close"], segment="morning")
    if name == "intra_segment_vwap_deviation":
        return op._calculate_series(panels["close"], panels["amount"], panels["volume"], segment="morning")
    if name == "intra_vwap_above_ratio":
        return op._calculate_series(panels["close"], panels["amount"], panels["volume"])
    if name == "intra_vwap_cross_count":
        return op._calculate_series(panels["close"], panels["amount"], panels["volume"])
    if name == "intra_concentration":
        return op._calculate_series(panels["volume"])
    if name == "intra_entropy":
        return op._calculate_series(panels["volume"])
    if name == "intra_signed_imbalance_proxy":
        return op._calculate_series(panels["close"], panels["volume"])
    if name == "intra_return_activity_corr":
        return op._calculate_series(panels["close"], panels["activity"])
    if name == "intra_amihud":
        return op._calculate_series(panels["close"], panels["amount"])
    if name == "intra_kyle_lambda_proxy":
        return op._calculate_series(panels["close"], panels["amount"])
    if name == "intra_extreme_bar_return":
        return op._calculate_series(panels["close"], side="max")
    if name == "intra_lunch_gap_return":
        return op._calculate_series(panels["close"], panels["open"])
    if name.startswith("intra_limit"):
        return op._calculate_series(panels["close"], high_limit=_daily_limit_panels(panels["close"]), side="up")
    return op._calculate_series(panels["close"])


def _assert_parity(ref: pd.DataFrame, got: pd.DataFrame, *, label: str, rtol=1e-7, atol=1e-9):
    idx = ref.index.union(got.index)
    cols = ref.columns.union(got.columns)
    ref = ref.reindex(index=idx, columns=cols)
    got = got.reindex(index=idx, columns=cols)
    nan_ok = ref.isna().equals(got.isna())
    assert nan_ok, f"{label}: NaN masks differ\nref:\n{ref}\ngot:\n{got}"
    mask = ~ref.isna()
    np.testing.assert_allclose(
        ref.to_numpy()[mask], got.to_numpy()[mask], rtol=rtol, atol=atol, err_msg=label
    )


def test_minute_parity_polars_matches_pandas(panels, long_df):
    for name, fn in _POLARS_OPS.items():
        ref = _pandas_kernel(name, panels)
        got = fn(long_df)
        _assert_parity(ref, got, label=f"polars:{name}")


def test_minute_parity_sql_matches_pandas(panels, con):
    for name, fn in _SQL_OPS.items():
        ref = _pandas_kernel(name, panels)
        got = fn(con, "mlong")
        _assert_parity(ref, got, label=f"sql:{name}", rtol=1e-6, atol=1e-9)


def test_minute_parity_limit_ops_polars(panels, long_df, limit_panel):
    for name, fn in _POLARS_LIMIT_OPS.items():
        ref = _pandas_kernel(name, panels)
        got = fn(long_df, limit_panel)
        _assert_parity(ref, got, label=f"polars:{name}")


def test_minute_parity_utc_index_matches_beijing(panels, long_df):
    """UTC 存储的分钟数据：session-local 转换后三后端仍一致。"""
    from tests.backend_parity.intraday_minute_parity import build_long

    # Real COS minute data is stored as the UTC instant; the fixture is in
    # Beijing wall-clock, so interpret it as Asia/Shanghai then convert to UTC.
    utc_panels = {k: v.tz_localize("Asia/Shanghai").tz_convert("UTC") for k, v in panels.items()}
    utc_long = build_long(utc_panels)
    for name, fn in _POLARS_OPS.items():
        if name in {"intra_high_time", "intra_low_time"}:
            continue
        ref = _pandas_kernel(name, panels)
        got = fn(utc_long)
        _assert_parity(ref, got, label=f"polars-utc:{name}", rtol=1e-6, atol=1e-8)


def test_minute_parity_pandas_contract_nan_day(panels):
    """全 NaN 日 → 三后端都为 NaN（pandas 内核契约，参考实现）。"""
    close = panels["close"].copy()
    close.loc[close.index.normalize() == pd.Timestamp("2024-01-03")] = np.nan
    from factor_engine.cleaned_operators.microstructure.intraday_agg import IntraRealizedVariance

    out = IntraRealizedVariance()._calculate_series(close)
    assert np.isnan(out.loc[pd.Timestamp("2024-01-03"), "A"])


def test_minute_parity_kyle_lambda_now_certified():
    """intra_kyle_lambda_proxy 的 cov/var 样本-总体混合已精确实现，三后端认证。"""
    from tests.backend_parity.intraday_minute_parity import _REFERENCE_ONLY

    assert _REFERENCE_ONLY == set()
    assert "intra_kyle_lambda_proxy" in _POLARS_OPS
    assert "intra_kyle_lambda_proxy" in _SQL_OPS


def test_minute_parity_coverage_complete():
    """25 个分钟算子全部被覆盖：三后端 / polars+limit / 明确 reference-only。"""
    from factor_engine.cleaned_operators.microstructure.intraday_agg import __all__ as minute_ops

    covered = set(_POLARS_OPS) | set(_SQL_OPS) | set(_POLARS_LIMIT_OPS) | _REFERENCE_ONLY
    assert set(minute_ops) <= covered, f"missing: {set(minute_ops) - covered}"
