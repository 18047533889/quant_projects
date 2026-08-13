# -*- coding: utf-8 -*-
"""Tests for R47 indicator / chip-surface / intraday-limit / panel operators.

The four R47 modules are NOT yet wired into ``cleaned_operators._LOAD_MODULES``
(they are wired centrally later), so they must be imported here BEFORE
``ensure_cleaned_loaded()`` so the ``@register_operator`` decorators run at
import time; ``load_all`` then runs governance/finalize over them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.technical.new_indicators  # noqa: F401
import cleaned_operators.technical.chip_ops  # noqa: F401
import cleaned_operators.intraday.limit_eod  # noqa: F401
import cleaned_operators.cross_section.panel_gap  # noqa: F401

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

INDICATOR_CANONICALS = (
    "HMA QQE RSX ALMA CoppockCurve ElderRay FisherTransform"
).split()
CHIP_CANONICALS = (
    "turnover_chip_age_cost_surface turnover_chip_overhang_surface"
).split()
INTRADAY_CANONICALS = (
    "intra_limit_pre_hit_pressure_profile intra_eod_reversal_decomposition"
).split()
PANEL_CANONICALS = (
    "panel_async_beta_ex_self panel_factor_pocket_strength "
    "cs_predictability_mosaic_score panel_predictability_mosaic_score"
).split()
ALL = INDICATOR_CANONICALS + CHIP_CANONICALS + INTRADAY_CANONICALS + PANEL_CANONICALS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _daily_panel(days: int = 100, cols: int = 2, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="B")
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((days, cols)) * 0.01, axis=0)) * 100.0,
        index=idx,
        columns=[f"C{i}" for i in range(cols)],
    )


def _minute_panel(days: int = 2, bars: int = 240, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        for m in list(range(571, 691))[: bars // 2] + list(range(781, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.005, axis=0)) * 100.0,
        index=pd.DatetimeIndex(timestamps),
        columns=[f"C{i}" for i in range(cols)],
    )


def _daily_limits(days: int = 2, cols: int = 2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp("2024-01-03") + pd.Timedelta(days=d) for d in range(days)]
    base = 100.0 * (1.0 + rng.standard_normal((days, cols)) * 0.02)
    return (
        pd.DataFrame(base * 1.10, index=pd.DatetimeIndex(dates), columns=[f"C{i}" for i in range(cols)]),
        pd.DataFrame(base * 0.90, index=pd.DatetimeIndex(dates), columns=[f"C{i}" for i in range(cols)]),
    )


def _all_nan(template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# registration + surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ALL)))
def test_registered_and_classified(name: str) -> None:
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


# ---------------------------------------------------------------------------
# daily technical indicators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(INDICATOR_CANONICALS)))
def test_indicator_shape_and_determinism(name: str) -> None:
    panel = _daily_panel(days=80, seed=1)
    op = OperatorRegistry.get(name)
    high = panel * 1.02
    low = panel * 0.98
    if name == "ElderRay":
        first = op.calculate(high, low, panel)
        second = op.calculate(high, low, panel)
    elif name == "FisherTransform":
        first = op.calculate(high, low)
        second = op.calculate(high, low)
    else:
        first = op.calculate(panel)
        second = op.calculate(panel)
    assert first.shape == panel.shape, name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_hma_golden() -> None:
    x = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    df = pd.DataFrame(x, index=pd.date_range("2024-01-01", periods=6), columns=["A"])
    out = OperatorRegistry.get("HMA").calculate(df, window=3, rounding="floor")["A"]
    # n2 = floor(3/2)=1, ns=floor(sqrt(3))=1 -> HMA = 2*x - WMA(x,3)
    wma3 = np.full(6, np.nan)
    for t in range(2, 6):
        wma3[t] = (x[t - 2] + 2.0 * x[t - 1] + 3.0 * x[t]) / 6.0
    expected = 2.0 * x - wma3
    for t in range(2, 6):
        assert out.iloc[t] == pytest.approx(expected[t], rel=1e-9)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])


def test_qqe_line_matches_wilder_rsi_ma() -> None:
    x = _daily_panel(days=60, cols=1, seed=3)["C0"]
    df = x.to_frame("A")
    out = OperatorRegistry.get("QQE").calculate(df, length=3, smooth=2, output="line")["A"]
    delta = x.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1.0 / 3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1.0 / 3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    expected = rsi.rolling(2).mean()
    valid = expected.notna() & out.notna()
    pd.testing.assert_series_equal(out[valid], expected[valid], check_dtype=False, check_names=False)


def test_rsx_constant_emits_neutral_50() -> None:
    df = pd.DataFrame(np.full(20, 100.0), index=pd.date_range("2024-01-01", periods=20), columns=["A"])
    out = OperatorRegistry.get("RSX").calculate(df, length=3)["A"]
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert (out.iloc[2:] == 50.0).all()


def test_rsx_bounded_and_warmup() -> None:
    df = _daily_panel(days=200, cols=1, seed=4)
    out = OperatorRegistry.get("RSX").calculate(df, length=14)["C0"]
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all() and (valid <= 100.0).all()
    assert out.isna().sum() >= 13  # NaN for first length-1 obs


def test_alma_golden() -> None:
    window = 3
    offset = 0.85
    sigma = 6.0
    x = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 16.0])
    df = pd.DataFrame(x, index=pd.date_range("2024-01-01", periods=6), columns=["A"])
    out = OperatorRegistry.get("ALMA").calculate(df, window=window, offset=offset, sigma=sigma)["A"]
    m = offset * (window - 1)
    s = window / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(window)])
    weights = weights / weights.sum()
    expected = np.full(6, np.nan)
    for t in range(window - 1, 6):
        expected[t] = float(np.dot(x[t - window + 1 : t + 1], weights))
    for t in range(window - 1, 6):
        assert out.iloc[t] == pytest.approx(expected[t], rel=1e-9)


def test_coppock_golden() -> None:
    close = np.array([10.0, 11.0, 12.0, 13.0, 15.0, 14.0])
    df = pd.DataFrame(close, index=pd.date_range("2024-01-01", periods=6), columns=["A"])
    out = OperatorRegistry.get("CoppockCurve").calculate(df, roc1=2, roc2=3, wma_window=2, roc_mode="pct")["A"]
    roc2 = close / np.roll(close, 2) - 1.0
    roc3 = close / np.roll(close, 3) - 1.0
    roc2[:2] = np.nan
    roc3[:3] = np.nan
    total = roc2 + roc3
    expected = np.full(6, np.nan)
    for t in range(1, 6):  # WMA(2) weights [1,2] newest-largest
        if np.isfinite(total[t - 1]) and np.isfinite(total[t]):
            expected[t] = (total[t - 1] + 2.0 * total[t]) / 3.0
    for t in range(6):
        if np.isfinite(expected[t]):
            assert out.iloc[t] == pytest.approx(expected[t], rel=1e-9)
        else:
            assert np.isnan(out.iloc[t])


def test_elder_ray_golden() -> None:
    high = np.array([10.5, 11.5, 12.5])
    low = np.array([8.5, 9.5, 10.5])
    close = np.array([10.0, 11.0, 12.0])
    idx = pd.date_range("2024-01-01", periods=3)
    hdf = pd.DataFrame(high, index=idx, columns=["A"])
    ldf = pd.DataFrame(low, index=idx, columns=["A"])
    cdf = pd.DataFrame(close, index=idx, columns=["A"])
    out = OperatorRegistry.get("ElderRay").calculate(hdf, ldf, cdf, ema=2, output="bull")["A"]
    e1 = (2.0 / 3.0) * close[1] + (1.0 / 3.0) * close[0]
    e2 = (2.0 / 3.0) * close[2] + (1.0 / 3.0) * e1
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(high[1] - e1, rel=1e-9)
    assert out.iloc[2] == pytest.approx(high[2] - e2, rel=1e-9)


def test_fisher_transform_golden() -> None:
    high = np.array([10.0, 11.0, 12.0, 13.0])
    low = np.array([8.0, 9.0, 10.0, 11.0])
    idx = pd.date_range("2024-01-01", periods=4)
    hdf = pd.DataFrame(high, index=idx, columns=["A"])
    ldf = pd.DataFrame(low, index=idx, columns=["A"])
    out = OperatorRegistry.get("FisherTransform").calculate(
        hdf, ldf, window=3, smooth=0.5, signal_smooth=0.5, output="value"
    )["A"]
    source = (high + low) / 2.0
    # t=2: min=9,max=11, raw=1 -> z=0.5 -> fisher=0.5*ln(3)
    z2 = 0.5 * 1.0
    exp2 = 0.5 * np.log((1.0 + z2) / (1.0 - z2))
    # t=3: min=9,max=12, raw=1 -> z=0.5*1+0.5*0.5=0.75 -> fisher=0.5*ln(7)
    z3 = 0.5 * 1.0 + 0.5 * z2
    exp3 = 0.5 * np.log((1.0 + z3) / (1.0 - z3))
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(exp2, rel=1e-9)
    assert out.iloc[3] == pytest.approx(exp3, rel=1e-9)


@pytest.mark.parametrize("name", sorted(set(INDICATOR_CANONICALS)))
def test_indicator_all_nan(name: str) -> None:
    panel = _daily_panel(days=60, seed=5)
    nan_panel = _all_nan(panel)
    op = OperatorRegistry.get(name)
    if name == "ElderRay":
        out = op.calculate(nan_panel, nan_panel, nan_panel)
    elif name == "FisherTransform":
        out = op.calculate(nan_panel, nan_panel)
    else:
        out = op.calculate(nan_panel)
    assert out.shape == panel.shape, name
    assert out.isna().all().all(), name


@pytest.mark.parametrize("name", sorted(set(INDICATOR_CANONICALS)))
def test_indicator_future_poison(name: str) -> None:
    panel = _daily_panel(days=80, seed=6)
    op = OperatorRegistry.get(name)
    high, low = panel * 1.02, panel * 0.98
    h_t, l_t = high.copy(), low.copy()
    h_t.iloc[50:] *= 10.0
    l_t.iloc[50:] *= 10.0
    if name == "ElderRay":
        base = op.calculate(high, low, panel)
        changed = op.calculate(h_t, l_t, panel)
    elif name == "FisherTransform":
        base = op.calculate(high, low)
        changed = op.calculate(h_t, l_t)
    else:
        base = op.calculate(panel)
        tampered = panel.copy()
        tampered.iloc[50:] *= 10.0
        changed = op.calculate(tampered)
    pd.testing.assert_series_equal(base.iloc[0], changed.iloc[0], check_dtype=False)


# ---------------------------------------------------------------------------
# daily chip ops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(CHIP_CANONICALS)))
def test_chip_shape_and_determinism(name: str) -> None:
    close = _daily_panel(days=200, seed=7)
    turnover = pd.DataFrame(
        np.abs(np.random.default_rng(8).standard_normal(close.shape)) * 0.05,
        index=close.index,
        columns=close.columns,
    )
    op = OperatorRegistry.get(name)
    first = op.calculate(close, turnover, window=30)
    second = op.calculate(close, turnover, window=30)
    assert first.shape == close.shape, name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_chip_age_cost_turnover_one_entropy_zero() -> None:
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.DataFrame(np.full(n, 100.0), index=idx, columns=["A"])
    turnover = pd.DataFrame(np.ones(n), index=idx, columns=["A"])
    op = OperatorRegistry.get("turnover_chip_age_cost_surface")
    # window=5, price_bins=4, age_bins=4 -> grid collapses to a single cell
    out = op.calculate(close, turnover, window=5, price_bins=4, age_bins=4)
    entropy = out["A"]
    assert np.isnan(entropy.iloc[0]) and np.isnan(entropy.iloc[3])
    assert entropy.iloc[4] == pytest.approx(0.0, abs=1e-9)
    mi = op.calculate(close, turnover, window=5, price_bins=4, age_bins=4, output="age_price_mi")["A"]
    assert mi.iloc[4] == pytest.approx(0.0, abs=1e-9)
    slope = op.calculate(close, turnover, window=5, price_bins=4, age_bins=4, output="cost_age_slope")["A"]
    assert np.isnan(slope.iloc[4])


def test_chip_overhang_turnover_one() -> None:
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.DataFrame(np.full(n, 100.0), index=idx, columns=["A"])
    turnover = pd.DataFrame(np.ones(n), index=idx, columns=["A"])
    op = OperatorRegistry.get("turnover_chip_overhang_surface")
    out = op.calculate(close, turnover, window=5, bins=8)["A"]
    # all mass sits in a single cost bin at the current close (bin rep slightly above)
    assert out.iloc[4] == pytest.approx(1.0, abs=1e-9)  # overhang_mass == 1
    under = op.calculate(close, turnover, window=5, bins=8, output="under_price_mass")["A"]
    assert under.iloc[4] == pytest.approx(0.0, abs=1e-9)
    ratio = op.calculate(close, turnover, window=5, bins=8, output="overhead_supply_ratio")["A"]
    assert np.isnan(ratio.iloc[4])
    vac = op.calculate(close, turnover, window=5, bins=8, output="supply_vacuum")["A"]
    assert np.isnan(vac.iloc[4])


@pytest.mark.parametrize("name", sorted(set(CHIP_CANONICALS)))
def test_chip_all_nan_and_future_poison(name: str) -> None:
    close = _daily_panel(days=100, seed=9)
    turnover = pd.DataFrame(np.abs(np.random.default_rng(10).standard_normal(close.shape)) * 0.05,
                            index=close.index, columns=close.columns)
    op = OperatorRegistry.get(name)
    out = op.calculate(_all_nan(close), _all_nan(turnover), window=30)
    assert out.isna().all().all(), name
    base = op.calculate(close, turnover, window=30)
    tampered = close.copy()
    tampered.iloc[60:] *= 10.0
    changed = op.calculate(tampered, turnover, window=30)
    pd.testing.assert_series_equal(base.iloc[0], changed.iloc[0], check_dtype=False)


# ---------------------------------------------------------------------------
# intraday limit + EOD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(INTRADAY_CANONICALS)))
def test_intraday_shape_and_determinism(name: str) -> None:
    close = _minute_panel(days=2, seed=11)
    volume = pd.DataFrame(np.abs(np.random.default_rng(12).standard_normal(close.shape)) + 1.0,
                          index=close.index, columns=close.columns)
    hl, ll = _daily_limits(days=2, seed=13)
    op = OperatorRegistry.get(name)
    if name == "intra_eod_reversal_decomposition":
        first = op.calculate(close, volume, window_minutes=30)
        second = op.calculate(close, volume, window_minutes=30)
    else:
        first = op.calculate(close, volume, hl, ll)
        second = op.calculate(close, volume, hl, ll)
    assert first.shape == (2, 2), name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def _one_day_rising() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A-share single day: morning flat at 100, afternoon rising 100->110."""
    minutes = list(range(571, 691)) + list(range(781, 901))
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-03") + pd.Timedelta(minutes=m) for m in minutes])
    afternoon = np.linspace(100.0, 110.0, 120)
    prices = np.concatenate([np.full(120, 100.0), afternoon])
    price = pd.DataFrame(prices, index=idx, columns=["A"])
    volume = pd.DataFrame(np.ones(len(idx)), index=idx, columns=["A"])
    hl = pd.DataFrame({"A": [108.0]}, index=pd.DatetimeIndex(["2024-01-03"]))
    ll = pd.DataFrame({"A": [80.0]}, index=pd.DatetimeIndex(["2024-01-03"]))
    return price, volume, hl, ll


def test_limit_pre_hit_return_accel_golden() -> None:
    price, volume, hl, ll = _one_day_rising()
    op = OperatorRegistry.get("intra_limit_pre_hit_pressure_profile")
    out = op.calculate(price, volume, hl, ll, side="up", pre_window=5, output="return_accel")
    vals = price["A"].to_numpy()
    hit_idx = int(np.argmax(vals) >= 108.0))
    assert 0 < hit_idx < len(vals)
    pre = vals[hit_idx - 5 : hit_idx]
    expected = float(np.polyfit(np.arange(5), np.log(pre), 1)[0])
    assert out["A"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_limit_pre_hit_volume_accel_flat() -> None:
    price, volume, hl, ll = _one_day_rising()
    op = OperatorRegistry.get("intra_limit_pre_hit_pressure_profile")
    out = op.calculate(price, volume, hl, ll, side="up", pre_window=5, output="volume_accel")
    assert out["A"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_limit_pre_hit_require_hit_false_no_hit() -> None:
    price, volume, hl, ll = _one_day_rising()
    op = OperatorRegistry.get("intra_limit_pre_hit_pressure_profile")
    high_no_hit = pd.DataFrame({"A": [999.0]}, index=pd.DatetimeIndex(["2024-01-03"]))
    out_true = op.calculate(price, volume, high_no_hit, ll, side="up", pre_window=5, require_hit=True)
    assert np.isnan(out_true["A"].iloc[0])
    out_false = op.calculate(price, volume, high_no_hit, ll, side="up", pre_window=5, require_hit=False)
    assert np.isfinite(out_false["A"].iloc[0])


def test_limit_pre_hit_open_at_limit_nan() -> None:
    # price opens already at the limit -> zero-length pre-hit path -> NaN
    price, volume, hl, ll = _one_day_rising()
    price = price.copy()
    price.iloc[:] = 200.0  # everything at/above the limit
    out = OperatorRegistry.get("intra_limit_pre_hit_pressure_profile").calculate(
        price, volume, hl, ll, side="up", pre_window=5, output="return_accel"
    )
    assert np.isnan(out["A"].iloc[0])


def test_eod_reversal_decomposition_pressure_golden() -> None:
    price, volume, hl, ll = _one_day_rising()
    op = OperatorRegistry.get("intra_eod_reversal_decomposition")
    out = op.calculate(price, volume, window_minutes=30, baseline="prior_window", output="pressure")["A"]
    vals = price["A"].to_numpy()
    n = len(vals)
    cw = vals[n - 30 :]
    bw = vals[n - 60 : n - 30]
    rets = np.diff(np.log(bw))
    expected = (cw[-1] - cw[0]) / float(np.std(rets))
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_eod_reversal_morning_baseline_pressure_nan() -> None:
    # morning is flat -> baseline std = 0 -> pressure NaN (never bridges lunch)
    price, volume, hl, ll = _one_day_rising()
    op = OperatorRegistry.get("intra_eod_reversal_decomposition")
    out = op.calculate(price, volume, window_minutes=30, baseline="morning", output="pressure")["A"]
    assert np.isnan(out.iloc[0])


def test_intraday_lunch_boundary_not_bridged() -> None:
    """A volatile morning session must never leak into the prior/close windows."""
    price, volume, hl, ll = _one_day_rising()
    price = price.copy()
    # make the morning session extremely volatile
    rng = np.random.default_rng(0)
    price.loc[price.index[:120], "A"] = 100.0 + np.cumsum(rng.standard_normal(120)) * 5.0
    op = OperatorRegistry.get("intra_eod_reversal_decomposition")
    out = op.calculate(price, volume, window_minutes=30, baseline="prior_window", output="retention")["A"]
    vals = price["A"].to_numpy()
    cw = vals[210:240]
    bw = vals[180:210]
    expected = (cw[-1] - cw[0]) / (abs(bw[-1] - bw[0]) + 1e-12)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("name", sorted(set(INTRADAY_CANONICALS)))
def test_intraday_future_day_poison(name: str) -> None:
    close = _minute_panel(days=3, seed=14)
    volume = pd.DataFrame(np.abs(np.random.default_rng(15).standard_normal(close.shape)) + 1.0,
                          index=close.index, columns=close.columns)
    hl, ll = _daily_limits(days=3, seed=16)
    op = OperatorRegistry.get(name)
    if name == "intra_limit_pre_hit_pressure_profile":
        base = op.calculate(close, volume, hl, ll, pre_window=10)
    else:
        base = op.calculate(close, volume, window_minutes=30)
    tampered = close.copy()
    tampered.iloc[-240:] *= 10.0  # corrupt the last day's bars
    if name == "intra_limit_pre_hit_pressure_profile":
        changed = op.calculate(tampered, volume, hl, ll, pre_window=10)
    else:
        changed = op.calculate(tampered, volume, window_minutes=30)
    pd.testing.assert_series_equal(base.iloc[0], changed.iloc[0], check_dtype=False)


# ---------------------------------------------------------------------------
# cross-sectional / panel ops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(PANEL_CANONICALS)))
def test_panel_shape_and_determinism(name: str) -> None:
    n, cols = 80, 3
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(17)
    ret = pd.DataFrame(rng.standard_normal((n, cols)) * 0.01, index=idx, columns=["A", "B", "C"])
    weight = pd.DataFrame(np.abs(rng.standard_normal((n, cols))) + 0.5, index=idx, columns=["A", "B", "C"])
    factor = pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=["A", "B", "C"])
    op = OperatorRegistry.get(name)
    if name == "panel_async_beta_ex_self":
        args = (ret, weight)
        kw = dict(window=30, min_periods=15, refresh_freq=5)
    elif name == "panel_factor_pocket_strength":
        args = (ret, factor)
        kw = dict(window=30, min_periods=15)
    else:
        args = (factor, ret, factor)
        kw = dict(window=30, min_history=15, clusters=2, lag=1)
    first = op.calculate(*args, **kw)
    second = op.calculate(*args, **kw)
    assert first.shape == ret.shape, name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_panel_async_beta_ex_self_golden() -> None:
    # two stocks, equal weights; A's ex-self market return == B's return
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    ret = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03, 0.04, 0.05],
         "B": [-0.01, -0.02, -0.03, -0.04, -0.05]},
        index=idx,
    )
    weight = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=idx)
    out = OperatorRegistry.get("panel_async_beta_ex_self").calculate(
        ret, weight, window=3, min_periods=2, refresh_freq=3
    )
    # beta of A on (ex-self market = B): r_A = -1 * r_B -> beta = -1 at t>=3
    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[3] == pytest.approx(-1.0, rel=1e-9)
    assert out["A"].iloc[4] == pytest.approx(-1.0, rel=1e-9)


def test_panel_async_beta_ex_self_constant_nan() -> None:
    n, cols = 60, 3
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    ret = pd.DataFrame(np.zeros((n, cols)), index=idx, columns=["A", "B", "C"])
    weight = pd.DataFrame(np.ones((n, cols)), index=idx, columns=["A", "B", "C"])
    out = OperatorRegistry.get("panel_async_beta_ex_self").calculate(
        ret, weight, window=30, min_periods=10, refresh_freq=5
    )
    assert out.isna().all().all()


def test_panel_factor_pocket_strength_golden() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    factor = pd.DataFrame({"A": [3.0, 3.0], "B": [2.0, 2.0], "C": [1.0, 1.0]}, index=idx)
    ret = pd.DataFrame({"A": [0.05, 0.05], "B": [0.01, 0.01], "C": [-0.02, -0.02]}, index=idx)
    out = OperatorRegistry.get("panel_factor_pocket_strength").calculate(
        ret, factor, window=2, min_periods=2, threshold=0.5
    )
    # each day: cs_rank pct = [1, 2/3, 1/3] -> A,B in pocket; cs_median ret = 0.01
    # A: in-pocket AND ret > median -> 1 on both days -> strength 1.0
    # B: in-pocket but ret == median -> 0 -> strength 0.0 ; C: not in pocket -> 0
    assert out["A"].iloc[1] == pytest.approx(1.0, abs=1e-9)
    assert out["B"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert out["C"].iloc[1] == pytest.approx(0.0, abs=1e-9)


def test_cs_predictability_mosaic_score_golden() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    state = pd.DataFrame(
        {"A": [1.0, 4.0, 1.0, 4.0],
         "B": [2.0, 3.0, 2.0, 3.0],
         "C": [3.0, 2.0, 3.0, 2.0],
         "D": [4.0, 1.0, 4.0, 1.0]},
        index=idx,
    )
    base_signal = pd.DataFrame(
        {"A": [1.0] * 4, "B": [2.0] * 4, "C": [3.0] * 4, "D": [4.0] * 4},
        index=idx,
    )
    realized = base_signal.copy()  # perfectly positively ranked
    out = OperatorRegistry.get("cs_predictability_mosaic_score").calculate(
        base_signal, realized, state, window=3, min_history=2, clusters=2, lag=1
    )
    # At t=3 the state buckets are [1,1,1,0] (A,B,C in bucket 1); the historical
    # per-date rank-ICs of bucket 1 are all +1.0 -> mosaic = 1.0 for A,B,C.
    assert out["A"].iloc[3] == pytest.approx(1.0, abs=1e-9)
    assert out["B"].iloc[3] == pytest.approx(1.0, abs=1e-9)
    assert out["C"].iloc[3] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(out["D"].iloc[3])


def test_panel_predictability_mosaic_matches_cs_mean() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    state = pd.DataFrame(
        {"A": [1.0, 4.0, 1.0, 4.0],
         "B": [2.0, 3.0, 2.0, 3.0],
         "C": [3.0, 2.0, 3.0, 2.0],
         "D": [4.0, 1.0, 4.0, 1.0]},
        index=idx,
    )
    base_signal = pd.DataFrame(
        {"A": [1.0] * 4, "B": [2.0] * 4, "C": [3.0] * 4, "D": [4.0] * 4}, index=idx
    )
    realized = base_signal.copy()
    cs = OperatorRegistry.get("cs_predictability_mosaic_score").calculate(
        base_signal, realized, state, window=3, min_history=2, clusters=2, lag=1
    )
    panel = OperatorRegistry.get("panel_predictability_mosaic_score").calculate(
        base_signal, realized, state, window=3, min_history=2, clusters=2, lag=1
    )
    row = cs.iloc[3].to_numpy(dtype=float)
    expected = float(np.nanmean(row))
    assert panel["A"].iloc[3] == pytest.approx(expected, abs=1e-9)
    assert panel["B"].iloc[3] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", sorted(set(PANEL_CANONICALS)))
def test_panel_future_poison(name: str) -> None:
    n, cols = 80, 3
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(18)
    ret = pd.DataFrame(rng.standard_normal((n, cols)) * 0.01, index=idx, columns=["A", "B", "C"])
    weight = pd.DataFrame(np.abs(rng.standard_normal((n, cols))) + 0.5, index=idx, columns=["A", "B", "C"])
    factor = pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=["A", "B", "C"])
    op = OperatorRegistry.get(name)
    if name == "panel_async_beta_ex_self":
        base = op.calculate(ret, weight, window=30, min_periods=15, refresh_freq=5)
        tampered = ret.copy()
        tampered.iloc[60:] *= 10.0
        changed = op.calculate(tampered, weight, window=30, min_periods=15, refresh_freq=5)
    elif name == "panel_factor_pocket_strength":
        base = op.calculate(ret, factor, window=30, min_periods=15)
        tampered = factor.copy()
        tampered.iloc[60:] *= 10.0
        changed = op.calculate(ret, tampered, window=30, min_periods=15)
    else:
        base = op.calculate(factor, ret, factor, window=30, min_history=15, clusters=2, lag=1)
        tampered = factor.copy()
        tampered.iloc[60:] *= 10.0
        changed = op.calculate(tampered, ret, tampered, window=30, min_history=15, clusters=2, lag=1)
    pd.testing.assert_series_equal(base.iloc[10], changed.iloc[10], check_dtype=False)


@pytest.mark.parametrize("name", sorted(set(PANEL_CANONICALS)))
def test_panel_all_nan(name: str) -> None:
    n, cols = 40, 3
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    nan_panel = pd.DataFrame(np.nan, index=idx, columns=["A", "B", "C"])
    op = OperatorRegistry.get(name)
    if name == "panel_async_beta_ex_self":
        out = op.calculate(nan_panel, nan_panel, window=30, min_periods=10, refresh_freq=5)
    elif name == "panel_factor_pocket_strength":
        out = op.calculate(nan_panel, nan_panel, window=30, min_periods=10)
    else:
        out = op.calculate(nan_panel, nan_panel, nan_panel, window=30, min_history=10, clusters=2, lag=1)
    assert out.shape == nan_panel.shape, name
    assert out.isna().all().all(), name
