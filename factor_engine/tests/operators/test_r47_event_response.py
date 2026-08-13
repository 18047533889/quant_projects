# -*- coding: utf-8 -*-
"""R47 intraday event/response operator tests (minute -> daily).

The module ``cleaned_operators.intraday.event_response`` is NOT yet wired into
the central ``_LOAD_MODULES`` list (that happens centrally later), so it is
imported explicitly here first to register its operators; ``ensure_cleaned_loaded``
then runs the governance/finalize pass over the whole registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.intraday.event_response  # noqa: F401  (registers operators at import time)

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

CANONICALS = (
    "intra_event_window_reduce intra_event_pre_post_contrast intra_impulse_event_detector "
    "intra_post_impulse_response intra_probe_outcome_score intra_supply_absorption_score "
    "intra_consolidation_quality intra_response_curve_features intra_liquidity_resilience_curve_fit"
).split()

_ALL = tuple(sorted(CANONICALS))


# ---------------------------------------------------------------------------
# Synthetic A-share minute panel helpers (tz-aware)
# ---------------------------------------------------------------------------

def _minute_panel(days: int = 3, bars: int = 240, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        # A-share session minute-of-day: 09:31..11:30 (571..690), 13:01..15:00 (781..900)
        for m in list(range(571, 691))[: bars // 2] + list(range(781, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    idx = pd.DatetimeIndex(timestamps).tz_localize("Asia/Shanghai")
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.01, axis=0)) * 100,
        index=idx,
        columns=[f"C{i}" for i in range(cols)],
    )
    return close


def _amount_volume(close: pd.DataFrame, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vol = pd.DataFrame(np.abs(rng.standard_normal(close.shape)) + 1.0, index=close.index, columns=close.columns)
    amount = vol * close
    return amount, vol


def _single_day(
    cols: dict[str, np.ndarray],
    minutes: list[int] | None = None,
    day: str = "2024-01-03",
) -> pd.DataFrame:
    """Build a one-day tz-aware minute panel from {col: array} aligned to minutes."""
    if minutes is None:
        n = len(next(iter(cols.values())))
        minutes = list(range(571, 571 + n))  # consecutive morning bars
    idx = pd.DatetimeIndex(
        [pd.Timestamp(day) + pd.Timedelta(minutes=int(m)) for m in minutes]
    ).tz_localize("Asia/Shanghai")
    return pd.DataFrame(cols, index=idx)


def _event_mask(n: int, *positions: int) -> np.ndarray:
    m = np.zeros(n, dtype=float)
    for p in positions:
        m[p] = 1.0
    return m


# ---------------------------------------------------------------------------
# Registration + surface + shape + determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ALL)
def test_registered_and_classified(name: str) -> None:
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


def test_shape_and_determinism_1_and_2() -> None:
    x = _minute_panel(seed=1)
    mask = pd.DataFrame((np.abs(x.to_numpy() > np.nanpercentile(np.abs(x.to_numpy()), 95)).astype(float),
                        index=x.index, columns=x.columns)
    for name in ("intra_event_window_reduce", "intra_event_pre_post_contrast"):
        op = OperatorRegistry.get(name)
        first = op.calculate(x, mask)
        second = op.calculate(x, mask)
        assert first.shape == (3, 2), name
        assert first.dtypes.eq("float64").all(), name
        pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_shape_and_determinism_3() -> None:
    close = _minute_panel(seed=1)
    op = OperatorRegistry.get("intra_impulse_event_detector")
    first = op.calculate(close)
    second = op.calculate(close)
    assert first.shape == (3, 2)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_shape_and_determinism_4_to_7() -> None:
    close = _minute_panel(seed=1)
    amount, vol = _amount_volume(close)
    for name in (
        "intra_post_impulse_response",
        "intra_probe_outcome_score",
        "intra_supply_absorption_score",
        "intra_consolidation_quality",
    ):
        op = OperatorRegistry.get(name)
        first = op.calculate(close, vol, amount)
        second = op.calculate(close, vol, amount)
        assert first.shape == (3, 2), name
        assert first.dtypes.eq("float64").all(), name
        pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_shape_and_determinism_8_and_9() -> None:
    close = _minute_panel(seed=1)
    amount, vol = _amount_volume(close)
    for name in ("intra_response_curve_features", "intra_liquidity_resilience_curve_fit"):
        op = OperatorRegistry.get(name)
        first = op.calculate(close, vol)
        second = op.calculate(close, vol)
        assert first.shape == (3, 2), name
        pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# Golden hand-computed small cases
# ---------------------------------------------------------------------------

def test_op1_golden_window_reduce() -> None:
    vals = np.arange(1.0, 11.0)
    x = _single_day({"A": vals})
    mask = _single_day({"A": _event_mask(10, 2)})
    op = OperatorRegistry.get("intra_event_window_reduce")
    out = op.calculate(x, mask, pre=2, post=3, reducer="mean", event_select="first", min_obs=3)
    # window = [0,5] -> x[0..5] = [1..6]; mean = 3.5
    assert out["A"].iloc[0] == pytest.approx(3.5, rel=1e-9)
    out = op.calculate(x, mask, pre=2, post=3, reducer="slope", event_select="first", min_obs=3)
    # slope of x over minute-of-day 571..576 for y=1..6 -> 1.0
    assert out["A"].iloc[0] == pytest.approx(1.0, rel=1e-9)
    out = op.calculate(x, mask, pre=2, post=3, reducer="std", event_select="first", min_obs=3)
    assert out["A"].iloc[0] == pytest.approx(np.std([1.0, 2, 3, 4, 5, 6], ddof=1), rel=1e-9)


def test_op2_golden_pre_post_contrast() -> None:
    vals = np.arange(1.0, 11.0)
    x = _single_day({"A": vals})
    mask = _single_day({"A": _event_mask(10, 4)})
    op = OperatorRegistry.get("intra_event_pre_post_contrast")
    out = op.calculate(x, mask, pre=2, post=3, metric="mean_diff", event_select="first", min_obs=2)
    # pre window [2,3] -> [3,4] mean 3.5 ; post [5,7] -> [6,7,8] mean 7.0 ; diff 3.5
    assert out["A"].iloc[0] == pytest.approx(3.5, rel=1e-9)
    out = op.calculate(x, mask, pre=2, post=3, metric="vol_ratio", event_select="first", min_obs=2)
    pre_std = float(np.std([3.0, 4.0], ddof=1))
    post_std = float(np.std([6.0, 7.0, 8.0], ddof=1))
    assert out["A"].iloc[0] == pytest.approx(post_std / pre_std, rel=1e-9)


def test_op3_golden_impulse_detector() -> None:
    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.005, 0.0, 0.0, 0.0, 0.0])
    prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(rets))])  # 10 bars
    p = _single_day({"A": prices})
    op = OperatorRegistry.get("intra_impulse_event_detector")
    kw = dict(event="both", threshold="quantile", z=0.5, merge_gap=2)
    # thr = quantile(|r|, 0.5) = 0.005 ; events at bar indices 1,2,3,4 -> one episode
    assert op.calculate(p, output="count", **kw)["A"].iloc[0] == 1.0
    assert op.calculate(p, output="strength", **kw)["A"].iloc[0] == pytest.approx(0.07, rel=1e-9)
    assert op.calculate(p, output="max_strength", **kw)["A"].iloc[0] == pytest.approx(0.07, rel=1e-9)
    assert op.calculate(p, output="first_time", **kw)["A"].iloc[0] == pytest.approx(0.1, rel=1e-9)
    assert op.calculate(p, output="last_time", **kw)["A"].iloc[0] == pytest.approx(0.4, rel=1e-9)
    assert op.calculate(p, output="duration", **kw)["A"].iloc[0] == 4.0


def test_op4_golden_post_impulse_response() -> None:
    prices = np.array([100.2, 100.1, 100.3, 100.2, 100.0, 105.0, 106.0, 104.0, 105.0, 105.0])
    vol = np.ones(10)
    p = _single_day({"A": prices})
    v = _single_day({"A": vol})
    a = _single_day({"A": prices.copy()})
    op = OperatorRegistry.get("intra_post_impulse_response")
    kw = dict(direction="up", threshold="quantile", z=0.95, horizon=3)
    # single impulse at bar 5 (r=log(105/100)); post window [6,7,8]
    assert op.calculate(p, v, a, output="retention", **kw)["A"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert op.calculate(p, v, a, output="giveback", **kw)["A"].iloc[0] == pytest.approx(104.0 / 106.0 - 1.0, rel=1e-9)
    assert op.calculate(p, v, a, output="max_drawdown", **kw)["A"].iloc[0] == pytest.approx(104.0 / 106.0 - 1.0, rel=1e-9)
    assert op.calculate(p, v, a, output="vwap_hold", **kw)["A"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert op.calculate(p, v, a, output="recovery_half_life", **kw)["A"].iloc[0] == pytest.approx(3.0)


def test_op5_golden_probe_outcome_score() -> None:
    prices = np.array([100.0, 100.0, 100.0, 103.0, 102.0, 102.5, 102.5, 106.0, 106.0, 105.0, 105.5, 106.0])
    vol = np.ones(12)
    p = _single_day({"A": prices})
    v = _single_day({"A": vol})
    a = _single_day({"A": prices.copy()})
    op = OperatorRegistry.get("intra_probe_outcome_score")
    kw = dict(direction="up", z=0.9, probe_horizon=2, response_horizon=3)
    # exactly two impulses at bars 3 and 7 (robust_z z=0.9); z-scores are +-1/sqrt(2)
    out = op.calculate(p, v, a, output="score", **kw)
    assert out["A"].iloc[0] == pytest.approx(0.025, abs=1e-6)
    out = op.calculate(p, v, a, output="failure", **kw)
    assert out["A"].iloc[0] == pytest.approx(0.5)
    out = op.calculate(p, v, a, output="holding", **kw)
    assert out["A"].iloc[0] == pytest.approx(0.5)
    # second push needs a longer response horizon to reach the second impulse
    out = op.calculate(p, v, a, output="second_push", direction="up", z=0.9,
                       probe_horizon=2, response_horizon=6)
    assert out["A"].iloc[0] == pytest.approx(0.5)


def test_op6_golden_supply_absorption() -> None:
    # returns: one clean up-impulse at bar 5 (robust_z z=3 fires only there)
    rets = np.array([0.001, -0.001, 0.001, -0.001, 0.15, 0.002, -0.002, 0.001, -0.001])
    prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(rets))])
    vol = np.ones(10)
    p = _single_day({"A": prices})
    v = _single_day({"A": vol})
    a = _single_day({"A": prices.copy()})
    op = OperatorRegistry.get("intra_supply_absorption_score")
    kw = dict(event="up_impulse", horizon=3)
    # impulse at bar 5; post window [6,7,8]; within-session returns into bars 6,7,8
    post_abs = (
        abs(np.log(prices[6] / prices[5]))
        + abs(np.log(prices[7] / prices[6]))
        + abs(np.log(prices[8] / prices[7]))
    )
    expected_absorption = 1.0 / (1.0 + post_abs / 3.0)
    out = op.calculate(p, v, a, output="absorption", **kw)
    assert out["A"].iloc[0] == pytest.approx(expected_absorption, rel=1e-9)
    out = op.calculate(p, v, a, output="volume_no_drop", **kw)
    assert out["A"].iloc[0] == pytest.approx(1.0, rel=1e-9)
    out = op.calculate(p, v, a, output="downside_resilience", **kw)
    path = np.concatenate([[prices[5]], prices[6:9]])
    dd = float(np.min(path / np.maximum.accumulate(path) - 1.0))
    assert out["A"].iloc[0] == pytest.approx(1.0 - dd, rel=1e-9)


def test_op7_golden_consolidation_quality() -> None:
    rets = np.array([0.001, -0.001, 0.001, -0.001, 0.15, 0.002, -0.002, 0.001, -0.001])
    prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(rets))])
    vol = np.ones(10)
    p = _single_day({"A": prices})
    v = _single_day({"A": vol})
    a = _single_day({"A": prices.copy()})
    op = OperatorRegistry.get("intra_consolidation_quality")
    kw = dict(trigger="impulse", trigger_z=3.0, horizon=3)
    # post window [6,7,8]
    post_mean = float(np.mean(prices[6:9]))
    tightness = 1.0 - (float(np.max(prices[6:9])) - float(np.min(prices[6:9]))) / post_mean
    out = op.calculate(p, v, a, output="tightness", **kw)
    assert out["A"].iloc[0] == pytest.approx(tightness, rel=1e-9)
    out = op.calculate(p, v, a, output="level", **kw)
    assert out["A"].iloc[0] == pytest.approx(post_mean / prices[5] - 1.0, rel=1e-9)
    out = op.calculate(p, v, a, output="volume_dryup", **kw)
    assert out["A"].iloc[0] == pytest.approx(1.0, rel=1e-9)
    out = op.calculate(p, v, a, output="rising_floor", **kw)
    min_pre = float(np.min(prices[2:5]))
    assert out["A"].iloc[0] == pytest.approx((float(np.min(prices[6:9])) - min_pre) / min_pre, rel=1e-9)
    out = op.calculate(p, v, a, output="breakout_readiness", **kw)
    sign = -1.0 if prices[8] / prices[6] - 1.0 < 0.0 else 1.0
    assert out["A"].iloc[0] == pytest.approx(sign * tightness, rel=1e-9)


def test_op8_golden_response_curve_features() -> None:
    rets = np.array([0.001, -0.001, 0.001, -0.001, 0.15, 0.002, -0.002, 0.001, -0.001])
    prices = np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(rets))])
    vol = np.ones(10)
    p = _single_day({"A": prices})
    act = _single_day({"A": vol})
    op = OperatorRegistry.get("intra_response_curve_features")
    kw = dict(trigger="impulse", horizon=3, curve="price")
    y = prices[6:9] / prices[5] - 1.0
    x = np.arange(3, dtype=float)
    out = op.calculate(p, act, output="slope", **kw)
    assert out["A"].iloc[0] == pytest.approx(np.polyfit(x, y, 1)[0], rel=1e-6)
    out = op.calculate(p, act, output="auc", **kw)
    assert out["A"].iloc[0] == pytest.approx(float(np.sum(y)), rel=1e-9)
    out = op.calculate(p, act, output="half_life", **kw)
    maxy = float(np.max(np.abs(y)))
    half_life = float(next(k for k in range(3) if abs(y[k]) <= 0.5 * maxy))
    assert out["A"].iloc[0] == pytest.approx(half_life)
    out = op.calculate(p, act, output="monotonicity", **kw)
    last_move = y[-1] - y[0]
    target = 1.0 if last_move > 0 else -1.0
    assert out["A"].iloc[0] == pytest.approx(float(np.mean(np.sign(np.diff(y)) == target)))
    out = op.calculate(p, act, output="change_count", **kw)
    assert out["A"].iloc[0] == pytest.approx(1.0)
    out = op.calculate(p, act, output="curvature", **kw)
    assert out["A"].iloc[0] == pytest.approx(np.polyfit(x, y, 2)[0], rel=1e-6)


def test_op9_golden_liquidity_resilience_curve_fit() -> None:
    prices = np.array([100.0] * 5 + [105.0] * 6)
    activity = np.array([10.0, 10, 10, 10, 10, 10, 30, 20, 15, 12, 11])
    p = _single_day({"A": prices})
    act = _single_day({"A": activity})
    op = OperatorRegistry.get("intra_liquidity_resilience_curve_fit")
    kw = dict(shock_threshold=2.5, horizon=4)
    # shock at bar 5 (log return 0.04879 > 2.5 * session_std); post activity [30,20,15,12]
    a_inf = 12.0
    res0 = 30.0 - a_inf
    out = op.calculate(p, act, output="residual", **kw)
    assert out["A"].iloc[0] == pytest.approx(res0, rel=1e-9)
    out = op.calculate(p, act, output="asymptote", **kw)
    assert out["A"].iloc[0] == pytest.approx(a_inf, rel=1e-9)
    out = op.calculate(p, act, output="slope", **kw)
    k = np.arange(4, dtype=float)
    assert out["A"].iloc[0] == pytest.approx(np.polyfit(k, activity[6:10], 1)[0], rel=1e-9)
    # log-residual OLS slope on usable points k=0,1,2 -> tau -> half-life
    lr = np.log(np.array([18.0, 8.0, 3.0]))
    kk = np.array([0.0, 1.0, 2.0])
    slope = np.polyfit(kk, lr, 1)[0]
    assert slope < 0
    tau = -1.0 / slope
    out = op.calculate(p, act, output="half_life", **kw)
    assert out["A"].iloc[0] == pytest.approx(tau * np.log(2.0), rel=1e-9)


# ---------------------------------------------------------------------------
# Causality: corrupting future days must not change past outputs
# ---------------------------------------------------------------------------

def test_future_poison_all() -> None:
    close = _minute_panel(days=4, seed=6, cols=2)
    amount, vol = _amount_volume(close)
    mask = pd.DataFrame((np.abs(close.to_numpy() > np.nanpercentile(np.abs(close.to_numpy()), 95)).astype(float),
                        index=close.index, columns=close.columns)
    tampered = close.copy()
    tampered.iloc[-240:] *= 10.0
    tampered_amount = amount.copy()
    tampered_amount.iloc[-240:] *= 10.0

    def _check(name, a, b):
        base = a
        changed = b
        assert base.shape == changed.shape, name
        pd.testing.assert_series_equal(base.iloc[0], changed.iloc[0], check_dtype=False)

    op1 = OperatorRegistry.get("intra_event_window_reduce")
    _check("op1", op1.calculate(close, mask), op1.calculate(tampered, mask))
    op2 = OperatorRegistry.get("intra_event_pre_post_contrast")
    _check("op2", op2.calculate(close, mask), op2.calculate(tampered, mask))
    op3 = OperatorRegistry.get("intra_impulse_event_detector")
    _check("op3", op3.calculate(close), op3.calculate(tampered))
    op4 = OperatorRegistry.get("intra_post_impulse_response")
    _check("op4", op4.calculate(close, vol, amount), op4.calculate(tampered, vol, tampered_amount))
    op5 = OperatorRegistry.get("intra_probe_outcome_score")
    _check("op5", op5.calculate(close, vol, amount), op5.calculate(tampered, vol, tampered_amount))
    op6 = OperatorRegistry.get("intra_supply_absorption_score")
    _check("op6", op6.calculate(close, vol, amount), op6.calculate(tampered, vol, tampered_amount))
    op7 = OperatorRegistry.get("intra_consolidation_quality")
    _check("op7", op7.calculate(close, vol, amount), op7.calculate(tampered, vol, tampered_amount))
    op8 = OperatorRegistry.get("intra_response_curve_features")
    _check("op8", op8.calculate(close, vol), op8.calculate(tampered, vol))
    op9 = OperatorRegistry.get("intra_liquidity_resilience_curve_fit")
    _check("op9", op9.calculate(close, vol), op9.calculate(tampered, vol))


# ---------------------------------------------------------------------------
# All-NaN and constant degeneracy behaviour
# ---------------------------------------------------------------------------

def test_all_nan_returns_all_nan() -> None:
    close = _minute_panel(seed=0)
    nan_close = close * np.nan
    mask = pd.DataFrame(np.ones_like(close.to_numpy()), index=close.index, columns=close.columns)
    nan_amount, nan_vol = _amount_volume(nan_close)
    ops = {
        "intra_event_window_reduce": (nan_close, mask),
        "intra_event_pre_post_contrast": (nan_close, mask),
        "intra_impulse_event_detector": (nan_close,),
        "intra_post_impulse_response": (nan_close, nan_vol, nan_amount),
        "intra_probe_outcome_score": (nan_close, nan_vol, nan_amount),
        "intra_supply_absorption_score": (nan_close, nan_vol, nan_amount),
        "intra_consolidation_quality": (nan_close, nan_vol, nan_amount),
        "intra_response_curve_features": (nan_close, nan_vol),
        "intra_liquidity_resilience_curve_fit": (nan_close, nan_vol),
    }
    for name, args in ops.items():
        out = OperatorRegistry.get(name).calculate(*args)
        assert out.shape == (3, 2), name
        assert np.isnan(out.to_numpy()).all(), name


def test_constant_behaviour() -> None:
    close = _minute_panel(days=1, seed=0, cols=1)
    const = pd.DataFrame(np.ones_like(close.to_numpy()) * 10.0, index=close.index, columns=close.columns)
    mask = pd.DataFrame(np.zeros_like(close.to_numpy()), index=close.index, columns=close.columns)
    mask.iloc[5] = 1.0
    # op1: mean of a constant window is the constant
    out = OperatorRegistry.get("intra_event_window_reduce").calculate(const, mask, reducer="mean")
    assert np.isfinite(out.to_numpy()).all()
    assert out.iloc[0, 0] == pytest.approx(10.0)
    # op2: vol_ratio on constant pre window -> NaN (zero variance denominator)
    out = OperatorRegistry.get("intra_event_pre_post_contrast").calculate(const, mask, metric="vol_ratio")
    assert np.isnan(out.to_numpy()).all()
    # op3: constant price -> no impulse -> count 0, strength NaN
    out = OperatorRegistry.get("intra_impulse_event_detector").calculate(const, output="count")
    assert out.iloc[0, 0] == 0.0
    out = OperatorRegistry.get("intra_impulse_event_detector").calculate(const, output="strength")
    assert np.isnan(out.to_numpy()).all()
    # op4-9: no event / no shock on constant data -> NaN
    vol_const = const.copy()
    amount_const = const.copy()
    for name, args in {
        "intra_post_impulse_response": (const, vol_const, amount_const),
        "intra_probe_outcome_score": (const, vol_const, amount_const),
        "intra_supply_absorption_score": (const, vol_const, amount_const),
        "intra_consolidation_quality": (const, vol_const, amount_const),
        "intra_response_curve_features": (const, vol_const),
        "intra_liquidity_resilience_curve_fit": (const, vol_const),
    }.items():
        out = OperatorRegistry.get(name).calculate(*args)
        assert np.isnan(out.to_numpy()).all(), name


# ---------------------------------------------------------------------------
# Lunch gap is never bridged
# ---------------------------------------------------------------------------

def test_lunch_gap_not_bridged() -> None:
    minutes = [685, 686, 687, 688, 689, 690, 781, 782, 783, 784]
    xvals = np.array([1.0, 2, 3, 4, 5, 6, 1000.0, 1000, 1000, 1000])
    m = _event_mask(10, 5)  # event at the last morning bar (minute 690)
    x = _single_day({"A": xvals}, minutes=minutes)
    mask = _single_day({"A": m}, minutes=minutes)
    op = OperatorRegistry.get("intra_event_window_reduce")
    out = op.calculate(x, mask, pre=2, post=3, reducer="mean", event_select="first", min_obs=3)
    # window must be clipped to the morning segment [688,689,690] -> mean 5.0
    assert out["A"].iloc[0] == pytest.approx(5.0, rel=1e-9)
    # if lunch were bridged the mean would include the 1000-valued afternoon bars
    assert out["A"].iloc[0] < 100.0


def test_param_validation_raises() -> None:
    close = _minute_panel(seed=1)
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_event_window_reduce").calculate(close, close, reducer="bogus")
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_event_pre_post_contrast").calculate(close, close, metric="bogus")
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_impulse_event_detector").calculate(close, event="sideways")
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_post_impulse_response").calculate(close, close, close, output="bogus")
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_supply_absorption_score").calculate(close, close, close, event="sideways")
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_response_curve_features").calculate(close, close, curve="bogus")
    with pytest.raises(ValueError):
        OperatorRegistry.get("intra_liquidity_resilience_curve_fit").calculate(close, close, output="bogus")
