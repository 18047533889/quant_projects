# -*- coding: utf-8 -*-
"""Tests for the R47 intraday slice-profile / volume-at-price operators.

Covers: intra_slice_mask_reduce, intra_slice_mask_pair_reduce,
intra_multiresolution_resample_reduce, intra_same_slot_zscore,
intra_session_boundary_jump, intra_volume_at_price_profile,
intra_volume_profile_peak_geometry, intra_volume_profile_supply_structure,
intra_volume_profile_value_area, intra_round_price_clustering_share,
intra_round_price_barrier_response.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.intraday.slice_profile  # noqa: F401  (registers operators)

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

CANONICALS = (
    "intra_slice_mask_reduce intra_slice_mask_pair_reduce "
    "intra_multiresolution_resample_reduce intra_same_slot_zscore "
    "intra_session_boundary_jump intra_volume_at_price_profile "
    "intra_volume_profile_peak_geometry intra_volume_profile_supply_structure "
    "intra_volume_profile_value_area intra_round_price_clustering_share "
    "intra_round_price_barrier_response"
).split()

VAP_NAMES = (
    "intra_volume_at_price_profile intra_volume_profile_peak_geometry "
    "intra_volume_profile_supply_structure intra_volume_profile_value_area"
).split()


def _minute_panel(days: int = 3, bars: int = 240, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        for m in list(range(571, 691))[: bars // 2] + list(range(781, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.01, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps, tz="Asia/Shanghai"),
        columns=[f"C{i}" for i in range(cols)],
    )
    return close


def _amount_volume(close: pd.DataFrame, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vol = pd.DataFrame(
        np.abs(rng.standard_normal(close.shape)) + 1.0, index=close.index, columns=close.columns
    )
    amount = vol * close
    return amount, vol


def _panel(dates, minutes, per_day, cols=None):
    """Explicit single- (or multi-) column minute panel.

    dates: list of date strings; minutes: list of minute-of-day; per_day maps a
    date string to a list of values aligned with ``minutes``.
    """
    ts = []
    vals = []
    for d in dates:
        day = pd.Timestamp(d)
        v = per_day[d]
        assert len(v) == len(minutes), (d, len(v), len(minutes))
        for m, val in zip(minutes, v):
            ts.append(day + pd.Timedelta(minutes=m))
            vals.append(float(val))
    idx = pd.DatetimeIndex(ts, tz="Asia/Shanghai")
    names = cols or ["A"]
    data = {c: list(vals) for c in names}
    return pd.DataFrame(data, index=idx)


def _all_nan_panel(dates, minutes, cols=2):
    ts = []
    for d in dates:
        day = pd.Timestamp(d)
        for m in minutes:
            ts.append(day + pd.Timedelta(minutes=m))
    idx = pd.DatetimeIndex(ts, tz="Asia/Shanghai")
    return pd.DataFrame(np.nan, index=idx, columns=[f"C{i}" for i in range(cols)])


def _build_args(name, close, amount, vol):
    if name == "intra_slice_mask_reduce":
        return (close, vol), {}
    if name == "intra_slice_mask_pair_reduce":
        return (close, vol, amount), {}
    if name in ("intra_multiresolution_resample_reduce", "intra_same_slot_zscore",
                "intra_round_price_clustering_share", "intra_round_price_barrier_response"):
        return (close,), {}
    if name == "intra_session_boundary_jump":
        return (close, vol), {}
    if name in VAP_NAMES:
        return (close, vol), {}
    return (close,), {}


# ---------------------------------------------------------------------------
# Registration + surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(CANONICALS)))
def test_registered_and_classified(name: str) -> None:
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


# ---------------------------------------------------------------------------
# Shape + determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(CANONICALS)))
def test_shape_and_determinism(name: str) -> None:
    close = _minute_panel(seed=1)
    amount, vol = _amount_volume(close)
    args, kwargs = _build_args(name, close, amount, vol)
    op = OperatorRegistry.get(name)
    first = op.calculate(*args, **kwargs)
    second = op.calculate(*args, **kwargs)
    assert first.shape == (3, 2), name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# Future-poison (PIT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(CANONICALS)))
def test_future_poison(name: str) -> None:
    close = _minute_panel(days=4, seed=6, cols=2)
    amount, vol = _amount_volume(close)
    args, kwargs = _build_args(name, close, amount, vol)
    op = OperatorRegistry.get(name)
    base = op.calculate(*args, **kwargs)
    tampered = close.copy()
    tampered.iloc[-240:] *= 10.0
    targs, tkwargs = _build_args(name, tampered, amount, vol)
    changed = op.calculate(*targs, **tkwargs)
    assert base.shape == changed.shape, name
    pd.testing.assert_frame_equal(base.iloc[:-1], changed.iloc[:-1], check_dtype=False)


# ---------------------------------------------------------------------------
# All-NaN => all-NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(CANONICALS)))
def test_all_nan_yields_all_nan(name: str) -> None:
    close = _all_nan_panel(["2024-01-03", "2024-01-04"], list(range(571, 591)) + list(range(781, 801)))
    amount, vol = _amount_volume(close)
    args, kwargs = _build_args(name, close, amount, vol)
    op = OperatorRegistry.get(name)
    out = op.calculate(*args, **kwargs)
    assert out.shape == (2, 2), name
    assert np.isnan(out.to_numpy()).all(), name


# ---------------------------------------------------------------------------
# 1. intra_slice_mask_reduce
# ---------------------------------------------------------------------------

def test_slice_mask_reduce_golden() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_reduce")
    out = op.calculate(x, mask, mask_side="high", mask_q=0.5, reducer="mean", min_bars=2)
    assert out["A"].iloc[0] == pytest.approx(35.0)
    out = op.calculate(x, mask, mask_side="low", mask_q=0.5, reducer="mean", min_bars=2)
    assert out["A"].iloc[0] == pytest.approx(15.0)
    out = op.calculate(x, mask, mask_side="high", mask_q=0.5, reducer="slope", min_bars=2)
    assert out["A"].iloc[0] == pytest.approx(10.0)
    out = op.calculate(x, mask, mask_side="high", mask_q=0.5, reducer="positive_share", min_bars=2)
    assert out["A"].iloc[0] == pytest.approx(1.0)


def test_slice_mask_reduce_slice_window() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_reduce")
    # slice=0.25 -> window [0.0, 0.5] -> only the two morning bars
    out = op.calculate(x, mask, slice=0.25, mask_side="high", mask_q=0.5, reducer="mean", min_bars=2)
    # ranks of [1,2] = [0,1]; keep rank>=0.5 -> x=[20]; mean=20 (sum/mean allows 1)
    assert out["A"].iloc[0] == pytest.approx(20.0)


def test_slice_mask_reduce_constant_variance_nan() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [5.0, 5.0, 5.0, 5.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_reduce")
    out = op.calculate(x, mask, mask_side="high", mask_q=0.5, reducer="std", min_bars=2)
    assert np.isnan(out["A"].iloc[0])


def test_slice_mask_reduce_param_validation() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_reduce")
    with pytest.raises(ValueError):
        op.calculate(x, mask, mask_side="sideways", mask_q=0.5)
    with pytest.raises(ValueError):
        op.calculate(x, mask, mask_side="high", mask_q=1.5)


# ---------------------------------------------------------------------------
# 2. intra_slice_mask_pair_reduce
# ---------------------------------------------------------------------------

def test_slice_mask_pair_reduce_golden() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 573]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0]})
    y = _panel(dates, minutes, {"2024-01-03": [5.0, 6.0, 7.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0]})
    op = OperatorRegistry.get("intra_slice_mask_pair_reduce")
    out = op.calculate(x, y, mask, mask_side="high", mask_q=0.5, y_lag=1,
                       reducer="euclidean", min_pairs=2)
    expected = np.sqrt(((20.0 - 5.0) ** 2 + (30.0 - 6.0) ** 2) / 2.0)
    assert out["A"].iloc[0] == pytest.approx(expected)

    minutes2 = [571, 572, 781, 782]
    x2 = _panel(dates, minutes2, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    y2 = _panel(dates, minutes2, {"2024-01-03": [5.0, 6.0, 7.0, 8.0]})
    mask2 = _panel(dates, minutes2, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    out = op.calculate(x2, y2, mask2, mask_side="high", mask_q=0.5, y_lag=0,
                       reducer="corr", min_pairs=2)
    assert out["A"].iloc[0] == pytest.approx(1.0)


def test_slice_mask_pair_reduce_lunch_not_bridged() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    y = _panel(dates, minutes, {"2024-01-03": [5.0, 6.0, 7.0, 8.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_pair_reduce")
    # y_lag=1: retained x bars are afternoon idx 2,3; x=30 pairs y[1] (morning) -> invalid
    # (lunch gap), x=40 pairs y[2] (afternoon) -> valid.  Only 1 pair -> corr NaN.
    out = op.calculate(x, y, mask, mask_side="high", mask_q=0.5, y_lag=1,
                       reducer="corr", min_pairs=2)
    assert np.isnan(out["A"].iloc[0])


def test_slice_mask_pair_reduce_constant_variance_nan() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    y = _panel(dates, minutes, {"2024-01-03": [5.0, 5.0, 5.0, 5.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_pair_reduce")
    out = op.calculate(x, y, mask, mask_side="high", mask_q=0.5, y_lag=0,
                       reducer="corr", min_pairs=2)
    assert np.isnan(out["A"].iloc[0])


def test_slice_mask_pair_reduce_negative_lag_raises() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 781, 782]
    x = _panel(dates, minutes, {"2024-01-03": [10.0, 20.0, 30.0, 40.0]})
    y = _panel(dates, minutes, {"2024-01-03": [5.0, 6.0, 7.0, 8.0]})
    mask = _panel(dates, minutes, {"2024-01-03": [1.0, 2.0, 3.0, 4.0]})
    op = OperatorRegistry.get("intra_slice_mask_pair_reduce")
    with pytest.raises(ValueError):
        op.calculate(x, y, mask, y_lag=-1, min_pairs=2)


# ---------------------------------------------------------------------------
# 3. intra_multiresolution_resample_reduce
# ---------------------------------------------------------------------------

def test_multiresolution_resample_reduce_golden() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 572, 781, 782]
    values = {
        "2024-01-03": [10.0, 20.0, 30.0, 40.0],
        "2024-01-04": [20.0, 30.0, 40.0, 50.0],
        "2024-01-05": [30.0, 40.0, 50.0, 60.0],
    }
    x = _panel(dates, minutes, values)
    op = OperatorRegistry.get("intra_multiresolution_resample_reduce")
    # 4 bars/day with 60-min buckets -> 2 of 4 expected buckets non-empty (coverage 0.5)
    out = op.calculate(x, bar_minutes=60, reducer="mean", min_coverage=0.5)
    assert np.isnan(out["A"].iloc[0])
    assert np.isnan(out["A"].iloc[1])
    assert out["A"].iloc[2] == pytest.approx(30.0)


def test_multiresolution_resample_lunch_not_bridged() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 781]
    values = {
        "2024-01-03": [100.0, 200.0],
        "2024-01-04": [300.0, 400.0],
        "2024-01-05": [500.0, 600.0],
    }
    x = _panel(dates, minutes, values)
    op = OperatorRegistry.get("intra_multiresolution_resample_reduce")
    # bar_minutes=240 -> morning and afternoon each form their OWN bar (never a
    # single 240-min bar bridging lunch).  coverage = 2/2 = 1.0, so min_coverage
    # 0.9 accepts the day.  Day summaries are last_minus_first of the two bars.
    out = op.calculate(x, bar_minutes=240, reducer="last_minus_first", min_coverage=0.9)
    assert out["A"].iloc[2] == pytest.approx(100.0)
    # If lunch were bridged into one bucket, coverage would be 1/2 = 0.5 < 0.9 -> NaN.
    out = op.calculate(x, bar_minutes=240, reducer="mean", min_coverage=0.9)
    assert np.isfinite(out["A"].iloc[2])


def test_multiresolution_resample_reducer_last_minus_first() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 572, 781, 782]
    values = {
        "2024-01-03": [10.0, 20.0, 30.0, 40.0],
        "2024-01-04": [20.0, 30.0, 40.0, 50.0],
        "2024-01-05": [30.0, 40.0, 50.0, 60.0],
    }
    x = _panel(dates, minutes, values)
    op = OperatorRegistry.get("intra_multiresolution_resample_reduce")
    out = op.calculate(x, bar_minutes=60, reducer="last_minus_first", min_coverage=0.5)
    # day summaries: day1 = 35-15=20, day2 = 45-25=20, day3 = 55-35=20
    assert out["A"].iloc[2] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 4. intra_same_slot_zscore
# ---------------------------------------------------------------------------

def test_same_slot_zscore_golden() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 781]
    values = {
        "2024-01-03": [10.0, 20.0],
        "2024-01-04": [12.0, 22.0],
        "2024-01-05": [14.0, 24.0],
    }
    x = _panel(dates, minutes, values)
    op = OperatorRegistry.get("intra_same_slot_zscore")
    out = op.calculate(x, history_days=20, ddof=1, min_history=2)
    assert np.isnan(out["A"].iloc[0])
    assert np.isnan(out["A"].iloc[1])
    expected = 3.0 / np.sqrt(2.0)
    assert out["A"].iloc[2] == pytest.approx(expected)


def test_same_slot_zscore_constant_prior_nan() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 781]
    values = {
        "2024-01-03": [10.0, 20.0],
        "2024-01-04": [10.0, 20.0],
        "2024-01-05": [10.0, 20.0],
    }
    x = _panel(dates, minutes, values)
    op = OperatorRegistry.get("intra_same_slot_zscore")
    out = op.calculate(x, history_days=20, ddof=1, min_history=2)
    # prior std = 0 for day3 -> z NaN
    assert np.isnan(out["A"].iloc[2])


def test_same_slot_zscore_bad_ddof_raises() -> None:
    dates = ["2024-01-03"]
    minutes = [571]
    x = _panel(dates, minutes, {"2024-01-03": [10.0]})
    op = OperatorRegistry.get("intra_same_slot_zscore")
    with pytest.raises(ValueError):
        op.calculate(x, ddof=2)


# ---------------------------------------------------------------------------
# 5. intra_session_boundary_jump
# ---------------------------------------------------------------------------

def test_session_boundary_jump_golden() -> None:
    dates = ["2024-01-03", "2024-01-04"]
    minutes = [571, 572, 573, 781, 782, 783]
    price = _panel(dates, minutes, {
        "2024-01-03": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "2024-01-04": [104.0, 104.5, 105.0, 106.0, 106.5, 107.0],
    })
    vol = _panel(dates, minutes, {
        "2024-01-03": [1.0] * 6,
        "2024-01-04": [1.0] * 6,
    })
    op = OperatorRegistry.get("intra_session_boundary_jump")
    out = op.calculate(price, boundary="lunch_restart", output="gap")
    assert out["A"].iloc[0] == pytest.approx(103.0 / 102.0 - 1.0)
    assert out["A"].iloc[1] == pytest.approx(106.0 / 105.0 - 1.0)

    pre_close = pd.DataFrame(
        {"A": [100.0, 103.0]}, index=pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    )
    out = op.calculate(price, pre_close=pre_close, boundary="open", output="gap")
    assert out["A"].iloc[0] == pytest.approx(100.0 / 100.0 - 1.0)
    assert out["A"].iloc[1] == pytest.approx(104.0 / 103.0 - 1.0)

    out = op.calculate(price, boundary="close", pre_bars=2, output="gap")
    assert out["A"].iloc[0] == pytest.approx(105.0 / 104.0 - 1.0)
    assert out["A"].iloc[1] == pytest.approx(107.0 / 106.5 - 1.0)

    out = op.calculate(price, boundary="lunch_restart", pre_bars=2, output="normalized_gap")
    gap2 = 106.0 / 105.0 - 1.0
    assert out["A"].iloc[1] == pytest.approx(gap2 / np.sqrt(0.125))

    out = op.calculate(price, vol, boundary="lunch_restart", pre_bars=2, post_bars=2,
                       output="volume_jump")
    assert out["A"].iloc[1] == pytest.approx(1.0)

    out = op.calculate(price, boundary="lunch_restart", pre_bars=2, post_bars=2,
                       output="recovery")
    assert out["A"].iloc[1] == pytest.approx(106.5 / 105.0 - 1.0 - (106.0 / 105.0 - 1.0))


def test_session_boundary_jump_constant_normalized_nan() -> None:
    dates = ["2024-01-03", "2024-01-04"]
    minutes = [571, 572, 573, 781, 782, 783]
    price = _panel(dates, minutes, {
        "2024-01-03": [100.0] * 6,
        "2024-01-04": [100.0] * 6,
    })
    op = OperatorRegistry.get("intra_session_boundary_jump")
    out = op.calculate(price, boundary="lunch_restart", pre_bars=2, output="normalized_gap")
    assert np.isnan(out["A"].iloc[1])
    # plain gap on constant data is well defined (0)
    out = op.calculate(price, boundary="lunch_restart", output="gap")
    assert out["A"].iloc[1] == pytest.approx(0.0)


def test_session_boundary_jump_bad_boundary_raises() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572]
    price = _panel(dates, minutes, {"2024-01-03": [100.0, 101.0]})
    op = OperatorRegistry.get("intra_session_boundary_jump")
    with pytest.raises(ValueError):
        op.calculate(price, boundary="midnight")


# ---------------------------------------------------------------------------
# 6. intra_volume_at_price_profile
# ---------------------------------------------------------------------------

def test_volume_at_price_profile_golden() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 579))
    prices = [100.0, 100.0, 102.0, 104.0, 100.0, 100.0, 102.0, 104.0]
    price = _panel(dates, minutes, {"2024-01-03": prices})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 8})
    op = OperatorRegistry.get("intra_volume_at_price_profile")
    out = op.calculate(price, vol, bins=8, output="entropy")
    assert out["A"].iloc[0] == pytest.approx(0.5)
    out = op.calculate(price, vol, bins=8, output="poc_price")
    assert out["A"].iloc[0] == pytest.approx(0.5 / 8.0)
    out = op.calculate(price, vol, bins=8, output="concentration")
    assert out["A"].iloc[0] == pytest.approx(0.25 + 0.0625 + 0.0625)


def test_volume_at_price_profile_constant_nan() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 579))
    price = _panel(dates, minutes, {"2024-01-03": [100.0] * 8})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 8})
    op = OperatorRegistry.get("intra_volume_at_price_profile")
    out = op.calculate(price, vol, bins=8, output="entropy")
    assert np.isnan(out["A"].iloc[0])


def test_volume_at_price_profile_zero_volume_nan() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 579))
    price = _panel(dates, minutes, {"2024-01-03": [100.0, 101.0, 102.0, 103.0,
                                                   100.0, 101.0, 102.0, 103.0]})
    vol = _panel(dates, minutes, {"2024-01-03": [0.0] * 8})
    op = OperatorRegistry.get("intra_volume_at_price_profile")
    out = op.calculate(price, vol, bins=8, output="entropy")
    assert np.isnan(out["A"].iloc[0])


# ---------------------------------------------------------------------------
# 7. intra_volume_profile_peak_geometry
# ---------------------------------------------------------------------------

def test_volume_profile_peak_geometry_golden() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 581))
    prices = [100.0, 100.0, 100.0, 100.0, 100.1, 100.5, 100.6, 100.6, 100.6, 100.6]
    price = _panel(dates, minutes, {"2024-01-03": prices})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 10})
    op = OperatorRegistry.get("intra_volume_profile_peak_geometry")
    out = op.calculate(price, vol, bins=8, smooth=0, output="peak_count")
    assert out["A"].iloc[0] == pytest.approx(2.0)
    out = op.calculate(price, vol, bins=8, smooth=0, output="top_peak_mass")
    assert out["A"].iloc[0] == pytest.approx(0.4)
    out = op.calculate(price, vol, bins=8, smooth=0, output="second_peak_mass")
    assert out["A"].iloc[0] == pytest.approx(0.4)
    out = op.calculate(price, vol, bins=8, smooth=0, output="peak_ratio")
    assert out["A"].iloc[0] == pytest.approx(1.0)
    out = op.calculate(price, vol, bins=8, smooth=0, output="peak_distance")
    assert out["A"].iloc[0] == pytest.approx(0.875)
    out = op.calculate(price, vol, bins=8, smooth=0, output="top_peak_width")
    assert out["A"].iloc[0] == pytest.approx(2.0)
    out = op.calculate(price, vol, bins=8, smooth=0, output="nearest_peak_distance")
    assert out["A"].iloc[0] == pytest.approx(0.875)
    out = op.calculate(price, vol, bins=8, smooth=0, output="valley_depth")
    assert out["A"].iloc[0] == pytest.approx(1.0)


def test_volume_profile_peak_geometry_constant_nan() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 581))
    price = _panel(dates, minutes, {"2024-01-03": [100.0] * 10})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 10})
    op = OperatorRegistry.get("intra_volume_profile_peak_geometry")
    out = op.calculate(price, vol, bins=8, output="peak_count")
    assert np.isnan(out["A"].iloc[0])


# ---------------------------------------------------------------------------
# 8. intra_volume_profile_supply_structure
# ---------------------------------------------------------------------------

def test_volume_profile_supply_structure_golden() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 581))
    prices = [100.0, 110.0, 100.0, 110.0, 106.0, 100.0, 110.0, 100.0, 110.0, 106.0]
    price = _panel(dates, minutes, {"2024-01-03": prices})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 10})
    op = OperatorRegistry.get("intra_volume_profile_supply_structure")
    out = op.calculate(price, vol, bins=8, output="overhead_mass")
    assert out["A"].iloc[0] == pytest.approx(0.4)
    out = op.calculate(price, vol, bins=8, output="under_price_mass")
    assert out["A"].iloc[0] == pytest.approx(0.6)
    # nearest upper peak: bin 7 center = 100 + 7.5*1.25 = 109.375, current = 106
    out = op.calculate(price, vol, bins=8, output="nearest_upper_peak")
    assert out["A"].iloc[0] == pytest.approx((109.375 - 106.0) / 106.0)


def test_volume_profile_supply_structure_constant_nan() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 581))
    price = _panel(dates, minutes, {"2024-01-03": [100.0] * 10})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 10})
    op = OperatorRegistry.get("intra_volume_profile_supply_structure")
    out = op.calculate(price, vol, bins=8, output="overhead_mass")
    assert np.isnan(out["A"].iloc[0])


# ---------------------------------------------------------------------------
# 9. intra_volume_profile_value_area
# ---------------------------------------------------------------------------

def test_volume_profile_value_area_golden() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 580))
    prices = [100.0, 100.0, 100.0, 100.0, 102.0, 102.0, 102.0, 104.0, 104.0]
    price = _panel(dates, minutes, {"2024-01-03": prices})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 9})
    op = OperatorRegistry.get("intra_volume_profile_value_area")
    out = op.calculate(price, vol, bins=8, target_mass=0.7, output="value_area_width")
    assert out["A"].iloc[0] == pytest.approx(0.625)
    out = op.calculate(price, vol, bins=8, target_mass=0.7, output="poc_price")
    assert out["A"].iloc[0] == pytest.approx(0.0625)
    out = op.calculate(price, vol, bins=8, target_mass=0.7, output="value_area_high")
    assert out["A"].iloc[0] == pytest.approx(0.5625)
    out = op.calculate(price, vol, bins=8, target_mass=0.7, output="value_area_low")
    assert out["A"].iloc[0] == pytest.approx(0.0625)
    out = op.calculate(price, vol, bins=8, target_mass=0.7, output="value_area_mid")
    assert out["A"].iloc[0] == pytest.approx(0.3125)
    out = op.calculate(price, vol, bins=8, target_mass=0.7, output="in_value_area")
    assert out["A"].iloc[0] == pytest.approx(7.0 / 9.0)


def test_volume_profile_value_area_constant_nan() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 580))
    price = _panel(dates, minutes, {"2024-01-03": [100.0] * 9})
    vol = _panel(dates, minutes, {"2024-01-03": [1.0] * 9})
    op = OperatorRegistry.get("intra_volume_profile_value_area")
    out = op.calculate(price, vol, bins=8, output="value_area_width")
    assert np.isnan(out["A"].iloc[0])


# ---------------------------------------------------------------------------
# 10. intra_round_price_clustering_share
# ---------------------------------------------------------------------------

def test_round_price_clustering_share_golden() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572, 573, 574, 575]
    price = _panel(dates, minutes, {"2024-01-03": [10.0, 10.01, 10.5, 10.0, 10.5]})
    op = OperatorRegistry.get("intra_round_price_clustering_share")
    out = op.calculate(price, lattice=1.0, tolerance_ticks=0.25, output="share", min_bars=3)
    assert out["A"].iloc[0] == pytest.approx(0.4)
    out = op.calculate(price, lattice=1.0, tolerance_ticks=0.25, output="excess_share", min_bars=3)
    assert out["A"].iloc[0] == pytest.approx(0.4 - 2.0 * 0.25 * 0.01 / 0.5)
    out = op.calculate(price, lattice=1.0, tolerance_ticks=0.25, output="run_length", min_bars=3)
    assert out["A"].iloc[0] == pytest.approx(1.0)


def test_round_price_clustering_share_constant_nan() -> None:
    dates = ["2024-01-03"]
    minutes = list(range(571, 580))
    price = _panel(dates, minutes, {"2024-01-03": [10.0] * 9})
    op = OperatorRegistry.get("intra_round_price_clustering_share")
    out = op.calculate(price, lattice=1.0, output="share", min_bars=3)
    assert np.isnan(out["A"].iloc[0])


def test_round_price_clustering_share_too_few_bars_nan() -> None:
    dates = ["2024-01-03"]
    minutes = [571, 572]
    price = _panel(dates, minutes, {"2024-01-03": [10.0, 10.01]})
    op = OperatorRegistry.get("intra_round_price_clustering_share")
    out = op.calculate(price, lattice=1.0, output="share", min_bars=3)
    assert np.isnan(out["A"].iloc[0])


# ---------------------------------------------------------------------------
# 11. intra_round_price_barrier_response
# ---------------------------------------------------------------------------

def test_round_price_barrier_response_golden() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 572, 573, 574]
    prices = {
        "2024-01-03": [99.99, 99.992, 100.01, 100.0],
        "2024-01-04": [99.99, 99.992, 99.99, 100.0],
        "2024-01-05": [100.0, 100.0, 100.0, 100.0],
    }
    price = _panel(dates, minutes, prices)
    op = OperatorRegistry.get("intra_round_price_barrier_response")
    out = op.calculate(price, lattice=1.0, tolerance_ticks=1.0, output="cross_rate", min_events=2)
    assert np.isnan(out["A"].iloc[0])
    assert np.isnan(out["A"].iloc[1])
    assert out["A"].iloc[2] == pytest.approx(1.0 / 3.0)
    out = op.calculate(price, lattice=1.0, tolerance_ticks=1.0, output="bounce_rate", min_events=2)
    assert out["A"].iloc[2] == pytest.approx(2.0 / 3.0)


def test_round_price_barrier_response_min_events_nan() -> None:
    dates = ["2024-01-03", "2024-01-04", "2024-01-05"]
    minutes = [571, 572, 573, 574]
    prices = {
        "2024-01-03": [99.99, 99.992, 100.01, 100.0],
        "2024-01-04": [99.99, 99.992, 99.99, 100.0],
        "2024-01-05": [100.0, 100.0, 100.0, 100.0],
    }
    price = _panel(dates, minutes, prices)
    op = OperatorRegistry.get("intra_round_price_barrier_response")
    out = op.calculate(price, lattice=1.0, tolerance_ticks=1.0, output="cross_rate", min_events=4)
    # only 3 events over the lookback -> NaN
    assert np.isnan(out["A"].iloc[2])


def test_round_price_barrier_response_constant_nan() -> None:
    dates = ["2024-01-03", "2024-01-04"]
    minutes = [571, 572, 573, 574]
    price = _panel(dates, minutes, {
        "2024-01-03": [100.0, 100.0, 100.0, 100.0],
        "2024-01-04": [100.0, 100.0, 100.0, 100.0],
    })
    op = OperatorRegistry.get("intra_round_price_barrier_response")
    out = op.calculate(price, lattice=1.0, output="cross_rate", min_events=2)
    assert np.isnan(out.to_numpy()).all()
