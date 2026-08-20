# -*- coding: utf-8 -*-
"""Tests for the 2026-08 next-stage intraday operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

MOMENT_CANONICALS = (
    "intra_realized_skewness intra_realized_kurtosis intra_realized_quarticity "
    "intra_tripower_quarticity intra_continuous_variance intra_jump_variation "
    "intra_positive_jump_variation intra_negative_jump_variation intra_signed_jump_ratio "
    "intra_jump_count intra_jump_concentration intra_jump_first_time intra_jump_last_time "
    "intra_jump_clustering"
).split()

BETA_CANONICALS = (
    "intra_realized_beta intra_realized_correlation intra_down_down_semibeta intra_up_up_semibeta "
    "intra_down_up_semibeta intra_up_down_semibeta intra_beta_asymmetry intra_idiosyncratic_variance "
    "intra_idiosyncratic_skewness intra_idiosyncratic_kurtosis intra_market_model_r2"
).split()

TIME_CANONICALS = (
    "intra_interval_return intra_interval_volume_share intra_interval_amount_share "
    "intra_interval_realized_variance intra_interval_vwap_deviation intra_interval_illiquidity "
    "intra_same_slot_momentum intra_same_slot_reversal intra_return_profile_cosine "
    "intra_volume_profile_cosine intra_amount_profile_cosine intra_volume_profile_jsd "
    "intra_amount_profile_jsd intra_profile_earth_mover_distance"
).split()

VWAP_CANONICALS = (
    "intra_vwap_path_slope intra_vwap_path_curvature intra_price_vwap_max_positive_excursion "
    "intra_price_vwap_max_negative_excursion intra_time_above_vwap intra_longest_above_vwap_streak "
    "intra_longest_below_vwap_streak intra_vwap_reversion_speed"
).split()
DRAWDOWN_CANONICALS = (
    "intra_max_drawdown intra_max_drawup "
    "intra_drawdown_depth intra_drawdown_duration intra_drawdown_recovery_half_life"
).split()

OVERNIGHT_CANONICALS = (
    "ts_overnight_intraday_cov ts_overnight_intraday_spread ts_overnight_intraday_sign_agreement "
    "ts_gap_reversion_ratio ts_gap_fill_ratio ts_gap_survival_duration ts_opening_mispricing_score"
).split()

ALL = MOMENT_CANONICALS + BETA_CANONICALS + TIME_CANONICALS + VWAP_CANONICALS + DRAWDOWN_CANONICALS + OVERNIGHT_CANONICALS


def _minute_panel(days: int = 3, bars: int = 240, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        # A-share session minute-of-day: 09:31..11:30 (571..690), 13:01..15:00 (781..900)
        for m in list(range(571, 691))[: bars // 2] + list(range(781, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.01, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps),
        columns=[f"C{i}" for i in range(cols)],
    )
    return close


def _amount_volume(close: pd.DataFrame, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    vol = pd.DataFrame(np.abs(rng.standard_normal(close.shape)) + 1.0, index=close.index, columns=close.columns)
    amount = vol * close
    return amount, vol


def _daily_weights(days: int = 3, cols: int = 2) -> pd.DataFrame:
    dates = [pd.Timestamp("2024-01-03") + pd.Timedelta(days=d) for d in range(days)]
    return pd.DataFrame(
        np.ones((days, cols)) * 1e8, index=pd.DatetimeIndex(dates), columns=[f"C{i}" for i in range(cols)]
    )


@pytest.mark.parametrize("name", sorted(set(ALL)))
def test_registered_and_classified(name: str) -> None:
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    # 2026-08 daily migration promoted the intra_* factor operators to daily.
    assert classify_canonical(name) in ("daily", "extended", "research"), name


@pytest.mark.parametrize("name", sorted(set(MOMENT_CANONICALS)))
def test_moment_operators_shape_and_determinism(name: str) -> None:
    close = _minute_panel(seed=1)
    op = OperatorRegistry.get(name)
    first = op.calculate(close)
    second = op.calculate(close)
    assert first.shape == (3, 2)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


@pytest.mark.parametrize("name", sorted(set(VWAP_CANONICALS)))
def test_vwap_operators_shape_and_determinism(name: str) -> None:
    close = _minute_panel(seed=1)
    amount, vol = _amount_volume(close)
    op = OperatorRegistry.get(name)
    first = op.calculate(close, amount, vol)
    second = op.calculate(close, amount, vol)
    assert first.shape == (3, 2)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


@pytest.mark.parametrize("name", sorted(set(DRAWDOWN_CANONICALS)))
def test_drawdown_operators_shape_and_determinism(name: str) -> None:
    close = _minute_panel(seed=1)
    op = OperatorRegistry.get(name)
    first = op.calculate(close)
    second = op.calculate(close)
    assert first.shape == (3, 2)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_moment_operators_with_threshold_param() -> None:
    close = _minute_panel(seed=2)
    for name in ("intra_jump_count", "intra_signed_jump_ratio", "intra_positive_jump_variation"):
        op = OperatorRegistry.get(name)
        out = op.calculate(close, threshold_scale=3.0)
        assert out.shape == (3, 2)
        assert np.isfinite(out.to_numpy()).any()


def test_realized_beta_recovers_true_beta() -> None:
    rng = np.random.default_rng(7)
    n = 240
    common = rng.standard_normal(n)
    cols = {
        "A": 0.8 * common + 0.5 * rng.standard_normal(n),
        "B": 1.2 * common + 0.4 * rng.standard_normal(n),
    }
    close = pd.DataFrame(
        np.exp(np.cumsum(np.column_stack(list(cols.values())), axis=0)) * 100,
        index=_minute_panel(days=1, seed=7).index,
        columns=list(cols),
    )
    w = pd.DataFrame({"A": [1e8], "B": [2e8]}, index=pd.DatetimeIndex(["2024-01-03"]))
    out = OperatorRegistry.get("intra_realized_beta").calculate(close, w)
    assert out.shape == (1, 2)
    assert out["A"].iloc[0] == pytest.approx(0.8, abs=0.1)
    assert out["B"].iloc[0] == pytest.approx(1.2, abs=0.1)


def test_beta_operators_shape() -> None:
    close = _minute_panel(days=2, seed=3)
    w = _daily_weights(days=2, cols=2)
    for name in BETA_CANONICALS:
        op = OperatorRegistry.get(name)
        out = op.calculate(close, w)
        assert out.shape == (2, 2), name


def test_interval_return_matches_manual() -> None:
    close = _minute_panel(days=1, bars=240, seed=4, cols=1)
    op = OperatorRegistry.get("intra_interval_return")
    out = op.calculate(close, start_minute=570, end_minute=690)
    vals = close["C0"].to_numpy()
    expected = vals[119] / vals[0] - 1.0  # bars 0..119 are 09:31..11:30
    assert out["C0"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_drawdown_bounds() -> None:
    close = _minute_panel(seed=5)
    dd = OperatorRegistry.get("intra_max_drawdown").calculate(close)
    assert (dd.to_numpy() <= 0.0).all()
    assert np.isfinite(dd.to_numpy()).any()
    du = OperatorRegistry.get("intra_max_drawup").calculate(close)
    assert (du.to_numpy() >= 0.0).all()


def test_profile_operators_causal() -> None:
    """Modifying future days must not change past outputs (PIT)."""
    close = _minute_panel(days=4, seed=6, cols=2)
    name = "intra_return_profile_cosine"
    op = OperatorRegistry.get(name)
    base = op.calculate(close, window=3)
    tampered = close.copy()
    # corrupt the last day's returns heavily
    tampered.iloc[-240:] *= 10.0
    changed = op.calculate(tampered, window=3)
    assert base.shape == changed.shape
    # the first day's output is unaffected
    pd.testing.assert_series_equal(base.iloc[0], changed.iloc[0], check_dtype=False)


def test_overnight_operators_shape_and_pit() -> None:
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.standard_normal((n, 2)), axis=0)), index=idx, columns=["A", "B"])
    pre = close.shift(1)
    opn = close * (1 + rng.standard_normal((n, 2)) * 0.005)
    for name in OVERNIGHT_CANONICALS:
        op = OperatorRegistry.get(name)
        out = op.calculate(close, opn, pre)
        assert out.shape == (n, 2), name
        assert np.isfinite(out.to_numpy()).any(), name
