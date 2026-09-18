"""Focused grid and numerical contracts for Polars intraday beta operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators.intraday.polars_intraday_full import (
    _beta_asymmetry,
    _beta_stat,
)


INSTRUMENTS = ("C0", "C1", "C2")


def _minute_close(day: str = "2024-01-03") -> tuple[pl.DataFrame, np.ndarray]:
    returns = np.array(
        [
            [0.010, 0.015, 0.020], [-0.020, -0.010, -0.015],
            [0.012, -0.006, 0.018], [-0.008, 0.014, -0.004],
            [0.020, 0.010, 0.005], [-0.015, -0.025, -0.010],
            [0.006, -0.012, 0.009], [-0.011, 0.018, -0.007],
            [0.013, 0.007, 0.016], [-0.018, -0.009, -0.022],
            [0.009, -0.004, 0.011], [-0.006, 0.012, -0.003],
        ],
        dtype=float,
    )
    log_prices = np.vstack([np.zeros(3), np.cumsum(returns, axis=0)])
    timestamps = pd.date_range(f"{day} 09:30", periods=len(log_prices), freq="min")
    close = pl.DataFrame(
        {"QuoteTime": timestamps, **{name: np.exp(log_prices[:, i]) * 100 for i, name in enumerate(INSTRUMENTS)}}
    )
    return close, returns


def _cap(day: str, values=(1.0, 2.0, 3.0)) -> pl.DataFrame:
    return pl.DataFrame(
        {"QuoteTime": [pd.Timestamp(day)], **{name: [float(value)] for name, value in zip(INSTRUMENTS, values)}}
    )


def _assert_axis_and_all_nan(frame: pl.DataFrame, dates: list[pd.Timestamp]) -> None:
    assert frame.columns == ["date", *INSTRUMENTS]
    assert frame["date"].to_list() == [date.date() for date in dates]
    assert np.isnan(frame.select(INSTRUMENTS).to_numpy()).all()


def test_no_daily_membership_overlap_retains_minute_axis_with_nan_values():
    close, _ = _minute_close()
    stale_cap = _cap("2024-01-01")
    for kind in ("down_down", "down_up", "idio_kurt"):
        _assert_axis_and_all_nan(_beta_stat(close, stale_cap, kind), [pd.Timestamp("2024-01-03")])
    _assert_axis_and_all_nan(_beta_asymmetry(close, stale_cap), [pd.Timestamp("2024-01-03")])


def test_all_missing_daily_weights_and_empty_input_preserve_declared_columns():
    close, _ = _minute_close()
    missing_cap = _cap("2024-01-03", (np.nan, np.nan, np.nan))
    _assert_axis_and_all_nan(_beta_stat(close, missing_cap, "idio_kurt"), [pd.Timestamp("2024-01-03")])

    empty = close.head(0)
    result = _beta_stat(empty, missing_cap, "down_down")
    assert result.columns == ["date", *INSTRUMENTS]
    assert result.height == 0


def test_matching_daily_weights_match_independent_beta_oracles():
    close, returns = _minute_close()
    weights = np.array([1.0, 2.0, 3.0])
    market = returns @ weights / weights.sum()
    cap = _cap("2024-01-03", tuple(weights))

    def semibeta(asset: np.ndarray, asset_sign: int, market_sign: int) -> float:
        market_mask = market_sign * market > 0
        joint = market_mask & (asset_sign * asset > 0)
        return float(np.sum(asset[joint] * market[joint]) / np.sum(market[market_mask] ** 2))

    expected_dd = np.array([semibeta(returns[:, j], -1, -1) for j in range(3)])
    expected_du = np.array([semibeta(returns[:, j], -1, 1) for j in range(3)])
    expected_uu = np.array([semibeta(returns[:, j], 1, 1) for j in range(3)])

    actual_dd = _beta_stat(close, cap, "down_down").select(INSTRUMENTS).to_numpy()[0]
    actual_du = _beta_stat(close, cap, "down_up").select(INSTRUMENTS).to_numpy()[0]
    actual_asym = _beta_asymmetry(close, cap).select(INSTRUMENTS).to_numpy()[0]
    np.testing.assert_allclose(actual_dd, expected_dd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_du, expected_du, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_asym, expected_dd - expected_uu, rtol=1e-12, atol=1e-12)

    expected_kurtosis = []
    for asset in returns.T:
        design = np.column_stack([np.ones(market.size), market])
        residual = asset - design @ np.linalg.lstsq(design, asset, rcond=None)[0]
        centered = residual - residual.mean()
        expected_kurtosis.append(np.mean(centered**4) / np.mean(centered**2) ** 2)
    actual_kurtosis = _beta_stat(close, cap, "idio_kurt").select(INSTRUMENTS).to_numpy()[0]
    np.testing.assert_allclose(actual_kurtosis, expected_kurtosis, rtol=1e-11, atol=1e-11)
