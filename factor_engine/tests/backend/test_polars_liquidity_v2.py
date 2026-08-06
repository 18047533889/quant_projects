# -*- coding: utf-8 -*-
"""Parity tests for native Polars liquidity / price-volume operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    load_all()


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(424242)
    index = pd.date_range("2022-01-01", periods=90, freq="D")
    close = pd.DataFrame(
        {
            "A": 80.0 + np.cumsum(rng.normal(0.1, 1.2, len(index))),
            "B": 45.0 + np.cumsum(rng.normal(-0.02, 0.7, len(index))),
        },
        index=index,
    )
    spread = pd.DataFrame(rng.uniform(0.5, 2.5, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    volume = pd.DataFrame(rng.integers(5_000, 500_000, close.shape), index=index, columns=close.columns).astype(float)
    ret = close.pct_change()
    turnover = volume / 1e6
    dollar_volume = (close * volume).abs()
    shares = pd.DataFrame(rng.uniform(1e8, 1e10, close.shape), index=index, columns=close.columns)
    return high, low, close, volume, ret, turnover, dollar_volume, shares


def _polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def _assert_parity(name, args, kwargs, rtol=1e-8, atol=1e-8):
    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")
    assert pandas_op is not None, f"{name} missing pandas"
    assert polars_op is not None, f"{name} missing polars"
    pandas_out = pandas_op.calculate(*args, **kwargs)
    polars_out = polars_op.calculate(*[_polars(arg) for arg in args], **kwargs)
    assert list(pandas_out.columns) == list(polars_out.columns)
    for column in pandas_out.columns:
        np.testing.assert_allclose(
            pandas_out[column].to_numpy(),
            polars_out[column].to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    ("name", "inputs", "kwargs"),
    [
        ("adv", (2, 3), {"window": 10}),
        ("abnormal_volume", (3,), {"window": 10}),
        ("abnormal_turnover", (5,), {"window": 10}),
        ("volume_volatility", (3,), {"window": 5}),
        ("turnover_volatility", (5,), {"window": 5}),
        ("volume_autocorr", (3,), {"window": 10, "lag": 1}),
        ("turnover_autocorr", (5,), {"window": 10, "lag": 2}),
        ("amihud_illiquidity", (4, 2, 3), {"window": 10}),
        ("price_impact", (4, 6), {"window": 10}),
        ("return_per_turnover", (4, 5), {}),
        ("volume_shock", (3,), {"window": 10}),
        ("turnover_shock", (5,), {"window": 10}),
        ("volume_acceleration", (3,), {"short_window": 3, "long_window": 10}),
        ("turnover_acceleration", (5,), {"short_window": 3, "long_window": 10}),
        ("up_volume_ratio", (4, 3), {"window": 10}),
        ("down_volume_ratio", (4, 3), {"window": 10}),
        ("signed_volume_imbalance", (4, 3), {"window": 10}),
        ("up_down_volume_ratio", (4, 3), {"window": 10}),
        ("volume_weighted_return", (4, 3), {"window": 10}),
        ("volume_weighted_momentum", (2, 3), {"window": 10}),
        ("price_volume_divergence", (2, 3), {"price_window": 5, "volume_window": 10}),
        ("price_turnover_divergence", (2, 5), {"price_window": 5, "turnover_window": 10}),
        ("return_volume_beta", (4, 3), {"window": 10}),
        ("return_turnover_beta", (4, 5), {"window": 10}),
        ("ADL", (0, 1, 2, 3), {"window": 10}),
        ("ChaikinOscillator", (0, 1, 2, 3), {"fast_window": 3, "slow_window": 10, "adl_window": 5}),
        ("ForceIndex", (2, 3), {"window": 5}),
        ("EaseOfMovement", (0, 1, 3), {"window": 5, "volume_scale": 1e4}),
        ("bounded_nvi", (2, 3), {"window": 10}),
        ("bounded_pvi", (2, 3), {"window": 10}),
        ("zero_return_ratio", (4,), {"window": 10, "epsilon": 1e-12}),
        ("roll_spread_proxy", (4,), {"window": 10}),
        ("corwin_schultz_spread", (0, 1), {"window": 10}),
        ("high_low_spread_proxy", (0, 1), {"window": 10}),
        ("turnover_adjusted_volatility", (4, 5), {"window": 10}),
        ("volume_to_range", (3, 0, 1), {"window": 10}),
        ("rolling_vwap", (2, 3), {"window": 10}),
        ("vwap_deviation", (2, 3), {"window": 10}),
        ("relative_volume", (3,), {"window": 10}),
        ("dollar_volume", (2, 3), {}),
        ("dollar_volume_zscore", (2, 3), {"window": 10}),
        ("volume_momentum", (3,), {"window": 5}),
        ("turnover_momentum", (5,), {"window": 5}),
        ("volume_zscore", (3,), {"window": 10}),
        ("turnover_zscore", (5,), {"window": 10}),
        ("return_volume_corr", (4, 3), {"window": 10}),
        ("abs_return_volume_corr", (4, 3), {"window": 10}),
        ("signed_volume", (4, 3), {}),
        ("signed_dollar_volume", (4, 2, 3), {}),
        ("rolling_obv", (2, 3), {"window": 10}),
        ("rolling_pvt", (2, 3), {"window": 10}),
        ("ts_average_volume", (3,), {"window": 10}),
        ("ts_impulse_volume", (3,), {"window": 5, "baseline_window": 10}),
        ("ts_consolidation_volume_decay", (3,), {"window": 10}),
        ("true_turnover_rate", (3, 7), {}),
        ("average_turnover", (5,), {"window": 10}),
        ("ts_impulse_return", (2,), {"window": 5}),
        ("ts_impulse_strength", (2,), {"window": 5, "vol_window": 10}),
    ],
)
def test_native_polars_volume_matches_pandas(panels, name, inputs, kwargs):
    _assert_parity(name, tuple(panels[index] for index in inputs), kwargs)


def test_native_polars_volume_nan_warmup_matches_pandas():
    rng = np.random.default_rng(99)
    index = pd.date_range("2023-01-01", periods=60, freq="D")
    mask = rng.random((60, 2)) < 0.12
    close = pd.DataFrame(rng.normal(10, 3, (60, 2)), index=index, columns=["A", "B"])
    close[mask] = np.nan
    high = close + 1.0
    low = close - 1.0
    volume = pd.DataFrame(rng.uniform(1e3, 1e5, (60, 2)), index=index, columns=["A", "B"])
    volume[mask] = np.nan
    ret = close.pct_change()
    for name, args, kwargs in (
        ("abnormal_volume", (volume,), {"window": 5}),
        ("volume_volatility", (volume,), {"window": 5}),
        ("volume_autocorr", (volume,), {"window": 5, "lag": 1}),
        ("amihud_illiquidity", (ret, close, volume), {"window": 5}),
        ("volume_shock", (volume,), {"window": 5}),
        ("volume_acceleration", (volume,), {"short_window": 3, "long_window": 6}),
        ("up_volume_ratio", (ret, volume), {"window": 5}),
        ("volume_weighted_return", (ret, volume), {"window": 5}),
        ("price_volume_divergence", (close, volume), {"price_window": 3, "volume_window": 5}),
        ("return_volume_beta", (ret, volume), {"window": 5}),
        ("ADL", (high, low, close, volume), {"window": 5}),
        ("ForceIndex", (close, volume), {"window": 3}),
        ("bounded_nvi", (close, volume), {"window": 5}),
        ("bounded_pvi", (close, volume), {"window": 5}),
        ("zero_return_ratio", (ret,), {"window": 5, "epsilon": 1e-12}),
        ("roll_spread_proxy", (ret,), {"window": 5}),
        ("corwin_schultz_spread", (high, low), {"window": 5}),
        ("turnover_adjusted_volatility", (ret, volume / 1e6), {"window": 5}),
        ("volume_to_range", (volume, high, low), {"window": 5}),
        ("rolling_vwap", (close, volume), {"window": 5}),
        ("relative_volume", (volume,), {"window": 5}),
        ("return_volume_corr", (ret, volume), {"window": 5}),
        ("rolling_obv", (close, volume), {"window": 5}),
        ("rolling_pvt", (close, volume), {"window": 5}),
        ("ts_impulse_volume", (volume,), {"window": 3, "baseline_window": 5}),
        ("ts_consolidation_volume_decay", (volume,), {"window": 5}),
    ):
        _assert_parity(name, args, kwargs)
