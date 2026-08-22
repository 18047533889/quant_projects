# -*- coding: utf-8 -*-
"""Parity tests for native Polars chart-structure / pattern operators."""
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
    rng = np.random.default_rng(555)
    n = 220
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    close = pd.DataFrame(
        {
            "A": 100.0 + np.cumsum(rng.normal(0.0, 1.5, n)),
            "B": 40.0 + np.cumsum(rng.normal(0.02, 0.8, n)),
        },
        index=index,
    )
    spread = pd.DataFrame(rng.uniform(0.5, 2.5, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    volume = pd.DataFrame(rng.integers(5_000, 300_000, close.shape), index=index, columns=close.columns).astype(float)
    return high, low, close, volume


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


_STRUCT = {"left_window": 5, "right_window": 5, "history_window": 40}


@pytest.mark.parametrize(
    ("name", "inputs", "kwargs"),
    [
        ("ts_nth_pivot_high", (0,), {**_STRUCT, "n": 1}),
        ("ts_nth_pivot_low", (1,), {**_STRUCT, "n": 1}),
        ("ts_nth_pivot_high_age", (0,), {**_STRUCT, "n": 1}),
        ("ts_nth_pivot_low_age", (1,), {**_STRUCT, "n": 2}),
        ("ts_pivot_high_count", (0,), _STRUCT),
        ("ts_pivot_low_count", (1,), _STRUCT),
        ("ts_pivot_high_spacing", (0,), _STRUCT),
        ("ts_pivot_low_spacing", (1,), _STRUCT),
        ("ts_last_pivot_high", (0,), _STRUCT),
        ("ts_last_pivot_low", (1,), _STRUCT),
        ("ts_pivot_high_age", (0,), _STRUCT),
        ("ts_pivot_low_age", (1,), _STRUCT),
        ("ts_resistance_level", (0,), {**_STRUCT, "points": 3}),
        ("ts_support_level", (1,), {**_STRUCT, "points": 3}),
        ("ts_resistance_slope", (0,), {**_STRUCT, "points": 3}),
        ("ts_support_slope", (1,), {**_STRUCT, "points": 3}),
        ("ts_distance_to_resistance", (2, 0), {**_STRUCT, "points": 3}),
        ("ts_distance_to_support", (2, 1), {**_STRUCT, "points": 3}),
        ("ts_resistance_break", (2, 0), {**_STRUCT, "points": 3}),
        ("ts_support_break", (2, 1), {**_STRUCT, "points": 3}),
        ("ts_resistance_fit_r2", (0,), {**_STRUCT, "points": 3}),
        ("ts_support_fit_r2", (1,), {**_STRUCT, "points": 3}),
        ("ts_swing_amplitude", (0, 1), _STRUCT),
        ("ts_swing_amplitude_pct", (0, 1, 2), _STRUCT),
        ("ts_swing_duration", (0, 1), _STRUCT),
        ("ts_swing_velocity", (0, 1), _STRUCT),
        ("ts_swing_amplitude_atr", (0, 1, 2), {**_STRUCT, "atr_window": 14}),
        ("ts_channel_width", (0, 1), {**_STRUCT, "points": 3}),
        ("ts_channel_width_pct", (2, 0, 1), {**_STRUCT, "points": 3}),
        ("ts_channel_width_atr", (0, 1, 2), {**_STRUCT, "points": 3, "atr_window": 14}),
        ("ts_channel_width_slope", (0, 1), {**_STRUCT, "points": 3, "window": 10}),
        ("ts_line_convergence", (0, 1), {**_STRUCT, "points": 3}),
        ("ts_line_parallelism", (0, 1), {**_STRUCT, "points": 3}),
        ("ts_pattern_symmetry", (0, 1), _STRUCT),
        ("ts_consolidation_width", (0, 1), {"window": 10}),
        ("ts_consolidation_slope", (2,), {"window": 10}),
        ("pattern_double_top", (0, 1), {**_STRUCT, "tolerance": 0.05, "min_depth": 0.02, "min_spacing": 5, "max_spacing": 40}),
        ("pattern_double_bottom", (0, 1), {**_STRUCT, "tolerance": 0.05, "min_depth": 0.02, "min_spacing": 5, "max_spacing": 40}),
        ("pattern_head_shoulders", (0, 1), {**_STRUCT, "shoulder_tolerance": 0.1, "head_min_prominence": 0.05, "max_neckline_slope": 0.5}),
        ("pattern_inverse_head_shoulders", (0, 1), {**_STRUCT, "shoulder_tolerance": 0.1, "head_min_prominence": 0.05, "max_neckline_slope": 0.5}),
        ("pattern_sym_triangle", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_ascending_triangle", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_descending_triangle", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_rising_wedge", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_falling_wedge", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_rectangle", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_rising_channel", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01, "parallel_tolerance": 0.02}),
        ("pattern_falling_channel", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01, "parallel_tolerance": 0.02}),
        ("pattern_broadening", (0, 1), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_bull_flag", (2, 0, 1, 3), {"impulse_window": 10, "flag_window": 8, "min_impulse": 0.03, "max_retracement": 0.5, "max_width": 0.1, "volume_decay_threshold": 0.01}),
        ("pattern_bear_flag", (2, 0, 1, 3), {"impulse_window": 10, "flag_window": 8, "min_impulse": 0.03, "max_retracement": 0.5, "max_width": 0.1, "volume_decay_threshold": 0.01}),
        ("pattern_triple_top", (0, 1), {**_STRUCT, "tolerance": 0.05, "min_depth": 0.02, "min_spacing": 5, "max_spacing": 40}),
        ("pattern_triple_bottom", (0, 1), {**_STRUCT, "tolerance": 0.05, "min_depth": 0.02, "min_spacing": 5, "max_spacing": 40}),
        ("pattern_123_bull", (0, 1), {**_STRUCT, "min_swing": 0.03}),
        ("pattern_123_bear", (0, 1), {**_STRUCT, "min_swing": 0.03}),
        ("pattern_rounding_bottom", (2,), {"window": 20, "min_fit": 0.3}),
        ("pattern_rounding_top", (2,), {"window": 20, "min_fit": 0.3}),
        ("pattern_cup", (2,), {"window": 20, "min_depth": 0.02, "max_edge_diff": 0.5, "min_fit": 0.3}),
        ("pattern_cup_handle", (2, 0, 1), {"cup_window": 20, "handle_window": 5, "min_depth": 0.02, "max_edge_diff": 0.5, "min_fit": 0.3, "max_handle_retracement": 0.5}),
        ("pattern_bull_pennant", (2, 0, 1, 3), {"impulse_window": 10, "pennant_window": 8, "min_impulse": 0.03, "max_width": 0.1, "volume_decay_threshold": 0.01}),
        ("pattern_bear_pennant", (2, 0, 1, 3), {"impulse_window": 10, "pennant_window": 8, "min_impulse": 0.03, "max_width": 0.1, "volume_decay_threshold": 0.01}),
        ("pattern_breakout_retest", (2,), {"window": 10, "max_wait": 5, "tolerance": 0.01}),
        ("pattern_breakdown_retest", (2,), {"window": 10, "max_wait": 5, "tolerance": 0.01}),
    ],
)
def test_native_polars_structure_matches_pandas(panels, name, inputs, kwargs):
    _assert_parity(name, tuple(panels[index] for index in inputs), kwargs)


def test_native_polars_structure_nan_warmup_matches_pandas():
    rng = np.random.default_rng(17)
    n = 160
    index = pd.date_range("2022-06-01", periods=n, freq="D")
    mask = rng.random((n, 2)) < 0.08
    close = pd.DataFrame(np.cumsum(rng.normal(0.0, 1.5, (n, 2)), axis=0) + 100, index=index, columns=["A", "B"])
    close[mask] = np.nan
    spread = pd.DataFrame(rng.uniform(0.5, 2.5, close.shape), index=index, columns=close.columns)
    high = close + spread
    low = close - spread
    high[mask] = np.nan
    low[mask] = np.nan
    volume = pd.DataFrame(rng.integers(5_000, 200_000, close.shape), index=index, columns=close.columns).astype(float)
    volume[mask] = np.nan
    for name, args, kwargs in (
        ("ts_nth_pivot_high", (high,), {**_STRUCT, "n": 1}),
        ("ts_pivot_high_count", (high,), _STRUCT),
        ("ts_pivot_high_spacing", (high,), _STRUCT),
        ("ts_last_pivot_high", (high,), _STRUCT),
        ("ts_pivot_high_age", (high,), _STRUCT),
        ("ts_resistance_level", (high,), {**_STRUCT, "points": 3}),
        ("ts_resistance_slope", (high,), {**_STRUCT, "points": 3}),
        ("ts_distance_to_resistance", (close, high), {**_STRUCT, "points": 3}),
        ("ts_resistance_break", (close, high), {**_STRUCT, "points": 3}),
        ("ts_resistance_fit_r2", (high,), {**_STRUCT, "points": 3}),
        ("ts_swing_amplitude", (high, low), _STRUCT),
        ("ts_swing_amplitude_pct", (high, low, close), _STRUCT),
        ("ts_swing_duration", (high, low), _STRUCT),
        ("ts_swing_velocity", (high, low), _STRUCT),
        ("ts_channel_width", (high, low), {**_STRUCT, "points": 3}),
        ("ts_channel_width_pct", (close, high, low), {**_STRUCT, "points": 3}),
        ("ts_line_convergence", (high, low), {**_STRUCT, "points": 3}),
        ("ts_line_parallelism", (high, low), {**_STRUCT, "points": 3}),
        ("ts_pattern_symmetry", (high, low), _STRUCT),
        ("ts_consolidation_width", (high, low), {"window": 8}),
        ("ts_consolidation_slope", (close,), {"window": 8}),
        ("pattern_double_top", (high, low), {**_STRUCT, "tolerance": 0.05, "min_depth": 0.02, "min_spacing": 5, "max_spacing": 40}),
        ("pattern_head_shoulders", (high, low), {**_STRUCT, "shoulder_tolerance": 0.1, "head_min_prominence": 0.05, "max_neckline_slope": 0.5}),
        ("pattern_sym_triangle", (high, low), {**_STRUCT, "points": 3, "slope_threshold": 0.01}),
        ("pattern_rising_channel", (high, low), {**_STRUCT, "points": 3, "slope_threshold": 0.01, "parallel_tolerance": 0.02}),
        ("pattern_123_bull", (high, low), {**_STRUCT, "min_swing": 0.03}),
        ("pattern_triple_top", (high, low), {**_STRUCT, "tolerance": 0.05, "min_depth": 0.02, "min_spacing": 5, "max_spacing": 40}),
        ("pattern_rounding_bottom", (close,), {"window": 15, "min_fit": 0.3}),
        ("pattern_cup", (close,), {"window": 15, "min_depth": 0.02, "max_edge_diff": 0.5, "min_fit": 0.3}),
        ("pattern_bull_flag", (close, high, low, volume), {"impulse_window": 8, "flag_window": 6, "min_impulse": 0.03, "max_retracement": 0.5, "max_width": 0.1, "volume_decay_threshold": 0.01}),
        ("pattern_bull_pennant", (close, high, low, volume), {"impulse_window": 8, "pennant_window": 6, "min_impulse": 0.03, "max_width": 0.1, "volume_decay_threshold": 0.01}),
        ("pattern_breakout_retest", (close,), {"window": 8, "max_wait": 4, "tolerance": 0.01}),
    ):
        _assert_parity(name, args, kwargs)
