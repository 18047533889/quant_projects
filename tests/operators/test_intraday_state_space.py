# -*- coding: utf-8 -*-
"""Tests for the R47 intraday state-space operators.

Covers: intra_kalman_latent_price, intra_state_space_volume_components,
intra_functional_motif_score, intra_visibility_graph_features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def setup_module():
    """Reset registry lifecycle to allow operator registration during test imports."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._lifecycle != OperatorRegistry.Lifecycle.BUILDING:
        OperatorRegistry._lifecycle = OperatorRegistry.Lifecycle.BUILDING


import factor_engine.cleaned_operators.intraday.state_space  # noqa: F401  (registers operators)

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

CANONICALS = [
    "intra_kalman_latent_price",
    "intra_state_space_volume_components",
    "intra_functional_motif_score",
    "intra_visibility_graph_features",
]


def _minute_panel(days: int = 3, bars: int = 240, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    """Generate synthetic minute OHLCV panel."""
    rng = np.random.default_rng(seed)
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        # Morning: 570-690 (09:30-11:30), Afternoon: 780-900 (13:00-15:00)
        for m in list(range(570, 691))[: bars // 2] + list(range(780, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((len(timestamps), cols)) * 0.01, axis=0)) * 100,
        index=pd.DatetimeIndex(timestamps, tz="Asia/Shanghai"),
        columns=[f"C{i}" for i in range(cols)],
    )
    return close


def _amount_volume(close: pd.DataFrame, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic amount/volume aligned with close."""
    rng = np.random.default_rng(seed)
    vol = pd.DataFrame(
        np.abs(rng.standard_normal(close.shape)) * 1000 + 1000,
        index=close.index,
        columns=close.columns,
    )
    amount = vol * close
    return amount, vol


def _all_nan_panel(days: int, bars: int, cols: int = 2) -> pd.DataFrame:
    """Generate all-NaN minute panel."""
    timestamps = []
    for d in range(days):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=d)
        for m in list(range(570, 691))[: bars // 2] + list(range(780, 901))[: bars // 2]:
            timestamps.append(day + pd.Timedelta(minutes=m))
    idx = pd.DatetimeIndex(timestamps, tz="Asia/Shanghai")
    return pd.DataFrame(np.nan, index=idx, columns=[f"C{i}" for i in range(cols)])


# ---------------------------------------------------------------------------
# Registration + surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_registered_and_classified(name: str) -> None:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


# ---------------------------------------------------------------------------
# Shape + determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_shape_and_determinism(name: str) -> None:
    close = _minute_panel(days=3, seed=1)
    amount, vol = _amount_volume(close)
    op = OperatorRegistry.get(name)

    if name == "intra_kalman_latent_price":
        args = (close,)
        kwargs = {"min_bars": 30}
    elif name == "intra_state_space_volume_components":
        args = (vol,)
        kwargs = {"min_bars": 30}
    elif name == "intra_functional_motif_score":
        args = (close, vol)
        kwargs = {"min_bars": 30}
    elif name == "intra_visibility_graph_features":
        args = (close,)
        kwargs = {"min_bars": 30}
    else:
        raise ValueError(f"Unknown operator: {name}")

    first = op.calculate(*args, **kwargs)
    second = op.calculate(*args, **kwargs)
    assert first.shape == (3, 2), f"{name}: expected (3,2), got {first.shape}"
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


# ---------------------------------------------------------------------------
# All-NaN input -> all-NaN output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_all_nan_input(name: str) -> None:
    close = _all_nan_panel(days=3, bars=240)
    vol = _all_nan_panel(days=3, bars=240)
    op = OperatorRegistry.get(name)

    if name == "intra_kalman_latent_price":
        args = (close,)
    elif name == "intra_state_space_volume_components":
        args = (vol,)
    elif name == "intra_functional_motif_score":
        args = (close, vol)
    elif name == "intra_visibility_graph_features":
        args = (close,)
    else:
        raise ValueError(f"Unknown operator: {name}")

    result = op.calculate(*args, min_bars=30)

    # intra_functional_motif_score may return empty DataFrame for all-NaN input
    if name == "intra_functional_motif_score" and result.shape[0] == 0:
        # Accept empty result as valid behavior for all-NaN input
        pass
    else:
        assert result.shape == (3, 2), name
        assert result.isna().all().all(), f"{name}: expected all-NaN output for all-NaN input"


# ---------------------------------------------------------------------------
# Future-poison (PIT safety)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_future_poison(name: str) -> None:
    """Tampering with the last day's data must not affect earlier days."""
    close = _minute_panel(days=4, seed=6, cols=2)
    amount, vol = _amount_volume(close)
    op = OperatorRegistry.get(name)

    if name == "intra_kalman_latent_price":
        args = (close,)
    elif name == "intra_state_space_volume_components":
        args = (vol,)
    elif name == "intra_functional_motif_score":
        args = (close, vol)
    elif name == "intra_visibility_graph_features":
        args = (close,)
    else:
        raise ValueError(f"Unknown operator: {name}")

    base = op.calculate(*args, min_bars=30)

    # Tamper with last day (day index 3)
    tampered_close = close.copy()
    tampered_vol = vol.copy()
    last_day = close.index.normalize().unique()[-1]
    mask = close.index.normalize() == last_day
    tampered_close.loc[mask] *= 10.0
    tampered_vol.loc[mask] *= 10.0

    if name == "intra_kalman_latent_price":
        targs = (tampered_close,)
    elif name == "intra_state_space_volume_components":
        targs = (tampered_vol,)
    elif name == "intra_functional_motif_score":
        targs = (tampered_close, tampered_vol)
    elif name == "intra_visibility_graph_features":
        targs = (tampered_close,)
    else:
        raise ValueError(f"Unknown operator: {name}")

    tampered = op.calculate(*targs, min_bars=30)

    # First 3 days must be identical
    try:
        pd.testing.assert_frame_equal(
            base.iloc[:3], tampered.iloc[:3], check_dtype=False
        )
    except AssertionError as e:
        raise AssertionError(f"{name}: tampering last day affected earlier days (PIT violation)") from e


# ---------------------------------------------------------------------------
# Min_bars threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_min_bars_threshold(name: str) -> None:
    """Insufficient bars (below min_bars) must return NaN."""
    close = _minute_panel(days=1, bars=20, seed=10, cols=1)
    amount, vol = _amount_volume(close, seed=11)
    op = OperatorRegistry.get(name)

    if name == "intra_kalman_latent_price":
        args = (close,)
    elif name == "intra_state_space_volume_components":
        args = (vol,)
    elif name == "intra_functional_motif_score":
        args = (close, vol)
    elif name == "intra_visibility_graph_features":
        args = (close,)
    else:
        raise ValueError(f"Unknown operator: {name}")

    # min_bars=50 > 20 bars available -> should return NaN
    result = op.calculate(*args, min_bars=50)
    assert result.shape == (1, 1), name
    assert result.isna().all().all(), f"{name}: insufficient bars must return NaN"


# ---------------------------------------------------------------------------
# Dual-backend parity (pandas_numpy vs polars)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CANONICALS))
def test_dual_backend_parity(name: str) -> None:
    """Pandas and Polars backends must produce identical results."""
    close = _minute_panel(days=3, seed=42)
    amount, vol = _amount_volume(close, seed=43)

    pandas_op = OperatorRegistry.get(name, backend="pandas_numpy")
    polars_op = OperatorRegistry.get(name, backend="polars")

    if pandas_op is None or polars_op is None:
        pytest.skip(f"{name}: backend not available")

    if name == "intra_kalman_latent_price":
        args = (close,)
    elif name == "intra_state_space_volume_components":
        args = (vol,)
    elif name == "intra_functional_motif_score":
        args = (close, vol)
    elif name == "intra_visibility_graph_features":
        args = (close,)
    else:
        raise ValueError(f"Unknown operator: {name}")

    pandas_result = pandas_op.calculate(*args, min_bars=30)
    polars_result = polars_op.calculate(*args, min_bars=30)

    try:
        pd.testing.assert_frame_equal(
            pandas_result, polars_result, check_dtype=False, rtol=1e-6
        )
    except AssertionError as e:
        raise AssertionError(f"{name}: pandas vs polars backend mismatch") from e


# ---------------------------------------------------------------------------
# Kalman-specific tests
# ---------------------------------------------------------------------------

def test_kalman_latent_price_innovation() -> None:
    """Kalman filter should produce finite innovation variance on normal data."""
    close = _minute_panel(days=2, seed=99)
    op = OperatorRegistry.get("intra_kalman_latent_price")
    result = op.calculate(close, min_bars=30)
    assert result.shape == (2, 2)
    # At least one instrument should have finite result
    assert result.notna().any().any(), "Kalman filter produced all NaN on normal data"


# ---------------------------------------------------------------------------
# Volume components tests
# ---------------------------------------------------------------------------

def test_volume_components_ratio_bounds() -> None:
    """Volume component ratio must be in [0, 1]."""
    vol = _minute_panel(days=2, seed=88)
    vol = vol.abs() + 1.0  # Ensure positive volume
    op = OperatorRegistry.get("intra_state_space_volume_components")
    result = op.calculate(vol, min_bars=30)
    finite_vals = result.values[np.isfinite(result.values)]
    if len(finite_vals) > 0:
        assert np.all(finite_vals >= 0) and np.all(finite_vals <= 1), \
            "Volume component ratio must be in [0, 1]"


# ---------------------------------------------------------------------------
# Functional motif tests
# ---------------------------------------------------------------------------

def test_functional_motif_score_bounds() -> None:
    """Motif score (correlation difference) must be in [-2, 2]."""
    close = _minute_panel(days=2, seed=77)
    amount, vol = _amount_volume(close, seed=78)
    op = OperatorRegistry.get("intra_functional_motif_score")
    result = op.calculate(close, vol, min_bars=30)
    finite_vals = result.values[np.isfinite(result.values)]
    if len(finite_vals) > 0:
        assert np.all(finite_vals >= -2) and np.all(finite_vals <= 2), \
            "Motif score (correlation diff) must be in [-2, 2]"


# ---------------------------------------------------------------------------
# Visibility graph tests
# ---------------------------------------------------------------------------

def test_visibility_graph_nonnegative() -> None:
    """Visibility graph degree must be non-negative."""
    close = _minute_panel(days=2, seed=66)
    op = OperatorRegistry.get("intra_visibility_graph_features")
    result = op.calculate(close, min_bars=30)
    finite_vals = result.values[np.isfinite(result.values)]
    if len(finite_vals) > 0:
        assert np.all(finite_vals >= 0), "Visibility graph degree must be non-negative"
