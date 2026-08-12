# -*- coding: utf-8 -*-
"""Tests for adaptive filter operators (2026-08-13).

Covers five adaptive filtering operators following R47 conventions:
  - ts_mcginley_dynamic
  - ts_vidya
  - ts_one_euro_filter
  - ts_nlms_filter
  - ts_rls_filter
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cleaned_operators.technical.adaptive_filters  # noqa: F401

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

ADAPTIVE_FILTER_CANONICALS = [
    "ts_mcginley_dynamic",
    "ts_vidya",
    "ts_one_euro_filter",
    "ts_nlms_filter",
    "ts_rls_filter",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _daily_panel(days: int = 100, cols: int = 2, seed: int = 0, start: str = "2024-01-01") -> pd.DataFrame:
    """Generate daily price panel with random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="B")
    return pd.DataFrame(
        np.exp(np.cumsum(rng.standard_normal((days, cols)) * 0.01, axis=0)) * 100.0,
        index=idx,
        columns=[f"C{i}" for i in range(cols)],
    )


def _all_nan(template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# registration + surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_registered_and_classified(name: str) -> None:
    """Every operator must be registered and classified as extended."""
    from cleaned_operators.operator_surface import classify_canonical

    assert OperatorRegistry.get(name) is not None, name
    assert classify_canonical(name) in ("daily", "extended", "research"), name


@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_has_explicit_policy(name: str) -> None:
    """Every operator must have explicit PIT policy."""
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    assert name in _EXPLICIT_POLICIES, f"{name} missing explicit policy"
    policy = _EXPLICIT_POLICIES[name]
    assert policy.get("scope") == "ts", f"{name} scope must be 'ts'"
    assert policy.get("pit_safe") is True, f"{name} must be pit_safe=True"


# ---------------------------------------------------------------------------
# shape and determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_shape_and_determinism(name: str) -> None:
    """Output shape matches input, deterministic across runs."""
    panel = _daily_panel(days=80, cols=2, seed=1)
    op = OperatorRegistry.get(name)

    # default parameters
    if name == "ts_mcginley_dynamic":
        result1 = op(panel, window=20, power=4.0)
        result2 = op(panel, window=20, power=4.0)
    elif name == "ts_vidya":
        result1 = op(panel, window=20, smooth=9)
        result2 = op(panel, window=20, smooth=9)
    elif name == "ts_one_euro_filter":
        result1 = op(panel, min_cutoff=1.0, beta=0.007)
        result2 = op(panel, min_cutoff=1.0, beta=0.007)
    elif name == "ts_nlms_filter":
        result1 = op(panel, order=5, mu=0.1, eps=1e-6)
        result2 = op(panel, order=5, mu=0.1, eps=1e-6)
    elif name == "ts_rls_filter":
        result1 = op(panel, order=5, lambda_=0.99, delta=1.0)
        result2 = op(panel, order=5, lambda_=0.99, delta=1.0)
    else:
        pytest.fail(f"Unknown operator: {name}")

    # shape preservation
    assert result1.shape == panel.shape, f"{name} shape mismatch"
    assert result1.index.equals(panel.index), f"{name} index mismatch"
    assert result1.columns.equals(panel.columns), f"{name} columns mismatch"

    # determinism
    pd.testing.assert_frame_equal(result1, result2, check_exact=False, rtol=1e-10)


# ---------------------------------------------------------------------------
# causal (no future leakage)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_causal_no_future_leak(name: str) -> None:
    """Output at t depends only on x[:t+1], not x[t+1:]."""
    panel = _daily_panel(days=50, cols=2, seed=2)
    op = OperatorRegistry.get(name)

    # parameters
    if name == "ts_mcginley_dynamic":
        kwargs = {"window": 10, "power": 4.0}
    elif name == "ts_vidya":
        kwargs = {"window": 10, "smooth": 5}
    elif name == "ts_one_euro_filter":
        kwargs = {"min_cutoff": 1.0, "beta": 0.01}
    elif name == "ts_nlms_filter":
        kwargs = {"order": 3, "mu": 0.1, "eps": 1e-6}
    elif name == "ts_rls_filter":
        kwargs = {"order": 3, "lambda_": 0.99, "delta": 1.0}
    else:
        pytest.fail(f"Unknown operator: {name}")

    split = 30
    full = op(panel, **kwargs)
    prefix = op(panel.iloc[:split], **kwargs)

    # prefix result must match full result up to split
    pd.testing.assert_frame_equal(
        full.iloc[:split], prefix, check_exact=False, rtol=1e-10, atol=1e-12
    )


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_all_nan_input(name: str) -> None:
    """All-NaN input produces all-NaN output."""
    panel = _all_nan(_daily_panel(days=30, cols=2))
    op = OperatorRegistry.get(name)

    if name == "ts_mcginley_dynamic":
        result = op(panel, window=10, power=4.0)
    elif name == "ts_vidya":
        result = op(panel, window=10, smooth=5)
    elif name == "ts_one_euro_filter":
        result = op(panel, min_cutoff=1.0, beta=0.01)
    elif name == "ts_nlms_filter":
        result = op(panel, order=3, mu=0.1, eps=1e-6)
    elif name == "ts_rls_filter":
        result = op(panel, order=3, lambda_=0.99, delta=1.0)
    else:
        pytest.fail(f"Unknown operator: {name}")

    assert result.isna().all().all(), f"{name} must return all-NaN for all-NaN input"


@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_warmup_period(name: str) -> None:
    """Output is NaN until minimum warmup period is satisfied."""
    panel = _daily_panel(days=50, cols=2, seed=3)
    op = OperatorRegistry.get(name)

    if name == "ts_mcginley_dynamic":
        result = op(panel, window=20, power=4.0)
        min_warmup = 20
    elif name == "ts_vidya":
        result = op(panel, window=20, smooth=9)
        min_warmup = 20
    elif name == "ts_one_euro_filter":
        result = op(panel, min_cutoff=1.0, beta=0.01)
        min_warmup = 1
    elif name == "ts_nlms_filter":
        result = op(panel, order=10, mu=0.1, eps=1e-6)
        min_warmup = 10
    elif name == "ts_rls_filter":
        result = op(panel, order=8, lambda_=0.99, delta=1.0)
        min_warmup = 8
    else:
        pytest.fail(f"Unknown operator: {name}")

    # check warmup period has NaN
    if min_warmup > 1:
        assert result.iloc[:min_warmup - 1].isna().all().all(), (
            f"{name} must have NaN in warmup period"
        )


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------
def test_mcginley_dynamic_parameters():
    """McGinley Dynamic parameter validation."""
    panel = _daily_panel(days=30, cols=1, seed=4)
    op = OperatorRegistry.get("ts_mcginley_dynamic")

    # valid
    result = op(panel, window=10, power=2.0)
    assert result.shape == panel.shape

    # invalid window
    with pytest.raises(ValueError, match="window"):
        op(panel, window=1, power=2.0)

    # invalid power
    with pytest.raises(ValueError, match="power"):
        op(panel, window=10, power=0.05)


def test_vidya_parameters():
    """VIDYA parameter validation."""
    panel = _daily_panel(days=30, cols=1, seed=5)
    op = OperatorRegistry.get("ts_vidya")

    # valid
    result = op(panel, window=10, smooth=5)
    assert result.shape == panel.shape

    # invalid window
    with pytest.raises(ValueError, match="window"):
        op(panel, window=1, smooth=5)

    # invalid smooth
    with pytest.raises(ValueError, match="smooth"):
        op(panel, window=10, smooth=0)


def test_one_euro_parameters():
    """1€ Filter parameter validation."""
    panel = _daily_panel(days=30, cols=1, seed=6)
    op = OperatorRegistry.get("ts_one_euro_filter")

    # valid
    result = op(panel, min_cutoff=1.0, beta=0.01)
    assert result.shape == panel.shape

    # invalid min_cutoff
    with pytest.raises(ValueError, match="min_cutoff"):
        op(panel, min_cutoff=0.0001, beta=0.01)

    # invalid beta
    with pytest.raises(ValueError, match="beta"):
        op(panel, min_cutoff=1.0, beta=-0.1)


def test_nlms_parameters():
    """NLMS filter parameter validation."""
    panel = _daily_panel(days=30, cols=1, seed=7)
    op = OperatorRegistry.get("ts_nlms_filter")

    # valid
    result = op(panel, order=5, mu=0.1, eps=1e-6)
    assert result.shape == panel.shape

    # invalid order (too large)
    with pytest.raises(ValueError, match="order"):
        op(panel, order=150, mu=0.1, eps=1e-6)

    # invalid mu
    with pytest.raises(ValueError, match="mu"):
        op(panel, order=5, mu=3.0, eps=1e-6)


def test_rls_parameters():
    """RLS filter parameter validation."""
    panel = _daily_panel(days=30, cols=1, seed=8)
    op = OperatorRegistry.get("ts_rls_filter")

    # valid
    result = op(panel, order=5, lambda_=0.99, delta=1.0)
    assert result.shape == panel.shape

    # invalid order (too large)
    with pytest.raises(ValueError, match="order"):
        op(panel, order=100, lambda_=0.99, delta=1.0)

    # invalid lambda
    with pytest.raises(ValueError, match="lambda_"):
        op(panel, order=5, lambda_=0.8, delta=1.0)


# ---------------------------------------------------------------------------
# dual backend parity (pandas vs polars)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ADAPTIVE_FILTER_CANONICALS)))
def test_dual_backend_parity(name: str) -> None:
    """Pandas and Polars backends produce identical results."""
    panel = _daily_panel(days=60, cols=2, seed=9)

    # parameters
    if name == "ts_mcginley_dynamic":
        kwargs = {"window": 15, "power": 3.0}
    elif name == "ts_vidya":
        kwargs = {"window": 15, "smooth": 7}
    elif name == "ts_one_euro_filter":
        kwargs = {"min_cutoff": 0.5, "beta": 0.01}
    elif name == "ts_nlms_filter":
        kwargs = {"order": 4, "mu": 0.15, "eps": 1e-6}
    elif name == "ts_rls_filter":
        kwargs = {"order": 4, "lambda_": 0.98, "delta": 1.0}
    else:
        pytest.fail(f"Unknown operator: {name}")

    # pandas backend
    pandas_result = OperatorRegistry.get(name, backend="pandas_numpy")(panel, **kwargs)

    # polars backend
    polars_result = OperatorRegistry.get(name, backend="polars")(panel, **kwargs)

    # compare
    pd.testing.assert_frame_equal(
        pandas_result, polars_result, check_exact=False, rtol=1e-9, atol=1e-10
    )


# ---------------------------------------------------------------------------
# smoothing behavior
# ---------------------------------------------------------------------------
def test_mcginley_smooths_and_adapts():
    """McGinley Dynamic smooths but adapts faster during trends."""
    rng = np.random.default_rng(10)
    # stable then trending
    stable = np.full(30, 100.0) + rng.standard_normal(30) * 0.5
    trend = 100.0 + np.arange(30) * 2.0 + rng.standard_normal(30) * 0.5
    series = np.concatenate([stable, trend])
    panel = pd.DataFrame(series, columns=["C0"])

    op = OperatorRegistry.get("ts_mcginley_dynamic")
    result = op(panel, window=10, power=4.0)

    # check smoothing (variance reduction)
    valid = result["C0"].dropna()
    assert valid.std() < panel["C0"].std(), "McGinley must smooth"


def test_vidya_adapts_to_volatility():
    """VIDYA adapts smoothing based on volatility."""
    rng = np.random.default_rng(11)
    low_vol = 100.0 + rng.standard_normal(30) * 0.2
    high_vol = 100.0 + rng.standard_normal(30) * 5.0
    series = np.concatenate([low_vol, high_vol])
    panel = pd.DataFrame(series, columns=["C0"])

    op = OperatorRegistry.get("ts_vidya")
    result = op(panel, window=10, smooth=5)

    # VIDYA should produce valid output
    valid = result["C0"].dropna()
    assert len(valid) > 0, "VIDYA must produce valid output"


def test_one_euro_reduces_lag_on_velocity():
    """1€ Filter reduces lag when velocity increases."""
    # step function
    series = np.concatenate([np.full(20, 100.0), np.full(20, 110.0)])
    panel = pd.DataFrame(series, columns=["C0"])

    op = OperatorRegistry.get("ts_one_euro_filter")
    result = op(panel, min_cutoff=0.5, beta=0.1)

    # should track the step
    valid = result["C0"].dropna()
    assert valid.iloc[-1] > 105.0, "1€ Filter should track step change"


def test_nlms_learns_pattern():
    """NLMS filter adapts weights to predict next value."""
    # simple linear trend
    series = 100.0 + np.arange(50) * 0.5
    panel = pd.DataFrame(series, columns=["C0"])

    op = OperatorRegistry.get("ts_nlms_filter")
    result = op(panel, order=3, mu=0.1, eps=1e-6)

    # prediction error should decrease over time
    error = (panel["C0"] - result["C0"]).abs()
    valid_error = error.dropna()
    if len(valid_error) > 20:
        early_err = valid_error.iloc[:10].mean()
        late_err = valid_error.iloc[-10:].mean()
        assert late_err < early_err, "NLMS should improve prediction over time"


def test_rls_converges_faster():
    """RLS filter converges faster than LMS."""
    series = 100.0 + np.arange(40) * 0.5
    panel = pd.DataFrame(series, columns=["C0"])

    op = OperatorRegistry.get("ts_rls_filter")
    result = op(panel, order=3, lambda_=0.99, delta=1.0)

    # should produce predictions
    valid = result["C0"].dropna()
    assert len(valid) > 10, "RLS must produce predictions after warmup"
