# -*- coding: utf-8 -*-
"""Prefix-invariance audit of causal vs non-causal transforms.

A transform ``F`` is *prefix-invariant* (a strict causality requirement) when
``F(X[:T])[:t] == F(X[:t])[:t]`` within tolerance for all ``t <= T``: each
output point depends only on observations at or before it (and for these
filter types, strictly before it).

This test file proves two things:
  1. ``trailing_sma`` (a member of the new causal smoother family) IS
     prefix-invariant — a passing test.
  2. ``hp_filter`` from ``transforms/decomposition/trend.py`` is NOT
     prefix-invariant despite its original ``"causal"`` claim — a failing
     (RED) test. It excludes the current observation via ``shift(1)`` but
     then solves the HP objective over the WHOLE lagged sample
     (``spsolve(A, y)``), so a historical trend point can depend on later
     lagged observations.
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.transforms.smoothing import trailing_sma
from factor_preprocess.transforms.decomposition.trend import hp_filter
from factor_preprocess.registry.transforms import (
    TransformRegistry,
    TransformCategory,
    create_default_registry,
)


def _make_single_asset(seed: int = 0, n: int = 60):
    """Single-asset synthetic series (trend + cycle + noise)."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = 0.05 * t
    cycle = 1.5 * np.sin(2 * np.pi * t / 12.0)
    noise = rng.normal(0, 0.3, n)
    values = trend + cycle + noise + rng.normal(0, 0.1, n)
    df = pd.DataFrame(
        {
            "asset_id": ["A"] * n,
            "date": list(t),
            "value": values,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 1. PASS: the causal smoother family IS prefix-invariant.
# ---------------------------------------------------------------------------
def test_trailing_sma_is_prefix_invariant():
    """Recomputing trailing_sma over a prefix reproduces the prefix outputs."""
    df = _make_single_asset(seed=1, n=60)
    full = trailing_sma(df, window=5)
    T = len(df)
    for t in [10, 20, 30, 45, 55]:
        prefix_df = df.iloc[:t]
        prefix_out = trailing_sma(prefix_df, window=5)
        expected = full.iloc[:t].to_numpy()
        actual = prefix_out.iloc[:t].to_numpy()
        # Compare only where both are defined (NaN warmup propagates the same).
        mask = ~np.isnan(expected) & ~np.isnan(actual)
        np.testing.assert_allclose(
            actual[mask], expected[mask], rtol=1e-9, atol=1e-9,
            err_msg=f"trailing_sma not prefix-invariant at t={t}",
        )


# ---------------------------------------------------------------------------
# 2. RED: hp_filter claims causal but is NOT prefix-invariant.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="hp_filter is NOT prefix-invariant: it solves the HP objective over "
    "the whole lagged sample (spsolve), so a historical trend point depends on "
    "later lagged observations. Known non-causal bug, documented here and "
    "barred from production as OFFLINE_ONLY.",
    strict=False,
)
def test_hp_filter_is_prefix_invariant():
    """This test documents that hp_filter is NOT prefix-invariant and FAILS.

    It asserts the causality property that hp_filter's docstring promises but
    does not deliver. The transform is therefore registered OFFLINE_ONLY and
    must never run in production.
    """
    df = _make_single_asset(seed=7, n=40)
    full = hp_filter(df, lambda_param=1600.0)

    # Truncate at several cut points and recompute over the prefix only.
    for t in [10, 15, 20, 25, 30, 35]:
        prefix_df = df.iloc[:t]
        prefix_out = hp_filter(prefix_df, lambda_param=1600.0)
        expected = full.iloc[:t].to_numpy()
        actual = prefix_out.iloc[:t].to_numpy()

        mask = ~np.isnan(expected) & ~np.isnan(actual)
        assert mask.sum() >= 5, f"too few comparable points at t={t}"
        np.testing.assert_allclose(
            actual[mask], expected[mask], rtol=1e-6, atol=1e-6,
            err_msg=f"hp_filter is NOT prefix-invariant at t={t}",
        )


# ---------------------------------------------------------------------------
# 3. Guard: hp_filter / hp_decompose must be OFFLINE_ONLY in the registry.
# ---------------------------------------------------------------------------
def test_hp_filters_are_offline_only_not_production_causal():
    """The non-prefix-invariant HP transforms must be barred from production."""
    registry = create_default_registry()
    for name in ("hp_filter", "hp_decompose"):
        meta = registry.get(name)
        assert meta is not None, f"{name} must be registered"
        assert meta.causal_safe is False, f"{name} must be causal_safe=False"
        assert meta.admission == "OFFLINE_ONLY", f"{name} must be OFFLINE_ONLY"
        # Production validation must fail closed.
        with pytest.raises(ValueError):
            registry.validate_production(name)

    # hp_filter must not appear among the causal-safe transforms.
    names = {m.name for m in registry.list_causal_safe()}
    assert "hp_filter" not in names
    assert "hp_decompose" not in names


# ---------------------------------------------------------------------------
# 4. The causal smoother family is PRODUCTION-admissible in the registry.
# ---------------------------------------------------------------------------
def test_causal_smoothers_are_production():
    registry = create_default_registry()
    for name in (
        "trailing_sma",
        "trailing_median",
        "robust_ewma",
        "kama",
        "one_sided_iir_lowpass",
        "kalman_local_level",
    ):
        meta = registry.get(name)
        assert meta is not None, f"{name} not registered"
        assert meta.causal_safe is True, f"{name} must be causal_safe"
        assert meta.admission == "PRODUCTION", f"{name} must be PRODUCTION"
        registry.validate_production(name)  # should not raise
