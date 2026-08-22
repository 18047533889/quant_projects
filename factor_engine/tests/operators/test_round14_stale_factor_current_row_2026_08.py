# -*- coding: utf-8 -*-
"""Round-14 closure: current-row stale-factor fixes.

Covers the P0-5 / P0-6 / P0-7 current-row-required fixes in the Hankel/SSA,
multifractal and persistence-entropy topology families, plus the P1-20 honest
rename (``ts_multifractal_width`` -> ``ts_generalized_hurst_spread_q1_q4`` with
a working alias), the new ``ts_multifractal_spectrum_width`` canonical, and the
P1-21 curvature q-grid identity alignment.

Each stale-factor test builds a valid trailing window, then flips ONLY the
current row to NaN (history untouched) and asserts the CURRENT output is NaN —
never a stale value computed from yesterday's finite run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()
    yield


def _panel(n: int = 120, *, nan_last: bool = False, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = np.cumsum(rng.normal(0, 1, n)).astype(float)
    if nan_last:
        x[-1] = np.nan
    return pd.DataFrame({"close": x}, index=idx)


def _last(name: str, df: pd.DataFrame, **kwargs) -> float:
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, f"{name} has no pandas_numpy backend"
    out = op.calculate(df, **kwargs)
    arr = np.asarray(out, dtype=float)
    return float(arr[-1, 0])


# ---------------------------------------------------------------------------
# P0-5 Hankel / SSA current-row stale factor
# ---------------------------------------------------------------------------

_HANKEL_KWARGS = {"window": 40, "embedding_dim": 10}


def test_hankel_effective_rank_current_row_nan_is_nan():
    assert np.isnan(_last("ts_hankel_effective_rank", _panel(nan_last=True), **_HANKEL_KWARGS))


def test_hankel_singular_gap_current_row_nan_is_nan():
    assert np.isnan(_last("ts_hankel_singular_gap", _panel(nan_last=True), **_HANKEL_KWARGS))


def test_ssa_reconstruction_residual_current_row_nan_is_nan():
    assert np.isnan(
        _last("ts_ssa_reconstruction_residual", _panel(nan_last=True), **_HANKEL_KWARGS, n_components=3)
    )


def test_hankel_happy_path_finite():
    for name in ("ts_hankel_effective_rank", "ts_hankel_singular_gap"):
        assert np.isfinite(_last(name, _panel(), **_HANKEL_KWARGS)), name
    assert np.isfinite(
        _last("ts_ssa_reconstruction_residual", _panel(), **_HANKEL_KWARGS, n_components=3)
    )


# ---------------------------------------------------------------------------
# P0-6 multifractal current-row stale factor
# ---------------------------------------------------------------------------

_MF_KWARGS = {"window": 60}


def test_multifractal_spread_current_row_nan_is_nan():
    assert np.isnan(_last("ts_generalized_hurst_spread_q1_q4", _panel(nan_last=True), **_MF_KWARGS))


def test_multifractal_curvature_current_row_nan_is_nan():
    assert np.isnan(_last("ts_multifractal_curvature", _panel(nan_last=True), **_MF_KWARGS))


def test_multifractal_spectrum_width_current_row_nan_is_nan():
    assert np.isnan(_last("ts_multifractal_spectrum_width", _panel(nan_last=True), **_MF_KWARGS))


def test_generalized_hurst_exponent_current_row_nan_is_nan():
    assert np.isnan(_last("ts_generalized_hurst_exponent", _panel(nan_last=True), **_MF_KWARGS, q=2.0))


def test_multifractal_happy_path_finite():
    for name in ("ts_generalized_hurst_spread_q1_q4", "ts_multifractal_spectrum_width"):
        assert np.isfinite(_last(name, _panel(), **_MF_KWARGS)), name


# ---------------------------------------------------------------------------
# P0-7 persistence-entropy H0/H1 topology current-row stale factor
# ---------------------------------------------------------------------------

_TOPOLOGY_KWARGS = {"window": 40, "tau": 1, "dim": 3}


def test_persistence_entropy_h0_current_row_nan_is_nan():
    assert np.isnan(_last("ts_persistence_entropy_h0", _panel(nan_last=True), **_TOPOLOGY_KWARGS))


def test_persistence_entropy_h1_current_row_nan_is_nan():
    assert np.isnan(_last("ts_persistence_entropy_h1", _panel(nan_last=True), **_TOPOLOGY_KWARGS))


def test_persistence_entropy_h0_happy_path_finite():
    assert np.isfinite(_last("ts_persistence_entropy_h0", _panel(), **_TOPOLOGY_KWARGS))


# ---------------------------------------------------------------------------
# P1-20 honest rename: old spelling is a working alias, same values
# ---------------------------------------------------------------------------

def test_multifractal_width_resolves_to_spread_canonical():
    assert (
        OperatorRegistry.resolve_canonical("ts_multifractal_width")
        == "ts_generalized_hurst_spread_q1_q4"
    )


def test_rename_alias_produces_same_values():
    df = _panel()
    new = _last("ts_generalized_hurst_spread_q1_q4", df, **_MF_KWARGS)
    alias = _last("ts_multifractal_width", df, **_MF_KWARGS)
    assert np.isfinite(new) and np.isfinite(alias)
    assert np.isclose(new, alias, equal_nan=True)
    # Same on a NaN-current-row input: both emit NaN.
    df_nan = _panel(nan_last=True)
    assert np.isnan(_last("ts_generalized_hurst_spread_q1_q4", df_nan, **_MF_KWARGS))
    assert np.isnan(_last("ts_multifractal_width", df_nan, **_MF_KWARGS))


# ---------------------------------------------------------------------------
# P1-20 new canonical: true singularity-spectrum width
# ---------------------------------------------------------------------------

def test_multifractal_spectrum_width_finite_non_negative_on_noisy_series():
    out = _last("ts_multifractal_spectrum_width", _panel(), **_MF_KWARGS)
    assert np.isfinite(out)
    assert out >= 0.0


def test_multifractal_spectrum_width_nan_on_insufficient_history():
    short = _panel(n=30)  # far below window=60 -> nothing can emit
    assert np.isnan(_last("ts_multifractal_spectrum_width", short, **_MF_KWARGS))


# ---------------------------------------------------------------------------
# P1-21 curvature q-grid identity is aligned (documented grid == real grid)
# ---------------------------------------------------------------------------

def test_multifractal_curvature_semantic_version_bumped():
    catalog = OperatorRegistry._catalog.get("ts_multifractal_curvature", {})
    assert str(catalog.get("semantic_version") or "") == "2.0"


def test_multifractal_spectrum_width_registered_with_both_backends():
    backends = sorted(OperatorRegistry.backends_for("ts_multifractal_spectrum_width"))
    assert "pandas_numpy" in backends and "polars" in backends


def test_generalized_hurst_spread_registered_with_both_backends():
    backends = sorted(OperatorRegistry.backends_for("ts_generalized_hurst_spread_q1_q4"))
    assert "pandas_numpy" in backends and "polars" in backends
