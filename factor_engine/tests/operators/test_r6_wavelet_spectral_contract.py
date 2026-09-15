from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model import wavelet_spectral as ws


NAMES = (
    "ts_wavelet_low_frequency_ratio",
    "ts_wavelet_high_frequency_ratio",
    "ts_wavelet_entropy",
    "ts_wavelet_energy_slope",
    "ts_spectral_low_frequency_ratio",
)


def test_complete_defaults_and_history_contracts():
    ensure_cleaned_loaded()
    for name in NAMES:
        canonical = OperatorRegistry.get(name, "pandas_numpy", mode="any")
        assert canonical.metadata.panel_params == ("x",)
        assert canonical.metadata.panel_arity == 1
        assert canonical.metadata.scalar_params == ("window",)
        assert set(canonical.metadata.param_specs) == {"window"}
        window = canonical.metadata.param_specs["window"]
        assert window.default == 128
        assert window.history_semantics == "max_rows"
        polars = OperatorRegistry.get(name, "polars", mode="any")
        assert polars.metadata.param_specs == canonical.metadata.param_specs
        assert polars.metadata.panel_params == canonical.metadata.panel_params
        assert polars.metadata.scalar_params == canonical.metadata.scalar_params


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_wavelet_statistics_are_scale_invariant_and_finite(factor):
    t = np.arange(128, dtype=float)
    values = np.sin(2.0 * np.pi * t / 32.0) + 0.3 * (-1.0) ** t
    stats = ("low", "high", "entropy", "slope")
    expected = [ws._wavelet_stats(values, 128, stat) for stat in stats]
    actual = [ws._wavelet_stats(values * factor, 128, stat) for stat in stats]
    assert np.isfinite(actual).all()
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_spectral_ratio_is_scale_invariant_and_finite(factor):
    t = np.arange(128, dtype=float)
    values = np.cos(2.0 * np.pi * 3.0 * t / 128.0) + 0.5 * (-1.0) ** t
    expected = ws._spectral_low_ratio(values, 128)
    actual = ws._spectral_low_ratio(values * factor, 128)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_full_window_and_frequency_semantics_survive_scaling():
    values = np.arange(128, dtype=float)
    with_gap = values.copy()
    with_gap[1] = np.nan
    assert np.isnan(ws._wavelet_stats(with_gap, 128, "low"))
    assert np.isnan(ws._spectral_low_ratio(with_gap, 128))
    assert np.isnan(ws._wavelet_stats(values[:-1], 128, "low"))
    assert np.isnan(ws._spectral_low_ratio(values[:-1], 128))
    t = np.arange(128, dtype=float)
    low = np.cos(2.0 * np.pi * 2.0 * t / 128.0)
    high = (-1.0) ** t
    assert ws._spectral_low_ratio(low, 128) == pytest.approx(1.0, abs=1e-12)
    assert ws._spectral_low_ratio(high, 128) == pytest.approx(0.0, abs=1e-12)


def test_registry_outputs_match_across_final_backends():
    ensure_cleaned_loaded()
    rng = np.random.default_rng(20260914)
    values = rng.normal(size=160)
    pandas_frame = pd.DataFrame({"A": values})
    polars_frame = pl.DataFrame({"A": values})
    for name in NAMES:
        canonical = OperatorRegistry.get(name, "pandas_numpy", mode="any")
        polars = OperatorRegistry.get(name, "polars", mode="any")
        expected = canonical.calculate(x=pandas_frame, window=128)
        actual = polars.calculate(x=polars_frame, window=128)
        if hasattr(actual, "to_pandas"):
            actual = actual.to_pandas()
        np.testing.assert_allclose(
            np.asarray(actual["A"], dtype=float),
            np.asarray(expected["A"], dtype=float),
            rtol=2e-12,
            atol=2e-12,
            equal_nan=True,
        )
