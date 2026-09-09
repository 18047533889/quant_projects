import numpy as np
import pytest
from scipy.signal import detrend


def _module():
    from factor_engine.cleaned_operators import research_spectral
    return research_spectral


def _direct_dft_reference(values, segments):
    size = len(values) // segments
    chunks = values[-size * segments:].reshape(segments, size)
    chunks = detrend(chunks, axis=1, type="linear") * np.hanning(size)
    upper = min(32, size // 2)
    # Independent matrix DFT; no use of the implementation's FFT or accumulators.
    basis = np.exp(-2j * np.pi * np.outer(np.arange(upper + 1), np.arange(size)) / size)
    spectra = chunks @ basis.T
    ratios = []
    for a in range(1, upper + 1):
        for b in range(1, upper - a + 1):
            product = spectra[:, a] * spectra[:, b]
            target = spectra[:, a + b]
            ratios.append(abs(np.vdot(target, product)) ** 2 /
                          (np.vdot(product, product).real * np.vdot(target, target).real))
    return np.asarray(ratios)


@pytest.mark.parametrize("size", [64, 65, 120])
def test_bicoherence_matches_direct_dft_and_changes_of_units(size):
    values = np.random.default_rng(size).normal(size=size)
    expected = _direct_dft_reference(values, 4)
    module = _module()
    for scale in (1.0, -1.0, 1e-300, 1e-3, 1e6, 1e300):
        actual = module._bicoherence_values(values * scale, 4)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_surrogate_excess_is_unit_invariant():
    values = np.random.default_rng(777).normal(size=64)
    fn = _module()._bicoherence_top_decile_excess
    expected = fn(values, 4)
    assert np.isfinite(expected)
    for scale in (1e-300, 1e-3, 1e6, 1e300, -2.0):
        assert fn(values * scale, 4) == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("level", [0.0, 100.0, 1e-300, 1e300])
def test_constant_detrended_windows_are_undefined(level):
    module = _module()
    values = np.full(64, level)
    assert module._bicoherence_values(values, 4) == []
    assert np.isnan(module._bicoherence_top_decile_mean(values, 4))
    assert np.isnan(module._bicoherence_top_decile_excess(values, 4))


@pytest.mark.parametrize("canonical", ["ts_bicoherence_top_decile_mean", "ts_bicoherence_top_decile_excess"])
def test_public_research_backends_and_prefix(canonical):
    import pandas as pd
    import polars as pl
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    ensure_cleaned_loaded()
    x = pd.DataFrame(np.random.default_rng(7).normal(size=(70, 1)) * 1e-6, columns=["A"])
    pandas_op = OperatorRegistry.get(canonical, "pandas_numpy", mode="research")
    polars_op = OperatorRegistry.get(canonical, "polars", mode="research")
    assert pandas_op is not None and polars_op is not None
    result = pandas_op.calculate(x, window=64, n_segments=4)
    pd.testing.assert_frame_equal(
        result.iloc[:67], pandas_op.calculate(x.iloc[:67], window=64, n_segments=4)
    )
    assert np.isfinite(result.to_numpy()).any()
    actual = polars_op.calculate(pl.from_pandas(x), window=64, n_segments=4)
    np.testing.assert_allclose(actual.to_numpy(), result.to_numpy(), equal_nan=True)
