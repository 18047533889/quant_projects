import numpy as np
import pytest
from scipy import signal


def _kernel():
    from factor_engine.cleaned_operators.cross_spectrum import _cross_spectrum_window
    return _cross_spectrum_window


def _welch_reference(x, y, band):
    from factor_engine.cleaned_operators.cross_spectrum import _band_range
    size = max(16, len(x) // 2)
    options = dict(fs=1, window=np.hanning(size), nperseg=size,
                   noverlap=size - size // 2, detrend="linear", return_onesided=False)
    _, xx = signal.welch(x, **options)
    _, yy = signal.welch(y, **options)
    _, xy = signal.csd(y, x, **options)
    lo, hi = _band_range(band, size // 2)
    xx, yy, xy = (values[1:size // 2 + 1][lo:hi] for values in (xx, yy, xy))
    return np.sum(np.abs(xy) ** 2) / np.sum(xx * yy), np.angle(np.sum(xy))


@pytest.mark.parametrize("n", [24, 63, 64, 65])
@pytest.mark.parametrize("band", ["all", "long", "medium", "short"])
def test_welch_reference_and_unit_scaling(n, band):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    y = np.roll(x, 1) + 0.1 * rng.normal(size=n)
    expected = _welch_reference(x, y, band)
    np.testing.assert_allclose(_kernel()(x, y, band), expected, rtol=1e-10, atol=1e-12)
    for factor in (1e-300, 1e-6, 1e6, 1e300):
        np.testing.assert_allclose(_kernel()(x * factor, y * factor, band),
                                   expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("level", [0.0, 100.0, 1e-300, 1e300])
def test_zero_detrended_energy_has_neither_coherence_nor_phase(level):
    coherence, phase = _kernel()(np.full(64, level), np.full(64, level))
    assert np.isnan(coherence) and np.isnan(phase)


def test_opposite_scale_changes_phase_but_not_coherence():
    x = np.random.default_rng(321).normal(size=64)
    coherence, phase = _kernel()(x, x)
    flipped_coherence, flipped_phase = _kernel()(x, -x)
    assert coherence == pytest.approx(1.0)
    assert flipped_coherence == pytest.approx(coherence)
    assert phase == pytest.approx(0.0)
    assert abs(flipped_phase) == pytest.approx(np.pi)


def test_low_coherence_gates_phase():
    rng = np.random.default_rng(90)
    x, y = rng.normal(size=(2, 64))
    coherence, phase = _kernel()(x, y, min_coherence=1.0)
    assert 0 <= coherence < 1
    assert np.isnan(phase)


def test_cancelling_coherent_phase_vectors_are_undefined(monkeypatch):
    # Isolate circular aggregation using exact equal opposite spectral vectors.
    # This is not a replacement for the real Welch/FFT comparisons above.
    calls = 0
    def spectrum(values):
        nonlocal calls
        result = np.zeros(len(values) // 2 + 1, dtype=complex)
        result[1:3] = [1, 1 if calls % 2 == 0 else -1]
        calls += 1
        return result
    monkeypatch.setattr(np.fft, "rfft", spectrum)
    x = np.arange(64, dtype=float)
    coherence, phase = _kernel()(x, x)
    assert coherence == 1.0
    assert np.isnan(phase)


@pytest.mark.parametrize("canonical", ["ts_cross_spectral_coherence", "ts_cross_spectral_phase"])
def test_public_research_backends_and_prefix(canonical):
    import pandas as pd
    import polars as pl
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    ensure_cleaned_loaded()
    values = np.random.default_rng(99).normal(size=(70, 2)) * 1e-6
    values[32, 0] = np.nan
    x = pd.DataFrame(values, columns=["A", "B"])
    y = x.shift(1)
    pandas_op = OperatorRegistry.get(canonical, "pandas_numpy", mode="research")
    polars_op = OperatorRegistry.get(canonical, "polars", mode="research")
    assert pandas_op is not None and polars_op is not None
    result = pandas_op.calculate(x, y, window=32)
    pd.testing.assert_frame_equal(
        result.iloc[:60], pandas_op.calculate(x.iloc[:60], y.iloc[:60], window=32)
    )
    assert np.isfinite(result.to_numpy()).any()
    actual = polars_op.calculate(pl.from_pandas(x), pl.from_pandas(y), window=32)
    np.testing.assert_allclose(actual.to_numpy(), result.to_numpy(), equal_nan=True)
