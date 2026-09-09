"""Separate digital-frequency and numerical-representation Bessel oracles."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import besselap, lp2lp_zpk, bilinear_zpk, zpk2sos, sosfilt, sosfreqz

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.technical import frequency_filters


def reference_sos(order, cutoff, *, prewarp=True):
    z, p, k = besselap(order, norm="phase")
    frequency = 2 * np.tan(np.pi * cutoff) if prewarp else 2 * np.pi * cutoff
    z, p, k = lp2lp_zpk(z, p, k, wo=frequency)
    return zpk2sos(*bilinear_zpk(z, p, k, fs=1.0))


@pytest.fixture(scope="module")
def op():
    load_all()
    result = OperatorRegistry.get("ts_bessel_lowpass_causal", backend="pandas_numpy", mode="research")
    assert type(result).__module__ == "factor_engine.cleaned_operators.technical.frequency_filters"
    return result


@pytest.mark.parametrize("cutoff", [.0001, .001, .05, .4, .499])
def test_first_order_digital_cutoff_has_analytic_half_power(cutoff):
    _, response = sosfreqz(frequency_filters._bessel_sos(1, cutoff), worN=[cutoff], fs=1.)
    assert abs(response[0]) == pytest.approx(1 / np.sqrt(2), rel=1e-10)


@pytest.mark.parametrize("order", range(1, 9))
@pytest.mark.parametrize("cutoff", [.0001, .001, .05, .4, .499])
def test_whole_declared_domain_boundary_grid_matches_zpk_reference(op, order, cutoff):
    x = np.ones(5000)
    if cutoff == .0001:
        x = np.ones(30000)
    reference = sosfilt(reference_sos(order, cutoff), x)
    actual = op.calculate(pd.DataFrame({"A": x}), order=order, cutoff=cutoff).A.to_numpy()
    assert np.isnan(actual[:order]).all()
    assert np.isfinite(actual[order:]).all()
    np.testing.assert_allclose(actual[order:], reference[order:], rtol=1e-10, atol=1e-10)
    assert np.max(np.abs(actual[order:])) < 2


def test_unwarped_same_model_control_is_stable_without_frequency_fix():
    # Isolate M29: changing BA representation, not M28's digital frequency.
    out = sosfilt(reference_sos(8, .001, prewarp=False), np.ones(5000))
    assert np.isfinite(out).all() and np.max(np.abs(out)) < 2
    assert out[-1] == pytest.approx(1., abs=1e-5)


@pytest.mark.parametrize("signal", ["impulse", "step", "noise"])
def test_public_prefix_and_sos_state_chunks(op, signal):
    rng = np.random.default_rng(932)
    x = rng.normal(size=1000) if signal == "noise" else np.zeros(1000)
    if signal == "impulse":
        x[0] = 1
    elif signal == "step":
        x[100:] = 1
    frame = pd.DataFrame({"A": x})
    full = op.calculate(frame, order=8, cutoff=.001)
    pd.testing.assert_frame_equal(op.calculate(frame.iloc[:417], order=8, cutoff=.001), full.iloc[:417])
    sos = frequency_filters._bessel_sos(8, .001)
    left, state = sosfilt(sos, x[:417], zi=np.zeros((len(sos), 2)))
    right, _ = sosfilt(sos, x[417:], zi=state)
    combined = np.concatenate([left, right])
    np.testing.assert_allclose(full.A.iloc[8:], combined[8:], rtol=1e-12, atol=1e-12)
    # This proves SOS state composition only, not runtime checkpoint wiring.
