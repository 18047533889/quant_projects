"""Independent time/support oracles for the actual registered FIR binding."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import firwin, lfilter

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.runtime.execution_contract import own_history_requirement


@pytest.fixture(scope="module")
def fir():
    load_all()
    op = OperatorRegistry.get("ts_fir_lowpass_causal", backend="pandas_numpy", mode="research")
    assert type(op).__module__ == "factor_engine.cleaned_operators.technical.frequency_filters"
    return op


@pytest.mark.parametrize("ntaps", [3, 7, 21])
def test_fir_matches_one_sided_reference_and_exact_first_valid_row(fir, ntaps):
    rng = np.random.default_rng(723)
    x = pd.DataFrame(rng.normal(size=(70, 2)), columns=["A", "B"])
    h = firwin(ntaps, .1, window="hamming", fs=1)
    expected = lfilter(h, [1.], x.to_numpy(), axis=0)
    expected[:ntaps - 1] = np.nan
    actual = fir.calculate(x, ntaps=ntaps, cutoff=.1)
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=1e-14, equal_nan=True)
    assert actual.iloc[ntaps - 1].notna().all()


def test_future_poison_and_every_prefix_preserve_acceptance_mask_and_value(fir):
    x = pd.DataFrame({"A": np.full(65, 100.), "B": np.linspace(-2, 2, 65)})
    baseline = fir.calculate(x, ntaps=21)
    poisoned = x.copy()
    poisoned.iloc[40:] += 1_000
    changed = fir.calculate(poisoned, ntaps=21)
    pd.testing.assert_frame_equal(baseline.iloc[:40], changed.iloc[:40])
    for stop in range(1, len(x) + 1):
        prefix = fir.calculate(x.iloc[:stop], ntaps=21)
        pd.testing.assert_frame_equal(prefix, baseline.iloc[:stop])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_invalidates_entire_filter_support_not_just_current(fir, missing):
    x = pd.DataFrame({"A": np.ones(35)})
    x.iloc[12, 0] = missing
    out = fir.calculate(x, ntaps=7)
    expected_valid = np.arange(len(x)) >= 6
    expected_valid[12:19] = False
    np.testing.assert_array_equal(out.A.notna(), expected_valid)
    np.testing.assert_allclose(out.A.dropna(), 1., atol=1e-14, rtol=1e-14)


def test_history_authority_covers_taps_and_exact_warmup_chunk_matches(fir):
    x = pd.DataFrame({"A": np.sin(np.arange(80) / 3.)})
    req = own_history_requirement("ts_fir_lowpass_causal", {"ntaps": 21})
    assert not req.is_full_history and req.rows == 20
    full = fir.calculate(x, ntaps=21)
    chunk = fir.calculate(x.iloc[50 - req.rows:], ntaps=21)
    pd.testing.assert_frame_equal(full.iloc[50:], chunk.loc[50:])


@pytest.mark.parametrize("ntaps", [2, 4, 20, 202, np.int64(0)])
def test_invalid_taps_reject_before_kernel_no_silent_effective_parameter(fir, ntaps, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid declared tap domain reached kernel")
    monkeypatch.setattr(fir, "_calculate_series", forbidden)
    with pytest.raises(OperatorParameterError):
        fir.calculate(pd.DataFrame({"A": [1., 2., 3.]}), ntaps=ntaps)


def test_declared_integral_float_normalization_remains_distinct_from_runtime_gate(fir):
    x = pd.DataFrame({"A": np.arange(40.)})
    pd.testing.assert_frame_equal(fir.calculate(x, ntaps=7.0), fir.calculate(x, ntaps=7))
    with pytest.raises(OperatorParameterError):
        fir._calculate_series(x, ntaps=7.0)


@pytest.mark.parametrize("name,kwargs,message", [
    ("ts_causal_savgol_endpoint", {"window": 3, "polyorder": 5}, "polyorder must be smaller"),
    ("ts_spectral_lowpass_trailing", {"window": 8, "cutoff_freq": 256}, "cutoff_freq exceeds"),
])
def test_filter_joint_domains_reject_before_kernel(name, kwargs, message, monkeypatch):
    load_all()
    op = OperatorRegistry.get(name, backend="pandas_numpy", mode="research")
    def forbidden(*args, **kwargs):
        pytest.fail("infeasible joint filter domain reached kernel")
    monkeypatch.setattr(op, "_calculate_series", forbidden)
    with pytest.raises(ValueError, match=message):
        op.calculate(pd.DataFrame({"A": np.arange(40.)}), **kwargs)
