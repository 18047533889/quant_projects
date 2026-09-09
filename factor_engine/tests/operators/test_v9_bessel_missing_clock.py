import numpy as np
import pandas as pd
import pytest
from scipy import signal


@pytest.fixture(scope="module")
def op():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    ensure_cleaned_loaded()
    return OperatorRegistry.get("ts_bessel_lowpass_causal", "pandas_numpy", mode="research")


@pytest.mark.parametrize("order", [1, 4, 8])
def test_event_clock_against_explicit_sos_state_oracle(op, order):
    values = np.full((80, 2), 100.0)
    values[:7, 0] = np.nan
    values[30:35, 0] = np.nan
    values[-3:, 0] = np.nan
    values[21, 1] = np.inf
    values[31, 1] = 0.0
    sos = signal.bessel(order, 0.05, fs=1.0, norm="phase", output="sos")
    expected = np.full_like(values, np.nan)
    for col in range(2):
        state = np.zeros((len(sos), 2))
        seen = 0
        for row, value in enumerate(values[:, col]):
            if not np.isfinite(value):
                continue
            current, state = signal.sosfilt(sos, [value], zi=state)
            if seen >= order:
                expected[row, col] = current[0]
            seen += 1
    actual = op.calculate(pd.DataFrame(values, columns=["A", "B"]), order=order)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(actual.iloc[:7 + order, 0]).all()


def test_gap_is_not_observed_zero_and_no_future_fill(op):
    gap = pd.DataFrame({"A": np.full(100, 100.0)})
    gap.iloc[50] = np.nan
    zero = gap.fillna(0.0)
    actual = op.calculate(gap)
    observed_zero = op.calculate(zero)
    assert np.isnan(actual.iloc[50, 0])
    assert np.isfinite(observed_zero.iloc[50, 0])
    assert abs(actual.iloc[55, 0] - observed_zero.iloc[55, 0]) > 1.0
    pd.testing.assert_frame_equal(actual.iloc[:60], op.calculate(gap.iloc[:60]))
    compressed = op.calculate(gap.dropna())
    pd.testing.assert_frame_equal(actual.drop(index=50), compressed)


def test_replayed_partitions_preserve_masks_and_state(op):
    values = np.random.default_rng(30).normal(size=(95, 2))
    values[18:30, 0] = np.nan
    values[65:72, 1] = np.nan
    frame = pd.DataFrame(values)
    full = op.calculate(frame)
    # The supported contract replays the prefix. This is not saved-state
    # checkpoint evidence and deliberately does not use a finite warmup.
    pieces = [op.calculate(frame.iloc[:end]).iloc[start:end]
              for start, end in [(0, 23), (23, 51), (51, 70), (70, 95)]]
    pd.testing.assert_frame_equal(pd.concat(pieces), full)
    from factor_engine.runtime.execution_contract import execution_contract
    assert execution_contract("ts_bessel_lowpass_causal").chunking == "required_full_history"
