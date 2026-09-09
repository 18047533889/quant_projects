import numpy as np
import pandas as pd
import pytest
from scipy import signal


@pytest.fixture(scope="module", autouse=True)
def _load():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    ensure_cleaned_loaded()


@pytest.mark.parametrize("period", [3, 20, 60])
@pytest.mark.parametrize("order", [1, 2, 4, 10])
def test_actual_filter_design_matches_digital_butterworth_magnitude(monkeypatch, period, order):
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    original = signal.butter
    designed = []
    def capture(*args, **kwargs):
        sos = original(*args, **kwargs)
        designed.append(sos)
        return sos
    monkeypatch.setattr(signal, "butter", capture)
    op = OperatorRegistry.get("ts_butterworth_lowpass_causal", "pandas_numpy", mode="research")
    op.calculate(pd.DataFrame({"A": [1.0, 2.0, 3.0]}), cutoff_period=period, order=order)
    assert len(designed) == 1
    cutoff = 1.0 / period
    frequencies = np.array([cutoff / 4, cutoff / 2, cutoff, min(0.49, cutoff * 1.4)])
    _, response = signal.sosfreqz(designed[0], worN=frequencies, fs=1.0)
    # Analytic bilinear-transform Butterworth magnitude, independent of design.
    expected = 1.0 / np.sqrt(1.0 + (np.tan(np.pi * frequencies) / np.tan(np.pi * cutoff)) ** (2 * order))
    np.testing.assert_allclose(np.abs(response), expected, rtol=2e-10, atol=2e-12)
    assert abs(response[2]) == pytest.approx(1 / np.sqrt(2), rel=2e-10)


def test_public_gap_freeze_prefix_and_polars_parity():
    import polars as pl
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    rng = np.random.default_rng(27)
    values = rng.normal(size=(80, 2)) + 100
    values[20:25, 0] = np.nan
    x = pd.DataFrame(values, columns=["A", "B"])
    op = OperatorRegistry.get("ts_butterworth_lowpass_causal", "pandas_numpy", mode="research")
    actual = op.calculate(x)
    pd.testing.assert_frame_equal(actual.iloc[:50], op.calculate(x.iloc[:50]))
    compressed = op.calculate(x[["A"]].dropna())
    np.testing.assert_allclose(actual["A"].dropna(), compressed["A"], rtol=0, atol=0)
    polars_op = OperatorRegistry.get("ts_butterworth_lowpass_causal", "polars", mode="research")
    assert polars_op is not None
    np.testing.assert_allclose(polars_op.calculate(pl.from_pandas(x)).to_numpy(), actual,
                               equal_nan=True)
    constant = op.calculate(pd.DataFrame({"A": np.full(50, 100.0)}))
    np.testing.assert_allclose(constant, 100.0, rtol=1e-12)


def test_butterworth_requires_full_replay_without_checkpoint_adapter():
    from factor_engine.runtime.execution_contract import execution_contract, own_history_requirement
    contract = execution_contract("ts_butterworth_lowpass_causal")
    assert contract.state_model == "recursive"
    assert contract.chunking == "required_full_history"
    assert own_history_requirement("ts_butterworth_lowpass_causal", {"cutoff_period":20, "order":2}).is_full_history
    from factor_engine.cleaned_operators.filter_smooth import (
        ButterworthLowpassCausalOperator, filter_smooth_contract,
    )
    from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
    for surface in (ButterworthLowpassCausalOperator.filter_contract,
                    filter_smooth_contract("ts_butterworth_lowpass_causal")):
        assert surface.stateful and not surface.checkpointable
        assert not surface.time_shard_safe
        assert surface.lag_class == "variable"
    assert "checkpointable" not in infer_operator_policy("ts_butterworth_lowpass_causal").tags
