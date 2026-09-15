"""Marked-event independent event-index oracles through fresh final registry."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.operator_snapshot import _parameter_contract

AUTO = "event_mark_autocorr"
COUPLING = "event_interval_mark_coupling"

@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()

def panels():
    event = np.zeros(45)
    positions = np.array([0, 2, 5, 9, 14, 20, 27, 35, 44])
    event[positions] = 1
    mark = np.zeros(45)
    mark[positions] = [1, 3, 2, 8, 4, 9, 12, 7, 15]
    index = pd.date_range("2025-01-01", periods=45, tz="Asia/Hong_Kong")
    return [pd.DataFrame({"A": v}, index=index) for v in (event, mark)]

def run(name, backend, frames, **kwargs):
    op = OperatorRegistry.get(name, backend, mode="research")
    assert op.metadata.panel_params == ("event", "mark")
    assert _parameter_contract(op, op.metadata.panel_params)[2]
    if backend == "polars":
        frames = [pl.from_pandas(v.rename_axis("date").reset_index()) for v in frames]
    result = op.calculate(event=frames[0], mark=frames[1], **kwargs)
    return result.to_pandas().set_index("date").rename_axis(None) if backend == "polars" else result

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("name", [AUTO, COUPLING])
def test_oracle_scaling_defaults_and_prefix(name, backend):
    inputs = panels()
    marks = inputs[1].A[inputs[0].A == 1].to_numpy()
    expected = np.corrcoef(marks[1:], marks[:-1])[0, 1] if name == AUTO else np.corrcoef(np.arange(2, 10), marks[1:])[0, 1]
    out = run(name, backend, inputs)
    assert out.iloc[-1, 0] == pytest.approx(expected, abs=1e-12)
    defaults = dict(history_window=252, event_lag=1, mark_missing_policy="censor") if name == AUTO else dict(window=252, max_boundary_extension=5, mark_missing_policy="censor")
    np.testing.assert_allclose(out, run(name, backend, inputs, **defaults), equal_nan=True)
    for scale in (1e-200, 1e200):
        scaled = [inputs[0], inputs[1] * scale]
        np.testing.assert_allclose(out, run(name, backend, scaled), equal_nan=True, atol=1e-12)
    np.testing.assert_allclose(out.iloc[:36], run(name, backend, [v.iloc[:36] for v in inputs]), equal_nan=True)
    if backend == "polars":
        spec = OperatorRegistry.get(name, backend, mode="research").physical_spec()
        assert spec.execution_kind.value == "polars_pandas_delegate"
        assert spec.implementation_source_hash

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_invalid_panels_controls_and_unknown_intervals(backend):
    frames = panels()
    for name in (AUTO, COUPLING):
        window = "history_window" if name == AUTO else "window"
        for bad in (True, 6.5, 0):
            with pytest.raises((TypeError, ValueError)):
                run(name, backend, frames, **{window: bad})
        for bad in (np.inf, 0.5):
            invalid = [v.copy() for v in frames]
            invalid[0].iloc[1] = bad
            with pytest.raises((TypeError, ValueError)):
                run(name, backend, invalid)
        with pytest.raises((TypeError, ValueError)):
            run(name, backend, [frames[0], frames[1].rename(columns={"A": "B"})])
    with pytest.raises((TypeError, ValueError)):
        run(AUTO, backend, frames, history_window=6, event_lag=2)
    missing = [v.copy() for v in frames]
    missing[1].iloc[-1] = np.nan
    assert np.isnan(run(AUTO, backend, missing).iloc[-1, 0])
    # Only the last two intervals survive the unknown-event gap.
    missing = [v.copy() for v in frames]
    missing[0].iloc[26] = np.nan
    assert np.isnan(run(COUPLING, backend, missing).iloc[-1, 0])
    for policy in ("censor", "drop"):
        assert np.isnan(run(AUTO, backend, missing, mark_missing_policy=policy).iloc[-1, 0])
