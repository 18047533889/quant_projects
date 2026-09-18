"""EVT threshold ladders must be feasible and independent of numeric units."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

def frame(scale=1.0):
    values = np.exp(np.linspace(0., 4., 60)) * scale
    return pd.DataFrame({"A": values}, index=pd.date_range("2024-01-01", periods=60))

def invoke(data, backend, **kwargs):
    load_all()
    if backend == "polars":
        data = pl.from_pandas(data.rename_axis("timestamp").reset_index())
    return OperatorRegistry.get("ts_evt_threshold_stability", backend).calculate(data, **kwargs)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_two_thresholds_are_rejected_at_public_boundary(backend):
    with pytest.raises(ValueError, match="k_min|k_max|threshold"):
        invoke(frame(), backend, window=20, k_min=2, k_max=3)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
def test_three_thresholds_are_finite_and_scale_invariant(backend, scale):
    expected = invoke(frame(), "pandas_numpy", window=10, k_min=2, k_max=4)
    actual = invoke(frame(scale), backend, window=10, k_min=2, k_max=4)
    a = actual["A"].to_numpy()
    assert np.isfinite(a).any()
    np.testing.assert_allclose(a, expected["A"].to_numpy(), equal_nan=True, rtol=1e-12)

def test_hill_large_finite_ratio_does_not_overflow():
    from factor_engine.cleaned_operators.evt_allan import _hill_xi
    ordered = np.array([1e-300, 1e-250, 1e100, 1e200, 1e300])
    expected = np.mean(np.log(ordered[-3:]) - np.log(ordered[-4]))
    assert _hill_xi(ordered, 3) == pytest.approx(expected)

@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("side", ["upper", "lower"])
def test_threshold_stability_reference_prefix_and_gap_recovery(backend, side):
    data = frame()
    if side == "lower":
        data = -data
    data.iloc[20, 0] = np.nan
    data.iloc[40, 0] = np.inf
    result = invoke(data, backend, window=10, k_min=2, k_max=4, side=side)
    values = result["A"].to_numpy()
    assert np.isnan(values[20:30]).all()
    assert np.isnan(values[40:50]).all()
    for end in (19, 39, 59):
        tail = data["A"].iloc[end-9:end+1].to_numpy()
        ordered = np.sort(tail if side == "upper" else -tail)
        hills = [np.mean(np.log(ordered[-k:])-np.log(ordered[-k-1])) for k in (2,3,4)]
        assert values[end] == pytest.approx(1/(1+np.std(hills)))
    prefix = invoke(data.iloc[:36], backend, window=10, k_min=2, k_max=4, side=side)
    np.testing.assert_allclose(prefix["A"].to_numpy(), values[:36], equal_nan=True)
    if backend == "polars":
        assert result["timestamp"].to_list() == list(data.index.to_pydatetime())
    else:
        pd.testing.assert_index_equal(result.index, data.index)
