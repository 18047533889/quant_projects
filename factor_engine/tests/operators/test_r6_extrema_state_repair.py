import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


load_all()


def _frame(values):
    return pd.DataFrame({"A": np.asarray(values, dtype=float)}, index=pd.date_range("2024-01-01", periods=len(values)))


def test_extrema_identical_series_independent_oracle_and_defaults():
    values = np.tile([1.0, 3.0, 1.0, 2.5, 1.0, 4.0, 1.0], 8)
    x = _frame(values)
    div = OperatorRegistry.get("ts_extrema_divergence_strength", "pandas_numpy")
    rate = OperatorRegistry.get("ts_extrema_confirmation_rate", "pandas_numpy")
    d = div.calculate(x, x, window=40, prominence=0.1, confirmation=1, match_lag=0, side="peak")["A"]
    r = rate.calculate(x, x, window=40, prominence=0.1, confirmation=1, tolerance=0, side="peak")["A"]
    assert np.allclose(d.dropna(), 0.0)
    assert np.allclose(r.dropna(), 1.0)
    assert div.calculate(x, x).shape == x.shape
    assert rate.calculate(x, x).shape == x.shape


@pytest.mark.parametrize("name, knob", [
    ("ts_extrema_divergence_strength", {"match_lag": 1.5}),
    ("ts_extrema_confirmation_rate", {"tolerance": 1.5}),
])
def test_extrema_strict_parameters_and_relation(name, knob):
    x = _frame(np.arange(20.0))
    op = OperatorRegistry.get(name, "pandas_numpy")
    with pytest.raises(ValueError):
        op.calculate(x, x, **knob)
    with pytest.raises(ValueError):
        op.calculate(x, x, window=4, confirmation=2)


def test_state_density_hand_oracle_scale_invariance_and_gap_position():
    op = OperatorRegistry.get("ts_state_density", "pandas_numpy")
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 2.0])
    got = op.calculate(_frame(values), window=5, bandwidth=1.0, min_periods=5)["A"].iloc[-1]
    past = values[:5]
    scale = 1.4826 * np.median(np.abs(past - np.median(past)))
    u = (past - values[-1]) / scale
    expected = np.mean(0.75 * (1.0 - u * u) * (np.abs(u) <= 1.0))
    assert got == pytest.approx(expected)
    tiny = op.calculate(_frame(values * 1e-200), window=5, bandwidth=1.0, min_periods=5)["A"].iloc[-1]
    assert tiny == pytest.approx(expected)
    gapped = values.copy(); gapped[2] = np.nan
    assert np.isnan(op.calculate(_frame(gapped), window=5, bandwidth=1.0, min_periods=5)["A"].iloc[-1])


def test_state_density_defaults_and_strict_relation():
    op = OperatorRegistry.get("ts_state_density", "pandas_numpy")
    x = _frame(np.linspace(1.0, 3.0, 80) + 0.1 * np.sin(np.arange(80)))
    assert op.calculate(x).shape == x.shape
    with pytest.raises(ValueError):
        op.calculate(x, window=10, min_periods=11)
    with pytest.raises(ValueError):
        op.calculate(x, bandwidth=0.0)
