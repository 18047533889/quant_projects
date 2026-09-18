"""Declared defaults must execute consistently at the public operator boundary."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("name,defaults", [
    ("ts_topk_sum", {"d": 20, "k": None, "min_periods": 1}),
    ("digital_count", {"d": 20, "threshold": 0.01, "run": 3}),
    ("ts_max_buildup", {"d": 20}),
])
def test_omitted_defaults_match_declared_defaults(backend, name, defaults):
    load_all()
    dates = pd.date_range("2026-01-01", periods=32)
    values = 1.0 + np.arange(32) * 0.001
    frame = pd.DataFrame({"A": values}, index=dates)
    arg = frame if backend == "pandas_numpy" else pl.DataFrame({"timestamp": dates, "A": values})
    op = OperatorRegistry.get(name, backend, mode="research")
    for key, value in defaults.items():
        assert op.metadata.param_specs[key].default == value
    omitted = op.calculate(arg)
    explicit = op.calculate(arg, **defaults)
    a = omitted["A"].to_numpy()
    b = explicit["A"].to_numpy()
    np.testing.assert_allclose(a, b, equal_nan=True)
    if name == "ts_topk_sum":
        expected = frame["A"].rolling(20, min_periods=20).sum().to_numpy()
    elif name == "digital_count":
        expected = np.minimum(np.arange(32), 20).astype(float)
        expected[expected < 3] = 0.0
    else:
        expected = np.minimum(np.arange(32) + 1, 20).astype(float)
    np.testing.assert_allclose(a, expected, equal_nan=True, rtol=1e-12)
