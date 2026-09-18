"""A residual is undefined when its current observed pair is non-finite."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("field", ["x", "y"])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_current_nonfinite_pair_is_missing_and_later_finite_pairs_recover(backend, field, bad):
    load_all()
    x = np.linspace(-1, 3, 25)
    y = 7.25 - 1.4 * x + 0.18 * x**2
    (x if field == "x" else y)[15] = bad
    time = pd.date_range("2025-01-01", periods=len(x))
    if backend == "polars":
        frames = [pl.DataFrame({"timestamp": time, "A": v}) for v in (y, x)]
    else:
        frames = [pd.DataFrame({"A": v}, index=time) for v in (y, x)]
    out = OperatorRegistry.get("ts_poly2_resid", backend, mode="any").calculate(*frames, d=8)
    values = out["A"].to_numpy()
    assert np.isnan(values[15]), "current non-finite observation cannot define a residual"
    assert not np.isinf(values).any()
    np.testing.assert_allclose(values[16:], 0.0, atol=1e-10)
