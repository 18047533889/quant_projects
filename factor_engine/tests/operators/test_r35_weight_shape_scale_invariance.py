"""Weight shares must not depend on their finite unit of measurement."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("canonical", ["ts_weighted_time_centroid", "ts_mass_concentration"])
@pytest.mark.parametrize("scale", [1.0, 1e308, 1e-308])
def test_weight_shape_matches_independent_scale_free_oracle(backend, canonical, scale):
    load_all()
    weights = np.array([0.125, 0.25, 0.5, 0.75, 1.0])
    shares = weights / weights.sum()
    if canonical == "ts_weighted_time_centroid":
        expected = 2.0 * np.dot(np.arange(5), shares) / 4.0 - 1.0
    else:
        expected = (np.dot(shares, shares) - 0.2) / 0.8
    frame = pd.DataFrame({"A": weights * scale})
    arg = frame if backend == "pandas_numpy" else pl.from_pandas(frame)
    result = OperatorRegistry.get(canonical, backend).calculate(arg, window=5, min_periods=5)
    assert result["A"].to_numpy()[-1] == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
@pytest.mark.parametrize("canonical", ["ts_weighted_time_centroid", "ts_mass_concentration"])
def test_invalid_weights_remain_undefined(backend, canonical):
    load_all()
    op = OperatorRegistry.get(canonical, backend)
    for values in ([0.0] * 5, [-0.1, 0.25, 0.5, 0.75, 1.0]):
        frame = pd.DataFrame({"A": values})
        arg = frame if backend == "pandas_numpy" else pl.from_pandas(frame)
        assert np.isnan(op.calculate(arg, window=5, min_periods=5)["A"].to_numpy()[-1])
