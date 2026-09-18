"""R29 regression for daily overhaul kernels whose declared defaults were unusable."""
import inspect

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


CANONICALS = (
    "ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if",
    "ts_tail_mean", "ts_argmax", "ts_argmin",
)


def _array(value):
    return value.to_numpy() if isinstance(value, pl.DataFrame) else value.to_numpy(dtype=float)


def test_declared_window_default_is_callable_and_backend_equal():
    load_all()
    x = pd.DataFrame({"A": np.linspace(-2.0, 3.0, 25), "B": np.sin(np.arange(25) / 3)})
    condition = x > 0
    for name in CANONICALS:
        arrays = {}
        for backend in ("pandas_numpy", "polars"):
            op = OperatorRegistry.get(name, backend, mode="research")
            assert op.metadata.param_specs["window"].default == 20
            args = (condition,) if name == "ts_count_if" else (
                (x, condition) if name.endswith("_if") else (x,)
            )
            if backend == "polars":
                args = tuple(pl.from_pandas(value) for value in args)
            omitted = _array(op.calculate(*args))
            explicit = _array(op.calculate(*args, window=20))
            np.testing.assert_allclose(omitted, explicit, rtol=1e-12, atol=1e-12, equal_nan=True)
            arrays[backend] = omitted
            signature = inspect.signature(op._fn)
            assert signature.parameters["window"].default == 20
        np.testing.assert_allclose(arrays["pandas_numpy"], arrays["polars"], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_tail_mean_default_uses_two_finite_rows_and_lower_decile_oracle():
    load_all()
    values = np.arange(1.0, 26.0)
    frame = pd.DataFrame({"A": values})
    expected = np.full((25, 1), np.nan)
    for row in range(1, 25):
        sample = values[max(0, row - 19) : row + 1]
        threshold = np.quantile(sample, 0.1)
        expected[row, 0] = np.mean(sample[sample <= threshold])
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("ts_tail_mean", backend, mode="research")
        assert op.metadata.param_specs["min_periods"].default == 2
        source = pl.from_pandas(frame) if backend == "polars" else frame
        actual = _array(op.calculate(source))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert np.isnan(actual[0, 0]) and np.isfinite(actual[1:, 0]).all()
