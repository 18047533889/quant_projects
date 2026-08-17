"""R2-P0-026 rolling variance parameter ABI regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry

load_all()

PANDAS = OperatorRegistry.get("ts_var", backend="pandas_numpy")
POLARS = OperatorRegistry.get("ts_var", backend="polars")


def test_ts_var_authorities_share_d_window_min_periods_ddof_abi() -> None:
    assert PANDAS is not None
    assert POLARS is not None
    expected = ["x", "window", "ddof", "min_periods"]
    assert PANDAS.metadata.param_names == expected
    assert POLARS.metadata.param_names == expected


def test_ts_var_polars_matches_pandas_for_explicit_window_and_ddof() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 8.0])
    kwargs = {"window": 3, "ddof": 0, "min_periods": 2}
    pandas_actual = PANDAS.calculate(pd.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    polars_actual = POLARS.calculate(pl.DataFrame({"x": values}), **kwargs)["x"].to_numpy()
    np.testing.assert_allclose(polars_actual, pandas_actual, equal_nan=True)


def test_ts_var_d_is_rejected_instead_of_overriding_window() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    canonical = POLARS.calculate(
        pl.DataFrame({"x": values}), window=3, ddof=1, min_periods=2
    )["x"].to_numpy()
    with_d = POLARS.calculate(
        pl.DataFrame({"x": values}), window=3, d=2, ddof=1, min_periods=2
    )["x"].to_numpy()
    np.testing.assert_array_equal(with_d, canonical)
