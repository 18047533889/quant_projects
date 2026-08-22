# -*- coding: utf-8 -*-
"""Focused pandas/Polars-long parity for canonical EWM standard deviation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _source(values_a: list[float], values_b: list[float]) -> InMemorySeriesSource:
    load_all()
    dates = pd.date_range("2024-01-02", periods=len(values_a), freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["timestamp", "instrument"]
    )
    values = np.asarray(list(zip(values_a, values_b)), dtype=float).ravel()
    return InMemorySeriesSource(data={"close": pd.Series(values, index=index)})


def _run(source: InMemorySeriesSource, backend_name: str) -> dict:
    expr = make_cleaned_call_factory("ewm_std")(col("close"), 3)
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="ewm_std_nan_hole", expr=expr))


@pytest.mark.parametrize(
    ("values_a", "values_b"),
    [
        (
            [1.0, 2.0, np.nan, 8.0, 16.0, np.nan, 64.0, 128.0],
            [100.0, 100.0, np.nan, 100.0, 100.0, np.nan, 100.0, 100.0],
        ),
        (
            [np.nan, 1.0, 2.0, 4.0, np.nan, 8.0, 16.0, 32.0],
            [10.0, np.nan, 10.0, 10.0, 10.0, np.nan, 10.0, 10.0],
        ),
        (
            [1.0, 2.0, np.inf, 8.0, 16.0, -np.inf, 64.0, 128.0],
            [100.0, 100.0, np.inf, 100.0, 100.0, -np.inf, 100.0, 100.0],
        ),
    ],
)
def test_ewm_std_pandas_polars_long_parity_through_nonfinite_holes(
    values_a: list[float], values_b: list[float]
) -> None:
    """EWM std treats NaN and infinities as missing without cross-instrument state."""
    source = _source(values_a, values_b)
    pandas_out = _run(source, "pandas")["result"].sort_index()
    polars_run = _run(source, "polars_long")

    assert polars_run.get("used_polars_long_path") is True
    pd.testing.assert_series_equal(
        pandas_out,
        polars_run["result"].sort_index(),
        check_names=False,
        rtol=1e-9,
        atol=1e-10,
    )
