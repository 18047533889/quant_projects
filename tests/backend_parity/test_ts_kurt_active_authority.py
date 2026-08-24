# -*- coding: utf-8 -*-
"""Focused pandas/Polars-long parity for the active ts_kurt authority."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _source(values_a: list[float], values_b: list[float]) -> InMemorySeriesSource:
    load_all()
    dates = pd.date_range("2024-01-02", periods=len(values_a), freq="B")
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["timestamp", "instrument"]
    )
    values = np.asarray(list(zip(values_a, values_b)), dtype=float).ravel()
    return InMemorySeriesSource(data={"close": pd.Series(values, index=index)})


def _run(
    source: InMemorySeriesSource, backend_name: str, window: int | None = None
) -> dict:
    factory = make_cleaned_call_factory("ts_kurt")
    expr = factory(col("close")) if window is None else factory(col("close"), window)
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="ts_kurt_active_authority", expr=expr))


@pytest.mark.parametrize("window", [4, 5])
def test_ts_kurt_pandas_polars_long_matches_stable_authority(window: int) -> None:
    source = _source(
        [
            1.0,
            2.0,
            3.0,
            np.nan,
            5.0,
            6.0,
            np.inf,
            8.0,
            9.0,
            10.0,
            -np.inf,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
        ],
        [7.0] * 17,
    )
    pandas_run = _run(source, "pandas", window)
    polars_run = _run(source, "polars_long", window)

    assert polars_run.get("used_polars_long_path") is True
    pandas_result = pandas_run["result"].sort_index()
    polars_result = polars_run["result"].sort_index()
    pd.testing.assert_series_equal(
        pandas_result,
        polars_result,
        check_names=False,
        rtol=1e-9,
        atol=1e-10,
    )
    values_a = pandas_result.xs("A", level="instrument")
    clean_start = 14 if window == 4 else 15
    assert values_a.iloc[:clean_start].isna().all()
    assert values_a.iloc[clean_start:].notna().all()
    assert pandas_result.xs("B", level="instrument").isna().all()


def test_ts_kurt_default_window_and_known_oracle() -> None:
    source = _source(
        [float(value) for value in range(1, 23)],
        [7.0] * 22,
    )
    pandas_run = _run(source, "pandas")
    polars_run = _run(source, "polars_long")

    assert polars_run.get("used_polars_long_path") is True
    pandas_result = pandas_run["result"].sort_index()
    polars_result = polars_run["result"].sort_index()
    pd.testing.assert_series_equal(
        pandas_result,
        polars_result,
        check_names=False,
        rtol=1e-9,
        atol=1e-10,
    )
    values_a = pandas_result.xs("A", level="instrument")
    assert values_a.iloc[:19].isna().all()
    assert values_a.iloc[19] == pytest.approx(-1.2, abs=1e-12)
    assert pandas_result.xs("B", level="instrument").isna().all()


@pytest.mark.parametrize("window", [1, 2, 3])
@pytest.mark.parametrize("backend_name", ["pandas", "polars_long"])
def test_ts_kurt_rejects_window_below_four(
    backend_name: str, window: int
) -> None:
    source = _source(
        [float(value) for value in range(1, 7)],
        [7.0] * 6,
    )

    with pytest.raises(ValueError, match="ts_kurt window must be >= 4"):
        _run(source, backend_name, window)
