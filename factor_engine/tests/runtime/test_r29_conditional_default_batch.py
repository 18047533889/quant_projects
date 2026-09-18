"""Conditional rolling defaults through actual shared-DAG execution."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource
from tests.runtime.test_r23_extreme_numeric_batch import _coherent_broker


@pytest.mark.parametrize("backend", ["pandas", "polars_long", "auto"])
def test_conditional_defaults_through_run_many(backend):
    dates = pd.date_range("2026-01-01", periods=32)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    values = np.sin(np.arange(32) / 2)
    source = InMemorySeriesSource(data={"x": pd.Series(values, index=index)})
    source.instrument_filter = ("A",)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {"x": "float64"}
    engine = FactorEngine(build_backend(backend), source, run_mode="research")
    engine.resource_broker = _coherent_broker()
    condition = F("gt")(col("x"), 0)
    factors = [Factor(name="ts_count_if", expr=F("ts_count_if")(condition))]
    for name in ("ts_sum_if", "ts_mean_if", "ts_std_if", "ts_last_if"):
        factors.append(Factor(name=name, expr=F(name)(col("x"), condition)))
    result = engine.run_many(factors)
    expected = {f.name: np.full(32, np.nan) for f in factors}
    for row in range(32):
        window = values[max(0, row - 19):row + 1]
        selected = window[window > 0]
        expected["ts_count_if"][row] = len(selected)
        if len(selected):
            expected["ts_sum_if"][row] = selected.sum()
            expected["ts_mean_if"][row] = selected.mean()
            expected["ts_last_if"][row] = selected[-1]
        if len(selected) >= 2:
            expected["ts_std_if"][row] = selected.std(ddof=1)
    for name, values in expected.items():
        pd.testing.assert_series_equal(result["results"][name], pd.Series(values, index=index),
                                       check_names=False, rtol=1e-12, atol=1e-12)
    assert result["dag"].shared_nodes
