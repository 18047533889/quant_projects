"""run_many must not change top-K semantics when defaults are omitted."""
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
def test_topk_sum_default_and_explicit_shared_batch(backend):
    dates = pd.date_range("2026-01-01", periods=32)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    values = 1.0 + np.arange(32) * 0.001
    source = InMemorySeriesSource(data={"x": pd.Series(values, index=index)})
    source.instrument_filter = ("A",)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {"x": "float64"}
    engine = FactorEngine(build_backend(backend), source, run_mode="research")
    engine.resource_broker = _coherent_broker()
    result = engine.run_many([
        Factor(name="omitted", expr=F("ts_topk_sum")(col("x"))),
        Factor(name="explicit", expr=F("ts_topk_sum")(col("x"), d=20, k=None)),
        Factor(name="five", expr=F("ts_topk_sum")(col("x"), d=20, k=5)),
    ])["results"]
    expected = pd.Series(values, index=index).rolling(20, min_periods=20).sum()
    for name in ("omitted", "explicit"):
        pd.testing.assert_series_equal(result[name], expected, check_names=False, rtol=1e-12)
    expected_five = pd.Series(values, index=index).rolling(20, min_periods=5).apply(
        lambda a: np.sort(a)[-5:].sum(), raw=True
    )
    pd.testing.assert_series_equal(result["five"], expected_five, check_names=False, rtol=1e-12)
