"""Small actual run_many tests for final backend routing, not just kernels."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.mark.parametrize("backend", ["pandas", "polars_long", "auto"])
def test_run_many_normalize_extrema_and_shared_subexpression(backend):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    values = pd.Series(np.tile([-1e308, 0., 1e308], 4), index=index)
    source = InMemorySeriesSource(data={"x": values})
    # Exact bounded fixture scope supplies the same shape evidence required
    # from real sources; auto must not invent row counts for unknown sources.
    source.instrument_filter = ("A", "B", "C")
    source.start_date = index.levels[0].min().tz_localize("UTC")
    source.end_date = index.levels[0].max().tz_localize("UTC")
    source.schema = {"x": "float64"}
    engine = FactorEngine(backend=build_backend(backend), data_source=source, run_mode="research")
    normalized = F("normalize")(col("x"))
    factors = [
        Factor(name="normalized", expr=normalized),
        Factor(name="normalized_abs", expr=F("abs")(normalized)),
    ]
    out = engine.run_many(factors)
    expected = pd.Series(np.tile([0., .5, 1.], 4), index=index)
    for name in ("normalized", "normalized_abs"):
        pd.testing.assert_series_equal(out["results"][name], expected,
                                       check_names=False, check_dtype=False)
    assert out["dag"].shared_nodes
