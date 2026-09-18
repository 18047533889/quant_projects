"""Small actual DAG execution for corrected causal AR kernels."""
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
from tests.runtime.test_r32_ar_family_contract import oracle


@pytest.mark.parametrize("backend", ["pandas", "polars_long", "auto"])
def test_causal_ar_models_through_real_run_many(backend):
    rng = np.random.default_rng(331)
    values = np.zeros((36, 3))
    for row in range(2, len(values)):
        values[row] = 0.2 + 0.4 * values[row - 1] - 0.1 * values[row - 2] + rng.normal(0, 0.1, 3)
    values[17, 0] = np.nan
    dates = pd.date_range("2026-01-01", periods=len(values))
    assets = ["A", "B", "C"]
    index = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])
    source = InMemorySeriesSource(data={"x": pd.Series(values.ravel(), index=index)})
    source.instrument_filter = tuple(assets)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {"x": "float64"}
    engine = FactorEngine(build_backend(backend), source, run_mode="research")
    engine.resource_broker = _coherent_broker()
    params = dict(window=12, order=2, warmup_policy="expanding")
    names = {
        "ts_ar_fitted_value": ("forecast", 0),
        "ts_ar_prior_forecast": ("forecast", 1),
        "ts_ar_prior_innovation": ("innovation", 1),
    }
    shared = F("ts_ar_prior_forecast")(col("x"), **params)
    factors = [Factor(name=name, expr=F(name)(col("x"), **params)) for name in names]
    factors.append(Factor(name="absolute_forecast", expr=F("abs")(shared)))
    result = engine.run_many(factors)
    for name, (stat, lag) in names.items():
        expected = np.column_stack([
            oracle(values[:, c], 12, 2, "expanding", stat, lag, 0)
            for c in range(values.shape[1])
        ])
        actual = result["results"][name].reindex(index).to_numpy().reshape(values.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
        if name == "ts_ar_prior_forecast":
            absolute = result["results"]["absolute_forecast"].reindex(index).to_numpy().reshape(values.shape)
            np.testing.assert_allclose(absolute, np.abs(expected), rtol=1e-10, atol=1e-10, equal_nan=True)
    assert result["dag"].shared_nodes
