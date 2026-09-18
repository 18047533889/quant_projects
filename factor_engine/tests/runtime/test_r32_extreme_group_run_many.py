"""Actual shared-DAG batches for extreme finite-member reductions."""
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
def test_run_many_extreme_group_reductions(backend):
    dates = pd.date_range("2026-01-01", periods=4)
    assets = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])
    magnitudes = np.array([1e308, 1e-308, 1.0, 0.0])
    x = np.column_stack([magnitudes, -magnitudes, np.full(4, np.inf)])
    source = InMemorySeriesSource(data={
        "x": pd.Series(x.ravel(), index=idx),
        "group_id": pd.Series(1.0, index=idx),
    })
    source.instrument_filter = tuple(assets)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {name: "float64" for name in source.data}
    engine = FactorEngine(build_backend(backend), source, run_mode="research")
    engine.resource_broker = _coherent_broker()
    std = F("group_std")(col("x"), col("group_id"))
    factors = [
        Factor(name="std", expr=std),
        Factor(name="std_abs", expr=F("abs")(std)),
        Factor(name="scale", expr=F("scale")(col("x"))),
        Factor(name="demean", expr=F("group_neutralize")(col("x"), col("group_id"))),
    ]
    result = engine.run_many(factors)
    expected_std = np.column_stack([magnitudes * np.sqrt(2.0)] * 2 + [np.full(4, np.nan)])
    expected_scale = np.tile([0.5, -0.5, np.nan], (4, 1))
    expected_scale[-1, :2] = 0.0
    expected_demean = x.copy()
    expected_demean[:, 2] = np.nan
    normalizer = np.where(magnitudes == 0, 1.0, magnitudes)[:, None]
    for name, expected in [("std", expected_std), ("std_abs", expected_std),
                           ("scale", expected_scale), ("demean", expected_demean)]:
        got = result["results"][name].reindex(idx).to_numpy().reshape(4, 3)
        divisor = 1.0 if name == "scale" else normalizer
        np.testing.assert_allclose(got / divisor, expected / divisor,
                                   rtol=1e-12, atol=1e-14, equal_nan=True)
    assert result["dag"].shared_nodes
