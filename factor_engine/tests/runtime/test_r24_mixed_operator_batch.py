"""Small real run_many integration for repaired operators and shared roots."""
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
from tests.backend_parity.test_r24_rsi_gap_oracle import _oracle


@pytest.mark.parametrize("backend", ["pandas", "polars_long", "auto"])
def test_run_many_mixed_group_and_recursive_shared_roots(backend):
    dates = pd.date_range("2026-01-01", periods=32)
    assets = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, assets], names=["timestamp", "instrument"])
    t = np.arange(32, dtype=float)
    prices = 100 + t[:, None] + np.sin(t[:, None] + np.arange(4))
    prices[[11, 12], 0] = np.nan
    x = np.tile([1.0, 3.0, 5.0, 9.0], (32, 1))
    x[8, 0], x[15, 2], x[20, 1] = np.inf, -np.inf, np.nan
    groups = np.tile([0.0, 0.0, 1.0, 1.0], (32, 1))
    groups[23, 0] = np.nan
    source = InMemorySeriesSource(data={
        "close": pd.Series(prices.ravel(), index=index),
        "x": pd.Series(x.ravel(), index=index),
        "group_id": pd.Series(groups.ravel(), index=index),
    })
    source.instrument_filter = tuple(assets)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {name: "float64" for name in source.data}
    engine = FactorEngine(backend=build_backend(backend), data_source=source, run_mode="research")
    engine.resource_broker = _coherent_broker()
    group = F("group_mean")(col("x"), col("group_id"))
    rsi = F("RSI_WILDER")(col("close"), 3)
    factors = [
        Factor(name="group", expr=group),
        Factor(name="group_abs", expr=F("abs")(group)),
        Factor(name="rsi", expr=rsi),
        Factor(name="rsi_abs", expr=F("abs")(rsi)),
    ]
    out = engine.run_many(factors)
    expected_group = np.full_like(x, np.nan)
    for i in range(len(dates)):
        valid = np.isfinite(x[i]) & np.isfinite(groups[i])
        for label in np.unique(groups[i, valid]):
            mask = valid & (groups[i] == label)
            expected_group[i, mask] = np.mean(x[i, mask])
    expected_rsi = np.column_stack([_oracle(prices[:, j], 3) for j in range(4)])
    for name, expected in (
        ("group", expected_group), ("group_abs", np.abs(expected_group)),
        ("rsi", expected_rsi), ("rsi_abs", np.abs(expected_rsi)),
    ):
        pd.testing.assert_series_equal(
            out["results"][name].sort_index(),
            pd.Series(expected.ravel(), index=index).sort_index(),
            check_names=False, check_dtype=False, rtol=1e-11, atol=1e-11,
        )
    assert out["dag"].shared_nodes
