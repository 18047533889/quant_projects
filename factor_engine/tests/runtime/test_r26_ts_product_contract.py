"""R26 regression: one authoritative ts_product contract on every path."""
import numpy as np
import pandas as pd

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.ir.analyzer import Analyzer
from factor_engine.planner.lowerer import Lowerer
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.execution_contract import own_history_requirement
from tests.helpers import InMemorySeriesSource
from tests.runtime.test_r23_extreme_numeric_batch import _coherent_broker


def test_ts_product_contract_binding_and_history_are_canonical():
    load_all()
    expected_names = ["x", "window", "min_periods", "skipna"]
    for backend in ("pandas_numpy", "polars"):
        meta = OperatorRegistry.get("ts_product", backend, mode="research").metadata
        assert meta.param_names == expected_names
        assert meta.param_specs["window"].default == 20
        assert set(meta.param_specs) == {"window", "min_periods", "skipna"}

    default = Analyzer().lower(F("ts_product")(col("close")))
    explicit = Analyzer().lower(F("ts_product")(col("close"), 20))
    controlled = Analyzer().lower(
        F("ts_product")(col("close"), window=3, min_periods=2, skipna=False)
    )
    plan = Lowerer().to_logical_plan(controlled.ir)
    assert default.lookback == explicit.lookback == 19
    assert own_history_requirement("ts_product", {}).rows == 19
    assert plan.attrs["window"] == 3
    assert plan.attrs["min_periods"] == 2
    assert plan.attrs["skipna"] is False


def test_run_many_default_equals_explicit_20_and_skipna_false_propagates():
    load_all()
    dates = pd.date_range("2026-01-01", periods=25)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    values = np.linspace(1.001, 1.025, 25)
    source = InMemorySeriesSource(data={"close": pd.Series(values, index=index)})
    source.instrument_filter = ("A",)
    source.start_date = dates.min().tz_localize("UTC")
    source.end_date = dates.max().tz_localize("UTC")
    source.schema = {"close": "float64"}
    engine = FactorEngine(
        backend=build_backend("pandas"), data_source=source, run_mode="research"
    )
    engine.resource_broker = _coherent_broker()
    out = engine.run_many([
        Factor(name="default", expr=F("ts_product")(col("close"))),
        Factor(name="explicit", expr=F("ts_product")(col("close"), 20)),
    ])["results"]
    pd.testing.assert_series_equal(
        out["default"], out["explicit"], check_names=False, check_dtype=False
    )
    expected = pd.Series(np.nan, index=index, dtype=float)
    for row in range(19, len(values)):
        expected.iloc[row] = np.prod(values[row - 19 : row + 1])
    pd.testing.assert_series_equal(
        out["default"], expected, check_names=False, check_dtype=False,
        rtol=1e-12, atol=1e-12,
    )
    assert out["default"].iloc[:19].isna().all()

    frame = pd.DataFrame({"A": [2.0, np.nan, -3.0]})
    op = OperatorRegistry.get("ts_product", "pandas_numpy", mode="research")
    assert op.calculate(frame, window=3, min_periods=2).iloc[-1, 0] == -6.0
    assert np.isnan(
        op.calculate(frame, window=3, min_periods=2, skipna=False).iloc[-1, 0]
    )
