from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.execution_contract import execution_contract
from factor_engine.runtime.operator_snapshot import build_runtime_operator_snapshot
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


def test_hump_decay_finite_state_empty_and_strict_hump():
    op = OperatorRegistry.get("hump_decay", "pandas_numpy", mode="any")
    x = pd.DataFrame({"A": [1.0, 1.03, 1.2, np.nan, np.inf, 1.21, 0.9]})
    np.testing.assert_allclose(op.calculate(x, 0.05)["A"], [1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 0.9], equal_nan=True)
    seed = pd.DataFrame({"A": [np.inf, np.nan, 2.0]})
    np.testing.assert_allclose(op.calculate(seed)["A"], [np.nan, np.nan, 2.0], equal_nan=True)
    empty = pd.DataFrame(index=pd.RangeIndex(0), columns=["A"], dtype=float)
    pd.testing.assert_frame_equal(op.calculate(empty), empty)
    no_columns = pd.DataFrame(index=pd.RangeIndex(3))
    pd.testing.assert_frame_equal(op.calculate(no_columns), no_columns.astype(float))
    np.testing.assert_allclose(op.calculate(pd.DataFrame({"A": [1.0, 1.0, 1.1]}), 0.0)["A"], [1.0, 1.0, 1.1])
    for invalid in (-0.1, np.nan, np.inf, "0.1"):
        with pytest.raises((TypeError, ValueError), match="hump"):
            op.calculate(x, invalid)


def test_hump_decay_declares_true_unbounded_recursive_state():
    op = OperatorRegistry.get("hump_decay", "pandas_numpy", mode="any")
    assert op.metadata.window_semantics == "full_history"
    contract = execution_contract("hump_decay")
    assert contract.state_model == "recursive"
    assert contract.chunking == "required_full_history"
    polars_op = OperatorRegistry.get("hump_decay", "polars", mode="any")
    assert polars_op._physical_spec.stateful is True


def test_trade_when_all_mixed_topologies_nan_truth_and_axes():
    op = OperatorRegistry.get("trade_when", "pandas_numpy", mode="any")
    condition = pd.DataFrame([[1.0, 0.0], [np.nan, np.inf]])
    signal = pd.DataFrame([[10.0, 20.0], [30.0, 40.0]])
    fallback = pd.DataFrame([[-1.0, -2.0], [-3.0, -4.0]])
    np.testing.assert_allclose(op.calculate(condition, signal, fallback), [[10.0, -2.0], [-3.0, 40.0]])
    np.testing.assert_allclose(op.calculate(condition, 7.0, fallback), [[7.0, -2.0], [-3.0, 7.0]])
    np.testing.assert_allclose(op.calculate(np.nan, signal, -9.0), np.full(signal.shape, -9.0))
    np.testing.assert_allclose(op.calculate(1.0, signal, -9.0), signal)
    np.testing.assert_allclose(op.calculate(0.0, signal, -9.0), np.full(signal.shape, -9.0))
    assert op.calculate(True, 3.0, -1.0) == 3.0
    assert op.calculate(np.nan, 3.0, -1.0) == -1.0
    assert op.calculate(0.0, 3.0, -1.0) == -1.0
    with pytest.raises(ValueError, match="axes|misaligned"):
        op.calculate(condition, signal.rename(columns={0: "other"}), fallback)


def test_trade_when_pandas_and_polars_backend_oracles():
    pl = pytest.importorskip("polars")
    condition_pd = pd.DataFrame({"A": [0.0, np.nan, np.inf, -2.0]})
    signal_pd = pd.DataFrame({"A": [10.0, 20.0, 30.0, 40.0]})
    expected = np.array([-1.0, -1.0, 30.0, 40.0])
    pandas_op = OperatorRegistry.get("trade_when", "pandas_numpy", mode="any")
    np.testing.assert_allclose(pandas_op.calculate(condition_pd, signal_pd, -1.0)["A"], expected)
    polars_op = OperatorRegistry.get("trade_when", "polars", mode="any")
    polars_out = polars_op.calculate(pl.DataFrame(condition_pd), pl.DataFrame(signal_pd), -1.0)
    np.testing.assert_allclose(polars_out["A"].to_numpy(), expected)


def test_hump_trade_schema_and_dsl_polars_parity():
    rows = {row["canonical"]: row for row in build_runtime_operator_snapshot(profile="runtime")["operators"]}
    hump_schema = rows["hump_decay"]["backends"][0]["parameter_schema"]
    assert hump_schema["properties"]["hump"]["default"] == 0.05
    for backend in rows["trade_when"]["backends"]:
        schema = backend["parameter_schema"]
        if not {"condition", "signal", "fallback"} <= set(schema["properties"]):
            continue
        for field in ("condition", "signal", "fallback"):
            assert schema["properties"][field]["x-factor-engine-role"] == "panel_or_scalar"

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=4), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(data={
        "x": pd.Series([1.0, 2.0, 1.02, 2.2, np.nan, np.inf, 1.3, 2.21], index=idx),
        "condition": pd.Series([1.0, 0.0, np.nan, 1.0, 0.0, np.nan, 1.0, 0.0], index=idx),
        "fallback": pd.Series([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], index=idx),
    })
    factors = [
        Factor(name="hump", expr=F("hump_decay")(col("x"), 0.05)),
        Factor(name="trade_ppp", expr=F("trade_when")(col("condition"), col("x"), col("fallback"))),
        Factor(name="trade_sps", expr=F("trade_when")(np.nan, col("x"), -9.0)),
    ]
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=source, run_mode="research").run_many(factors)
    pl_out = FactorEngine(backend=build_backend("polars_long"), data_source=source, run_mode="research").run_many(factors)
    for name in ("hump", "trade_ppp", "trade_sps"):
        pd.testing.assert_series_equal(pd_out["results"][name], pl_out["results"][name], check_names=False, check_dtype=False)
        assert pl_out["backend_paths"][name]["used_polars_long_path"] is True
