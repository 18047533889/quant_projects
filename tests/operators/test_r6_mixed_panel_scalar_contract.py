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
from factor_engine.runtime.operator_snapshot import build_runtime_operator_snapshot
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    load_all()


def test_power_all_panel_scalar_topologies_and_domain():
    op = OperatorRegistry.get("power", "pandas_numpy", mode="any")
    base = pd.DataFrame([[-2.0, 0.0], [4.0, np.nan]])
    exp = pd.DataFrame([[2.0, -1.0], [0.5, 3.0]])
    np.testing.assert_allclose(op.calculate(base, exp), [[4.0, np.nan], [2.0, np.nan]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(2.0, exp), [[4.0, 0.5], [np.sqrt(2), 8.0]])
    np.testing.assert_allclose(op.calculate(base, 2.0), [[4.0, 0.0], [16.0, np.nan]], equal_nan=True)
    assert op.calculate(2.0, 3.0) == 8.0
    assert np.isnan(op.calculate(-2.0, 0.5))


def test_coalesce_is_nullable_variadic_panel_or_scalar():
    op = OperatorRegistry.get("coalesce", "pandas_numpy", mode="any")
    panel = pd.DataFrame([[np.nan, 2.0], [3.0, np.nan]])
    np.testing.assert_allclose(op.calculate(None, panel, 7.0), [[7.0, 2.0], [3.0, 7.0]])
    assert op.calculate(None, np.nan, 5.0) == 5.0


def test_safe_div_null_all_topologies_domain_and_axes():
    op = OperatorRegistry.get("safe_div_null", "pandas_numpy", mode="any")
    x = pd.DataFrame([[4.0, np.nan], [8.0, 3.0]])
    y = pd.DataFrame([[2.0, 1e-13], [0.0, -3.0]])
    np.testing.assert_allclose(op.calculate(x, y), [[2.0, np.nan], [np.nan, -1.0]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(x, 2.0), [[2.0, np.nan], [4.0, 1.5]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(6.0, y), [[3.0, np.nan], [np.nan, -2.0]], equal_nan=True)
    assert op.calculate(6.0, 2.0) == 3.0
    with pytest.raises(ValueError, match="epsilon"):
        op.calculate(1.0, 2.0, epsilon=0.0)
    with pytest.raises(ValueError, match="axes|aligned"):
        op.calculate(x, y.rename(columns={0: "other"}))


def test_protected_div_all_topologies_default_missing_and_axes():
    op = OperatorRegistry.get("protected_div", "pandas_numpy", mode="any")
    x = pd.DataFrame([[4.0, np.nan], [8.0, 3.0]])
    y = pd.DataFrame([[2.0, 1e-13], [0.0, -3.0]])
    np.testing.assert_allclose(op.calculate(x, y, default=-7.0), [[2.0, np.nan], [-7.0, -1.0]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(x, 2.0), [[2.0, np.nan], [4.0, 1.5]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(6.0, y, default=-7.0), [[3.0, -7.0], [-7.0, -2.0]], equal_nan=True)
    assert op.calculate(6.0, 2.0) == 3.0
    assert op.calculate(6.0, 0.0, default=-7.0) == -7.0
    with pytest.raises((TypeError, ValueError), match="epsilon"):
        op.calculate(1.0, 2.0, epsilon=-1.0)
    with pytest.raises(ValueError, match="axes|misaligned"):
        op.calculate(x, y.rename(columns={0: "other"}))


def test_signed_power_all_topologies_domain_and_axes():
    op = OperatorRegistry.get("signed_power", "pandas_numpy", mode="any")
    x = pd.DataFrame([[-2.0, 0.0], [4.0, np.nan]])
    c = pd.DataFrame([[2.0, -1.0], [0.5, 3.0]])
    np.testing.assert_allclose(op.calculate(x, c), [[-4.0, np.nan], [2.0, np.nan]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(x, 2.0), [[-4.0, 0.0], [16.0, np.nan]], equal_nan=True)
    np.testing.assert_allclose(op.calculate(-2.0, c), [[-4.0, -0.5], [-np.sqrt(2), -8.0]], equal_nan=True)
    assert op.calculate(-2.0, 0.5) == pytest.approx(-np.sqrt(2))
    assert np.isnan(op.calculate(0.0, -1.0))
    assert np.isnan(op.calculate(1e308, 2.0))
    with pytest.raises(ValueError, match="axes|misaligned"):
        op.calculate(x, c.rename(columns={0: "other"}))


def test_mixed_schema_is_declared_and_variadic_is_visible():
    rows = {row["canonical"]: row for row in build_runtime_operator_snapshot(profile="runtime")["operators"]}
    for name in ("power", "coalesce", "safe_div_null", "protected_div", "signed_power"):
        checked = 0
        for backend in rows[name]["backends"]:
            schema = backend["parameter_schema"]
            assert schema["x-factor-engine-contract-status"] == "declared"
            mixed_fields = {"x", "c"} if name == "signed_power" else {"x", "y"}
            if not mixed_fields <= set(schema["properties"]):
                continue
            checked += 1
            for field in mixed_fields:
                prop = schema["properties"][field]
                runtime_op = OperatorRegistry.get(name, backend["backend"], mode="any")
                assert prop["x-factor-engine-role"] == "panel_or_scalar", (
                    name, backend["backend"], field, prop, type(runtime_op), runtime_op.metadata.mixed_params,
                )
                assert {"array", "number"} <= {branch["type"] for branch in prop["anyOf"]}
        assert checked >= 2
    schema = rows["coalesce"]["backends"][0]["parameter_schema"]
    assert schema["x-factor-engine-variadic"]["name"] == "values"


def test_mixed_dsl_run_many_pandas_polars_parity():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=3), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    source = InMemorySeriesSource(data={
        "base": pd.Series([-2.0, 0.0, 4.0, np.nan, 3.0, 2.0], index=idx),
        "exp": pd.Series([2.0, -1.0, 0.5, 3.0, 2.0, 3.0], index=idx),
        "alt": pd.Series([9.0, 8.0, 7.0, 6.0, 5.0, 4.0], index=idx),
        "denom": pd.Series([2.0, 0.0, 2.0, 1e-13, -3.0, 4.0], index=idx),
    })
    factors = [
        Factor(name="panel_panel", expr=F("power")(col("base"), col("exp"))),
        Factor(name="scalar_panel", expr=F("power")(2.0, col("exp"))),
        Factor(name="coalesce3", expr=F("coalesce")(col("base"), col("alt"), 0.0)),
        Factor(name="safe_div", expr=F("safe_div_null")(col("base"), col("denom"))),
        Factor(name="protected", expr=F("protected_div")(2.0, col("denom"), default=-7.0)),
        Factor(name="signed", expr=F("signed_power")(col("base"), col("exp"))),
    ]
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=source, run_mode="research").run_many(factors)
    pl_out = FactorEngine(backend=build_backend("polars_long"), data_source=source, run_mode="research").run_many(factors)
    for name in ("panel_panel", "scalar_panel", "coalesce3", "safe_div", "protected", "signed"):
        pd.testing.assert_series_equal(pd_out["results"][name], pl_out["results"][name], check_names=False, check_dtype=False)
    for name in ("panel_panel", "scalar_panel", "coalesce3", "safe_div", "protected", "signed"):
        assert pl_out["backend_paths"][name]["used_polars_long_path"] is True
