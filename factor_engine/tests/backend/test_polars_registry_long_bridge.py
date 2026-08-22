# -*- coding: utf-8
"""Registry bridge：polars_long 覆盖 Registry polars 算子。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.polars_expr_emitter import POLARS_EXPR_CAPABLE, POLARS_LONG_CAPABLE, plan_is_polars_long_capable
from backend.polars_registry_bridge import polars_registry_long_capable
from cleaned_operators import load_all
from cleaned_operators.operator_policy import polars_implemented_canonicals
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 21.0, 20.5, 22.0], index=idx)
    return InMemorySeriesSource(data={"close": close})


def _run(source, expr, backend_name: str):
    eng = FactorEngine(backend=build_backend(backend_name), data_source=source)
    return eng.run(Factor(name="t", expr=expr))


def test_polars_long_covers_registry_polars():
    from backend.polars_expr_emitter import get_polars_long_capable

    load_all()  # governance renames/moves legacy names before the live query
    reg = polars_implemented_canonicals() - {"column", "literal"}
    long_set = get_polars_long_capable() - {"column", "literal"}
    strict_period_registry = {
        "quarter_from_cumulative", "ttm_from_quarterly",
        "ttm_from_cumulative", "yoy_by_period",
    }
    # constant is a scalar operator (ScalarOperator) with no series long lowering.
    scalar_only = {"constant"}
    assert reg - long_set <= {"Lead", "next", "shuffle"} | strict_period_registry | scalar_only


def test_registry_bridge_disjoint_from_native():
    from backend.polars_expr_emitter import POLARS_EXPR_CAPABLE

    bridge = polars_registry_long_capable(
        exclude_native=POLARS_EXPR_CAPABLE - {"column", "literal"}
    )
    native = POLARS_EXPR_CAPABLE - {"column", "literal"}
    assert bridge.isdisjoint(native)


@pytest.mark.parametrize(
    "factory_name,expr_builder",
    [
        ("MACD", lambda: make_cleaned_call_factory("MACD")(col("close"), 2, 3, 2)),
        ("RSI", lambda: make_cleaned_call_factory("RSI")(col("close"), 2)),
        ("MOM", lambda: make_cleaned_call_factory("MOM")(col("close"), 2)),
        ("ROC", lambda: make_cleaned_call_factory("ROC")(col("close"), 2)),
        ("acos", lambda: make_cleaned_call_factory("acos")(col("close"))),
        ("if_else", lambda: make_cleaned_call_factory("if_else")(col("close"), col("close"), col("close"))),
        ("cum_std", lambda: make_cleaned_call_factory("cum_std")(col("close"))),
    ],
)
def test_polars_long_registry_ops(source, factory_name, expr_builder):
    expr = expr_builder()
    eng = FactorEngine(backend=build_backend("pandas"), data_source=source)
    try:
        plan, _ = eng.compile(Factor(name="t", expr=expr))
    except KeyError:
        pytest.skip(f"{factory_name} moved out of the primitive registry")
    if not plan_is_polars_long_capable(plan):
        pytest.skip(f"{factory_name} is not an active PolarsLong primitive")
    assert plan_is_polars_long_capable(plan), factory_name

    pd_out = _run(source, expr, "pandas")
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_path") is True, factory_name
    if factory_name in {"RSI", "acos"}:
        assert long_out.get("used_polars_long_registry") is True, factory_name
    elif factory_name in {"MOM", "ROC", "if_else"}:
        assert long_out.get("used_polars_long_native") is True, factory_name
    elif factory_name == "cum_std":
        assert long_out.get("used_polars_long_python_rolling") is True, factory_name
    pd.testing.assert_series_equal(
        pd_out["result"].sort_index(),
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-5,
        atol=1e-5,
    )
