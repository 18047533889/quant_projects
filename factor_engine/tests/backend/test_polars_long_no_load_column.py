# -*- coding: utf-8
"""polars_long 不得调用 load_column / load_columns / prefetch_columns。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import NoLoadColumnSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "A"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 20.0, 21.0, 20.5], index=idx)
    volume = pd.Series([100.0, 110.0, 105.0, 200.0, 210.0, 205.0], index=idx)
    open_ = pd.Series([9.5, 10.5, 10.0, 19.5, 20.5, 20.0], index=idx)
    return NoLoadColumnSource(data={"close": close, "volume": volume, "open": open_})


@pytest.mark.parametrize(
    "factory_name,expr_builder",
    [
        ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(col("close"), 5)),
        ("rank", lambda: make_cleaned_call_factory("rank")(col("close"))),
        (
            "group_mean",
            lambda: make_cleaned_call_factory("group_mean")(col("close"), col("volume")),
        ),
        (
            "vwap",
            lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 3),
        ),
        (
            "protected_div",
            lambda: make_cleaned_call_factory("protected_div")(col("close"), col("volume")),
        ),
        (
            "where",
            lambda: make_cleaned_call_factory("where")(
                make_cleaned_call_factory("gt")(col("close"), 0),
                col("close"),
                col("volume"),
            ),
        ),
        (
            "add_combo",
            lambda: make_cleaned_call_factory("add")(
                make_cleaned_call_factory("ts_mean")(col("close"), 2),
                make_cleaned_call_factory("ts_delta")(col("close"), 1),
            ),
        ),
        (
            "multi_column_plan",
            lambda: make_cleaned_call_factory("add")(
                make_cleaned_call_factory("ts_mean")(col("close"), 2),
                make_cleaned_call_factory("ts_mean")(col("open"), 2),
            ),
        ),
    ],
)
def test_polars_long_never_loads_columns(source, factory_name, expr_builder):
    if factory_name in {"vwap", "protected_div"}:
        pytest.skip(f"{factory_name} is not an active daily Polars canonical")
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run(Factor(name="t", expr=expr_builder()))
    assert out.get("used_polars_long_path") is True
    assert not out.get("polars_long_fallback_reason")
    assert len(out["result"]) == 6
