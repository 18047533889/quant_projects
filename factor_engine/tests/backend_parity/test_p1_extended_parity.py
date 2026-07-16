# -*- coding: utf-8
"""P1 Extended 算子 PolarsLong vs Pandas parity。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.production_fastpath_tiers import P1_EXTENDED_CANONICALS
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0, 13.0, 14.0], index=idx)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, -0.01], index=idx)
    grp = pd.Series([1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "ret": ret, "grp": grp})


CASES = [
    ("ts_mad", lambda: make_cleaned_call_factory("ts_mad")(col("close"), 3)),
    ("ts_skew", lambda: make_cleaned_call_factory("ts_skew")(col("close"), 3)),
    ("ts_argmax", lambda: make_cleaned_call_factory("ts_argmax")(col("close"), 3)),
    ("ts_argmin", lambda: make_cleaned_call_factory("ts_argmin")(col("close"), 3)),
    ("ts_ema", lambda: make_cleaned_call_factory("ewm_mean")(col("close"), 3)),
    ("cum_sum", lambda: make_cleaned_call_factory("cum_sum")(col("close"))),
    ("log_abs", lambda: make_cleaned_call_factory("log_abs")(col("close"))),
    (
        "group_decay_linear",
        lambda: make_cleaned_call_factory("group_decay_linear")(col("close"), col("grp")),
    ),
]


@pytest.mark.parametrize("name,expr_builder", CASES)
def test_p1_extended_polars_long_matches_pandas(source, name, expr_builder):
    assert name in P1_EXTENDED_CANONICALS
    expr = expr_builder()
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()
    long_out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(
        Factor(name="t", expr=expr)
    )["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-5, atol=1e-5)
