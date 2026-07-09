# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER2：双序列 / 截面 / 价量 parity CI。"""

from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from cleaned_operators.operator_policy import (
    POLARS_PARITY_VERIFIED_TIER2,
    POLARS_PRODUCTION_SAFE,
)
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
    x = pd.Series([1.0, 2.0, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0], index=idx)
    y = pd.Series([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 40.0], index=idx)
    grp = pd.Series([1, 1, 1, 1, 2, 2, 2, 2], index=idx, dtype=float)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05], index=idx)
    bm = pd.Series([0.005, 0.015, -0.005, 0.02, 0.03, 0.008, 0.015, 0.04], index=idx)
    vol = pd.Series([100, 200, 150, 300, 400, 500, 450, 600], index=idx)
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 20.0, 21.0, 20.5, 22.0], index=idx)
    return InMemorySeriesSource(
        data={"x": x, "y": y, "grp": grp, "ret": ret, "bm": bm, "vol": vol, "close": close}
    )


def _run_pair(source, expr):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
        a = eng_pd.run(Factor(name="t", expr=expr))["result"]
        b = eng_pl.run(Factor(name="t", expr=expr))["result"]
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
    return a, b


def _parity_exprs():
    x, y = col("x"), col("y")
    return {
        "ts_corr": make_cleaned_call_factory("ts_corr")(x, y, 2),
        "ts_rank": make_cleaned_call_factory("ts_rank")(x, 2),
        "coalesce": make_cleaned_call_factory("coalesce")(x, y),
        "protected_div": make_cleaned_call_factory("protected_div")(x, y),
        "protected_log": make_cleaned_call_factory("protected_log")(x),
        "where": make_cleaned_call_factory("where")(x, x, y),
        "cs_demean": make_cleaned_call_factory("cs_demean")(x),
        "scale": make_cleaned_call_factory("scale")(x),
        "normalize": make_cleaned_call_factory("normalize")(x),
        "group_rank": make_cleaned_call_factory("group_rank")(x, col("grp")),
        "rolling_beta": make_cleaned_call_factory("rolling_beta")(col("ret"), col("bm"), 2),
        "vwap": make_cleaned_call_factory("vwap")(col("close"), col("vol"), 2),
    }


def test_parity_tier2_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER2 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER2))
def test_polars_parity_verified_tier2(source, canonical):
    exprs = _parity_exprs()
    assert canonical in exprs, f"缺少 parity fixture: {canonical}"
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
