# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER8：截面聚合 / EWM·WMA / 清洗·group 扩展 parity CI。"""

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
    POLARS_PARITY_VERIFIED_TIER8,
    POLARS_PRODUCTION_SAFE,
)
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

c_mean = make_cleaned_call_factory("c_mean")
c_std = make_cleaned_call_factory("c_std")
c_sum = make_cleaned_call_factory("c_sum")
c_count = make_cleaned_call_factory("c_count")
ewm_mean = make_cleaned_call_factory("ewm_mean")
wma = make_cleaned_call_factory("WMA")
nan_to_num = make_cleaned_call_factory("nan_to_num")
is_finite = make_cleaned_call_factory("is_finite")
group_std = make_cleaned_call_factory("group_std")
group_normalize = make_cleaned_call_factory("group_normalize")
ts_cov = make_cleaned_call_factory("ts_cov")


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
    return InMemorySeriesSource(data={"x": x, "y": y, "grp": grp})


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
        "c_mean": c_mean(x),
        "c_std": c_std(x),
        "c_sum": c_sum(x),
        "c_count": c_count(x),
        "ewm_mean": ewm_mean(x, 2),
        "WMA": wma(x, 2),
        "nan_to_num": nan_to_num(x, 0),
        "is_finite": is_finite(x),
        "group_std": group_std(x, col("grp")),
        "group_normalize": group_normalize(x, col("grp")),
        "ts_cov": ts_cov(x, y, 2),
    }


def test_parity_tier8_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER8 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER8))
def test_polars_parity_verified_tier8(source, canonical):
    exprs = _parity_exprs()
    assert canonical in exprs, f"缺少 parity fixture: {canonical}"
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-6, atol=1e-6)
