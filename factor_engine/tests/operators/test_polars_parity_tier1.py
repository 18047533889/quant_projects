# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED：parity 通过后才允许并入 POLARS_PRODUCTION_SAFE。"""

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
    POLARS_PARITY_VERIFIED,
    POLARS_PARITY_VERIFIED_TIER1,
    POLARS_PRODUCTION_SAFE,
)
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_std = make_cleaned_call_factory("ts_std")
ts_pct = make_cleaned_call_factory("ts_pct")
rank_ = make_cleaned_call_factory("rank")
zscore_ = make_cleaned_call_factory("zscore")
ffill_ = make_cleaned_call_factory("ffill")
winsorize_ = make_cleaned_call_factory("winsorize")


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
    return InMemorySeriesSource(data={"x": x})


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
    x = col("x")
    return {
        "rank": rank_(x),
        "zscore": zscore_(x),
        "ts_std": ts_std(x, 2),
        "ffill": ffill_(x),
        "ts_pct": ts_pct(x, 1),
        "winsorize": winsorize_(x, 0.01),
    }


def test_parity_verified_tier1_subset_of_production_safe():
    from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE

    assert POLARS_PARITY_VERIFIED_TIER1 <= POLARS_PRODUCTION_SAFE
    assert POLARS_PARITY_VERIFIED_TIER1 <= POLARS_PARITY_VERIFIED


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER1))
def test_polars_parity_verified_tier1(source, canonical):
    exprs = _parity_exprs()
    assert canonical in exprs, f"缺少 parity fixture: {canonical}"
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
