# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER6：双序列 Beta / 截面回归 parity CI。"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skip(reason="legacy Polars rollout tier superseded by six-way evidence certification")

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.operator_policy import (
    POLARS_PARITY_VERIFIED_TIER6,
    POLARS_PRODUCTION_SAFE,
)
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_beta = make_cleaned_call_factory("ts_beta")
cs_resid = make_cleaned_call_factory("cs_resid")
cs_regression = make_cleaned_call_factory("cs_regression")


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
    y = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05], index=idx)
    x = pd.Series([0.005, 0.015, -0.005, 0.02, 0.03, 0.008, 0.015, 0.04], index=idx)
    fac = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], index=idx)
    size = pd.Series([0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0], index=idx)
    return InMemorySeriesSource(data={"y": y, "x": x, "fac": fac, "size": size})


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
    return {
        "ts_beta": ts_beta(col("y"), col("x"), 2),
        "cs_resid": cs_resid(col("fac"), col("size")),
        "cs_regression": cs_regression(col("fac"), col("size"), 0),
    }


def test_parity_tier6_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER6 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER6))
def test_polars_parity_verified_tier6(source, canonical):
    exprs = _parity_exprs()
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_cs_regression_all_modes(source, mode: int):
    a, b = _run_pair(source, cs_regression(col("fac"), col("size"), mode))
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-8, atol=1e-8)
