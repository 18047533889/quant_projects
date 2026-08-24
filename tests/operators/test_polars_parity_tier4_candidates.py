# -*- coding: utf-8
"""Tier-4 parity：Sharpe / 自相关（已并入 POLARS_PRODUCTION_SAFE）。"""

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
    POLARS_PARITY_VERIFIED_TIER4,
    POLARS_PRODUCTION_SAFE,
)
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_sharpe = make_cleaned_call_factory("ts_sharpe")
ts_autocorr = make_cleaned_call_factory("ts_autocorr")


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
        "ts_sharpe": ts_sharpe(x, 2),
        "ts_autocorr": ts_autocorr(x, 2, 1),
    }


def test_parity_tier4_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER4 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER4))
def test_polars_parity_verified_tier4(source, canonical):
    exprs = _parity_exprs()
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-8, atol=1e-8)
