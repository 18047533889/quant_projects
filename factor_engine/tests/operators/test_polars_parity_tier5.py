# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER5：PRODUCTION_CORE 缺口 parity CI。"""

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
    POLARS_PARITY_VERIFIED_TIER5,
    POLARS_PRODUCTION_SAFE,
)
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_zscore = make_cleaned_call_factory("ts_zscore")
protected_sqrt = make_cleaned_call_factory("protected_sqrt")


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
    neg = pd.Series([-1.0, 4.0, -9.0, 16.0, 0.0, 25.0, -36.0, 49.0], index=idx)
    flat = pd.Series([5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0], index=idx)
    return InMemorySeriesSource(data={"x": x, "neg": neg, "flat": flat})


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
        "ts_zscore": ts_zscore(col("x"), 2),
        "protected_sqrt": protected_sqrt(col("neg")),
    }


def test_parity_tier5_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER5 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER5))
def test_polars_parity_verified_tier5(source, canonical):
    exprs = _parity_exprs()
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)


def test_ts_zscore_constant_window_zero_std(source):
    """窗口内常数序列：std→0 替换为 1，zscore=0。"""
    a, b = _run_pair(source, ts_zscore(col("flat"), 2))
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
    assert a.iloc[-1] == pytest.approx(0.0)


def test_protected_sqrt_negative_inputs_zero(source):
    a, b = _run_pair(source, protected_sqrt(col("neg")))
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
    assert a.iloc[0] == pytest.approx(0.0)
    assert a.iloc[2] == pytest.approx(0.0)
