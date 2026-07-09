# -*- coding: utf-8
"""POLARS_PRODUCTION_SAFE 白名单算子：pandas vs polars 数值 parity CI。"""

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
from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_sum = make_cleaned_call_factory("ts_sum")
ts_min = make_cleaned_call_factory("ts_min")
ts_max = make_cleaned_call_factory("ts_max")
ts_delta = make_cleaned_call_factory("ts_delta")
ts_delay = make_cleaned_call_factory("ts_delay")
delay = make_cleaned_call_factory("delay")
abs_ = make_cleaned_call_factory("abs")
log_ = make_cleaned_call_factory("log")
clip_ = make_cleaned_call_factory("clip")
sqrt_ = make_cleaned_call_factory("sqrt")
sign_ = make_cleaned_call_factory("sign")
neg_ = make_cleaned_call_factory("neg")
exp_ = make_cleaned_call_factory("exp")
ts_mean = make_cleaned_call_factory("ts_mean")
ts_sharpe = make_cleaned_call_factory("ts_sharpe")
ts_autocorr = make_cleaned_call_factory("ts_autocorr")
ts_std = make_cleaned_call_factory("ts_std")
rank_ = make_cleaned_call_factory("rank")
zscore_ = make_cleaned_call_factory("zscore")


def _panel_series():
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
    return x, y


@pytest.fixture(scope="module")
def source():
    load_all()
    x, y = _panel_series()
    return InMemorySeriesSource(data={"x": x, "y": y})


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
        "ts_mean": ts_mean(x, 2),
        "ts_sum": ts_sum(x, 2),
        "ts_min": ts_min(x, 2),
        "ts_max": ts_max(x, 2),
        "ts_delta": ts_delta(x, 1),
        "ts_delay": ts_delay(x, 1),
        "delay": delay(x, 1),
        "add": x + y,
        "subtract": x - y,
        "multiply": x * y,
        "divide": x / y,
        "abs": abs_(x),
        "log": log_(x),
        "clip": clip_(x, 0.5, 50.0),
        "neg": neg_(x),
        "exp": exp_(x),
        "sqrt": sqrt_(x),
        "sign": sign_(x),
        "rank": rank_(x),
        "zscore": zscore_(x),
        "ts_std": ts_std(x, 2),
        "ffill": make_cleaned_call_factory("ffill")(x),
        "ts_pct": make_cleaned_call_factory("ts_pct")(x, 1),
        "winsorize": make_cleaned_call_factory("winsorize")(x, 0.01),
    }


@pytest.mark.parametrize("canonical", sorted(POLARS_PRODUCTION_SAFE))
def test_polars_production_safe_parity(source, canonical):
    exprs = _parity_exprs()
    if canonical not in exprs:
        pytest.skip(f"无 parity fixture: {canonical}")
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
