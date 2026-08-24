# -*- coding: utf-8
"""POLARS_PARITY_VERIFIED_TIER3：EMA / decay / fillna / group 统计 parity CI。"""

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
    POLARS_PARITY_VERIFIED_TIER3,
    POLARS_PRODUCTION_SAFE,
)
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_ema = make_cleaned_call_factory("ts_ema")
ts_decay_linear = make_cleaned_call_factory("ts_decay_linear")
fillna_const = make_cleaned_call_factory("fillna_const")
fillna = make_cleaned_call_factory("fillna")
group_zscore = make_cleaned_call_factory("group_zscore")
group_mean = make_cleaned_call_factory("group_mean")
group_neutralize = make_cleaned_call_factory("group_neutralize")
ts_var = make_cleaned_call_factory("ts_var")
ts_median = make_cleaned_call_factory("ts_median")


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
        "ts_ema": ts_ema(x, 2),
        "ts_decay_linear": ts_decay_linear(x, 2),
        "fillna_const": fillna_const(x, 0),
        "fillna": fillna(x, 0),
        "group_zscore": group_zscore(x, col("grp")),
        "group_mean": group_mean(x, col("grp")),
        "group_neutralize": group_neutralize(x, col("grp")),
        "ts_var": ts_var(x, 2),
        "ts_median": ts_median(x, 2),
    }


def test_parity_tier3_subset_of_production_safe():
    assert POLARS_PARITY_VERIFIED_TIER3 <= POLARS_PRODUCTION_SAFE


@pytest.mark.parametrize("canonical", sorted(POLARS_PARITY_VERIFIED_TIER3))
def test_polars_parity_verified_tier3(source, canonical):
    exprs = _parity_exprs()
    assert canonical in exprs, f"缺少 parity fixture: {canonical}"
    a, b = _run_pair(source, exprs[canonical])
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
