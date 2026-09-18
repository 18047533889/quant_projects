# -*- coding: utf-8
"""Tier-3 历史 parity gap 回归：修复后应持续与 pandas 对齐。"""

from __future__ import annotations

import os
import numpy as np

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource

ts_decay_linear = make_cleaned_call_factory("ts_decay_linear")
group_zscore = make_cleaned_call_factory("group_zscore")


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
    grp = pd.Series([1] * len(idx), index=idx, dtype=float)
    return InMemorySeriesSource(data={"x": x, "grp": grp})


def _run_pair(source, expr):
    os.environ["FACTOR_ENGINE_DISABLE_BOTTLENECK"] = "1"
    try:
        eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=source)
        eng_pl = FactorEngine(backend=build_backend("polars"), data_source=source)
        a = eng_pd.run(Factor(name="t", expr=expr))["result"]
        b = eng_pl.run(Factor(name="t", expr=expr))["result"]
    finally:
        os.environ.pop("FACTOR_ENGINE_DISABLE_BOTTLENECK", None)
    # Compare labeled values, not backend-specific storage row order.
    assert a.index.is_unique and b.index.is_unique
    return a.sort_index(), b.sort_index()


def test_ts_decay_linear_parity_regression(source):
    a, b = _run_pair(source, ts_decay_linear(col("x"), 2))
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a.xs("A", level="instrument").iloc[1:], [5/3, 10/3, 20/3])


def test_group_zscore_parity_regression(source):
    a, b = _run_pair(source, group_zscore(col("x"), col("grp")))
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a.xs("A", level="instrument"), -1 / np.sqrt(2))
    np.testing.assert_allclose(a.xs("B", level="instrument"), 1 / np.sqrt(2))
