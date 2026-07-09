# -*- coding: utf-8
"""Tier-3 parity 已知语义差：未纳入 POLARS_PARITY_VERIFIED，修复后应升级 tier3。"""

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
from runtime.engine import FactorEngine
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
    grp = pd.Series([1, 1, 1, 1, 2, 2, 2, 2], index=idx, dtype=float)
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
    return a, b


def _series_equal(a, b) -> bool:
    try:
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)
        return True
    except AssertionError:
        return False


def test_ts_decay_linear_first_bar_min_periods_gap(source):
    """pandas 窗口首 bar 有值；polars 首 bar 为 NaN（min_periods 语义差）。"""
    a, b = _run_pair(source, ts_decay_linear(col("x"), 2))
    if _series_equal(a, b):
        pytest.skip("ts_decay_linear parity 已对齐，可移入 POLARS_PARITY_VERIFIED_TIER3")
    assert pd.notna(a.iloc[0])
    assert pd.isna(b.iloc[0])


def test_group_zscore_polars_passthrough_gap(source):
    """polars 路径未做分组 zscore 时，输出应与 pandas 不同。"""
    a, b = _run_pair(source, group_zscore(col("x"), col("grp")))
    if _series_equal(a, b):
        pytest.skip("group_zscore parity 已对齐，可移入 POLARS_PARITY_VERIFIED_TIER3")
    assert (a.to_numpy() == 0.0).all()
    pd.testing.assert_series_equal(b, source.data["x"], check_names=False)
