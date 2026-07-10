# -*- coding: utf-8
"""PolarsLongBackend vs PandasBackend parity（long-table native 路径）。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
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
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 10.5, 12.0, 11.5, 20.0, 21.0, 20.5, 22.0, 21.5], index=idx)
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([100.0, 110.0, 105.0, 120.0, 115.0, 200.0, 210.0, 205.0, 220.0, 215.0], index=idx)
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02], index=idx)
    grp = pd.Series([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], index=idx, dtype=float)
    return InMemorySeriesSource(data={"close": close, "high": high, "low": low, "volume": volume, "ret": ret, "grp": grp})


def _run(source, expr, backend_name: str):
    eng = FactorEngine(backend=build_backend(backend_name), data_source=source)
    return eng.run(Factor(name="t", expr=expr))


POLARS_LONG_PARITY_CASES = [
        ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(col("close"), 5)),
        ("ts_delta", lambda: make_cleaned_call_factory("ts_delta")(col("close"), 1)),
        ("ts_pct", lambda: make_cleaned_call_factory("ts_pct")(col("close"), 1)),
        ("rank", lambda: make_cleaned_call_factory("rank")(col("close"))),
        ("zscore", lambda: make_cleaned_call_factory("zscore")(col("close"))),
        ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(col("close"), 10)),
        (
            "protected_div",
            lambda: make_cleaned_call_factory("protected_div")(col("close"), col("volume")),
        ),
        ("log_returns", lambda: make_cleaned_call_factory("log_returns")(col("close"))),
        (
            "vwap",
            lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 3),
        ),
        (
            "ts_beta",
            lambda: make_cleaned_call_factory("ts_beta")(col("ret"), col("close"), 3),
        ),
        (
            "cs_resid",
            lambda: make_cleaned_call_factory("cs_resid")(col("close"), col("volume")),
        ),
        (
            "group_percentile",
            lambda: make_cleaned_call_factory("group_percentile")(col("close"), col("grp"), 0.5),
        ),
        (
            "ts_ema",
            lambda: make_cleaned_call_factory("ts_ema")(col("close"), 3),
        ),
        (
            "fillna",
            lambda: make_cleaned_call_factory("fillna")(col("close"), 0),
        ),
        (
            "group_winsorize",
            lambda: make_cleaned_call_factory("group_winsorize")(col("close"), col("grp")),
        ),
        (
            "group_decay_linear",
            lambda: make_cleaned_call_factory("group_decay_linear")(col("close"), col("grp"), 5),
        ),
        (
            "cs_mad",
            lambda: make_cleaned_call_factory("cs_mad")(col("close")),
        ),
        (
            "cs_mad_zscore",
            lambda: make_cleaned_call_factory("cs_mad_zscore")(col("close")),
        ),
        (
            "c_mean",
            lambda: make_cleaned_call_factory("c_mean")(col("close")),
        ),
        (
            "ts_decay_linear",
            lambda: make_cleaned_call_factory("ts_decay_linear")(col("close"), 3),
        ),
        (
            "WMA",
            lambda: make_cleaned_call_factory("WMA")(col("close"), 3),
        ),
        (
            "ts_mad",
            lambda: make_cleaned_call_factory("ts_mad")(col("close"), 2),
        ),
        (
            "ts_quantile",
            lambda: make_cleaned_call_factory("ts_quantile")(col("close"), 3, 0.5),
        ),
        (
            "ts_regression",
            lambda: make_cleaned_call_factory("ts_regression")(col("close"), col("volume"), 3),
        ),
        (
            "Slope",
            lambda: make_cleaned_call_factory("Slope")(col("close"), 3),
        ),
        (
            "log_abs",
            lambda: make_cleaned_call_factory("log_abs")(col("close")),
        ),
        (
            "cum_delta",
            lambda: make_cleaned_call_factory("cum_delta")(col("close")),
        ),
        (
            "expanding_mean",
            lambda: make_cleaned_call_factory("expanding_mean")(col("close")),
        ),
        (
            "ewm_corr",
            lambda: make_cleaned_call_factory("ewm_corr")(col("ret"), col("close"), 3),
        ),
        (
            "count",
            lambda: make_cleaned_call_factory("count")(col("close")),
        ),
        (
            "ts_ratio",
            lambda: make_cleaned_call_factory("ts_ratio")(col("close")),
        ),
        (
            "ts_kurt",
            lambda: make_cleaned_call_factory("ts_kurt")(col("close"), 4),
        ),
        (
            "expanding_rank",
            lambda: make_cleaned_call_factory("expanding_rank")(col("close")),
        ),
        (
            "quantile",
            lambda: make_cleaned_call_factory("quantile")(col("close"), 2),
        ),
        (
            "RSI_WILDER",
            lambda: make_cleaned_call_factory("RSI_WILDER")(col("close"), 2),
        ),
        (
            "ATR_WILDER",
            lambda: make_cleaned_call_factory("ATR_WILDER")(col("high"), col("low"), col("close"), 2),
        ),
        (
            "add_combo",
            lambda: make_cleaned_call_factory("add")(
                make_cleaned_call_factory("ts_mean")(col("close"), 5),
                make_cleaned_call_factory("ts_delta")(col("close"), 1),
            ),
        ),
]


@pytest.mark.parametrize(
    "factory_name,expr_builder",
    POLARS_LONG_PARITY_CASES,
)
def test_polars_long_matches_pandas(source, factory_name, expr_builder):
    expr = expr_builder()
    pd_out = _run(source, expr, "pandas")
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_path") is True, factory_name
    assert not long_out.get("polars_long_fallback_reason"), factory_name
    base = pd_out["result"].sort_index()
    fast = long_out["result"].sort_index()
    pd.testing.assert_series_equal(base, fast, check_names=False, rtol=1e-6, atol=1e-6)


def test_polars_long_fallback_on_unsupported(source):
    from api.cleaned_ops import make_cleaned_call_factory

    fft = make_cleaned_call_factory("fft")
    out = _run(source, fft(col("close")), "polars_long")
    assert not out.get("used_polars_long_path")
    assert out.get("polars_long_fallback_reason") or out.get("result") is not None


def test_hybrid_long_backend_factory():
    from backend.factory import build_backend
    from backend.hybrid_long_backend import HybridLongBackend

    assert isinstance(build_backend("auto_long"), HybridLongBackend)
    assert isinstance(build_backend("hybrid_long"), HybridLongBackend)


def test_scan_polars_long_skips_panel(source):
    """scan_polars_long 存在且返回 LazyFrame。"""
    lf = source.scan_polars_long(["close", "volume"])
    cols = lf.collect_schema().names()
    assert "ts" in cols and "inst" in cols
    assert "close" in cols and "volume" in cols
