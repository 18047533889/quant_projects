# -*- coding: utf-8
"""宽表 column 融合与 native tier 审计。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.polars_long_policy import (
    POLARS_LONG_MAP_GROUPS,
    POLARS_LONG_NATIVE,
    POLARS_LONG_PYTHON_ROLLING,
)
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0], index=idx)
    volume = pd.Series([100.0, 110.0, 200.0, 210.0, 150.0, 160.0], index=idx)
    flag = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    grp = pd.Series([1.0, 1.0, 2.0, 2.0, 1.0, 2.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "volume": volume, "flag": flag, "grp": grp})


def _run(source, expr, backend_name: str):
    return FactorEngine(backend=build_backend(backend_name), data_source=source).run(
        Factor(name="t", expr=expr)
    )


@pytest.mark.parametrize(
    "factory_name,expr_builder",
    [
        ("power", lambda: make_cleaned_call_factory("power")(col("close"), col("volume"))),
        ("gt", lambda: make_cleaned_call_factory("gt")(col("close"), col("volume"))),
        ("and_", lambda: make_cleaned_call_factory("and_")(col("flag"), col("volume"))),
        ("coalesce", lambda: make_cleaned_call_factory("coalesce")(col("close"), col("volume"))),
        (
            "where",
            lambda: make_cleaned_call_factory("where")(col("flag"), col("close"), col("volume")),
        ),
        ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(col("close"), col("volume"), 2)),
        ("vwap", lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 2)),
    ],
)
def test_fused_base_column_ops_match_pandas(source, factory_name, expr_builder):
    expr = expr_builder()
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


# emitter 内已知使用 rolling_map / map_groups 的算子，不得出现在 POLARS_LONG_NATIVE
_ROLLING_MAP_OPS = frozenset(
    {
        "ts_kurt",
        "ts_moment",
        "ts_max_buildup",
        "ewm_corr",
    }
)


_PYTHON_ROLLING_OPS = frozenset(
    {
        "ts_decay_linear",
        "WMA",
        "Slope",
        "ts_skew",
        "ts_quantile",
    }
)

_NATIVE_BATCH2_ROLLING_OPS = frozenset({"ts_argmax", "ts_argmin"})


def test_native_tier_excludes_rolling_map_ops():
    overlap = POLARS_LONG_NATIVE & _ROLLING_MAP_OPS
    assert not overlap, f"rolling_map ops mislabeled native: {sorted(overlap)}"
    py_overlap = POLARS_LONG_PYTHON_ROLLING & _PYTHON_ROLLING_OPS
    assert py_overlap == _PYTHON_ROLLING_OPS


def test_ts_sharpe_native_ts_argmax_native_batch2():
    for op in ("ts_sharpe", "ts_autocorr"):
        assert op in POLARS_LONG_NATIVE
        assert op not in POLARS_LONG_MAP_GROUPS
    for op in _NATIVE_BATCH2_ROLLING_OPS:
        assert op in POLARS_LONG_NATIVE
        assert op not in POLARS_LONG_PYTHON_ROLLING


def test_group_window_ops_are_native_not_map_groups():
    for op in (
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "group_rank",
        "group_percentile",
        "group_winsorize",
    ):
        assert op in POLARS_LONG_NATIVE
        assert op not in POLARS_LONG_MAP_GROUPS


def test_group_percentile_native_matches_pandas(source):
    expr = make_cleaned_call_factory("group_percentile")(col("close"), col("grp"), 0.5)
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_group_winsorize_native_matches_pandas(source):
    expr = make_cleaned_call_factory("group_winsorize")(col("close"), col("grp"))
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_dag_fusion_ts_mean_add_matches_pandas(source):
    """``add(ts_mean, ts_mean)`` 应走宽表 DAG fusion，结果与 pandas 一致。"""
    f = make_cleaned_call_factory
    expr = f("add")(f("ts_mean")(col("close"), 2), f("ts_mean")(col("close"), 3))
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_dag_fusion_ts_mean_delta_matches_pandas(source):
    f = make_cleaned_call_factory
    expr = f("subtract")(f("ts_mean")(col("close"), 2), f("ts_delta")(col("close"), 1))
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_dag_fusion_nary_add_ts_matches_pandas(source):
    """``add(add(ts_mean, ts_mean), ts_delta)`` 同列 n-ary fusion。"""
    f = make_cleaned_call_factory
    inner = f("add")(f("ts_mean")(col("close"), 2), f("ts_mean")(col("close"), 3))
    expr = f("add")(inner, f("ts_delta")(col("close"), 1))
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_dag_fusion_rank_ts_mean_matches_pandas(source):
    """``rank(ts_mean(col,w))`` unary-over-ts fusion，与 pandas 一致且无 registry fallback。"""
    expr = make_cleaned_call_factory("rank")(make_cleaned_call_factory("ts_mean")(col("close"), 2))
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    assert not (long_out.get("polars_long_registry_ops") or [])
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_dag_fusion_zscore_ts_mean_matches_pandas(source):
    expr = make_cleaned_call_factory("zscore")(make_cleaned_call_factory("ts_mean")(col("close"), 2))
    pd_out = _run(source, expr, "pandas")["result"].sort_index()
    long_out = _run(source, expr, "polars_long")
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out,
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )
