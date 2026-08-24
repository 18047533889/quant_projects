# -*- coding: utf-8
"""backend path summary 与 native tier 修正测试。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api import rank, ts_mean
from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.backend.path_summary import infer_primary_route, snapshot_backend_path
from factor_engine.backend.polars_long_policy import classify_plan_op, infer_polars_long_tier
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0, 10.5, 12.0], index=idx)
    grp = pd.Series([1.0, 1.0, 2.0, 2.0, 1.0, 2.0], index=idx)
    return InMemorySeriesSource(data={"close": close, "grp": grp})


@pytest.mark.parametrize(
    "op",
    [
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "group_rank",
        "group_std",
        "group_normalize",
        "group_percentile",
        "group_winsorize",
        "group_decay_linear",
        "cs_mad",
        "cs_mad_zscore",
    ],
)
def test_window_group_ops_are_native_tier(op):
    assert classify_plan_op(op) == "native"
    assert infer_polars_long_tier(op) == "native"


@pytest.mark.parametrize(
    "op",
    ["ts_kurt"],
)
def test_complex_group_ops_remain_map_groups(op):
    assert classify_plan_op(op) == "map_groups"
    assert infer_polars_long_tier(op) == "map_groups"


def test_recursive_ewm_corr_is_not_admitted_to_segmented_long_execution():
    # 2026-08: ts_ewm_corr was promoted to daily and the polars-expr emitter now
    # handles it via full-series per-instrument pandas ewm (map_groups tier).
    # It is still NOT native segmented execution, so it stays out of the native
    # fastpath tiers (stateful / python_rolling / native).
    assert classify_plan_op("ts_ewm_corr") == "map_groups"
    assert infer_polars_long_tier("ts_ewm_corr") == "map_groups"


@pytest.mark.parametrize(
    "op",
    ["ts_decay_linear", "WMA", "ts_skew", "ts_quantile", "ts_argmax", "ts_argmin"],
)
def test_python_rolling_ops_tier(op):
    assert classify_plan_op(op) == "python_rolling"
    assert infer_polars_long_tier(op) == "python_rolling"


def test_group_mean_long_path_native_telemetry(source):
    expr = make_cleaned_call_factory("group_zscore")(col("close"), col("grp"))
    out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    assert out.get("used_polars_long_path") is True
    assert out.get("used_polars_long_native") is True
    assert not out.get("used_polars_long_map_groups")


def test_infer_primary_route_native():
    route = infer_primary_route(
        {"backend": "polars_long", "used_polars_long_path": True, "used_polars_long_native": True}
    )
    assert route == "polars_long_native"


def test_infer_primary_route_python_rolling():
    route = infer_primary_route(
        {
            "backend": "polars_long",
            "used_polars_long_path": True,
            "used_polars_long_python_rolling": True,
            "used_polars_long_native": False,
            "polars_long_python_rolling_ops": ["ts_argmax"],
        }
    )
    assert route == "polars_long_python_rolling"


def test_ts_argmax_python_rolling_telemetry(source):
    expr = make_cleaned_call_factory("ts_argmax")(col("close"), 2)
    out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    assert out.get("used_polars_long_path") is True
    assert out.get("used_polars_long_python_rolling") is True
    assert not out.get("used_polars_long_native")
    assert out.get("polars_long_python_rolling_ops") == ["ts_argmax"]
    assert out.get("backend_path_summary", {}).get("primary_route") == "polars_long_python_rolling"


def test_run_many_backend_path_summary(source):
    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run_many([f1, f2])
    assert "backend_paths" in out
    assert "backend_path_summary" in out
    assert out["backend_paths"]["a"]["primary_route"] == "polars_long_native"
    assert out["backend_paths"]["b"]["primary_route"] == "polars_long_native"
    assert out["backend_path_summary"]["factor_count"] == 2
    assert out["backend_path_summary"]["all_native"] is True


def test_snapshot_backend_path_includes_ops():
    snap = snapshot_backend_path(
        {
            "backend": "polars_long",
            "used_polars_long_path": True,
            "used_polars_long_native": True,
            "polars_long_native_ops": ["group_mean"],
        }
    )
    assert snap["primary_route"] == "polars_long_native"
    assert snap["polars_long_native_ops"] == ["group_mean"]
