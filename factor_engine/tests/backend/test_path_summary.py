# -*- coding: utf-8
"""backend path summary 与 native tier 修正测试。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api import rank, ts_mean
from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.path_summary import infer_primary_route, snapshot_backend_path
from backend.polars_long_policy import classify_plan_op, infer_polars_long_tier
from cleaned_operators import load_all
from runtime.engine import FactorEngine
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
    ["ewm_corr", "ts_kurt", "quantile"],
)
def test_complex_group_ops_remain_map_groups(op):
    assert classify_plan_op(op) == "map_groups"
    assert infer_polars_long_tier(op) == "map_groups"


@pytest.mark.parametrize(
    "op",
    ["ts_decay_linear", "WMA", "ts_skew", "ts_quantile"],
)
def test_python_rolling_ops_tier(op):
    assert classify_plan_op(op) == "python_rolling"
    assert infer_polars_long_tier(op) == "python_rolling"


@pytest.mark.parametrize("op", ["ts_argmax", "ts_argmin"])
def test_batch2_rolling_ops_native_tier(op):
    assert infer_polars_long_tier(op) == "native"
    assert classify_plan_op(op) == "native"


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
