# -*- coding: utf-8
"""polars_long 能力分层与 scan_index_long 测试。"""
from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.polars_long_policy import (
    POLARS_LONG_PASSTHROUGH,
    classify_plan_op,
    infer_polars_long_tier,
)
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def source():
    load_all()
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, 20.0, 21.0], index=idx)
    return InMemorySeriesSource(data={"close": close})


def test_bfill_is_passthrough_not_native():
    assert "bfill" in POLARS_LONG_PASSTHROUGH
    assert classify_plan_op("bfill") == "passthrough"
    assert infer_polars_long_tier("bfill") == "passthrough"


def test_if_else_production_tier():
    load_all()
    from backend.operator_capability import summarize_operator

    spec = summarize_operator("if_else")
    assert spec.polars_long_tier == "native"
    # panel polars 以 ``where`` 为 production parity 代表（if_else 为 DSL 别名）
    assert summarize_operator("where").polars == "production_safe"


def test_bfill_long_path_passthrough_telemetry(source):
    expr = make_cleaned_call_factory("bfill")(col("close"))
    out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    assert out.get("used_polars_long_path") is True
    assert out.get("used_polars_long_passthrough") is True
    assert not out.get("used_polars_long_native")


def test_compile_memo_dedupes_duplicate_subtree(source):
    """add(ts_mean(x), ts_mean(x)) 应命中 memo，结果与 pandas 一致。"""
    sub = make_cleaned_call_factory("ts_mean")(col("close"), 2)
    expr = make_cleaned_call_factory("add")(sub, sub)
    pd_out = FactorEngine(backend=build_backend("pandas"), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    long_out = FactorEngine(backend=build_backend("polars_long"), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    assert long_out.get("used_polars_long_native") is True
    pd.testing.assert_series_equal(
        pd_out["result"].sort_index(),
        long_out["result"].sort_index(),
        check_names=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_scan_index_long_on_source(source):
    lf = source.scan_index_long()
    rows = lf.collect().sort(["ts", "inst"])
    assert rows.height == 4
    assert set(rows["inst"].to_list()) == {"A", "B"}


def test_optional_universe_align_without_load_column(source):
    os.environ["FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE"] = "1"
    try:
        expr = make_cleaned_call_factory("ts_mean")(col("close"), 2)
        eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
        out = eng.run(Factor(name="t", expr=expr))
        assert out.get("used_polars_long_path") is True
        assert len(out["result"]) == 4
    finally:
        os.environ.pop("FACTOR_ENGINE_POLARS_LONG_ALIGN_UNIVERSE", None)
