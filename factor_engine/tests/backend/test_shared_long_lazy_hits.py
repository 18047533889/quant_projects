# -*- coding: utf-8
"""shared_long_lazy_cache plan_ref 命中计数。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from backend.context import ExecutionContext
from backend.polars_long_backend import PolarsLongBackend
from backend.polars_expr_emitter import compile_polars_long_lazy
from planner.logical_plan import PlanNode
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def source():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0], index=idx)
    return InMemorySeriesSource(data={"close": close})


def test_plan_ref_lazy_cache_increments_hits(source):
    sub = PlanNode(op="ts_mean", inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])], attrs={"d": 2})
    ctx = ExecutionContext(
        data_source=source,
        timestamp_col="timestamp",
        instrument_col="instrument",
        shared_long_lazy_cache={},
        runtime_stats={},
    )
    compile_polars_long_lazy(sub, ctx, lazy_cache_key="sid1")
    root = PlanNode(
        op="rank",
        attrs={},
        inputs=[PlanNode(op="plan_ref", attrs={"sid": "sid1"}, inputs=[])],
    )
    PolarsLongBackend().execute(root, ctx)
    assert int((ctx.runtime_stats or {}).get("shared_long_lazy_hits") or 0) >= 1
