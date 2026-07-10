# -*- coding: utf-8
"""runtime event log + native tier 收紧。"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from backend.polars_long_production import (
    is_polars_long_native_production_safe,
    polars_long_production_tier,
)
from backend.runtime_events import append_runtime_event, rollup_runtime_fields
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def loaded():
    load_all()


@pytest.fixture
def source(loaded):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0], index=idx)
    return InMemorySeriesSource(data={"close": close})


def test_runtime_events_recorded_on_run(source):
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run(Factor(name="t", expr=make_cleaned_call_factory("ts_mean")(col("close"), 2)))
    events = out.get("runtime_events") or []
    assert any(e.get("type") == "polars_long_native" for e in events)
    bps = out.get("backend_path_summary") or {}
    assert int(bps.get("runtime_event_count") or 0) >= 1


def test_rollup_runtime_fields_merges_latest():
    class _Ctx:
        runtime_stats = {
            "events": [{"type": "sql_fully_pushed", "backend": "duckdb_sql"}],
            "latest": {"sql_fully_pushed": True, "backend": "duckdb_sql"},
            "fully_sql": True,
        }

    rolled = rollup_runtime_fields(_Ctx())
    assert rolled.get("sql_fully_pushed") is True
    assert rolled.get("runtime_event_count") == 1


def test_winsorize_native_tier_implemented():
    assert polars_long_production_tier("winsorize") == "implemented"
    assert not is_polars_long_native_production_safe("winsorize")


def test_ts_std_native_tier_production_safe():
    assert polars_long_production_tier("ts_std") == "production_safe"
    assert is_polars_long_native_production_safe("ts_std")
    assert polars_long_production_tier("ts_mean") == "production_safe"
    assert is_polars_long_native_production_safe("ts_mean")


def test_rebuild_runtime_from_events_only():
    from backend.path_summary import build_backend_path_summary
    from backend.runtime_events import rebuild_runtime_from_events

    runtime = {
        "events": [
            {"type": "sql_partial_pushed", "backend": "duckdb_sql", "sql_query_count": 2},
            {
                "type": "polars_long_native",
                "backend": "polars_long",
                "polars_long_native_ops": ["ts_mean", "rank"],
            },
        ],
        "latest": {"backend": "polars_long"},
    }
    rebuilt = rebuild_runtime_from_events(runtime)
    assert rebuilt.get("sql_partial_pushed") is True
    assert rebuilt.get("used_polars_long_native") is True
    assert "ts_mean" in (rebuilt.get("polars_long_native_ops") or [])
    assert rebuilt.get("runtime_event_count") == 2
    assert "duckdb_sql" in (rebuilt.get("runtime_event_backends") or [])

    summary = build_backend_path_summary(runtime)
    assert summary.get("runtime_event_count") == 2
    assert summary.get("used_polars_long_native") is True
