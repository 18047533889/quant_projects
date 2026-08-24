"""Phase G：LazyColumnBundle / scheduling_hints / incremental_event_service。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from factor_engine.backend.polars_lazy import LazyColumnBundle, build_lazy_column_bundle
from factor_engine.planner.scheduling_hints import derive_scheduling_hints
from factor_engine.runtime.incremental_event_service import normalize_data_event, plan_incremental_from_event
from factor_engine.runtime.incremental_scheduler import DataEvent


def test_derive_scheduling_hints_high_memory():
    summary = {
        "high_memory_ops": ["ts_rank", "neutralize"],
        "tier_histogram": {"3": 2},
        "factor_count": 2,
    }
    out = derive_scheduling_hints(summary)
    actions = {h["action"] for h in out["hints"]}
    assert "materialize_sharded" in actions
    assert "prefer_polars_lazy" in actions
    assert out["recommended"].get("shard_by") == "asset_bucket"


def test_derive_scheduling_hints_parallel_batch():
    summary = {
        "high_memory_ops": [],
        "tier_histogram": {"0": 5, "2": 3},
        "factor_count": 6,
    }
    out = derive_scheduling_hints(summary)
    actions = {h["action"] for h in out["hints"]}
    assert "run_many_parallel" in actions


def test_normalize_data_event_dict():
    ev = normalize_data_event(
        {"dataset": "d", "column": "close", "updated_date": "2026-07-09"}
    )
    assert isinstance(ev, DataEvent)
    assert ev.dataset == "d"


def test_lazy_bundle_materialize_uses_single_collect():
    pytest.importorskip("polars")
    import polars as pl

    lf = pl.LazyFrame(
        {
            "ts": [1, 2, 3],
            "inst": ["A", "A", "B"],
            "close": [1.0, 2.0, 3.0],
            "open": [0.5, 1.5, 2.5],
        }
    )
    bundle = LazyColumnBundle(
        lf=lf,
        time_column="ts",
        instrument_column="inst",
        physical_columns=("close", "open"),
        output_names={},
        normalize_timestamp=False,
        timestamp_unit=None,
    )
    calls = {"n": 0}
    original_collect = pl.LazyFrame.collect

    def counted_collect(self, *args, **kwargs):
        calls["n"] += 1
        return original_collect(self, *args, **kwargs)

    idx = pd.MultiIndex.from_tuples(
        [(1, "A"), (2, "A"), (3, "B")],
        names=["timestamp", "instrument"],
    )
    with patch.object(pl.LazyFrame, "collect", counted_collect):
        with patch(
            "data_access.read.adapters.arrow_table_to_multiindex_columns",
            return_value={
                "close": pd.Series([1.0, 2.0, 3.0], index=idx),
                "open": pd.Series([0.5, 1.5, 2.5], index=idx),
            },
        ):
            first = bundle.materialize_columns(["close", "open"])
            second = bundle.materialize_columns(["close"])
    assert calls["n"] == 1
    assert set(first.keys()) == {"close", "open"}
    assert "close" in second


def test_plan_incremental_from_event_empty_catalog(tmp_path):
    out = plan_incremental_from_event(
        DataEvent(dataset="ds", column="close", updated_date="2026-07-01"),
        lake_root=tmp_path / "lake",
    )
    assert out["factor_count"] == 0
    assert out["event"]["column"] == "close"
