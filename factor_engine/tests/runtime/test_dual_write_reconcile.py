# -*- coding: utf-8
"""双写对账与日内增量窗口测试。"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from factor_engine.runtime.dual_write_reconcile import list_open_dual_write_failures, reconcile_dual_write_state
from factor_engine.runtime.incremental import build_incremental_plan
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.time_window import resolve_incremental_window_for_bar_freq


def test_resolve_incremental_window_intraday_uses_tick_precise_mode():
    window = resolve_incremental_window_for_bar_freq(
        watermark_end="2024-06-01",
        lookback_bars=20 * 78,
        recompute_tail_bars=5,
        bar_freq="5m",
    )
    assert window["window_mode"] == "intraday_tick_precise"
    assert window["load_start"] is not None
    approx = resolve_incremental_window_for_bar_freq(
        watermark_end="2024-06-01",
        lookback_bars=20 * 78,
        recompute_tail_bars=5,
        bar_freq="5m",
        use_tick_precise=False,
    )
    assert approx["window_mode"] == "intraday_calendar_approx"
    assert window["load_start"] > approx["load_start"]


def test_build_incremental_plan_intraday_window_mode():
    plan = build_incremental_plan(
        factor_id="intraday_f",
        analysis_lookback=20,
        watermark={"end_date": "2024-06-01"},
        factor_freq="1d",
        source_bar_freq="5m",
        lookback_extra=0,
    )
    assert plan.window_mode == "intraday_tick_precise"
    assert plan.lookback_bars >= 20 * 78


def test_reconcile_dual_write_state_no_failures(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.register(
        factor_id="f1",
        author="test",
        frequency="1d",
        ast_hash="abc",
    )
    cat.update_watermark("f1", "2024-01-01", "2024-01-10", row_count=10)
    report = reconcile_dual_write_state(factor_id="f1", lake_root=tmp_path)
    assert report["ok"] is True
    assert report["open_dual_write_failures"] == 0


def test_list_dual_write_failures_from_catalog(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.record_run(
        {
            "run_id": "run_bad",
            "factor_id": "f2",
            "factor_name": "f2",
            "ast_hash": "h",
            "operator_catalog_hash": "o",
            "lookback": 1,
            "referenced_columns": [],
            "extra": {
                "dual_write_failed": True,
                "dual_write_error": "ch down",
                "staging_written": True,
            },
        }
    )
    rows = list_open_dual_write_failures(lake_root=tmp_path, factor_id="f2")
    assert len(rows) == 1
    assert rows[0]["error"] == "ch down"
    report = reconcile_dual_write_state(factor_id="f2", lake_root=tmp_path)
    assert report["ok"] is False
    assert report["latest_failure"]["staging_written"] is True


def test_dual_write_closed_after_repair(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.record_run(
        {
            "run_id": "bad",
            "factor_id": "f3",
            "factor_name": "f3",
            "ast_hash": "h",
            "operator_catalog_hash": "o",
            "lookback": 1,
            "referenced_columns": [],
            "extra": {"dual_write_failed": True},
        }
    )
    cat.record_run(
        {
            "run_id": "fix",
            "factor_id": "f3",
            "factor_name": "f3",
            "ast_hash": "h",
            "operator_catalog_hash": "repair",
            "lookback": 0,
            "referenced_columns": [],
            "extra": {"dual_write_repaired": True},
        }
    )
    assert reconcile_dual_write_state(factor_id="f3", lake_root=tmp_path)["ok"] is True
    assert list_open_dual_write_failures(lake_root=tmp_path) == []
