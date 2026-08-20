# -*- coding: utf-8
"""run_pipeline reconcile 子命令与双写修复测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from runtime.dual_write_reconcile import (
    reconcile_all_dual_write_states,
    reconcile_dual_write_state,
    repair_dual_write_clickhouse,
)
from storage.catalog import FactorCatalog
from storage.materializer import ParquetMaterializer


def test_repair_dual_write_from_local_parquet(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(factor_id="repair_me", result=series, ast_hash="h1")

    mock_summary = MagicMock(
        factor_id="repair_me",
        table="fv",
        rows_written=3,
        database="quant",
    )
    with patch(
        "storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(
            factor_id="repair_me",
            lake_root=tmp_path,
            clickhouse_table="fv",
        )
    assert out["ok"] is True
    assert out["load_source"] == "local"
    assert out["rows_repaired"] == 3
    wm = mat.catalog.get_watermark("repair_me")
    assert wm is not None


def test_repair_without_data_returns_guidance(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.register(factor_id="no_local", author="t", frequency="1d", ast_hash="h")
    out = repair_dual_write_clickhouse(factor_id="no_local", lake_root=tmp_path)
    assert out["ok"] is False
    assert out["reason"] == "no_data_source"


def test_repair_from_staging_when_no_local(tmp_path, monkeypatch):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.register(factor_id="stg_only", author="t", frequency="1d", ast_hash="h")

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["X"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0], index=idx)

    def _fake_staging(fid, **kwargs):
        assert fid == "stg_only"
        return series

    mock_summary = MagicMock(rows_written=2, table="fv", database="q", factor_id="stg_only")
    monkeypatch.setattr(
        "storage.staging_loader.load_factor_series_from_staging",
        _fake_staging,
    )
    with patch(
        "storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(factor_id="stg_only", lake_root=tmp_path)
    assert out["ok"] is True
    assert out["load_source"] == "staging"


def test_dual_write_closed_after_repair_run(tmp_path):
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
            "run_id": "fixed",
            "factor_id": "f3",
            "factor_name": "f3",
            "ast_hash": "h",
            "operator_catalog_hash": "repair",
            "lookback": 0,
            "referenced_columns": [],
            "extra": {"dual_write_repaired": True},
        }
    )
    report = reconcile_dual_write_state(factor_id="f3", lake_root=tmp_path)
    assert report["ok"] is True
    assert report["dual_write_still_open"] is False


def test_reconcile_all_dual_write_states(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.record_run(
        {
            "run_id": "r1",
            "factor_id": "f1",
            "factor_name": "f1",
            "ast_hash": "h",
            "operator_catalog_hash": "o",
            "lookback": 1,
            "referenced_columns": [],
            "extra": {"dual_write_failed": True, "dual_write_error": "down"},
        }
    )
    report = reconcile_all_dual_write_states(lake_root=tmp_path)
    assert report["factors_with_failures"] == 1
    assert report["reports"][0]["factor_id"] == "f1"


def test_repair_dual_write_respects_no_compensate_staging(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(factor_id="no_comp", result=series, ast_hash="h1")

    called = {"n": 0}

    def _skip_delete(*args, **kwargs):
        called["n"] += 1
        return {"deleted_rows": 0}

    monkeypatch.setattr(
        "storage.staging_loader.delete_staging_rows",
        _skip_delete,
    )
    mock_summary = MagicMock(rows_written=2, table="fv", database="q", factor_id="no_comp")
    with patch(
        "storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(
            factor_id="no_comp",
            lake_root=tmp_path,
            compensate_staging=False,
        )
    assert out["ok"] is True
    assert called["n"] == 0


def test_repair_dual_write_compensates_staging_by_default(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0], index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(factor_id="with_comp", result=series, ast_hash="h1")

    called = {"n": 0}

    def _do_delete(*args, **kwargs):
        called["n"] += 1
        return {"deleted_rows": 1}

    monkeypatch.setattr(
        "storage.staging_loader.delete_staging_rows",
        _do_delete,
    )
    mock_summary = MagicMock(rows_written=2, table="fv", database="q", factor_id="with_comp")
    with patch(
        "storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        repair_dual_write_clickhouse(factor_id="with_comp", lake_root=tmp_path)
    assert called["n"] == 1


from tests.helpers import FE_ROOT


def test_run_pipeline_reconcile_dual_write_cli(tmp_path):
    run_pipeline = FE_ROOT / "run_pipeline.py"
    env = {**dict(__import__("os").environ), "PYTHONPATH": f"{FE_ROOT.parent}:{FE_ROOT}"}
    proc = subprocess.run(
        [
            sys.executable,
            str(run_pipeline),
            "reconcile",
            "dual-write",
            "--lake-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(FE_ROOT),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload.get("ok") is True
