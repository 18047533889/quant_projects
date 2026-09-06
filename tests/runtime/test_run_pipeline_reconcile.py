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

from factor_engine.runtime.dual_write_reconcile import (
    reconcile_all_dual_write_states,
    reconcile_dual_write_state,
    repair_dual_write_clickhouse,
)
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.materializer import ParquetMaterializer


@pytest.fixture(autouse=True)
def _isolated_staging_root(tmp_path, monkeypatch):
    """Keep every staging read/delete inside this test's temporary workspace."""
    from data_access import reset_store

    workspace = tmp_path / "data_access_workspace"
    monkeypatch.setenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT", str(workspace))
    monkeypatch.setenv("RUN_NAMESPACE", "dual_write_reconcile_test")
    reset_store()
    yield workspace
    reset_store()


def _write_summary(factor_id: str, rows: int):
    return MagicMock(
        factor_id=factor_id,
        table="fv",
        rows_written=rows,
        database="quant",
    )


def test_repair_dual_write_from_local_parquet(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["A"]],
        names=["timestamp", "instrument"],
    )
    series = pd.Series([1.0, 2.0, 3.0], index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(factor_id="repair_me", result=series, ast_hash="h1")
    mat.catalog.record_run(
        {
            "run_id": "failed-before-repair",
            "factor_id": "repair_me",
            "factor_name": "repair_me",
            "ast_hash": "h1",
            "operator_catalog_hash": "test",
            "lookback": 0,
            "referenced_columns": [],
            "extra": {"dual_write_failed": True},
        }
    )

    mock_summary = _write_summary("repair_me", 3)
    with patch(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(
            factor_id="repair_me",
            lake_root=tmp_path,
            clickhouse_table="fv",
        )
    assert out["ok"] is False
    assert out["status"] == "SUBMITTED_UNVERIFIED"
    assert out["load_source"] == "local"
    assert out["rows_submitted"] == 3
    assert out["staging_retained"] is True
    assert reconcile_dual_write_state(
        factor_id="repair_me", lake_root=tmp_path
    )["dual_write_still_open"] is True
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

    mock_summary = _write_summary("stg_only", 2)
    monkeypatch.setattr(
        "factor_engine.storage.staging_loader.load_factor_series_from_staging",
        _fake_staging,
    )
    with patch(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(factor_id="stg_only", lake_root=tmp_path)
    assert out["ok"] is False
    assert out["status"] == "SUBMITTED_UNVERIFIED"
    assert out["load_source"] == "staging"


def test_dual_write_closed_after_repair_run(tmp_path):
    cat = FactorCatalog(tmp_path / "_catalog.sqlite")
    cat.register(factor_id="f3", author="test", frequency="1d", ast_hash="h")
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
    cat.register(factor_id="f1", author="test", frequency="1d", ast_hash="h")
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
        "factor_engine.storage.staging_loader.delete_staging_rows",
        _skip_delete,
    )
    mock_summary = _write_summary("no_comp", 2)
    with patch(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(
            factor_id="no_comp",
            lake_root=tmp_path,
            compensate_staging=False,
        )
    assert out["status"] == "SUBMITTED_UNVERIFIED"
    assert called["n"] == 0


def test_repair_dual_write_retains_staging_by_default(tmp_path, monkeypatch):
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
        "factor_engine.storage.staging_loader.delete_staging_rows",
        _do_delete,
    )
    mock_summary = _write_summary("with_comp", 2)
    with patch(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=mock_summary,
    ):
        out = repair_dual_write_clickhouse(factor_id="with_comp", lake_root=tmp_path)
    assert called["n"] == 0
    assert out["staging_retained"] is True


@pytest.mark.parametrize("summary", [None, pytest.param(_write_summary("partial", 1), id="partial")])
def test_unverified_or_partial_write_never_deletes_staging(tmp_path, monkeypatch, summary):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(
        factor_id="partial",
        result=pd.Series([1.0, 2.0], index=idx),
        ast_hash="h1",
    )
    monkeypatch.setattr(
        "factor_engine.storage.staging_loader.delete_staging_rows",
        lambda *_args, **_kwargs: pytest.fail("staging must be retained"),
    )
    with patch(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        return_value=summary,
    ):
        out = repair_dual_write_clickhouse(
            factor_id="partial", lake_root=tmp_path, compensate_staging=True
        )

    expected_status = "WRITE_RESULT_UNPROVEN" if summary is None else "SUBMITTED_UNVERIFIED"
    assert out["status"] == expected_status
    assert out["staging_retained"] is True
    assert out["staging_compensation"]["reason"] == "retained_for_reconciliation_unverified"


def test_clickhouse_exception_preserves_staging_and_watermark(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=2), ["A"]],
        names=["timestamp", "instrument"],
    )
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(
        factor_id="write_error",
        result=pd.Series([1.0, 2.0], index=idx),
        ast_hash="h1",
    )
    before = mat.catalog.get_watermark("write_error")
    monkeypatch.setattr(
        "factor_engine.storage.staging_loader.delete_staging_rows",
        lambda *_args, **_kwargs: pytest.fail("staging must be retained"),
    )
    with patch(
        "factor_engine.storage.clickhouse_materializer.ClickHouseMaterializer.materialize",
        side_effect=RuntimeError("clickhouse unavailable"),
    ):
        with pytest.raises(RuntimeError, match="clickhouse unavailable"):
            repair_dual_write_clickhouse(
                factor_id="write_error", lake_root=tmp_path, compensate_staging=True
            )
    assert mat.catalog.get_watermark("write_error") == before


@pytest.mark.parametrize("cwd_kind", ["repository", "package"])
def test_run_pipeline_reconcile_dual_write_cli(tmp_path, cwd_kind):
    import factor_engine

    package_root = Path(factor_engine.__file__).resolve().parent
    repository_root = package_root.parent
    cwd = repository_root if cwd_kind == "repository" else package_root
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(repository_root)}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "factor_engine.run_pipeline",
            "reconcile",
            "dual-write",
            "--lake-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload.get("ok") is True
