"""Phase 8：Prometheus/OTLP、并行调度、publish 审批、snapshot 对账。"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from factor_engine.runtime.metrics_export import (
    to_otlp_json,
    to_prometheus_text,
    write_otlp_metrics_file,
    write_prometheus_metrics_file,
)
from factor_engine.runtime.snapshot_reconcile import reconcile_data_snapshot
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.lake_publish import (
    PublishNotApprovedError,
    publish_factor_lake,
    sync_local_factor_to_staging,
)
from factor_engine.storage.materializer import ParquetMaterializer


def test_prometheus_text_contains_core_gauges():
    summary = {
        "metrics": {
            "pipeline": {
                "configs_total": 3,
                "configs_success": 2,
                "configs_failed": 1,
                "total_result_rows": 100,
                "total_materialized_rows": 80,
                "incremental_runs": 1,
            },
            "data_access": {"available": False, "operators": {}},
            "per_config": [],
        }
    }
    text = to_prometheus_text(summary)
    assert "factor_engine_configs_total 3" in text
    assert "factor_engine_configs_success 2" in text
    assert "factor_engine_total_materialized_rows 80" in text


def test_otlp_json_has_resource_metrics():
    summary = {
        "metrics": {
            "exported_at": "2026-07-08T00:00:00+00:00",
            "pipeline": {"configs_total": 1, "configs_success": 1, "configs_failed": 0},
        }
    }
    payload = to_otlp_json(summary)
    assert "resourceMetrics" in payload
    metrics = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    names = {item["name"] for item in metrics}
    assert "factor_engine.configs_total" in names


def test_write_prometheus_and_otlp_files(tmp_path):
    summary = {"metrics": {"pipeline": {"configs_total": 1, "configs_success": 1, "configs_failed": 0}}}
    prom = write_prometheus_metrics_file(summary, tmp_path / "m.prom")
    otlp = write_otlp_metrics_file(summary, tmp_path / "m.otlp.json")
    assert "factor_engine_configs_total" in prom.read_text(encoding="utf-8")
    assert "resourceMetrics" in otlp.read_text(encoding="utf-8")


def _make_series() -> pd.Series:
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "AAA"),
            (pd.Timestamp("2024-01-03"), "AAA"),
        ],
        names=["timestamp", "instrument"],
    )
    return pd.Series([1.0, 2.0], index=idx)


def test_reconcile_data_snapshot_ok(tmp_path):
    from factor_engine.runtime.lineage import hash_data_source_config

    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake)
    ds_cfg = {"type": "data_access", "dataset": "ashare_stock_daily", "fields": {"close": "Close"}}
    snap = hash_data_source_config(ds_cfg)
    mat.materialize(
        factor_id="snap_ok",
        result=_make_series(),
        ast_hash="h1",
        data_source_config=ds_cfg,
        data_snapshot_id=snap,
        write_metadata=True,
        run_lineage={
            "run_id": "r1",
            "factor_id": "snap_ok",
            "ast_hash": "h1",
            "referenced_columns": ["close"],
            "extra": {"data_snapshot_id": snap},
        },
    )
    report = reconcile_data_snapshot(
        factor_id="snap_ok",
        lake_root=lake,
        data_source_config=ds_cfg,
        catalog=mat.catalog,
    )
    assert report["expected_snapshot_id"] == snap
    assert report["registry_snapshot_id"] == snap
    assert report["latest_run_snapshot_id"] == snap
    assert report["ok"] is True


def test_reconcile_data_snapshot_uses_registry_without_config(tmp_path):
    from factor_engine.runtime.lineage import hash_data_source_config

    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake)
    ds_cfg = {"type": "data_access", "dataset": "ashare_stock_daily", "fields": {"close": "Close"}}
    snap = hash_data_source_config(ds_cfg)
    mat.materialize(
        factor_id="snap_registry",
        result=_make_series(),
        ast_hash="h1",
        data_source_config=ds_cfg,
        data_snapshot_id=snap,
        write_metadata=True,
        run_lineage={
            "run_id": "r2",
            "factor_id": "snap_registry",
            "ast_hash": "h1",
            "referenced_columns": ["close"],
            "extra": {"data_snapshot_id": snap},
        },
    )
    report = reconcile_data_snapshot(
        factor_id="snap_registry",
        lake_root=lake,
        data_source_config=None,
        catalog=mat.catalog,
    )
    assert report["expected_snapshot_id"] == snap
    assert report["ok"] is True


def test_publish_requires_approval(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    mat = ParquetMaterializer(lake_root=lake)
    mat.materialize(factor_id="pub_test", result=_make_series(), ast_hash="h1")

    monkeypatch.delenv("QUANT_PUBLISH_APPROVED", raising=False)
    with pytest.raises(PublishNotApprovedError):
        publish_factor_lake(factor_id="pub_test", lake_root=lake, approve=False, reconcile=False)


def test_publish_with_approve_uses_staging(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    lake = tmp_path / "lake"
    workspace.mkdir()
    lake.mkdir()

    monkeypatch.setenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT", str(workspace))
    monkeypatch.setenv("FACTOR_LAKE_ROOT", str(lake))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "phase8test")
    monkeypatch.setenv("QUANT_OPERATOR", "phase8@test")
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")

    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        dedent(
            f"""
            factor_lake:
              kind: parametric
              access_mode: published
              layout: hive
              root_template: {lake}/factors/{{factor_id}}
              glob_template: "year=*/*.parquet"
              params_schema:
                factor_id: str
              time_column: datetime
              instrument_column: asset
              hive_partitioning: true
              union_by_name: true

            factor_lake_staging:
              kind: parametric
              access_mode: staging
              layout: hive
              root_template: {workspace}/staging/${{RUN_NAMESPACE}}/factor_lake/factors/{{factor_id}}
              glob_template: "year=*/*.parquet"
              params_schema:
                factor_id: str
              time_column: datetime
              instrument_column: asset
              hive_partitioning: true
              union_by_name: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(yaml_path))

    from data_access import reset_store

    reset_store()

    local_lake = tmp_path / "local_lake"
    mat = ParquetMaterializer(lake_root=local_lake)
    mat.materialize(factor_id="pub_ok", result=_make_series(), ast_hash="h1")

    sync = sync_local_factor_to_staging(factor_id="pub_ok", lake_root=local_lake)
    assert sync["rows"] == 2

    result = publish_factor_lake(
        factor_id="pub_ok",
        lake_root=local_lake,
        approve=True,
        sync_from_local=False,
        reconcile=False,
    )
    assert result["publish"]["rows"] == 2
    assert (lake / "factors" / "pub_ok").exists()


def test_pipeline_config_dir_parallel_runs(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for idx in range(2):
        (cfg_dir / f"factor_{idx}.yaml").write_text(
            dedent(
                f"""
                factor:
                  name: smoke_{idx}
                  expr: close
                data_source:
                  type: parquet
                  root: {tmp_path}/data
                  timestamp_col: TradeDate
                  instrument_col: Symbol
                  fields:
                    close: Close
                  max_files: 1
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    def _fake_execute(config, *, config_name, **kwargs):
        return {
            "config_name": config_name,
            "factor_name": config.factor.name,
            "status": "success",
            "mode": "run",
            "analysis": {"lookback": 0, "referenced_columns": []},
            "plan": {"root_op": "col", "node_count": 1},
            "result": {"row_count": 1, "non_null_count": 1},
            "errors": [],
        }

    monkeypatch.setattr("factor_engine.pipeline._execute_config_with_retries", _fake_execute)

    from factor_engine.pipeline import run_config_directory

    out = run_config_directory(cfg_dir, output_root=tmp_path / "out", n_jobs=2)
    assert out["summary"]["configs_total"] == 2
    assert out["summary"]["configs_success"] == 2
    assert (tmp_path / "out" / "metrics.prom").exists()
    assert (tmp_path / "out" / "metrics.otlp.json").exists()
