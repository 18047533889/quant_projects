"""Phase 9：OTLP 推送、分片调度、版本回滚、audit 关联。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from runtime.metrics_export import push_otlp_http, to_otlp_json
from runtime.run_audit_correlate import correlate_runs_with_audit, read_audit_log
from runtime.task_queue import FileTaskQueue, shard_config_paths
from storage.catalog import FactorCatalog
from storage.lake_version import (
    diff_factor_trees,
    list_publish_archives,
    rollback_factor_publish,
    summarize_factor_tree,
)
from storage.materializer import ParquetMaterializer


def test_to_otlp_json_has_service_name():
    payload = to_otlp_json({"metrics": {"pipeline": {"configs_total": 1}}})
    attrs = payload["resourceMetrics"][0]["resource"]["attributes"]
    assert any(item["key"] == "service.name" for item in attrs)


def test_to_otlp_json_time_is_unix_nano():
    payload = to_otlp_json(
        {
            "metrics": {
                "exported_at": "2026-07-08T00:00:00+00:00",
                "pipeline": {"configs_total": 1},
            }
        }
    )
    point = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["gauge"]["dataPoints"][0]
    assert point["timeUnixNano"].isdigit()
    assert int(point["timeUnixNano"]) > 0


def test_push_otlp_grpc_falls_back_to_http():
    from runtime.metrics_export import push_otlp_grpc

    summary = {"metrics": {"pipeline": {"configs_total": 1, "configs_success": 1, "configs_failed": 0}}}
    result = push_otlp_grpc(summary, "127.0.0.1:4317", timeout_sec=0.2)
    assert result["transport"] == "http_fallback_from_grpc"
    assert result["ok"] is False


def test_push_otlp_http_handles_connection_error():
    summary = {"metrics": {"pipeline": {"configs_total": 1, "configs_success": 1, "configs_failed": 0}}}
    result = push_otlp_http(summary, "http://127.0.0.1:1", timeout_sec=0.2)
    assert result["ok"] is False


def test_shard_config_paths_is_stable():
    paths = [Path(f"/tmp/cfg_{i}.yaml") for i in range(10)]
    a = shard_config_paths(paths, shard_index=0, shard_count=3)
    b = shard_config_paths(paths, shard_index=0, shard_count=3)
    assert a == b
    assert len(a) <= len(paths)


def test_file_task_queue_roundtrip(tmp_path):
    queue = FileTaskQueue(tmp_path / "queue")
    job = queue.enqueue("/tmp/a.yaml", profile="dev")
    claimed = queue.claim()
    assert claimed is not None
    assert claimed.job_id == job.job_id
    queue.complete(job.job_id, result={"ok": True})
    assert (tmp_path / "queue" / "done" / f"{job.job_id}.json").exists()


def test_file_task_queue_requeue_stale_running(tmp_path):
    queue = FileTaskQueue(tmp_path / "queue")
    job = queue.enqueue("/tmp/b.yaml")
    claimed = queue.claim()
    assert claimed is not None
    moved = queue.requeue_stale_running()
    assert moved == 1
    assert queue.stats()["pending"] == 1
    assert queue.stats()["running"] == 0


def _seed_factor(tmp_path: Path, *, factor_id: str, values: list[float]) -> Path:
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "AAA"), (pd.Timestamp("2024-01-03"), "AAA")],
        names=["timestamp", "instrument"],
    )
    series = pd.Series(values, index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.materialize(factor_id=factor_id, result=series, ast_hash="h1")
    return tmp_path / "factors" / factor_id


def test_lake_version_diff_and_rollback(tmp_path):
    current = _seed_factor(tmp_path, factor_id="ver_test", values=[1.0, 2.0])
    archive = tmp_path / "factors" / "_archive" / "ver_test_backup"
    archive.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(current, archive)

    # mutate current
    _seed_factor(tmp_path, factor_id="ver_test", values=[9.0, 9.0])
    report = diff_factor_trees(archive, current)
    assert report["rows_delta"] == 0
    assert report["left"]["rows"] == 2

    result = rollback_factor_publish(factor_id="ver_test", lake_root=tmp_path, archive_path=archive)
    assert result["after"]["rows"] == 2


def test_list_publish_archives_detects_entries(tmp_path):
    archive = tmp_path / "factors" / "_archive" / "fid_20260101T000000Z_abcd1234"
    archive.mkdir(parents=True)
    found = list_publish_archives("fid", tmp_path)
    assert found == [archive]


def test_correlate_runs_with_audit_by_time(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    now = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(
            {
                "ts": now,
                "op": "publish",
                "dataset": "factor_lake",
                "ok": True,
                "params": {"factor_id": "corr_f"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    catalog = FactorCatalog(tmp_path / "catalog.sqlite")
    catalog.register("corr_f", author="u", frequency="1d", ast_hash="abc")
    catalog.record_run(
        {
            "run_id": "run-corr",
            "factor_id": "corr_f",
            "ast_hash": "abc",
            "referenced_columns": ["close"],
            "created_at": now,
            "extra": {"data_snapshot_id": "snap1"},
        }
    )

    items = correlate_runs_with_audit(
        catalog,
        read_audit_log(audit_path),
        factor_id="corr_f",
        max_delta_seconds=600,
    )
    assert len(items) == 1
    assert items[0]["audit_matches"]


def test_pipeline_shard_dispatch(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    for idx in range(6):
        (cfg_dir / f"f_{idx}.yaml").write_text(
            dedent(
                f"""
                factor:
                  name: smoke_{idx}
                  expr: close
                data_source:
                  type: parquet
                  root: {tmp_path}
                  timestamp_col: TradeDate
                  instrument_col: Symbol
                  fields:
                    close: Close
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

    monkeypatch.setattr("pipeline._execute_config_with_retries", _fake_execute)

    from pipeline import run_config_directory

    out = run_config_directory(
        cfg_dir,
        output_root=tmp_path / "out",
        shard_index=0,
        shard_count=3,
    )
    assert out["summary"]["configs_selected"] <= 6
    assert out["summary"]["shard_index"] == 0
