"""Phase 7：metrics 导出测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from factor_engine.runtime.metrics_export import (
    enrich_pipeline_summary,
    export_factor_runs_jsonl,
    summarize_pipeline_results,
    write_pipeline_metrics_file,
)
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.materializer import ParquetMaterializer


def test_summarize_pipeline_results_counts_rows():
    results = [
        {
            "status": "success",
            "config_name": "a",
            "mode": "materialize",
            "result": {"row_count": 10, "non_null_count": 9},
            "materialization": {"rows_written": 10, "run_id": "r1"},
        },
        {"status": "failed", "config_name": "b"},
    ]
    metrics = summarize_pipeline_results(results)
    assert metrics["configs_total"] == 2
    assert metrics["configs_success"] == 1
    assert metrics["total_result_rows"] == 10
    assert metrics["total_materialized_rows"] == 10


def test_enrich_pipeline_summary_adds_metrics_block():
    summary = {"configs_total": 1}
    results = [{"status": "success", "config_name": "x", "result": {"row_count": 3}}]
    enriched = enrich_pipeline_summary(summary, results)
    assert "metrics" in enriched
    assert enriched["metrics"]["pipeline"]["configs_success"] == 1


def test_export_factor_runs_jsonl(tmp_path):
    catalog = FactorCatalog(tmp_path / "catalog.sqlite")
    catalog.register("f1", author="u", frequency="1d", ast_hash="abc")
    catalog.record_run(
        {
            "run_id": "run-1",
            "factor_id": "f1",
            "ast_hash": "abc",
            "row_count": 5,
            "non_null_count": 5,
            "referenced_columns": ["close"],
        }
    )
    out = tmp_path / "runs.jsonl"
    count = export_factor_runs_jsonl(catalog, out, factor_id="f1")
    assert count == 1
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert line["run_id"] == "run-1"


def test_materializer_partition_isolation_allows_partial_success(tmp_path, monkeypatch):
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-12-31"), "A"),
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-03"), "A"),
                (pd.Timestamp("2024-01-04"), "A"),
            ],
            names=["timestamp", "instrument"],
        ),
    )
    mat = ParquetMaterializer(lake_root=tmp_path)
    calls: list[int] = []

    original = mat._upsert_partition

    def flaky_upsert(factor_dir, part_values, new_df, *, policy):
        year = int(part_values.get("year", 0))
        calls.append(year)
        if year == 2024:
            raise OSError("simulated partition failure")
        return original(factor_dir, part_values, new_df, policy=policy)

    monkeypatch.setattr(mat, "_upsert_partition", flaky_upsert)

    with pytest.raises(Exception, match="分区落盘部分失败"):
        mat.materialize(
            factor_id="part_fail",
            result=series,
            ast_hash="hash1",
            isolate_partition_failures=True,
        )

    assert 2023 in calls and 2024 in calls
    failed = mat.catalog.list_partition_checkpoints("part_fail", status="failed")
    assert failed and failed[0]["partition_year"] == 2024
    assert (tmp_path / "factors" / "part_fail" / "year=2023" / "data.parquet").exists()


def test_materializer_resume_skips_successful_partition(tmp_path):
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-12-31"), "A"),
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-03"), "A"),
                (pd.Timestamp("2024-01-04"), "A"),
            ],
            names=["timestamp", "instrument"],
        ),
    )
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.catalog.record_partition_checkpoint(
        factor_id="resume_test",
        partition_year=2023,
        run_id="prev-run",
        status="success",
    )

    calls: list[int] = []

    def track_upsert(factor_dir, part_values, new_df, *, policy):
        year = int(part_values.get("year", 0))
        calls.append(year)
        return ParquetMaterializer._upsert_partition(
            mat, factor_dir, part_values, new_df, policy=policy
        )

    mat._upsert_partition = track_upsert  # type: ignore[method-assign]
    summary = mat.materialize(
        factor_id="resume_test",
        result=series,
        ast_hash="hash1",
        resume=True,
    )
    assert 2023 not in calls
    assert 2024 in calls
    assert summary["partitions_skipped"] == [2023]
