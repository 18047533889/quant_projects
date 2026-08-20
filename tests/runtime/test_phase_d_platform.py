"""Phase D：dependency catalog、队列重试、perf 阈值、bucket 校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.dependency_catalog import DependencyCatalog
from runtime.incremental_scheduler import DataEvent
from runtime.task_queue import FileTaskQueue
from storage.catalog import FactorCatalog


def test_dependency_catalog_reverse_index(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    catalog.record_factor_dependency(
        "alpha_001",
        referenced_columns=["close", "volume"],
        lookback=20,
        source_dataset="us_stocks",
    )
    catalog.record_factor_dependency(
        "alpha_002",
        referenced_columns=["open"],
        lookback=5,
        source_dataset="us_stocks",
    )
    dep = DependencyCatalog(catalog)
    close_factors = dep.factors_for_column("close", dataset="us_stocks")
    assert len(close_factors) == 1
    assert close_factors[0]["factor_id"] == "alpha_001"

    idx = dep.reverse_index(["close", "open"])
    by_col = {s.column: s for s in idx}
    assert "alpha_001" in by_col["close"].factor_ids
    assert "alpha_002" in by_col["open"].factor_ids

    plans = dep.plan_for_event(
        DataEvent(dataset="us_stocks", column="close", updated_date="2026-07-09")
    )
    assert len(plans) == 1


def test_dependency_catalog_list_dataset(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    catalog.record_factor_dependency(
        "f1",
        referenced_columns=["close"],
        lookback=3,
        source_dataset="ds_a",
    )
    catalog.record_factor_dependency(
        "f2",
        referenced_columns=["close"],
        lookback=3,
        source_dataset="ds_b",
    )
    dep = DependencyCatalog(catalog)
    rows = dep.factors_for_dataset("ds_a")
    assert len(rows) == 1
    assert rows[0]["factor_id"] == "f1"


def test_queue_retry_or_fail(tmp_path):
    queue = FileTaskQueue(tmp_path / "queue")
    job = queue.enqueue("cfg.yaml")
    claimed = queue.claim()
    assert claimed is not None
    status = queue.retry_or_fail(claimed.job_id, error="boom", max_retries=2)
    assert status == "requeued"
    pending = list((tmp_path / "queue" / "pending").glob("*.json"))
    payload = json.loads(pending[0].read_text(encoding="utf-8"))
    assert payload["payload"]["attempts"] == 1

    claimed2 = queue.claim()
    status2 = queue.retry_or_fail(claimed2.job_id, error="boom2", max_retries=1)
    assert status2 == "failed"


def test_catalog_list_dependency_columns(tmp_path):
    catalog = FactorCatalog(tmp_path / "_catalog.sqlite")
    catalog.record_factor_dependency(
        "x1",
        referenced_columns=["close", "volume"],
        lookback=1,
        source_dataset="d",
    )
    cols = catalog.list_dependency_columns()
    assert cols == ["close", "volume"]
