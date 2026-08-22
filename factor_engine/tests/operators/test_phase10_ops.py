# -*- coding: utf-8 -*-
"""Phase 10：SLO、灾备复制、Redis 队列工厂。"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.lake_replication import disaster_recovery_verify, replicate_lake_directory
from runtime.slo_rules import evaluate_slo_rules
from runtime.task_queue import FileTaskQueue, build_task_queue


def test_evaluate_slo_rules_pass():
    out = evaluate_slo_rules({"pipeline_success_ratio": 0.995, "dual_write_open_failures": 0})
    assert out["ok"] is True


def test_evaluate_slo_rules_violation():
    out = evaluate_slo_rules({"pipeline_success_ratio": 0.5})
    assert out["ok"] is False
    assert out["violations"]


def test_replicate_lake_directory_dry_run(tmp_path):
    src = tmp_path / "primary"
    (src / "factors" / "f1").mkdir(parents=True)
    (src / "factors" / "f1" / "x.txt").write_text("1", encoding="utf-8")
    dst = tmp_path / "replica"
    report = replicate_lake_directory(src, dst, dry_run=True)
    assert report["ok"] is True
    assert not dst.exists()


def test_disaster_recovery_verify(tmp_path):
    primary = tmp_path / "primary"
    replica = tmp_path / "replica"
    (primary / "factors" / "a").mkdir(parents=True)
    (replica / "factors" / "a").mkdir(parents=True)
    out = disaster_recovery_verify(primary, replica)
    assert out["ok"] is True


def test_build_task_queue_file(tmp_path):
    q = build_task_queue(backend="file", root=tmp_path / "q")
    assert isinstance(q, FileTaskQueue)
    job = q.enqueue("/tmp/cfg.yaml")
    assert q.stats()["pending"] == 1
    claimed = q.claim()
    assert claimed is not None
    q.complete(claimed.job_id)


def test_build_task_queue_object_store_delegates_local(tmp_path):
    from runtime.task_queue import ObjectStoreTaskQueue

    q = build_task_queue(backend="object_store", root=tmp_path / "objq")
    assert isinstance(q, ObjectStoreTaskQueue)
    job = q.enqueue("/tmp/cfg2.yaml")
    assert q.stats()["pending"] == 1
    claimed = q.claim()
    assert claimed is not None
    q.complete(claimed.job_id)


def test_build_task_queue_redis_requires_package():
    with pytest.raises(ImportError):
        build_task_queue(backend="redis", redis_url="redis://localhost:6379/0")
