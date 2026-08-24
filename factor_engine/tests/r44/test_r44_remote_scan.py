# -*- coding: utf-8 -*-
"""R44-P0: 远程扫描 / 零本地磁盘 / 远程 IO 成本 测试（dataaccess-only，不镜像 factor_engine）。

覆盖：
  1. LocalObjectStore head/list/range/put/multipart 往返；缺失 key → head None、
     range_read 抛错。
  2. 小 parquet 数据集（date + asset + close 列，按 date 分区）写入 store 后，
     RemoteDatasetScanner.scan 的投影 + date 区间 + universe 过滤精确返回预期 batch。
  3. resolve_snapshot 从 manifest JSON 解析对象 key。
  4. local_disk_policy：STRICT_REMOTE 下 tracked 写入后 assert 抛违规；
     HIGH_PERFORMANCE 放行；spill_policy_for 尊重 env。
  5. estimate_remote_io 返回全字段 RemoteIOCost，total_bytes == 字节字段之和。
"""
from __future__ import annotations

import io
import json
import shutil
from datetime import date

import pytest

import pyarrow as pa
import pyarrow.parquet as pq

from dataaccess.read.object_store import LocalObjectStore
from dataaccess.read.remote_dataset_scanner import (
    RemoteDatasetScanner,
    RemoteScanPlan,
    RemoteScanStats,
)
from dataaccess.read.remote_io_cost import RemoteIOCost, estimate_remote_io
from dataaccess.read.local_disk_policy import (
    LocalDiskPolicy,
    LocalDiskPolicyViolation,
    assert_no_local_persistent_write,
    spill_policy_for,
    track_local_persistent_bytes_written,
)


# ---------------------------------------------------------------------------
# 1. LocalObjectStore 往返
# ---------------------------------------------------------------------------
def test_local_object_store_round_trip(tmp_path):
    store = LocalObjectStore(tmp_path / "objs")
    store.put_object("ds/2024-01-02.parquet", b"DATA-A")
    store.put_object("ds/2024-01-03.parquet", b"DATA-B")

    assert store.list_objects("ds/") == [
        "ds/2024-01-02.parquet",
        "ds/2024-01-03.parquet",
    ]
    head = store.head_object("ds/2024-01-02.parquet")
    assert head is not None
    assert head["size"] == len(b"DATA-A")
    assert "etag" in head

    assert store.range_read("ds/2024-01-02.parquet", offset=2, length=4) == b"TA-A"
    reader = store.open_reader("ds/2024-01-02.parquet")
    assert reader is not None
    assert reader.read() == b"DATA-A"
    reader.close()

    # 缺失 key
    assert store.head_object("missing.parquet") is None
    assert store.open_reader("missing.parquet") is None
    with pytest.raises(Exception):
        store.range_read("missing.parquet", offset=0, length=10)


def test_local_object_store_multipart(tmp_path):
    store = LocalObjectStore(tmp_path / "objs")
    upload_id = store.begin_multipart("big/blob.bin")
    store.upload_part(upload_id, "big/blob.bin", 0, b"part0-")
    store.upload_part(upload_id, "big/blob.bin", 1, b"part1")
    store.complete_multipart(upload_id, "big/blob.bin")
    assert store.range_read("big/blob.bin", offset=0, length=100) == b"part0-part1"
    # part staging 已清理
    assert store.list_objects("big/") == ["big/blob.bin"]

    # abort 清理
    upload_id2 = store.begin_multipart("big/dropped.bin")
    store.upload_part(upload_id2, "big/dropped.bin", 0, b"x")
    store.abort_multipart(upload_id2, "big/dropped.bin")
    assert store.head_object("big/dropped.bin") is None


# ---------------------------------------------------------------------------
# 2. RemoteDatasetScanner：投影 + date 区间 + universe 过滤
# ---------------------------------------------------------------------------
@pytest.fixture()
def scanned_store(tmp_path):
    root = tmp_path / "objs"
    store = LocalObjectStore(root)
    # 按 date 分区的三个对象。
    for day, assets, close in (
        ("2024-01-02", ("A", "B"), (10.0, 20.0)),
        ("2024-01-03", ("A", "C"), (30.0, 40.0)),
        ("2024-01-04", ("B", "C"), (50.0, 60.0)),
    ):
        table = pa.table(
            {"date": [day] * len(assets), "asset": list(assets), "close": list(close)}
        )
        buf = io.BytesIO()
        pq.write_table(table, buf)
        store.put_object(f"ds/{day}.parquet", buf.getvalue())

    manifest = {
        "dataset": "ds",
        "snapshot_id": "snap-1",
        "objects": [
            {
                "key": f"ds/{day}.parquet",
                "etag": f"e{i}",
                "size": store.head_object(f"ds/{day}.parquet")["size"],
            }
            for i, day in enumerate(("2024-01-02", "2024-01-03", "2024-01-04"))
        ],
    }
    store.put_object(
        "manifests/ds/snap-1.json", json.dumps(manifest).encode("utf-8")
    )
    return store


def test_scanner_projection_and_predicates(scanned_store):
    scanner = RemoteDatasetScanner(scanned_store)
    batches = list(
        scanner.scan(
            dataset="ds",
            snapshot_id="snap-1",
            fields=("date", "asset", "close"),
            start="2024-01-03",
            end="2024-01-03",
            universe=("A", "C"),
        )
    )
    # 逐 batch 摊平成行元组后比较（扫描器按对象切 batch，不按行切）。
    flat_rows = [
        (row["date"], row["asset"], row["close"])
        for b in batches
        for row in b.to_pylist()
    ]
    assert sorted(flat_rows) == [
        ("2024-01-03", "A", 30.0),
        ("2024-01-03", "C", 40.0),
    ]
    assert scanner.last_stats.rows_read == 2
    assert scanner.last_stats.object_count >= 1
    assert scanner.last_stats.batches >= 1


def test_scanner_projection_subset_columns(scanned_store):
    scanner = RemoteDatasetScanner(scanned_store)
    batches = list(
        scanner.scan(
            dataset="ds",
            snapshot_id="snap-1",
            fields=("close",),
            start="2024-01-02",
            end="2024-01-02",
            universe=("B",),
        )
    )
    assert len(batches) == 1
    assert batches[0].column_names == ["close"]
    assert batches[0].to_pydict() == {"close": [20.0]}


def test_scanner_no_predicate_reads_all(scanned_store):
    scanner = RemoteDatasetScanner(scanned_store)
    batches = list(
        scanner.scan(dataset="ds", snapshot_id="snap-1", fields=("date", "asset", "close"))
    )
    assert sum(b.num_rows for b in batches) == 6


def test_scanner_to_tables(scanned_store):
    scanner = RemoteDatasetScanner(scanned_store)
    tables = scanner.scan_to_tables(
        dataset="ds",
        snapshot_id="snap-1",
        fields=("date", "asset", "close"),
        start="2024-01-04",
        end="2024-01-04",
    )
    assert len(tables) == 1
    assert tables[0].num_rows == 2
    assert tables[0].column_names == ["date", "asset", "close"]


# ---------------------------------------------------------------------------
# 3. resolve_snapshot
# ---------------------------------------------------------------------------
def test_resolve_snapshot_manifest(scanned_store):
    scanner = RemoteDatasetScanner(scanned_store)
    keys = scanner.resolve_snapshot("ds", "snap-1")
    assert keys == [
        "ds/2024-01-02.parquet",
        "ds/2024-01-03.parquet",
        "ds/2024-01-04.parquet",
    ]


def test_resolve_snapshot_mismatch_raises(scanned_store):
    scanner = RemoteDatasetScanner(scanned_store)
    with pytest.raises(ValueError):
        scanner.resolve_snapshot("ds", "snap-2")  # snapshot_id 不匹配


def test_resolve_snapshot_custom_loader(scanned_store):
    manifest = {
        "dataset": "ds",
        "snapshot_id": "s1",
        "objects": [{"key": "ds/2024-01-03.parquet"}],
    }
    scanner = RemoteDatasetScanner(
        scanned_store, manifest_loader=lambda store, ds, sid: manifest
    )
    assert scanner.resolve_snapshot("ds", "s1") == ["ds/2024-01-03.parquet"]


# ---------------------------------------------------------------------------
# 4. 零本地磁盘策略
# ---------------------------------------------------------------------------
def test_strict_remote_tracked_write_fails_closed(monkeypatch):
    # 测试进程默认可能继承 FACTOR_ENGINE_PRODUCTION；这里显式切 STRICT_REMOTE，
    # 保证 fail-closed 断言与机器环境无关。
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "STRICT_REMOTE")
    monkeypatch.delenv("FACTOR_ENGINE_PRODUCTION", raising=False)
    with track_local_persistent_bytes_written() as track:
        track(4096)  # 模拟一次本地持久写入
        with pytest.raises(LocalDiskPolicyViolation):
            assert_no_local_persistent_write()


def test_high_performance_tracked_write_allowed(monkeypatch):
    monkeypatch.setenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", "HIGH_PERFORMANCE")
    with track_local_persistent_bytes_written() as track:
        track(4096)
        assert_no_local_persistent_write()  # 放行，不抛


def test_no_tracked_write_no_violation():
    with track_local_persistent_bytes_written():
        assert_no_local_persistent_write()  # 0 字节 → 通过


def test_spill_policy_for_env(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_PRODUCTION", raising=False)
    monkeypatch.delenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", raising=False)
    assert spill_policy_for() is LocalDiskPolicy.HIGH_PERFORMANCE

    assert spill_policy_for({"FACTOR_ENGINE_LOCAL_DISK_POLICY": "STRICT_REMOTE"}) is (
        LocalDiskPolicy.STRICT_REMOTE
    )
    assert spill_policy_for({"FACTOR_ENGINE_LOCAL_DISK_POLICY": "strict_remote"}) is (
        LocalDiskPolicy.STRICT_REMOTE
    )
    assert spill_policy_for({"FACTOR_ENGINE_LOCAL_DISK_POLICY": "high_performance"}) is (
        LocalDiskPolicy.HIGH_PERFORMANCE
    )
    # production 缺省显式策略 → STRICT_REMOTE
    assert spill_policy_for({"FACTOR_ENGINE_PRODUCTION": "1"}) is (
        LocalDiskPolicy.STRICT_REMOTE
    )
    assert spill_policy_for({"FACTOR_ENGINE_PRODUCTION": "true"}) is (
        LocalDiskPolicy.STRICT_REMOTE
    )
    # 非法值 fail-closed
    with pytest.raises(ValueError):
        spill_policy_for({"FACTOR_ENGINE_LOCAL_DISK_POLICY": "BOGUS"})


def test_strict_remote_via_production_tracks_and_fails(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_LOCAL_DISK_POLICY", raising=False)
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION", "1")
    assert spill_policy_for() is LocalDiskPolicy.STRICT_REMOTE
    with track_local_persistent_bytes_written() as track:
        track(1)
        with pytest.raises(LocalDiskPolicyViolation):
            assert_no_local_persistent_write()


# ---------------------------------------------------------------------------
# 5. 远程 IO 成本模型
# ---------------------------------------------------------------------------
def test_estimate_remote_io_fields_and_total():
    plan = RemoteScanPlan(
        dataset="ds",
        snapshot_id="snap-1",
        fields=("close",),
        start="2024-01-01",
        end="2024-01-31",
        universe=("A", "B"),
        object_keys=("ds/a.parquet", "ds/b.parquet"),
    )
    cost = estimate_remote_io(plan)
    assert isinstance(cost, RemoteIOCost)
    assert cost.range_read_bytes > 0
    assert cost.get_count > 0
    assert cost.network_transfer_bytes > 0
    assert cost.multipart_sink_bytes >= 0
    assert cost.time_to_durable_commit_ms > 0
    assert cost.total_bytes() == (
        cost.network_transfer_bytes + cost.multipart_sink_bytes
    )
    data = cost.estimate()
    assert data["total_bytes"] == cost.total_bytes()
    assert set(data) == {
        "range_read_bytes",
        "get_count",
        "network_transfer_bytes",
        "multipart_sink_bytes",
        "time_to_durable_commit_ms",
        "total_bytes",
    }


def test_estimate_remote_io_custom_params():
    plan = RemoteScanPlan(
        dataset="ds",
        snapshot_id="s",
        fields=("close",),
        object_keys=("ds/a.parquet",),
    )
    cost = estimate_remote_io(plan, bytes_per_row=64.0, object_overhead_bytes=1024)
    assert cost.network_transfer_bytes > 1024
