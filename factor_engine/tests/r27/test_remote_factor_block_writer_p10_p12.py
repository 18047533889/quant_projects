# -*- coding: utf-8 -*-
"""P0-10..12: RemoteFactorBlockWriter 直写 COS（bounded-memory）+ 新 MemoryLeaseKinds。

验证：
    (a) writer 把 block 写到 LocalObjectStore，CURRENT 只在全部 block 存在+校验后
        翻转；部分代次 resolve_current 不可见；
    (b) bounded-memory：小 broker 预算下写多因子，serializer 峰值≈单块（不整块
        artifact 一次 hold RAM），summary telemetry 可观测；
    (c) round-trip：writer → reader 回读 → float32 值一致；
    (d) abort 清理部分代次；
    (e) 新 MemoryLeaseKinds 注册进 soft-pool map，且 acquire/release 经 broker 可用。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from data_access.read.object_store import LocalObjectStore
from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
from factor_engine.runtime.resource_broker import ResourceBroker
from factor_engine.runtime.remote_factor_block_writer import (
    RemoteFactorBlockWriter,
    deserialize_block,
    make_remote_sink_writer,
    serialize_block,
)

NEW_KINDS = [
    MemoryLeaseKind.COS_READ_BUFFER,
    MemoryLeaseKind.PARQUET_DECODE,
    MemoryLeaseKind.REMOTE_RANGE_BUFFER,
    MemoryLeaseKind.FEATURE_BLOCK_ASSEMBLY,
    MemoryLeaseKind.COS_UPLOAD_PART,
    MemoryLeaseKind.COS_UPLOAD_INFLIGHT,
    MemoryLeaseKind.MANIFEST_BUFFER,
]


def _make_broker(hard: int) -> ResourceBroker:
    return ResourceBroker(
        hard_memory_limit=hard,
        cpu_slots=4,
        min_host_reserve_gb=0.0,
        min_host_reserve_fraction=0.0,
    )


def _make_writer(tmp_path, store=None, broker=None, **kw):
    store = store or LocalObjectStore(tmp_path / "obj")
    w = RemoteFactorBlockWriter(store, "factors/r27", broker=broker, **kw)
    return w, store


class _RemoteReader:
    """极简 reader：经 feature-block manifest 反序列化各 block 并按列回读。"""

    def __init__(self, store, writer):
        self.store = store
        self.manifest = writer.manifest()
        self.gid = writer.generation_id()
        self.prefix = writer.prefix

    def read(self, factor_id: str) -> np.ndarray:
        resolved = self.manifest.resolve(factor_id)
        assert len(resolved) == 1
        _partition, spec, column = resolved[0]
        full_key = f"{self.prefix}/{self.gid}/{spec.file}"
        head = self.store.head_object(full_key)
        payload = self.store.range_read(full_key, offset=0, length=int(head["size"]))
        block = deserialize_block(payload)
        return np.ascontiguousarray(block[:, column])


# ---- (e) 新 MemoryLeaseKinds 注册 + acquire/release -------------------------
def test_new_lease_kinds_exist_in_enum():
    for k in NEW_KINDS:
        assert k in MemoryLeaseKind
        assert str(k) == k.value


def test_new_lease_kinds_registered_in_soft_pool():
    from factor_engine.runtime.resource_broker import DEFAULT_SOFT_POOL_FRACTIONS

    for k in NEW_KINDS:
        assert k in DEFAULT_SOFT_POOL_FRACTIONS
        assert DEFAULT_SOFT_POOL_FRACTIONS[k] > 0.0
    assert sum(DEFAULT_SOFT_POOL_FRACTIONS.values()) <= 1.0


def test_new_lease_kinds_acquire_release_via_broker():
    broker = _make_broker(8 * 1024**3)
    for k in NEW_KINDS:
        lease = broker.acquire_memory(k, 1024**3)
        assert lease is not None, f"{k} acquire 失败"
        assert broker._lease_sum_bytes() == 1024**3
        lease.release()
        assert broker._lease_sum_bytes() == 0


def test_remote_io_lease_budget_resolves():
    broker = _make_broker(8 * 1024**3)
    budget = broker.current_remote_io_lease_budget()
    exec_budget = broker.execution_budget()
    assert 0 < budget <= exec_budget


# ---- serialization round-trip ----------------------------------------------
def test_serialize_deserialize_roundtrip():
    block = np.arange(2500 * 16, dtype=np.float32).reshape(2500, 16) + 0.5
    payload = serialize_block(block)
    back = deserialize_block(payload)
    assert back.shape == block.shape
    assert back.dtype == np.dtype("float32")
    np.testing.assert_array_equal(back, block)


# ---- (a) CURRENT 只在全部 block 上传后翻转 ----------------------------------
def test_current_flips_only_after_all_blocks_present(tmp_path):
    w, store = _make_writer(tmp_path, columns_per_block=4)
    rows = 500
    for i in range(9):
        w.submit(f"f{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    assert w.resolve_current() is None  # 尚未发布
    gid = w.finish()
    assert gid
    assert w.resolve_current() == gid
    gen_prefix = f"factors/r27/{gid}"
    objs = store.list_objects(gen_prefix)
    assert sum(1 for o in objs if o.endswith(".fb")) == 3
    assert w.block_count() == 3
    assert store.head_object("factors/r27/CURRENT.json") is not None


def test_partial_generation_not_visible_before_finish(tmp_path):
    w, store = _make_writer(tmp_path, columns_per_block=4)
    rows = 200
    for i in range(5):
        w.submit(f"p{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    assert w.block_count() == 1  # 前 4 列已 flush 出 1 个 block
    assert w.resolve_current() is None  # 但 CURRENT 未翻转


# ---- (d) abort 清理部分代次 --------------------------------------------------
def test_abort_cleans_partial_generation(tmp_path):
    w, store = _make_writer(tmp_path, columns_per_block=4)
    rows = 300
    for i in range(5):
        w.submit(f"a{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    assert w.block_count() == 1
    w.abort()
    assert w.resolve_current() is None
    assert w.buffered_factors() == 0
    current_keys = [k for k in store.list_objects("factors/r27/") if k.endswith("CURRENT.json")]
    assert len(current_keys) == 0


def test_aborted_generation_rejects_new_submit(tmp_path):
    w, store = _make_writer(tmp_path, columns_per_block=4)
    rows = 100
    for i in range(5):
        w.submit(f"z{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    w.abort()
    with pytest.raises(RuntimeError):
        w.submit("zz", np.arange(rows, dtype=np.float32))


# ---- (c) round-trip writer → reader -----------------------------------------
def test_roundtrip_values_match_float32(tmp_path):
    w, store = _make_writer(tmp_path, columns_per_block=4)
    rows = 1000
    n = 9
    for i in range(n):
        w.submit(f"f{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    w.finish()
    manifest = w.manifest()
    assert manifest is not None
    assert manifest.dtype == "float32"
    r = _RemoteReader(store, w)
    for i in range(n):
        fid = f"f{i:02d}"
        got = r.read(fid)
        expected = np.arange(rows, dtype=np.float32) + float(i)
        assert got.shape == (rows,)
        assert got.dtype == np.dtype("float32")
        np.testing.assert_array_equal(got, expected)


# ---- (b) bounded-memory -----------------------------------------------------
def test_bounded_memory_within_small_broker_budget(tmp_path):
    broker = _make_broker(64 * 1024**2)  # 64MB
    rows = 500
    cols_per_block = 8
    w, store = _make_writer(tmp_path, broker=broker, columns_per_block=cols_per_block)
    for i in range(20):
        w.submit(f"b{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    w.finish()
    s = w.summary()
    assert s["total_factors_written"] == 20
    assert s["block_count"] == 3
    assert s["serializer_peak_bytes"] > 0
    assert s["inflight_parts"] == 0  # 上传完成 → RAM 已释放
    assert s["upload_buffer_bytes"] == 0
    single_block_bytes = rows * cols_per_block * 4 + 4096
    assert s["serializer_peak_bytes"] <= single_block_bytes * 2


def test_bounded_memory_never_holds_whole_artifact(tmp_path):
    broker = _make_broker(256 * 1024**2)
    rows = 50_000
    cols_per_block = 16
    w, store = _make_writer(tmp_path, broker=broker, columns_per_block=cols_per_block)
    for i in range(64):
        w.submit(f"m{i:02d}", np.arange(rows, dtype=np.float32) + float(i))
    w.finish()
    s = w.summary()
    assert s["total_factors_written"] == 64
    all_at_once = 64 * rows * 4
    # 峰值 ≈ 单 block（3.2MB），而非整 artifact 的 1/3+。
    assert s["serializer_peak_bytes"] < all_at_once // 3
    assert s["serializer_peak_bytes"] <= rows * cols_per_block * 4 + 8192


# ---- sink 接线：remote writer ------------------------------------------------
def test_remote_sink_writer_integration(tmp_path):
    from factor_engine.runtime.streaming_result_sink import StreamingResultSink

    w, store = _make_writer(tmp_path, columns_per_block=4)
    sink = StreamingResultSink(
        writer=make_remote_sink_writer(w),
        batch_size=8,
        writer_threads=1,
        queue_bytes=1 << 24,
    )
    sink.start()
    rows = 400
    n = 12
    for i in range(n):
        arr = np.arange(rows, dtype=np.float32) + float(i)
        accepted = sink.submit(name=f"s{i:02d}", value=arr, partition="2026-08")
        assert accepted
    sink.finish()
    w.finish()
    s = w.summary()
    assert s["total_factors_written"] == n
    assert w.block_count() == math.ceil(n / 4)
    assert w.resolve_current() == w.generation_id()
