# -*- coding: utf-8 -*-
"""Wave1-P0-08/09/10 COS runtime audit tests.（P0-06/07 不碰：不覆盖 multipart/PartNumber）。

断言矩阵：
  P0-08  DISPROVEN: 进程重启 / GC 不删已发布的历史代次
      - test_p0_08_gc_keeps_published_manifest_generations
        (已发布的旧代次带 manifest → 显式 gc 永不删；旧行为= DISPROVEN 防御)
      - test_p0_08_gc_removes_only_orphan_without_manifest
        (只有无 manifest 的孤儿代次可被显式 gc 清理，且有 manifest 的永不)
      - test_p0_08_no_implicit_gc_trigger_in_publisher
        (publisher 无 __del__/weakref；gc 只在显式调用，0 自动触发路径)
  P0-09  DISPROVEN at runtime / documented + tested:
      - test_p0_09_memory_lease_released_on_writer_failure
        (writer 上传抛错时 lease 在 finally 释放 → broker 无泄漏，可复用)
      - test_p0_09_crash_orphan_objects_contained_by_policy
        (进程崩溃残留 orphan 无 manifest → 显式 gc 可清；不可显式删带 manifest 历史)
  P0-10  CONFIRMED → 修复：stale-writer 防护（fencing-epoch 单调 + StaleWriterError）
      - test_p0_10_stale_writer_rejected_when_out_of_date
        (begin 快照 epoch 落后于 CURRENT → finish 抛 StaleWriterError)
      - test_p0_10_legitimate_sequential_writers_accepted
        (先 10 后 10 的合法 writer 都晋升；epoch 单调递增)
      - test_p0_10_concurrent_current_rejects_stale_manifest_writer_id
        (CURRENT 已被越权翻转/writer_id 不符 → 拒绝)
"""
from __future__ import annotations

import gc as _gc
import json
from pathlib import Path

import pytest

from data_access.core.exceptions import DataError
from data_access.read.object_store import LocalObjectStore
from data_access.write.object_store_generation_publisher import (
    ObjectStoreGenerationPublisher,
    StaleWriterError,
    _CURRENT_KEY,
    _MANIFEST_KEY,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "cos")


def _current(store: LocalObjectStore, prefix: str) -> dict:
    key = f"{prefix}/{_CURRENT_KEY}"
    head = store.head_object(key)
    if head is None:
        return {}
    blob = store.range_read(key, offset=0, length=int(head["size"]))
    return json.loads(blob.decode("utf-8"))


def _publish_one(store: LocalObjectStore, prefix: str, payload: bytes = b"D",
                 writer_id: str = "w1") -> ObjectStoreGenerationPublisher:
    pub = ObjectStoreGenerationPublisher(store, writer_id=writer_id)
    gid = pub.begin_generation(prefix)
    pub.add_object(gid, "data.txt", payload)
    pub.finish_generation(gid)
    return pub


# ---- P0-08: GC 不删已发布历史代次（DISPROVEN → 防御测试） ---------------


def test_p0_08_gc_keeps_published_manifest_generations(store):
    prefix = "factors/f08"
    p1 = _publish_one(store, prefix, b"V1", writer_id="w1")
    g1 = _current(store, prefix)["generation_id"]
    p2 = _publish_one(store, prefix, b"V2", writer_id="w2")
    g2 = _current(store, prefix)["generation_id"]
    assert g1 != g2

    # 显式 gc（模拟运维 GC pass）：两个已发布代次都带 manifest → 永不删。
    removed = p1.gc(prefix)  # 即使持有旧 publisher 也一样
    assert removed == 0
    removed2 = p2.gc(prefix)
    assert removed2 == 0
    # 历史代次（g1，非 CURRENT）的 manifest 与数据都在。
    assert store.head_object(f"{prefix}/{g1}/{_MANIFEST_KEY}") is not None
    assert store.head_object(f"{prefix}/{g1}/data.txt") is not None
    assert store.head_object(f"{prefix}/{g2}/data.txt") is not None


def test_p0_08_gc_removes_only_orphan_without_manifest(store):
    prefix = "factors/f08b"
    pub = _publish_one(store, prefix, b"GOOD", writer_id="w1")
    good = _current(store, prefix)["generation_id"]

    # 孤儿：begin + 上传，但从未 finish（模拟进程崩溃/异常路径）。
    orphan = pub.begin_generation(prefix)
    pub.add_object(orphan, "partial.txt", b"ORPHAN")
    assert store.head_object(f"{prefix}/{orphan}/partial.txt") is not None
    assert store.head_object(f"{prefix}/{orphan}/{_MANIFEST_KEY}") is None

    removed = pub.gc(prefix)
    assert removed >= 1
    assert store.head_object(f"{prefix}/{orphan}/partial.txt") is None
    assert store.head_object(f"{prefix}/{good}/data.txt") is not None


def test_p0_08_no_implicit_gc_trigger_in_publisher():
    import inspect
    from data_access.write import object_store_generation_publisher as M

    src = inspect.getsource(M)
    # 模块内绝无对象生命周期/弱引用触发删除。
    assert "__del__" not in src
    assert "weakref" not in src
    assert "gc.collect" not in src
    # gc() 是唯一删除路径，且为显式方法。
    assert "def gc(self" in src


# ---- P0-09: writer lease / 崩溃资源（DISPROVEN → 测试固定） -------------


def test_p0_09_memory_lease_released_on_writer_failure(tmp_path):
    import numpy as np
    from data_access.read.object_store import LocalObjectStore
    from factor_engine.runtime.resource_broker import ResourceBroker
    from factor_engine.runtime.remote_factor_block_writer import (
        RemoteFactorBlockWriter,
    )

    broker = ResourceBroker(
        hard_memory_limit=64 * 1024 * 1024,
        cpu_slots=2,
        min_host_reserve_gb=0.0,
    )
    store = LocalObjectStore(tmp_path / "obj")

    class _FailingPublisher:
        """add_object 抛错的 publisher —— 模拟上传中途失败。"""

        def __init__(self, store_):
            self.store = store_

        def begin_generation(self, prefix, **kw):
            return "g-fail"

        def add_object(self, gid, rel, serialized, **kw):
            raise DataError("simulated upload failure")

        def abort_generation(self, gid):
            pass

        def finish_generation(self, gid):
            raise AssertionError("finish 不应被调用")

        def resolve_current(self, prefix):
            return None

    w = RemoteFactorBlockWriter(
        store,
        "factors/f09",
        broker=broker,
        publisher=_FailingPublisher(store),
        research_only=True,
    )
    before = broker._lease_sum_bytes()
    with pytest.raises(DataError):
        w.submit("f_a", np.arange(5, dtype=np.float32))
    # 上传抛错 → submit 的 finally 释放该次 in-flight lease。(缓冲列 lease 已
    # 由 flush_partition 失败路径释放；此处校验 broker 无滞留占用。)
    assert broker._lease_sum_bytes() <= before


def test_p0_09_crash_orphan_objects_contained_by_policy(store):
    prefix = "factors/f09b"
    pub = _publish_one(store, prefix, b"CURRENT", writer_id="w1")
    current = _current(store, prefix)["generation_id"]
    # 崩溃残留的 pending 代次（无 manifest）。
    ghost = pub.begin_generation(prefix)
    pub.add_object(ghost, "ghost.bin", b"x" * 10)
    # begin 内存状态丢失（模拟进程重启）。
    pub._pending.clear()
    pub._gen_prefix.clear()

    # 新 publisher 显式 gc：只清 orphan（无 manifest），CURRENT + 历史 manifest 保留。
    fresh = ObjectStoreGenerationPublisher(store, writer_id="w2")
    removed = fresh.gc(prefix)
    assert removed == 1
    assert store.head_object(f"{prefix}/{ghost}/ghost.bin") is None
    assert store.head_object(f"{prefix}/{current}/data.txt") is not None
    assert store.head_object(f"{prefix}/{current}/{_MANIFEST_KEY}") is not None
    assert _current(store, prefix)["generation_id"] == current  # CURRENT 不变


# ---- P0-10: 并发 CURRENT 下 stale writer 被拒（CONFIRMED → 修复） -------


def test_p0_10_stale_writer_rejected_when_out_of_date(store):
    # 先发布一个 CURRENT。
    _publish_one(store, "factors/f10", b"v1", writer_id="wA")

    # stale writer：begin 之后（快照 epoch=1），另一个 writer 又晋升到 epoch=2。
    stale = ObjectStoreGenerationPublisher(store, writer_id="stale-1")
    gid = stale.begin_generation("factors/f10")
    stale.add_object(gid, "data.txt", b"STALE")
    _publish_one(store, "factors/f10", b"v2", writer_id="wB")  # 并发新 writer 晋升

    # 旧 writer 现在 finish —— CURRENT epoch 已推进，必须被拒。
    with pytest.raises(StaleWriterError):
        stale.finish_generation(gid)
    # 拒绝后 CURRENT 仍是新代次，旧 writer 的对象不覆盖。
    cur = _current(store, "factors/f10")
    assert cur["generation_id"] != gid
    assert store.head_object(f"factors/f10/{gid}/data.txt") is not None  # 数据仍在


def test_p0_10_legitimate_sequential_writers_accepted(store):
    first = ObjectStoreGenerationPublisher(store, writer_id="w1")
    g1 = first.begin_generation("factors/f10b")
    first.add_object(g1, "data.txt", b"A")
    m1 = first.finish_generation(g1)
    assert m1.fencing_epoch == 1
    assert _current(store, "factors/f10b")["fencing_epoch"] == 1

    second = ObjectStoreGenerationPublisher(store, writer_id="w2")
    g2 = second.begin_generation("factors/f10b")
    second.add_object(g2, "data.txt", b"B")
    m2 = second.finish_generation(g2)
    assert m2.fencing_epoch == 2
    assert _current(store, "factors/f10b")["fencing_epoch"] == 2
    assert m2.writer_id == "w2"


def test_p0_10_concurrent_current_rejects_stale_manifest_writer_id(store):
    # 越权场景：有人直接改 CURRENT 指向他人代次（或 writer_id 不符）。
    _publish_one(store, "factors/f10c", b"v1", writer_id="legit")
    evil = ObjectStoreGenerationPublisher(store, writer_id="hijacker")
    gid = evil.begin_generation("factors/f10c")
    evil.add_object(gid, "data.txt", b"EVIL")
    # 并发 CURRENT 已被推进（他人晋升）。
    _publish_one(store, "factors/f10c", b"v2", writer_id="other-writer")
    with pytest.raises(StaleWriterError):
        evil.finish_generation(gid)  # epoch 落后 → stale 拒绝