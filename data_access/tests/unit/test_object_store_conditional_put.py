# -*- coding: utf-8 -*-
"""ObjectStore 条件写（put_object_conditional）单测（UPSTREAM_FIX_PLAN 问题二）。

覆盖 LocalObjectStore 的 If-Match / If-None-Match 语义与并发 CURRENT 抢占。
COSObjectStore 的条件写由 publisher 回退路径 + test_object_store_cos 覆盖。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_access.read.object_store import LocalObjectStore
from data_access.write.publish_errors import StaleWriterError
from data_access.write.object_store_generation_publisher import (
    ObjectStoreGenerationPublisher,
    _CURRENT_KEY,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "objs")


def test_conditional_put_if_match_success(store: LocalObjectStore):
    store.put_object("a/b.json", b"v1")
    etag = store.head_object("a/b.json")["etag"]
    # ETag 匹配 → 覆盖成功
    store.put_object_conditional("a/b.json", b"v2", if_match=etag)
    assert store.range_read("a/b.json", offset=0, length=2) == b"v2"


def test_conditional_put_if_match_stale_rejected(store: LocalObjectStore):
    store.put_object("a/b.json", b"v1-longer-content")
    old_etag = store.head_object("a/b.json")["etag"]
    # 别的 writer 已覆盖（内容长度不同 → mtime+size etag 必然不同；同长度
    # 同 mtime 会被 LocalObjectStore 的 mtime+size etag 碰撞）。
    store.put_object("a/b.json", b"v2-different-length")
    with pytest.raises(StaleWriterError):
        store.put_object_conditional("a/b.json", b"v3", if_match=old_etag)
    # 当前内容仍是 v2（旧 writer 未覆盖成功）
    assert store.range_read("a/b.json", offset=0, length=2) == b"v2"


def test_conditional_put_if_none_match_create_once(store: LocalObjectStore):
    store.put_object_conditional("cur.json", b"first", if_none_match=True)
    assert store.head_object("cur.json") is not None
    # 对象已存在 → If-None-Match 抢占必须失败
    with pytest.raises(StaleWriterError):
        store.put_object_conditional("cur.json", b"second", if_none_match=True)
    assert store.range_read("cur.json", offset=0, length=5) == b"first"


def test_conditional_put_missing_object_if_match_fails(store: LocalObjectStore):
    with pytest.raises(StaleWriterError):
        store.put_object_conditional("nope.json", b"x", if_match="whatever")


def test_current_pointer_conditional_flip_concurrent(store: LocalObjectStore):
    """两个 publisher 并发晋升：只有一个能翻转 CURRENT（条件写 fencing）。"""
    p1 = ObjectStoreGenerationPublisher(store, writer_id="w1", multipart_threshold=4)
    p2 = ObjectStoreGenerationPublisher(store, writer_id="w2", multipart_threshold=4)

    g1 = p1.begin_generation("factors/f")
    p1.add_object(g1, "data.parquet", b"gen1")
    g2 = p2.begin_generation("factors/f")
    p2.add_object(g2, "data.parquet", b"gen2")

    p1.finish_generation(g1)
    # w2 的 begin 快照 epoch=0 已落后（CURRENT 存在但 w2 begin 在翻转前）——
    # 应用层 epoch 检查或对象层 If-Match 至少一道拒绝；CURRENT 必须仍是 gen1。
    with pytest.raises((StaleWriterError, Exception)):
        p2.finish_generation(g2)
    cur = store.range_read(
        f"factors/f/{_CURRENT_KEY}", offset=0, length=10**6
    )
    assert json.loads(cur)["generation_id"] == g1
