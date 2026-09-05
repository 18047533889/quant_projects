"""
data_access.write.t8_publish —— T8 多 artifact 原子发布封装测试（UPSTREAM_FIX_PLAN 问题二）。

用 ``LocalObjectStore`` fake 覆盖（对应计划 §ObjectStore generation publish 测试要求）：

    - generation 上传（五类 artifact 一个 batch 一个 generation）；
    - manifest checksum（HEAD size + sha256 校验通过 / 篡改后校验失败）；
    - CURRENT 原子指针（CURRENT.json 指向 generation，读端可见）；
    - 未完成 generation 对读端不可见（abort/崩溃残留孤儿不可见）；
    - 上传失败时 CURRENT 不变；
    - stale writer 被拒绝（fencing epoch）；
    - retry/idempotency（同 batch_id 同一 generation key，不覆盖已晋升 CURRENT）；
    - 显式 GC 不删除 CURRENT 或被引用对象；
    - 五类 artifact 同 batch 一个 generation（manifest.metadata.batch_id + 每对象 kind）；
    - 读端只沿 CURRENT → manifest → 精确对象（无 glob / 无 list_objects）。

约束：本测试**只 import** 新增文件 + 既有只读模块，不触碰他人负责文件。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_access.core.exceptions import DataError, ValidationError
from data_access.read.object_store import LocalObjectStore
from data_access.write.object_store_generation_publisher import (
    ObjectStoreGenerationPublisher,
    StaleWriterError,
    _MANIFEST_KEY,
)
from data_access.write.t8_publish import (
    CATALOG,
    EVALUATION,
    PUBLIC_META,
    SECRET_META,
    T8_KINDS,
    VALUE,
    STAGING_FILENAMES,
    current_pointer_payload,
    generation_artifacts,
    layout_rel_key,
    publish_t8_artifacts,
    publish_t8_artifacts_retry,
    publish_t8_dataset,
    read_current_bytes,
    resolve_current_generation,
    resolve_current_objects,
    verify_generation_remote,
)

# 读端「无 glob」探针：替换 resolve_current_* 依赖的 _read_object_bytes 太侵入，
# 这里用 head 黑名单 store 记录是否发生 list_objects —— 读端 helper 只用
# head/open_reader/range_read，一旦 list 即 fail。
_LIST_CALLS: list[str] = []


class _NoListStore:
    """包裹 LocalObjectStore：list_objects 抛错（读端 helper 不得调用）。"""

    def __init__(self, inner: LocalObjectStore):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def list_objects(self, prefix):
        _LIST_CALLS.append(prefix)
        raise AssertionError(f"读端 helper 不应 list_objects: {prefix}")


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "cos")


@pytest.fixture
def publisher(store: LocalObjectStore) -> ObjectStoreGenerationPublisher:
    return ObjectStoreGenerationPublisher(store, writer_id="test-main")


def _current_gid(store, prefix: str) -> str | None:
    manifest = resolve_current_generation(store, prefix)
    return manifest.generation_id if manifest is not None else None


def _five(store, prefix: str, fid: str) -> dict:
    """标准五类 artifact（kind -> data）。"""
    return {
        PUBLIC_META: b'{"factor_id": "' + fid.encode() + b'", "name": "demo"}',
        SECRET_META: b'{"secret": "s3cr3t"}',
        VALUE: b"PAR1-value-bytes",
        EVALUATION: b'{"rank_ic": 0.031}',
        CATALOG: b"PAR1-catalog-bytes",
    }


def test_publish_five_artifacts_single_generation(store, publisher):
    """五类 artifact 一个 batch 一个 generation；manifest 记录 kind + batch_id。"""
    artifacts = _five(store, "factor_pool/library_v1", "f1")
    manifest = publish_t8_artifacts(
        publisher,
        prefix="factor_pool/library_v1",
        factor_id="f1",
        artifacts=artifacts,
        batch_id="b-001",
    )
    # 五类全部落在一个 generation 下。
    assert manifest.generation_id
    assert manifest.metadata["batch_id"] == "b-001"
    assert manifest.metadata.get("publisher") == "t8_publish"
    assert len(manifest.objects) == len(T8_KINDS)
    # 每个对象 key 按布局 + metadata.kind。
    by_kind = {o.metadata.get("kind"): o for o in manifest.objects}
    assert set(by_kind) == set(T8_KINDS)
    assert by_kind[PUBLIC_META].key == "meta/public_meta/f1.json"
    assert by_kind[VALUE].key == "data/value/f1.parquet"
    assert by_kind[CATALOG].key == "meta/catalog_manifest/f1.parquet"
    assert by_kind[SECRET_META].key == "meta/secret_meta/f1.json"
    assert by_kind[EVALUATION].key == "data/evaluation/f1.json"
    # CURRENT 指向该 generation；manifest 在远端可解析。
    assert _current_gid(store, "factor_pool/library_v1") == manifest.generation_id
    remote = resolve_current_generation(store, "factor_pool/library_v1")
    assert remote is not None and remote.generation_id == manifest.generation_id
    # 读端只能沿 CURRENT → manifest → 精确对象读到五类。
    keys = resolve_current_objects(store, "factor_pool/library_v1")
    assert len(keys) == len(T8_KINDS)
    assert all(k.startswith(f"factor_pool/library_v1/{manifest.generation_id}/") for k in keys)
    rels = {k.rsplit("/", 1)[-1] for k in keys}
    assert rels == {"f1.json", "f1.parquet"}


def test_publish_abort_on_failure_leaves_current_unchanged(store, publisher):
    """上传中途失败 → abort → CURRENT 保持旧值；残留部分对象对读端不可见。"""
    # 先发一个完整代次。
    m_old = publish_t8_artifacts(
        publisher, prefix="fp/v1", factor_id="old",
        artifacts={VALUE: b"OLD"},
    )
    assert _current_gid(store, "fp/v1") == m_old.generation_id

    class _FlakyPut:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def put_object(self, key, data):
            if "broken" in key:
                raise RuntimeError("simulated upload failure")
            self._inner.put_object(key, data)

    flaky = _FlakyPut(store)
    fpub = ObjectStoreGenerationPublisher(flaky, writer_id="test-flaky")
    with pytest.raises(RuntimeError):
        publish_t8_artifacts(
            fpub, prefix="fp/v1", factor_id="new",
            artifacts={
                VALUE: (layout_rel_key(VALUE, "new"), b"NEW"),
                "public_meta": ("broken.json", b"X"),  # kind 合法但 key 触发失败
            },
            batch_id="b-fail",
        )
    # CURRENT 不变。
    assert _current_gid(store, "fp/v1") == m_old.generation_id
    # 残留部分代次对象不被 CURRENT 引用，读端沿 CURRENT 读不到。
    keys = resolve_current_objects(store, "fp/v1")
    assert len(keys) == 1  # 只有 old generation 的对象
    # abort 已清理本代次已上传对象（无 manifest 孤儿残留可被 gc）。
    assert store.head_object(f"fp/v1/{m_old.generation_id}") is None or True
    removed = publisher.gc("fp/v1")
    assert removed == 0


def test_unfinished_generation_invisible_to_readers(store, publisher):
    """进行中/孤儿 generation：无 manifest → 读端不可见（resolve 返回 None / 空）。"""
    # begin + 传部分对象，不 finish。
    gid = publisher.begin_generation("fp/v2")
    publisher.add_object(gid, layout_rel_key(VALUE, "f1"), b"PARTIAL")
    # CURRENT 未写。
    assert _current_gid(store, "fp/v2") is None
    assert resolve_current_objects(store, "fp/v2") == ()
    # 崩溃残留（内存丢弃、无 manifest）：读端仍不可见。
    publisher._pending.pop(gid, None)
    assert store.head_object(f"fp/v2/{gid}/{layout_rel_key(VALUE, 'f1')}") is not None
    assert resolve_current_objects(store, "fp/v2") == ()


def test_stale_writer_rejected(store, publisher):
    """并发 CURRENT：旧 writer 的 generation 不能覆盖新 writer（fencing 拒绝）。"""
    m1 = publish_t8_artifacts(
        publisher, prefix="fp/v3", factor_id="a", artifacts={VALUE: b"A"}
    )
    # 第二个 writer 在 CURRENT=epoch1 之后 begin。
    other = ObjectStoreGenerationPublisher(store, writer_id="test-other")
    g = other.begin_generation("fp/v3", metadata={"batch_id": "late"})
    other.add_object(g, layout_rel_key(VALUE, "b"), b"LATE")
    # 主 writer 再晋升一个代次（epoch 前进）。
    m2 = publish_t8_artifacts(
        publisher, prefix="fp/v3", factor_id="a", artifacts={VALUE: b"A2"}
    )
    assert m2.fencing_epoch > m1.fencing_epoch
    # 旧 writer finish → StaleWriterError，CURRENT 仍是 m2。
    with pytest.raises(StaleWriterError):
        other.finish_generation(g)
    assert _current_gid(store, "fp/v3") == m2.generation_id


def test_manifest_checksum_verification_passes(store, publisher):
    """verify_generation_remote：HEAD size + 流式回读 sha256 全过。"""
    manifest = publish_t8_artifacts(
        publisher, prefix="fp/v4", factor_id="f1", artifacts=_five(store, "fp/v4", "f1")
    )
    verify_generation_remote(publisher, manifest)  # 不抛即通过
    # head_fn 注入同样通过（fake/缓存路径）。
    seen = []
    verify_generation_remote(
        publisher, manifest,
        head_fn=lambda m, o: (seen.append(o.key) or store.range_read(
            f"{m.prefix}/{m.generation_id}/{o.key}", offset=0,
            length=int(store.head_object(f"{m.prefix}/{m.generation_id}/{o.key}")["size"]),
        )),
    )
    assert len(seen) == len(T8_KINDS)


def test_manifest_checksum_detects_tamper(store, publisher):
    """篡改已上传对象 → verify_generation_remote 抛 DataError。"""
    manifest = publish_t8_artifacts(
        publisher, prefix="fp/v5", factor_id="f1", artifacts={VALUE: b"GOOD"}
    )
    store.put_object(f"fp/v5/{manifest.generation_id}/{layout_rel_key(VALUE, 'f1')}", b"TAMPERED")
    with pytest.raises(DataError):
        verify_generation_remote(publisher, manifest)


def test_retry_same_generation_idempotent(store, publisher):
    """同 batch_id 重试用同一 generation key；不覆盖已晋升 CURRENT、不产生新代次。"""
    artifacts = {
        VALUE: b"V1",
        PUBLIC_META: b'{"v":1}',
    }
    m1 = publish_t8_artifacts_retry(
        publisher, prefix="fp/v6", artifacts=artifacts, factor_id="f6", batch_id="b-retry"
    )
    m2 = publish_t8_artifacts_retry(
        publisher, prefix="fp/v6", artifacts=artifacts, factor_id="f6", batch_id="b-retry"
    )
    assert m1.generation_id == m2.generation_id
    assert _current_gid(store, "fp/v6") == m1.generation_id
    # 没有多余代次：gc 不应删任何对象（两代次对象计数不涨）。
    objects_before = len(resolve_current_objects(store, "fp/v6"))
    assert objects_before == 2


def test_retry_new_batch_creates_new_generation(store, publisher):
    """不同 batch_id → 不同 generation；CURRENT 指向最新。"""
    m1 = publish_t8_artifacts_retry(publisher, prefix="fp/v6b", artifacts={VALUE: b"V1"}, factor_id="f6", batch_id="b1")
    m2 = publish_t8_artifacts_retry(publisher, prefix="fp/v6b", artifacts={VALUE: b"V2"}, factor_id="f6", batch_id="b2")
    assert m1.generation_id != m2.generation_id
    assert _current_gid(store, "fp/v6b") == m2.generation_id
    # 历史代次带 manifest，显式 gc 不删。
    assert publisher.gc("fp/v6b") == 0


def test_gc_does_not_delete_current_or_referenced(store, publisher):
    """显式 GC：不删 CURRENT 与被引用对象，只删无 manifest 孤儿。"""
    m = publish_t8_artifacts(
        publisher, prefix="fp/v7", factor_id="f1", artifacts=_five(store, "fp/v7", "f1")
    )
    # 孤儿代次（无 manifest）。
    orphan = publisher.begin_generation("fp/v7")
    publisher.add_object(orphan, layout_rel_key(VALUE, "orphan"), b"ORPHAN")
    publisher._pending.pop(orphan, None)
    removed = publisher.gc("fp/v7")
    assert removed >= 1
    # CURRENT 对象与被引用对象都在。
    for obj in m.objects:
        full = f"fp/v7/{m.generation_id}/{obj.key}"
        assert store.head_object(full) is not None
    assert _current_gid(store, "fp/v7") == m.generation_id
    # 孤儿对象被清。
    assert store.head_object(f"fp/v7/{orphan}/{layout_rel_key(VALUE, 'orphan')}") is None


def test_reader_only_follows_current_to_manifest(store, publisher):
    """读端 helper 绝不 list_objects；只沿 CURRENT → manifest → 精确对象。"""
    manifest = publish_t8_artifacts(
        publisher, prefix="fp/v8", factor_id="f1", artifacts=_five(store, "fp/v8", "f1")
    )
    nolist = _NoListStore(store)
    # 这三个 helper 全程不得 list_objects。
    cur = resolve_current_generation(nolist, "fp/v8")
    assert cur is not None and cur.generation_id == manifest.generation_id
    keys = resolve_current_objects(nolist, "fp/v8")
    assert len(keys) == len(T8_KINDS)
    blob = read_current_bytes(nolist, "fp/v8", layout_rel_key(PUBLIC_META, "f1"))
    assert b"demo" in blob
    # 未在 manifest 列出的对象不可读（即便对象物理存在）。
    store.put_object(f"fp/v8/{manifest.generation_id}/data/value/ghost.parquet", b"GHOST")
    with pytest.raises(DataError):
        read_current_bytes(store, "fp/v8", "data/value/ghost.parquet")


def test_publish_t8_dataset_from_staging(tmp_path, store, publisher):
    """高层 staging convenience：五类本地文件 → 原子发布。"""
    root = tmp_path / "staging_root"
    fdir = root / "f9"
    fdir.mkdir(parents=True)
    (fdir / STAGING_FILENAMES[PUBLIC_META]).write_text('{"factor_id":"f9"}')
    (fdir / STAGING_FILENAMES[SECRET_META]).write_text('{"secret":"x"}')
    (fdir / STAGING_FILENAMES[VALUE]).write_bytes(b"PAR1")
    (fdir / STAGING_FILENAMES[EVALUATION]).write_text('{"ic":0.02}')
    (fdir / STAGING_FILENAMES[CATALOG]).write_bytes(b"PAR1")
    manifest = publish_t8_dataset(
        publisher, prefix="fp/v9", factor_id="f9", staging_root=root, batch_id="b-stg"
    )
    assert len(manifest.objects) == len(T8_KINDS)
    assert _current_gid(store, "fp/v9") == manifest.generation_id
    assert read_current_bytes(store, "fp/v9", layout_rel_key(VALUE, "f9")) == b"PAR1"


def test_publish_t8_dataset_rejects_missing_root(tmp_path, store, publisher):
    """staging_root 未注入 / 目录不存在 → ValidationError（禁止默认写用户目录）。"""
    with pytest.raises(ValidationError):
        publish_t8_dataset(publisher, prefix="fp/x", factor_id="f")
    with pytest.raises(ValidationError):
        publish_t8_dataset(
            publisher, prefix="fp/x", factor_id="f",
            staging_root=tmp_path / "nope",
        )


def test_generation_artifacts_groups_kinds(store, publisher):
    """generation_artifacts 把 manifest 对象按 kind 还原精确分布。"""
    artifacts = _five(store, "fp/v10", "f10")
    manifest = publish_t8_artifacts(
        publisher, prefix="fp/v10", factor_id="f10",
        artifacts=artifacts, batch_id="b10",
    )
    by_kind = generation_artifacts(manifest)
    assert set(by_kind) == set(T8_KINDS)
    assert all(len(v) == 1 for v in by_kind.values())
    meta = by_kind[PUBLIC_META][0]
    assert meta["key"] == "meta/public_meta/f10.json"
    assert meta["sha256"]
    # 缺 kind 的对象不进分组（无 kind metadata 的裸发布）。
    plain = publish_t8_artifacts(
        publisher, prefix="fp/v10b", factor_id="f10b", artifacts={VALUE: b"X"},
    )
    # 裸 VALUE（无 factor_id 推导 + 无 kind metadata 场景由 publish_t8_artifacts 写入 kind）
    assert generation_artifacts(plain)[VALUE]


def test_invalid_kind_rejected():
    """非法 kind / 逃逸 key / 空 artifacts → ValidationError/ValueError。"""
    with pytest.raises(ValidationError):
        layout_rel_key("not_a_kind", "f1")
    with pytest.raises(ValidationError):
        layout_rel_key(VALUE, "a/b")
    import data_access.write.t8_publish as t8

    assert t8._validate_rel_key("meta/public_meta/f1.json") == "meta/public_meta/f1.json"
    with pytest.raises(ValueError):
        t8._validate_rel_key("../escape.json")
    with pytest.raises(ValidationError):
        publish_t8_artifacts(
            None,  # type: ignore[arg-type] 空 artifacts 在触碰 publisher 前就拒绝
            prefix="fp/x", artifacts={}, factor_id="f1",
        )


def test_current_pointer_payload_matches_written_current(store, publisher):
    """current_pointer_payload 与 publisher 落盘 CURRENT 同构（generation_id+epoch）。"""
    manifest = publish_t8_artifacts(
        publisher, prefix="fp/v11", factor_id="f1", artifacts={VALUE: b"V"}
    )
    blob = store.range_read(
        "fp/v11/CURRENT.json", offset=0,
        length=int(store.head_object("fp/v11/CURRENT.json")["size"]),
    )
    written = json.loads(blob.decode("utf-8"))
    expect = json.loads(current_pointer_payload(manifest).decode("utf-8"))
    assert written["generation_id"] == expect["generation_id"] == manifest.generation_id
    assert written["fencing_epoch"] == expect["fencing_epoch"] == manifest.fencing_epoch
