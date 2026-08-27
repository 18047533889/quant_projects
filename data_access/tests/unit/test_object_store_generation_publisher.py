# -*- coding: utf-8
"""ObjectStoreGenerationPublisher 单元测试。

覆盖：
    (a) 发布 → CURRENT 只在所有对象存在+校验后才翻转；中途失败 → CURRENT 不变，
        部分代次不可见（resolve_current 不指向它）。
    (b) checksum 不匹配 → 代次不晋升。
    (c) abort 清理部分代次；后续显式 gc 删除未引用对象。
    (d) 授权边界：因子写解析到正确 bucket+prefix 且按资源 scope 允许/拒绝；跨 prefix 拒绝。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_access.core.exceptions import AccessDeniedError, DataError, ValidationError
from data_access.read.object_store import LocalObjectStore
from data_access.write.object_store_generation_publisher import (
    ObjectStoreGenerationPublisher,
    _CURRENT_KEY,
    _MANIFEST_KEY,
)
from data_access.write.object_storage_boundary import (
    ObjectResource,
    ObjectResourceScope,
    ObjectStorageAuthorizationBoundary,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "cos")


@pytest.fixture
def publisher(store: LocalObjectStore) -> ObjectStoreGenerationPublisher:
    return ObjectStoreGenerationPublisher(store, multipart_threshold=4)


def _current_gid(store: LocalObjectStore, prefix: str) -> str | None:
    key = f"{prefix}/{_CURRENT_KEY}"
    head = store.head_object(key)
    if head is None:
        return None
    blob = store.range_read(key, offset=0, length=int(head["size"]))
    return json.loads(blob.decode("utf-8"))["generation_id"]


# ---- (a) CURRENT 只在完整校验后翻转；中途失败 CURRENT 不变 ------------------


def test_publish_flips_current_only_after_all_objects_verified(store, publisher):
    gid = publisher.begin_generation("factors/f1", metadata={"factor_id": "f1"})
    publisher.add_object(gid, "data.parquet", b"AAA")
    publisher.add_object(gid, "meta.json", b'{"x":1}')

    # 晋升前 CURRENT 不存在。
    assert _current_gid(store, "factors/f1") is None
    assert publisher.resolve_current("factors/f1") is None

    manifest = publisher.finish_generation(gid)

    assert manifest.generation_id == gid
    assert _current_gid(store, "factors/f1") == gid
    assert publisher.resolve_current("factors/f1") == gid
    # 对象确实落盘。
    assert store.head_object(f"factors/f1/{gid}/data.parquet") is not None
    assert store.head_object(f"factors/f1/{gid}/meta.json") is not None
    # manifest 存在且含内容哈希。
    mhead = store.head_object(f"factors/f1/{gid}/{_MANIFEST_KEY}")
    assert mhead is not None
    assert manifest.content_hash


def test_mid_upload_failure_leaves_current_unchanged(store, publisher):
    # 先发布一个完整代次作为 CURRENT。
    g1 = publisher.begin_generation("factors/f1")
    publisher.add_object(g1, "data.parquet", b"OLD")
    publisher.finish_generation(g1)
    assert publisher.resolve_current("factors/f1") == g1

    # 新代次中途失败（add_object 抛错）。
    g2 = publisher.begin_generation("factors/f1")
    publisher.add_object(g2, "data.parquet", b"NEW")
    with pytest.raises(DataError):
        # 模拟上传后 head 缺失 → 校验失败。
        def _fail_upload(full_key, blob):
            raise DataError(f"simulated upload failure for {full_key}")

        publisher._upload = _fail_upload  # type: ignore[assignment]
        publisher.add_object(g2, "broken.parquet", b"X")

    # CURRENT 仍指向旧完整代次；部分代次不可见。
    assert publisher.resolve_current("factors/f1") == g1
    assert _current_gid(store, "factors/f1") == g1
    # 部分代次对象存在但未被 CURRENT 引用。
    assert store.head_object(f"factors/f1/{g2}/data.parquet") is not None


# ---- (b) checksum 不匹配 → 不晋升 -------------------------------------------


def test_checksum_mismatch_not_promoted(store, publisher):
    gid = publisher.begin_generation("factors/f2")
    publisher.add_object(gid, "data.parquet", b"GOOD")

    # 篡改已上传对象内容 → 校验时 sha256 不匹配。
    store.put_object(f"factors/f2/{gid}/data.parquet", b"TAMPERED")

    with pytest.raises(DataError):
        publisher.finish_generation(gid)

    # CURRENT 未翻转。
    assert publisher.resolve_current("factors/f2") is None
    assert _current_gid(store, "factors/f2") is None


# ---- (c) abort 清理部分代次；显式 gc 删除未引用对象 -------------------------


def test_abort_cleans_partial_generation(store, publisher):
    gid = publisher.begin_generation("factors/f3")
    publisher.add_object(gid, "data.parquet", b"PARTIAL")
    assert store.head_object(f"factors/f3/{gid}/data.parquet") is not None

    publisher.abort_generation(gid)

    # abort 删除该代次对象。
    assert store.head_object(f"factors/f3/{gid}/data.parquet") is None
    assert publisher.resolve_current("factors/f3") is None


def test_gc_removes_unreferenced_generation(store, publisher):
    # 发布一个完整代次作为 CURRENT。
    g1 = publisher.begin_generation("factors/f4")
    publisher.add_object(g1, "data.parquet", b"KEEP")
    publisher.finish_generation(g1)

    # 一个被 abort 的孤儿代次（对象残留，无 manifest）。
    orphan = publisher.begin_generation("factors/f4")
    publisher.add_object(orphan, "data.parquet", b"ORPHAN")
    # 模拟进程崩溃：直接丢弃内存状态，对象残留。
    publisher._pending.pop(orphan, None)
    assert store.head_object(f"factors/f4/{orphan}/data.parquet") is not None

    removed = publisher.gc("factors/f4")

    assert removed >= 1
    # 孤儿对象被删，CURRENT 代次保留。
    assert store.head_object(f"factors/f4/{orphan}/data.parquet") is None
    assert store.head_object(f"factors/f4/{g1}/data.parquet") is not None
    assert publisher.resolve_current("factors/f4") == g1


def test_gc_keeps_manifest_referenced_generation(store, publisher):
    # 发布两个完整代次（都有 manifest），CURRENT 指向第二个。
    g1 = publisher.begin_generation("factors/f5")
    publisher.add_object(g1, "data.parquet", b"V1")
    publisher.finish_generation(g1)
    g2 = publisher.begin_generation("factors/f5")
    publisher.add_object(g2, "data.parquet", b"V2")
    publisher.finish_generation(g2)
    assert publisher.resolve_current("factors/f5") == g2

    # gc 不删有 manifest 的历史代次（回滚/审计用）。
    removed = publisher.gc("factors/f5")
    assert removed == 0
    assert store.head_object(f"factors/f5/{g1}/data.parquet") is not None
    assert store.head_object(f"factors/f5/{g2}/data.parquet") is not None


# ---- (d) 授权边界 -----------------------------------------------------------


def test_authorize_factor_write_resolves_bucket_and_prefix():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*"),
        ],
        factor_target_resolver=lambda fid: ("factor-lake", f"factors/{fid}"),
    )
    resource = boundary.authorize_factor_write("f1")
    assert resource.bucket == "factor-lake"
    assert resource.key_prefix == "factors/f1"
    assert resource.dataset == "f1"


def test_authorize_factor_write_denied_cross_prefix():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[
            ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*"),
        ],
        factor_target_resolver=lambda fid: ("factor-lake", f"other/{fid}"),
    )
    with pytest.raises(AccessDeniedError):
        boundary.authorize_factor_write("f1")


def test_authorize_object_denied_wrong_bucket():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*")]
    )
    with pytest.raises(AccessDeniedError):
        boundary.authorize_object(
            ObjectResource(bucket="other-bucket", key_prefix="factors/f1"),
            "object:write",
        )


def test_authorize_object_allowed_within_scope():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*")]
    )
    # 不抛错即通过。
    boundary.authorize_object(
        ObjectResource(bucket="factor-lake", key_prefix="factors/f1/data.parquet"),
        "object:write",
    )
    boundary.authorize_object(
        ObjectResource(bucket="factor-lake", key_prefix="factors/f1/data.parquet"),
        "object:read",
    )


def test_unknown_action_rejected():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[ObjectResourceScope(bucket="factor-lake")]
    )
    with pytest.raises(AccessDeniedError):
        boundary.authorize_object(
            ObjectResource(bucket="factor-lake", key_prefix="x"), "factor:write"
        )


def test_require_any_object_write_action():
    boundary = ObjectStorageAuthorizationBoundary(
        allowed_scopes=[ObjectResourceScope(bucket="factor-lake", key_prefix="factors/*")]
    )
    boundary.require_any_object_write_action(
        ObjectResource(bucket="factor-lake", key_prefix="factors/f1")
    )
    with pytest.raises(AccessDeniedError):
        boundary.require_any_object_write_action(
            ObjectResource(bucket="other", key_prefix="factors/f1")
        )
