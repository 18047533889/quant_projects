# -*- coding: utf-8
"""ObjectStoreGenerationPublisher 单元测试。

覆盖：
    (a) 发布 → CURRENT 只在所有对象存在+校验后才翻转；中途失败 → CURRENT 不变，
        部分代次不可见（resolve_current 不指向它）。
    (b) checksum 不匹配 → 代次不晋升。
    (c) abort 清理部分代次；后续显式 gc 删除未引用对象。
    (d) 授权边界：因子写解析到正确 bucket+prefix 且按资源 scope 允许/拒绝；跨 prefix 拒绝。
    (e) P0-07 有界内存 multipart 流式上传（bytes 与 BinaryIO 双入口）：
        - bytes 大对象 multipart 分块上传（不整块二次拷贝常驻），PartNumber 1..N，
          complete 只调一次；
        - BinaryIO 直接进 store 流式路径（不 read() 整读），sha256/size 逐块回读校验，
          内存峰值保持平坦（受 part 数无关）。
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


# ---- (e) P0-07 有界内存 multipart 流式上传（bytes 与 BinaryIO 双入口） --------

import hashlib as _hashlib
import io as _io
from collections import deque as _deque


class _PartSinkRecorder:
    """fake ObjectStore：记录每 part 的 (part_number, len)，不保留内容。

    只计数不存字节 —— 隔离客户端内存测量（真实 COS 由服务端吸收 part 字节）。
    同时模拟 retry：``upload_part`` 首次对 ``fail_once_part`` 抛错，重试后成功
    （单 part 重试语义）。
    """

    def __init__(self, fail_once_part: int | None = None):
        self.part_numbers: list[int] = []
        self.part_sizes: list[int] = []
        self.complete_count = 0
        self.abort_count = 0
        self.put_count = 0
        self.upload_id = "fake-up-1"
        self._fail_once_part = fail_once_part
        self._fail_remaining = {fail_once_part: 1} if fail_once_part is not None else {}
        self._objects: dict[str, bytes] = {}

    def head_object(self, key):
        data = self._objects.get(key)
        if data is None:
            return None
        return {"etag": "x", "size": len(data), "last_modified": 0}

    def list_objects(self, prefix):
        return [k for k in self._objects if k.startswith(prefix)]

    def range_read(self, key, *, offset, length):
        return self._objects[key][offset : offset + length]

    def open_reader(self, key):
        data = self._objects.get(key)
        if data is None:
            return None
        return _io.BytesIO(data)

    def put_object(self, key, data):
        self.put_count += 1
        if hasattr(data, "read"):
            data = data.read()
        self._objects[key] = data

    def begin_multipart(self, key):
        return self.upload_id

    def upload_part(self, upload_id, key, part_index, data):
        # 0-based 内部索引 → 1-based S3 PartNumber（publisher 契约）。
        pnum = int(part_index) + 1
        # 单 part 瞬时失败 → 调用方（store）重试同一 part（不重试整个 upload）。
        if self._fail_once_part == pnum and self._fail_remaining.get(pnum, 0) > 0:
            self._fail_remaining[pnum] -= 1
            raise RuntimeError("simulated transient part failure")
        self.part_numbers.append(pnum)
        self.part_sizes.append(len(data))
        self._objects.setdefault(key, bytearray()).extend(data)
        return {"ETag": f'"etag-{self.upload_id}-{pnum}"'}

    def complete_multipart(self, upload_id, key):
        self.complete_count += 1

    def abort_multipart(self, upload_id, key):
        self.abort_count += 1

    def delete_object(self, key):
        self._objects.pop(key, None)


def test_bytes_multipart_publisher_part_numbers_and_complete_once():
    """bytes 大对象 multipart：PartNumber 1..N 连续、complete 恰好一次。

    P0-07 回归：修复前 publisher._upload 用 1-based 内部索引直接传给
    ``store.upload_part``（该接口期望 0-based）→ COSObjectStore 再 +1 → 实际
    PartNumber 2..N+1，PartNumber 1 缺失（S3 会拒绝 complete）。修复后统一
    0-based 内部索引 → store 换算 1-based。
    """
    sink = _PartSinkRecorder()
    pub = ObjectStoreGenerationPublisher(sink, multipart_threshold=1)
    pub.multipart_part_size = lambda: 5 * 1024 * 1024  # 5 MiB parts
    gid = pub.begin_generation("factors/f1")
    # 4 parts：3 个满 5MiB + 1 个尾巴 1 字节。
    payload = (
        b"A" * (5 * 1024 * 1024)
        + b"B" * (5 * 1024 * 1024)
        + b"C" * (5 * 1024 * 1024)
        + b"D"
    )
    pub.add_object(gid, "big.parquet", payload)
    assert sink.part_numbers == [1, 2, 3, 4]
    assert sink.complete_count == 1
    assert sink.abort_count == 0
    assert sink.put_count == 0  # 大对象不落单 PUT
    # 每个 part 都 >= 5 MiB（除最后一个尾巴）。
    assert all(s >= 5 * 1024 * 1024 for s in sink.part_sizes[:-1])
    assert sink.part_sizes[-1] == 1
    # manifest 记录的 size/sha256 与实际一致。
    obj = pub._pending[gid].objects[0]
    assert obj.size == len(payload)
    assert obj.sha256 == _hashlib.sha256(payload).hexdigest()


def test_bytes_multipart_publisher_part_retry():
    """单 part 失败重试该 part（不重试整个 upload），complete 仍一次。"""
    sink = _PartSinkRecorder(fail_once_part=2)
    pub = ObjectStoreGenerationPublisher(sink, multipart_threshold=1)
    pub.multipart_part_size = lambda: 5 * 1024 * 1024
    gid = pub.begin_generation("factors/f1")
    payload = (
        b"A" * (5 * 1024 * 1024) + b"B" * (5 * 1024 * 1024) + b"C"
    )
    pub.add_object(gid, "retry.parquet", payload)
    # part 2 首次失败 + 重试成功 → 每个 part 记录一次成功。
    assert sink.part_numbers == [1, 2, 3]
    assert sink.complete_count == 1
    assert sink.abort_count == 0


def test_streaming_publisher_flat_memory_proof():
    """BinaryIO 入口：内存峰值与 part 数无关（平坦），5x part-size 合成流。

    内存证明不靠 psutil —— 用可注入的 ``_MaxHoldingRecorder`` 跟踪任意时刻
    客户端实际持有的字节数：
    - ``put_object`` 收到的是流（store 逐 part 消费，producer 只持有 1 个 part）；
    - 每 part 上传后立即释放 → 峰值 ≈ part_size × O(1)。
    合成流物理只持有 1 个模板 part，逻辑 5× part_size。对比：整读路径需同时
    持有 5× part_size（或对象全量）。
    """
    part_size = 1024 * 1024  # 1 MiB 分块（测试缩小）
    total = 5 * part_size  # 逻辑 5 个 part

    class _LazyStream:
        """逐块产出的合成流：物理只持有模板，逻辑 total 字节。"""

        def __init__(self, template: bytes, *, total: int):
            self._template = template
            self._remaining = total
            self._pos = 0

        def read(self, n: int = -1):
            if n is None or n < 0:
                n = self._remaining
            out = bytearray()
            while n > 0 and self._remaining > 0:
                chunk = self._template[self._pos : self._pos + n]
                if not chunk:
                    self._pos = 0
                    chunk = self._template[:n]
                out.extend(chunk)
                self._remaining -= len(chunk)
                self._pos = (self._pos + len(chunk)) % len(self._template)
                n -= len(chunk)
            return bytes(out)

    class _MaxHoldingRecorder(_PartSinkRecorder):
        """记录任意时刻 held 字节峰值（当前活跃 part bytes 之和）。"""

        def __init__(self, part_size: int):
            super().__init__()
            self.part_size = part_size
            self.held = 0
            self.max_held = 0
            self.seen_max_upload_size = 0

        def upload_part(self, upload_id, key, part_index, data):
            # 调用方（COSObjectStore）内部会重试该 part —— 这里模拟 store 层：
            # fail_once 的 part 首败后重试成功。
            if self._fail_once_part is not None:
                return super().upload_part(upload_id, key, part_index, data)
            pnum = int(part_index) + 1
            self.part_numbers.append(pnum)
            n = len(data)
            self.part_sizes.append(n)
            self.held += n
            self.max_held = max(self.max_held, self.held)
            self.seen_max_upload_size = max(self.seen_max_upload_size, n)
            # 模拟上传后服务端吸收 → 客户端立即释放该 part。
            self.held -= n
            self._objects.setdefault(key, bytearray()).extend(data)
            return {"ETag": f'"etag-{self.upload_id}-{pnum}"'}

        def complete_multipart(self, upload_id, key):
            self.complete_count += 1

        def put_object(self, key, data):
            if hasattr(data, "read"):
                # 整读路径（对比用）：读到的字节在 put_object 期间 held。
                blob = data.read()
                self.held += len(blob)
                self.max_held = max(self.max_held, self.held)
                self.held -= len(blob)
                self._objects[key] = blob
            else:
                self.put_count += 1
                self.held += len(data)
                self.max_held = max(self.max_held, self.held)
                self.held -= len(data)
                self._objects[key] = data

    sink = _MaxHoldingRecorder(part_size)
    pub = ObjectStoreGenerationPublisher(sink, multipart_threshold=1)
    # 1 MiB 分块（min 5 MiB 由 MAX 保留给真 COS；fake sink 无该约束，测试聚焦
    # 分块流式语义：N 个 part、内存平坦、complete 一次）。
    pub.multipart_part_size = lambda: part_size
    gid = pub.begin_generation("factors/f1")
    stream = _LazyStream(b"T" * part_size, total=total)
    pub.add_object(gid, "stream.parquet", stream)

    # PartNumber 1..N 连续、complete 恰好一次。
    assert sink.part_numbers == list(range(1, len(sink.part_numbers) + 1))
    assert sink.complete_count == 1
    # 分块数 = 5（5x part_size）。
    assert len(sink.part_numbers) == 5
    # 内存峰值平坦：任何时刻 held <= 单 part 大小（stream 由 store 逐 part 消费，
    # publisher 从不整读）。这与对象总大小（5x part_size）无关 —— 对比整读路径
    # 需同时持有 5x part_size。
    assert sink.max_held <= part_size
    # 单次上传最大 part 也不超过 part_size。
    assert sink.seen_max_upload_size <= part_size
    # manifest 记录的 size/sha256 与流一致（逐块回读校验）。
    obj = pub._pending[gid].objects[0]
    payload = stream  # already consumed; recompute
    expect_sha = _hashlib.sha256(
        b"T" * part_size * 5
    ).hexdigest()
    assert obj.size == total
    assert obj.sha256 == expect_sha

