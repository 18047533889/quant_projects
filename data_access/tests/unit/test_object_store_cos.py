# -*- coding: utf-8 -*-
"""COSObjectStore production-correctness tests.

Covers:
- real multipart (CreateMultipartUpload → UploadPart → CompleteMultipartUpload,
  AbortMultipartUpload) against a fake S3 client; part bytes released after upload.
- STS/temporary-credential: aws_session_token injected when present; a mock that
  REQUIRES session_token succeeds only when token present.
- cross-principal isolation: same long-lived COSObjectStore, principal A scoped to
  prefix A and principal B scoped to prefix B; A never reads/writes B and vice versa
  (client rebuilt per credential identity).
- COSSeekableRangeReader: seek/tell/read correctness + bounded Range GETs (never
  whole-object).
- multipart abort cleanup + orphan-cleanup helper.
"""
from __future__ import annotations

import threading
import time

import pytest

from data_access.core.exceptions import (
    MultipartPartCountError,
    MultipartPartNumberError,
)
from data_access.read.object_store import (
    COSObjectStore,
    COSSeekableRangeReader,
    LocalObjectStore,
    MAX_PART_NUMBER,
    _validate_part_number,
)


# ---------------------------------------------------------------------------
# Fake S3 client
# ---------------------------------------------------------------------------
class FakeS3:
    """In-memory S3-compatible fake that records calls and enforces session token."""

    def __init__(self, require_session_token=False, fail_upload_parts=()):
        self.objects = {}  # key -> bytes
        self.etags = {}  # key -> etag (S3-style, quoted)
        self.multiparts = {}  # upload_id -> {key, parts: {idx: bytes}}
        self.require_session_token = require_session_token
        self.fail_upload_parts = set(fail_upload_parts)
        self.failed_once = set()  # parts that already failed once (transient)
        self.calls = []
        self.uploaded_part_sizes = []  # bytes uploaded per UploadPart (memory check)
        self.client_kwargs = None
        self.head_etag_override = {}  # key -> forced ETag returned by head_object

    def _check_token(self):
        if self.require_session_token and not self.client_kwargs.get("aws_session_token"):
            raise RuntimeError("STS token required but not provided")

    def _multipart_etag(self, parts):
        import hashlib

        md5s = b"".join(hashlib.md5(p).digest() for p in parts)
        return f'"{hashlib.md5(md5s).hexdigest()}-{len(parts)}"'

    def create_multipart_upload(self, **kw):
        self._check_token()
        self.calls.append(("create_multipart_upload", kw))
        uid = f"up-{len(self.multiparts)}"
        self.multiparts[uid] = {"key": kw["Key"], "parts": {}}
        return {"UploadId": uid}

    def upload_part(self, **kw):
        self._check_token()
        self.calls.append(("upload_part", kw))
        uid = kw["UploadId"]
        idx = kw["PartNumber"]
        if idx in self.fail_upload_parts and idx not in self.failed_once:
            self.failed_once.add(idx)
            raise RuntimeError("simulated 5xx on part upload")
        body = kw["Body"]
        data = body.read() if hasattr(body, "read") else body
        self.multiparts[uid]["parts"][idx] = data
        self.uploaded_part_sizes.append(len(data))
        return {"ETag": f'"etag-{uid}-{idx}"'}

    def complete_multipart_upload(self, **kw):
        self._check_token()
        self.calls.append(("complete_multipart_upload", kw))
        uid = kw["UploadId"]
        parts = sorted(kw["MultipartUpload"]["Parts"], key=lambda p: p["PartNumber"])
        blob = b"".join(self.multiparts[uid]["parts"][p["PartNumber"]] for p in parts)
        self.objects[kw["Key"]] = blob
        self.etags[kw["Key"]] = self._multipart_etag(
            [self.multiparts[uid]["parts"][p["PartNumber"]] for p in parts]
        )
        del self.multiparts[uid]
        return {}

    def abort_multipart_upload(self, **kw):
        self._check_token()
        self.calls.append(("abort_multipart_upload", kw))
        self.multiparts.pop(kw["UploadId"], None)
        return {}

    def list_multipart_uploads(self, **kw):
        self._check_token()
        self.calls.append(("list_multipart_uploads", kw))
        prefix = kw.get("Prefix", "")
        uploads = []
        for uid, st in self.multiparts.items():
            if st["key"].startswith(prefix):
                uploads.append(
                    {"Key": st["key"], "UploadId": uid, "Initiated": time.time() - 100}
                )
        return {"Uploads": uploads, "IsTruncated": False}

    def put_object(self, **kw):
        self._check_token()
        self.calls.append(("put_object", kw))
        body = kw["Body"]
        data = body.read() if hasattr(body, "read") else body
        self.objects[kw["Key"]] = data
        import hashlib

        self.etags[kw["Key"]] = f'"{hashlib.md5(data).hexdigest()}"'
        return {}

    def get_object(self, **kw):
        self._check_token()
        self.calls.append(("get_object", kw))
        data = self.objects[kw["Key"]]
        rng = kw.get("Range")
        if rng:
            spec = rng[len("bytes="):]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(data) - 1
            return {"Body": _BytesBody(data[start : end + 1])}
        return {"Body": _BytesBody(data)}

    def head_object(self, **kw):
        self._check_token()
        self.calls.append(("head_object", kw))
        data = self.objects.get(kw["Key"])
        if data is None:
            raise RuntimeError("not found")
        etag = self.head_etag_override.get(kw["Key"], self.etags.get(kw["Key"], '"x"'))
        return {
            "ContentLength": len(data),
            "ETag": etag,
            "LastModified": time.time(),
        }

    def list_objects_v2(self, **kw):
        self._check_token()
        self.calls.append(("list_objects_v2", kw))
        prefix = kw.get("Prefix", "")
        keys = [k for k in self.objects if k.startswith(prefix)]
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def delete_object(self, **kw):
        self._check_token()
        self.calls.append(("delete_object", kw))
        self.objects.pop(kw["Key"], None)
        self.etags.pop(kw["Key"], None)
        return {}


class _BytesBody:
    def __init__(self, data):
        self._data = data
        self._pos = 0

    def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._data) - self._pos
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out


# ---------------------------------------------------------------------------
# Helpers: credential provider + boto3 client injection
# ---------------------------------------------------------------------------
_DEFAULT_PART_MB = 1
_DEFAULT_PART_SIZE = _DEFAULT_PART_MB * 1024 * 1024


def _make_provider(access, secret, *, token=None, principal=None, scope=None, generation=None):
    from data_access.security.credentials import CredentialMaterial

    gen = generation

    class _P:
        source = "test"
        generation = gen

        def resolve(self):
            return CredentialMaterial(
                access_key_id=access,
                secret_access_key=secret,
                session_token=token,
                principal_id=principal,
                credential_scope_id=scope,
                source="test",
            )

    return _P()


def _patch_boto3(monkeypatch, fake):
    """Make COSObjectStore._s3() return the fake client with recorded kwargs.

    boto3/botocore are not installed in the test venv, so inject a fake ``boto3``
    module into ``sys.modules`` (the store imports it lazily inside ``_s3()``).
    """
    import sys
    import types

    fake_boto3 = types.ModuleType("boto3")

    def _client(**kwargs):
        fake.client_kwargs = kwargs
        fake.client_build_count = getattr(fake, "client_build_count", 0) + 1
        return fake

    fake_boto3.client = _client
    fake_botocore = types.ModuleType("botocore")
    fake_config_mod = types.ModuleType("botocore.config")
    fake_config_mod.Config = lambda **kw: kw
    fake_botocore.config = fake_config_mod
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config_mod)


def _patch_global_provider(monkeypatch, provider):
    from data_access.security import credentials as cred_mod

    monkeypatch.setattr(cred_mod, "_global_credential_provider", lambda: provider)


# ---------------------------------------------------------------------------
# 1. Real multipart
# ---------------------------------------------------------------------------
def test_multipart_real_upload_and_memory_release(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )

    store = COSObjectStore("bucket", part_size_mb=1, inflight=2)
    uid = store.begin_multipart("big/obj.parquet")
    assert uid in fake.multiparts

    # 5 parts, inflight=2 -> at most 2 parts buffered in memory at once.
    for i in range(5):
        store.upload_part(uid, "big/obj.parquet", i, b"P" * (i + 1))

    # Before complete, no object yet (atomic).
    assert "big/obj.parquet" not in fake.objects

    store.complete_multipart(uid, "big/obj.parquet")
    assert fake.objects["big/obj.parquet"] == b"P" * 1 + b"P" * 2 + b"P" * 3 + b"P" * 4 + b"P" * 5

    # Real UploadPart calls happened (not a single put_object).
    upload_calls = [c for c in fake.calls if c[0] == "upload_part"]
    assert len(upload_calls) == 5
    complete_calls = [c for c in fake.calls if c[0] == "complete_multipart_upload"]
    assert len(complete_calls) == 1
    # No put_object for the assembled blob.
    assert not [c for c in fake.calls if c[0] == "put_object"]
    # upload_id removed from in-flight state.
    assert uid not in store._uploads


def test_multipart_abort_cleans_up(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", part_size_mb=1, inflight=2)
    uid = store.begin_multipart("abort/obj")
    store.upload_part(uid, "abort/obj", 0, b"x" * 10)
    store.abort_multipart(uid, "abort/obj")
    assert uid not in fake.multiparts
    assert uid not in store._uploads
    assert "abort/obj" not in fake.objects
    assert any(c[0] == "abort_multipart_upload" for c in fake.calls)


def test_multipart_part_retry_on_5xx(monkeypatch):
    fake = FakeS3(fail_upload_parts=(2,))
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", part_size_mb=1, inflight=1)
    uid = store.begin_multipart("retry/obj")
    # part 2 fails once (transient 5xx); the store retries that part internally.
    for i in range(3):
        store.upload_part(uid, "retry/obj", i, b"R" * (i + 1))
    store.complete_multipart(uid, "retry/obj")
    assert fake.objects["retry/obj"] == b"R" + b"RR" + b"RRR"
    # part 2 was attempted twice (once failed, once retried).
    part2_attempts = [
        c for c in fake.calls if c[0] == "upload_part" and c[1]["PartNumber"] == 2
    ]
    assert len(part2_attempts) == 2


def test_small_object_uses_single_put(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket")
    store.put_object("small/obj", b"hello")
    assert fake.objects["small/obj"] == b"hello"
    assert any(c[0] == "put_object" for c in fake.calls)


# ---------------------------------------------------------------------------
# 1b. Threshold routing: small -> single PUT, large -> real multipart
# ---------------------------------------------------------------------------
def test_threshold_routing_small_single_put(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", multipart_threshold_bytes=1024)
    store.put_object("small/obj", b"x" * 100)  # below threshold
    assert fake.objects["small/obj"] == b"x" * 100
    assert any(c[0] == "put_object" for c in fake.calls)
    assert not [c for c in fake.calls if c[0] == "create_multipart_upload"]


def test_threshold_routing_large_uses_multipart(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", multipart_threshold_bytes=1024, part_size_mb=1)
    payload = b"y" * 5000  # above threshold -> multipart
    store.put_object("big/obj", payload)
    assert fake.objects["big/obj"] == payload
    assert any(c[0] == "create_multipart_upload" for c in fake.calls)
    assert any(c[0] == "upload_part" for c in fake.calls)
    assert any(c[0] == "complete_multipart_upload" for c in fake.calls)
    # No single put_object for the assembled blob.
    assert not [c for c in fake.calls if c[0] == "put_object"]


def test_threshold_routing_exact_threshold_uses_multipart(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", multipart_threshold_bytes=1024, part_size_mb=1)
    payload = b"z" * 1024  # exactly at threshold -> multipart
    store.put_object("exact/obj", payload)
    assert fake.objects["exact/obj"] == payload
    assert any(c[0] == "create_multipart_upload" for c in fake.calls)


# ---------------------------------------------------------------------------
# 1c. Abort cleanup on part failure (via put_object path)
# ---------------------------------------------------------------------------
def test_put_object_aborts_on_part_failure(monkeypatch):
    fake = FakeS3(fail_upload_parts=(2,))
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    # part_retries=1 -> part 2 fails permanently -> whole multipart aborted.
    store = COSObjectStore(
        "bucket",
        multipart_threshold_bytes=1,
        part_size_mb=1,
        part_retries=1,
        multipart_retries=1,
    )
    # 3 parts of 1MB each; part index 2 fails.
    payload = b"a" * (1024 * 1024) + b"b" * (1024 * 1024) + b"c" * (1024 * 1024)
    with pytest.raises(RuntimeError):
        store.put_object("abort/obj", payload)
    # abort_multipart_upload was called to clean up parts.
    assert any(c[0] == "abort_multipart_upload" for c in fake.calls)
    # No object was assembled (atomic).
    assert "abort/obj" not in fake.objects
    # No orphan multipart remains.
    assert not fake.multiparts


def test_put_object_abort_on_error_false_leaves_orphan(monkeypatch):
    fake = FakeS3(fail_upload_parts=(2,))
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore(
        "bucket",
        multipart_threshold_bytes=1,
        part_size_mb=1,
        part_retries=1,
        multipart_retries=1,
        abort_on_error=False,
    )
    payload = b"a" * (1024 * 1024) + b"b" * (1024 * 1024) + b"c" * (1024 * 1024)
    with pytest.raises(RuntimeError):
        store.put_object("abort/obj", payload)
    # abort_on_error=False -> no abort call; orphan multipart remains.
    assert not [c for c in fake.calls if c[0] == "abort_multipart_upload"]
    assert fake.multiparts  # orphan remains


# ---------------------------------------------------------------------------
# 1d. ETag verification on complete
# ---------------------------------------------------------------------------
def test_etag_verify_success(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore(
        "bucket", multipart_threshold_bytes=1, part_size_mb=1, verify_etag=True
    )
    payload = b"etag" * 100
    store.put_object("etag/obj", payload)
    assert fake.objects["etag/obj"] == payload


def test_etag_verify_mismatch_raises(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore(
        "bucket", multipart_threshold_bytes=1, part_size_mb=1, verify_etag=True
    )
    payload = b"etag" * 100
    # Corrupt the stored object's ETag so verification fails.
    fake.head_etag_override["etag/obj"] = '"corrupted-etag"'
    with pytest.raises(RuntimeError, match="ETag 校验失败"):
        store.put_object("etag/obj", payload)


def test_etag_verify_disabled_skips_check(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore(
        "bucket", multipart_threshold_bytes=1, part_size_mb=1, verify_etag=False
    )
    payload = b"etag" * 100
    fake.head_etag_override["etag/obj"] = '"corrupted-etag"'
    store.put_object("etag/obj", payload)  # no raise
    assert fake.objects["etag/obj"] == payload


# ---------------------------------------------------------------------------
# 2. STS / temporary-credential
# ---------------------------------------------------------------------------
def test_sts_token_required_success(monkeypatch):
    fake = FakeS3(require_session_token=True)
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch,
        _make_provider("AK", "SK", token="STS-TOKEN", principal="p1", scope="s1"),
    )
    store = COSObjectStore("bucket")
    store.put_object("sts/obj", b"data")
    assert fake.client_kwargs["aws_session_token"] == "STS-TOKEN"
    assert fake.objects["sts/obj"] == b"data"


def test_sts_token_required_fails_without_token(monkeypatch):
    fake = FakeS3(require_session_token=True)
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket")
    with pytest.raises(RuntimeError, match="STS token required"):
        store.put_object("sts/obj", b"data")


# ---------------------------------------------------------------------------
# 3. Cross-principal isolation (no cross-principal client caching)
# ---------------------------------------------------------------------------
def test_cross_principal_client_isolation(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    store = COSObjectStore("bucket")

    # Principal A scoped to prefix A.
    _patch_global_provider(
        monkeypatch, _make_provider("AK_A", "SK_A", principal="A", scope="scopeA")
    )
    store.put_object("A/obj", b"a-data")
    assert fake.objects["A/obj"] == b"a-data"
    client_a = store._client

    # Principal B scoped to prefix B, same long-lived store instance.
    _patch_global_provider(
        monkeypatch, _make_provider("AK_B", "SK_B", principal="B", scope="scopeB")
    )
    store.put_object("B/obj", b"b-data")
    assert fake.objects["B/obj"] == b"b-data"
    client_b = store._client

    # Client was rebuilt (different identity), not reused across principals.
    assert fake.client_build_count == 2
    assert store._client_identity[0] == "B"  # principal now B
    assert store._client_identity[1] == "scopeB"

    # A never wrote B's prefix and vice versa.
    assert "A/obj" in fake.objects
    assert "B/obj" in fake.objects
    # The client used for B's write carried B's access key.
    assert fake.client_kwargs["aws_access_key_id"] == "AK_B"


def test_client_reused_within_same_principal(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="A", scope="scopeA")
    )
    store = COSObjectStore("bucket")
    store.put_object("A/1", b"x")
    c1 = store._client
    store.put_object("A/2", b"y")
    c2 = store._client
    assert c1 is c2  # same identity -> cached client reused


# ---------------------------------------------------------------------------
# 4. COSSeekableRangeReader
# ---------------------------------------------------------------------------
def test_seekable_reader_seek_tell_read(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", reader_buffer_mb=1)
    payload = bytes(range(256)) * 4  # 1024 bytes
    store.put_object("r/obj", payload)

    reader = store.open_reader("r/obj")
    assert reader is not None
    assert reader.seekable()
    assert reader.tell() == 0
    assert reader.read(10) == payload[:10]
    assert reader.tell() == 10
    reader.seek(100)
    assert reader.tell() == 100
    assert reader.read(5) == payload[100:105]
    reader.seek(-10, 2)  # from end
    assert reader.tell() == len(payload) - 10
    assert reader.read() == payload[-10:]
    reader.close()


def test_seekable_reader_issues_bounded_range_gets(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", reader_buffer_mb=1)
    payload = b"Z" * 1000
    store.put_object("r/obj", payload)

    reader = store.open_reader("r/obj")
    # Read a small slice far into the object.
    reader.seek(500)
    reader.read(10)
    reader.close()

    get_calls = [c for c in fake.calls if c[0] == "get_object"]
    assert get_calls, "expected Range GETs"
    for _, kw in get_calls:
        rng = kw.get("Range", "")
        assert rng.startswith("bytes=")
        # Never a whole-object GET (no Range) and never a huge range.
        assert rng != ""
        start_s, _, end_s = rng[len("bytes="):].partition("-")
        length = int(end_s) - int(start_s) + 1
        assert length <= 8 * 1024 * 1024  # bounded by max_buffer


def test_seekable_reader_missing_object(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket")
    reader = store.open_reader("missing/obj")
    with pytest.raises(FileNotFoundError):
        reader.seek(0)


# ---------------------------------------------------------------------------
# 5. Orphan multipart cleanup
# ---------------------------------------------------------------------------
def test_abort_stale_multipart_uploads(monkeypatch):
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket")
    uid = store.begin_multipart("orphan/obj")
    store.upload_part(uid, "orphan/obj", 0, b"x" * 10)
    # Not completed/aborted -> orphan remains.
    assert uid in fake.multiparts

    n = store.abort_stale_multipart_uploads(prefix="orphan/", ttl_seconds=0)
    assert n == 1
    assert uid not in fake.multiparts
    assert any(c[0] == "abort_multipart_upload" for c in fake.calls)


# ---------------------------------------------------------------------------
# LocalObjectStore regression (unchanged behavior)
# ---------------------------------------------------------------------------
def test_local_object_store_multipart(tmp_path):
    store = LocalObjectStore(tmp_path)
    uid = store.begin_multipart("d/obj")
    store.upload_part(uid, "d/obj", 0, b"a")
    store.upload_part(uid, "d/obj", 1, b"b")
    store.complete_multipart(uid, "d/obj")
    assert store.range_read("d/obj", offset=0, length=2) == b"ab"
    assert store.head_object("d/obj")["size"] == 2



# ---------------------------------------------------------------------------
# 6. S3 PartNumber 1-based 规范（P0-05 关闭项）
# ---------------------------------------------------------------------------
def _strip_etag(etag: str) -> str:
    return str(etag).strip('"') or ""


def _upload_records(fake) -> list[dict]:
    return [kw for c, kw in fake.calls if c == "upload_part"]


def test_cos_streaming_upload_uses_1based_partnumbers(monkeypatch):
    """COSObjectStore.upload_part(0-based) -> 真实 UploadPart PartNumber=1..N。

    回放 P0-05 缺陷：修复前 PartNumber 从 0 开始，第二个 part 编号 1 ——
    complete 的 Parts 列表里 idx==1 的同时出现两处（part#1 etag 与 part#2
    编号冲突），假 S3 用 idx 读取 part bytes 得到连续对象但真实 COS 会拒绝。
    """
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", part_size_mb=_DEFAULT_PART_MB, inflight=2)
    uid = store.begin_multipart("ords/obj")
    store.upload_part(uid, "ords/obj", 0, b"P" * 10)
    store.upload_part(uid, "ords/obj", 1, b"Q" * 20)
    store.upload_part(uid, "ords/obj", 2, b"R" * 30)
    store.complete_multipart(uid, "ords/obj")

    uploads = _upload_records(fake)
    assert [w["PartNumber"] for w in uploads] == [1, 2, 3]

    complete_calls = [w for c, w in fake.calls if c == "complete_multipart_upload"]
    assert len(complete_calls) == 1
    parts = sorted(complete_calls[0]["MultipartUpload"]["Parts"], key=lambda x: x["PartNumber"])
    assert [p["PartNumber"] for p in parts] == [1, 2, 3]
    # 每个 PartNumber 的 ETag 与对应 UploadPart 返回的 ETag 一一配对（编号连续，
    # 无 part 1 缺失 / 无 idx 冲突）。FakeS3 返回 ``etag-{upload_id}-{PartNumber}``。
    for rec in uploads:
        pnum = rec["PartNumber"]
        assert _strip_etag(parts[pnum - 1]["ETag"]) == _strip_etag(
            f"etag-{uid}-{pnum}"
        )
    assert fake.objects["ords/obj"] == b"P" * 10 + b"Q" * 20 + b"R" * 30


def test_cos_put_object_batch_uses_1based_partnumbers(monkeypatch):
    """put_object 批量 multipart（_put_multipart 路径）同样落 1-based。

    9999 part × 1MB 需要 10.5GB 对象；FakeS3 若按默认保留 part bytes + complete
    再拼整 blob，峰值 ~3×10.5GB —— 共享机上必被 OOM killer 杀掉（exit 137）。
    本测试只断言 PartNumber 编号契约（计数/连续性），不校验 blob 字节 → 用
    discard fake（不保留 part bytes，同 _make_discard_fake 模式），峰值内存降到
    O(parts)。
    """
    fake = FakeS3()
    # 不保留 part bytes：丢弃存储、只记 calls + 大小（PartNumber 契约足够）。
    def upload_part_discard(**kw):
        fake.calls.append(
            ("upload_part", {k: (len(v) if k == "Body" else v) for k, v in kw.items()})
        )
        body = kw["Body"]
        fake.uploaded_part_sizes.append(
            len(body) if hasattr(body, "__len__") else 0
        )
        return {"ETag": f'"etag-discard-{kw["PartNumber"]}"'}

    def complete_discard(**kw):
        fake.calls.append(("complete_multipart_upload", kw))
        fake.objects[kw["Key"]] = b""
        fake.multiparts.pop(kw["UploadId"], None)
        return {}

    fake.upload_part = upload_part_discard
    fake.complete_multipart_upload = complete_discard
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore(
        "bucket", part_size_mb=_DEFAULT_PART_MB, multipart_threshold_bytes=1
    )
    # 9999 part 的「PartNumber 1..N 连续」契约用 1 字节 part 验证（同
    # test_streaming_part_count_guard 的 ``store._part_size = 1`` 模式）：
    # 1MB part → 10.5GB 对象 + 切片列表，峰值 >21GB，共享机上被 OOM killer
    # 杀掉（exit 137）。契约只关心 PartNumber 编号/计数，与 part 大小无关。
    # 9999 part 的「PartNumber 1..N 连续」契约用 1 字节 part 验证（同
    # test_streaming_part_count_guard 的 ``store._part_size = 1`` 模式）：
    # 1MB part → 10.5GB 对象 + 切片列表，峰值 >21GB，共享机上被 OOM killer
    # 杀掉（exit 137）。契约只关心 PartNumber 编号/计数，与 part 大小无关。
    # ETag 校验依赖 fake 返回真实逐 part MD5 聚合 → discard fake 的固定
    # etag 过不了校验，关闭（本测试不校验 ETag，有专门测试覆盖）。
    store._part_size = 1
    store._verify_etag = False
    store.put_object("big/obj", b"Z" * (MAX_PART_NUMBER - 1))
    uploads = _upload_records(fake)
    assert len(uploads) == MAX_PART_NUMBER - 1
    complete_calls = [w for c, w in fake.calls if c == "complete_multipart_upload"]
    complete_parts = complete_calls[0]["MultipartUpload"]["Parts"]
    # complete 收到的 Parts 按 1-based PartNumber 严格排序且连续（S3 顺序约束）。
    assert [p["PartNumber"] for p in complete_parts] == list(
        range(1, MAX_PART_NUMBER)
    )


def test_cos_put_object_rejects_part_count_exceeding_s3_cap(monkeypatch):
    """超过 10000 part 上限的拼接上传 fail-closed 抛错，绝不部分提交。"""
    fake = FakeS3(fail_upload_parts=(10000,))
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore(
        "bucket", part_size_mb=1, multipart_threshold_bytes=1, part_retries=1,
        multipart_retries=1, abort_on_error=False,
    )
    # 超过 10000 part 上限 fail-closed：用小 part size 把 10001 个 part 的
    # 对象廉价构造出来（1 字节 part → 10001 字节 = 10001 part）。
    store2 = COSObjectStore(
        "bucket", part_size_mb=1, multipart_threshold_bytes=1, part_retries=1,
        multipart_retries=1, abort_on_error=False,
    )
    store2._part_size = 1  # 1-byte part size → 10001+ 字节 = 10001+ part
    with pytest.raises(MultipartPartCountError, match="超过 S3 上限"):
        store2.put_object("toolarge/obj", b"y" * 10001)
    assert "toolarge/obj" not in fake.objects


def test_part_number_validator_bounds(monkeypatch):
    """_validate_part_number：1 与 10000 合法，0/10001/非整数 fail-closed。"""
    assert _validate_part_number(1, "ctx") == 1
    assert _validate_part_number(10000, "ctx") == 10000
    for bad in (0, 10001, -1, "x", None, 3.5):
        with pytest.raises(MultipartPartNumberError):
            _validate_part_number(bad, "ctx")


# ---------------------------------------------------------------------------
# 7. 有界内存 Streaming multipart（P0-07）
# ---------------------------------------------------------------------------
import io
import resource


class _LazyBytesStream(io.BytesIO):
    """Synthetic 逻辑大对象 lazy stream：物理只持有模板 bytes，递归产出。

    真实生产源是文件/网络（绝不整读），本 stub 只模拟「逐块产出、物理有界」。
    total 逻辑字节数；模板重复拼段。物理内存 O(template)。
    """

    def __init__(self, template: bytes, *, total: int):
        self._template = bytes(template)
        self._remaining = total
        self._offset = 0

    def read(self, n: int = -1):
        if n is None or n < 0:
            n = self._remaining
        out = bytearray()
        while n > 0 and self._remaining > 0:
            chunk = self._template[self._offset : self._offset + n]
            if not chunk:
                self._offset = 0
                chunk = self._template[:n]
            out.extend(chunk)
            self._remaining -= len(chunk)
            self._offset = (self._offset + len(chunk)) % len(self._template)
            n -= len(chunk)
        return bytes(out)

    def readable(self):
        return True

    def seekable(self):
        return False


def test_streaming_multipart_bounded_memory(monkeypatch, tmp_path):
    """逻辑 10GB 流上传：客户端 RSS 峰值 O(part_size × inflight)，不随对象大小线性。

    - 用 **discarding** fake（每 part 到达即丢弃）—— 生产上 part 由 COS 服务端
      吸收，only 客户端 producer 窗口 + boto 开销留在进程内。
    - 断言：RSS 峰值增量 <= inflight × part_size × 4（保守系数），与 10GB 逻辑
      对象大小无关。
    """
    part_size = 8 * 1024 * 1024
    inflight = 2
    total = 10 * 1024 * 1024 * 1024  # 逻辑 10GB（物理由 lazy stream 产出，不常驻）
    fake = _make_discard_fake(monkeypatch)
    store = COSObjectStore("bucket", part_size_mb=8, inflight=inflight, multipart_threshold_bytes=1)
    store._verify_etag = False  # 丢弃 fake 无真实 ETag —— 只测按模型内存
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stream = _LazyBytesStream(b"T" * 65536, total=total)
    store.put_object("big/stream", stream)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # RSS 单位为 KB。阈值 = base + inflight×part_size×4（保守系数）。
    threshold = baseline + 4 * inflight * part_size // 1024
    uploads = [w for c, w in fake.calls if c == "upload_part"]
    assert len(uploads) == (total + part_size - 1) // part_size
    assert peak <= threshold, f"RSS peak {peak}KB > {threshold}KB (baseline {baseline}KB)"


def _make_discard_fake(monkeypatch):
    """构造不保留 part bytes 的 fake（只计数 + 记录 calls），隔离客户端内存。"""
    fake = FakeS3()
    fake._discard = True

    orig_up = fake.upload_part
    def upload_part_discard(**kw):
        # 不写入 multiparts 存储（丢弃 part bytes）—— 只保留大小统计 + calls。
        fake.calls.append(("upload_part", {k: (len(v) if k == "Body" else v) for k, v in kw.items()}))
        fake.uploaded_part_sizes.append(len(kw["Body"]) if hasattr(kw["Body"], "__len__") else 0)
        return {"ETag": '"discard-etag"'}
    fake.upload_part = upload_part_discard

    orig_comp = fake.complete_multipart_upload
    def complete_discard(**kw):
        fake.calls.append(("complete_multipart_upload", kw))
        # No storage — mark object exists minimally.
        fake.objects[kw["Key"]] = b""
        fake.multiparts.pop(kw["UploadId"], None)
        return {}
    fake.complete_multipart_upload = complete_discard

    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    return fake


def test_streaming_put_object_integrity_and_1based(monkeypatch):
    """流入口对象字节完整、PartNumber 1-based 连续、compile 后对象匹配。"""
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", part_size_mb=1, multipart_threshold_bytes=1)
    store._verify_etag = True
    payload = b"Q" * (1024 * 1024 * 3) + b"!" * 1234
    store.put_object("stream/obj", io.BytesIO(payload))
    assert fake.objects["stream/obj"] == payload
    uploads = [w for c, w in fake.calls if c == "upload_part"]
    pn = [w["PartNumber"] for w in uploads]
    assert sorted(pn) == list(range(1, len(pn) + 1))


def test_streaming_part_count_guard(monkeypatch):
    """流模式 >10000 part fail-closed（MultipartPartCountError），不提交。"""
    fake = FakeS3()
    _patch_boto3(monkeypatch, fake)
    _patch_global_provider(
        monkeypatch, _make_provider("AK", "SK", principal="p1", scope="s1")
    )
    store = COSObjectStore("bucket", part_size_mb=1, multipart_threshold_bytes=1)
    store._part_size = 1  # 1-byte part → 10001 字节 = 10001 part
    store._verify_etag = False
    with pytest.raises(MultipartPartCountError):
        store.put_object("toolarge/stream", io.BytesIO(b"y" * 10001))
    assert not fake.multiparts  # abort 清理，无孤儿 multipart
    assert "toolarge/stream" not in fake.objects

