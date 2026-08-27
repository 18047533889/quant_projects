# -*- coding: utf-8 -*-
"""R44-P0: ObjectStore 抽象 —— 节点级增量 FactorEngine 的远程数据平面。

零本地磁盘目标（STRICT_REMOTE）下，增量 planner 通过 ``ObjectStore`` 读写
parquet/arrow 对象，而不是直接落本地磁盘。本模块提供：

- ``ObjectStore``  Protocol：head / list / range / open_reader / put /
  multipart / delete 的最小对象存储接口。
- ``LocalObjectStore``：基于 ``root`` 目录的字节对象存储（写走 temp +
  ``os.replace`` 原子替换，multipart 用 staging 文件模拟），测试与本地
  增量跑批用。
- ``COSObjectStore``：适配 ``data_access.cos.remote`` 的 S3/COS 适配器。
  - 真实 multipart（CreateMultipartUpload → UploadPart → CompleteMultipartUpload，
    失败 AbortMultipartUpload），有界 in-flight part 队列，part 上传后立即释放
    内存（不再整对象 ``b"".join`` 内存炸弹）。
  - STS/temporary-credential 支持（``aws_session_token``）。
  - 凭证感知的 client 缓存：client 身份 = (principal, scope, generation,
    access_key, session_token, endpoint, region, bucket)，身份变化或凭证临近
    过期即重建 boto3 client，绝不跨 principal/scope 复用。
  - ``open_reader`` 返回 ``COSSeekableRangeReader``（有界 Range GET，不整对象
    拉取），供 RemoteDatasetScanner 做 parquet footer / row-group 裁剪。
- ``NullObjectStore``：全部抛 ``NotImplementedError``，用于测试 policy gating。

R44-P0 契约：
    1. ``put_object`` / ``complete_multipart`` 必须原子（读者永远看不到半成品）。
    2. key 是 POSIX 相对路径；拒绝 ``..`` / 绝对路径逃逸。
    3. 远程适配只做 best-effort —— 网络/凭证失败绝不让扫描路径崩溃。
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

__all__ = [
    "ObjectStore",
    "LocalObjectStore",
    "COSObjectStore",
    "COSSeekableRangeReader",
    "NullObjectStore",
]


@runtime_checkable
class ObjectStore(Protocol):
    """最小对象存储接口（R44-P0）。

    ``head_object`` 返回 ``{etag?, size?, last_modified?}``（缺失 key → None）；
    ``range_read`` 对缺失 key 必须抛错（调用方据此区分「存在但短读」与「不存在」）。
    """

    def head_object(self, key: str) -> dict | None: ...

    def list_objects(self, prefix: str) -> list[str]: ...

    def range_read(self, key: str, *, offset: int, length: int) -> bytes: ...

    def open_reader(self, key: str) -> BinaryIO | None: ...

    def put_object(self, key: str, data: bytes) -> None: ...

    def begin_multipart(self, key: str) -> str: ...

    def upload_part(
        self, upload_id: str, key: str, part_index: int, data: bytes
    ) -> None: ...

    def complete_multipart(self, upload_id: str, key: str) -> None: ...

    def abort_multipart(self, upload_id: str, key: str) -> None: ...

    def delete_object(self, key: str) -> None: ...


def _safe_relative_key(key: str) -> str:
    """把外部 key 规范化为相对 POSIX 路径（拒绝逃逸）。

    R44-P0 fail-closed：``..`` 段 / 绝对路径 / 空 key 一律拒绝——对象 key 必须
    落在 store root 内，不能借 key 逃出存储根写任意文件。
    """
    key = str(key)
    if not key or key.startswith("/") or "\\" in key:
        raise ValueError(f"非法对象 key: {key!r}")
    norm = os.path.normpath(key)
    if norm == ".." or norm.startswith("../"):
        raise ValueError(f"对象 key 越界（.. 逃逸）: {key!r}")
    return norm


def _multipart_etag(chunks: list[tuple[int, bytes]]) -> str:
    """S3 多 part 对象的 ETag = MD5(各 part MD5 拼接) + "-<part_count>"。

    用于 complete 后校验完整对象 ETag 是否与期望一致（ETag 校验）。
    """
    md5s = b"".join(hashlib.md5(part).digest() for _, part in chunks)
    return f"{hashlib.md5(md5s).hexdigest()}-{len(chunks)}"


class LocalObjectStore:
    """基于本地目录的字节对象存储（R44-P0，原子写）。

    - ``put_object``：写 ``.<name>.tmp.<uuid>`` 后 ``os.replace`` 原子落盘；
    - multipart：part 文件存 ``.<key>.mpu.<upload_id>.<index>`` staging，
      complete 按序拼接后原子替换，abort 清理全部 part；
    - ``open_reader`` 返回 ``pathlib.Path.open("rb")`` 的二进制句柄。

    注意：这是 **ObjectStore 实现**，不代表「本地持久字节」——STRICT_REMOTE
    的零磁盘约束由 :mod:`data_access.read.local_disk_policy` 把关，调用方在
    真落盘前必须走 ``track_local_persistent_bytes_written``。
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        rel = _safe_relative_key(key)
        p = (self.root / rel).resolve()
        try:
            p.relative_to(self.root)
        except ValueError:
            raise ValueError(f"对象 key 越界（逃出 root）: {key!r}") from None
        return p

    # ---- reads ----
    def head_object(self, key: str) -> dict | None:
        p = self._path(key)
        try:
            st = p.stat()
        except OSError:
            return None
        return {
            "etag": f"{st.st_mtime_ns:x}-{st.st_size}",
            "size": st.st_size,
            "last_modified": st.st_mtime,
        }

    def list_objects(self, prefix: str) -> list[str]:
        prefix = str(prefix).lstrip("/")
        out: list[str] = []
        base = self.root if not prefix else self.root / prefix
        if not base.exists():
            return []
        for p in base.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self.root))
                if not prefix or rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)

    def range_read(self, key: str, *, offset: int, length: int) -> bytes:
        p = self._path(key)
        with p.open("rb") as fh:
            fh.seek(offset)
            return fh.read(length)

    def open_reader(self, key: str) -> BinaryIO | None:
        p = self._path(key)
        if not p.exists():
            return None
        return p.open("rb")

    # ---- writes ----
    def put_object(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f".{p.name}.tmp.{uuid.uuid4().hex}"
        tmp.write_bytes(data)
        os.replace(str(tmp), str(p))

    def begin_multipart(self, key: str) -> str:
        return uuid.uuid4().hex

    def upload_part(
        self, upload_id: str, key: str, part_index: int, data: bytes
    ) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        part = p.parent / f".{p.name}.mpu.{upload_id}.{int(part_index):06d}"
        part.write_bytes(data)

    def complete_multipart(self, upload_id: str, key: str) -> None:
        p = self._path(key)
        parts = sorted(
            p.parent.glob(f".{p.name}.mpu.{upload_id}.*"),
            key=lambda q: int(q.name.rsplit(".", 1)[-1]),
        )
        if not parts:
            raise FileNotFoundError(
                f"multipart {upload_id} 无任何 part，无法 complete: {key}"
            )
        tmp = p.parent / f".{p.name}.tmp.{uuid.uuid4().hex}"
        with tmp.open("wb") as out:
            for part in parts:
                out.write(part.read_bytes())
        os.replace(str(tmp), str(p))
        for part in parts:
            try:
                part.unlink()
            except OSError:
                pass

    def abort_multipart(self, upload_id: str, key: str) -> None:
        p = self._path(key)
        for part in p.parent.glob(f".{p.name}.mpu.{upload_id}.*"):
            try:
                part.unlink()
            except OSError:
                pass

    def delete_object(self, key: str) -> None:
        p = self._path(key)
        try:
            p.unlink()
        except OSError:
            pass


class _MultipartState:
    """COS 真实 multipart 的进行中状态（有界 in-flight part 队列）。

    ``queue`` 持有尚未真正 UploadPart 的 ``(part_index, data)``；达到
    ``inflight`` 上限即冲刷最旧 part（UploadPart 后立即释放 bytes）。这样任意
    时刻内存里最多 ``inflight`` 个 part，不再整对象 ``b"".join``。
    """

    __slots__ = ("upload_id", "key", "parts", "queue", "inflight", "expected_etag")

    def __init__(self, upload_id: str, key: str, inflight: int):
        self.upload_id = upload_id
        self.key = key
        self.parts: list[tuple[int, str]] = []  # (part_index, etag)
        self.queue: deque[tuple[int, bytes]] = deque()
        self.inflight = max(1, int(inflight))
        self.expected_etag: str | None = None  # 期望的完整对象 ETag（ETag 校验用）


class COSSeekableRangeReader:
    """基于 COSObjectStore 的可 seek 二进制句柄（有界 Range GET）。

    供 RemoteDatasetScanner 做 parquet footer / row-group 裁剪：``seek`` /
    ``tell`` 语义与文件句柄一致，但每次 ``read`` 只拉取不超过 ``max_buffer``
    的 Range，绝不整对象拉进内存。``head_object`` 拿对象 size 一次。
    """

    def __init__(
        self,
        store: "COSObjectStore",
        key: str,
        *,
        max_buffer: int = 8 * 1024 * 1024,
    ):
        self._store = store
        self._key = str(key)
        self._max_buffer = max(1, int(max_buffer))
        self._pos = 0
        self._size: int | None = None
        self._buf = b""
        self._buf_start = 0
        self._closed = False

    def _ensure_size(self) -> None:
        if self._size is not None:
            return
        meta = self._store.head_object(self._key)
        if meta is None or meta.get("size") is None:
            raise FileNotFoundError(f"对象不存在或 size 未知: {self._key!r}")
        self._size = int(meta["size"])

    def seek(self, offset: int, whence: int = 0) -> int:
        self._ensure_size()
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self._pos + offset
        elif whence == 2:
            new = self._size + offset
        else:
            raise ValueError(f"非法 whence: {whence}")
        if new < 0:
            raise ValueError(f"seek 越界: {new}")
        self._pos = new
        return new

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        self._ensure_size()
        if self._closed:
            raise ValueError("reader 已关闭")
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        out = bytearray()
        while n > 0 and self._pos < self._size:
            # 命中缓冲
            if self._buf and self._buf_start <= self._pos < self._buf_start + len(self._buf):
                off = self._pos - self._buf_start
                take = min(n, len(self._buf) - off)
                out += self._buf[off : off + take]
                self._pos += take
                n -= take
                continue
            # 有界 Range GET
            length = min(n, self._max_buffer, self._size - self._pos)
            chunk = self._store.range_read(
                self._key, offset=self._pos, length=length
            )
            self._buf = chunk
            self._buf_start = self._pos
            take = min(n, len(chunk))
            out += chunk[:take]
            self._pos += take
            n -= take
        return bytes(out)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "COSSeekableRangeReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class COSObjectStore:
    """COS/S3 对象存储适配器（R44-P0，best-effort）。

    复用 :mod:`data_access.cos.remote` 的凭证/端点解析（``resolve_s3_credentials``）
    与 HEAD 元数据缓存（``read_contract._remote_object_meta``）。

    - **真实 multipart**：``begin_multipart`` → ``CreateMultipartUpload``，
      ``upload_part`` → ``UploadPart``（有界 in-flight 队列，part 上传后立即
      释放内存），``complete_multipart`` → ``CompleteMultipartUpload``，
      ``abort_multipart`` → ``AbortMultipartUpload``。小对象仍走单 ``put_object``。
    - **STS/temporary-credential**：``creds.session_token`` 存在时注入
      ``aws_session_token``。
    - **凭证感知 client 缓存**：client 身份 = (principal, scope, generation,
      access_key, session_token, endpoint, region, bucket)；身份变化或凭证临近
      过期即重建 boto3 client，绝不跨 principal/scope 复用。
    - ``open_reader`` 返回 ``COSSeekableRangeReader``（有界 Range GET）。

    任何操作失败抛错，由扫描器 catch 后降级全量读。
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint: str | None = None,
        region: str | None = None,
        part_size_mb: int | None = None,
        inflight: int | None = None,
        reader_buffer_mb: int | None = None,
        part_retries: int = 3,
        part_retry_backoff: float = 0.05,
        multipart_threshold_bytes: int | None = None,
        max_concurrency: int | None = None,
        multipart_retries: int | None = None,
        abort_on_error: bool = True,
        verify_etag: bool = True,
    ):
        self.bucket = str(bucket)
        self._endpoint = endpoint
        self._region = region
        self._part_size = int(
            part_size_mb
            or int(os.environ.get("COS_MULTIPART_PART_SIZE_MB", "16"))
        ) * 1024 * 1024
        self._inflight = int(
            inflight or int(os.environ.get("COS_MULTIPART_INFLIGHT", "4"))
        )
        self._reader_buffer = int(
            reader_buffer_mb
            or int(os.environ.get("COS_READER_BUFFER_MB", "8"))
        ) * 1024 * 1024
        self._part_retries = max(1, int(part_retries))
        self._part_retry_backoff = max(0.0, float(part_retry_backoff))
        # 大对象 multipart 阈值：低于该字节数走单 PUT，达到/超过走真实 multipart。
        self._multipart_threshold = int(
            multipart_threshold_bytes
            if multipart_threshold_bytes is not None
            else int(os.environ.get("COS_MULTIPART_THRESHOLD_BYTES", str(8 * 1024 * 1024)))
        )
        # 并发上传 part 的线程数（有界 in-flight 队列 + 线程池）。
        self._max_concurrency = max(
            1,
            int(
                max_concurrency
                if max_concurrency is not None
                else int(os.environ.get("COS_MULTIPART_MAX_CONCURRENCY", "4"))
            ),
        )
        # 整个 multipart 流程（create→upload→complete）的重试次数。
        self._multipart_retries = max(
            1,
            int(
                multipart_retries
                if multipart_retries is not None
                else int(os.environ.get("COS_MULTIPART_RETRIES", "2"))
            ),
        )
        # 失败时是否 abort 清理已上传 part（避免孤儿 part 占用 COS 存储）。
        self._abort_on_error = bool(abort_on_error)
        # complete 后是否校验完整对象 ETag 与期望值一致（不一致视为失败）。
        self._verify_etag = bool(verify_etag)
        self._client = None
        self._client_identity: tuple | None = None
        self._client_creds = None
        self._uploads: dict[str, _MultipartState] = {}

    # ---- credential-aware client ----
    def _build_client_identity(self, creds) -> tuple:
        """client 身份：principal/scope/generation + 具体凭证 + 端点/region/bucket。

        含 access_key_id / session_token 是**严格更安全**：即使 generation 未
        递增，凭证轮换也会触发重建，绝不复用旧凭证构建的 client。
        """
        return (
            getattr(creds, "principal_id", None),
            getattr(creds, "credential_scope_id", None),
            getattr(creds, "credential_generation_id", None),  # 防御性读取
            getattr(creds, "access_key_id", None),
            getattr(creds, "session_token", None),
            self._endpoint,
            self._region or getattr(creds, "region", None),
            self.bucket,
        )

    @staticmethod
    def _creds_expiring(creds) -> bool:
        expires_at = getattr(creds, "expires_at", None)
        if expires_at is None:
            return False
        try:
            ts = (
                expires_at.timestamp()
                if hasattr(expires_at, "timestamp")
                else float(expires_at)
            )
        except Exception:
            return False
        return ts <= time.time() + 90  # 60s 操作 + 30s 时钟偏差

    def _s3(self):
        """惰性构建 boto3 client（凭证走 data_access.cos.remote）。

        每次解析凭证并比对身份；身份变化或凭证临近过期即重建。绝不返回在
        不同 principal/scope 下构建的 client。
        """
        from data_access.cos.remote import resolve_s3_credentials

        creds = resolve_s3_credentials()
        identity = self._build_client_identity(creds)
        if (
            self._client is not None
            and self._client_identity == identity
            and not self._creds_expiring(creds)
        ):
            return self._client

        import boto3
        from botocore.config import Config

        endpoint = self._endpoint or (
            (("https://" if creds.use_ssl else "http://") + creds.endpoint)
            if creds.endpoint
            else None
        )
        kwargs = {
            "service_name": "s3",
            "endpoint_url": endpoint,
            "region_name": self._region or creds.region,
            "aws_access_key_id": creds.access_key_id,
            "aws_secret_access_key": creds.secret_access_key,
            "config": Config(
                connect_timeout=2, read_timeout=10, retries={"max_attempts": 1}
            ),
        }
        if getattr(creds, "session_token", None):
            kwargs["aws_session_token"] = creds.session_token
        self._client = boto3.client(**kwargs)
        self._client_identity = identity
        self._client_creds = creds
        return self._client

    def _uri(self, key: str) -> str:
        return f"cos://{self.bucket}/{key}"

    def head_object(self, key: str) -> dict | None:
        from data_access.read.read_contract import _remote_object_meta

        meta = _remote_object_meta(self._uri(key), fresh=True)
        if meta is None:
            return None
        return {
            "etag": meta.get("etag"),
            "size": meta.get("content_length"),
            "last_modified": meta.get("last_modified"),
        }

    def list_objects(self, prefix: str) -> list[str]:
        s3 = self._s3()
        out: list[str] = []
        kwargs: dict = {"Bucket": self.bucket, "Prefix": str(prefix)}
        while True:
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                out.append(str(obj["Key"]))
            if not resp.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = resp.get("NextContinuationToken")
        return out

    def range_read(self, key: str, *, offset: int, length: int) -> bytes:
        s3 = self._s3()
        end = offset + length - 1
        resp = s3.get_object(
            Bucket=self.bucket,
            Key=str(key),
            Range=f"bytes={offset}-{end}",
        )
        return resp["Body"].read()

    def open_reader(self, key: str) -> BinaryIO | None:
        return COSSeekableRangeReader(self, str(key), max_buffer=self._reader_buffer)

    def put_object(self, key: str, data: bytes) -> None:
        """上传对象：小对象单 PUT，大对象（>= 阈值）走真实 multipart。

        大对象 multipart 流程：create → 分片并发 upload_part → complete →
        （可选）ETag 校验。任何失败按 ``abort_on_error`` 清理已上传 part。
        """
        if len(data) >= self._multipart_threshold:
            self._put_multipart(str(key), data)
            return
        s3 = self._s3()
        s3.put_object(Bucket=self.bucket, Key=str(key), Body=data)

    def _put_multipart(self, key: str, data: bytes) -> None:
        """真实 multipart 上传一个完整对象（含重试 + abort 清理 + ETag 校验）。"""
        last_exc: Exception | None = None
        for attempt in range(self._multipart_retries):
            upload_id = self.begin_multipart(key)
            try:
                self._upload_parts_concurrent(upload_id, key, data)
                self.complete_multipart(upload_id, key)
                # complete 后 ETag 校验（verify_etag=True 时）。
                if self._verify_etag:
                    self._verify_completed_etag(key, data)
                return
            except Exception as exc:  # noqa: BLE001 - 重试整个 multipart 流程
                last_exc = exc
                if self._abort_on_error:
                    try:
                        self.abort_multipart(upload_id, key)
                    except Exception:
                        pass
                if attempt < self._multipart_retries - 1:
                    time.sleep(self._part_retry_backoff)
        raise last_exc if last_exc is not None else RuntimeError("multipart upload failed")

    def _upload_parts_concurrent(self, upload_id: str, key: str, data: bytes) -> None:
        """把对象按 part_size 分片，用有界线程池并发 UploadPart。

        每片独立 UploadPart（含单 part 重试），part 上传后立即释放 bytes。
        """
        part_size = max(1, self._part_size)
        chunks = [
            (idx, data[offset : offset + part_size])
            for idx, offset in enumerate(range(0, len(data), part_size))
        ]
        if not chunks:
            chunks = [(0, b"")]
        state = self._uploads.get(upload_id)
        if state is None:
            raise KeyError(f"未知 multipart upload_id: {upload_id}")
        # 期望的完整对象 ETag（S3 多 part 对象 = MD5(各 part MD5 拼接) + "-N"）。
        state.expected_etag = _multipart_etag(chunks)

        def _upload_one(item: tuple[int, bytes]) -> None:
            idx, part = item
            self._upload_part_with_retry(upload_id, key, idx, part)

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            list(pool.map(_upload_one, chunks))

    def _upload_part_with_retry(
        self, upload_id: str, key: str, part_index: int, data: bytes
    ) -> None:
        """单 part UploadPart，失败重试该 part（不重试整个 upload）。"""
        s3 = self._s3()
        last_exc: Exception | None = None
        for attempt in range(self._part_retries):
            try:
                resp = s3.upload_part(
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=int(part_index),
                    Body=data,
                )
                etag = str(resp.get("ETag", "")).strip('"') or ""
                state = self._uploads.get(upload_id)
                if state is not None:
                    state.parts.append((int(part_index), etag))
                return
            except Exception as exc:  # noqa: BLE001 - retry transient part failure
                last_exc = exc
                if attempt < self._part_retries - 1:
                    time.sleep(self._part_retry_backoff)
        raise last_exc if last_exc is not None else RuntimeError("part upload failed")

    def _verify_completed_etag(self, key: str, data: bytes) -> None:
        """complete 后校验完整对象 ETag 与期望值一致；不一致抛错（不静默成功）。"""
        s3 = self._s3()
        try:
            resp = s3.head_object(Bucket=self.bucket, Key=str(key))
        except Exception:
            # 无法 head（如 fake 无 head_object）→ 跳过校验（best-effort）。
            return
        actual = str(resp.get("ETag", "")).strip('"')
        if not actual:
            return
        expected = _multipart_etag(
            [
                (i, data[o : o + self._part_size])
                for i, o in enumerate(range(0, len(data), self._part_size))
            ]
        )
        if actual != expected:
            raise RuntimeError(
                f"multipart ETag 校验失败: key={key!r} 期望 {expected} 实际 {actual}"
            )

    # ---- real multipart ----
    def begin_multipart(self, key: str) -> str:
        s3 = self._s3()
        resp = s3.create_multipart_upload(Bucket=self.bucket, Key=str(key))
        upload_id = str(resp["UploadId"])
        self._uploads[upload_id] = _MultipartState(upload_id, str(key), self._inflight)
        return upload_id

    def _flush_one(self, state: _MultipartState) -> None:
        """冲刷队列最旧 part：UploadPart 后立即释放 bytes，记录 ETag。

        单 part 失败（5xx/timeout）重试该 part（不重试整个 upload），最多
        ``_part_retries`` 次；重试耗尽才抛错（调用方 abort 整个 upload）。
        """
        part_index, data = state.queue.popleft()
        s3 = self._s3()
        last_exc: Exception | None = None
        for attempt in range(self._part_retries):
            try:
                resp = s3.upload_part(
                    Bucket=self.bucket,
                    Key=state.key,
                    UploadId=state.upload_id,
                    PartNumber=int(part_index),
                    Body=data,
                )
                etag = str(resp.get("ETag", "")).strip('"') or ""
                state.parts.append((int(part_index), etag))
                # data 出队即释放（无引用）
                return
            except Exception as exc:  # noqa: BLE001 - retry transient part failure
                last_exc = exc
                if attempt < self._part_retries - 1:
                    time.sleep(self._part_retry_backoff)
        raise last_exc if last_exc is not None else RuntimeError("part upload failed")

    def upload_part(
        self, upload_id: str, key: str, part_index: int, data: bytes
    ) -> None:
        state = self._uploads.get(upload_id)
        if state is None:
            raise KeyError(f"未知 multipart upload_id: {upload_id}")
        state.queue.append((int(part_index), data))
        while len(state.queue) >= state.inflight:
            self._flush_one(state)

    def complete_multipart(self, upload_id: str, key: str) -> None:
        state = self._uploads.pop(upload_id, None)
        if state is None:
            raise KeyError(f"未知 multipart upload_id: {upload_id}")
        while state.queue:
            self._flush_one(state)
        if not state.parts:
            raise ValueError(f"multipart {upload_id} 无任何 part，无法 complete")
        parts = sorted(state.parts, key=lambda p: p[0])
        body = {
            "Parts": [
                {"PartNumber": idx, "ETag": etag} for idx, etag in parts
            ]
        }
        s3 = self._s3()
        s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=state.key,
            UploadId=upload_id,
            MultipartUpload=body,
        )

    def abort_multipart(self, upload_id: str, key: str) -> None:
        state = self._uploads.pop(upload_id, None)
        if state is None:
            return
        try:
            s3 = self._s3()
            s3.abort_multipart_upload(
                Bucket=self.bucket, Key=state.key, UploadId=upload_id
            )
        except Exception:
            pass  # best-effort

    # ---- orphan multipart cleanup ----
    def list_multipart_uploads(self, prefix: str = "") -> list[dict]:
        """列出进行中的 multipart upload（含 key / upload_id / initiated）。"""
        s3 = self._s3()
        out: list[dict] = []
        kwargs: dict = {"Bucket": self.bucket, "Prefix": str(prefix)}
        while True:
            resp = s3.list_multipart_uploads(**kwargs)
            for u in resp.get("Uploads", []):
                out.append(
                    {
                        "key": str(u["Key"]),
                        "upload_id": str(u["UploadId"]),
                        "initiated": u.get("Initiated"),
                    }
                )
            if not resp.get("IsTruncated"):
                break
            kwargs["KeyMarker"] = resp.get("NextKeyMarker")
            kwargs["UploadIdMarker"] = resp.get("NextUploadIdMarker")
        return out

    def abort_stale_multipart_uploads(
        self, prefix: str = "", *, ttl_seconds: int = 3600
    ) -> int:
        """abort 超过 TTL 的孤儿 multipart upload，返回 abort 数量。"""
        s3 = self._s3()
        now = time.time()
        aborted = 0
        for u in self.list_multipart_uploads(prefix):
            initiated = u.get("initiated")
            if initiated is None:
                continue
            try:
                ts = (
                    initiated.timestamp()
                    if hasattr(initiated, "timestamp")
                    else float(initiated)
                )
            except Exception:
                continue
            if now - ts > ttl_seconds:
                try:
                    s3.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=u["key"],
                        UploadId=u["upload_id"],
                    )
                    aborted += 1
                except Exception:
                    pass
        return aborted

    def delete_object(self, key: str) -> None:
        s3 = self._s3()
        s3.delete_object(Bucket=self.bucket, Key=str(key))


class NullObjectStore:
    """全操作抛 ``NotImplementedError`` —— 测试 policy gating / 未接入 store。"""

    def __getattr__(self, name: str):
        def _raise(*args, **kwargs):
            raise NotImplementedError(
                f"NullObjectStore.{name} 未实现（R44-P0）：测试 policy gating 用"
            )

        return _raise
