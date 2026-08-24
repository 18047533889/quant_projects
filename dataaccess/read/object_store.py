# -*- coding: utf-8 -*-
"""R44-P0: ObjectStore 抽象 —— 节点级增量 FactorEngine 的远程数据平面。

零本地磁盘目标（STRICT_REMOTE）下，增量 planner 通过 ``ObjectStore`` 读写
parquet/arrow 对象，而不是直接落本地磁盘。本模块提供：

- ``ObjectStore``  Protocol：head / list / range / open_reader / put /
  multipart / delete 的最小对象存储接口。
- ``LocalObjectStore``：基于 ``root`` 目录的字节对象存储（写走 temp +
  ``os.replace`` 原子替换，multipart 用 staging 文件模拟），测试与本地
  增量跑批用。
- ``COSObjectStore``：适配 ``data_access.cos.remote`` 的薄适配器（只保证
  head/range/get/put 干净可用；multipart 是 no-op 委托 put_object，见类文档）。
- ``NullObjectStore``：全部抛 ``NotImplementedError``，用于测试 policy gating。

R44-P0 契约：
    1. ``put_object`` / ``complete_multipart`` 必须原子（读者永远看不到半成品）。
    2. key 是 POSIX 相对路径；拒绝 ``..`` / 绝对路径逃逸。
    3. 远程适配只做 best-effort —— 网络/凭证失败绝不让扫描路径崩溃。
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

__all__ = [
    "ObjectStore",
    "LocalObjectStore",
    "COSObjectStore",
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


class COSObjectStore:
    """COS 对象存储薄适配器（R44-P0，best-effort）。

    复用 :mod:`data_access.cos.remote` 的凭证/端点解析与 HEAD 元数据缓存
    （``_remote_object_meta``）；head/range/get/put 通过 boto3（S3 兼容 API）
    干净可用。**multipart 是 no-op 委托 ``put_object``**——COS 模块未暴露
    完整 multipart API，对中小对象单 PUT 即够；增量 planner 的 COS 写入路径
    由调用方选择是否分批。

    ``open_reader`` 返回 None（无流式句柄）→ 调用方回退 ``range_read``。
    任何操作失败抛错，由扫描器 catch 后降级全量读。
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint: str | None = None,
        region: str | None = None,
    ):
        self.bucket = str(bucket)
        self._endpoint = endpoint
        self._region = region
        self._client = None
        self._parts: dict[str, list[bytes]] = {}

    def _s3(self):
        """惰性构建 boto3 client（凭证走 data_access.cos.remote）。"""
        if self._client is not None:
            return self._client
        import boto3
        from botocore.config import Config

        from data_access.cos.remote import resolve_s3_credentials

        creds = resolve_s3_credentials()
        endpoint = self._endpoint or (
            ("https://" if creds.use_ssl else "http://") + creds.endpoint
            if creds.endpoint
            else None
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=self._region or creds.region,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            config=Config(connect_timeout=2, read_timeout=10, retries={"max_attempts": 1}),
        )
        return self._client

    def _uri(self, key: str) -> str:
        return f"cos://{self.bucket}/{key}"

    def head_object(self, key: str) -> dict | None:
        from data_access.cos.remote import _remote_object_meta

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
        return None  # 无流式句柄 → 扫描器回退 range_read

    def put_object(self, key: str, data: bytes) -> None:
        s3 = self._s3()
        s3.put_object(Bucket=self.bucket, Key=str(key), Body=data)

    def begin_multipart(self, key: str) -> str:
        return uuid.uuid4().hex

    def upload_part(
        self, upload_id: str, key: str, part_index: int, data: bytes
    ) -> None:
        # 薄适配：暂存内存，complete 时单 PUT 拼接（COS 模块无干净 multipart API）。
        self._parts.setdefault(upload_id, [])
        while len(self._parts[upload_id]) <= int(part_index):
            self._parts[upload_id].append(b"")
        self._parts[upload_id][int(part_index)] = data

    def complete_multipart(self, upload_id: str, key: str) -> None:
        blob = b"".join(self._parts.pop(upload_id, []))
        self.put_object(key, blob)

    def abort_multipart(self, upload_id: str, key: str) -> None:
        self._parts.pop(upload_id, None)

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
