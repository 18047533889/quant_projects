"""
data_access.write.object_store_generation_publisher —— 对象存储原生的不可变代次发布。

COS 没有原子 rename。正确的生产发布 = 不可变代次（immutable generation）：

    1. 数据对象上传到 ``<prefix>/<generation_id>/...``（generation_id 由本模块生成，
       或调用方提供 generation key）。
    2. 每个对象上传后 head/verify 其 size + sha256。
    3. 写不可变 ``GenerationManifest``（JSON）：generation_id、layout_version、
       有序对象 key + size + sha256 + 行/列/日期区间/因子清单等元数据、created_at、
       manifest 自身内容哈希。
    4. 晋升前校验代次 COMPLETE（所有列出的对象存在且 checksum 匹配）。
    5. **最后**更新 ``CURRENT`` / active-generation 指针（``<prefix>/CURRENT.json``）。
       读者永远只看到旧的完整代次或新的完整代次——绝看不到部分代次。
    6. 步骤 1-4 任何失败都让 CURRENT 不变。提供 abort/cleanup：只删除「非 CURRENT
       且不被不可变 manifest 引用」的对象，且只在显式 GC 调用时，不自动删。

本模块是对象存储发布的**唯一权威**。FactorEngine writer（后续 wave）将消费这套
精确 API，签名必须清晰稳定。

依赖：接受一个 data_access ObjectStore 实例（见 data_access/read/object_store.py 的
``ObjectStore`` Protocol：head_object / list_objects / range_read / open_reader /
put_object / begin_multipart / upload_part / complete_multipart / abort_multipart /
delete_object）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, BinaryIO, Callable, Mapping, Sequence

from data_access.core.exceptions import DataError, ValidationError
from data_access.read.object_store import ObjectStore

logger = logging.getLogger("data_access.object_store_generation_publisher")

# CURRENT 指针对象 key（相对 prefix）。
_CURRENT_KEY = "CURRENT.json"
# 代次 manifest 对象 key（相对 generation 前缀）。
_MANIFEST_KEY = "_manifest.json"
# 代次元数据对象 key（相对 generation 前缀）。
_GENERATION_META_KEY = "_generation.json"

_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class GenerationObject:
    """代次内单个数据对象的清单条目。"""

    key: str  # 相对 generation 前缀的 key
    size: int
    sha256: str
    # 可选业务元数据：行数 / 列 / 日期区间 / 因子清单等，由调用方提供。
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size": self.size,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GenerationObject":
        return cls(
            key=str(d["key"]),
            size=int(d["size"]),
            sha256=str(d["sha256"]),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass(frozen=True)
class GenerationManifest:
    """不可变代次清单。"""

    generation_id: str
    layout_version: int
    prefix: str
    objects: tuple[GenerationObject, ...]
    metadata: dict[str, Any]
    created_at: str
    content_hash: str  # manifest 自身内容哈希（不含本字段）

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "layout_version": self.layout_version,
            "prefix": self.prefix,
            "objects": [o.to_dict() for o in self.objects],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GenerationManifest":
        return cls(
            generation_id=str(d["generation_id"]),
            layout_version=int(d["layout_version"]),
            prefix=str(d["prefix"]),
            objects=tuple(
                GenerationObject.from_dict(o) for o in d.get("objects") or []
            ),
            metadata=dict(d.get("metadata") or {}),
            created_at=str(d["created_at"]),
            content_hash=str(d.get("content_hash") or ""),
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_content_hash(manifest_dict: dict[str, Any]) -> str:
    """对 manifest 的规范 JSON 求内容哈希（排除 content_hash 字段自身）。"""
    body = {k: v for k, v in manifest_dict.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _PendingGeneration:
    """进行中的代次（尚未晋升）。"""

    __slots__ = (
        "generation_id",
        "prefix",
        "layout_version",
        "metadata",
        "objects",
        "created_at",
    )

    def __init__(
        self,
        generation_id: str,
        prefix: str,
        layout_version: int,
        metadata: dict[str, Any],
    ) -> None:
        self.generation_id = generation_id
        self.prefix = prefix
        self.layout_version = layout_version
        self.metadata = metadata
        self.objects: list[GenerationObject] = []
        self.created_at = _now_iso()


class ObjectStoreGenerationPublisher:
    """对象存储不可变代次发布器（COS 无原子 rename 的正确生产发布）。

    公共 API 契约（FactorEngine writer 消费的稳定接口）：
        begin_generation(prefix, layout_version, metadata) -> generation_id
        add_object(generation_id, key, data)
        finish_generation(generation_id) -> GenerationManifest
        abort_generation(generation_id)
        resolve_current(prefix) -> generation_id | None
        list_objects_for_generation(generation_id) -> list[str]
        gc(prefix) -> int   # 显式垃圾回收未引用代次

    线程/进程安全：本实例维护进行中代次的内存状态，单实例单线程使用；跨进程
    协调由调用方（FactorEngine writer）负责（见 atomic_generation.DistributedWriteCoordinator）。
    """

    def __init__(
        self,
        store: ObjectStore,
        *,
        bucket: str | None = None,
        multipart_threshold: int = 8 * 1024 * 1024,
    ) -> None:
        self.store = store
        self.bucket = bucket
        self.multipart_threshold = multipart_threshold
        self._pending: dict[str, _PendingGeneration] = {}
        # generation_id -> prefix（begin 时记录，finish 后保留，供 list/gc 定位 manifest）。
        self._gen_prefix: dict[str, str] = {}

    # ---- 生命周期 ----------------------------------------------------------

    def begin_generation(
        self,
        prefix: str,
        layout_version: int = _LAYOUT_VERSION,
        metadata: Mapping[str, Any] | None = None,
        *,
        generation_id: str | None = None,
    ) -> str:
        """开始一个新代次，返回 generation_id。

        prefix 是代次集合的根（如 ``factors/{factor_id}``）。数据对象将上传到
        ``<prefix>/<generation_id>/<key>``。
        """
        prefix = _normalize_prefix(prefix)
        gid = generation_id or uuid.uuid4().hex
        if gid in self._pending:
            raise ValidationError(f"generation {gid} 已在进行中")
        self._pending[gid] = _PendingGeneration(
            gid, prefix, int(layout_version), dict(metadata or {})
        )
        self._gen_prefix[gid] = prefix
        return gid

    def add_object(
        self,
        generation_id: str,
        key: str,
        data: bytes | BinaryIO,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """向代次添加一个数据对象并上传。

        上传后立即 head/verify size + sha256。key 是相对 generation 前缀的 POSIX
        相对路径（拒绝 ``..`` 逃逸）。
        """
        pending = self._require_pending(generation_id)
        rel_key = _safe_object_key(key)
        full_key = f"{pending.prefix}/{pending.generation_id}/{rel_key}"

        blob = _coerce_bytes(data)
        self._upload(full_key, blob)

        # 上传后验证 size + sha256。
        head = self.store.head_object(full_key)
        if head is None:
            raise DataError(
                f"generation {generation_id} 对象 {full_key} 上传后 head 缺失"
            )
        actual_size = head.get("size")
        if actual_size is not None and int(actual_size) != len(blob):
            raise DataError(
                f"generation {generation_id} 对象 {full_key} size 不匹配："
                f"期望 {len(blob)} 实际 {actual_size}"
            )
        pending.objects.append(
            GenerationObject(
                key=rel_key,
                size=len(blob),
                sha256=_sha256_bytes(blob),
                metadata=dict(metadata or {}),
            )
        )

    def finish_generation(self, generation_id: str) -> GenerationManifest:
        """写不可变 manifest、校验代次 COMPLETE、最后翻转 CURRENT 指针。

        任何失败（写 manifest / 校验 / 翻转）都让 CURRENT 保持原值。
        """
        pending = self._require_pending(generation_id)
        if not pending.objects:
            raise ValidationError(
                f"generation {generation_id} 无任何对象，拒绝晋升空代次"
            )

        # 1) 写不可变 manifest（含自身内容哈希）。
        manifest = self._write_manifest(pending)

        # 2) 晋升前校验代次 COMPLETE：所有列出的对象存在且 checksum 匹配。
        self._validate_generation_complete(pending, manifest)

        # 3) 最后翻转 CURRENT 指针（读者只看到旧完整代次或新完整代次）。
        self._flip_current(pending.prefix, generation_id)

        self._pending.pop(generation_id, None)
        logger.info(
            "generation %s promoted under prefix %s (%d objects)",
            generation_id, pending.prefix, len(pending.objects),
        )
        return manifest

    def abort_generation(self, generation_id: str) -> None:
        """中止一个未晋升的代次：删除其已上传对象并丢弃内存状态。

        只删除该代次自己的对象；CURRENT 指针不受影响。未引用对象的最终清理由
        显式 ``gc`` 负责（本方法不自动 GC 其它代次）。
        """
        pending = self._pending.pop(generation_id, None)
        if pending is None:
            return
        for obj in pending.objects:
            full_key = f"{pending.prefix}/{pending.generation_id}/{obj.key}"
            try:
                self.store.delete_object(full_key)
            except Exception as exc:  # best-effort 清理
                logger.warning("abort 清理对象 %s 失败：%s", full_key, exc)
        logger.info("generation %s aborted", generation_id)

    # ---- 读取 --------------------------------------------------------------

    def resolve_current(self, prefix: str) -> str | None:
        """解析当前 active generation_id；无 CURRENT 指针返回 None。"""
        prefix = _normalize_prefix(prefix)
        current_key = f"{prefix}/{_CURRENT_KEY}"
        head = self.store.head_object(current_key)
        if head is None:
            return None
        reader = self.store.open_reader(current_key)
        if reader is None:
            # 无流式句柄（如 COS）→ 回退 range_read 全量读。
            size = head.get("size")
            if size is None:
                return None
            blob = self.store.range_read(current_key, offset=0, length=int(size))
        else:
            try:
                blob = reader.read()
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
        try:
            payload = json.loads(blob.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        gid = payload.get("generation_id")
        return str(gid) if gid else None

    def list_objects_for_generation(self, generation_id: str) -> list[str]:
        """列出某代次下的对象 key（相对 generation 前缀）。"""
        # 从 manifest 读取（权威清单），而非扫描目录。
        manifest = self._read_manifest_for_generation(generation_id)
        if manifest is None:
            return []
        return [o.key for o in manifest.objects]

    def gc(self, prefix: str) -> int:
        """显式垃圾回收：删除「非 CURRENT 且不被不可变 manifest 引用」的对象。

        只删除 CURRENT 指针未指向、且其代次目录下没有不可变 manifest 的对象。
        返回删除的对象数。绝不自动调用。
        """
        prefix = _normalize_prefix(prefix)
        current = self.resolve_current(prefix)
        removed = 0
        # 扫描 prefix 下所有对象，按代次分组。
        all_keys = self.store.list_objects(prefix)
        by_gen: dict[str, list[str]] = {}
        for key in all_keys:
            rel = key[len(prefix) + 1:] if key.startswith(prefix + "/") else key
            if rel == _CURRENT_KEY:
                continue
            parts = rel.split("/", 1)
            if len(parts) < 2:
                continue  # 顶层非代次对象，不动
            by_gen.setdefault(parts[0], []).append(key)
        for gid, keys in by_gen.items():
            if gid == current:
                continue
            # 有不可变 manifest 的代次是已发布历史，保留（回滚/审计用）。
            if self._read_manifest_for_generation(gid) is not None:
                continue
            for key in keys:
                try:
                    self.store.delete_object(key)
                    removed += 1
                except Exception as exc:
                    logger.warning("gc 删除对象 %s 失败：%s", key, exc)
        return removed

    # ---- 内部实现 ----------------------------------------------------------

    def _upload(self, full_key: str, blob: bytes) -> None:
        """上传单个对象；大对象走 multipart。"""
        if len(blob) >= self.multipart_threshold:
            upload_id = self.store.begin_multipart(full_key)
            try:
                # 分片上传（每片 8MB）。
                chunk = self.multipart_threshold
                idx = 0
                for offset in range(0, len(blob), chunk):
                    self.store.upload_part(
                        upload_id, full_key, idx, blob[offset : offset + chunk]
                    )
                    idx += 1
                self.store.complete_multipart(upload_id, full_key)
            except Exception:
                try:
                    self.store.abort_multipart(upload_id, full_key)
                except Exception:
                    pass
                raise
        else:
            self.store.put_object(full_key, blob)

    def _write_manifest(self, pending: _PendingGeneration) -> GenerationManifest:
        manifest_dict = {
            "generation_id": pending.generation_id,
            "layout_version": pending.layout_version,
            "prefix": pending.prefix,
            "objects": [o.to_dict() for o in pending.objects],
            "metadata": pending.metadata,
            "created_at": pending.created_at,
            "content_hash": "",  # 占位，下面填
        }
        manifest_dict["content_hash"] = _manifest_content_hash(manifest_dict)
        manifest = GenerationManifest.from_dict(manifest_dict)
        manifest_key = (
            f"{pending.prefix}/{pending.generation_id}/{_MANIFEST_KEY}"
        )
        self.store.put_object(
            manifest_key, json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8")
        )
        return manifest

    def _validate_generation_complete(
        self, pending: _PendingGeneration, manifest: GenerationManifest
    ) -> None:
        """晋升前校验：所有列出的对象存在且 sha256 匹配。"""
        for obj in manifest.objects:
            full_key = f"{pending.prefix}/{pending.generation_id}/{obj.key}"
            head = self.store.head_object(full_key)
            if head is None:
                raise DataError(
                    f"generation {pending.generation_id} 校验失败：对象 {full_key} 缺失"
                )
            actual = self._read_object_bytes(full_key, head)
            if _sha256_bytes(actual) != obj.sha256:
                raise DataError(
                    f"generation {pending.generation_id} 校验失败：对象 {full_key} "
                    f"sha256 不匹配（期望 {obj.sha256}）"
                )

    def _flip_current(self, prefix: str, generation_id: str) -> None:
        """最后翻转 CURRENT 指针（单对象原子写）。"""
        current_key = f"{prefix}/{_CURRENT_KEY}"
        payload = json.dumps(
            {"generation_id": generation_id, "updated_at": _now_iso()},
            sort_keys=True,
        ).encode("utf-8")
        self.store.put_object(current_key, payload)

    def _read_manifest_for_generation(
        self, generation_id: str
    ) -> GenerationManifest | None:
        """读取某代次的不可变 manifest；不存在返回 None。"""
        prefix = self._gen_prefix.get(generation_id)
        if prefix is None:
            return None
        manifest_key = f"{prefix}/{generation_id}/{_MANIFEST_KEY}"
        head = self.store.head_object(manifest_key)
        if head is None:
            return None
        blob = self._read_object_bytes(manifest_key, head)
        try:
            return GenerationManifest.from_dict(json.loads(blob.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError):
            return None

    def _read_object_bytes(self, full_key: str, head: dict) -> bytes:
        reader = self.store.open_reader(full_key)
        if reader is not None:
            try:
                return reader.read()
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
        size = head.get("size")
        if size is None:
            raise DataError(f"对象 {full_key} 无 size 且无流式句柄，无法校验")
        return self.store.range_read(full_key, offset=0, length=int(size))

    def _require_pending(self, generation_id: str) -> _PendingGeneration:
        pending = self._pending.get(generation_id)
        if pending is None:
            raise ValidationError(
                f"generation {generation_id} 不存在或已结束（需先 begin_generation）"
            )
        return pending


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix).strip("/")
    if not prefix:
        raise ValidationError("prefix 不能为空")
    return prefix


def _safe_object_key(key: str) -> str:
    key = str(key)
    if not key or key.startswith("/") or "\\" in key:
        raise ValueError(f"非法对象 key: {key!r}")
    norm = key
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"对象 key 越界（.. 逃逸）: {key!r}")
    return "/".join(parts)


def _coerce_bytes(data: bytes | BinaryIO) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.read()


__all__ = [
    "ObjectStoreGenerationPublisher",
    "GenerationManifest",
    "GenerationObject",
    "_CURRENT_KEY",
    "_MANIFEST_KEY",
]
