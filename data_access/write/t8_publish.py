"""
data_access.write.t8_publish —— T8 多 artifact 原子发布封装（问题二，DATA_ACCESS_UPSTREAM_FIX_PLAN）。

T8 因子的 public_meta / secret_meta / value / evaluation / catalog 五类数据必须作为
**一个逻辑 batch** 原子发布到远端 factor_pool/library_v1 布局，禁止五类数据跨
generation。本模块是 :class:`ObjectStoreGenerationPublisher` 之上的**语义封装层**：

- 一次 ``begin_generation`` → 逐 artifact ``add_object`` → ``finish_generation``，
  任何一步失败都 ``abort_generation``（CURRENT 指针不被部分代次污染）；
- T8 语义（kind + batch_id）放进 **现有 manifest 结构**（``metadata`` 与每个
  ``GenerationObject.metadata["kind"]``），**不改** ``GenerationManifest`` schema；
- retry 幂等：同一 ``batch_id`` 重试复用同一 generation key（``generation_id``
  参数），CURRENT 不被重放覆盖（fencing epoch 由 publisher 保证）；
- 读端 helper：读端只能沿 ``CURRENT.json → generation manifest → manifest 列出的
  精确对象`` 读取，**禁止**对整个 factor_pool 做 wildcard glob —— 本模块读路径
  只用 ``head_object / open_reader / range_read``，绝不 ``list_objects``。

对象 key 布局（相对 generation 前缀）：

.. code-block:: text

    meta/public_meta/<factor_id>.json
    meta/secret_meta/<factor_id>.json
    data/value/<factor_id>.parquet
    data/evaluation/<factor_id>.json
    meta/catalog_manifest/<factor_id>.parquet

CURRENT 指针与 generation manifest 的写盘（``_manifest.json``、fencing_epoch、
writer_id、sha256 校验）全部由 ``ObjectStoreGenerationPublisher`` 负责，本模块不
重复实现，只编排多 artifact 语义。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from data_access.core.exceptions import DataError, ValidationError
from data_access.read.object_store import ObjectStore
from data_access.write.object_store_generation_publisher import (
    GenerationManifest,
    ObjectStoreGenerationPublisher,
    StaleWriterError,
)

logger = logging.getLogger("data_access.write.t8_publish")

# T8 artifact 类别（五类一个 batch）。
PUBLIC_META = "public_meta"
SECRET_META = "secret_meta"
VALUE = "value"
EVALUATION = "evaluation"
CATALOG = "catalog"
T8_KINDS: tuple[str, ...] = (PUBLIC_META, SECRET_META, VALUE, EVALUATION, CATALOG)

# kind -> 远端子目录（相对 generation 前缀）。
_KIND_DIRS: dict[str, str] = {
    PUBLIC_META: "meta/public_meta",
    SECRET_META: "meta/secret_meta",
    CATALOG: "meta/catalog_manifest",
    VALUE: "data/value",
    EVALUATION: "data/evaluation",
}
# kind -> 单文件扩展名（publish_t8_dataset 默认文件名）。
_KIND_EXT: dict[str, str] = {
    PUBLIC_META: "json",
    SECRET_META: "json",
    CATALOG: "parquet",
    VALUE: "parquet",
    EVALUATION: "json",
}
# publish_t8_dataset 从 staging 目录识别的文件名（<staging_root>/<factor_id>/<name>）。
STAGING_FILENAMES: dict[str, str] = {
    kind: f"{kind}.{_KIND_EXT[kind]}" for kind in T8_KINDS
}

_LAYOUT_VERSION = 1

# CURRENT 指针与代次 manifest 对象 key（与 publisher 常量一致）。
_CURRENT_KEY = "CURRENT.json"
_MANIFEST_KEY = "_manifest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- kind / key 校验 --------------------------------------------------------


def _require_valid_kind(kind: str) -> None:
    if kind not in T8_KINDS:
        raise ValidationError(
            f"非法 T8 artifact kind: {kind!r}，允许 {list(T8_KINDS)}"
        )


def _safe_factor_token(factor_id: str) -> str:
    """factor_id 只能作为单段 token 出现在对象 key（拒绝 ``/``/``..``/绝对路径逃逸）。"""
    fid = str(factor_id)
    if not fid or fid in (".", ".."):
        raise ValidationError(f"非法 factor_id: {factor_id!r}")
    if "/" in fid or "\\" in fid or fid.startswith("."):
        raise ValidationError(
            f"factor_id 不能含路径分隔符或点开头（防对象 key 逃逸）: {factor_id!r}"
        )
    return fid


def layout_rel_key(kind: str, factor_id: str) -> str:
    """按 T8 远端布局拼 artifact 的**相对 generation 前缀**对象 key。

    ``meta/public_meta/<factor_id>.json`` / ``data/value/<factor_id>.parquet`` 等。
    需要多文件 value 的调用方可直接传自定义 key 给 :func:`publish_t8_artifacts`。
    """
    _require_valid_kind(kind)
    fid = _safe_factor_token(factor_id)
    return f"{_KIND_DIRS[kind]}/{fid}.{_KIND_EXT[kind]}"


def _validate_rel_key(rel_key: str) -> str:
    rel_key = str(rel_key)
    if not rel_key or rel_key.startswith("/") or "\\" in rel_key:
        raise ValueError(f"非法对象 key: {rel_key!r}")
    parts = [p for p in rel_key.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"对象 key 越界（.. 逃逸）: {rel_key!r}")
    return "/".join(parts)


def _coerce_data(data: bytes | BinaryIO) -> bytes:
    """bytes 原样返回；BinaryIO 整读一次（流不可回放，只允许消费一次）。"""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    blob = data.read()
    if not isinstance(blob, bytes):
        blob = bytes(blob)
    return blob


def _normalize_artifacts(
    artifacts: Mapping[str, Any] | None,
    factor_id: str | None,
) -> list[tuple[str, str, bytes]]:
    """把 ``{kind: (rel_key, data_bytes|BinaryIO)}`` 或 ``{kind: data}`` 规范化。

    值缺省 rel_key 时按 ``layout_rel_key(kind, factor_id)`` 生成（需提供 factor_id）。
    返回 [(kind, rel_key, data_bytes), ...]，保序（metadata/调用顺序）。
    """
    artifacts = dict(artifacts or {})
    if not artifacts:
        raise ValidationError("T8 artifacts 不能为空：至少一个 artifact")
    out: list[tuple[str, str, bytes]] = []
    for raw_kind, spec in artifacts.items():
        kind = str(raw_kind)
        _require_valid_kind(kind)
        if isinstance(spec, tuple) and len(spec) == 2:
            key, data = spec
            rel_key = _validate_rel_key(str(key))
        else:
            if not factor_id:
                raise ValidationError(
                    f"artifact {kind} 未提供对象 key，且无 factor_id 可推导默认 key"
                )
            data = spec
            rel_key = layout_rel_key(kind, factor_id)
        out.append((kind, rel_key, _coerce_data(data)))
    return out


# ---- 发布 -------------------------------------------------------------------


def publish_t8_artifacts(
    publisher: ObjectStoreGenerationPublisher,
    *,
    prefix: str,
    factor_id: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
    batch_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
) -> GenerationManifest:
    """把五类 T8 artifact 作为**一个 batch 一个 generation** 原子发布。

    语义（对齐 ObjectStoreGenerationPublisher 的不可变代次协议）：
        1. ``begin_generation(prefix, metadata={batch_id, ...})``；
        2. 逐 artifact ``add_object``（每个对象的 ``GenerationObject.metadata["kind"]``
           记录其 T8 类别，manifest 序列化后读者可精确还原五类分布）；
        3. ``finish_generation`` → 写不可变 manifest → 校验全部对象 sha256 →
           **最后**翻转 CURRENT；
        4. 1-3 任何失败 ``abort_generation``（删除本代次已上传对象），CURRENT 不变。

    ``artifacts``：``{kind: (rel_key, data_bytes|BinaryIO)}`` 或 ``{kind: data}``
    （后者用 ``factor_id`` 推导默认 key，见 :func:`layout_rel_key`）。

    retry/idempotency：同一 ``batch_id`` 的重试调用应传**同一个** ``generation_id``
    （可用 :func:`publish_t8_artifacts_retry`，它自动保证失败重试用同一 generation
    key）。CURRENT 不被重放覆盖：重试期间若有别的 writer 已晋升新代次，
    ``finish_generation`` 的 fencing epoch 判定抛 :class:`StaleWriterError`。

    返回晋升后的 :class:`GenerationManifest`（只读，不可变）。
    """
    norm = _normalize_artifacts(artifacts, factor_id)
    if not norm:
        raise ValidationError("T8 artifacts 为空，拒绝发布空 batch")

    meta: dict[str, Any] = dict(metadata or {})
    if batch_id is not None:
        meta["batch_id"] = str(batch_id)
    # 记录这是一次 T8 多 artifact batch 发布（供审计/读端识别布局语义）。
    meta["publisher"] = "t8_publish"
    meta["layout_version"] = _LAYOUT_VERSION

    gid: str | None = None
    try:
        gid = publisher.begin_generation(
            prefix,
            layout_version=_LAYOUT_VERSION,
            metadata=meta,
            generation_id=generation_id,
        )
        for kind, rel_key, data in norm:
            publisher.add_object(
                gid, rel_key, data, metadata={"kind": kind, "batch_id": str(batch_id) if batch_id is not None else ""}
            )
        manifest = publisher.finish_generation(gid)
        logger.info(
            "T8 batch %s promoted: generation=%s prefix=%s artifacts=%d (batch_id=%s)",
            batch_id, manifest.generation_id, prefix, len(norm), batch_id,
        )
        return manifest
    except BaseException:
        if gid is not None:
            try:
                publisher.abort_generation(gid)
            except Exception as exc:  # noqa: BLE001 - best-effort 清理，不掩盖原错
                logger.warning("T8 abort_generation(%s) 清理失败：%s", gid, exc)
        raise


def publish_t8_artifacts_retry(
    publisher: ObjectStoreGenerationPublisher,
    *,
    prefix: str,
    artifacts: Mapping[str, Any] | None = None,
    factor_id: str | None = None,
    batch_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    retries: int = 3,
    retry_backoff: float | None = None,
    generation_id: str | None = None,
) -> GenerationManifest:
    """带有限重试的 T8 发布：**每次尝试用同一 generation key**（幂等）。

    - 同一 ``batch_id``（或显式 ``generation_id``）→ 同一 generation key：
      ``publish_t8_artifacts`` 失败时已 abort 清理，重试从头重传，最终对象集合一致；
    - :class:`StaleWriterError` 表示并发 fencing 拒绝（CURRENT 已被更新 writer
      晋升）——**不重试**，立即原样抛（fail-closed，禁止覆盖已晋升 CURRENT）；
    - 其余瞬时错误按 ``retries``（默认 3）重试，间隔 ``retry_backoff``（默认取
      环境 ``T8_RETRY_BACKOFF``，0.2s）。
    """
    gen_key = generation_id
    if gen_key is None:
        gen_key = str(batch_id) if batch_id is not None else uuid.uuid4().hex
    n = max(1, int(retries))
    backoff = (
        retry_backoff
        if retry_backoff is not None
        else float(os.environ.get("T8_RETRY_BACKOFF", "0.2"))
    )
    last_exc: Exception | None = None
    for attempt in range(n):
        try:
            return publish_t8_artifacts(
                publisher,
                prefix=prefix,
                factor_id=factor_id,
                artifacts=artifacts,
                batch_id=batch_id,
                metadata=metadata,
                generation_id=gen_key,
            )
        except StaleWriterError:
            raise
        except Exception as exc:  # noqa: BLE001 - 瞬时上传失败才重试
            last_exc = exc
            logger.warning(
                "T8 publish attempt %d/%d failed (batch_id=%s gen=%s): %s",
                attempt + 1, n, batch_id, gen_key, exc,
            )
            if attempt < n - 1:
                import time as _time

                _time.sleep(max(0.0, backoff))
    raise last_exc if last_exc is not None else RuntimeError("T8 publish retries exhausted")


# ---- staging 高层 convenience ----------------------------------------------


def _resolve_staging_root(staging_root: str | Path | None) -> Path:
    """解析 staging 根目录。

    显式注入优先；否则取环境 ``T8_STAGING_ROOT`` / ``ALPHAFLOW_STAGING_ROOT``。
    **无默认用户目录**——两者都未提供时抛 ``ValidationError``（禁止默认写不存在
    的其它用户目录，见 UPSTREAM_FIX_PLAN 问题二 dataset 配置段）。
    """
    if staging_root is not None:
        return Path(staging_root).expanduser().resolve()
    env = os.environ.get("T8_STAGING_ROOT") or os.environ.get(
        "ALPHAFLOW_STAGING_ROOT"
    )
    if env:
        return Path(env).expanduser().resolve()
    raise ValidationError(
        "staging_root 未注入：请显式传 staging_root，或设 T8_STAGING_ROOT / "
        "ALPHAFLOW_STAGING_ROOT（禁止默认写不存在的用户目录）"
    )


def publish_t8_dataset(
    publisher: ObjectStoreGenerationPublisher,
    *,
    prefix: str,
    factor_id: str,
    staging_root: str | Path | None = None,
    batch_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    generation_id: str | None = None,
    include: tuple[str, ...] | None = None,
) -> GenerationManifest:
    """高层封装：读本地 staging 目录 → 按 T8 布局上传 → 原子发布。

    staging 目录约定（本地写盘由上游/调用方完成，本函数只读）：

    .. code-block:: text

        <staging_root>/<factor_id>/
            public_meta.json      # -> meta/public_meta/<factor_id>.json
            secret_meta.json      # -> meta/secret_meta/<factor_id>.json
            value.parquet         # -> data/value/<factor_id>.parquet
            evaluation.json       # -> data/evaluation/<factor_id>.json
            catalog.parquet       # -> meta/catalog_manifest/<factor_id>.parquet

    ``include`` 指定本次发布的类别子集（默认全部五类）；存在即上传，**允许部分
    类别发布**（原子性只约束纳入 batch 的对象），但至少一类存在，否则抛
    ``ValidationError``。staging 目录路径必须显式注入（见 :func:`_resolve_staging_root`）。
    """
    root = _resolve_staging_root(staging_root)
    base = root / factor_id
    if not base.is_dir():
        raise ValidationError(
            f"staging 目录不存在: {base}（请先由上游写入本地 staging）"
        )
    include_kinds = tuple(include) if include is not None else T8_KINDS
    artifacts: dict[str, tuple[str, bytes]] = {}
    for kind in include_kinds:
        _require_valid_kind(kind)
        fp = base / STAGING_FILENAMES[kind]
        if not fp.is_file():
            logger.warning("staging 缺少 %s（跳过）: %s", kind, fp)
            continue
        artifacts[kind] = (layout_rel_key(kind, factor_id), fp.read_bytes())
    if not artifacts:
        raise ValidationError(
            f"staging 目录 {base} 无任何可发布的 T8 artifact（include={list(include_kinds)}）"
        )
    return publish_t8_artifacts(
        publisher,
        prefix=prefix,
        factor_id=factor_id,
        artifacts=artifacts,
        batch_id=batch_id,
        metadata=metadata,
        generation_id=generation_id,
    )


# ---- CURRENT / 校验 ----------------------------------------------------------


def current_pointer_payload(manifest: GenerationManifest) -> bytes:
    """构造 CURRENT 指针对象内容（与 publisher 写盘格式一致，可用于对拍/审计）。

    ``{generation_id, updated_at, fencing_epoch}`` sort_keys JSON。
    """
    payload = {
        "generation_id": manifest.generation_id,
        "updated_at": _now_iso(),
        "fencing_epoch": int(manifest.fencing_epoch or 0),
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def verify_generation_remote(
    publisher: ObjectStoreGenerationPublisher,
    manifest: GenerationManifest,
    *,
    head_fn: Callable[[GenerationManifest, Any], bytes] | None = None,
) -> None:
    """对 manifest 每个对象做 HEAD + size + sha256 校验。

    - ``HEAD`` 拿 size（缺失对象 → ``DataError``）；
    - sha256：默认顺序流式回读整对象计算（有界内存，1 MiB 分块）；传 ``head_fn``
      时用它返回对象字节做哈希（fake store / 已缓存场景）。
    任一不匹配抛 ``DataError``。只读校验，不改任何对象。
    """
    store: ObjectStore = publisher.store
    for obj in manifest.objects:
        full_key = f"{manifest.prefix}/{manifest.generation_id}/{obj.key}"
        head = store.head_object(full_key)
        if head is None:
            raise DataError(
                f"verify_generation_remote: 对象缺失 {full_key} "
                f"(generation={manifest.generation_id})"
            )
        actual_size = head.get("size")
        if actual_size is not None and int(actual_size) != int(obj.size):
            raise DataError(
                f"verify_generation_remote: 对象 {full_key} size 不匹配，"
                f"期望 {obj.size} 实际 {actual_size}"
            )
        if head_fn is not None:
            blob = head_fn(manifest, obj)
            if _sha256_bytes(blob) != obj.sha256:
                raise DataError(
                    f"verify_generation_remote: 对象 {full_key} sha256 不匹配 "
                    f"（期望 {obj.sha256}）"
                )
            continue
        # 顺序流式回读算 sha256（不整读进内存）。
        sha = hashlib.sha256()
        total = 0
        chunk = 1024 * 1024
        reader = store.open_reader(full_key)
        if reader is not None:
            try:
                while True:
                    part = reader.read(chunk)
                    if not part:
                        break
                    total += len(part)
                    sha.update(part)
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
        else:
            size = head.get("size")
            if size is None:
                raise DataError(f"对象 {full_key} 无 size 且无流式句柄，无法校验")
            blob = store.range_read(full_key, offset=0, length=int(size))
            total = len(blob)
            sha.update(blob)
        if total != int(obj.size):
            raise DataError(
                f"verify_generation_remote: 对象 {full_key} 回读 size 不匹配，"
                f"期望 {obj.size} 实际 {total}"
            )
        if sha.hexdigest() != obj.sha256:
            raise DataError(
                f"verify_generation_remote: 对象 {full_key} sha256 不匹配 "
                f"（期望 {obj.sha256}）"
            )


def generation_artifacts(manifest: GenerationManifest) -> dict[str, list[dict[str, Any]]]:
    """从 manifest 还原五类 artifact 的精确分布（kind -> [{key,size,sha256}, ...]）。

    T8 语义存于 ``GenerationObject.metadata["kind"]``（本模块 add 时写入）；
    这里把 ``manifest.objects`` 按 kind 分组，供读端/验收确认五类同 batch 同 generation。
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for obj in manifest.objects:
        kind = str((obj.metadata or {}).get("kind") or "")
        if not kind:
            continue
        out.setdefault(kind, []).append(
            {"key": obj.key, "size": obj.size, "sha256": obj.sha256}
        )
    return out


# ---- 读端 helper（只沿 CURRENT → manifest → 精确对象，无 glob） ---------------


def _read_object_bytes(store: ObjectStore, full_key: str) -> bytes:
    """读单个对象全文：优先 open_reader 流式，其次 range_read 全量。"""
    head = store.head_object(full_key)
    if head is None:
        raise DataError(f"对象不存在: {full_key}")
    reader = store.open_reader(full_key)
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
        raise DataError(f"对象 {full_key} 无 size 且无流式句柄，无法读取")
    return store.range_read(full_key, offset=0, length=int(size))


def resolve_current_generation(store: ObjectStore, prefix: str) -> GenerationManifest | None:
    """读端权威解析：``CURRENT.json → generation_id → _manifest.json``。

    - CURRENT 不存在或损坏 → 返回 None（读端视为无已发布数据）；
    - manifest 缺失/损坏 → 返回 None（fail-closed：绝不退回 glob 扫描）。

    本函数只 ``head/open_reader/range_read`` 精确 key，**从不** ``list_objects``。
    """
    prefix = str(prefix).strip("/")
    if not prefix:
        raise ValidationError("prefix 不能为空")
    current_key = f"{prefix}/{_CURRENT_KEY}"
    head = store.head_object(current_key)
    if head is None:
        return None
    try:
        payload = json.loads(_read_object_bytes(store, current_key).decode("utf-8"))
        gid = str(payload.get("generation_id") or "")
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError, DataError):
        return None
    if not gid:
        return None
    manifest_key = f"{prefix}/{gid}/{_MANIFEST_KEY}"
    mhead = store.head_object(manifest_key)
    if mhead is None:
        return None
    try:
        return GenerationManifest.from_dict(
            json.loads(_read_object_bytes(store, manifest_key).decode("utf-8"))
        )
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError, DataError):
        return None


def resolve_current_objects(store: ObjectStore, prefix: str) -> tuple[str, ...]:
    """返回 CURRENT 指向 generation 的**精确对象完整 key**（无 glob）。

    只沿 CURRENT → manifest → ``manifest.objects[].key`` 展开；未晋升/孤儿代次
    绝不返回。
    """
    manifest = resolve_current_generation(store, prefix)
    if manifest is None:
        return ()
    prefix = str(prefix).strip("/")
    return tuple(
        f"{prefix}/{manifest.generation_id}/{o.key}" for o in manifest.objects
    )


def read_current_bytes(store: ObjectStore, prefix: str, rel_key: str) -> bytes:
    """沿 CURRENT → manifest 精确读取 ``rel_key``（相对 generation 前缀）对象字节。

    ``rel_key`` 必须出现在 CURRENT manifest 的对象清单里，否则 ``DataError``
    （读端只能读 manifest 列出的精确对象——不 glob、不猜测未引用对象）。
    """
    rel_key = _validate_rel_key(rel_key)
    manifest = resolve_current_generation(store, prefix)
    if manifest is None:
        raise DataError(f"{prefix} 无 CURRENT generation，拒绝读取 {rel_key}")
    found = next((o for o in manifest.objects if o.key == rel_key), None)
    if found is None:
        raise DataError(
            f"rel_key {rel_key!r} 不在 CURRENT manifest 对象清单内"
            f"（generation={manifest.generation_id}，只允许精确对象读取）"
        )
    full_key = f"{manifest.prefix}/{manifest.generation_id}/{rel_key}"
    return _read_object_bytes(store, full_key)


__all__ = [
    "PUBLIC_META",
    "SECRET_META",
    "VALUE",
    "EVALUATION",
    "CATALOG",
    "T8_KINDS",
    "STAGING_FILENAMES",
    "layout_rel_key",
    "publish_t8_artifacts",
    "publish_t8_artifacts_retry",
    "publish_t8_dataset",
    "current_pointer_payload",
    "verify_generation_remote",
    "generation_artifacts",
    "resolve_current_generation",
    "resolve_current_objects",
    "read_current_bytes",
]
