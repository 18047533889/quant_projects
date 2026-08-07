"""
data_access.core.storage —— 存储后端抽象（local / s3 / cos / http / cli / clickhouse）

职责
    1. 归一化「数据放哪」：registry 的 ``storage:`` 声明 + COS mirror 注册表 +
       默认 local，解析成统一的 ``StorageSpec``
    2. 提供 ``is_remote_storage`` / ``to_s3_uri`` / ``authorize_path`` 等分派，
       让 COS 不再是 Store 里的特殊分支
    3. 未来接 S3 / MinIO / R2 / NAS / HTTP / ClickHouse 时，只加 backend + 分派

设计要点
    1. 不重写 COS mirror/remote 机制（已 battle-tested）：把它的决策结果
       （cos 前缀 / mode）归一化为 StorageSpec(type=cos)。
    2. 显式 ``storage:`` 声明优先；否则按数据集名查 COS mirror 注册表；都没有
       视为 local。
    3. 路径鉴权分派：local → PathAuthorizer；s3/cos → authorize_s3_path。

非职责
    不负责凭证解析（cos.remote.resolve_s3_credentials / cos.s3_duckdb）；
    不负责文件同步（cos.mirror）。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from data_access.core.exceptions import ValidationError


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    COS = "cos"
    HTTP = "http"
    CLI = "cli"          # clean-cos-ro 按需拉取
    CLICKHOUSE = "clickhouse"


_REMOTE_BACKENDS = frozenset({StorageBackend.S3, StorageBackend.COS, StorageBackend.HTTP})


@dataclass(frozen=True)
class StorageSpec:
    """统一存储声明。"""

    type: str = "local"
    uri: str | None = None          # 如 cos://bucket/prefix、s3://bucket/prefix
    root: str | None = None         # 本地根
    bucket: str | None = None
    prefix: str | None = None
    endpoint: str | None = None
    region: str | None = None
    secret: str | None = None       # DuckDB Secret 名
    mode: str | None = None         # cos: mirror | remote | auto
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def backend(self) -> StorageBackend:
        try:
            return StorageBackend(self.type)
        except ValueError:
            return StorageBackend.LOCAL

    @property
    def is_remote(self) -> bool:
        return self.backend in _REMOTE_BACKENDS

    @property
    def scheme(self) -> str:
        return "s3" if self.backend in {StorageBackend.S3, StorageBackend.COS} else self.type


def parse_storage_spec(raw: Any) -> StorageSpec | None:
    """解析 YAML ``storage:`` 块。None/空返回 None。"""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"type": raw}
    if not isinstance(raw, dict):
        raise ValidationError(
            f"storage 必须是字符串或 mapping，收到 {type(raw).__name__}"
        )
    stype = str(raw.get("type", "local")).strip().lower()
    try:
        backend = StorageBackend(stype)
    except ValueError:
        raise ValidationError(
            f"未知 storage.type={stype!r}。支持: {[b.value for b in StorageBackend]}"
        )
    options = raw.get("options") or {}
    if not isinstance(options, dict):
        raise ValidationError("storage.options 必须是 mapping")
    return StorageSpec(
        type=backend.value,
        uri=raw.get("uri"),
        root=raw.get("root"),
        bucket=raw.get("bucket"),
        prefix=raw.get("prefix"),
        endpoint=raw.get("endpoint"),
        region=raw.get("region"),
        secret=raw.get("secret"),
        mode=str(raw.get("mode", "")).strip() or None,
        options={str(k): v for k, v in options.items()},
    )


def resolve_storage_for_dataset(ds: Any) -> StorageSpec:
    """数据集 → StorageSpec。

    优先级：
        1. registry 显式 ``storage:`` 声明
        2. COS mirror 注册表命中 → type=cos（prefix/mode 对齐当前读模式）
        3. 默认 local
    """
    explicit = getattr(ds, "storage", None)
    if explicit:
        spec = parse_storage_spec(explicit)
        if spec is not None:
            return spec

    # COS mirror 注册表（A 股/美股本地镜像或 remote 源）
    try:
        from data_access.cos.mirror import mirror_spec_for_dataset
        from data_access.cos.remote import cos_read_mode

        mirror = mirror_spec_for_dataset(ds.name)
        if mirror is not None:
            return StorageSpec(
                type="cos",
                uri=mirror.cos_prefix,
                prefix=mirror.cos_prefix,
                root=str(mirror.local_root),
                mode=cos_read_mode(),
            )
    except Exception:
        pass
    return StorageSpec(type="local")


def is_remote_storage(ds: Any) -> bool:
    return resolve_storage_for_dataset(ds).is_remote


def to_s3_uri(uri: str) -> str:
    """``cos://bucket/key`` → ``s3://bucket/key``。"""
    if uri.startswith("cos://"):
        return "s3://" + uri[len("cos://") :]
    if uri.startswith("s3://"):
        return uri
    raise ValidationError(f"非 COS/S3 URI: {uri!r}")


def authorize_storage_path(
    ds: Any,
    path: str,
    *,
    authorizer: Any = None,
) -> None:
    """按存储后端分派路径鉴权。

    - local：PathAuthorizer.resolve_and_authorize
    - s3/cos：cos.remote.authorize_s3_path（前缀白名单）
    """
    if str(path).startswith("s3://") or str(path).startswith("cos://"):
        from data_access.cos.remote import authorize_s3_path

        authorize_s3_path(to_s3_uri(str(path)))
        return
    if authorizer is not None:
        authorizer.resolve_and_authorize(str(path))


def storage_description(ds: Any) -> str:
    """数据集存储描述（诊断/日志用）。"""
    spec = resolve_storage_for_dataset(ds)
    if spec.type == "local":
        return f"local({spec.root or getattr(ds, 'root', '?')})"
    return f"{spec.type}({spec.uri or spec.bucket or ''})" + (
        f":{spec.mode}" if spec.mode else ""
    )
