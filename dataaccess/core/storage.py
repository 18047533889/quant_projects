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

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
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

# ``scheme://``（URI 形态）匹配器。`..`/`/` 等非 URI 字符串不匹配。
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _backend_from_uri_scheme(uri: Any) -> str | None:
    """从 URI scheme 推导 storage backend；非 URI（无 scheme）返回 None。

    #7 类型未显式声明时**必须**从 scheme 推导，绝不能 fallback LOCAL——
    ``StorageSpec.from_yaml("cos://bucket/path")`` 之前被静默解析成
    ``type=local, uri=cos://...``，直接破坏 authorization / reader 路由 /
    snapshot / remote capability。未实现 scheme（如 ``oss://``）直接拒绝。
    """
    text = str(uri or "").strip()
    if not text:
        return None
    low = text.lower()
    if low.startswith("cos://"):
        return "cos"
    if low.startswith("s3://"):
        return "s3"
    if low.startswith(("http://", "https://")):
        return "http"
    if _SCHEME_RE.match(text):
        raise ValidationError(
            f"storage URI {uri!r} 的 scheme 未实现（支持 cos:// / s3:// / http(s)://）。"
            "不能把带 scheme 的 URI 静默当 local 处理。"
        )
    return None


def _deep_freeze(value: Any) -> Any:
    """递归把 mapping/list/set 转成不可变形式（深冻结）。

    #12：``StorageSpec`` 是 frozen typed IR，但 ``options`` 是可变 dict——
    ``ds.storage.options["x"]=...`` 可以绕过 frozen dataclass 的不可变契约，
    且 plan/fingerprint 未必感知内部 mutation。构造时深冻结，任何写路径都会
    TypeError，从根上把「registry compile 后即 immutable」落实。
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(v) for v in value)
    return value


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
    # #P0-final closure 4：remote 路径形态与文件格式（cos/remote 消费的
    # ``storage.source.layout`` / ``source.format`` 归一化后落在 typed spec 上，
    # 不再让各模块各自读 raw dict）。
    layout: str | None = None       # plain / daily_parquet / hive_date / hive_year
    format: str | None = None       # parquet / arrow / feather / csv ...
    credential_profile: str | None = None  # COS 凭证 profile 名
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """#39 programmatic construction 也 fail-closed：非法 type 直接抛。

        旧代码 ``StorageSpec(type="cosss")`` 会在 ``.backend`` 静默退化成
        LOCAL——把 remote 数据集当 local 读，是灾难性的。YAML 路径
        ``parse_storage_spec`` 早已校验；这里补上代码构造路径。
        """
        try:
            StorageBackend(str(self.type or "").strip().lower())
        except ValueError:
            raise ValidationError(
                f"未知 storage.type={self.type!r}。支持: "
                f"{[b.value for b in StorageBackend]}"
            )
        # #12 深冻结 options（见 ``_deep_freeze``）：frozen dataclass 不再留可变 dict。
        object.__setattr__(self, "options", _deep_freeze(self.options))

    @property
    def backend(self) -> StorageBackend:
        # __post_init__ 已 fail-closed；这里直接映射（防御保留）。
        return StorageBackend(str(self.type).strip().lower())

    @property
    def is_remote(self) -> bool:
        return self.backend in _REMOTE_BACKENDS

    @property
    def scheme(self) -> str:
        return "s3" if self.backend in {StorageBackend.S3, StorageBackend.COS} else self.type

    @classmethod
    def from_yaml(cls, raw: Any, *, context: str = "storage") -> "StorageSpec":
        """**唯一** ``storage:`` 解析入口（#P0-final closure 4）。

        - typed ``StorageSpec`` 原样返回（对象自身即合法，不依赖创建入口）；
        - 字符串 ``"cos"`` / ``"cos://bucket/prefix"``；
        - dict：顶层与嵌套 ``source``（``{source: {type: cos, ...}}``）合并解析，
          顶层字段优先；**任何 unknown key fail-closed**（``storage: {typo: ...}``
          不再静默忽略走默认 local）。
        """
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            text = str(raw).strip()
            if text.lower().startswith(("cos://", "s3://", "oss://", "http://", "https://")):
                raw = {"uri": text}
            else:
                raw = {"type": text}
        if not isinstance(raw, dict):
            raise ValidationError(
                f"{context} 必须是字符串或 mapping，收到 {type(raw).__name__}"
            )
        _reject_unknown_storage_keys(raw, allowed=_STORAGE_TOP_KEYS, context=context)
        src = raw.get("source")
        if src is not None:
            if isinstance(src, str):
                src = {"uri": str(src)}
            if not isinstance(src, dict):
                raise ValidationError(
                    f"{context}.source 必须是 mapping 或 uri 字符串，"
                    f"收到 {type(src).__name__}"
                )
            _reject_unknown_storage_keys(
                src, allowed=_STORAGE_SOURCE_KEYS, context=f"{context}.source"
            )
        merged: dict[str, Any] = {}
        if isinstance(src, dict):
            merged.update(src)
        for k, v in raw.items():
            if k != "source":
                merged[k] = v  # 顶层字段优先

        # #7 URI scheme → backend：``type`` 未显式声明时从 ``uri`` 的 scheme 推导，
        # 绝不 fallback LOCAL。``_backend_from_uri_scheme`` 对未知 scheme（oss://）
        # 直接拒绝；显式 type 与 scheme 推导矛盾也拒绝（配置写错的两种形态）。
        uri_raw = merged.get("uri")
        inferred = _backend_from_uri_scheme(uri_raw) if uri_raw else None
        stype_raw = merged.get("type")
        if stype_raw:
            stype = str(stype_raw).strip().lower()
            if inferred is not None and stype != inferred:
                raise ValidationError(
                    f"{context}: uri={uri_raw!r} 的 scheme 推导 backend={inferred!r}，"
                    f"与显式 type={stype!r} 矛盾。请删除其一使两者一致。"
                )
        else:
            stype = inferred if inferred is not None else "local"
        try:
            backend = StorageBackend(stype)
        except ValueError:
            raise ValidationError(
                f"{context}: 未知 storage.type={stype!r}。支持: "
                f"{[b.value for b in StorageBackend]}"
            )
        options = merged.get("options") or {}
        if not isinstance(options, dict):
            raise ValidationError(f"{context}.options 必须是 mapping")
        return cls(
            type=backend.value,
            uri=merged.get("uri"),
            root=merged.get("root"),
            bucket=merged.get("bucket"),
            prefix=merged.get("prefix"),
            endpoint=merged.get("endpoint"),
            region=merged.get("region"),
            secret=merged.get("secret"),
            mode=str(merged.get("mode", "")).strip() or None,
            layout=str(merged.get("layout", "")).strip() or None,
            format=str(merged.get("format", "")).strip() or None,
            credential_profile=(
                str(merged.get("credential_profile", "")).strip() or None
            ),
            options={str(k): v for k, v in options.items()},
        )


# #P0-final closure 4：唯一 ``StorageSpec`` schema。顶层与嵌套 ``source`` 都收，
# 合并后编译成 typed ``StorageSpec``——registry 保存 typed spec，任何模块不再
# 各自解析 raw dict（旧 split-brain：registry 认 ``storage.source.type``，而
# ``parse_storage_spec`` 只读顶层 ``type``，合法的 ``{source:{type:cos}}`` 会
# 被默认成 local）。
_STORAGE_TOP_KEYS = frozenset({
    "type", "source", "uri", "root", "bucket", "prefix", "endpoint", "region",
    "secret", "mode", "options", "layout", "format", "credential_profile",
})
_STORAGE_SOURCE_KEYS = frozenset({
    "type", "uri", "bucket", "prefix", "endpoint", "region", "secret", "mode",
    "options", "layout", "format", "credential_profile",
})


def _reject_unknown_storage_keys(mapping: Mapping[str, Any], *, allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValidationError(
            f"{context} 含未知配置 key {unknown}；应为 {sorted(allowed)} 之一"
        )


def parse_storage_spec(raw: Any) -> StorageSpec | None:
    """解析 YAML ``storage:`` 块。None/空返回 None。typed StorageSpec 原样返回。"""
    if raw is None:
        return None
    return StorageSpec.from_yaml(raw, context="storage")


def _storage_attr(ds: Any, key: str) -> Any:
    """从 ``ds.storage`` 取声明字段：typed ``StorageSpec`` 或旧 raw dict 都支持。"""
    storage = getattr(ds, "storage", None)
    if isinstance(storage, StorageSpec):
        return getattr(storage, key, None)
    if isinstance(storage, dict):
        src = storage.get("source")
        if isinstance(src, dict):
            return storage.get(key) if key in storage else src.get(key)
        if isinstance(src, str) and key == "uri":
            return src
        return storage.get(key)
    return None


def declared_storage_type(ds: Any) -> str | None:
    """数据集声明的 storage backend 类型（local/s3/cos/...）。"""
    val = _storage_attr(ds, "type")
    if val:
        return str(val).strip().lower()
    return None


def declared_storage_uri(ds: Any) -> str | None:
    """数据集声明的 storage uri（cos://... / s3://... / 远程前缀）。"""
    val = _storage_attr(ds, "uri")
    if val:
        return str(val).strip()
    return None


def declared_storage_layout(ds: Any) -> str | None:
    """数据集声明的远程路径形态（daily_parquet / hive_date / hive_year / plain）。"""
    val = _storage_attr(ds, "layout")
    if val:
        return str(val).strip().lower()
    return None


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
    except Exception as exc:
        # #38 fail-closed：COS 模块真实 bug / 配置损坏 / import 异常时，
        # **不得**把 remote 数据集静默当 local。research 降级 local + 告警；
        # production/strict 直接抛（明确要求修配置，而不是换一种语义）。
        from data_access.read.query_budget import is_strict_semantics

        msg = (
            f"解析数据集 {ds.name!r} 的 COS 存储声明失败：{type(exc).__name__}: {exc}。"
            "镜像注册表查询不可信时禁止静默降级为 local。"
        )
        if is_strict_semantics():
            raise ValidationError(msg) from exc
        import logging

        logging.getLogger("data_access.storage").warning("%s（research 降级 local）", msg)
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


@dataclass(frozen=True)
class BackendCapability:
    """#40 一个 StorageBackend 的真实能力声明（authorization / reader / snapshot）。

    枚举里声明了 HTTP/CLI/CLICKHOUSE，但若真实实现没有对应能力，语义就漂了。
    这里显式列出每后端的实际支持程度，路由/鉴权/快照按它分派，不再靠
    scheme 字符串猜。
    """

    authorization: str        # "local_authorizer" | "s3_prefix" | "none"
    duckdb_httpfs: bool       # DuckDB 能否直接 read_parquet(uri)
    snapshot_metadata: bool   # 能否取对象版本（etag/version_id/…）
    read_implemented: bool    # 当前是否真正可读（HTTP/CLI/CH 尚未实现 reader）


_BACKEND_CAPABILITIES: dict[StorageBackend, BackendCapability] = {
    StorageBackend.LOCAL: BackendCapability(
        authorization="local_authorizer", duckdb_httpfs=False,
        snapshot_metadata=False, read_implemented=True,
    ),
    StorageBackend.S3: BackendCapability(
        authorization="s3_prefix", duckdb_httpfs=True,
        snapshot_metadata=True, read_implemented=True,
    ),
    StorageBackend.COS: BackendCapability(
        authorization="s3_prefix", duckdb_httpfs=True,
        snapshot_metadata=True, read_implemented=True,
    ),
    StorageBackend.HTTP: BackendCapability(
        authorization="none", duckdb_httpfs=False,
        snapshot_metadata=False, read_implemented=False,
    ),
    StorageBackend.CLI: BackendCapability(
        authorization="local_authorizer", duckdb_httpfs=False,
        snapshot_metadata=True, read_implemented=True,  # clean-cos-ro 落盘后本地读
    ),
    StorageBackend.CLICKHOUSE: BackendCapability(
        authorization="none", duckdb_httpfs=False,
        snapshot_metadata=False, read_implemented=False,
    ),
}


def storage_backend_capabilities(backend: StorageBackend) -> BackendCapability:
    """取一个后端的真实能力；未登记后端 fail-closed（宁抛不猜）。"""
    try:
        return _BACKEND_CAPABILITIES[backend]
    except KeyError:
        raise ValidationError(f"StorageBackend {backend!r} 未登记能力矩阵")


def backend_readable(backend: StorageBackend) -> bool:
    """该后端当前是否真正可读（否则路径解析/读路由应拒绝而非静默降级）。"""
    return storage_backend_capabilities(backend).read_implemented


def authorize_storage_path(
    ds: Any,
    path: str,
    *,
    authorizer: Any = None,
) -> None:
    """按存储后端能力分派路径鉴权（#40，替代硬编码 scheme 判断）。

    - local_authorizer：PathAuthorizer.resolve_and_authorize
    - s3_prefix：cos.remote.authorize_s3_path（前缀白名单）
    - none：不支持鉴权（HTTP/CH）→ fail-closed
    """
    spec = resolve_storage_for_dataset(ds)
    cap = storage_backend_capabilities(spec.backend)
    if cap.authorization == "s3_prefix":
        from data_access.cos.remote import authorize_s3_path

        authorize_s3_path(to_s3_uri(str(path)))
        return
    if cap.authorization == "local_authorizer":
        if authorizer is not None:
            authorizer.resolve_and_authorize(str(path))
        return
    raise ValidationError(
        f"数据集 {ds.name!r} 的存储后端 {spec.type!r} 未实现鉴权（HTTP/ClickHouse）"
    )


def storage_description(ds: Any) -> str:
    """数据集存储描述（诊断/日志用）。"""
    spec = resolve_storage_for_dataset(ds)
    if spec.type == "local":
        return f"local({spec.root or getattr(ds, 'root', '?')})"
    return f"{spec.type}({spec.uri or spec.bucket or ''})" + (
        f":{spec.mode}" if spec.mode else ""
    )
