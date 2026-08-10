# -*- coding: utf-8
"""读路径快照契约：ReadResult / DataSnapshot / lineage。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """稳定序列化 params（用于 scope / snapshot fingerprint）。"""
    if not params:
        return {}
    out: dict[str, Any] = {}
    for key in sorted(params):
        val = params[key]
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[str(key)] = val
        elif isinstance(val, (list, tuple)):
            out[str(key)] = [_jsonable(v) for v in val]
        elif isinstance(val, dict):
            out[str(key)] = canonicalize_params(val)
        else:
            out[str(key)] = repr(val)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return repr(value)


@dataclass(frozen=True)
class FileVersion:
    """单个 parquet 文件或 glob 模式版本描述。

    #8 远程对象版本：s3:// / cos:// 对象同 key 被覆盖后，只记 path 的
    snapshot 永远不变——必须带上 etag/version_id/content_length/last_modified
    才构成「真实快照身份」。本地文件只填 size/mtime_ns。
    """

    path: str
    size: int | None = None
    mtime_ns: int | None = None
    checksum: str | None = None
    etag: str | None = None
    version_id: str | None = None
    content_length: int | None = None
    last_modified: Any = None


@dataclass(frozen=True)
class DataSnapshot:
    """一次读操作绑定的数据快照身份。"""

    snapshot_id: str
    dataset: str
    registry_hash: str
    schema_hash: str
    file_manifest_hash: str
    files: tuple[FileVersion, ...]
    params: tuple[tuple[str, Any], ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "dataset": self.dataset,
            "registry_hash": self.registry_hash,
            "schema_hash": self.schema_hash,
            "file_manifest_hash": self.file_manifest_hash,
            "files": [
                {
                    "path": f.path,
                    "size": f.size,
                    "mtime_ns": f.mtime_ns,
                    "checksum": f.checksum,
                    "etag": f.etag,
                    "version_id": f.version_id,
                    "content_length": f.content_length,
                    "last_modified": (
                        f.last_modified.isoformat()
                        if getattr(f.last_modified, "isoformat", None)
                        else f.last_modified
                    ),
                }
                for f in self.files
            ],
            "params": dict(self.params),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class ReadStats:
    rows: int
    bytes: int
    elapsed_ms: float
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadLineage:
    dataset: str
    columns: tuple[str, ...] = ()
    time_range: tuple[Any, Any] | None = None
    # #P1-final closure：保留 None（全市场）与 ()（空股票池）的区别——lineage 是
    # provenance/reproducibility 记录，``None`` 和 ``[]`` 语义不同（None=不限制，
    # []=空池 WHERE FALSE），不能折叠成同一个 ()。
    instrument_filter: tuple[str, ...] | None = ()
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        # 统一把序列型字段冻结成 tuple（list/set 传入时防 lineage 被外部变异）。
        object.__setattr__(
            self,
            "columns",
            tuple(self.columns) if self.columns is not None else (),
        )
        if self.instrument_filter is not None:
            object.__setattr__(self, "instrument_filter", tuple(self.instrument_filter))
        object.__setattr__(
            self,
            "params",
            tuple(self.params) if self.params is not None else (),
        )


def lineage_params(params: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    """把 params canonicalize 成**不可变**的 (k, v) 元组序列（lineage 用）。

    #P1-final closure：stream/aggregation 路径此前把 mutable dict 直接塞给声明为
    tuple 的 ``ReadLineage.params``——调用方之后改 params 会污染 lineage。
    canonicalize（稳定排序 + jsonable 化）后按值冻结成不可变嵌套结构。
    """
    if not params:
        return ()
    canon = canonicalize_params(params)
    return tuple(
        sorted((str(k), _freeze_lineage_value(v)) for k, v in canon.items())
    )


def _freeze_lineage_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted((str(k), _freeze_lineage_value(v)) for k, v in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_lineage_value(v) for v in value)
    return value


@dataclass(frozen=True)
class ReadResult:
    table: pa.Table
    snapshot: DataSnapshot
    stats: ReadStats
    lineage: ReadLineage


def schema_hash_from_decl(schema: Mapping[str, str] | None) -> str:
    if not schema:
        return "empty"
    payload = json.dumps(dict(sorted(schema.items())), sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)[:16]


# 进程内 memo：s3:// URI → (ts, 对象头元数据)（避免每次 manifest 都 head 一次）。
# #P0-C6 不永久缓存：remote 对象同 key 会被覆盖（ETag/version_id 变化，snapshot_id
# 必须跟着变）；失败的 None（无凭证/网络）也不该永久缓存（首次无凭证、之后补凭证
# 必须能重试）。统一短 TTL，过期即 revalidation——strict/pin 永远拿不到超过 TTL
# 的陈旧 HEAD。
_remote_meta_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_REMOTE_META_TTL_SECONDS = 30.0


def _remote_object_meta(uri: str, *, fresh: bool = False) -> dict[str, Any] | None:
    """s3:// / cos:// 对象头元数据（best-effort，带 TTL memo）。

    需要已配置 S3 凭证（``cos.remote.resolve_s3_credentials``）与 boto3。
    失败（无凭证/网络/未装 boto3）返回 None 并 memoize 为 None，**绝不阻塞**
    读路径。对象同 key 被覆盖 → etag/version_id 变化 → snapshot_id 跟着变。

    ``fresh=True``（#P0 收官 0.9.5）：**绕过 TTL 缓存强制重新 HEAD**。ScanHandle
    collect 前的 snapshot revalidation 用它——collect(t1) 必须拿到 scan(t0) 之后
    对象的最新身份，不能复用 30s 内的旧 HEAD 而漏掉同 key 覆盖。
    """
    key = str(uri)
    now = time.monotonic()
    if not fresh:
        cached = _remote_meta_cache.get(key)
        if cached is not None and now - cached[0] < _REMOTE_META_TTL_SECONDS:
            return cached[1]
    try:
        from data_access.cos.remote import cos_uri_to_s3_uri, resolve_s3_credentials

        creds = resolve_s3_credentials()
        import boto3
        from botocore.config import Config

        # #P0-C7 先统一 cos:// → s3:// 再切 bucket/key：旧代码对所有 scheme 用
        # ``key[len("s3://"):]``，cos:// 前缀长度不同导致 bucket/key 直接错位
        # （generic cos:// 对象 snapshot 元数据解析错误）。
        s3_uri = cos_uri_to_s3_uri(key)
        path = s3_uri[len("s3://") :]
        bucket, sep, obj = path.partition("/")
        if not sep or not obj:
            _remote_meta_cache[key] = (now, None)
            return None
        s3 = boto3.client(
            "s3",
            endpoint_url=(
                ("https://" if creds.use_ssl else "http://") + creds.endpoint
                if creds.endpoint
                else None
            ),
            region_name=creds.region,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            config=Config(
                connect_timeout=2, read_timeout=5, retries={"max_attempts": 0}
            ),
        )
        resp = s3.head_object(Bucket=bucket, Key=obj)
        meta = {
            "etag": str(resp.get("ETag", "")).strip('"') or None,
            "version_id": resp.get("VersionId"),
            "content_length": resp.get("ContentLength"),
            "last_modified": resp.get("LastModified"),
        }
        _remote_meta_cache[key] = (now, meta)
        return meta
    except Exception:
        _remote_meta_cache[key] = (now, None)
        return None


def _remote_snapshot_meta_enabled() -> bool:
    """是否对 s3:// 对象 head 元数据（默认：strict/production 才开，避免 dev
    每次读都打网络；可用 DATA_ACCESS_REMOTE_SNAPSHOT_META=1 强制开启）。"""
    import os

    raw = os.environ.get("DATA_ACCESS_REMOTE_SNAPSHOT_META", "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    try:
        from data_access.read.query_budget import is_strict_semantics

        return is_strict_semantics()
    except Exception:
        return False


def _path_has_glob(path: str) -> bool:
    """路径是否含 glob 通配（* / ? / {..}）。"""
    return any(c in path for c in ("*", "?", "{", "}"))


def build_file_manifest(paths: Sequence[str]) -> tuple[FileVersion, ...]:
    """从 DuckDB glob 路径列表构建文件 manifest（本地 stat / 远程对象头）。

    R25 P0-009：remote wildcard（``s3://bucket/table/*.parquet``）**不再是**一个
    FileVersion——它不构成可证明的 snapshot 对象。含通配的远程 pattern 标注
    ``etag=None`` 且 ``content_length=None``，并把原始 glob 记录在 path 里；
    strict snapshot 校验（``data_access.snapshot.SnapshotVerifier``）据此在
    production 下拒绝 unresolved wildcard 直接进 executor。
    """
    import glob as glob_mod

    enable_remote = _remote_snapshot_meta_enabled()
    versions: list[FileVersion] = []
    seen: set[str] = set()
    for pattern in paths:
        if str(pattern).startswith("s3://") or str(pattern).startswith("cos://"):
            if pattern not in seen:
                seen.add(str(pattern))
                extra: dict[str, Any] = {}
                if enable_remote and not _path_has_glob(str(pattern)):
                    meta = _remote_object_meta(pattern)
                    if meta:
                        extra = meta
                # R25 P0-009：glob 仍出现在 path 里（供 snapshot resolver / audit
                # 识别 unresolved wildcard），但不带 etag/content_length 冒充 exact
                # object。snapshot 层负责把它解析成 exact object 集。
                versions.append(FileVersion(path=str(pattern), **extra))
            continue
        expanded = sorted(glob_mod.glob(pattern, recursive=True))
        if not expanded:
            if pattern not in seen:
                seen.add(pattern)
                versions.append(FileVersion(path=pattern))
            continue
        for fp in expanded:
            if fp in seen:
                continue
            seen.add(fp)
            p = Path(fp)
            try:
                st = p.stat()
                versions.append(
                    FileVersion(path=fp, size=st.st_size, mtime_ns=st.st_mtime_ns)
                )
            except OSError:
                versions.append(FileVersion(path=fp))
    return tuple(versions)


def file_versions_from_manifest(manifest: Any, paths: Sequence[str]) -> tuple[FileVersion, ...]:
    """从一份 fresh DatasetManifest 构建 FileVersion 列表，避免逐文件 ``stat``。

    快照构建时 manifest 已存在且新鲜：把 pruned glob 展开后按路径查 manifest 的
    bytes/mtime_ns（省掉 NFS/COS 上昂贵的 O(N) stat）；manifest 里没有的文件
    回退 stat。与 ``build_file_manifest`` 输出结构完全一致。
    """
    import glob as glob_mod

    by_path = {str(f.path): f for f in getattr(manifest, "files", ())}
    enable_remote = _remote_snapshot_meta_enabled()
    versions: list[FileVersion] = []
    seen: set[str] = set()
    for pattern in paths:
        if str(pattern).startswith("s3://") or str(pattern).startswith("cos://"):
            if pattern not in seen:
                seen.add(str(pattern))
                extra: dict[str, Any] = {}
                if enable_remote:
                    meta = _remote_object_meta(pattern)
                    if meta:
                        extra = meta
                versions.append(FileVersion(path=str(pattern), **extra))
            continue
        expanded = sorted(glob_mod.glob(pattern, recursive=True))
        if not expanded:
            if pattern not in seen:
                seen.add(pattern)
                versions.append(FileVersion(path=pattern))
            continue
        for fp in expanded:
            if fp in seen:
                continue
            seen.add(fp)
            mf = by_path.get(fp)
            if mf is not None:
                versions.append(
                    FileVersion(
                        path=fp,
                        size=mf.bytes,
                        mtime_ns=mf.mtime_ns,
                        etag=getattr(mf, "etag", None),
                        version_id=getattr(mf, "version_id", None),
                        content_length=getattr(mf, "content_length", None),
                        last_modified=getattr(mf, "last_modified", None),
                    )
                )
                continue
            p = Path(fp)
            try:
                st = p.stat()
                versions.append(
                    FileVersion(path=fp, size=st.st_size, mtime_ns=st.st_mtime_ns)
                )
            except OSError:
                versions.append(FileVersion(path=fp))
    return tuple(versions)


def file_manifest_hash(files: Sequence[FileVersion]) -> str:
    payload = [
        {
            "path": f.path,
            "size": f.size,
            "mtime_ns": f.mtime_ns,
            "checksum": f.checksum,
            "etag": f.etag,
            "version_id": f.version_id,
            "content_length": f.content_length,
            "last_modified": f.last_modified,
        }
        for f in files
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_text(text)[:16]


def build_data_snapshot(
    *,
    dataset: str,
    registry_hash: str,
    schema: Mapping[str, str] | None,
    paths: Sequence[str],
    params: Mapping[str, Any] | None = None,
    files: Sequence[FileVersion] | None = None,
) -> DataSnapshot:
    canon = canonicalize_params(params)
    file_versions = tuple(files) if files is not None else build_file_manifest(paths)
    manifest_hash = file_manifest_hash(file_versions)
    schema_hash = schema_hash_from_decl(schema)
    identity = json.dumps(
        {
            "dataset": dataset,
            "registry_hash": registry_hash,
            "schema_hash": schema_hash,
            "manifest_hash": manifest_hash,
            "params": canon,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    snapshot_id = _sha256_text(identity)[:24]
    return DataSnapshot(
        snapshot_id=snapshot_id,
        dataset=dataset,
        registry_hash=registry_hash,
        schema_hash=schema_hash,
        file_manifest_hash=manifest_hash,
        files=file_versions,
        params=tuple(sorted(canon.items())),
    )


def rebuild_snapshot_files(
    snapshot: DataSnapshot, files: Sequence[FileVersion]
) -> DataSnapshot:
    """重建 snapshot（文件版本已变化，如 scan collect 前底层文件被覆盖）。

    保持 dataset / registry_hash / schema_hash / params / created_at 不变，
    按新 files 重算 file_manifest_hash 与 snapshot_id——lineage/缓存身份与
    「实际读到什么」重新对齐，不再把旧 snapshot 当成刚读的数据（#7）。
    """
    file_versions = tuple(files)
    manifest_hash = file_manifest_hash(file_versions)
    identity = json.dumps(
        {
            "dataset": snapshot.dataset,
            "registry_hash": snapshot.registry_hash,
            "schema_hash": snapshot.schema_hash,
            "manifest_hash": manifest_hash,
            "params": dict(snapshot.params),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return DataSnapshot(
        snapshot_id=_sha256_text(identity)[:24],
        dataset=snapshot.dataset,
        registry_hash=snapshot.registry_hash,
        schema_hash=snapshot.schema_hash,
        file_manifest_hash=manifest_hash,
        files=file_versions,
        params=snapshot.params,
        created_at=snapshot.created_at,
    )


@dataclass(frozen=True)
class SqlReadLineage:
    """sql() 多 dataset 读 lineage。"""

    datasets: tuple[str, ...]
    read_params: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    query_preview: str = ""


@dataclass(frozen=True)
class SqlReadResult:
    """sql() 返回：结果表 + 合并 snapshot。"""

    table: pa.Table
    snapshot: DataSnapshot
    stats: ReadStats
    lineage: SqlReadLineage


def merge_sql_data_snapshots(
    snapshots: Sequence[DataSnapshot],
    *,
    registry_hash: str,
) -> DataSnapshot:
    """多 dataset sql() 路径：合并各 dataset snapshot 为单一 identity。"""
    if not snapshots:
        raise ValueError("merge_sql_data_snapshots 需要至少一个 snapshot")
    if len(snapshots) == 1:
        return snapshots[0]

    ordered = sorted(snapshots, key=lambda s: s.dataset)
    combined_files: list[FileVersion] = []
    seen_paths: set[str] = set()
    for snap in ordered:
        for fv in snap.files:
            if fv.path not in seen_paths:
                seen_paths.add(fv.path)
                combined_files.append(fv)

    manifest_hash = file_manifest_hash(combined_files)
    merged_params: dict[str, Any] = {}
    for snap in ordered:
        for k, v in snap.params:
            merged_params[f"{snap.dataset}.{k}"] = v

    identity = json.dumps(
        {
            "kind": "sql_merge",
            "datasets": [s.dataset for s in ordered],
            "child_snapshot_ids": [s.snapshot_id for s in ordered],
            "registry_hash": registry_hash,
            "manifest_hash": manifest_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    snapshot_id = _sha256_text(identity)[:24]
    schema_hash = _sha256_text(
        "|".join(s.schema_hash for s in ordered)
    )[:16]
    return DataSnapshot(
        snapshot_id=snapshot_id,
        dataset=",".join(s.dataset for s in ordered),
        registry_hash=registry_hash,
        schema_hash=schema_hash,
        file_manifest_hash=manifest_hash,
        files=tuple(combined_files),
        params=tuple(sorted(merged_params.items())),
    )

