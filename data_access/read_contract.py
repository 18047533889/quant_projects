# -*- coding: utf-8
"""读路径快照契约：ReadResult / DataSnapshot / lineage。"""
from __future__ import annotations

import hashlib
import json
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
    """单个 parquet 文件或 glob 模式版本描述。"""

    path: str
    size: int | None = None
    mtime_ns: int | None = None
    checksum: str | None = None


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
    instrument_filter: tuple[str, ...] = ()
    params: tuple[tuple[str, Any], ...] = ()


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


def build_file_manifest(paths: Sequence[str]) -> tuple[FileVersion, ...]:
    """从 DuckDB glob 路径列表构建文件 manifest（本地存在则补 stat）。"""
    import glob as glob_mod

    versions: list[FileVersion] = []
    seen: set[str] = set()
    for pattern in paths:
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


def file_manifest_hash(files: Sequence[FileVersion]) -> str:
    payload = [
        {"path": f.path, "size": f.size, "mtime_ns": f.mtime_ns, "checksum": f.checksum}
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
) -> DataSnapshot:
    canon = canonicalize_params(params)
    files = build_file_manifest(paths)
    manifest_hash = file_manifest_hash(files)
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
        files=files,
        params=tuple(sorted(canon.items())),
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

