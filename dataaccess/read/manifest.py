"""
data_access.read.manifest —— 数据集物理元数据清单（_manifest.parquet）

职责
    1. 每个数据集根放一个 ``_manifest.parquet``，记录每个文件：
       path / rows / bytes / min_time / max_time / min_instrument /
       max_instrument / schema_hash / mtime_ns / etag
    2. ``prune_by_time`` / ``prune_by_instruments``：读前按 manifest 的
       min/max 直接裁剪文件列表，避免 ``**/*.parquet`` 全量 glob + 逐文件
       ``pq.read_metadata()``
    3. 可扩展 row-group 级 statistics（column min/max/null_count）

设计要点
    1. min/max 存为规范化字符串（date/datetime→isoformat，int→str），
       同格式下字符串序 == 时间序，可直接比较。
    2. 仅在 manifest 存在且新鲜时使用；``is_fresh`` 用文件名级 glob 做
       数量比对（不读 footer），计数不一致视为过期。
    3. 构建是显式操作（``store.build_dataset_manifest`` / CLI），读路径只消费。

非职责
    不做谓词过滤（那是 DuckDB）；不负责时间分区 pattern 展开
    （partition_planner.py）；不读全表数据。

维护人：quant 基础平台组    最后更新：2026-08-07
"""

from __future__ import annotations

import glob as glob_module
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_FILENAME = "_manifest.parquet"

_MANIFEST_COLUMNS = [
    "path",
    "rows",
    "bytes",
    "min_time",
    "max_time",
    "min_instrument",
    "max_instrument",
    "schema_hash",
    "mtime_ns",
    "etag",
]


def _norm_value(value: Any) -> str | None:
    """把 parquet 统计值规范化为可比较字符串。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _time_key(value: Any) -> Any:
    """查询 time_range 端点的规范化键（与 _norm_value 对齐）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)[:10]


@dataclass(frozen=True)
class ManifestFile:
    path: str
    rows: int | None = None
    bytes: int | None = None
    min_time: str | None = None
    max_time: str | None = None
    min_instrument: str | None = None
    max_instrument: str | None = None
    schema_hash: str | None = None
    mtime_ns: int | None = None
    etag: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "bytes": self.bytes,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "min_instrument": self.min_instrument,
            "max_instrument": self.max_instrument,
            "schema_hash": self.schema_hash,
            "mtime_ns": self.mtime_ns,
            "etag": self.etag,
        }


@dataclass(frozen=True)
class ManifestRowGroup:
    path: str
    row_group: int
    column: str
    min: Any | None = None
    max: Any | None = None
    null_count: int | None = None
    rows: int | None = None


@dataclass
class DatasetManifest:
    """数据集物理元数据清单。"""

    dataset: str
    time_column: str | None = None
    instrument_column: str | None = None
    format: str = "parquet"
    files: tuple[ManifestFile, ...] = ()
    row_groups: tuple[ManifestRowGroup, ...] | None = None
    created_at: str | None = None

    @property
    def total_rows(self) -> int:
        return sum(f.rows or 0 for f in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes or 0 for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_table(self) -> pa.Table:
        rows = [f.to_row() for f in self.files]
        arrays: dict[str, list[Any]] = {c: [] for c in _MANIFEST_COLUMNS}
        for row in rows:
            for col in _MANIFEST_COLUMNS:
                arrays[col].append(row.get(col))
        return pa.table(arrays)

    def save(self, root: Path) -> Path:
        """写入 ``{root}/_manifest.parquet``（原子替换）。"""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        table = self.to_table()
        out = root / MANIFEST_FILENAME
        tmp = root / f".{MANIFEST_FILENAME}.tmp"
        pq.write_table(table, tmp)
        os.replace(str(tmp), str(out))
        return out

    @classmethod
    def load(cls, root: Path) -> "DatasetManifest | None":
        path = Path(root) / MANIFEST_FILENAME
        if not path.exists():
            return None
        try:
            table = pq.read_table(str(path))
        except Exception:
            return None
        data = table.to_pydict()
        files: list[ManifestFile] = []
        paths = data.get("path", [])
        for i in range(len(paths)):
            files.append(
                ManifestFile(
                    path=str(paths[i]),
                    rows=_int_or_none(_at(data, "rows", i)),
                    bytes=_int_or_none(_at(data, "bytes", i)),
                    min_time=_str_or_none(_at(data, "min_time", i)),
                    max_time=_str_or_none(_at(data, "max_time", i)),
                    min_instrument=_str_or_none(_at(data, "min_instrument", i)),
                    max_instrument=_str_or_none(_at(data, "max_instrument", i)),
                    schema_hash=_str_or_none(_at(data, "schema_hash", i)),
                    mtime_ns=_int_or_none(_at(data, "mtime_ns", i)),
                    etag=_str_or_none(_at(data, "etag", i)),
                )
            )
        meta = _read_manifest_meta(path)
        return cls(
            dataset=meta.get("dataset", ""),
            time_column=meta.get("time_column"),
            instrument_column=meta.get("instrument_column"),
            format=meta.get("format", "parquet"),
            files=tuple(files),
            created_at=meta.get("created_at"),
        )

    def prune_by_time(
        self, time_range: tuple[Any, Any] | None
    ) -> list[str]:
        """返回与 time_range 有交集的文件路径（按 min/max_time 裁剪）。"""
        if time_range is None or self.time_column is None:
            return [f.path for f in self.files]
        start_key = _time_key(time_range[0])
        end_key = _time_key(time_range[1])
        out: list[str] = []
        for f in self.files:
            if f.min_time is None or f.max_time is None:
                out.append(f.path)
                continue
            if start_key is not None and f.max_time < start_key:
                continue
            if end_key is not None and f.min_time > end_key:
                continue
            out.append(f.path)
        return out

    def prune_by_instruments(
        self, instrument_filter: Sequence[str] | None
    ) -> list[str]:
        """按 instrument 的字典序 min/max 裁剪（仅当字符串列统计可用）。"""
        if not instrument_filter:
            return [f.path for f in self.files]
        if self.instrument_column is None:
            return [f.path for f in self.files]
        lo = min(instrument_filter)
        hi = max(instrument_filter)
        out: list[str] = []
        for f in self.files:
            if f.min_instrument is None or f.max_instrument is None:
                out.append(f.path)
                continue
            if f.max_instrument < lo or f.min_instrument > hi:
                continue
            out.append(f.path)
        return out

    def prune(
        self,
        time_range: tuple[Any, Any] | None = None,
        instrument_filter: Sequence[str] | None = None,
    ) -> list[str]:
        """time + instrument 联合裁剪。"""
        paths = set(self.prune_by_time(time_range))
        if instrument_filter:
            paths &= set(self.prune_by_instruments(instrument_filter))
        return sorted(paths)


def _at(data: Mapping[str, list[Any]], col: str, i: int) -> Any:
    arr = data.get(col)
    if arr is None or i >= len(arr):
        return None
    return arr[i]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _read_manifest_meta(path: Path) -> dict[str, str]:
    """读 parquet key-value metadata（dataset/time_column/format/created_at）。"""
    try:
        meta = pq.read_metadata(str(path))
        kvs = meta.metadata
        out: dict[str, str] = {}
        if kvs is not None:
            for i in range(kvs.num_items):
                out[kvs.key(i)] = kvs.value(i)
        return out
    except Exception:
        return {}


def manifest_root_for_paths(paths: Sequence[str]) -> Path | None:
    """从解析出的 glob 路径推断 manifest 根目录（第一个路径的静态前缀）。"""
    for p in paths:
        if not p:
            continue
        static = p.split("*", 1)[0].rstrip("/")
        if static:
            return Path(static)
    return None


def is_manifest_fresh(manifest: DatasetManifest, glob_paths: Sequence[str]) -> bool:
    """文件名级新鲜度检查：glob 展开数量与 manifest 文件数一致才信任。

    只做文件名 glob（不读 footer），远快于逐文件 pq.read_metadata。
    """
    count = 0
    for pattern in glob_paths:
        if str(pattern).startswith("s3://"):
            # 远程无法本地计数 → 信任 manifest（COS 对象按日不变）
            return True
        if "*" in pattern or "?" in pattern or "[" in pattern:
            count += len(glob_module.glob(pattern, recursive=True))
        else:
            p = Path(pattern)
            if p.is_dir():
                count += len(list(p.rglob("*.parquet")))
            elif p.suffix in {".parquet", ".csv", ".tsv", ".jsonl", ".arrow", ".feather"} and p.exists():
                count += 1
    return count == manifest.file_count


def build_manifest_for_dataset(
    store: Any,
    dataset: str,
    *,
    params: Mapping[str, Any] | None = None,
    include_row_groups: bool = False,
    force: bool = False,
) -> DatasetManifest | None:
    """为数据集构建 manifest 并落盘。返回 None 表示没有可枚举文件。"""
    import glob as _glob

    from data_access.read.stats import expand_parquet_paths

    ds = store._registry.get(dataset)
    paths = store._resolve_raw_paths(ds, time_range=None, params=dict(params or {}))
    root = manifest_root_for_paths(paths)
    if root is None:
        return None

    # 只处理本层能读 footer 的格式（parquet 原生；csv 等无 footer → 略过）
    if getattr(ds, "format", "parquet") not in {"parquet", "pq"}:
        return None

    files = expand_parquet_paths(paths)
    if not files:
        return None

    time_col = ds.time_column
    inst_col = ds.instrument_column
    file_rows: list[ManifestFile] = []
    row_groups: list[ManifestRowGroup] = []
    for fp in files:
        try:
            meta = pq.read_metadata(str(fp))
        except Exception:
            continue
        rows = meta.num_rows
        fbytes = meta.serialized_size or _safe_stat(fp)
        schema_names = list(meta.schema.names)
        schema_hash = _schema_hash(meta.schema)
        min_t = max_t = min_i = max_i = None
        t_idx = schema_names.index(time_col) if time_col in schema_names else None
        i_idx = schema_names.index(inst_col) if inst_col in schema_names else None
        for rg in range(meta.num_row_groups):
            rg_meta = meta.row_group(rg)
            if include_row_groups and t_idx is not None:
                stats = rg_meta.column(t_idx).statistics
                if stats is not None:
                    row_groups.append(
                        ManifestRowGroup(
                            path=str(fp),
                            row_group=rg,
                            column=str(time_col),
                            min=_norm_value(stats.min),
                            max=_norm_value(stats.max),
                            null_count=stats.null_count,
                            rows=rg_meta.num_rows,
                        )
                    )
            if t_idx is not None:
                stats = rg_meta.column(t_idx).statistics
                if stats is not None:
                    lo, hi = _norm_value(stats.min), _norm_value(stats.max)
                    min_t = lo if (min_t is None or (lo is not None and lo < min_t)) else min_t
                    max_t = hi if (max_t is None or (hi is not None and hi > max_t)) else max_t
            if i_idx is not None:
                stats = rg_meta.column(i_idx).statistics
                if stats is not None:
                    lo, hi = _norm_value(stats.min), _norm_value(stats.max)
                    min_i = lo if (min_i is None or (lo is not None and lo < min_i)) else min_i
                    max_i = hi if (max_i is None or (hi is not None and hi > max_i)) else max_i
        try:
            st = fp.stat()
            mtime_ns = st.st_mtime_ns
        except OSError:
            mtime_ns = None
        file_rows.append(
            ManifestFile(
                path=str(fp),
                rows=rows,
                bytes=fbytes,
                min_time=min_t,
                max_time=max_t,
                min_instrument=min_i,
                max_instrument=max_i,
                schema_hash=schema_hash,
                mtime_ns=mtime_ns,
            )
        )

    if not file_rows:
        return None
    manifest = DatasetManifest(
        dataset=dataset,
        time_column=time_col,
        instrument_column=inst_col,
        format=str(getattr(ds, "format", "parquet")),
        files=tuple(file_rows),
        row_groups=tuple(row_groups) if row_groups else None,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    out_path = manifest.save(root)
    try:
        meta = pq.read_metadata(str(out_path))
        # 把 dataset 信息塞进 key-value metadata（重写一次）
        table = pq.read_table(str(out_path))
        kv = dict(table.schema.metadata or {})
        kv.update(
            {
                b"dataset": manifest.dataset.encode(),
                b"time_column": (manifest.time_column or "").encode(),
                b"instrument_column": (manifest.instrument_column or "").encode(),
                b"format": manifest.format.encode(),
                b"created_at": (manifest.created_at or "").encode(),
            }
        )
        tmp = out_path.with_suffix(".tmp")
        pq.write_table(table, tmp, metadata=kv)
        os.replace(str(tmp), str(out_path))
    except Exception:
        pass
    return manifest


def _safe_stat(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _schema_hash(schema: Any) -> str:
    # schema_arrow 是 pa.Schema；没有时退回 parquet 物理类型
    arrow_schema = getattr(schema, "schema_arrow", None)
    if arrow_schema is not None:
        names = list(arrow_schema.names)
        types = [str(arrow_schema.field(i).type) for i in range(len(names))]
    else:
        names = list(schema.names)
        types = [str(schema.column(i).physical_type) for i in range(len(names))]
    text = json.dumps(list(zip(names, types)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_manifest_for_dataset(store: Any, dataset: str, **params: Any) -> DatasetManifest | None:
    """按数据集加载 manifest（存在且新鲜时返回）。"""
    ds = store._registry.get(dataset)
    paths = store._resolve_raw_paths(ds, time_range=None, params=params)
    root = manifest_root_for_paths(paths)
    if root is None:
        return None
    manifest = DatasetManifest.load(root)
    if manifest is None:
        return None
    if manifest.dataset and manifest.dataset != dataset:
        return None
    if not is_manifest_fresh(manifest, paths):
        return None
    return manifest
