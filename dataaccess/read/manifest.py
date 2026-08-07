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
_MANIFEST_META_FILENAME = "_manifest.json"

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


def _norm_value(value: Any) -> Any:
    """把 parquet 统计值规范化为可比较键（数值保留数值，时间保留全精度 iso）。

    注意：不再把 int/float 字符串化（避免 ``"9" > "10"`` 字典序 bug），
    也不再对字符串 ``[:10]`` 截断（保留 intraday 全精度）。
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return value
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
        return value
    return str(value)


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
    time_dtype: str | None = None       # date / timestamp / int64 / float64 / string（typed 比较用）
    manifest_epoch: str | None = None   # 写路径 mutation 后递增；读路径信任它（O(1) freshness）

    def _typed(self, value: str | None, *, ints: bool = False) -> Any:
        """把 manifest 里存的 min/max 按 time_dtype 还原成可比较类型。"""
        if value is None:
            return None
        td = (self.time_dtype or "").lower()
        if "int" in td or ints:
            try:
                return int(value)
            except (TypeError, ValueError):
                return value
        if "float" in td:
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        return value

    @property
    def total_rows(self) -> int:
        return sum(f.rows or 0 for f in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes or 0 for f in self.files)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def schema_hashes(self) -> tuple[str, ...]:
        return tuple(sorted({f.schema_hash for f in self.files if f.schema_hash}))

    @property
    def dataset_version(self) -> str:
        """数据集内容版本：schema + 列定义 + 文件集合（结构变 → 版本变）。"""
        payload = {
            "format": self.format,
            "time_column": self.time_column,
            "instrument_column": self.instrument_column,
            "schema_hashes": list(self.schema_hashes),
            "file_count": self.file_count,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    @property
    def partition_version(self) -> str:
        """物理分区版本：文件数 + 字节 + 最大 mtime（新文件/替换 → 版本变）。"""
        max_mtime = max((f.mtime_ns or 0) for f in self.files) if self.files else 0
        payload = {
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "max_mtime_ns": max_mtime,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def to_table(self) -> pa.Table:
        rows = [f.to_row() for f in self.files]
        arrays: dict[str, list[Any]] = {c: [] for c in _MANIFEST_COLUMNS}
        for row in rows:
            for col in _MANIFEST_COLUMNS:
                arrays[col].append(row.get(col))
        return pa.table(arrays)

    def save(self, root: Path) -> Path:
        """写入 ``{root}/_manifest.parquet`` + ``_manifest.json``（meta，原子替换）。"""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        table = self.to_table()
        out = root / MANIFEST_FILENAME
        tmp = root / f".{MANIFEST_FILENAME}.tmp"
        pq.write_table(table, tmp)
        os.replace(str(tmp), str(out))
        if self.row_groups:
            _save_row_groups(root, self.row_groups)
        # meta 放独立 JSON sidecar（footer key-value metadata 各版本 pyarrow 行为不稳）
        epoch = self.manifest_epoch or _next_epoch(_read_epoch(root))
        meta_path = root / _MANIFEST_META_FILENAME
        tmp_meta = root / f".{_MANIFEST_META_FILENAME}.tmp"
        tmp_meta.write_text(
            json.dumps(
                {
                    "dataset": self.dataset,
                    "time_column": self.time_column,
                    "instrument_column": self.instrument_column,
                    "format": self.format,
                    "time_dtype": self.time_dtype,
                    "manifest_epoch": epoch,
                    "created_at": self.created_at,
                    "file_count": self.file_count,
                    "dataset_version": self.dataset_version,
                    "partition_version": self.partition_version,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(str(tmp_meta), str(meta_path))
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
        meta = _read_manifest_meta_json(root)
        return cls(
            dataset=meta.get("dataset", ""),
            time_column=meta.get("time_column"),
            instrument_column=meta.get("instrument_column"),
            format=meta.get("format", "parquet"),
            files=tuple(files),
            row_groups=_load_row_groups(root),
            created_at=meta.get("created_at"),
            time_dtype=meta.get("time_dtype"),
            manifest_epoch=meta.get("manifest_epoch"),
        )

    def prune_by_time(
        self, time_range: tuple[Any, Any] | None
    ) -> list[str]:
        """返回与 time_range 有交集的文件路径（按 min/max_time 裁剪）。

        min/max 比较是类型化的（#19）：int64 时间列按数值比较，避免
        ``"9" > "10"`` 字典序 bug；date/timestamp 按 isoformat 字符串比较。
        """
        if time_range is None or self.time_column is None:
            return [f.path for f in self.files]
        start_key = _time_key(time_range[0])
        end_key = _time_key(time_range[1])
        ints = "int" in (self.time_dtype or "").lower()
        if isinstance(start_key, (int, float)):
            ints = True
        if isinstance(end_key, (int, float)):
            ints = True
        out: list[str] = []
        for f in self.files:
            if f.min_time is None or f.max_time is None:
                out.append(f.path)
                continue
            lo = self._typed(f.min_time, ints=ints)
            hi = self._typed(f.max_time, ints=ints)
            if start_key is not None and hi is not None:
                try:
                    if hi < start_key:
                        continue
                except TypeError:
                    pass
            if end_key is not None and lo is not None:
                try:
                    if lo > end_key:
                        continue
                except TypeError:
                    pass
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


def _read_manifest_meta_json(root: Path) -> dict[str, str]:
    """读 ``_manifest.json`` sidecar（dataset/time_column/format/created_at）。"""
    path = Path(root) / _MANIFEST_META_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "dataset": str(payload.get("dataset", "")),
        "time_column": payload.get("time_column"),
        "instrument_column": payload.get("instrument_column"),
        "format": str(payload.get("format", "parquet")),
        "created_at": payload.get("created_at"),
        "time_dtype": payload.get("time_dtype"),
        "manifest_epoch": payload.get("manifest_epoch"),
    }


def manifest_root_for_paths(paths: Sequence[str]) -> Path | None:
    """从解析出的 glob 路径推断 manifest 根目录（第一个路径的静态前缀）。"""
    for p in paths:
        if not p:
            continue
        static = p.split("*", 1)[0].rstrip("/")
        if static:
            return Path(static)
    return None


def _count_data_files(glob_paths: Sequence[str]) -> int | None:
    """文件名级 glob 计数（只算数据文件，排除 manifest 自身）。

    返回 None 表示无法本地计数（远程 glob）→ 调用方应信任 manifest。
    """
    count = 0
    for pattern in glob_paths:
        if str(pattern).startswith("s3://"):
            # 远程无法本地计数 → 信任 manifest（COS 对象按日不变）
            return None
        matches = []
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matches = glob_module.glob(pattern, recursive=True)
        else:
            p = Path(pattern)
            if p.is_dir():
                matches = [str(x) for x in p.rglob("*.parquet")]
            elif p.suffix in {".parquet", ".csv", ".tsv", ".jsonl", ".arrow", ".feather"} and p.exists():
                matches = [str(p)]
        for m in matches:
            # manifest 自身（_manifest.parquet / _manifest.json）不算数据文件
            name = str(m).split("/")[-1]
            if name == MANIFEST_FILENAME or name == _MANIFEST_META_FILENAME:
                continue
            count += 1
    return count


def is_manifest_fresh(manifest: DatasetManifest, glob_paths: Sequence[str]) -> bool:
    """新鲜度检查（默认 O(1)）：写路径维护 ``manifest_epoch`` → 读路径信任。

    只有旧格式（无 epoch 的 ``_manifest.json``）才退回文件名 glob 比对，保证
    老 manifest 在重建前仍然可用。新 manifest 不再在 read path 做 O(N) glob。
    """
    root = manifest_root_for_paths(list(glob_paths) if glob_paths else [])
    if root is not None:
        meta = manifest_version_token(root)
        if meta is not None and meta.get("manifest_epoch") is not None:
            return True
    count = _count_data_files(glob_paths)
    if count is None:
        return True
    return count == manifest.file_count


def manifest_version_token(root: Path) -> dict[str, Any] | None:
    """廉价版本 token：只读 ``_manifest.json`` sidecar，不读 manifest parquet。

    供 query-scoped snapshot 的 ``store.manifest_version()`` 使用——几十微秒级，
    避免每次 describe 整个 dataset 再执行一次实际 read。
    """
    meta_path = Path(root) / _MANIFEST_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


_ROW_GROUPS_FILENAME = "_manifest_rowgroups.parquet"


def _next_epoch(existing: str | None) -> str:
    try:
        return str(int(existing or 0) + 1)
    except ValueError:
        return "1"


def _read_epoch(root: Path) -> str | None:
    meta = manifest_version_token(root)
    if meta is None:
        return None
    return meta.get("manifest_epoch")


def bump_manifest_epoch(root: Path) -> str | None:
    """写路径 mutation 后调用：递增 ``_manifest.json`` 的 ``manifest_epoch``。

    只改 sidecar（O(1)，不重建 manifest、不 glob）。返回新 epoch；无 sidecar
    返回 None（该数据集没有 manifest 可失效）。
    """
    meta_path = Path(root) / _MANIFEST_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    epoch = _next_epoch(payload.get("manifest_epoch"))
    payload["manifest_epoch"] = epoch
    tmp_meta = meta_path.with_name(f".{_MANIFEST_META_FILENAME}.tmp")
    tmp_meta.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(str(tmp_meta), str(meta_path))
    return epoch


def _save_row_groups(root: Path, row_groups: Sequence[ManifestRowGroup]) -> Path:
    """把 row-group 级统计持久化到 ``{root}/_manifest_rowgroups.parquet``。"""
    root = Path(root)
    rows = [
        {
            "path": rg.path,
            "row_group": rg.row_group,
            "column": rg.column,
            "min": _norm_value(rg.min),
            "max": _norm_value(rg.max),
            "null_count": rg.null_count,
            "rows": rg.rows,
        }
        for rg in row_groups
    ]
    table = pa.table(rows)
    out = root / _ROW_GROUPS_FILENAME
    tmp = root / f".{_ROW_GROUPS_FILENAME}.tmp"
    pq.write_table(table, tmp)
    os.replace(str(tmp), str(out))
    return out


def _load_row_groups(root: Path) -> tuple[ManifestRowGroup, ...] | None:
    """读取持久化的 row-group 级统计；无则返回 None。"""
    path = Path(root) / _ROW_GROUPS_FILENAME
    if not path.exists():
        return None
    try:
        table = pq.read_table(str(path))
    except Exception:
        return None
    data = table.to_pydict()
    out: list[ManifestRowGroup] = []
    paths = data.get("path", [])
    for i in range(len(paths)):
        out.append(
            ManifestRowGroup(
                path=str(paths[i]),
                row_group=int(_at(data, "row_group", i) or 0),
                column=str(_at(data, "column", i) or ""),
                min=_at(data, "min", i),
                max=_at(data, "max", i),
                null_count=_int_or_none(_at(data, "null_count", i)),
                rows=_int_or_none(_at(data, "rows", i)),
            )
        )
    return tuple(out) if out else None


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
    time_dtype = None
    if time_col is not None:
        td = str((ds.schema or {}).get(time_col, "")).lower()
        if td:
            time_dtype = "timestamp" if ("timestamp" in td or "datetime" in td) else td
    manifest = DatasetManifest(
        dataset=dataset,
        time_column=time_col,
        instrument_column=inst_col,
        format=str(getattr(ds, "format", "parquet")),
        files=tuple(file_rows),
        row_groups=tuple(row_groups) if row_groups else None,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        time_dtype=time_dtype,
    )
    manifest.save(root)
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
