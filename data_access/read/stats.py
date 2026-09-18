"""数据集轻量统计：parquet footer 行数估算 + 读路径路由建议 + sidecar 持久化。"""

from __future__ import annotations

import glob as glob_module
import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from .store import DataAccessStore

ReadMode = Literal["arrow", "stream", "polars"]

_DEFAULT_ARROW_MAX_ROWS = 5_000_000
_DEFAULT_STREAM_MIN_ROWS = 1_000_000
STATS_FILENAME = ".data_access_stats.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class DatasetReadStats:
    dataset: str
    parquet_files: int
    estimated_rows: int
    suggested_mode: ReadMode

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "parquet_files": self.parquet_files,
            "estimated_rows": self.estimated_rows,
            "suggested_mode": self.suggested_mode,
        }


@dataclass(frozen=True)
class DatasetStatsSnapshot:
    """数据集 sidecar 统计（持久化）。

    #P1-71 绑定 source identity：数据更新后旧 sidecar 不能继续参与 CBO /
    read_auto 路由——``source_epoch`` / ``manifest_generation`` 与当前不一致
    时视为 stale。
    """

    dataset: str
    num_rows: int
    num_files: int
    partition_columns: tuple[str, ...]
    min_time: str | None = None
    max_time: str | None = None
    instruments: int | None = None
    #: 列名 → 非空率（0~1）；由采样 parquet 估算
    column_null_ratio: dict[str, float] | None = None
    # #P1-71 source identity
    source_epoch: str | None = None
    manifest_generation: str | None = None
    # #P1-72 versioned schema
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "num_rows": self.num_rows,
            "num_files": self.num_files,
            "partition_columns": list(self.partition_columns),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "instruments": self.instruments,
            "source_epoch": self.source_epoch,
            "manifest_generation": self.manifest_generation,
            "version": self.version,
        }
        if self.column_null_ratio:
            payload["column_null_ratio"] = dict(self.column_null_ratio)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetStatsSnapshot":
        if not isinstance(payload, dict):
            raise ValueError("stats sidecar 必须是 mapping")
        raw_ratios = payload.get("column_null_ratio")
        ratios: dict[str, float] | None = None
        if isinstance(raw_ratios, dict) and raw_ratios:
            ratios = {str(k): float(v) for k, v in raw_ratios.items()}
        # #P1-72 严格 bounds validation：脏数值（num_rows=-100 / null_ratio=2.5）
        # 不能进入 CBO / read_auto 路由。
        try:
            num_rows = int(payload.get("num_rows", 0))
            num_files = int(payload.get("num_files", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stats sidecar num_rows/num_files 必须是整数: {exc}"
            ) from exc
        if num_rows < 0 or num_files < 0:
            raise ValueError(
                f"stats sidecar 非法：num_rows={num_rows} num_files={num_files}（不能为负）"
            )
        if ratios is not None:
            bad = [
                k for k, v in ratios.items()
                if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0)
            ]
            if bad:
                raise ValueError(
                    f"stats sidecar column_null_ratio 越界 {bad}（null 率必须在 [0,1]）"
                )
        return cls(
            dataset=str(payload.get("dataset", "")),
            num_rows=num_rows,
            num_files=num_files,
            # #P2-4 与 registry 默认一致：不再残留 ("year",)，缺省即无分区列。
            partition_columns=tuple(payload.get("partition_columns") or ()),
            min_time=payload.get("min_time"),
            max_time=payload.get("max_time"),
            instruments=payload.get("instruments"),
            column_null_ratio=ratios,
            source_epoch=payload.get("source_epoch"),
            manifest_generation=payload.get("manifest_generation"),
            version=int(payload.get("version", 1)),
        )


def expand_parquet_paths(glob_paths: list[str]) -> list[Path]:
    """把 registry glob 展开为具体 parquet 文件列表（只读 footer，不扫 payload）。"""
    files: list[Path] = []
    seen: set[str] = set()
    for pattern in glob_paths:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matches = glob_module.glob(pattern, recursive=True)
        else:
            p = Path(pattern)
            if p.is_dir():
                matches = [str(x) for x in sorted(p.rglob("*.parquet"))]
            elif p.suffix == ".parquet" and p.exists():
                matches = [str(p)]
            else:
                matches = []
        for match in matches:
            mp = str(Path(match).resolve())
            if not mp.endswith(".parquet") or mp in seen:
                continue
            seen.add(mp)
            files.append(Path(mp))
    return files


def _footer_file_identity(path: str) -> tuple[int, int, int, int, int]:
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=8192)
def _parquet_footer_rows(path: str, identity: tuple[int, int, int, int, int]) -> int:
    """Cache only an integer row count for an unchanged local file generation."""
    rows = int(pq.read_metadata(path).num_rows)
    if _footer_file_identity(path) != identity:
        # Never cache a count under a generation that changed during inspection.
        # The caller's existing unknown-cost policy handles this failed estimate.
        raise RuntimeError(f"Parquet file changed during row-count estimation: {path}")
    return rows


def estimate_parquet_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.name.startswith(".") or path.is_symlink():
            continue
        local_path = str(path.absolute())
        total += _parquet_footer_rows(local_path, _footer_file_identity(local_path))
    return total


@lru_cache(maxsize=8192)
def _parquet_footer_cost_summary(
    path: str, identity: tuple[int, int, int, int, int]
) -> tuple[int, tuple[str, ...], tuple[int, ...]]:
    """Return immutable row/schema/column-byte facts for one file generation.

    ``total_uncompressed_size`` is Parquet page metadata, not an assumed dtype
    width.  It is an encoded-column estimate, not an Arrow/runtime memory bound.
    """
    metadata = pq.read_metadata(path)
    names = tuple(str(metadata.schema.column(i).path) for i in range(metadata.num_columns))
    uncompressed = [0] * len(names)
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            size = int(row_group.column(column_index).total_uncompressed_size or 0)
            if size > 0:
                uncompressed[column_index] += size
    # For fixed-width Arrow values, dictionary/RLE encoded page sizes can be far
    # below the materialized value buffer.  The schema and raw row count prove a
    # stronger metadata-only estimate; retain the larger of the two facts.
    arrow_schema = metadata.schema.to_arrow_schema()
    by_name = {field.name: field.type for field in arrow_schema}
    for index, name in enumerate(names):
        dtype = by_name.get(name)
        width: int | None = None
        if dtype is not None:
            if pa.types.is_fixed_size_binary(dtype):
                width = int(dtype.byte_width)
            elif (
                pa.types.is_integer(dtype)
                or pa.types.is_floating(dtype)
                or pa.types.is_decimal(dtype)
                or pa.types.is_date(dtype)
                or pa.types.is_time(dtype)
                or pa.types.is_timestamp(dtype)
                or pa.types.is_duration(dtype)
            ):
                width = max(1, int(dtype.bit_width) // 8)
            elif pa.types.is_boolean(dtype):
                # Arrow boolean buffers are bit-packed.
                width = 0
        if width is not None:
            value_buffer = (
                (int(metadata.num_rows) + 7) // 8
                if width == 0
                else int(metadata.num_rows) * width
            )
            uncompressed[index] = max(uncompressed[index], value_buffer)
    if _footer_file_identity(path) != identity:
        raise RuntimeError(f"Parquet file changed during cost estimation: {path}")
    return int(metadata.num_rows), names, tuple(uncompressed)


@lru_cache(maxsize=8192)
def _parquet_footer_row_group_summary(
    path: str, identity: tuple[int, int, int, int, int]
) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
    """Immutable row-group facts cached for one unchanged file generation."""
    metadata = pq.read_metadata(path)
    names = tuple(str(metadata.schema.column(i).path) for i in range(metadata.num_columns))
    arrow_schema = metadata.schema.to_arrow_schema()
    by_name = {field.name: field.type for field in arrow_schema}
    groups = []
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        sizes = []
        statistics = []
        for column_index, name in enumerate(names):
            column = row_group.column(column_index)
            size = int(column.total_uncompressed_size or 0)
            dtype = by_name.get(name)
            width = None
            if dtype is not None:
                if pa.types.is_fixed_size_binary(dtype):
                    width = int(dtype.byte_width)
                elif (
                    pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
                    or pa.types.is_decimal(dtype) or pa.types.is_date(dtype)
                    or pa.types.is_time(dtype) or pa.types.is_timestamp(dtype)
                    or pa.types.is_duration(dtype)
                ):
                    width = max(1, int(dtype.bit_width) // 8)
                elif pa.types.is_boolean(dtype):
                    width = 0
            if width is not None:
                fixed = (
                    (int(row_group.num_rows) + 7) // 8
                    if width == 0 else int(row_group.num_rows) * width
                )
                size = max(size, fixed)
            sizes.append(size)
            stats = column.statistics
            statistics.append(
                (
                    bool(stats is not None and stats.has_min_max),
                    None if stats is None or not stats.has_min_max else stats.min,
                    None if stats is None or not stats.has_min_max else stats.max,
                )
            )
        groups.append((
            int(row_group.num_rows),
            sum(
                int(row_group.column(i).total_compressed_size or 0)
                for i in range(row_group.num_columns)
            ),
            tuple(sizes),
            tuple(statistics),
        ))
    if _footer_file_identity(path) != identity:
        raise RuntimeError(f"Parquet file changed during cost estimation: {path}")
    return names, tuple(groups)


def _coerce_footer_bound(value: object, sample: object) -> object | None:
    """Coerce a predicate endpoint only when ordering remains provable."""
    import datetime as dt

    if value is None:
        return None
    if isinstance(sample, bytes):
        # DuckDB BLOB/VARCHAR coercion and ordering are not interchangeable.
        # Only a bytes predicate proves ordering against bytes footer stats.
        return value if isinstance(value, bytes) else None
    if isinstance(sample, str):
        return value if isinstance(value, str) else None
    if isinstance(sample, dt.datetime):
        try:
            parsed = dt.datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        # Naive and timezone-aware values are not order-compatible.
        if (parsed.tzinfo is None) != (sample.tzinfo is None):
            return None
        return parsed
    if isinstance(sample, dt.date):
        try:
            return dt.date.fromisoformat(str(value)[:10])
        except (TypeError, ValueError):
            return None
    if isinstance(sample, bool):
        return value if isinstance(value, bool) else None
    if isinstance(sample, int) and not isinstance(value, bool):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if isinstance(value, float) and not value.is_integer():
            return None
        return parsed
    if isinstance(sample, float) and not isinstance(value, bool):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _coerce_time_footer_bound(value: object, sample: object) -> object | None:
    """Coerce only native temporal footer values with identical SQL ordering."""
    import datetime as dt

    if isinstance(sample, dt.datetime):
        if isinstance(value, dt.datetime):
            parsed = value
        elif isinstance(value, dt.date):
            # DuckDB compares a DATE parameter to a naive TIMESTAMP column by
            # casting the date to midnight.  Do not apply this to timezone-aware
            # footer values because that cast's timezone is not proven here.
            if sample.tzinfo is not None:
                return None
            parsed = dt.datetime.combine(value, dt.time.min)
        elif isinstance(value, str):
            try:
                parsed = dt.datetime.fromisoformat(value)
            except ValueError:
                return None
        else:
            return None
        if (parsed.tzinfo is None) != (sample.tzinfo is None):
            return None
        if parsed.tzinfo is not None and parsed.utcoffset() != sample.utcoffset():
            return None
        return parsed
    if isinstance(sample, dt.date):
        if isinstance(value, dt.datetime):
            return None
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str) and len(value) == 10:
            try:
                return dt.date.fromisoformat(value)
            except ValueError:
                return None
    # String/bytes/numeric time columns may be cast or normalized by the SQL
    # path; lexical/numeric footer ordering is not proof of execution ordering.
    return None


def _row_group_disjoint(
    names: tuple[str, ...],
    statistics: tuple[tuple[object, object, object], ...],
    *,
    time_column: str | None,
    time_range: tuple[object, object] | None,
    instrument_column: str | None,
    instrument_filter: list[str] | tuple[str, ...] | None,
    filters: object,
    time_column_is_timestamp: bool,
) -> bool:
    """Return True only when footer min/max proves a row group cannot match."""
    predicates: list[tuple[str, tuple[object, ...], bool]] = []
    if time_column and time_range is not None:
        predicates.append((time_column, tuple(time_range), True))
    if instrument_column and instrument_filter is not None:
        predicates.append((instrument_column, tuple(instrument_filter), False))
    if isinstance(filters, dict):
        for name, value in filters.items():
            if value is None or isinstance(value, (dict, list, tuple, set)):
                continue
            predicates.append((str(name), (value,), False))
    for name, values, is_range in predicates:
        if name not in names:
            continue
        has_min_max, lower, upper = statistics[names.index(name)]
        if not has_min_max:
            continue
        if lower is None or upper is None:
            continue
        try:
            # Corrupt/inconsistent footer ordering is unknown evidence.  It
            # must never become proof that a row group is disjoint.
            if lower > upper:
                continue
            if is_range:
                start = _coerce_time_footer_bound(values[0], lower)
                end_input = values[1]
                hi_op = "<="
                if end_input is not None:
                    from data_access.read.predicate import expand_end_bound

                    end_input, hi_op = expand_end_bound(
                        end_input,
                        time_column_is_timestamp=time_column_is_timestamp,
                    )
                end = _coerce_time_footer_bound(end_input, upper)
                # Closed execution interval: equality at either endpoint stays.
                if start is not None and start > upper:
                    return True
                if end is not None and (
                    end <= lower if hi_op == "<" else end < lower
                ):
                    return True
            else:
                coerced = tuple(_coerce_footer_bound(value, lower) for value in values)
                if coerced and all(
                    value is not None and (value < lower or value > upper)
                    for value in coerced
                ):
                    return True
        except (TypeError, ValueError, OverflowError):
            # Incomparable or malformed statistics retain the row group.
            continue
    return False


def estimate_parquet_scope_cost_filtered(
    paths: list[Path],
    columns: list[str] | None = None,
    *,
    time_column: str | None = None,
    time_range: tuple[object, object] | None = None,
    instrument_column: str | None = None,
    instrument_filter: list[str] | tuple[str, ...] | None = None,
    filters: object = None,
    time_column_is_timestamp: bool = False,
) -> tuple[int, int | None, int | None, int, int, int, int]:
    """Exact footer-bound work for row groups not disproved by predicates.

    Unknown/missing/incomparable statistics always retain the row group.  The
    returned bytes are aggregate scan/decode work, never a resident-memory or
    RSS bound.  Returns the legacy four fields plus selected row groups/files.
    """
    """Estimate raw rows and projected encoded-column bytes from footer facts.

    Returns ``(rows, projection_bytes, total_columns, projected_columns)``.
    A requested column absent from every footer makes projection bytes unknown;
    callers must not substitute a guessed variable-width value.
    """
    total_rows = 0
    projection_bytes = 0
    all_names: set[str] = set()
    requested = set(columns or ())
    project_all = not columns
    seen_requested: set[str] = set()
    inspected = 0
    selected_rowgroups = 0
    selected_files = 0
    selected_compressed_bytes = 0
    for path in paths:
        from data_access.runtime.prepared_read import current_deadline

        deadline = current_deadline()
        if deadline is not None:
            deadline.check(context="estimate_scan_cost(parquet footer)")
        if path.name.startswith(".") or path.is_symlink():
            continue
        local_path = str(path.absolute())
        identity = _footer_file_identity(local_path)
        names, row_groups = _parquet_footer_row_group_summary(local_path, identity)
        inspected += 1
        all_names.update(names)
        selected = set(names) if project_all else requested.intersection(names)
        seen_requested.update(selected)
        file_selected = False
        for rows, compressed_bytes, sizes, statistics in row_groups:
            if _row_group_disjoint(
                names, statistics,
                time_column=time_column, time_range=time_range,
                instrument_column=instrument_column,
                instrument_filter=instrument_filter, filters=filters,
                time_column_is_timestamp=time_column_is_timestamp,
            ):
                continue
            file_selected = True
            selected_rowgroups += 1
            total_rows += int(rows)
            selected_compressed_bytes += int(compressed_bytes)
            for column_index, name in enumerate(names):
                if name not in selected:
                    continue
                projection_bytes += int(sizes[column_index])
        if file_selected:
            selected_files += 1
        if _footer_file_identity(local_path) != identity:
            raise RuntimeError(f"Parquet file changed during cost estimation: {path}")
    missing = bool(requested - seen_requested)
    projected_columns = len(all_names) if project_all else len(seen_requested)
    projection = None if missing or not inspected else int(projection_bytes)
    return (
        total_rows, projection, (len(all_names) or None), projected_columns,
        selected_rowgroups, selected_files,
        selected_compressed_bytes,
    )


def estimate_parquet_scope_cost(
    paths: list[Path], columns: list[str] | None = None
) -> tuple[int, int | None, int | None, int]:
    """Backward-compatible unfiltered footer work estimate."""
    return estimate_parquet_scope_cost_filtered(paths, columns)[:4]


def infer_time_bounds_from_paths(paths: list[Path]) -> tuple[str | None, str | None]:
    """从 hive 分区路径推断 min/max year（粗粒度）。"""
    years: list[int] = []
    for path in paths:
        for part in path.parts:
            match = re.fullmatch(r"year=(\d{4})", part)
            if match:
                years.append(int(match.group(1)))
    if not years:
        return None, None
    return f"{min(years)}-01-01", f"{max(years)}-12-31"


def suggest_read_mode(
    estimated_rows: int,
    *,
    column_count: int | None = None,
    prefer_polars: bool = False,
    arrow_max_rows: int | None = None,
    stream_min_rows: int | None = None,
) -> ReadMode:
    """根据估算行数选择 read_arrow / read_arrow_stream / scan_polars。"""
    _ = column_count
    arrow_cap = arrow_max_rows if arrow_max_rows is not None else _env_int(
        "DATA_ACCESS_READ_AUTO_ARROW_MAX_ROWS", _DEFAULT_ARROW_MAX_ROWS
    )
    stream_floor = stream_min_rows if stream_min_rows is not None else _env_int(
        "DATA_ACCESS_READ_AUTO_STREAM_MIN_ROWS", _DEFAULT_STREAM_MIN_ROWS
    )
    if prefer_polars and estimated_rows >= stream_floor:
        return "polars"
    if estimated_rows <= arrow_cap:
        return "arrow"
    if estimated_rows >= stream_floor:
        return "stream"
    return "arrow"


def write_stats_sidecar(root: Path, snapshot: DatasetStatsSnapshot) -> Path:
    """写入 ``{root}/.data_access_stats.json``。"""
    root.mkdir(parents=True, exist_ok=True)
    out = root / STATS_FILENAME
    tmp = root / f".{STATS_FILENAME}.tmp"
    tmp.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(out))
    return out


def load_stats_sidecar(root: Path) -> DatasetStatsSnapshot | None:
    path = Path(root) / STATS_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return DatasetStatsSnapshot.from_dict(payload)
    except (ValueError, TypeError):
        # #P1-72 脏 sidecar（越界/坏类型）不能进 CBO → 视为不存在
        return None


def estimate_column_null_ratios(
    paths: list[Path],
    columns: list[str] | None = None,
    *,
    sample_files: int = 3,
) -> dict[str, float]:
    """采样若干 parquet 文件估算各列 **null 率**（null_count/total，0~1）。

    #P1-16 字段名叫 ``column_null_ratio``，之前却存的是非空率——名称与数值含义
    相反会误导 read_auto / CBO。这里真正保存 ``null_count / total``。
    """
    if not paths:
        return {}
    sample = paths[: max(1, int(sample_files))]
    null_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    for path in sample:
        if path.name.startswith("."):
            continue
        table = pq.read_table(str(path), columns=columns if columns else None)
        for name in table.column_names:
            col = table.column(name)
            null_counts[name] = null_counts.get(name, 0) + col.null_count
            row_counts[name] = row_counts.get(name, 0) + len(col)
    ratios: dict[str, float] = {}
    for name, total in row_counts.items():
        if total <= 0:
            ratios[name] = 0.0
            continue
        ratios[name] = null_counts.get(name, 0) / total
    return ratios


def build_dataset_stats_snapshot(
    store: "DataAccessStore",
    dataset: str,
    *,
    time_range: tuple[Any, Any] | None = None,
    with_null_ratio: bool = False,
    null_ratio_sample_files: int = 3,
    **params: Any,
) -> DatasetStatsSnapshot:
    """扫描 parquet footer 构建 sidecar 统计。"""
    ds = store._registry.get(dataset)
    paths = store._prepare_dataset_read(ds, time_range=time_range, params=params)
    parquet_files = expand_parquet_paths(paths)
    num_rows = estimate_parquet_rows(parquet_files)
    min_time, max_time = infer_time_bounds_from_paths(parquet_files)
    column_null_ratio: dict[str, float] | None = None
    if with_null_ratio and parquet_files:
        schema_cols = sorted(getattr(ds, "schema", {}) or {})
        column_null_ratio = estimate_column_null_ratios(
            parquet_files,
            schema_cols or None,
            sample_files=null_ratio_sample_files,
        )
    # #P1-71 记录 source identity：数据 mutation 后 sidecar 自动 stale，
    # read_auto 不再用旧统计做路由。
    source_epoch: str | None = None
    manifest_generation: str | None = None
    try:
        token = store.manifest_version(dataset)
        if isinstance(token, dict):
            source_epoch = token.get("source_epoch") or token.get("manifest_epoch")
            manifest_generation = token.get("manifest_generation_id")
    except Exception:
        pass
    return DatasetStatsSnapshot(
        dataset=dataset,
        num_rows=num_rows,
        num_files=len(parquet_files),
        partition_columns=tuple(getattr(ds, "partition_columns", ())),
        min_time=min_time,
        max_time=max_time,
        column_null_ratio=column_null_ratio,
        source_epoch=source_epoch,
        manifest_generation=manifest_generation,
    )


def dataset_read_stats(
    store: "DataAccessStore",
    dataset: str,
    *,
    columns: list[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    prefer_polars: bool = False,
    sidecar: DatasetStatsSnapshot | None = None,
    **params: Any,
) -> DatasetReadStats:
    """估算数据集行数并给出 read_auto 路由建议。

    若提供 ``sidecar`` 且行数 > 0，优先使用 sidecar 估算（避免重复扫 footer）。
    """
    ds = store._registry.get(dataset)
    if sidecar is not None and sidecar.num_rows > 0:
        estimated_rows = sidecar.num_rows
        parquet_files = sidecar.num_files
    else:
        # 优先用 manifest：避免 glob + 逐文件 footer
        from data_access.read.manifest import load_manifest_for_dataset

        try:
            manifest = load_manifest_for_dataset(store, dataset, **params)
        except Exception:
            manifest = None
        if manifest is not None and manifest.total_rows > 0:
            estimated_rows = manifest.total_rows
            parquet_files = manifest.file_count
        else:
            paths = store._prepare_dataset_read(ds, time_range=time_range, params=params)
            parquet_files_list = expand_parquet_paths(paths)
            parquet_files = len(parquet_files_list)
            estimated_rows = estimate_parquet_rows(parquet_files_list)
    mode = suggest_read_mode(
        estimated_rows,
        column_count=len(columns) if columns else None,
        prefer_polars=prefer_polars,
    )
    return DatasetReadStats(
        dataset=dataset,
        parquet_files=parquet_files,
        estimated_rows=estimated_rows,
        suggested_mode=mode,
    )
