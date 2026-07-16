"""数据集轻量统计：parquet footer 行数估算 + 读路径路由建议 + sidecar 持久化。"""

from __future__ import annotations

import glob as glob_module
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
    """数据集 sidecar 统计（持久化）。"""

    dataset: str
    num_rows: int
    num_files: int
    partition_columns: tuple[str, ...]
    min_time: str | None = None
    max_time: str | None = None
    instruments: int | None = None
    #: 列名 → 非空率（0~1）；由采样 parquet 估算
    column_null_ratio: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset": self.dataset,
            "num_rows": self.num_rows,
            "num_files": self.num_files,
            "partition_columns": list(self.partition_columns),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "instruments": self.instruments,
        }
        if self.column_null_ratio:
            payload["column_null_ratio"] = dict(self.column_null_ratio)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetStatsSnapshot":
        raw_ratios = payload.get("column_null_ratio")
        ratios: dict[str, float] | None = None
        if isinstance(raw_ratios, dict) and raw_ratios:
            ratios = {str(k): float(v) for k, v in raw_ratios.items()}
        return cls(
            dataset=str(payload.get("dataset", "")),
            num_rows=int(payload.get("num_rows", 0)),
            num_files=int(payload.get("num_files", 0)),
            partition_columns=tuple(payload.get("partition_columns") or ("year",)),
            min_time=payload.get("min_time"),
            max_time=payload.get("max_time"),
            instruments=payload.get("instruments"),
            column_null_ratio=ratios,
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


def estimate_parquet_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if path.name.startswith(".") or path.is_symlink():
            continue
        meta = pq.read_metadata(str(path))
        total += meta.num_rows
    return total


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
    return DatasetStatsSnapshot.from_dict(payload)


def estimate_column_null_ratios(
    paths: list[Path],
    columns: list[str] | None = None,
    *,
    sample_files: int = 3,
) -> dict[str, float]:
    """采样若干 parquet 文件估算各列非空率（0~1）。"""
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
        non_null = total - null_counts.get(name, 0)
        ratios[name] = non_null / total
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
    return DatasetStatsSnapshot(
        dataset=dataset,
        num_rows=num_rows,
        num_files=len(parquet_files),
        partition_columns=tuple(getattr(ds, "partition_columns", ("year",))),
        min_time=min_time,
        max_time=max_time,
        column_null_ratio=column_null_ratio,
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
