# -*- coding: utf-8 -*-
"""R39-PERF-077：per-column parquet manifest 真实 decoded footprint。

当前 ScanCost 只有 selected_bytes / projection_bytes / estimated_rows / files /
remote / instrument_count，没有 per-column 真实解码占用。本模块提供：

    - :class:`ColumnFootprintStats`：per-column 统计
      ``compressed_bytes / uncompressed_bytes / null_count / dictionary_bytes /
      avg_variable_width``。
    - :func:`produce_column_footprint_stats`：读 pyarrow parquet 元数据
      （row-group column stats）生成上述统计。
    - :func:`read_column_footprint_stats`：reader helper——manifest 带有
      ``column_footprints`` 字段时读取；缺失（旧 manifest / 新字段未写入）时返回
      ``None``（lazy fallback，现有 manifest 不受影响）。

读者（read-wave cluster）直接消费 ``read_column_footprint_stats`` 的结果做真实
footprint 内存打包（R39-PERF-006/010 依赖）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ColumnFootprintStats:
    """per-column parquet 真实解码 footprint。"""

    column: str
    compressed_bytes: int | None = None
    uncompressed_bytes: int | None = None
    null_count: int | None = None
    dictionary_bytes: int | None = None
    avg_variable_width: float | None = None

    @property
    def total_bytes(self) -> int | None:
        if self.uncompressed_bytes is not None:
            return int(self.uncompressed_bytes)
        return int(self.compressed_bytes) if self.compressed_bytes is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "compressed_bytes": self.compressed_bytes,
            "uncompressed_bytes": self.uncompressed_bytes,
            "null_count": self.null_count,
            "dictionary_bytes": self.dictionary_bytes,
            "avg_variable_width": self.avg_variable_width,
        }


def _as_stats(column: str, raw: Any) -> ColumnFootprintStats | None:
    if isinstance(raw, ColumnFootprintStats):
        return raw
    if isinstance(raw, Mapping):
        return ColumnFootprintStats(
            column=column,
            compressed_bytes=_int_or_none(raw.get("compressed_bytes")),
            uncompressed_bytes=_int_or_none(raw.get("uncompressed_bytes")),
            null_count=_int_or_none(raw.get("null_count")),
            dictionary_bytes=_int_or_none(raw.get("dictionary_bytes")),
            avg_variable_width=_float_or_none(raw.get("avg_variable_width")),
        )
    return None


def _int_or_none(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def read_column_footprint_stats(manifest: Any, columns: Sequence[str]) -> dict[str, ColumnFootprintStats] | None:
    """从 manifest 读取 per-column footprint。

    - manifest 有 ``column_footprints``（dict[str, ColumnFootprintStats|dict]）时
      按请求列返回；缺失列跳过。
    - 无该字段（旧 manifest / 未写入）→ 返回 ``None``（lazy fallback）。
    """
    raw = getattr(manifest, "column_footprints", None)
    if not raw:
        return None
    out: dict[str, ColumnFootprintStats] = {}
    for column in columns:
        entry = raw.get(column)
        if entry is None:
            continue
        stats = _as_stats(column, entry)
        if stats is not None:
            out[column] = stats
    return out if out else None


def produce_column_footprint_stats(
    parquet_path: str,
    columns: Sequence[str] | None = None,
) -> dict[str, ColumnFootprintStats]:
    """读 pyarrow parquet 元数据（row-group column stats）生成 per-column footprint。

    对每个 row group 的每个 column chunk 累加：
      - compressed_bytes：``total_compressed_size``
      - uncompressed_bytes：``total_uncompressed_size``
      - null_count：``statistics.null_count``（统计关闭时为 None）
      - dictionary_bytes：``dictionary_page.uncompressed_size``（无字典页为 None/0）
      - avg_variable_width：``uncompressed_bytes / num_values``（每列总体）
    """
    import pyarrow.parquet as pq  # 延迟导入（pyarrow 是可选依赖）

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    requested = list(columns) if columns else list(schema.names)
    stats: dict[str, ColumnFootprintStats] = {}
    for column in requested:
        compressed = 0
        uncompressed = 0
        null_count = 0
        dictionary_bytes = 0
        num_values = 0
        for i in range(parquet_file.metadata.num_row_groups):
            rg = parquet_file.metadata.row_group(i)
            col_idx = None
            for j in range(rg.num_columns):
                if rg.column(j).path_in_schema == column:
                    col_idx = j
                    break
            if col_idx is None:
                continue
            cc = rg.column(col_idx)
            compressed += int(getattr(cc, "total_compressed_size", 0) or 0)
            uncompressed += int(getattr(cc, "total_uncompressed_size", 0) or 0)
            num_values += int(getattr(cc, "num_values", 0) or 0)
            try:
                stats_meta = cc.statistics
                if stats_meta is not None and getattr(stats_meta, "null_count", None) is not None:
                    null_count += int(stats_meta.null_count)
            except Exception:  # noqa: BLE001
                pass
            try:
                dp = cc.dictionary_page
                if dp is not None:
                    dictionary_bytes += int(getattr(dp, "uncompressed_size", 0) or 0)
            except Exception:  # noqa: BLE001
                pass
        avg_width = None
        if uncompressed > 0 and num_values > 0:
            avg_width = round(uncompressed / num_values, 3)
        stats[column] = ColumnFootprintStats(
            column=column,
            compressed_bytes=compressed,
            uncompressed_bytes=uncompressed,
            null_count=null_count,
            dictionary_bytes=dictionary_bytes,
            avg_variable_width=avg_width,
        )
    return stats


def attach_column_footprints(manifest: Any, stats: Mapping[str, ColumnFootprintStats]) -> None:
    """把 per-column footprint 挂到 manifest 对象上（原地，读者直接消费）。

    data_access 的 ``DatasetManifest`` 是 frozen dataclass——本函数通过
    ``object.__setattr__`` 附加 ``column_footprints`` 字段，供
    :func:`read_column_footprint_stats` 读取。旧 manifest 无此字段 → reader 返回
    ``None``。
    """
    object.__setattr__(manifest, "column_footprints", dict(stats))
