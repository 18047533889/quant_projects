# -*- coding: utf-8 -*-
"""R39 PERF-055/056 —— ``FactorBlockLake`` 布局 + ``LakeWriteLayoutPolicy``。

§16 引入 block 物理布局，替换“每因子一个 parquet 文件”的逻辑：
逻辑 API 仍是 ``factor_id → Series``，物理写改为：

    factor_blocks/
      date_bucket=2026-08/
        block=<id>/
          data.parquet

两种候选布局（由真实读取模式决定，基准脚本在 scripts/storage_tuning_benchmark.py）：

- **Layout A（long block）**：``datetime, instrument, factor_id, value, generation``
- **Layout B（column block）**：``datetime, instrument, factor_0001..factor_0256``

目录/布局为 ``opt-in`` 基础设施，本轮**不接线**默认 materialize 写路径
（storage-format agent 负责接线）。

设计要点：
- ``FactorBlockLakeWriter``：按 layout 缓冲因子，Layout B 每 256 因子落一个
  宽表 block，Layout A 每 256 因子落一个长表 block；``close()`` 冲刷剩余并
  写 ``manifest.json``（factor_id → block/column/generation/date_bucket）。
- ``FactorBlockLakeReader.read_factor``：block 级读取——Layout B 列裁剪只读
  该因子列，Layout A 用 ``factor_id`` 谓词下推；``time_range`` 按
  date_bucket 分区裁剪，``instrument_filter`` 走 row-group 谓词。
- ``LakeWriteLayoutPolicy.choose_layout``：纯函数，按
  factor_count / update_frequency / read_pattern / matrix_demand / storage_class
  选择 SINGLE_FACTOR_DELTA / FACTOR_BLOCK_LONG / FACTOR_BLOCK_WIDE /
  MATRIX_COLUMN_BLOCK。
"""

from __future__ import annotations

import json
from datetime import datetime as _dt
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .parquet_batch_writer import BatchParquetWriter

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: Layout B 每个 column block 的因子列数（factor_0001..factor_0256）。
BLOCK_COLUMN_COUNT = 256
#: Layout A 每个 long block 的因子数。
BLOCK_FACTOR_COUNT = 256

LAYOUT_A_COLUMNS = ("datetime", "instrument", "factor_id", "value", "generation")
LAYOUT_B_PREFIX = "factor_"

# 布局策略枚举
SINGLE_FACTOR_DELTA = "SINGLE_FACTOR_DELTA"
FACTOR_BLOCK_LONG = "FACTOR_BLOCK_LONG"
FACTOR_BLOCK_WIDE = "FACTOR_BLOCK_WIDE"
MATRIX_COLUMN_BLOCK = "MATRIX_COLUMN_BLOCK"

ALL_LAYOUTS = (
    SINGLE_FACTOR_DELTA,
    FACTOR_BLOCK_LONG,
    FACTOR_BLOCK_WIDE,
    MATRIX_COLUMN_BLOCK,
)

# Layout A/B 的物理实现名（写盘时用）
_PHYSICAL_LONG = "LONG"
_PHYSICAL_WIDE = "WIDE"

_MANIFEST_FILE = "manifest.json"


# ---------------------------------------------------------------------------
# 布局策略（PERF-056）
# ---------------------------------------------------------------------------


class LakeWriteLayoutPolicy:
    """根据写入/读取特征选择 block 布局的纯函数策略（§16）。

    规则（确定性、可测试）：
      1. matrix_demand 为 high / read_pattern 为 matrix → MATRIX_COLUMN_BLOCK
         （矩阵训练要一整块宽表，避免逐因子外连接）。
      2. 高因子数 + object store 或列式读取 → FACTOR_BLOCK_WIDE
         （对象存储喜欢少而大的文件；宽表一块 256 因子 = 1 文件）。
      3. 中高因子数或中高频更新 → FACTOR_BLOCK_LONG
         （长块便于增量追加、单因子谓词下推）。
      4. 单因子 + 低频 + 点读 → SINGLE_FACTOR_DELTA（每个因子独立小文件）。
    """

    _HIGH = {"high", "matrix"}
    _MED = {"medium"}

    @staticmethod
    def _norm_frequency(value) -> str:
        if isinstance(value, (int, float)):
            if value >= 10:
                return "high"
            if value >= 1:
                return "medium"
            return "low"
        s = str(value).lower()
        if s in ("h", "frequent", "high"):
            return "high"
        if s in ("m", "medium"):
            return "medium"
        return "low"

    @classmethod
    def _norm_matrix_demand(cls, value) -> str:
        if isinstance(value, bool):
            return "high" if value else "low"
        s = str(value).lower()
        if s in ("high", "yes", "true", "1", "matrix"):
            return "high"
        if s in ("medium", "mid"):
            return "medium"
        return "low"

    @classmethod
    def _read_pattern(cls, value) -> str:
        s = str(value).lower()
        if "," in s or " " in s:
            tokens = {w.strip() for w in s.replace(",", " ").split()}
            if "matrix" in tokens:
                return "matrix"
            if "column" in tokens:
                return "column"
            if "point" in tokens:
                return "point"
            return "mixed"
        return s

    @classmethod
    def choose_layout(
        cls,
        factor_count: int,
        update_frequency="medium",
        read_pattern="mixed",
        matrix_demand=False,
        storage_class="local",
    ) -> str:
        """返回四种布局之一（纯函数，无副作用）。

        参数:
            factor_count: 因子数
            update_frequency: "low"/"medium"/"high" 或每周期更新次数（数值）
            read_pattern: "point"/"column"/"matrix"/"mixed"（可含多个，空格或逗号分隔）
            matrix_demand: bool 或 "low"/"medium"/"high"
            storage_class: "local" 或 "object"
        """
        factor_count = int(factor_count)
        freq = cls._norm_frequency(update_frequency)
        pattern = cls._read_pattern(read_pattern)
        matrix = cls._norm_matrix_demand(matrix_demand)
        storage = str(storage_class).lower()

        if matrix == "high" or pattern == "matrix":
            return MATRIX_COLUMN_BLOCK
        if factor_count >= 64 and (storage == "object" or pattern == "column"):
            return FACTOR_BLOCK_WIDE
        if matrix == "medium" and factor_count >= 16:
            return MATRIX_COLUMN_BLOCK
        if factor_count >= 8 or freq in ("high", "medium"):
            return FACTOR_BLOCK_LONG
        return SINGLE_FACTOR_DELTA


# ---------------------------------------------------------------------------
# 日期分桶
# ---------------------------------------------------------------------------


def _bucket_of(ts) -> str:
    return ts.strftime("%Y-%m")


def _buckets_between(start, end) -> list[str]:
    """返回 [start, end] 覆盖的 YYYY-MM 桶列表（含端点）。"""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start > end:
        return []
    out = []
    cur = start.normalize().replace(day=1)
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        cur += pd.DateOffset(months=1)
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class FactorBlockLakeWriter:
    """把因子按 layout 写入 block lake。

    API：``write_factor`` / ``write_factors`` 缓冲 → ``close()`` 冲刷全部 block
    并写 manifest。也可用 ``with FactorBlockLakeWriter(...) as w`` 上下文。

    - layout="FACTOR_BLOCK_WIDE"/"MATRIX_COLUMN_BLOCK" → Layout B（column block）
    - layout="FACTOR_BLOCK_LONG" → Layout A（long block，多因子一 block）
    - layout="SINGLE_FACTOR_DELTA" → Layout A（每因子独立 block）
    """

    def __init__(
        self,
        root,
        *,
        layout: str = FACTOR_BLOCK_WIDE,
        generation: int | str = 1,
        compression: str = "SNAPPY",
        block_column_count: int = BLOCK_COLUMN_COUNT,
        block_factor_count: int = BLOCK_FACTOR_COUNT,
        row_group_size: int | None = None,
        compress_row_group_bytes: int | None = None,
        stats: bool = True,
        dictionary_encode_cols: Sequence[str] = ("instrument",),
    ) -> None:
        layout = str(layout).upper()
        if layout not in ALL_LAYOUTS:
            raise ValueError(f"未知 layout: {layout!r}，可选 {ALL_LAYOUTS}")
        self.layout = layout
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._physical = _PHYSICAL_LONG if layout in (
            SINGLE_FACTOR_DELTA, FACTOR_BLOCK_LONG,
        ) else _PHYSICAL_WIDE
        self._generation = generation
        self._compression = compression
        self._block_column_count = int(block_column_count)
        self._block_factor_count = int(block_factor_count)
        self._row_group_size = row_group_size
        self._compress_row_group_bytes = compress_row_group_bytes
        self._stats = stats
        self._dict_cols = tuple(dictionary_encode_cols)

        self._pending: dict[str, pd.DataFrame] = {}
        self._gen: dict[str, object] = {}
        self._manifest: dict[str, list[dict]] = {}
        self._next_block = 1
        self._closed = False

    # -- 数据录入 -----------------------------------------------------------

    @staticmethod
    def _normalize_factor_frame(data) -> pd.DataFrame:
        """把 Series(MultiIndex) / DataFrame 归一化为
        ``columns=[datetime, instrument, value]`` 的 DataFrame。"""
        if isinstance(data, pd.Series):
            if isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 2:
                frame = data.reset_index()
                frame.columns = ["datetime", "instrument", "value"]
            else:
                raise ValueError("Series 索引必须是 MultiIndex(datetime, instrument)")
        elif isinstance(data, pd.DataFrame):
            cols = {str(c).lower() for c in data.columns}
            if {"datetime", "instrument", "value"} <= cols:
                frame = data.rename(
                    columns={
                        c: c.lower()
                        for c in data.columns
                        if str(c).lower() in ("datetime", "instrument", "value")
                    }
                )[["datetime", "instrument", "value"]].copy()
            else:
                raise ValueError("DataFrame 必须含 datetime/instrument/value 列")
        elif isinstance(data, pa.Table):
            names = [str(c).lower() for c in data.column_names]
            if {"datetime", "instrument", "value"} <= set(names):
                frame = data.select(
                    [data.column_names[i] for i, n in enumerate(names) if n in ("datetime", "instrument", "value")]
                ).to_pandas()
                frame.columns = ["datetime", "instrument", "value"]
            else:
                raise ValueError("Arrow Table 必须含 datetime/instrument/value 列")
        else:
            raise TypeError(f"不支持的数据类型: {type(data).__name__}")

        out = pd.DataFrame(
            {
                "datetime": pd.to_datetime(frame["datetime"]),
                "instrument": frame["instrument"].astype(str),
                "value": pd.to_numeric(frame["value"], errors="coerce").astype("float64"),
            }
        )
        return out.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    def write_factor(self, factor_id: str, data, generation=None) -> None:
        """写入单个因子（缓冲；Layout B 每 block_column_count 个自动落 block）。"""
        if self._closed:
            raise RuntimeError("writer 已 close")
        factor_id = str(factor_id)
        if factor_id in self._pending:
            raise ValueError(f"重复写入因子: {factor_id}")
        frame = self._normalize_factor_frame(data)
        if frame.empty:
            raise ValueError(f"因子 {factor_id} 无数据")
        gen = self._generation if generation is None else generation
        self._pending[factor_id] = frame
        self._gen[factor_id] = gen

        if self._physical == _PHYSICAL_WIDE:
            if len(self._pending) >= self._block_column_count:
                self._flush_block()
        elif self.layout == FACTOR_BLOCK_LONG:
            if len(self._pending) >= self._block_factor_count:
                self._flush_block()
        # SINGLE_FACTOR_DELTA：每因子独立 block，立即落。
        if self.layout == SINGLE_FACTOR_DELTA and factor_id in self._pending:
            self._flush_block()

    def write_factors(self, mapping: Mapping[str, object], generation=None) -> None:
        """批量写入 {factor_id: Series/DataFrame/ArrowTable}。"""
        for factor_id, data in mapping.items():
            self.write_factor(factor_id, data, generation=generation)

    # -- block 落盘 ---------------------------------------------------------

    def _flush_block(self) -> None:
        if not self._pending:
            return
        block_id = f"{self._next_block:04d}"
        bucket = _bucket_of(
            min(f["datetime"].iloc[0] for f in self._pending.values())
        )
        block_dir = self.root / f"date_bucket={bucket}" / f"block={block_id}"
        block_dir.mkdir(parents=True, exist_ok=True)
        path = block_dir / "data.parquet"

        if self._physical == _PHYSICAL_WIDE:
            table = self._build_wide_table()
            BatchParquetWriter.write_batches(
                table,
                table.schema,
                str(path),
                compression=self._compression,
                stats=self._stats,
                dictionary_encode_cols=self._dict_cols,
                row_group_size=self._row_group_size,
                compress_row_group_bytes=self._compress_row_group_bytes,
            )
        else:
            table = self._build_long_table()
            BatchParquetWriter.write_batches(
                table,
                table.schema,
                str(path),
                compression=self._compression,
                stats=self._stats,
                dictionary_encode_cols=("instrument", "factor_id"),
                row_group_size=self._row_group_size,
                compress_row_group_bytes=self._compress_row_group_bytes,
            )

        # 登记 manifest（每因子一条 entry 指向本 block）。
        order = list(self._pending)
        for fid in order:
            frame = self._pending[fid]
            self._manifest.setdefault(fid, []).append(
                {
                    "block": block_id,
                    "date_bucket": bucket,
                    "date_bucket_end": _bucket_of(frame["datetime"].iloc[-1]),
                    "column": (
                        f"{LAYOUT_B_PREFIX}{order.index(fid) + 1:04d}"
                        if self._physical == _PHYSICAL_WIDE
                        else None
                    ),
                    "layout": self._physical,
                    "generation": self._gen.get(fid, self._generation),
                    "rows": int(len(frame)),
                }
            )
        self._next_block += 1
        self._pending = {}
        self._gen = {}

    def _build_wide_table(self) -> pa.Table:
        frames = {
            fid: df.set_index(["datetime", "instrument"])["value"]
            for fid, df in self._pending.items()
        }
        wide = pd.concat(frames, axis=1)  # 单次 concat（非 pairwise merge）
        wide.columns = [
            f"{LAYOUT_B_PREFIX}{i + 1:04d}" for i in range(wide.shape[1])
        ]
        wide = wide.reset_index()  # datetime, instrument, factor_*
        fields = [
            pa.field("datetime", pa.timestamp("us")),
            pa.field("instrument", pa.string()),
        ]
        for c in wide.columns[2:]:
            fields.append(pa.field(str(c), pa.float64()))
        schema = pa.schema(fields)
        table = pa.Table.from_pandas(wide, schema=schema, preserve_index=False)
        return table

    def _build_long_table(self) -> pa.Table:
        chunks = []
        for fid, df in self._pending.items():
            chunk = df.copy()
            chunk["factor_id"] = fid
            chunk["generation"] = str(self._gen.get(fid, self._generation))
            chunks.append(chunk)
        long = pd.concat(chunks, ignore_index=True)
        schema = pa.schema(
            [
                pa.field("datetime", pa.timestamp("us")),
                pa.field("instrument", pa.string()),
                pa.field("factor_id", pa.string()),
                pa.field("value", pa.float64()),
                pa.field("generation", pa.string()),
            ]
        )
        table = pa.Table.from_pandas(long, schema=schema, preserve_index=False)
        return table

    # -- 收尾 ---------------------------------------------------------------

    def close(self) -> dict:
        """冲刷剩余数据、写 manifest.json，返回完整 manifest。"""
        if self._closed:
            return dict(self._manifest)
        self._flush_block()
        payload = {
            "schema_version": 1,
            "generated_at": _dt.now().astimezone().isoformat(timespec="seconds"),
            "layout": self.layout,
            "physical": self._physical,
            "factors": self._manifest,
        }
        (self.root / _MANIFEST_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._closed = True
        return dict(self._manifest)

    def __enter__(self) -> "FactorBlockLakeWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class FactorBlockLakeReader:
    """从 block lake 读取因子。

    ``read_factor(factor_id, time_range=None, instrument_filter=None)`` 返回
    MultiIndex(datetime, instrument) Series。

    - Layout B：列裁剪只读 ``datetime/instrument/<该因子列>``；
    - Layout A：``factor_id`` 谓词下推 + 列裁剪；
    - ``time_range``：先按 manifest 里的 date_bucket 分区裁剪，再做 row-group
      谓词过滤；
    - ``instrument_filter``：row-group 谓词过滤。
    """

    def __init__(self, root) -> None:
        self.root = Path(root)
        manifest_path = self.root / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"缺少 manifest: {manifest_path}")
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._factors = self._manifest.get("factors", {})

    # -- 元数据 -------------------------------------------------------------

    def factor_ids(self) -> list[str]:
        return sorted(self._factors)

    def manifest_entry(self, factor_id: str) -> list[dict]:
        if factor_id not in self._factors:
            raise KeyError(f"未知因子: {factor_id!r}")
        return self._factors[factor_id]

    def block_path(self, entry: dict) -> Path:
        return (
            self.root
            / f"date_bucket={entry['date_bucket']}"
            / f"block={entry['block']}"
            / "data.parquet"
        )

    # -- 读取 ---------------------------------------------------------------

    def _partition_entries(self, factor_id: str, time_range=None) -> list[dict]:
        """按 date_bucket 分区裁剪：只打开与查询区间相交的 block。

        block 可能覆盖多个月份（``date_bucket``..``date_bucket_end``），
        只要两个月份区间相交就保留（行级过滤仍保证精确）。
        """
        entries = self.manifest_entry(factor_id)
        if not time_range:
            return entries
        start, end = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
        qb = _buckets_between(start, end)
        if not qb:
            return []
        qb_start, qb_end = qb[0], qb[-1]
        return [
            e
            for e in entries
            if e.get("date_bucket_end", e.get("date_bucket") >= qb_start
            and e["date_bucket"] <= qb_end
        ]

    def read_factor(
        self,
        factor_id: str,
        time_range=None,
        instrument_filter=None,
    ) -> pd.Series:
        """读取因子为 MultiIndex(datetime, instrument) Series。"""
        entries = self._partition_entries(factor_id, time_range)
        if not entries:
            return pd.Series(dtype="float64")

        parts: list[pd.DataFrame] = []
        filters = self._row_filters(
            time_range=time_range, instrument_filter=instrument_filter
        )
        for entry in entries:
            layout = entry.get("layout")
            if layout == _PHYSICAL_WIDE:
                if not entry.get("column"):
                    raise RuntimeError(f"Layout B manifest 缺 column: {entry}")
                columns = ["datetime", "instrument", entry["column"]]
                table = pq.read_table(
                    str(self.block_path(entry)), columns=columns, filters=filters
                )
                frame = table.to_pandas()
                frame = frame.rename(columns={entry["column"]: "value"})
            else:
                columns = ["datetime", "instrument", "value"]
                row_filters = [("factor_id", "=", factor_id)]
                if filters:
                    row_filters = row_filters + list(filters)
                table = pq.read_table(
                    str(self.block_path(entry)), columns=columns, filters=row_filters
                )
                frame = table.to_pandas()
            if not frame.empty:
                frame["datetime"] = pd.to_datetime(frame["datetime"])
                frame["instrument"] = frame["instrument"].astype(str)
                parts.append(frame[["datetime", "instrument", "value"]])

        if not parts:
            return pd.Series(dtype="float64")
        out = pd.concat(parts, ignore_index=True)
        if time_range is not None:
            start, end = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
            out = out[(out["datetime"] >= start) & (out["datetime"] <= end)]
        if instrument_filter is not None:
            insts = [instrument_filter] if isinstance(instrument_filter, str) else list(instrument_filter)
            out = out[out["instrument"].isin(insts)]
        out = out.sort_values(["datetime", "instrument"]).drop_duplicates(
            ["datetime", "instrument"], keep="last"
        )
        return out.set_index(["datetime", "instrument"])["value"]

    @staticmethod
    def _row_filters(time_range=None, instrument_filter=None):
        filters = []
        if time_range is not None:
            start, end = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
            filters.append(("datetime", ">=", start.to_pydatetime()))
            filters.append(("datetime", "<=", end.to_pydatetime()))
        if instrument_filter is not None:
            insts = [instrument_filter] if isinstance(instrument_filter, str) else list(instrument_filter)
            filters.append(("instrument", "in", list(insts)))
        return filters or None


__all__ = [
    "ALL_LAYOUTS",
    "BLOCK_COLUMN_COUNT",
    "BLOCK_FACTOR_COUNT",
    "FACTOR_BLOCK_LONG",
    "FACTOR_BLOCK_WIDE",
    "FactorBlockLakeReader",
    "FactorBlockLakeWriter",
    "LakeWriteLayoutPolicy",
    "MATRIX_COLUMN_BLOCK",
    "SINGLE_FACTOR_DELTA",
    "_bucket_of",
    "_buckets_between",
]
