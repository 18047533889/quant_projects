# -*- coding: utf-8 -*-
"""P11+P12: FeatureBlock —— 100k 因子不能是 100k 文件。

A-share ~5000 只股票。100k 因子 × 5000 × float32 ≈ 2GB/日，~2500 交易日 ≈ 5TB
全历史。因此**全历史永久物化所有 100k 因子不可行**，且物理落盘必须分组，绝不
逐因子一文件。

物理布局（date/time partition × factor block × 全 universe）：:

    base_dir/
      <partition>/block_<i:04d>.npy   值（float32，rows×ncols，ncols<=512）
      manifest.json                  元数据（与值分离）

manifest 建立 ``FactorID -> BlockID -> Column`` 映射，reader 据此回读。块大小由
row_count / writer 内存 / 读模式动态选择，绝不固定为 100k 文件。多个因子共享一个
值文件（同一 block），每因子一列，由 manifest 的 column_map 定位。

dtype 契约：因子值一律 ``float32``（``FACTOR_VALUE_DTYPE``）。写入强制 cast 到
float32；回读返回 float32 ndarray。元数据（manifest）单独存放，与值文件分离。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

# ---- dtype contract ---------------------------------------------------------
FACTOR_VALUE_DTYPE = "float32"  # 显式 dtype 契约：因子值一律 float32
_BYTES_PER_CELL = np.dtype(FACTOR_VALUE_DTYPE).itemsize  # 4

# ---- 块大小动态选择边界 -------------------------------------------------------
MIN_COLUMNS_PER_BLOCK = 64
MAX_COLUMNS_PER_BLOCK = 512
_DEFAULT_WRITER_MEMORY_BYTES = 1 << 30  # 1GiB 缺省 writer 内存预算
_HEADROOM_FACTOR = 4  # 保守系数：把 block 字节上限压到预算的 1/4 以下
_ROWS_PER_GROWTH_STEP = 100  # 每行数增量 → 每块多一列的增速

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
VALUES_SUFFIX = ".npy"


def choose_columns_per_block(
    total_factors: int | None = None,
    row_count: int | None = None,
    writer_memory_bytes: int | None = None,
    *,
    min_cols: int = MIN_COLUMNS_PER_BLOCK,
    max_cols: int = MAX_COLUMNS_PER_BLOCK,
) -> int:
    """动态块大小（列数）：row_count / writer 内存 / 读模式共同决定。

    两条相反的作用力：
      * 行驱动增长 —— 行越多单块列越多（分摊 axis/header，读模式偏好大块）；
      * 内存上限 —— block 值字节 = rows×cols×4 必须 <= writer 内存预算，行太多
        时压回小块，绝不超预算。

    最终值夹在 ``[min_cols, max_cols]``。纯函数，便于单测。
    """
    max_cols = max(min_cols, int(max_cols))
    rows = max(1, int(row_count if row_count is not None else (total_factors or 1)))
    budget = max(1, int(writer_memory_bytes or _DEFAULT_WRITER_MEMORY_BYTES))

    # 内存上限（硬约束）：cols <= budget / (rows × bytes_per_cell)，留 1/4 headroom。
    # 内存逼到 min_cols 以下时以内存为准（绝不让单 block 字节超预算）。
    mem_capped = max(1, budget // max(1, rows * _BYTES_PER_CELL * _HEADROOM_FACTOR))
    # 行驱动增长（软下限）：行越多 → 单块列越多（读模式偏好整块读取）。
    row_based = min(max_cols, max(min_cols, min_cols + rows // _ROWS_PER_GROWTH_STEP))

    cols = min(max_cols, mem_capped, row_based)
    return max(1, cols)


def block_bytes(row_count: int, columns: int, dtype: str = FACTOR_VALUE_DTYPE) -> int:
    """一个 block 值文件的字节估算（float32 契约）。"""
    return max(1, int(row_count)) * max(1, int(columns)) * np.dtype(dtype).itemsize


# ---- manifest 结构 ----------------------------------------------------------
@dataclass
class BlockSpec:
    """一个物理 block 的元数据（与值文件分离存放）。"""

    block_id: str
    partition: str
    file: str  # 相对 base_dir 的值文件路径
    row_count: int
    factor_ids: list[str]
    column_map: dict[str, int]  # FactorID -> 列号（该 block 内）

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "partition": self.partition,
            "file": self.file,
            "row_count": self.row_count,
            "factor_ids": list(self.factor_ids),
            "column_map": dict(self.column_map),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockSpec":
        return cls(
            block_id=d["block_id"],
            partition=d["partition"],
            file=d["file"],
            row_count=int(d["row_count"]),
            factor_ids=list(d["factor_ids"]),
            column_map={k: int(v) for k, v in d["column_map"].items()},
        )


@dataclass
class BlockManifest:
    """根 manifest：把 FactorID -> BlockID -> Column 全图映射。"""

    schema_version: int
    dtype: str
    blocks: dict[str, BlockSpec]  # block_id -> spec
    factor_to_block: dict[str, dict[str, str]]  # FactorID -> {partition: block_id}
    partition_rows: dict[str, int]  # partition -> 行数
    columns_per_block: int | None = None

    def resolve(self, factor_id: str) -> list[tuple[str, BlockSpec, int]]:
        """FactorID -> [(partition, BlockSpec, column), ...]（跨全部分区）。"""
        out: list[tuple[str, BlockSpec, int]] = []
        per_partition = self.factor_to_block.get(factor_id)
        if not per_partition:
            return out
        for partition, block_id in per_partition.items():
            block = self.blocks.get(block_id)
            if block is None:
                continue
            column = block.column_map.get(factor_id)
            if column is None:
                continue
            out.append((partition, block, column))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dtype": self.dtype,
            "columns_per_block": self.columns_per_block,
            "partition_rows": dict(self.partition_rows),
            "factor_to_block": {
                k: dict(v) for k, v in self.factor_to_block.items()
            },
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlockManifest":
        return cls(
            schema_version=int(d["schema_version"]),
            dtype=str(d["dtype"]),
            columns_per_block=d.get("columns_per_block"),
            partition_rows=dict(d.get("partition_rows", {})),
            factor_to_block={
                k: dict(v) for k, v in d.get("factor_to_block", {}).items()
            },
            blocks={
                k: BlockSpec.from_dict(v) for k, v in d.get("blocks", {}).items()
            },
        )


# ---- writer -----------------------------------------------------------------
def _partition_token(partition: str) -> str:
    """把 partition 压成短 token 用于 block_id（同 partition 内块连续编号）。"""
    import hashlib

    return hashlib.blake2b(str(partition).encode("utf-8"), digest_size=3).hexdigest()


class FeatureBlockWriter:
    """物理 writer：把多个因子分组进一个 block 值文件，绝不逐因子一文件。

    每个 partition 缓冲因子值；缓冲满 ``columns_per_block`` 个即 flush 成一个
    ``(rows, ncols)`` float32 block 文件，manifest 记录每因子 → block → 列。
    ``finish()`` 落残余块 + 写根 manifest。元数据（manifest）与值分离。

    块大小由 ``choose_columns_per_block`` 动态决定（row_count / writer 内存 /
    读模式），并遵守 writer 内存上限 —— 不固定为 100k 文件。
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        dtype: str = FACTOR_VALUE_DTYPE,
        columns_per_block: int | None = None,
        writer_memory_bytes: int | None = None,
        default_partition: str = "default",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.dtype = str(dtype or FACTOR_VALUE_DTYPE)
        self.default_partition = str(default_partition)
        self.columns_per_block = columns_per_block
        self.writer_memory_bytes = writer_memory_bytes
        # partition -> {factor_id: 1D float32 ndarray}
        self._buffer: dict[str, dict[str, np.ndarray]] = {}
        self._order: dict[str, list[str]] = {}  # partition -> 插入顺序 factor_id
        self._partition_rows: dict[str, int] = {}
        self._blocks: dict[str, BlockSpec] = {}
        self._factor_to_block: dict[str, dict[str, str]] = {}
        self._block_seq = 0
        self._manifest: BlockManifest | None = None
        self._finished = False

    # -- 状态 ---------------------------------------------------------------
    def buffered_factors(self) -> int:
        return sum(len(v) for v in self._buffer.values())

    def block_count(self) -> int:
        return len(self._blocks)

    # -- 主写入 -------------------------------------------------------------
    def add(self, factor_id: str, values: Any, partition: str | None = None) -> None:
        """把一个因子的一段值（1D float32）加入缓冲，满了即 flush 成整块。"""
        if self._finished:
            raise RuntimeError("FeatureBlockWriter.finish() 已调用，不能再 add")
        partition = str(partition) if partition is not None else self.default_partition
        arr = self._coerce_values(values, factor_id)
        part_buf = self._buffer.setdefault(partition, {})
        part_order = self._order.setdefault(partition, [])
        if factor_id not in part_buf:
            part_order.append(factor_id)
        part_buf[factor_id] = arr
        self._partition_rows[partition] = int(arr.shape[0])
        target = self._resolve_columns_per_block(
            row_count=int(arr.shape[0]), factor_total=len(part_buf)
        )
        if len(part_buf) >= target:
            self.flush_partition(partition)

    def write_results(self, items: Iterable[Any]) -> None:
        """StreamingResultSink / writer 路径接线：item.name=factor_id，
        item.value=1D 值，item.meta["partition"] 可选覆盖。"""
        for item in items:
            name = getattr(item, "name", None)
            value = getattr(item, "value", None)
            meta = getattr(item, "meta", None) or {}
            if name is None or value is None:
                continue
            self.add(str(name), value, partition=meta.get("partition"))

    def flush_partition(self, partition: str) -> None:
        """把某个 partition 的缓冲落成一个（或多个）block 值文件。"""
        part_buf = self._buffer.get(partition)
        if not part_buf:
            return
        part_order = self._order.get(partition, list(part_buf.keys()))
        fids = [f for f in part_order if f in part_buf]
        if not fids:
            return
        rows = int(part_buf[fids[0]].shape[0])
        self._partition_rows[partition] = rows
        cols = self._resolve_columns_per_block(row_count=rows, factor_total=len(fids))
        for chunk_start in range(0, len(fids), cols):
            chunk = fids[chunk_start:chunk_start + cols]
            self._write_one_block(partition, rows, chunk, part_buf)
        # flush 后清空该 partition 缓冲
        self._buffer[partition] = {}
        self._order[partition] = []

    def _write_one_block(
        self,
        partition: str,
        rows: int,
        fids: Sequence[str],
        part_buf: Mapping[str, np.ndarray],
    ) -> None:
        block_id = f"b{_partition_token(partition)}_{self._block_seq:04d}"
        seq = self._block_seq
        self._block_seq += 1
        rel = f"{partition}/block_{seq:04d}{VALUES_SUFFIX}"
        path = self.base_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        # 一个 block 值文件：rows × len(fids) float32。列顺序与 fids 对齐。
        block = np.empty((rows, len(fids)), dtype=self.dtype)
        column_map: dict[str, int] = {}
        for col, fid in enumerate(fids):
            col_arr = part_buf[fid]
            if col_arr.shape[0] == rows:
                block[:, col] = col_arr
            else:
                # 行数不一致兜底：取重叠（不应发生；同 partition 行数一致）。
                block[:, col] = col_arr[:rows]
            column_map[fid] = col
        np.save(path, np.ascontiguousarray(block))
        spec = BlockSpec(
            block_id=block_id,
            partition=partition,
            file=rel,
            row_count=rows,
            factor_ids=list(fids),
            column_map=column_map,
        )
        self._blocks[block_id] = spec
        for fid in fids:
            self._factor_to_block.setdefault(fid, {})[partition] = block_id
        self._partition_rows[partition] = rows

    def _resolve_columns_per_block(self, *, row_count: int, factor_total: int) -> int:
        if self.columns_per_block is not None:
            return max(1, int(self.columns_per_block))
        return choose_columns_per_block(
            total_factors=factor_total,
            row_count=row_count,
            writer_memory_bytes=self.writer_memory_bytes,
        )

    def finish(self) -> Path:
        """落残余缓冲 + 写根 manifest。返回 manifest 路径。"""
        if self._finished:
            return self.manifest_path
        for partition in list(self._buffer.keys()):
            self.flush_partition(partition)
        self._manifest = BlockManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            dtype=self.dtype,
            blocks=dict(self._blocks),
            factor_to_block={
                k: dict(v) for k, v in self._factor_to_block.items()
            },
            partition_rows=dict(self._partition_rows),
            columns_per_block=self.columns_per_block,
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.manifest_path, self._manifest.to_dict())
        self._finished = True
        return self.manifest_path

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / MANIFEST_NAME

    @property
    def manifest(self) -> BlockManifest | None:
        return self._manifest

    # -- 内部工具 -----------------------------------------------------------
    def _coerce_values(self, values: Any, factor_id: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                raise ValueError(
                    f"FeatureBlockWriter: 因子 {factor_id} 值必须为 1D，got ndim={arr.ndim}"
                )
        if arr.dtype != np.dtype(FACTOR_VALUE_DTYPE):
            arr = arr.astype(FACTOR_VALUE_DTYPE)  # 显式 dtype 契约强制 float32
        return np.ascontiguousarray(arr)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---- reader -----------------------------------------------------------------
def load_manifest(base_dir: str | Path) -> BlockManifest:
    """读取根 manifest。"""
    path = Path(base_dir) / MANIFEST_NAME
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if int(data.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema 不匹配：{data.get('schema_version')} != {MANIFEST_SCHEMA_VERSION}"
        )
    return BlockManifest.from_dict(data)


class FeatureBlockReader:
    """reader：FactorID -> BlockID -> Column，回读因子值（float32 契约）。"""

    def __init__(self, base_dir: str | Path, manifest: BlockManifest | None = None):
        self.base_dir = Path(base_dir)
        self.manifest = manifest or load_manifest(base_dir)

    def resolve(self, factor_id: str) -> list[tuple[str, BlockSpec, int]]:
        return self.manifest.resolve(factor_id)

    def read_block(self, block_id: str) -> np.ndarray:
        spec = self.manifest.blocks.get(block_id)
        if spec is None:
            raise KeyError(f"block 不存在: {block_id}")
        arr = np.load(self.base_dir / spec.file, allow_pickle=False)
        if arr.dtype != np.dtype(FACTOR_VALUE_DTYPE):
            arr = arr.astype(FACTOR_VALUE_DTYPE)
        return np.ascontiguousarray(arr)

    def read_factor(self, factor_id: str) -> dict[str, np.ndarray]:
        """返回 ``{partition: 1D float32 ndarray}``。"""
        out: dict[str, np.ndarray] = {}
        for partition, spec, column in self.resolve(factor_id):
            block = self.read_block(spec.block_id)
            out[partition] = np.ascontiguousarray(block[:, column])
        return out

    def read_factor_full(self, factor_id: str) -> np.ndarray:
        """跨全部分区拼接（partition 排序）—— TIER2 全历史读路径。"""
        per = self.read_factor(factor_id)
        if not per:
            raise KeyError(f"因子不存在: {factor_id}")
        parts = [per[k] for k in sorted(per)]
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def factor_count(self) -> int:
        return len(self.manifest.factor_to_block)

    def block_count(self) -> int:
        return len(self.manifest.blocks)


# ---- sink 接线：result-complete -> hand to writer ---------------------------
def make_sink_writer(block_writer: FeatureBlockWriter) -> Any:
    """构造 StreamResultSink 兼容的 ``Callable[[list[ResultItem]], None]``。

    用法（result-complete → hand to writer）：:

        sink = StreamingResultSink(writer=make_sink_writer(block_writer), ...)
        sink.start(); ... sink.submit(name=fid, value=1d_array, partition=p) ...
        sink.finish()
        block_writer.finish()   # 落残余块 + 写 manifest
    """

    def _sink_writer(items: list[Any]) -> None:
        block_writer.write_results(items)

    return _sink_writer
