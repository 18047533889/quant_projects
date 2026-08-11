# -*- coding: utf-8 -*-
"""R39 PERF-057 —— 直接使用 PyArrow ``ParquetWriter`` / ``RecordBatch`` 流式落盘。

不经 ``pandas.to_parquet``，避免把 Arrow 数据先物化回 DataFrame 再重新转换的
开销（转换、复制、二次校验）。本模块只做一件事：把一批 ``pa.RecordBatch``
以**流式、固定 schema** 的方式写入 parquet：

- row-group streaming：按行数（``row_group_size``）或未压缩字节预算
  （``compress_row_group_bytes``）自动切分 row group；
- 固定 schema：所有 batch 统一 cast 到给定 schema，杜绝列序/类型漂移；
- statistics：默认写 column statistics（min/max/null_count…）供谓词下推；
- dictionary encoding：可对指定列开启（字符串/低基数列受益最大）；
- 压缩：SNAPPY / ZSTD(-1/-3) / UNCOMPRESSED，可配置 compression_level。

本模块是**独立基础设施**，本轮不接线默认 materialize 写路径（storage-format
agent 负责接线）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 压缩规格
# ---------------------------------------------------------------------------

#: 接受的压缩字面量 → (codec, compression_level)。SNAPPY 无级；ZSTD 可带级。
_COMPRESSION_SPECS: Mapping[str, tuple[str, int | None]] = {
    "SNAPPY": ("snappy", None),
    "SNAPPY-1": ("snappy", None),
    "ZSTD": ("zstd", None),
    "ZSTD-1": ("zstd", 1),
    "ZSTD-3": ("zstd", 3),
    "UNCOMPRESSED": ("none", None),
    "NONE": ("none", None),
    "LZ4": ("lz4", None),
}

#: 允许的 row-group 字节预算字面量（基准脚本用）。
ROW_GROUP_BYTES_LITERALS: Mapping[str, int] = {
    "64MB": 64 * 1024 * 1024,
    "128MB": 128 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
}


def normalize_compression(compression: str) -> tuple[str, int | None]:
    """把用户给的压缩名规整为 (codec, level)。

    ``ZSTD-1`` / ``ZSTD-3`` 被映射为 ``("zstd", 1|3)``；``SNAPPY`` 等原样映射；
    未知的 ``ZSTD-<n>`` 自动解析级别；其余按小写透传。
    """
    if compression is None:
        raise ValueError("compression 不能为 None")
    key = str(compression).upper().strip()
    if key in _COMPRESSION_SPECS:
        return _COMPRESSION_SPECS[key]
    if key.startswith("ZSTD-"):
        try:
            level = int(key.split("-", 1)[1])
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"无法解析压缩级别: {compression!r}") from exc
        return ("zstd", level)
    return (str(compression).lower(), None)


# ---------------------------------------------------------------------------
# 写结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParquetWriteResult:
    """一次 ``BatchParquetWriter`` 写盘的元数据。"""

    path: str
    num_row_groups: int
    num_rows: int
    num_columns: int
    bytes_written: int
    compression: str
    compression_level: int | None
    dictionary_encode_cols: tuple[str, ...]
    row_group_size: int | None
    compress_row_group_bytes: int | None
    write_cpu_seconds: float
    write_wall_seconds: float

    def to_dict(self) -> dict:
        out = dict(self.__dict__)
        out["dictionary_encode_cols"] = list(out["dictionary_encode_cols"])
        return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class BatchParquetWriter:
    """流式 parquet writer（可增量 ``write_batch``，也可一次性 ``write_batches``）。

    行组切分规则（二选一，优先字节预算）：
      - ``compress_row_group_bytes`` 给定：按**未压缩字节数**自动切 row group；
      - 否则 ``row_group_size`` 给定：按**行数**切 row group；
      - 两者都未给：``close()`` 时整表写为单个 row group（仅适合小数据）。
    """

    def __init__(
        self,
        path,
        schema: pa.Schema,
        *,
        compression: str = "SNAPPY",
        stats: bool = True,
        dictionary_encode_cols: Sequence[str] = (),
        use_byte_stream_split: bool = False,
        row_group_size: int | None = None,
        compress_row_group_bytes: int | None = None,
        version: str = "2.6",
    ) -> None:
        if not isinstance(schema, pa.Schema):
            raise TypeError(f"schema 必须是 pa.Schema，实际 {type(schema).__name__}")
        if compress_row_group_bytes is not None:
            if compress_row_group_bytes <= 0:
                raise ValueError("compress_row_group_bytes 必须为正")
            row_group_size = None
        elif row_group_size is not None:
            if row_group_size <= 0:
                raise ValueError("row_group_size 必须为正")

        self.path = str(path)
        self.schema = schema
        codec, level = normalize_compression(compression)
        self.compression = codec
        self.compression_level = level

        dict_cols = tuple(dictionary_encode_cols)
        unknown = [c for c in dict_cols if c not in schema.names]
        if unknown:
            raise ValueError(f"dictionary_encode_cols 含未知列: {unknown}")
        # 非空列表 → 精确指定；空 → 交给 pyarrow 默认（True）。
        use_dictionary: object = list(dict_cols) if dict_cols else True

        self._writer = pq.ParquetWriter(
            self.path,
            schema,
            compression=codec,
            compression_level=level,
            write_statistics=stats,
            use_dictionary=use_dictionary,
            use_byte_stream_split=use_byte_stream_split,
            version=version,
        )
        self._pending: list[pa.RecordBatch] = []
        self._pending_rows = 0
        self._pending_bytes = 0
        self._num_row_groups = 0
        self._num_rows = 0
        self._closed = False
        self._row_group_size = row_group_size
        self._compress_row_group_bytes = compress_row_group_bytes
        self._dict_cols = dict_cols
        self._stats = bool(stats)

    # -- 输入归一化 ---------------------------------------------------------

    @staticmethod
    def _coerce_batches(reader_or_batches) -> Iterator[pa.RecordBatch]:
        if isinstance(reader_or_batches, pa.RecordBatchReader):
            yield from reader_or_batches
            return
        if isinstance(reader_or_batches, pa.Table):
            yield from reader_or_batches.to_batches()
            return
        if isinstance(reader_or_batches, pa.RecordBatch):
            yield reader_or_batches
            return
        if isinstance(reader_or_batches, (bytes, str, Path)):
            raise TypeError("reader_or_batches 不能是路径；请先读成 RecordBatch/Table")
        # 通用可迭代：逐元素归一化（兼容 list[RecordBatch]、list[Table] 等）。
        for item in reader_or_batches:
            if isinstance(item, pa.RecordBatch):
                yield item
            elif isinstance(item, pa.Table):
                yield from item.to_batches()
            else:
                raise TypeError(
                    f"期望 pa.RecordBatch / pa.Table，实际 {type(item).__name__}"
                )

    # -- 流式写入 -----------------------------------------------------------

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """累积一个 batch；到达切分阈值时立即落一个 row group。

        - ``compress_row_group_bytes`` 模式：按未压缩字节数累积，达到预算即冲刷
          （单个超预算 batch 会独占一个 row group）。
        - ``row_group_size`` 模式：单个超长 batch 会被切成多个 row group，
          保证每个 row group 行数不超过 ``row_group_size``。
        """
        if self._closed:
            raise RuntimeError("writer 已 close")
        if not isinstance(batch, pa.RecordBatch):
            if isinstance(batch, pa.Table):
                for b in batch.to_batches():
                    self.write_batch(b)
                return
            raise TypeError(f"期望 pa.RecordBatch，实际 {type(batch).__name__}")
        if batch.schema != self.schema:
            try:
                batch = batch.cast(self.schema)
            except Exception as exc:  # noqa: BLE001 - 转为友好错误
                raise ValueError(
                    f"batch schema 与固定 schema 不兼容:\n  batch={batch.schema}\n"
                    f"  fixed={self.schema}"
                ) from exc

        if self._row_group_size is not None:
            self._write_batch_row_limited(batch)
            return

        self._pending.append(batch)
        self._pending_rows += batch.num_rows
        self._pending_bytes += batch.nbytes
        if (
            self._compress_row_group_bytes is not None
            and self._pending_bytes >= self._compress_row_group_bytes
        ):
            self._flush()

    def _write_batch_row_limited(self, batch: pa.RecordBatch) -> None:
        """按行数切分：把 batch 切成 row_group_size 的整数份逐一写。"""
        size = self._row_group_size
        remaining = batch
        while remaining is not None and remaining.num_rows > 0:
            space = size - self._pending_rows
            if space <= 0:
                self._flush()
                space = size
            if remaining.num_rows <= space:
                self._pending.append(remaining)
                self._pending_rows += remaining.num_rows
                self._pending_bytes += remaining.nbytes
                remaining = None
            else:
                head, tail = remaining.slice(0, space), remaining.slice(space)
                self._pending.append(head)
                self._pending_rows += head.num_rows
                self._pending_bytes += head.nbytes
                remaining = tail
            if self._pending_rows >= size:
                self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        table = pa.Table.from_batches(self._pending)
        # row_group_size=None → 整表作为一个 row group，从而精确控制行组数。
        self._writer.write_table(table, row_group_size=None)
        self._num_row_groups += 1
        self._num_rows += table.num_rows
        self._pending = []
        self._pending_rows = 0
        self._pending_bytes = 0

    def close(self) -> ParquetWriteResult:
        """冲刷剩余数据并关闭文件，返回写盘元数据。"""
        if not self._closed:
            self._flush()
            self._writer.close()
            self._closed = True
        return ParquetWriteResult(
            path=self.path,
            num_row_groups=self._num_row_groups,
            num_rows=self._num_rows,
            num_columns=len(self.schema.names),
            bytes_written=_file_size(self.path),
            compression=self.compression,
            compression_level=self.compression_level,
            dictionary_encode_cols=self._dict_cols,
            row_group_size=self._row_group_size,
            compress_row_group_bytes=self._compress_row_group_bytes,
            write_cpu_seconds=0.0,
            write_wall_seconds=0.0,
        )

    def __enter__(self) -> "BatchParquetWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 一次性便捷入口 ------------------------------------------------------

    @classmethod
    def write_batches(
        cls,
        reader_or_batches,
        schema: pa.Schema,
        path,
        *,
        row_group_size: int | None = None,
        compression: str = "SNAPPY",
        stats: bool = True,
        dictionary_encode_cols: Sequence[str] = (),
        compress_row_group_bytes: int | None = None,
        use_byte_stream_split: bool = False,
    ) -> ParquetWriteResult:
        """把 ``reader_or_batches``（RecordBatchReader / Table / RecordBatch /
        可迭代）流式写入 ``path``。

        ``schema`` 给定固定 schema（所有 batch cast 到它）；传 ``None`` 时从
        首 batch / reader 推导。
        """
        if schema is None:
            if isinstance(reader_or_batches, pa.RecordBatchReader):
                schema = reader_or_batches.schema
            else:
                for probe in cls._coerce_batches(reader_or_batches):
                    schema = probe.schema
                    break
            if schema is None:  # pragma: no cover - defensive
                raise ValueError("无法推导 schema：输入为空")
        if not isinstance(schema, pa.Schema):
            raise TypeError(f"schema 必须是 pa.Schema，实际 {type(schema).__name__}")

        cpu0 = time.process_time()
        wall0 = time.perf_counter()
        writer = cls(
            path,
            schema,
            compression=compression,
            stats=stats,
            dictionary_encode_cols=dictionary_encode_cols,
            use_byte_stream_split=use_byte_stream_split,
            row_group_size=row_group_size,
            compress_row_group_bytes=compress_row_group_bytes,
        )
        try:
            for batch in cls._coerce_batches(reader_or_batches):
                writer.write_batch(batch)
        finally:
            res = writer.close()
        write_cpu = time.process_time() - cpu0
        write_wall = time.perf_counter() - wall0
        return ParquetWriteResult(
            path=res.path,
            num_row_groups=res.num_row_groups,
            num_rows=res.num_rows,
            num_columns=res.num_columns,
            bytes_written=res.bytes_written,
            compression=res.compression,
            compression_level=res.compression_level,
            dictionary_encode_cols=res.dictionary_encode_cols,
            row_group_size=res.row_group_size,
            compress_row_group_bytes=res.compress_row_group_bytes,
            write_cpu_seconds=write_cpu,
            write_wall_seconds=write_wall,
        )

    @classmethod
    def write_pa_table(
        cls,
        table: pa.Table,
        path,
        *,
        row_group_size: int | None = None,
        compression: str = "SNAPPY",
        stats: bool = True,
        dictionary_encode_cols: Sequence[str] = (),
        compress_row_group_bytes: int | None = None,
    ) -> ParquetWriteResult:
        """把一张 ``pa.Table`` 直接写入 parquet（便捷入口）。"""
        if not isinstance(table, pa.Table):
            raise TypeError(f"write_pa_table 期望 pa.Table，实际 {type(table).__name__}")
        return cls.write_batches(
            table,
            table.schema,
            path,
            row_group_size=row_group_size,
            compression=compression,
            stats=stats,
            dictionary_encode_cols=dictionary_encode_cols,
            compress_row_group_bytes=compress_row_group_bytes,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _file_size(path) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:  # pragma: no cover - defensive
        return -1


def read_back_parquet(path) -> pa.Table:
    """读取 parquet 文件为 Arrow Table（用于 round-trip 校验）。"""
    return pq.read_table(str(path))


__all__ = [
    "BatchParquetWriter",
    "ParquetWriteResult",
    "ROW_GROUP_BYTES_LITERALS",
    "normalize_compression",
    "read_back_parquet",
]
