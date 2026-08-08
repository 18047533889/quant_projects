# -*- coding: utf-8
"""
ManagedBatchReader —— Arrow RecordBatchReader 的生命周期托管

背景
    DuckDB 流式读：``cursor.execute(sql).to_arrow_reader(batch_size)`` 返回的
    reader 内部持有 cursor 状态。之前没有明确的 owner 包装，cursor 的释放
    依赖 Python GC；生产环境不应依赖「GC 最后会帮我收」。

职责
    1. 把 (reader, cursor) 打包成 ManagedBatchReader，提供显式 ``close()``
    2. ``__enter__ / __exit__ / __del__`` 保证：正常结束、break、异常、
       client disconnect、GC 回收，都释放 cursor
    3. 可迭代、支持 ``read_next_batch()`` / ``schema``，对调用方透明
    4. #P1-8 ``read_next_batch()`` 在 EOF / 异常时也自动 close（不再要求调用方
       自己循环外手工 close）
    5. #P1-7 记录 ``opened_at / first_batch_at / closed_at / rows / bytes``，
       close 时经 ``telemetry_fn`` 上报真实 stream lifetime（而不是 reader 创建耗时）

    注意（#P0-6）：pooled 连接**绝不**作为 ``cursor`` 传入——本类的 ``close()``
    会调用 ``cursor.close()``，pooled 连接的 close 归 pool 所有。pooled 场景
    传 ``cursor=None`` + ``on_close`` 回调。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterator

import pyarrow as pa

logger = logging.getLogger("data_access.managed_reader")


class ManagedBatchReader:
    """显式管理 DuckDB RecordBatchReader + cursor 的生命周期。"""

    def __init__(
        self,
        reader: Any,
        cursor: Any = None,
        *,
        on_close: Callable[["ManagedBatchReader"], Any] | None = None,
        on_before_read: Callable[[], None] | None = None,
        telemetry_fn: Callable[[dict[str, Any]], None] | None = None,
        setup_ms: float | None = None,
    ) -> None:
        self._reader = reader
        self._cursor = cursor
        self._on_close = on_close
        self._on_before_read = on_before_read
        self._telemetry_fn = telemetry_fn
        self._setup_ms = float(setup_ms) if setup_ms is not None else 0.0
        self._closed = False
        self._had_error = False
        # #P1-7 真实 stream lifetime 观测
        self._opened_at = time.perf_counter()
        self._first_batch_at: float | None = None
        self._closed_at: float | None = None
        self._rows = 0
        self._bytes = 0

    @property
    def schema(self) -> pa.Schema:
        return self._reader.schema

    @property
    def had_error(self) -> bool:
        """流内是否发生过异常（on_close 用于判定连接健康）。"""
        return self._had_error

    def read_next_batch(self) -> pa.RecordBatch:
        """#P1-8 EOF / 异常都自动 close，连接生命周期不依赖调用方手工收。"""
        if self._closed:
            raise StopIteration
        if self._on_before_read is not None:
            self._on_before_read()
        try:
            batch = self._reader.read_next_batch()
        except StopIteration:
            self.close()
            raise
        except Exception:
            self._had_error = True
            self.close()
            raise
        if self._first_batch_at is None:
            self._first_batch_at = time.perf_counter()
        self._rows += batch.num_rows
        self._bytes += batch.nbytes
        return batch

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        return self

    def __next__(self) -> pa.RecordBatch:
        return self.read_next_batch()

    def close(self) -> None:
        """释放 reader / cursor / 连接。可重复调用。"""
        if self._closed:
            return
        self._closed = True
        self._closed_at = time.perf_counter()
        try:
            if hasattr(self._reader, "close"):
                self._reader.close()
        except Exception:
            logger.debug("managed_reader: reader.close 失败", exc_info=True)
        # #P0-6 pooled 连接（cursor=None）不在这里 close；由 on_close 归还 pool。
        if self._cursor is not None and hasattr(self._cursor, "close"):
            try:
                self._cursor.close()
            except Exception:
                logger.debug("managed_reader: cursor.close 失败", exc_info=True)
        self._reader = None
        self._cursor = None
        if self._telemetry_fn is not None:
            try:
                self._telemetry_fn(self._telemetry_dict())
            except Exception:
                logger.debug("managed_reader: telemetry 上报失败", exc_info=True)
            self._telemetry_fn = None
        if self._on_close is not None:
            try:
                self._on_close(self)
            except Exception:
                logger.debug("managed_reader: on_close 回调失败", exc_info=True)
            self._on_close = None

    def _telemetry_dict(self) -> dict[str, Any]:
        """#P1-7 stream 生命周期指标（ms / rows / bytes）。"""
        total = 0.0
        if self._closed_at is not None:
            total = (self._closed_at - self._opened_at) * 1000.0
        ttf = None
        if self._first_batch_at is not None:
            ttf = (self._first_batch_at - self._opened_at) * 1000.0
        setup = self._setup_ms
        stream = max(0.0, total - setup)
        return {
            "op": "reader_stream",
            "setup_ms": setup,
            "time_to_first_batch_ms": ttf,
            "stream_duration_ms": stream,
            "total_duration_ms": total,
            "rows": self._rows,
            "bytes": self._bytes,
        }

    def __enter__(self) -> "ManagedBatchReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"ManagedBatchReader(closed={self._closed}, "
            f"reader={type(self._reader).__name__ if self._reader else None})"
        )
