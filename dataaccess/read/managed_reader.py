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
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pyarrow as pa

logger = logging.getLogger("data_access.managed_reader")


class ManagedBatchReader:
    """显式管理 DuckDB RecordBatchReader + cursor 的生命周期。"""

    def __init__(
        self,
        reader: Any,
        cursor: Any = None,
        *,
        on_close: Any = None,
    ) -> None:
        self._reader = reader
        self._cursor = cursor
        self._on_close = on_close
        self._closed = False

    @property
    def schema(self) -> pa.Schema:
        return self._reader.schema

    def read_next_batch(self) -> pa.RecordBatch:
        return self._reader.read_next_batch()

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._closed:
            raise StopIteration
        try:
            return self._reader.read_next_batch()
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        """释放 cursor / reader / 连接。可重复调用。"""
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self._reader, "close"):
                self._reader.close()
        except Exception:
            logger.debug("managed_reader: reader.close 失败", exc_info=True)
        try:
            if self._cursor is not None and hasattr(self._cursor, "close"):
                self._cursor.close()
        except Exception:
            logger.debug("managed_reader: cursor.close 失败", exc_info=True)
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception:
                logger.debug("managed_reader: on_close 回调失败", exc_info=True)
            self._on_close = None
        self._reader = None
        self._cursor = None

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
