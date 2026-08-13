# -*- coding: utf-8 -*-
"""MB-P1-016: Batch transfer instead of per-operator.

Backend 间数据传输批量化（减少转换开销）：
    - Pandas → DuckDB：批量注册多张表（一次连接初始化）
    - DuckDB → Polars：批量 Arrow transfer（共享 RecordBatchReader）
    - Polars → Pandas：批量 to_pandas()（共享类型推断）
    - 避免逐算子 format 转换（N 次 → 1 次）

示例场景：
    100 个因子，每个因子从 Pandas source 执行 DuckDB SQL
    - 逐算子转换：100 次 pandas→duckdb→pandas（300 次转换）
    - 批量转换：1 次 register batch + 100 次 SQL + 1 次 batch export（102 次）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class BatchTransferMetrics:
    """批量传输统计。"""

    batch_transfers: int = 0
    single_transfers: int = 0
    total_tables_transferred: int = 0
    total_rows_transferred: int = 0
    estimated_saved_conversions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_transfers": self.batch_transfers,
            "single_transfers": self.single_transfers,
            "total_tables_transferred": self.total_tables_transferred,
            "total_rows_transferred": self.total_rows_transferred,
            "estimated_saved_conversions": self.estimated_saved_conversions,
        }


class BatchTransferOptimizer:
    """Backend 间批量数据传输优化器。

    集成点：
        - HybridExecutor 在 backend 切换时调用 batch_register()
        - PhysicalLowerer 识别连续相同 source 的 region（批量传输候选）
        - AdaptiveBatchScheduler fusion group 共享 batch transfer
    """

    def __init__(self, *, batch_threshold: int = 5) -> None:
        """
        Args:
            batch_threshold: 累积多少张表触发批量传输（平衡延迟 vs 吞吐）
        """
        self.batch_threshold = batch_threshold
        self._metrics = BatchTransferMetrics()
        self._lock = threading.RLock()
        # 待批量传输的表队列（按 target backend 分组）
        self._pending_batches: dict[str, list[dict[str, Any]]] = {}

    def register_transfer_request(
        self,
        table_name: str,
        data: Any,
        source_backend: str,
        target_backend: str,
    ) -> str | None:
        """注册一次传输请求（不立即执行，累积到批次）。

        Args:
            table_name: 表名
            data: 数据（DataFrame / Arrow Table）
            source_backend: 源 backend
            target_backend: 目标 backend

        Returns:
            batch_id（已触发批量传输）或 None（仍在累积）
        """
        key = f"{source_backend}→{target_backend}"

        with self._lock:
            if key not in self._pending_batches:
                self._pending_batches[key] = []

            self._pending_batches[key].append(
                {
                    "table_name": table_name,
                    "data": data,
                    "source_backend": source_backend,
                    "target_backend": target_backend,
                }
            )

            # 达到阈值：触发批量传输
            if len(self._pending_batches[key]) >= self.batch_threshold:
                batch_id = self._execute_batch_transfer(key)
                return batch_id

        return None

    def flush_pending(self, source_backend: str, target_backend: str) -> str | None:
        """强制刷新待传输批次（即使未达阈值）。

        Args:
            source_backend: 源 backend
            target_backend: 目标 backend

        Returns:
            batch_id（已执行批量传输）或 None（无待传输数据）
        """
        key = f"{source_backend}→{target_backend}"

        with self._lock:
            if key not in self._pending_batches or not self._pending_batches[key]:
                return None

            return self._execute_batch_transfer(key)

    def _execute_batch_transfer(self, key: str) -> str:
        """执行一次批量传输（内部调用，已持锁）。

        Args:
            key: 传输方向 key（"pandas→duckdb"）

        Returns:
            batch_id
        """
        batch = self._pending_batches.pop(key, [])
        if not batch:
            return ""

        batch_id = f"batch_{self._metrics.batch_transfers + 1}"
        source_backend = batch[0]["source_backend"]
        target_backend = batch[0]["target_backend"]

        _logger.info(
            "batch transfer: %s (%d tables)", key, len(batch),
        )

        try:
            if target_backend == "duckdb":
                self._batch_to_duckdb(batch, batch_id)
            elif target_backend == "polars":
                self._batch_to_polars(batch, batch_id)
            elif target_backend == "pandas":
                self._batch_to_pandas(batch, batch_id)
            else:
                # 未知 target：逐表 fallback
                for item in batch:
                    self._single_transfer(item)
                self._metrics.single_transfers += len(batch)

            self._metrics.batch_transfers += 1
            self._metrics.total_tables_transferred += len(batch)
            # 估算节省的转换次数：batch (1 setup + N transfers) vs N × (setup + transfer)
            self._metrics.estimated_saved_conversions += len(batch) - 1

        except Exception as exc:
            _logger.error("batch transfer %s failed: %s; fallback to single", key, exc)
            for item in batch:
                try:
                    self._single_transfer(item)
                except Exception:
                    pass
            self._metrics.single_transfers += len(batch)

        return batch_id

    def _batch_to_duckdb(self, batch: list[dict[str, Any]], batch_id: str) -> None:
        """批量传输到 DuckDB（共享连接初始化）。

        Args:
            batch: 待传输表列表
            batch_id: 批次标识
        """
        try:
            import duckdb
        except ImportError:
            _logger.warning("duckdb not installed; batch transfer disabled")
            for item in batch:
                self._single_transfer(item)
            return

        # 创建共享 DuckDB 连接（或复用全局连接）
        conn = duckdb.connect(":memory:")

        # 批量注册表（一次连接初始化开销）
        for item in batch:
            table_name = item["table_name"]
            data = item["data"]

            try:
                # 尝试 zero-copy Arrow 路径
                from runtime.multibackend.duckdb_arrow_boundary import (
                    get_global_arrow_optimizer,
                )

                optimizer = get_global_arrow_optimizer()
                optimizer.register_arrow_table(conn, table_name, data)
            except Exception:
                # fallback: 直接注册 Pandas
                conn.register(table_name, data)

            # 记录行数
            try:
                n_rows = len(data)
                self._metrics.total_rows_transferred += n_rows
            except Exception:
                pass

        # 将注册好的连接存入 context（供后续 SQL 使用）
        # 实际集成需要 backend executor 接收 batch conn
        _logger.debug("batch %s: registered %d tables to DuckDB", batch_id, len(batch))

    def _batch_to_polars(self, batch: list[dict[str, Any]], batch_id: str) -> None:
        """批量传输到 Polars（共享 Arrow RecordBatchReader）。

        Args:
            batch: 待传输表列表
            batch_id: 批次标识
        """
        try:
            import polars as pl
        except ImportError:
            _logger.warning("polars not installed; batch transfer disabled")
            for item in batch:
                self._single_transfer(item)
            return

        # 批量转换（共享类型推断 / schema 解析）
        for item in batch:
            data = item["data"]
            try:
                # Pandas → Polars（zero-copy 如果可能）
                if hasattr(data, "to_pandas"):
                    data = data.to_pandas()
                pl_df = pl.from_pandas(data)
                item["converted"] = pl_df

                n_rows = len(pl_df)
                self._metrics.total_rows_transferred += n_rows
            except Exception as exc:
                _logger.warning("polars conversion failed: %s", exc)
                item["converted"] = data

        _logger.debug("batch %s: converted %d tables to Polars", batch_id, len(batch))

    def _batch_to_pandas(self, batch: list[dict[str, Any]], batch_id: str) -> None:
        """批量传输到 Pandas（共享 dtype 推断）。

        Args:
            batch: 待传输表列表
            batch_id: 批次标识
        """
        # Pandas 是常见 target（DuckDB/Polars → Pandas）
        for item in batch:
            data = item["data"]
            try:
                if hasattr(data, "to_pandas"):
                    # Arrow / Polars / DuckDB Relation → Pandas
                    pd_df = data.to_pandas()
                elif hasattr(data, "arrow"):
                    # DuckDB Relation.arrow() → Pandas
                    arrow_table = data.arrow()
                    pd_df = arrow_table.to_pandas()
                else:
                    pd_df = data

                item["converted"] = pd_df

                n_rows = len(pd_df)
                self._metrics.total_rows_transferred += n_rows
            except Exception as exc:
                _logger.warning("pandas conversion failed: %s", exc)
                item["converted"] = data

        _logger.debug("batch %s: converted %d tables to Pandas", batch_id, len(batch))

    def _single_transfer(self, item: dict[str, Any]) -> None:
        """单表传输（fallback 路径）。"""
        # 简化实现：实际需要完整格式转换逻辑
        with self._lock:
            self._metrics.single_transfers += 1

    def metrics(self) -> BatchTransferMetrics:
        """返回累计统计。"""
        with self._lock:
            return BatchTransferMetrics(
                batch_transfers=self._metrics.batch_transfers,
                single_transfers=self._metrics.single_transfers,
                total_tables_transferred=self._metrics.total_tables_transferred,
                total_rows_transferred=self._metrics.total_rows_transferred,
                estimated_saved_conversions=self._metrics.estimated_saved_conversions,
            )

    def reset_metrics(self) -> None:
        """重置统计。"""
        with self._lock:
            self._metrics = BatchTransferMetrics()


# 全局单例
_global_batch_transfer: BatchTransferOptimizer | None = None
_global_lock = threading.Lock()


def get_global_batch_transfer_optimizer() -> BatchTransferOptimizer:
    """返回全局 BatchTransferOptimizer（进程级单例）。"""
    global _global_batch_transfer
    if _global_batch_transfer is None:
        with _global_lock:
            if _global_batch_transfer is None:
                _global_batch_transfer = BatchTransferOptimizer()
    return _global_batch_transfer
