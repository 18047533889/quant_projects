# -*- coding: utf-8 -*-
"""MB-P1-019: Smart representation selector.

Backend 表示选择器（数据格式智能选择）：
    - Pandas DataFrame vs Series vs ndarray
    - Polars DataFrame vs LazyFrame
    - DuckDB Relation vs Arrow Table vs Pandas
    - Arrow RecordBatch vs Table

目标：根据下游算子选择最优中间表示，减少格式转换开销。

决策依据：
    1. 下游算子支持的格式（capability）
    2. 当前格式转换成本（Pandas→Arrow 便宜，Arrow→Pandas 贵）
    3. 内存压力（LazyFrame 延迟物化 < DataFrame）
    4. 后续算子数量（多次复用 → eager；单次使用 → lazy）
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class RepresentationCost:
    """表示转换成本估算。"""

    from_repr: str
    to_repr: str
    conversion_ms: float
    memory_overhead_bytes: int
    supports_zero_copy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_repr": self.from_repr,
            "to_repr": self.to_repr,
            "conversion_ms": round(self.conversion_ms, 2),
            "memory_overhead_bytes": self.memory_overhead_bytes,
            "supports_zero_copy": self.supports_zero_copy,
        }


class SmartRepresentationSelector:
    """智能表示选择器（最小化格式转换开销）。

    集成点：
        - PhysicalLowerer 为每个算子选择输入/输出表示
        - HybridExecutor 根据推荐表示执行转换
        - CSE cache 存储最优表示（下次直接复用）
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 预定义转换成本矩阵（经验值，可通过 calibration 更新）
        self._conversion_costs = self._build_conversion_cost_matrix()
        # 统计：表示选择决策
        self._selection_count: dict[str, int] = {}

    def _build_conversion_cost_matrix(self) -> dict[tuple[str, str], RepresentationCost]:
        """构建表示转换成本矩阵（from → to）。

        成本单位：转换 1GB 数据需要的毫秒数。
        """
        costs: dict[tuple[str, str], RepresentationCost] = {}

        # Pandas 相关
        costs[("pandas_dataframe", "pandas_series")] = RepresentationCost(
            "pandas_dataframe", "pandas_series", conversion_ms=1.0,
            memory_overhead_bytes=0, supports_zero_copy=True,
        )
        costs[("pandas_dataframe", "numpy_array")] = RepresentationCost(
            "pandas_dataframe", "numpy_array", conversion_ms=5.0,
            memory_overhead_bytes=0, supports_zero_copy=False,
        )
        costs[("pandas_dataframe", "arrow_table")] = RepresentationCost(
            "pandas_dataframe", "arrow_table", conversion_ms=50.0,
            memory_overhead_bytes=100 * 1024**2, supports_zero_copy=False,
        )
        costs[("pandas_dataframe", "polars_dataframe")] = RepresentationCost(
            "pandas_dataframe", "polars_dataframe", conversion_ms=80.0,
            memory_overhead_bytes=50 * 1024**2, supports_zero_copy=False,
        )

        # Arrow 相关
        costs[("arrow_table", "pandas_dataframe")] = RepresentationCost(
            "arrow_table", "pandas_dataframe", conversion_ms=100.0,
            memory_overhead_bytes=200 * 1024**2, supports_zero_copy=False,
        )
        costs[("arrow_table", "polars_dataframe")] = RepresentationCost(
            "arrow_table", "polars_dataframe", conversion_ms=10.0,
            memory_overhead_bytes=0, supports_zero_copy=True,
        )
        costs[("arrow_table", "duckdb_relation")] = RepresentationCost(
            "arrow_table", "duckdb_relation", conversion_ms=5.0,
            memory_overhead_bytes=0, supports_zero_copy=True,
        )

        # Polars 相关
        costs[("polars_dataframe", "polars_lazyframe")] = RepresentationCost(
            "polars_dataframe", "polars_lazyframe", conversion_ms=1.0,
            memory_overhead_bytes=0, supports_zero_copy=True,
        )
        costs[("polars_lazyframe", "polars_dataframe")] = RepresentationCost(
            "polars_lazyframe", "polars_dataframe", conversion_ms=500.0,
            memory_overhead_bytes=0, supports_zero_copy=False,  # collect 需要计算
        )
        costs[("polars_dataframe", "arrow_table")] = RepresentationCost(
            "polars_dataframe", "arrow_table", conversion_ms=20.0,
            memory_overhead_bytes=0, supports_zero_copy=True,
        )
        costs[("polars_dataframe", "pandas_dataframe")] = RepresentationCost(
            "polars_dataframe", "pandas_dataframe", conversion_ms=100.0,
            memory_overhead_bytes=100 * 1024**2, supports_zero_copy=False,
        )

        # DuckDB 相关
        costs[("duckdb_relation", "arrow_table")] = RepresentationCost(
            "duckdb_relation", "arrow_table", conversion_ms=10.0,
            memory_overhead_bytes=0, supports_zero_copy=True,
        )
        costs[("duckdb_relation", "pandas_dataframe")] = RepresentationCost(
            "duckdb_relation", "pandas_dataframe", conversion_ms=150.0,
            memory_overhead_bytes=200 * 1024**2, supports_zero_copy=False,
        )

        return costs

    def select_representation(
        self,
        current_repr: str,
        downstream_operators: list[dict[str, Any]],
        *,
        memory_pressure: str = "NORMAL",
        reuse_count: int = 1,
    ) -> str:
        """选择最优中间表示（最小化总转换成本）。

        Args:
            current_repr: 当前数据表示
            downstream_operators: 下游算子列表（每项包含 backend 和 preferred_repr）
            memory_pressure: 内存压力阶段（"NORMAL" | "PRESSURE_1" | ...）
            reuse_count: 后续复用次数（多次复用优先 eager）

        Returns:
            推荐的表示类型
        """
        if not downstream_operators:
            return current_repr

        with self._lock:
            # 候选表示：当前表示 + 下游算子支持的表示
            candidates: set[str] = {current_repr}
            for op in downstream_operators:
                preferred = op.get("preferred_repr", "")
                if preferred:
                    candidates.add(preferred)

            # 计算每个候选的总成本
            best_repr = current_repr
            best_cost = float("inf")

            for candidate in candidates:
                # 转换成本：current → candidate
                conversion_cost = self._get_conversion_cost(current_repr, candidate)

                # 下游使用成本：candidate → 每个下游算子的 preferred
                usage_cost = 0.0
                for op in downstream_operators:
                    op_preferred = op.get("preferred_repr", candidate)
                    usage_cost += self._get_conversion_cost(candidate, op_preferred)

                # 内存压力惩罚（lazy > eager）
                memory_penalty = 0.0
                if memory_pressure in {"PRESSURE_3", "PRESSURE_4", "CRITICAL"}:
                    if "lazy" not in candidate.lower():
                        memory_penalty = 100.0

                # 复用次数奖励（多次复用优先 eager，避免重复 collect）
                reuse_penalty = 0.0
                if reuse_count > 3 and "lazy" in candidate.lower():
                    reuse_penalty = 200.0 * reuse_count

                total_cost = conversion_cost + usage_cost + memory_penalty + reuse_penalty

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_repr = candidate

            self._selection_count[best_repr] = self._selection_count.get(best_repr, 0) + 1

            _logger.debug(
                "representation selected: %s → %s (cost=%.1f)",
                current_repr, best_repr, best_cost,
            )

            return best_repr

    def _get_conversion_cost(self, from_repr: str, to_repr: str) -> float:
        """返回表示转换成本（毫秒 / GB）。"""
        if from_repr == to_repr:
            return 0.0

        key = (from_repr, to_repr)
        cost_info = self._conversion_costs.get(key)
        if cost_info:
            return cost_info.conversion_ms

        # 未知转换：保守估算（两次转换：from→arrow→to）
        return 200.0

    def suggest_cache_representation(
        self,
        operator: str,
        consumers: list[dict[str, Any]],
    ) -> str:
        """为 CSE cache 推荐存储表示（最小化后续读取成本）。

        Args:
            operator: 算子类型
            consumers: 消费者列表（每项包含 backend 和 preferred_repr）

        Returns:
            推荐的 cache 表示
        """
        # Arrow Table 是最通用的中间格式（Pandas/Polars/DuckDB 均可零拷贝或低成本转换）
        if not consumers:
            return "arrow_table"

        # 统计消费者的 preferred_repr
        repr_counts: dict[str, int] = {}
        for consumer in consumers:
            preferred = consumer.get("preferred_repr", "")
            if preferred:
                repr_counts[preferred] = repr_counts.get(preferred, 0) + 1

        # 返回最多消费者需要的表示（减少读取时转换）
        if repr_counts:
            return max(repr_counts, key=repr_counts.get)  # type: ignore

        return "arrow_table"

    def supports_zero_copy(self, from_repr: str, to_repr: str) -> bool:
        """判断转换是否支持 zero-copy。"""
        if from_repr == to_repr:
            return True

        key = (from_repr, to_repr)
        cost_info = self._conversion_costs.get(key)
        return cost_info.supports_zero_copy if cost_info else False

    def get_conversion_cost_info(
        self, from_repr: str, to_repr: str
    ) -> RepresentationCost | None:
        """返回详细转换成本信息（诊断用）。"""
        key = (from_repr, to_repr)
        with self._lock:
            return self._conversion_costs.get(key)

    def summary(self) -> dict[str, Any]:
        """返回选择器状态摘要。"""
        with self._lock:
            return {
                "total_selections": sum(self._selection_count.values()),
                "selection_distribution": dict(self._selection_count),
                "known_conversions": len(self._conversion_costs),
            }


# 全局单例
_global_selector: SmartRepresentationSelector | None = None
_global_lock = threading.Lock()


def get_global_representation_selector() -> SmartRepresentationSelector:
    """返回全局 SmartRepresentationSelector（进程级单例）。"""
    global _global_selector
    if _global_selector is None:
        with _global_lock:
            if _global_selector is None:
                _global_selector = SmartRepresentationSelector()
    return _global_selector
