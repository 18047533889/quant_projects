# -*- coding: utf-8 -*-
"""DuckDB 自适应并行配置模块。

根据查询复杂度和数据量动态调整线程数和内存限制。

基于 benchmark 结果的最佳实践：
- 小查询（< 100K rows）：4 threads, 2GB memory
- 中等查询（100K - 1M rows）：8 threads, 9GB memory
- 大查询（> 1M rows）：8-16 threads, 15GB memory
- 聚合密集型：32 threads（高并行度）
- 窗口函数：16 threads（平衡并行与内存）
- Arrow 零拷贝：32 threads（IO 密集）

2026-08-13: 基于 9 场景 benchmark 结果实现自适应配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from factor_engine.backend.sql_pushdown.duckdb_performance import DuckDBParallelConfig


class QueryComplexity(Enum):
    """查询复杂度分类。"""
    SIMPLE = "simple"  # 简单查询（单表扫描、简单聚合）
    MODERATE = "moderate"  # 中等复杂度（JOIN、窗口函数）
    COMPLEX = "complex"  # 复杂查询（多重窗口、嵌套子查询）
    AGGREGATION_INTENSIVE = "aggregation_intensive"  # 聚合密集型
    WINDOW_INTENSIVE = "window_intensive"  # 窗口函数密集型
    JOIN_INTENSIVE = "join_intensive"  # JOIN 密集型


@dataclass(frozen=True)
class WorkloadProfile:
    """工作负载特征。"""
    estimated_rows: int
    complexity: QueryComplexity
    is_batch: bool = False  # 是否批量查询（UNION ALL）
    use_arrow: bool = False  # 是否使用 Arrow 零拷贝


class AdaptiveParallelConfig:
    """自适应并行配置决策器。"""

    def __init__(
        self,
        *,
        cpu_count: int | None = None,
        total_memory_mb: int | None = None,
    ):
        """初始化自适应配置器。

        Args:
            cpu_count: CPU 核心数（默认：系统实际核心数）
            total_memory_mb: 总内存（MB，默认：系统总内存的 50%）
        """
        import multiprocessing
        import psutil

        self.cpu_count = cpu_count or multiprocessing.cpu_count()

        if total_memory_mb is None:
            total_memory_bytes = psutil.virtual_memory().total
            total_memory_mb = int(total_memory_bytes * 0.5 / (1024 * 1024))

        self.total_memory_mb = total_memory_mb

        # Benchmark 结果的最佳配置（从实际测试得出）
        self._best_configs = {
            "small_simple": {"threads": 4, "memory_factor": 0.15},
            "medium_simple": {"threads": 8, "memory_factor": 0.30},
            "large_simple": {"threads": 8, "memory_factor": 0.50},
            "aggregation_intensive": {"threads": min(32, self.cpu_count), "memory_factor": 0.50},
            "window_intensive": {"threads": min(16, self.cpu_count), "memory_factor": 0.50},
            "join_intensive": {"threads": min(4, self.cpu_count), "memory_factor": 0.20},
            "batch_union": {"threads": 8, "memory_factor": 0.40},
            "arrow_zero_copy": {"threads": min(32, self.cpu_count), "memory_factor": 0.50},
        }

    def decide_config(self, profile: WorkloadProfile) -> DuckDBParallelConfig:
        """根据工作负载特征决定最佳配置。

        Args:
            profile: 工作负载特征

        Returns:
            DuckDB 并行配置
        """
        # 1. 优先处理特殊情况
        if profile.use_arrow:
            config_key = "arrow_zero_copy"
        elif profile.is_batch:
            config_key = "batch_union"
        elif profile.complexity == QueryComplexity.AGGREGATION_INTENSIVE:
            config_key = "aggregation_intensive"
        elif profile.complexity == QueryComplexity.WINDOW_INTENSIVE:
            config_key = "window_intensive"
        elif profile.complexity == QueryComplexity.JOIN_INTENSIVE:
            config_key = "join_intensive"
        else:
            # 2. 根据数据量分类
            if profile.estimated_rows < 100_000:
                config_key = "small_simple"
            elif profile.estimated_rows < 1_000_000:
                config_key = "medium_simple"
            else:
                config_key = "large_simple"

        best = self._best_configs[config_key]

        threads = min(best["threads"], self.cpu_count)
        memory_limit_mb = int(self.total_memory_mb * best["memory_factor"])

        # 确保最小配置
        threads = max(1, threads)
        memory_limit_mb = max(512, memory_limit_mb)

        return DuckDBParallelConfig(
            threads=threads,
            memory_limit_mb=memory_limit_mb,
            enable_object_cache=True,
        )

    def estimate_complexity_from_sql(self, sql: str) -> QueryComplexity:
        """从 SQL 字符串估算查询复杂度。

        Args:
            sql: SQL 查询字符串

        Returns:
            查询复杂度分类
        """
        sql_upper = sql.upper()

        # 统计关键字出现次数
        window_count = sql_upper.count(" OVER ")
        join_count = sql_upper.count(" JOIN ")
        group_count = sql_upper.count(" GROUP BY ")
        agg_keywords = ["AVG(", "SUM(", "COUNT(", "MIN(", "MAX(", "STDDEV(", "PERCENTILE_"]
        agg_count = sum(sql_upper.count(kw) for kw in agg_keywords)

        # 决策逻辑
        if window_count >= 3:
            return QueryComplexity.WINDOW_INTENSIVE
        elif join_count >= 2:
            return QueryComplexity.JOIN_INTENSIVE
        elif agg_count >= 5:
            return QueryComplexity.AGGREGATION_INTENSIVE
        elif window_count >= 1 or join_count >= 1:
            return QueryComplexity.MODERATE
        elif agg_count >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def estimate_row_count_from_tables(
        self,
        conn: Any,
        table_names: list[str],
    ) -> int:
        """从表名估算行数（快速探测）。

        Args:
            conn: DuckDB 连接
            table_names: 表名列表

        Returns:
            估算的总行数
        """
        total_rows = 0

        for table_name in table_names:
            try:
                # 快速 COUNT 估算（DuckDB 对小表很快）
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                if result:
                    total_rows += result[0]
            except Exception:
                # 如果失败，假设中等规模
                total_rows += 500_000

        return total_rows

    def auto_config_from_sql(
        self,
        sql: str,
        conn: Any | None = None,
        *,
        hint_rows: int | None = None,
        is_batch: bool = False,
        use_arrow: bool = False,
    ) -> DuckDBParallelConfig:
        """从 SQL 查询自动推荐配置（便捷方法）。

        Args:
            sql: SQL 查询字符串
            conn: DuckDB 连接（可选，用于估算行数）
            hint_rows: 提示行数（如果已知）
            is_batch: 是否批量查询
            use_arrow: 是否使用 Arrow 零拷贝

        Returns:
            推荐的 DuckDB 并行配置
        """
        complexity = self.estimate_complexity_from_sql(sql)

        # 估算行数
        if hint_rows is not None:
            estimated_rows = hint_rows
        elif conn is not None:
            # 尝试从 SQL 提取表名
            table_names = self._extract_table_names(sql)
            estimated_rows = self.estimate_row_count_from_tables(conn, table_names)
        else:
            # 默认假设中等规模
            estimated_rows = 500_000

        profile = WorkloadProfile(
            estimated_rows=estimated_rows,
            complexity=complexity,
            is_batch=is_batch,
            use_arrow=use_arrow,
        )

        return self.decide_config(profile)

    def _extract_table_names(self, sql: str) -> list[str]:
        """从 SQL 提取表名（简单实现）。"""
        import re

        # 匹配 FROM/JOIN 后的表名
        pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        matches = re.findall(pattern, sql, re.IGNORECASE)

        return matches


# 全局实例（延迟初始化）
_GLOBAL_ADAPTIVE_CONFIG: AdaptiveParallelConfig | None = None


def get_adaptive_config() -> AdaptiveParallelConfig:
    """获取全局自适应配置实例。"""
    global _GLOBAL_ADAPTIVE_CONFIG

    if _GLOBAL_ADAPTIVE_CONFIG is None:
        _GLOBAL_ADAPTIVE_CONFIG = AdaptiveParallelConfig()

    return _GLOBAL_ADAPTIVE_CONFIG


# ============================================================================
# 便捷函数
# ============================================================================


def recommend_config(
    sql: str,
    *,
    conn: Any | None = None,
    hint_rows: int | None = None,
    is_batch: bool = False,
    use_arrow: bool = False,
) -> DuckDBParallelConfig:
    """推荐 DuckDB 配置（便捷函数）。

    Args:
        sql: SQL 查询字符串
        conn: DuckDB 连接（可选）
        hint_rows: 提示行数（如果已知）
        is_batch: 是否批量查询
        use_arrow: 是否使用 Arrow 零拷贝

    Returns:
        推荐的 DuckDB 并行配置
    """
    adaptive = get_adaptive_config()
    return adaptive.auto_config_from_sql(
        sql,
        conn=conn,
        hint_rows=hint_rows,
        is_batch=is_batch,
        use_arrow=use_arrow,
    )
