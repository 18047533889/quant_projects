# -*- coding: utf-8 -*-
"""测试 DuckDB 自适应并行配置。"""
from __future__ import annotations

import pytest

from factor_engine.backend.sql_pushdown.adaptive_parallel_config import (
    AdaptiveParallelConfig,
    QueryComplexity,
    WorkloadProfile,
    recommend_config,
)
from factor_engine.backend.sql_pushdown.duckdb_performance import DuckDBParallelConfig


class TestAdaptiveParallelConfig:
    """测试自适应并行配置。"""

    def test_small_workload(self):
        """测试小规模工作负载配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=50_000,
            complexity=QueryComplexity.SIMPLE,
        )

        config = adaptive.decide_config(profile)

        assert config.threads == 4
        assert config.memory_limit_mb > 0
        assert config.enable_object_cache is True

    def test_medium_workload(self):
        """测试中等规模工作负载配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=500_000,
            complexity=QueryComplexity.MODERATE,
        )

        config = adaptive.decide_config(profile)

        assert config.threads == 8
        assert config.memory_limit_mb > 0

    def test_large_workload(self):
        """测试大规模工作负载配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=2_000_000,
            complexity=QueryComplexity.COMPLEX,
        )

        config = adaptive.decide_config(profile)

        assert config.threads == 8
        assert config.memory_limit_mb > 0

    def test_aggregation_intensive(self):
        """测试聚合密集型配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=32, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=1_000_000,
            complexity=QueryComplexity.AGGREGATION_INTENSIVE,
        )

        config = adaptive.decide_config(profile)

        # 聚合密集型应该使用高并行度
        assert config.threads >= 8
        assert config.enable_object_cache is True

    def test_window_intensive(self):
        """测试窗口函数密集型配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=16, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=1_000_000,
            complexity=QueryComplexity.WINDOW_INTENSIVE,
        )

        config = adaptive.decide_config(profile)

        # 窗口函数应该使用中高并行度
        assert config.threads >= 8
        assert config.threads <= 16

    def test_join_intensive(self):
        """测试 JOIN 密集型配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=500_000,
            complexity=QueryComplexity.JOIN_INTENSIVE,
        )

        config = adaptive.decide_config(profile)

        # JOIN 密集型应该使用较少线程（避免内存压力）
        assert config.threads <= 4

    def test_arrow_zero_copy(self):
        """测试 Arrow 零拷贝路径配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=32, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=1_000_000,
            complexity=QueryComplexity.SIMPLE,
            use_arrow=True,
        )

        config = adaptive.decide_config(profile)

        # Arrow 零拷贝应该使用高并行度
        assert config.threads >= 16

    def test_batch_union(self):
        """测试批量 UNION ALL 配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=16384)

        profile = WorkloadProfile(
            estimated_rows=1_000_000,
            complexity=QueryComplexity.SIMPLE,
            is_batch=True,
        )

        config = adaptive.decide_config(profile)

        assert config.threads == 8

    def test_estimate_complexity_simple(self):
        """测试简单查询复杂度估算。"""
        adaptive = AdaptiveParallelConfig()

        sql = "SELECT * FROM table WHERE price > 100"
        complexity = adaptive.estimate_complexity_from_sql(sql)

        assert complexity == QueryComplexity.SIMPLE

    def test_estimate_complexity_aggregation(self):
        """测试聚合查询复杂度估算。"""
        adaptive = AdaptiveParallelConfig()

        sql = """
            SELECT instrument,
                   AVG(close), SUM(volume), MIN(low),
                   MAX(high), STDDEV(close), COUNT(*)
            FROM data
            GROUP BY instrument
        """
        complexity = adaptive.estimate_complexity_from_sql(sql)

        assert complexity == QueryComplexity.AGGREGATION_INTENSIVE

    def test_estimate_complexity_window(self):
        """测试窗口函数查询复杂度估算。"""
        adaptive = AdaptiveParallelConfig()

        sql = """
            SELECT timestamp, instrument,
                   AVG(close) OVER w1, AVG(close) OVER w2,
                   AVG(close) OVER w3
            FROM data
            WINDOW w1 AS (PARTITION BY instrument ORDER BY timestamp),
                   w2 AS (PARTITION BY instrument ORDER BY timestamp),
                   w3 AS (PARTITION BY instrument ORDER BY timestamp)
        """
        complexity = adaptive.estimate_complexity_from_sql(sql)

        assert complexity == QueryComplexity.WINDOW_INTENSIVE

    def test_estimate_complexity_join(self):
        """测试 JOIN 查询复杂度估算。"""
        adaptive = AdaptiveParallelConfig()

        sql = """
            SELECT t1.*, t2.*, t3.*
            FROM table1 t1
            JOIN table2 t2 ON t1.id = t2.id
            JOIN table3 t3 ON t2.id = t3.id
        """
        complexity = adaptive.estimate_complexity_from_sql(sql)

        assert complexity == QueryComplexity.JOIN_INTENSIVE

    def test_auto_config_from_sql(self):
        """测试从 SQL 自动推荐配置。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=16384)

        sql = """
            SELECT instrument, AVG(close) as avg_close
            FROM data
            GROUP BY instrument
        """

        config = adaptive.auto_config_from_sql(sql, hint_rows=100_000)

        assert isinstance(config, DuckDBParallelConfig)
        assert config.threads > 0
        assert config.memory_limit_mb > 0

    def test_recommend_config_convenience(self):
        """测试便捷函数。"""
        sql = "SELECT * FROM data WHERE close > 100"

        config = recommend_config(sql, hint_rows=50_000)

        assert isinstance(config, DuckDBParallelConfig)
        assert config.threads >= 1
        assert config.enable_object_cache is True

    def test_cpu_bound_respects_cpu_count(self):
        """测试配置不超过 CPU 核心数。"""
        adaptive = AdaptiveParallelConfig(cpu_count=4, total_memory_mb=16384)

        # 尝试聚合密集型（通常推荐 32 threads）
        profile = WorkloadProfile(
            estimated_rows=1_000_000,
            complexity=QueryComplexity.AGGREGATION_INTENSIVE,
        )

        config = adaptive.decide_config(profile)

        # 应该被限制为 4（CPU 核心数）
        assert config.threads <= 4

    def test_memory_minimum_guaranteed(self):
        """测试最小内存保证。"""
        adaptive = AdaptiveParallelConfig(cpu_count=8, total_memory_mb=256)

        profile = WorkloadProfile(
            estimated_rows=100_000,
            complexity=QueryComplexity.SIMPLE,
        )

        config = adaptive.decide_config(profile)

        # 即使总内存很小，也要保证最小 512MB
        assert config.memory_limit_mb >= 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
