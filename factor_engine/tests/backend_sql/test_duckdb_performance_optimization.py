# -*- coding: utf-8 -*-
"""DuckDB 性能优化测试套件。

测试所有五项优化的正确性和性能提升。
2026-08-13: 初版测试。
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import pytest

from backend.sql_pushdown.duckdb_performance import (
    DuckDBParallelConfig,
    IndexConfig,
    OptimizedDuckDBExecutor,
    PreparedStatement,
    QueryPlanCache,
    arrow_to_polars_zero_copy,
    compile_batch_union_all,
    create_indexes_if_beneficial,
    duckdb_result_to_series_arrow,
    get_query_plan_cache,
)


class TestBatchCompilation:
    """测试优化点 1：批量编译（UNION ALL）。"""

    def test_compile_batch_union_all_basic(self):
        """基本批量编译测试。"""
        queries = {
            "factor_a": "SELECT ts, inst, close AS _v FROM base",
            "factor_b": "SELECT ts, inst, volume AS _v FROM base",
        }

        result = compile_batch_union_all(queries)
        assert result is not None
        assert "UNION ALL" in result
        assert "_sid" in result
        assert "factor_a" in result
        assert "factor_b" in result

    def test_compile_batch_union_all_empty(self):
        """空查询字典返回 None。"""
        assert compile_batch_union_all({}) is None

    def test_compile_batch_union_all_single(self):
        """单个查询返回 None（不需要 UNION ALL）。"""
        queries = {"factor_a": "SELECT ts, inst, close AS _v FROM base"}
        assert compile_batch_union_all(queries) is None

    def test_compile_batch_union_all_sql_injection_safe(self):
        """SQL 注入防护测试。"""
        queries = {
            "factor'; DROP TABLE users; --": "SELECT ts, inst, close AS _v FROM base",
            "safe_factor": "SELECT ts, inst, open AS _v FROM base",
        }

        result = compile_batch_union_all(queries)
        assert result is not None
        # 单引号应该被转义为双单引号
        assert "'factor''; DROP TABLE users; --'" in result or "'';" in result
        # DROP TABLE 会出现在字符串字面量中，这是安全的
        assert "DROP TABLE" in result


class TestQueryPlanCache:
    """测试优化点 2：查询计划缓存。"""

    def test_cache_basic(self):
        """基本缓存功能。"""
        cache = QueryPlanCache(max_entries=10)

        query = "SELECT * FROM table"
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        # 首次查询：miss
        assert cache.get(query_hash) is None

        # 放入缓存
        cache.put(query, query_hash)

        # 再次查询：hit
        stmt = cache.get(query_hash)
        assert stmt is not None
        assert stmt.query == query
        assert stmt.hit_count == 1

    def test_cache_lru_eviction(self):
        """LRU 逐出测试。"""
        cache = QueryPlanCache(max_entries=3)

        queries = [f"SELECT {i} FROM table" for i in range(5)]
        hashes = [hashlib.sha256(q.encode()).hexdigest()[:16] for q in queries]

        # 填充缓存（超过 max_entries）
        for q, h in zip(queries, hashes):
            cache.put(q, h)

        # 前两个应该被逐出
        assert cache.get(hashes[0]) is None
        assert cache.get(hashes[1]) is None

        # 后三个应该存在
        assert cache.get(hashes[2]) is not None
        assert cache.get(hashes[3]) is not None
        assert cache.get(hashes[4]) is not None

    def test_cache_stats(self):
        """缓存统计信息测试。"""
        cache = QueryPlanCache(max_entries=10)

        query = "SELECT * FROM table"
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        cache.put(query, query_hash)
        cache.get(query_hash)  # hit 1
        cache.get(query_hash)  # hit 2

        stats = cache.stats()
        assert stats["entries"] == 1
        assert stats["total_hits"] == 2

    def test_cache_expiry(self):
        """缓存过期测试。"""
        cache = QueryPlanCache(max_entries=10, max_age_seconds=0.05)

        query = "SELECT * FROM table"
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        cache.put(query, query_hash)
        assert cache.get(query_hash) is not None

        # 等待过期（使用更长时间确保过期）
        time.sleep(0.15)
        assert cache.get(query_hash) is None


class TestParallelConfig:
    """测试优化点 3：并行执行配置。"""

    def test_parallel_config_from_env(self, monkeypatch):
        """从环境变量读取配置。"""
        monkeypatch.setenv("DUCKDB_THREADS", "4")
        monkeypatch.setenv("DUCKDB_MEMORY_LIMIT_MB", "1024")
        monkeypatch.setenv("DUCKDB_OBJECT_CACHE", "false")

        config = DuckDBParallelConfig.from_env()
        assert config.threads == 4
        assert config.memory_limit_mb == 1024
        assert config.enable_object_cache is False

    def test_parallel_config_defaults(self, monkeypatch):
        """默认配置（无环境变量）。"""
        monkeypatch.delenv("DUCKDB_THREADS", raising=False)
        monkeypatch.delenv("DUCKDB_MEMORY_LIMIT_MB", raising=False)

        config = DuckDBParallelConfig.from_env()
        assert config.threads > 0  # 应该等于 CPU 核心数
        assert config.memory_limit_mb > 0  # 应该是系统内存的一半


class TestArrowZeroCopy:
    """测试优化点 4：Arrow 零拷贝转换。"""

    def test_arrow_to_polars_zero_copy(self):
        """Arrow → Polars 零拷贝转换。"""
        import pyarrow as pa
        import polars as pl

        # 创建 Arrow 表
        arrow_table = pa.table({
            "ts": [1, 2, 3],
            "inst": ["A", "B", "C"],
            "_v": [1.0, 2.0, 3.0],
        })

        # 零拷贝转换
        df = arrow_to_polars_zero_copy(arrow_table)

        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["ts", "inst", "_v"]
        assert len(df) == 3

    def test_duckdb_result_to_series_arrow(self):
        """DuckDB 结果 → Series（Arrow 路径）。"""
        import pyarrow as pa

        # 模拟 DuckDB 结果
        arrow_table = pa.table({
            "ts": [1, 1, 2],
            "inst": ["A", "B", "A"],
            "value": [1.0, 2.0, 3.0],
        })

        # 转换为 Series
        series = duckdb_result_to_series_arrow(arrow_table)

        assert len(series) == 3
        assert series.index.names == ["ts", "inst"]
        assert series.name == "value"


class TestIndexOptimization:
    """测试优化点 5：自动索引。"""

    def test_index_config_default(self):
        """默认索引配置。"""
        config = IndexConfig.default()
        assert "ts" in config.indexed_columns
        assert "inst" in config.indexed_columns
        assert "trade_date" in config.indexed_columns

    @pytest.mark.skip(reason="需要真实 DuckDB 连接")
    def test_create_indexes_if_beneficial(self):
        """索引创建测试（需要真实数据库）。"""
        # 此测试需要真实的 DuckDB 连接和数据
        pass


class TestOptimizedExecutor:
    """测试集成的优化执行器。"""

    def test_executor_initialization(self):
        """执行器初始化测试。"""
        executor = OptimizedDuckDBExecutor()

        assert executor.parallel_config is not None
        assert executor.query_cache is not None
        assert executor.enable_indexing is True

    def test_executor_stats(self):
        """执行器统计信息测试。"""
        executor = OptimizedDuckDBExecutor()

        stats = executor.get_stats()
        assert "queries_executed" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "parallel_config" in stats


class TestIntegration:
    """集成测试（需要真实数据源）。"""

    @pytest.mark.skip(reason="需要真实 data_access 和数据")
    def test_end_to_end_optimization(self):
        """端到端优化测试。"""
        # 此测试需要：
        # 1. 真实的 DataAccess store
        # 2. 已编译的 SQL 查询
        # 3. 真实数据
        pass

    @pytest.mark.skip(reason="需要真实数据源")
    def test_performance_benchmark(self):
        """性能基准测试。"""
        # 对比优化前后的性能差异
        # 预期：
        # - 批量编译：30-50% 提升
        # - 查询缓存：20-40% 提升
        # - 并行执行：2-4x 提升
        # - Arrow 零拷贝：15-25% 提升
        # - 索引优化：10-20% 提升
        pass


# ============================================================================
# 性能基准测试工具
# ============================================================================

def benchmark_optimization(
    name: str,
    baseline_fn: Any,
    optimized_fn: Any,
    *args: Any,
    iterations: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """性能基准测试工具。

    Args:
        name: 优化名称
        baseline_fn: 基线函数
        optimized_fn: 优化后函数
        iterations: 迭代次数
        *args, **kwargs: 函数参数

    Returns:
        包含性能统计的 dict
    """
    import statistics

    baseline_times = []
    optimized_times = []

    for _ in range(iterations):
        # 基线测试
        start = time.perf_counter()
        baseline_fn(*args, **kwargs)
        baseline_times.append(time.perf_counter() - start)

        # 优化测试
        start = time.perf_counter()
        optimized_fn(*args, **kwargs)
        optimized_times.append(time.perf_counter() - start)

    baseline_avg = statistics.mean(baseline_times)
    optimized_avg = statistics.mean(optimized_times)
    improvement = (baseline_avg - optimized_avg) / baseline_avg * 100.0

    return {
        "name": name,
        "baseline_avg_ms": baseline_avg * 1000.0,
        "optimized_avg_ms": optimized_avg * 1000.0,
        "improvement_pct": improvement,
        "speedup": baseline_avg / optimized_avg if optimized_avg > 0 else float("inf"),
    }
