# -*- coding: utf-8 -*-
"""Polars 性能优化测试。

验证 streaming 模式、线程池调优、内存估算等优化功能。
"""
import os
import pytest


class TestPolarsPerformanceConfig:
    """测试 Polars 性能配置模块。"""

    def test_config_from_env_defaults(self):
        """测试默认配置（无环境变量）。"""
        from backend.polars_performance_config import PolarsPerformanceConfig

        config = PolarsPerformanceConfig.from_env()
        assert config.enable_streaming is True
        assert config.adaptive_batch_size is True
        assert config.lazy_optimization_level == 2
        assert config.predicate_pushdown is True
        assert config.projection_pushdown is True

    def test_config_from_env_custom(self, monkeypatch):
        """测试自定义环境变量配置。"""
        from backend.polars_performance_config import PolarsPerformanceConfig

        monkeypatch.setenv("POLARS_ENABLE_STREAMING", "0")
        monkeypatch.setenv("POLARS_MAX_THREADS", "4")
        monkeypatch.setenv("POLARS_MEMORY_BUDGET_MB", "256")
        monkeypatch.setenv("POLARS_LAZY_OPTIMIZATION", "1")

        config = PolarsPerformanceConfig.from_env()
        assert config.enable_streaming is False
        assert config.max_threads == 4
        assert config.memory_budget_mb == 256.0
        assert config.lazy_optimization_level == 1

    def test_get_optimal_thread_count(self):
        """测试最优线程数计算。"""
        from backend.polars_performance_config import PolarsPerformanceConfig

        # 测试自动检测
        config = PolarsPerformanceConfig()
        thread_count = config.get_optimal_thread_count()
        assert 1 <= thread_count <= 32

        # 测试显式设置
        config = PolarsPerformanceConfig(max_threads=8)
        assert config.get_optimal_thread_count() == 8

        # 测试边界条件
        config = PolarsPerformanceConfig(max_threads=100)
        assert config.get_optimal_thread_count() == 32  # 最大 32

        config = PolarsPerformanceConfig(max_threads=0)
        assert config.get_optimal_thread_count() >= 1  # 最小 1

    def test_get_collect_kwargs_streaming_enabled(self):
        """测试 collect 参数生成（streaming 启用）。"""
        from backend.polars_performance_config import PolarsPerformanceConfig

        config = PolarsPerformanceConfig(enable_streaming=True)

        # 大数据集：应该启用 streaming
        kwargs = config.get_collect_kwargs(estimated_rows=2_000_000)
        assert kwargs.get("streaming") is True

        # 小数据集：不启用 streaming
        kwargs = config.get_collect_kwargs(estimated_rows=10_000)
        assert "streaming" not in kwargs or kwargs.get("streaming") is False

        # 未知大小：保守策略，启用 streaming
        kwargs = config.get_collect_kwargs(estimated_rows=None)
        assert kwargs.get("streaming") is True

    def test_get_collect_kwargs_streaming_disabled(self):
        """测试 collect 参数生成（streaming 禁用）。"""
        from backend.polars_performance_config import PolarsPerformanceConfig

        config = PolarsPerformanceConfig(enable_streaming=False)

        # 即使大数据集，也不启用 streaming
        kwargs = config.get_collect_kwargs(estimated_rows=2_000_000)
        assert "streaming" not in kwargs or kwargs.get("streaming") is False

    def test_configure_polars_global(self):
        """测试 Polars 全局配置（需要 polars 安装）。"""
        pytest.importorskip("polars")
        from backend.polars_performance_config import PolarsPerformanceConfig

        config = PolarsPerformanceConfig(max_threads=4)
        # 应该不抛出异常
        config.configure_polars_global()


class TestPolarsMemoryOptimizer:
    """测试内存优化工具。"""

    def test_estimate_dataframe_memory(self):
        """测试 DataFrame 内存估算。"""
        from backend.polars_performance_config import estimate_dataframe_memory

        # 基本测试
        memory = estimate_dataframe_memory(rows=1000, cols=10, avg_col_size=8)
        expected_data = 1000 * 10 * 8  # 80,000 bytes
        expected_overhead = int(expected_data * 0.2)  # 16,000 bytes
        assert memory == expected_data + expected_overhead

        # 零行或零列
        assert estimate_dataframe_memory(0, 10) == 0
        assert estimate_dataframe_memory(1000, 0) == 0

    def test_get_adaptive_batch_size(self):
        """测试自适应批大小计算。"""
        from backend.polars_performance_config import get_adaptive_batch_size

        # 基本测试
        batch_size = get_adaptive_batch_size(
            total_items=10000,
            memory_budget_bytes=10 * 1024 * 1024,  # 10MB
            item_size_bytes=1024,  # 1KB per item
            min_batch=16,
            max_batch=512,
        )
        assert 16 <= batch_size <= 512

        # 小数据集：返回全部
        batch_size = get_adaptive_batch_size(
            total_items=10,
            memory_budget_bytes=10 * 1024 * 1024,
            item_size_bytes=1024,
        )
        assert batch_size == 10

        # 内存约束：不超过预算
        batch_size = get_adaptive_batch_size(
            total_items=100000,
            memory_budget_bytes=1024 * 1024,  # 1MB
            item_size_bytes=10 * 1024,  # 10KB per item
            min_batch=16,
            max_batch=512,
        )
        # 1MB / 10KB = 102 items，但不超过 max_batch
        assert batch_size <= 512

    def test_memory_estimate(self):
        """测试 MemoryEstimate 类。"""
        from backend.polars_memory_optimizer import MemoryEstimate

        estimate = MemoryEstimate(
            rows=1000, cols=10, estimated_bytes=80000, overhead_bytes=16000
        )
        assert estimate.total_bytes == 96000
        assert estimate.total_mb == pytest.approx(0.0915, rel=0.01)

    def test_estimate_polars_dataframe_memory(self):
        """测试精确的 Polars DataFrame 内存估算。"""
        from backend.polars_memory_optimizer import estimate_polars_dataframe_memory

        # 纯数值列
        estimate = estimate_polars_dataframe_memory(rows=1000, cols=10, dtype_size=8)
        assert estimate.rows == 1000
        assert estimate.cols == 10
        assert estimate.estimated_bytes == 1000 * 10 * 8
        assert estimate.overhead_bytes > 0

        # 包含字符串列
        estimate = estimate_polars_dataframe_memory(
            rows=1000, cols=5, dtype_size=8, string_cols=2, string_avg_len=20
        )
        assert estimate.cols == 7  # 5 numeric + 2 string
        numeric_mem = 1000 * 5 * 8
        string_mem = 1000 * 2 * (20 + 8)
        assert estimate.estimated_bytes == numeric_mem + string_mem

    def test_calculate_optimal_chunk_size(self):
        """测试最优分块大小计算。"""
        from backend.polars_memory_optimizer import calculate_optimal_chunk_size

        # 充足内存
        chunk_size = calculate_optimal_chunk_size(
            total_rows=1_000_000,
            total_cols=50,
            available_memory_bytes=1024 * 1024 * 1024,  # 1GB
            safety_factor=0.7,
        )
        assert 1000 <= chunk_size <= 1_000_000

        # 受限内存
        chunk_size = calculate_optimal_chunk_size(
            total_rows=1_000_000,
            total_cols=50,
            available_memory_bytes=10 * 1024 * 1024,  # 10MB
            safety_factor=0.7,
        )
        assert chunk_size >= 1000  # 至少 min_chunk_rows

        # 小数据集
        chunk_size = calculate_optimal_chunk_size(
            total_rows=500, total_cols=50, available_memory_bytes=1024 * 1024 * 1024
        )
        assert chunk_size == 500  # 不超过总行数

    def test_should_use_streaming(self):
        """测试 streaming 模式判断。"""
        from backend.polars_memory_optimizer import should_use_streaming

        # 未知行数：使用 streaming
        assert should_use_streaming(None, 50, 1024 * 1024 * 1024) is True

        # 大数据集（超过阈值）
        assert (
            should_use_streaming(
                10_000_000, 50, 1024 * 1024 * 1024, streaming_threshold_mb=100.0
            )
            is True
        )

        # 小数据集
        assert (
            should_use_streaming(
                10_000, 50, 1024 * 1024 * 1024, streaming_threshold_mb=100.0
            )
            is False
        )

        # 内存不足：使用 streaming
        assert should_use_streaming(1_000_000, 50, 10 * 1024 * 1024) is True

    def test_polars_memory_monitor(self):
        """测试内存监控器。"""
        from backend.polars_memory_optimizer import PolarsMemoryMonitor

        monitor = PolarsMemoryMonitor(budget_bytes=1024 * 1024 * 1024)  # 1GB

        # 记录操作
        monitor.record_collect(result_bytes=100 * 1024 * 1024, used_streaming=False)
        assert monitor.peak_usage_bytes == 100 * 1024 * 1024
        assert monitor.current_usage_bytes == 100 * 1024 * 1024

        monitor.record_collect(result_bytes=200 * 1024 * 1024, used_streaming=True)
        assert monitor.peak_usage_bytes == 200 * 1024 * 1024
        assert monitor.current_usage_bytes == 200 * 1024 * 1024

        # 获取统计
        stats = monitor.get_stats()
        assert stats["collect_count"] == 2
        assert stats["streaming_count"] == 1
        assert stats["streaming_ratio"] == 0.5
        assert stats["peak_usage_mb"] == pytest.approx(200.0, rel=0.1)

        # 判断是否启用 streaming
        assert monitor.should_enable_streaming(700 * 1024 * 1024) is True  # 超过 60%
        assert monitor.should_enable_streaming(100 * 1024 * 1024) is False


class TestPolarsBackendIntegration:
    """测试 Polars 后端集成。"""

    def test_backend_initialization_with_performance_config(self):
        """测试后端初始化时配置性能参数。"""
        pytest.importorskip("polars")
        from backend.polars_backend import PolarsBackend

        # 应该不抛出异常
        backend = PolarsBackend()
        assert backend is not None

    def test_optimized_collect_in_emitter(self, monkeypatch):
        """测试 polars_expr_emitter 中的优化 collect 调用。"""
        pytest.importorskip("polars")

        # 设置环境变量启用 streaming
        monkeypatch.setenv("POLARS_ENABLE_STREAMING", "1")

        # 这个测试需要完整的执行上下文，这里只验证模块可导入
        from backend import polars_expr_emitter

        assert hasattr(polars_expr_emitter, "execute_polars_long_plan")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
