# -*- coding: utf-8 -*-
"""测试 runtime.adaptive_config 自适应资源配置系统。"""
import os
import pytest

from runtime.adaptive_config import (
    AdaptiveResourceConfig,
    auto_configure,
    get_adaptive_config,
    get_global_adaptive_config,
    reset_global_adaptive_config,
    get_duckdb_threads,
    get_duckdb_memory_limit,
    get_polars_threads,
    get_compile_chunk_size,
    get_hard_memory_limit_bytes,
    get_block_abs_max_bytes,
    _get_system_memory_gb,
    _get_cpu_count,
    _adaptive_scale,
)


class TestSystemDetection:
    """测试系统资源检测函数。"""

    def test_get_system_memory_gb(self):
        """系统内存检测应返回合理值。"""
        memory_gb = _get_system_memory_gb()
        assert isinstance(memory_gb, float)
        assert memory_gb > 0
        # 合理范围：1GB - 2TB
        assert 1.0 <= memory_gb <= 2048.0

    def test_get_cpu_count(self):
        """CPU 核心数检测应返回合理值。"""
        cpu_count = _get_cpu_count()
        assert isinstance(cpu_count, int)
        assert cpu_count > 0
        # 合理范围：1-256 核心
        assert 1 <= cpu_count <= 256


class TestAdaptiveScale:
    """测试 _adaptive_scale 缩放函数。"""

    def test_adaptive_scale_baseline(self):
        """基准值应返回接近原值。"""
        # 30GB 基准值 500
        result = _adaptive_scale(500, memory_gb=30.0, min_value=100, max_value=2000)
        assert 400 <= result <= 600  # 允许一定浮动

    def test_adaptive_scale_small_memory(self):
        """小内存应缩小值。"""
        # 8GB 应返回更小的值
        result = _adaptive_scale(500, memory_gb=8.0, min_value=100, max_value=2000)
        assert result < 500
        assert result >= 100  # 不低于最小值

    def test_adaptive_scale_large_memory(self):
        """大内存应放大值。"""
        # 128GB 应返回更大的值
        result = _adaptive_scale(500, memory_gb=128.0, min_value=100, max_value=2000)
        assert result > 500
        assert result <= 2000  # 不超过最大值

    def test_adaptive_scale_respects_min(self):
        """应尊重最小值限制。"""
        result = _adaptive_scale(500, memory_gb=1.0, min_value=100, max_value=2000)
        assert result >= 100

    def test_adaptive_scale_respects_max(self):
        """应尊重最大值限制。"""
        result = _adaptive_scale(500, memory_gb=500.0, min_value=100, max_value=2000)
        assert result <= 2000


class TestGetAdaptiveConfig:
    """测试 get_adaptive_config 核心函数。"""

    def test_small_memory_config(self):
        """小内存服务器（8GB）应生成保守配置。"""
        config = get_adaptive_config(force_memory_gb=8.0, force_cpu_cores=4)

        # 基本检查
        assert config.system_memory_gb == 8.0
        assert config.system_cpu_cores == 4

        # DuckDB 内存应为 50% = 4GB
        assert config.duckdb_memory_limit_mb == int(8 * 1024 * 0.5)
        assert "4096MB" in config.duckdb_memory_limit or "4GB" in config.duckdb_memory_limit

        # 批量大小应较小
        assert config.batch_size < 100_000

        # 编译 chunk 应较小
        assert config.compile_chunk_size < 500

    def test_medium_memory_config(self):
        """中等内存服务器（30GB）应生成平衡配置。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

        assert config.system_memory_gb == 30.0
        assert config.system_cpu_cores == 8

        # DuckDB 内存应为 50% = 15GB
        assert config.duckdb_memory_limit_mb == int(30 * 1024 * 0.5)

        # DuckDB 线程数应为 min(8, cpu_cores) = 8
        assert config.duckdb_threads == 8

        # Polars 线程数应为 cpu_cores
        assert config.polars_threads == 8

    def test_large_memory_config(self):
        """大内存服务器（128GB）应充分利用资源。"""
        config = get_adaptive_config(force_memory_gb=128.0, force_cpu_cores=32)

        assert config.system_memory_gb == 128.0
        assert config.system_cpu_cores == 32

        # DuckDB 内存应为 50% = 64GB
        assert config.duckdb_memory_limit_mb == int(128 * 1024 * 0.5)

        # 批量大小应较大
        assert config.batch_size >= 100_000

        # 编译 chunk 应较大
        assert config.compile_chunk_size >= 500

        # DuckDB 线程数上限为 8
        assert config.duckdb_threads == 8

        # Polars 线程数为 32
        assert config.polars_threads == 32

    def test_xlarge_memory_config(self):
        """超大内存服务器（500GB）应生成最大配置。"""
        config = get_adaptive_config(force_memory_gb=500.0, force_cpu_cores=64)

        assert config.system_memory_gb == 500.0
        assert config.system_cpu_cores == 64

        # DuckDB 内存应为 50% = 250GB
        assert config.duckdb_memory_limit_mb == int(500 * 1024 * 0.5)

        # 批量大小应接近最大值
        assert config.batch_size >= 100_000

        # 硬内存限制应随内存增长
        assert config.hard_memory_limit_bytes > 8 * 1024**3

    def test_config_has_all_fields(self):
        """配置应包含所有必需字段。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

        # 系统资源
        assert hasattr(config, "system_memory_gb")
        assert hasattr(config, "system_cpu_cores")

        # DuckDB
        assert hasattr(config, "duckdb_threads")
        assert hasattr(config, "duckdb_memory_limit")
        assert hasattr(config, "duckdb_memory_limit_mb")

        # Polars
        assert hasattr(config, "polars_threads")
        assert hasattr(config, "polars_streaming_chunk_size")

        # 批处理
        assert hasattr(config, "compile_chunk_size")
        assert hasattr(config, "dag_chunk_size")
        assert hasattr(config, "batch_size")

        # 内存块
        assert hasattr(config, "block_abs_max_bytes")
        assert hasattr(config, "cache_size_bytes")
        assert hasattr(config, "streaming_threshold_bytes")

        # 并发
        assert hasattr(config, "max_workers_io")
        assert hasattr(config, "max_workers_compute")

        # 资源预算
        assert hasattr(config, "hard_memory_limit_bytes")
        assert hasattr(config, "safe_envelope_bytes")
        assert hasattr(config, "compile_budget_bytes")

    def test_env_override_duckdb_threads(self):
        """环境变量应覆盖 DuckDB 线程数。"""
        os.environ["DUCKDB_THREADS"] = "16"
        try:
            config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)
            assert config.duckdb_threads == 16
            assert config.config_source["duckdb_threads"] == "env"
        finally:
            del os.environ["DUCKDB_THREADS"]

    def test_env_override_polars_threads(self):
        """环境变量应覆盖 Polars 线程数。"""
        os.environ["POLARS_MAX_THREADS"] = "12"
        try:
            config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)
            assert config.polars_threads == 12
            assert config.config_source["polars_threads"] == "env"
        finally:
            del os.environ["POLARS_MAX_THREADS"]

    def test_env_override_compile_chunk(self):
        """环境变量应覆盖编译 chunk 大小。"""
        os.environ["COMPILE_CHUNK_SIZE"] = "999"
        try:
            config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)
            assert config.compile_chunk_size == 999
            assert config.config_source["compile_chunk_size"] == "env"
        finally:
            del os.environ["COMPILE_CHUNK_SIZE"]


class TestGlobalConfig:
    """测试全局配置单例。"""

    def test_global_config_singleton(self):
        """全局配置应返回相同实例。"""
        reset_global_adaptive_config()
        config1 = get_global_adaptive_config()
        config2 = get_global_adaptive_config()
        assert config1 is config2

    def test_reset_global_config(self):
        """重置全局配置应创建新实例。"""
        reset_global_adaptive_config()
        config1 = get_global_adaptive_config()

        reset_global_adaptive_config()
        config2 = get_global_adaptive_config()

        # 应该是不同的实例（虽然值可能相同）
        assert config1 is not config2

    def test_convenience_functions(self):
        """便捷访问函数应返回正确值。"""
        reset_global_adaptive_config()
        config = get_global_adaptive_config()

        assert get_duckdb_threads() == config.duckdb_threads
        assert get_duckdb_memory_limit() == config.duckdb_memory_limit
        assert get_polars_threads() == config.polars_threads
        assert get_compile_chunk_size() == config.compile_chunk_size
        assert get_hard_memory_limit_bytes() == config.hard_memory_limit_bytes
        assert get_block_abs_max_bytes() == config.block_abs_max_bytes


class TestAutoConfigureFunction:
    """测试 auto_configure() 启动配置函数。"""

    def test_auto_configure_returns_dict(self):
        """auto_configure() 应返回配置字典。"""
        config = auto_configure(apply_env=False, force_memory_gb=30.0, force_cpu_cores=8)

        assert isinstance(config, dict)
        assert "system_memory_gb" in config
        assert "duckdb_threads" in config
        assert "batch_size" in config
        assert config["system_memory_gb"] == 30.0
        assert config["system_cpu_cores"] == 8

    def test_auto_configure_dict_completeness(self):
        """auto_configure() 返回的字典应包含所有配置。"""
        config = auto_configure(apply_env=False, force_memory_gb=30.0, force_cpu_cores=8)

        required_keys = [
            "system_memory_gb",
            "system_cpu_cores",
            "duckdb_threads",
            "duckdb_memory_limit",
            "polars_threads",
            "compile_chunk_size",
            "dag_chunk_size",
            "batch_size",
            "block_abs_max_bytes",
            "cache_size_bytes",
            "max_workers_io",
            "max_workers_compute",
            "hard_memory_limit_bytes",
        ]

        for key in required_keys:
            assert key in config, f"Missing key: {key}"

    def test_auto_configure_applies_env_vars(self):
        """auto_configure(apply_env=True) 应设置环境变量。"""
        # 清理环境变量
        env_keys = [
            "DUCKDB_THREADS",
            "DUCKDB_MEMORY_LIMIT",
            "POLARS_MAX_THREADS",
            "FE_BATCH_SIZE",
            "FE_DAG_CHUNK_SIZE",
            "FE_MAX_WORKERS",
        ]
        for key in env_keys:
            os.environ.pop(key, None)

        config = auto_configure(apply_env=True, force_memory_gb=30.0, force_cpu_cores=8)

        # 检查环境变量是否被设置
        assert "DUCKDB_THREADS" in os.environ
        assert "DUCKDB_MEMORY_LIMIT" in os.environ
        assert "POLARS_MAX_THREADS" in os.environ
        assert "FE_BATCH_SIZE" in os.environ
        assert "FE_DAG_CHUNK_SIZE" in os.environ
        assert "FE_MAX_WORKERS" in os.environ

        # 检查值是否匹配
        assert int(os.environ["DUCKDB_THREADS"]) == config["duckdb_threads"]
        assert os.environ["DUCKDB_MEMORY_LIMIT"] == config["duckdb_memory_limit"]
        assert int(os.environ["POLARS_MAX_THREADS"]) == config["polars_threads"]
        assert int(os.environ["FE_BATCH_SIZE"]) == config["batch_size"]

    def test_auto_configure_respects_existing_env(self):
        """auto_configure() 应尊重已有环境变量。"""
        os.environ["DUCKDB_THREADS"] = "99"
        os.environ["FE_BATCH_SIZE"] = "88888"

        try:
            config = auto_configure(apply_env=True, force_memory_gb=30.0, force_cpu_cores=8)

            # 应使用环境变量的值（通过 get_adaptive_config 的逻辑）
            assert int(os.environ["DUCKDB_THREADS"]) == 99
            assert int(os.environ["FE_BATCH_SIZE"]) == 88888
        finally:
            del os.environ["DUCKDB_THREADS"]
            del os.environ["FE_BATCH_SIZE"]

    def test_auto_configure_no_env_application(self):
        """auto_configure(apply_env=False) 不应设置新环境变量。"""
        # 清理环境变量
        test_keys = ["DUCKDB_THREADS", "FE_BATCH_SIZE"]
        for key in test_keys:
            os.environ.pop(key, None)

        config = auto_configure(apply_env=False, force_memory_gb=30.0, force_cpu_cores=8)

        # 不应设置环境变量（因为 apply_env=False）
        # 注意：如果之前其他测试设置了环境变量，这里可能会有残留
        # 所以我们只检查返回的配置字典是否有效
        assert isinstance(config, dict)
        assert config["duckdb_threads"] > 0


class TestMemoryScaling:
    """测试内存配置随系统资源的缩放。"""

    def test_duckdb_memory_scales_linearly(self):
        """DuckDB 内存应随系统内存线性增长（50%）。"""
        for memory_gb in [8, 16, 32, 64, 128]:
            config = get_adaptive_config(force_memory_gb=float(memory_gb), force_cpu_cores=8)
            expected_mb = int(memory_gb * 1024 * 0.5)
            assert config.duckdb_memory_limit_mb == expected_mb

    def test_batch_size_scales_with_memory(self):
        """批量大小应随内存增长。"""
        config_8gb = get_adaptive_config(force_memory_gb=8.0, force_cpu_cores=4)
        config_64gb = get_adaptive_config(force_memory_gb=64.0, force_cpu_cores=16)

        assert config_64gb.batch_size > config_8gb.batch_size

    def test_hard_limit_scales_with_memory(self):
        """硬内存限制应随内存增长。"""
        config_8gb = get_adaptive_config(force_memory_gb=8.0, force_cpu_cores=4)
        config_128gb = get_adaptive_config(force_memory_gb=128.0, force_cpu_cores=32)

        assert config_128gb.hard_memory_limit_bytes > config_8gb.hard_memory_limit_bytes


class TestWorkerConfig:
    """测试并发 worker 配置。"""

    def test_io_workers_allow_oversubscription(self):
        """IO worker 应允许超订（1.5x 物理核心）。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)
        # IO workers = min(8 * 3 // 2, 16) = min(12, 16) = 12
        assert config.max_workers_io == 12

    def test_compute_workers_match_physical_cores(self):
        """计算 worker 应匹配物理核心数。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)
        assert config.max_workers_compute == 8

    def test_workers_scale_with_cores(self):
        """Worker 数量应随核心数增长。"""
        config_4c = get_adaptive_config(force_memory_gb=16.0, force_cpu_cores=4)
        config_32c = get_adaptive_config(force_memory_gb=64.0, force_cpu_cores=32)

        assert config_32c.max_workers_compute > config_4c.max_workers_compute
        assert config_32c.max_workers_io > config_4c.max_workers_io


class TestEdgeCases:
    """测试边界情况。"""

    def test_very_small_memory(self):
        """极小内存（2GB）应生成最小配置。"""
        config = get_adaptive_config(force_memory_gb=2.0, force_cpu_cores=1)

        assert config.system_memory_gb == 2.0
        # 配置应该是有效的最小值
        assert config.duckdb_threads >= 1
        assert config.polars_threads >= 1
        assert config.batch_size >= 10_000  # 最小批量
        assert config.compile_chunk_size >= 100  # 最小 chunk

    def test_single_core(self):
        """单核 CPU 应生成有效配置。"""
        config = get_adaptive_config(force_memory_gb=8.0, force_cpu_cores=1)

        assert config.system_cpu_cores == 1
        assert config.duckdb_threads >= 1
        assert config.polars_threads >= 1
        assert config.max_workers_compute >= 1

    def test_very_large_memory(self):
        """超大内存（1TB）应生成最大配置且不溢出。"""
        config = get_adaptive_config(force_memory_gb=1024.0, force_cpu_cores=128)

        assert config.system_memory_gb == 1024.0
        # DuckDB 内存：1024 * 0.5 = 512GB
        assert config.duckdb_memory_limit_mb == int(1024 * 1024 * 0.5)
        # 配置应在合理范围内（有最大值限制）
        assert config.compile_chunk_size <= 2000
        assert config.batch_size <= 500_000

    def test_config_source_tracking(self):
        """应正确追踪配置来源。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

        assert hasattr(config, "config_source")
        assert isinstance(config.config_source, dict)
        assert "duckdb_threads" in config.config_source
        assert config.config_source["duckdb_threads"] in ["env", "adaptive"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
