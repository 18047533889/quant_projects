# -*- coding: utf-8 -*-
"""测试 runtime.adaptive_config 自适应资源配置系统。"""
import os
import pytest
from unittest.mock import patch, mock_open

from factor_engine.runtime.adaptive_config import (
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
    _compute_host_fraction,
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
        """CPU 核心数检测应返回合理值（逻辑核心）。"""
        cpu_count = _get_cpu_count()
        assert isinstance(cpu_count, int)
        assert cpu_count > 0
        # 合理范围：1-256 核心
        assert 1 <= cpu_count <= 256

    def test_cgroup_v2_memory_limit(self):
        """应正确解析 cgroup v2 memory.max。"""
        mock_cgroup_v2 = "4294967296"  # 4GB
        with patch("builtins.open", mock_open(read_data=mock_cgroup_v2)):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value.total = 32 * 1024**3  # 32GB host
                memory_gb = _get_system_memory_gb()
                # 应取 min(32GB, 4GB) = 4GB
                assert 3.9 <= memory_gb <= 4.1

    def test_cgroup_v2_unlimited(self):
        """应识别 cgroup v2 'max' 为无限制。"""
        with patch("builtins.open", mock_open(read_data="max")):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value.total = 32 * 1024**3
                memory_gb = _get_system_memory_gb()
                # 应使用 host 的 32GB
                assert 31 <= memory_gb <= 33

    def test_cgroup_v1_memory_limit(self):
        """应正确解析 cgroup v1 memory.limit_in_bytes。"""
        mock_cgroup_v1 = str(8 * 1024**3)  # 8GB
        def mock_open_side_effect(path, *args, **kwargs):
            if "memory.max" in path:
                raise FileNotFoundError
            elif "memory.limit_in_bytes" in path:
                return mock_open(read_data=mock_cgroup_v1)()
            raise FileNotFoundError

        with patch("builtins.open", side_effect=mock_open_side_effect):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value.total = 64 * 1024**3
                memory_gb = _get_system_memory_gb()
                # 应取 min(64GB, 8GB) = 8GB
                assert 7.9 <= memory_gb <= 8.1

    def test_cgroup_v1_unlimited_sentinel(self):
        """应识别 cgroup v1 巨大哨兵值为无限制。"""
        sentinel = "9223372036854771712"  # cgroup v1 unlimited sentinel
        def mock_open_side_effect(path, *args, **kwargs):
            if "memory.max" in path:
                raise FileNotFoundError
            elif "memory.limit_in_bytes" in path:
                return mock_open(read_data=sentinel)()
            raise FileNotFoundError

        with patch("builtins.open", side_effect=mock_open_side_effect):
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value.total = 16 * 1024**3
                memory_gb = _get_system_memory_gb()
                # 应使用 host 的 16GB（忽略哨兵）
                assert 15 <= memory_gb <= 17

    def test_cpu_affinity_detection(self):
        """应优先使用 sched_getaffinity（容器感知）。"""
        with patch("os.sched_getaffinity", return_value={0, 1, 2, 3}):
            cpu_count = _get_cpu_count()
            # 应返回 affinity 集合大小
            assert cpu_count == 4

    def test_cgroup_cpu_quota(self):
        """应考虑 cgroup CPU 配额。"""
        # 模拟 affinity 8 核，但 cgroup 限制为 2 核
        mock_quota = "200000"  # 200ms quota
        mock_period = "100000"  # 100ms period => 2 cores

        def mock_open_side_effect(path, *args, **kwargs):
            if "cpu.max" in path:
                return mock_open(read_data=f"{mock_quota} {mock_period}")()
            raise FileNotFoundError

        with patch("os.sched_getaffinity", return_value=set(range(8))):
            with patch("builtins.open", side_effect=mock_open_side_effect):
                cpu_count = _get_cpu_count()
                # 应取 min(8, 2) = 2
                assert cpu_count == 2


class TestHostFraction:
    """测试 _compute_host_fraction 小机器保守策略。"""

    def test_small_host_conservative(self):
        """小机器应使用较低比例。"""
        assert _compute_host_fraction(4.0) == 0.55
        assert _compute_host_fraction(8.0) == 0.60

    def test_large_host_aggressive(self):
        """大机器应使用较高比例。"""
        assert _compute_host_fraction(128.0) == 0.82
        assert _compute_host_fraction(512.0) == 0.85

    def test_fraction_monotonic(self):
        """比例应随内存单调递增。"""
        fractions = [_compute_host_fraction(m) for m in [4, 8, 16, 32, 64, 128, 256]]
        assert fractions == sorted(fractions)


class TestAdaptiveScale:
    """测试 _adaptive_scale 缩放函数。"""

    def test_adaptive_scale_sqrt_for_counts(self):
        """COUNT 应使用 sqrt 缩放（scale_power=0.5）。"""
        base = 500
        # 30GB 基准
        result_30 = _adaptive_scale(base, 30.0, scale_power=0.5)
        assert 450 <= result_30 <= 550

        # 8GB 应更小（sqrt 缩放）
        result_8 = _adaptive_scale(base, 8.0, scale_power=0.5)
        assert result_8 < result_30

        # 128GB 应更大
        result_128 = _adaptive_scale(base, 128.0, scale_power=0.5)
        assert result_128 > result_30

    def test_adaptive_scale_linear_for_bytes(self):
        """BYTE BUDGET 应使用线性缩放（scale_power=1.0）。"""
        base = 8 * 1024**3  # 8GB

        # 线性缩放：内存加倍，预算也加倍
        result_30 = _adaptive_scale(base, 30.0, scale_power=1.0)
        result_60 = _adaptive_scale(base, 60.0, scale_power=1.0)

        # 60GB 应约为 30GB 的 2 倍
        ratio = result_60 / result_30
        assert 1.9 <= ratio <= 2.1

    def test_adaptive_scale_respects_min(self):
        """应尊重最小值限制。"""
        result = _adaptive_scale(500, 1.0, min_value=100, scale_power=0.5)
        assert result >= 100

    def test_adaptive_scale_respects_max(self):
        """应尊重最大值限制。"""
        result = _adaptive_scale(500, 500.0, max_value=2000, scale_power=0.5)
        assert result <= 2000


class TestGetAdaptiveConfig:
    """测试 get_adaptive_config 核心函数。"""

    def test_small_memory_conservative_fraction(self):
        """小内存服务器（4GB）应使用保守比例，避免 OOM。"""
        config = get_adaptive_config(force_memory_gb=4.0, force_cpu_cores=4)

        assert config.system_memory_gb == 4.0
        # hard_limit 应约为 4GB * 0.55 * 0.85 = 1.87GB < 2.2GB（避免 OOM）
        assert config.hard_memory_limit_bytes < 2.5 * 1024**3
        # DuckDB 应远小于 hard_limit
        assert config.duckdb_memory_limit_mb * 1024**2 < config.hard_memory_limit_bytes

    def test_medium_memory_config(self):
        """中等内存服务器（30GB）应生成平衡配置。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

        assert config.system_memory_gb == 30.0
        assert config.system_cpu_cores == 8

        # DuckDB 内存应为 40% = 12.2GB（不是 50%）
        expected_duckdb_mb = int(30 * 1024 * 0.40)
        assert config.duckdb_memory_limit_mb == expected_duckdb_mb

        # 验证层级关系
        assert config.safe_envelope_bytes <= config.hard_memory_limit_bytes
        duckdb_bytes = config.duckdb_memory_limit_mb * 1024**2
        assert duckdb_bytes <= config.hard_memory_limit_bytes

    def test_large_memory_linear_scaling(self):
        """大内存服务器（512GB）应线性扩展绝对字节预算。"""
        config_64 = get_adaptive_config(force_memory_gb=64.0, force_cpu_cores=16)
        config_512 = get_adaptive_config(force_memory_gb=512.0, force_cpu_cores=64)

        # 512GB 的 hard_limit 应远大于 64GB（接近 8x）
        ratio = config_512.hard_memory_limit_bytes / config_64.hard_memory_limit_bytes
        assert ratio > 5.0  # 至少 5 倍以上

        # cache_size 应也线性增长
        cache_ratio = config_512.cache_size_bytes / config_64.cache_size_bytes
        assert cache_ratio > 5.0

    def test_memory_hierarchy_invariant(self):
        """所有配置应满足内存层级不变式。"""
        for memory_gb in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            config = get_adaptive_config(force_memory_gb=float(memory_gb), force_cpu_cores=8)

            duckdb_bytes = config.duckdb_memory_limit_mb * 1024**2

            # 核心不变式：duckdb <= hard_limit, safe <= hard_limit
            assert duckdb_bytes <= config.hard_memory_limit_bytes, \
                f"{memory_gb}GB: duckdb {duckdb_bytes/1024**3:.1f}G > hard {config.hard_memory_limit_bytes/1024**3:.1f}G"
            assert config.safe_envelope_bytes <= config.hard_memory_limit_bytes, \
                f"{memory_gb}GB: safe {config.safe_envelope_bytes/1024**3:.1f}G > hard {config.hard_memory_limit_bytes/1024**3:.1f}G"
            assert config.block_abs_max_bytes <= config.hard_memory_limit_bytes
            assert config.cache_size_bytes <= config.hard_memory_limit_bytes

    def test_distinct_byte_budgets(self):
        """不同的字节预算应有明确差异（不再全部相同）。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

        budgets = {
            "hard_memory_limit": config.hard_memory_limit_bytes,
            "safe_envelope": config.safe_envelope_bytes,
            "block_abs_max": config.block_abs_max_bytes,
            "cache_size": config.cache_size_bytes,
            "streaming_threshold": config.streaming_threshold_bytes,
            "compile_budget": config.compile_budget_bytes,
        }

        # 应至少有 4 个不同的值
        unique_values = len(set(budgets.values()))
        assert unique_values >= 4, f"Only {unique_values} unique budgets: {budgets}"

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

    def test_env_override_duckdb_max_threads(self):
        """DUCKDB_MAX_THREADS 应覆盖 8 核上限。"""
        os.environ["DUCKDB_MAX_THREADS"] = "32"
        try:
            config = get_adaptive_config(force_memory_gb=512.0, force_cpu_cores=64)
            # 应使用 min(32, 64) = 32 而非默认的 8
            assert config.duckdb_threads == 32
        finally:
            del os.environ["DUCKDB_MAX_THREADS"]


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

        # 应该是不同的实例
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


class TestEdgeCases:
    """测试边界情况。"""

    def test_very_small_memory(self):
        """极小内存（2GB）应生成最小配置。"""
        config = get_adaptive_config(force_memory_gb=2.0, force_cpu_cores=1)

        assert config.system_memory_gb == 2.0
        # 配置应该是有效的最小值
        assert config.duckdb_threads >= 1
        assert config.polars_threads >= 1
        assert config.batch_size >= 10_000
        assert config.compile_chunk_size >= 100
        assert config.hard_memory_limit_bytes >= 2 * 1024**3  # 至少 2GB

    def test_very_large_memory(self):
        """超大内存（1TB）应生成最大配置且不溢出。"""
        config = get_adaptive_config(force_memory_gb=1024.0, force_cpu_cores=128)

        assert config.system_memory_gb == 1024.0
        # DuckDB 内存：1024 * 0.4 = 409.6GB
        assert config.duckdb_memory_limit_mb == int(1024 * 1024 * 0.4)
        # hard_limit 应受上限约束
        assert config.hard_memory_limit_bytes <= 768 * 1024**3
        # COUNT 配置应受上限约束
        assert config.compile_chunk_size <= 2000
        assert config.batch_size <= 500_000

    def test_config_source_tracking(self):
        """应正确追踪配置来源。"""
        config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

        assert hasattr(config, "config_source")
        assert isinstance(config.config_source, dict)
        assert "duckdb_threads" in config.config_source
        assert config.config_source["duckdb_threads"] in ["env", "adaptive"]

    def test_duckdb_memory_cap_enforced(self):
        """DuckDB 内存超过 hard_limit 时应自动 cap 到 80%。"""
        # 模拟用户通过环境变量设置了过大的 DuckDB 内存
        with patch.dict(os.environ, {"DUCKDB_MEMORY_LIMIT_MB": "300000"}):  # 300GB
            config = get_adaptive_config(force_memory_gb=30.0, force_cpu_cores=8)

            # hard_limit 约为 18.67GB
            # DuckDB 应被 cap 到 hard_limit * 0.8 ≈ 14.9GB
            duckdb_bytes = config.duckdb_memory_limit_mb * 1024**2
            assert duckdb_bytes <= config.hard_memory_limit_bytes, \
                "DuckDB memory should be capped below hard_limit"

            # 应该约为 80% of hard_limit
            expected_capped = int(config.hard_memory_limit_bytes * 0.80 / (1024**2))
            assert abs(config.duckdb_memory_limit_mb - expected_capped) < 100, \
                f"Expected ~{expected_capped}MB, got {config.duckdb_memory_limit_mb}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
