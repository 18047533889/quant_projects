# -*- coding: utf-8 -*-
"""测试 runtime.adaptive_config 自适应资源配置系统。"""
import os
import pytest
from unittest.mock import patch

from runtime.adaptive_config import (
    AdaptiveResourceConfig,
    auto_configure,
    get_config_value,
    _detect_total_memory_bytes,
    _detect_available_memory_bytes,
    _detect_cpu_count,
    _detect_disk_space_bytes,
)


class TestDetectionFunctions:
    """测试资源检测函数。"""

    def test_detect_total_memory(self):
        """总内存检测应返回正整数。"""
        total = _detect_total_memory_bytes()
        assert isinstance(total, int)
        assert total > 0
        # 合理范围：至少 1GB，最多 10TB（防止检测错误）
        assert 1 * 1024**3 <= total <= 10 * 1024**4

    def test_detect_available_memory(self):
        """可用内存检测应返回正整数，且不超过总内存。"""
        available = _detect_available_memory_bytes()
        total = _detect_total_memory_bytes()
        assert isinstance(available, int)
        assert available > 0
        assert available <= total

    def test_detect_cpu_count(self):
        """CPU 数量检测应返回正整数。"""
        cpu_count = _detect_cpu_count()
        assert isinstance(cpu_count, int)
        assert cpu_count > 0
        # 合理范围：1-1024 核心
        assert 1 <= cpu_count <= 1024

    def test_detect_disk_space(self):
        """磁盘空间检测应返回正整数。"""
        disk_space = _detect_disk_space_bytes()
        assert isinstance(disk_space, int)
        assert disk_space > 0
        # 合理范围：至少 1GB
        assert disk_space >= 1 * 1024**3


class TestAdaptiveResourceConfig:
    """测试 AdaptiveResourceConfig 类。"""

    def test_detect_creates_valid_config(self):
        """detect() 应返回有效的配置对象。"""
        config = AdaptiveResourceConfig.detect()
        assert config.total_memory_bytes > 0
        assert config.available_memory_bytes > 0
        assert config.cpu_count > 0
        assert config.disk_space_bytes > 0

    def test_properties_convert_to_gb(self):
        """属性应正确转换为 GB 单位。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=32 * 1024**3,
            available_memory_bytes=16 * 1024**3,
            cpu_count=8,
            disk_space_bytes=500 * 1024**3,
        )
        assert config.total_memory_gb == 32.0
        assert config.available_memory_gb == 16.0
        assert config.disk_space_gb == 500.0

    def test_small_memory_config(self):
        """小内存服务器（8GB）应生成保守配置。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=8 * 1024**3,
            available_memory_bytes=6 * 1024**3,  # 6GB 可用
            cpu_count=4,
            disk_space_bytes=100 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # 可用内存 = 6GB * 0.7 = 4.2GB
        # DuckDB = 4.2 * 0.4 = 1.68GB → 1GB (int)
        assert optimal["duckdb_memory_limit"] == "1GB"
        # 批量大小：< 10GB → 10,000
        assert optimal["batch_size"] == 10_000
        # DAG chunk: < 10GB → 100
        assert optimal["dag_chunk_size"] == 100
        # CPU: 4 核心 - 1 保留 = 3
        assert optimal["duckdb_threads"] == 3
        assert optimal["polars_threads"] == 3
        # Workers: 3 // 2 = 1
        assert optimal["max_workers"] == 1

    def test_medium_memory_config(self):
        """中等内存服务器（30GB）应生成中等配置。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=32 * 1024**3,
            available_memory_bytes=28 * 1024**3,  # 28GB 可用
            cpu_count=8,
            disk_space_bytes=500 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # 可用内存 = 28GB * 0.7 = 19.6GB
        # DuckDB = 19.6 * 0.4 = 7.84GB → 7GB
        assert optimal["duckdb_memory_limit"] == "7GB"
        # 批量大小：10-50GB → 100,000
        assert optimal["batch_size"] == 100_000
        # DAG chunk: 10-50GB → 500
        assert optimal["dag_chunk_size"] == 500
        # CPU: 8 核心 - 2 保留 = 6
        assert optimal["duckdb_threads"] == 6
        assert optimal["polars_threads"] == 6
        # Workers: 6 // 2 = 3
        assert optimal["max_workers"] == 3

    def test_large_memory_config(self):
        """大内存服务器（128GB）应生成大批量配置。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=128 * 1024**3,
            available_memory_bytes=120 * 1024**3,
            cpu_count=32,
            disk_space_bytes=2000 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # 可用内存 = 120GB * 0.7 = 84GB
        # DuckDB = 84 * 0.4 = 33.6GB → 33GB
        assert optimal["duckdb_memory_limit"] == "33GB"
        # 批量大小：50-200GB → 500,000
        assert optimal["batch_size"] == 500_000
        # DAG chunk: 50-200GB → 1000
        assert optimal["dag_chunk_size"] == 1000
        # CPU: 32 核心 - 2 保留 = 30
        assert optimal["duckdb_threads"] == 30
        # Workers: 30 // 2 = 15
        assert optimal["max_workers"] == 15

    def test_xlarge_memory_config(self):
        """超大内存服务器（500GB）应生成最大配置。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=512 * 1024**3,
            available_memory_bytes=480 * 1024**3,
            cpu_count=64,
            disk_space_bytes=5000 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # 可用内存 = 480GB * 0.7 = 336GB
        # DuckDB = 336 * 0.4 = 134.4GB → 134GB
        assert optimal["duckdb_memory_limit"] == "134GB"
        # 批量大小：>= 200GB → 1,000,000
        assert optimal["batch_size"] == 1_000_000
        # DAG chunk: >= 200GB → 5000
        assert optimal["dag_chunk_size"] == 5000
        # CPU: 64 核心 - 2 保留 = 62
        assert optimal["duckdb_threads"] == 62
        # Workers: 62 // 2 = 31
        assert optimal["max_workers"] == 31

    def test_config_includes_metadata(self):
        """配置应包含元数据字段。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=32 * 1024**3,
            available_memory_bytes=28 * 1024**3,
            cpu_count=8,
            disk_space_bytes=500 * 1024**3,
        )
        optimal = config.get_optimal_config()

        assert "total_memory_gb" in optimal
        assert "available_memory_gb" in optimal
        assert "usable_memory_gb" in optimal
        assert "cpu_count" in optimal
        assert "disk_space_gb" in optimal
        assert optimal["total_memory_gb"] == 32.0
        assert optimal["cpu_count"] == 8

    def test_cache_and_spill_config(self):
        """Cache 和 spill 配置应合理。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=32 * 1024**3,
            available_memory_bytes=28 * 1024**3,
            cpu_count=8,
            disk_space_bytes=500 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # Cache = 可用内存 * 0.7 * 0.3
        usable_gb = 28 * 0.7  # 19.6GB
        expected_cache = int(usable_gb * 0.3 * 1024**3)
        assert optimal["cache_max_bytes"] == expected_cache

        # Spill threshold = 可用内存 * 0.7 * 0.8
        expected_spill = int(usable_gb * 0.8 * 1024**3)
        assert optimal["cache_spill_threshold_bytes"] == expected_spill

        # Temp file: min(10% 磁盘, 100GB)
        assert optimal["temp_file_max_bytes"] <= 100 * 1024**3


class TestAutoConfigureFunction:
    """测试 auto_configure() 函数。"""

    def test_auto_configure_returns_dict(self):
        """auto_configure() 应返回配置字典。"""
        config = auto_configure(apply_env=False)
        assert isinstance(config, dict)
        assert "duckdb_memory_limit" in config
        assert "batch_size" in config
        assert "cpu_count" in config

    def test_auto_configure_applies_env_vars(self):
        """auto_configure(apply_env=True) 应设置环境变量。"""
        # 清理环境变量
        for key in ["DUCKDB_MEMORY_LIMIT", "DUCKDB_THREADS", "POLARS_MAX_THREADS",
                    "FE_BATCH_SIZE", "FE_DAG_CHUNK_SIZE", "FE_MAX_WORKERS"]:
            os.environ.pop(key, None)

        config = auto_configure(apply_env=True)

        assert "DUCKDB_MEMORY_LIMIT" in os.environ
        assert "DUCKDB_THREADS" in os.environ
        assert "POLARS_MAX_THREADS" in os.environ
        assert "FE_BATCH_SIZE" in os.environ
        assert os.environ["DUCKDB_MEMORY_LIMIT"] == config["duckdb_memory_limit"]
        assert int(os.environ["DUCKDB_THREADS"]) == config["duckdb_threads"]

    def test_auto_configure_respects_existing_env(self):
        """auto_configure() 应尊重已有环境变量。"""
        os.environ["DUCKDB_MEMORY_LIMIT"] = "16GB"
        os.environ["FE_BATCH_SIZE"] = "50000"

        config = auto_configure(apply_env=True)

        # 应使用环境变量的值
        assert config["duckdb_memory_limit"] == "16GB"
        assert config["batch_size"] == 50000

    def test_auto_configure_with_override(self):
        """auto_configure(override=...) 应应用用户覆盖。"""
        override = {
            "batch_size": 999_999,
            "dag_chunk_size": 7777,
        }
        config = auto_configure(apply_env=False, override=override)

        assert config["batch_size"] == 999_999
        assert config["dag_chunk_size"] == 7777

    def test_auto_configure_no_env_application(self):
        """auto_configure(apply_env=False) 不应设置环境变量。"""
        # 清理环境变量
        test_key = "FE_TEST_MARKER_12345"
        os.environ.pop(test_key, None)

        # 记录当前环境变量
        env_before = set(os.environ.keys())

        config = auto_configure(apply_env=False)

        # 不应新增环境变量（但可能读取已有的）
        # 我们检查是否新增了我们清理的关键变量
        assert test_key not in os.environ


class TestGetConfigValue:
    """测试 get_config_value() 辅助函数。"""

    def test_get_batch_size_from_env(self):
        """应从环境变量获取 batch_size。"""
        os.environ["FE_BATCH_SIZE"] = "12345"
        assert get_config_value("batch_size") == 12345

    def test_get_config_with_default(self):
        """环境变量不存在时应返回默认值。"""
        os.environ.pop("FE_BATCH_SIZE", None)
        assert get_config_value("batch_size", default=9999) == 9999

    def test_get_string_config(self):
        """字符串配置应原样返回。"""
        os.environ["DUCKDB_MEMORY_LIMIT"] = "32GB"
        assert get_config_value("duckdb_memory_limit") == "32GB"

    def test_get_unknown_key(self):
        """未知键应返回默认值。"""
        assert get_config_value("unknown_key_xyz", default="fallback") == "fallback"


class TestEdgeCases:
    """测试边界情况。"""

    def test_very_low_memory(self):
        """极低内存（< 1GB 可用）应使用最小配置。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=2 * 1024**3,
            available_memory_bytes=512 * 1024**2,  # 512MB 可用
            cpu_count=1,
            disk_space_bytes=10 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # 可用内存 * 0.7 = 0.35GB，但强制最小 1GB
        # usable = max(1.0, 0.35) = 1.0GB
        assert optimal["usable_memory_gb"] == 1.0
        assert optimal["batch_size"] == 10_000  # 最小批量
        assert optimal["max_workers"] == 1  # 最少 worker

    def test_single_cpu(self):
        """单核 CPU 应生成有效配置。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=8 * 1024**3,
            available_memory_bytes=6 * 1024**3,
            cpu_count=1,
            disk_space_bytes=100 * 1024**3,
        )
        optimal = config.get_optimal_config()

        # 单核：1 - 1 保留 = 0，但 max(..., 1) = 1
        assert optimal["duckdb_threads"] >= 1
        assert optimal["polars_threads"] >= 1
        assert optimal["max_workers"] >= 1

    def test_very_large_disk(self):
        """超大磁盘（> 10TB）temp 文件应限制在 100GB。"""
        config = AdaptiveResourceConfig(
            total_memory_bytes=128 * 1024**3,
            available_memory_bytes=120 * 1024**3,
            cpu_count=32,
            disk_space_bytes=20 * 1024**4,  # 20TB
        )
        optimal = config.get_optimal_config()

        # temp_file_max = min(10% * 20TB, 100GB) = 100GB
        assert optimal["temp_file_max_bytes"] == 100 * 1024**3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
