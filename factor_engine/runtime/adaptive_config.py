# -*- coding: utf-8 -*-
"""统一的自适应资源配置系统。

根据系统实际资源（内存、CPU）自动调整所有资源相关配置，避免硬编码。

设计原则：
1. 感知实际硬件：探测物理内存、CPU核心数
2. 自适应缩放：小内存环境保守配置，大内存环境充分利用
3. 环境变量覆盖：允许用户手动调整
4. 向后兼容：保留原有环境变量支持

2026-08-13: 初版实现，替代所有硬编码配置。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


def _get_system_memory_gb() -> float:
    """获取系统总内存（GB）。"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        _logger.warning("psutil not available, assuming 16GB RAM")
        return 16.0


def _get_cpu_count() -> int:
    """获取CPU核心数（物理核心）。"""
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return physical
    except ImportError:
        pass

    # 回退：逻辑核心数的一半（保守估计超线程）
    logical = os.cpu_count()
    if logical and logical > 0:
        return max(1, logical // 2)

    return 4  # 最终默认


@dataclass(frozen=True)
class AdaptiveResourceConfig:
    """自适应资源配置（基于系统实际资源计算）。

    所有配置根据实际硬件资源自动缩放：
    - 小内存环境（< 16GB）：保守配置
    - 中等内存环境（16-64GB）：平衡配置
    - 大内存环境（> 64GB）：充分利用
    """

    # ========== 系统探测结果 ==========
    system_memory_gb: float
    system_cpu_cores: int

    # ========== DuckDB 配置 ==========
    duckdb_threads: int
    duckdb_memory_limit: str  # 例如 "15GB"
    duckdb_memory_limit_mb: int

    # ========== Polars 配置 ==========
    polars_threads: int
    polars_streaming_chunk_size: int

    # ========== 批处理配置 ==========
    compile_chunk_size: int  # compile_many_chunked 的 chunk_size
    dag_chunk_size: int  # DAG 处理的 chunk_size
    batch_size: int  # 批量处理大小

    # ========== 内存块配置 ==========
    block_abs_max_bytes: int  # 单个块最大内存（BLOCK_ABS_MAX）
    cache_size_bytes: int  # 缓存大小
    streaming_threshold_bytes: int  # 流式处理阈值

    # ========== 并发配置 ==========
    max_workers_io: int  # IO 密集型线程池大小
    max_workers_compute: int  # CPU 密集型线程池大小

    # ========== 资源预算 ==========
    hard_memory_limit_bytes: int  # ResourceBroker 硬内存限制
    safe_envelope_bytes: int  # OOM replan 安全阈值
    compile_budget_bytes: int  # 编译阶段内存预算

    # ========== 配置来源 ==========
    config_source: dict[str, str]  # 记录每个配置的来源（system/env/default）

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于日志/序列化）。"""
        return {
            "system": {
                "memory_gb": self.system_memory_gb,
                "cpu_cores": self.system_cpu_cores,
            },
            "duckdb": {
                "threads": self.duckdb_threads,
                "memory_limit": self.duckdb_memory_limit,
                "memory_limit_mb": self.duckdb_memory_limit_mb,
            },
            "polars": {
                "threads": self.polars_threads,
                "streaming_chunk_size": self.polars_streaming_chunk_size,
            },
            "batch": {
                "compile_chunk_size": self.compile_chunk_size,
                "dag_chunk_size": self.dag_chunk_size,
                "batch_size": self.batch_size,
            },
            "memory": {
                "block_abs_max_bytes": self.block_abs_max_bytes,
                "cache_size_bytes": self.cache_size_bytes,
                "streaming_threshold_bytes": self.streaming_threshold_bytes,
                "hard_memory_limit_bytes": self.hard_memory_limit_bytes,
                "safe_envelope_bytes": self.safe_envelope_bytes,
                "compile_budget_bytes": self.compile_budget_bytes,
            },
            "concurrency": {
                "max_workers_io": self.max_workers_io,
                "max_workers_compute": self.max_workers_compute,
            },
            "config_source": self.config_source,
        }


def _adaptive_scale(
    base_value: float,
    memory_gb: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    scale_power: float = 0.5,
) -> int:
    """根据内存大小自适应缩放配置值。

    Args:
        base_value: 基准值（在 30GB 内存下的配置）
        memory_gb: 实际内存大小（GB）
        min_value: 最小值限制
        max_value: 最大值限制
        scale_power: 缩放幂次（0.5 = 平方根缩放，1.0 = 线性缩放）

    Returns:
        缩放后的整数值

    示例:
        30GB 基准 -> 15GB 环境：缩放 0.71x
        30GB 基准 -> 60GB 环境：缩放 1.41x
        30GB 基准 -> 500GB 环境：缩放 4.08x
    """
    # 缩放因子：(实际内存 / 30GB) ** scale_power
    scale_factor = (memory_gb / 30.0) ** scale_power
    scaled = base_value * scale_factor

    if min_value is not None:
        scaled = max(min_value, scaled)
    if max_value is not None:
        scaled = min(max_value, scaled)

    return int(scaled)


def get_adaptive_config(
    *,
    force_memory_gb: float | None = None,
    force_cpu_cores: int | None = None,
) -> AdaptiveResourceConfig:
    """获取自适应资源配置（基于实际系统资源）。

    Args:
        force_memory_gb: 强制指定内存大小（用于测试）
        force_cpu_cores: 强制指定CPU核心数（用于测试）

    Returns:
        自适应资源配置对象

    环境变量覆盖（优先级最高）：
        DUCKDB_THREADS: DuckDB 线程数
        DUCKDB_MEMORY_LIMIT_MB: DuckDB 内存限制（MB）
        POLARS_MAX_THREADS: Polars 线程数
        COMPILE_CHUNK_SIZE: compile_many_chunked 的 chunk_size
        MAX_WORKERS_IO: IO 线程池大小
        MAX_WORKERS_COMPUTE: CPU 线程池大小
    """
    # 1. 探测系统资源
    memory_gb = force_memory_gb or _get_system_memory_gb()
    cpu_cores = force_cpu_cores or _get_cpu_count()

    config_source: dict[str, str] = {}

    # 2. DuckDB 配置（基于 30GB 系统的最佳实践）
    # - 线程数：8（benchmark 最佳）
    # - 内存：50% 系统内存
    duckdb_threads_env = os.environ.get("DUCKDB_THREADS")
    if duckdb_threads_env:
        duckdb_threads = int(duckdb_threads_env)
        config_source["duckdb_threads"] = "env"
    else:
        duckdb_threads = min(8, cpu_cores)
        config_source["duckdb_threads"] = "adaptive"

    duckdb_memory_mb_env = os.environ.get("DUCKDB_MEMORY_LIMIT_MB")
    if duckdb_memory_mb_env:
        duckdb_memory_mb = int(duckdb_memory_mb_env)
        config_source["duckdb_memory_limit_mb"] = "env"
    else:
        # 50% 系统内存
        duckdb_memory_mb = int(memory_gb * 1024 * 0.5)
        config_source["duckdb_memory_limit_mb"] = "adaptive"

    duckdb_memory_limit = f"{duckdb_memory_mb}MB"

    # 3. Polars 配置
    polars_threads_env = os.environ.get("POLARS_MAX_THREADS")
    if polars_threads_env:
        polars_threads = int(polars_threads_env)
        config_source["polars_threads"] = "env"
    else:
        polars_threads = cpu_cores
        config_source["polars_threads"] = "adaptive"

    # 流式块大小：小内存环境用小块，大内存环境用大块
    if memory_gb < 16:
        polars_streaming_chunk_size = 50_000
    else:
        polars_streaming_chunk_size = 100_000
    config_source["polars_streaming_chunk_size"] = "adaptive"

    # 4. 批处理配置
    compile_chunk_env = os.environ.get("COMPILE_CHUNK_SIZE")
    if compile_chunk_env:
        compile_chunk_size = int(compile_chunk_env)
        config_source["compile_chunk_size"] = "env"
    else:
        # 基准：30GB 系统 -> 500，自适应缩放
        compile_chunk_size = _adaptive_scale(500, memory_gb, min_value=100, max_value=2000)
        config_source["compile_chunk_size"] = "adaptive"

    # DAG chunk size（较小，避免内存峰值）
    dag_chunk_size = _adaptive_scale(1000, memory_gb, min_value=500, max_value=5000)
    config_source["dag_chunk_size"] = "adaptive"

    # 通用 batch size
    batch_size = _adaptive_scale(100_000, memory_gb, min_value=10_000, max_value=500_000)
    config_source["batch_size"] = "adaptive"

    # 5. 内存块配置
    # BLOCK_ABS_MAX: 30GB 系统 -> 2GB
    block_abs_max_bytes = _adaptive_scale(
        2 * 1024**3, memory_gb,
        min_value=512 * 1024**2,  # 最小 512MB
        max_value=16 * 1024**3,   # 最大 16GB
    )
    config_source["block_abs_max_bytes"] = "adaptive"

    # 缓存大小（与 block_abs_max 一致）
    cache_size_bytes = block_abs_max_bytes
    config_source["cache_size_bytes"] = "adaptive"

    # 流式处理阈值（与 block_abs_max 一致）
    streaming_threshold_bytes = block_abs_max_bytes
    config_source["streaming_threshold_bytes"] = "adaptive"

    # 6. 资源预算
    # ResourceBroker hard_memory_limit: 30GB 系统 -> 8GB (约 27%)
    hard_memory_limit_bytes = _adaptive_scale(
        8 * 1024**3, memory_gb,
        min_value=2 * 1024**3,   # 最小 2GB
        max_value=128 * 1024**3, # 最大 128GB
    )
    config_source["hard_memory_limit_bytes"] = "adaptive"

    # OOM replan 安全阈值: 30GB 系统 -> 2GB
    safe_envelope_bytes = _adaptive_scale(
        2 * 1024**3, memory_gb,
        min_value=512 * 1024**2, # 最小 512MB
        max_value=16 * 1024**3,  # 最大 16GB
    )
    config_source["safe_envelope_bytes"] = "adaptive"

    # 编译阶段内存预算（与 block_abs_max 一致）
    compile_budget_bytes = block_abs_max_bytes
    config_source["compile_budget_bytes"] = "adaptive"

    # 7. 并发配置
    max_workers_io_env = os.environ.get("MAX_WORKERS_IO")
    if max_workers_io_env:
        max_workers_io = int(max_workers_io_env)
        config_source["max_workers_io"] = "env"
    else:
        # IO 密集型：允许超订（1.5x 物理核心）
        max_workers_io = min(cpu_cores * 3 // 2, 16)
        config_source["max_workers_io"] = "adaptive"

    max_workers_compute_env = os.environ.get("MAX_WORKERS_COMPUTE")
    if max_workers_compute_env:
        max_workers_compute = int(max_workers_compute_env)
        config_source["max_workers_compute"] = "env"
    else:
        # CPU 密集型：严格遵守物理核心数
        max_workers_compute = cpu_cores
        config_source["max_workers_compute"] = "adaptive"

    config = AdaptiveResourceConfig(
        system_memory_gb=memory_gb,
        system_cpu_cores=cpu_cores,
        duckdb_threads=duckdb_threads,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_memory_limit_mb=duckdb_memory_mb,
        polars_threads=polars_threads,
        polars_streaming_chunk_size=polars_streaming_chunk_size,
        compile_chunk_size=compile_chunk_size,
        dag_chunk_size=dag_chunk_size,
        batch_size=batch_size,
        block_abs_max_bytes=block_abs_max_bytes,
        cache_size_bytes=cache_size_bytes,
        streaming_threshold_bytes=streaming_threshold_bytes,
        max_workers_io=max_workers_io,
        max_workers_compute=max_workers_compute,
        hard_memory_limit_bytes=hard_memory_limit_bytes,
        safe_envelope_bytes=safe_envelope_bytes,
        compile_budget_bytes=compile_budget_bytes,
        config_source=config_source,
    )

    _logger.info(
        f"Adaptive config: {memory_gb:.1f}GB RAM, {cpu_cores} cores -> "
        f"DuckDB {duckdb_threads}t/{duckdb_memory_limit}, "
        f"Polars {polars_threads}t, "
        f"compile_chunk={compile_chunk_size}, "
        f"hard_limit={hard_memory_limit_bytes / 1024**3:.1f}GB"
    )

    return config


# 全局单例（延迟初始化）
_GLOBAL_CONFIG: AdaptiveResourceConfig | None = None


def get_global_adaptive_config() -> AdaptiveResourceConfig:
    """获取全局自适应配置单例。

    注意：首次调用时初始化，后续调用返回缓存的配置。
    如果需要重新探测系统资源，调用 reset_global_adaptive_config()。
    """
    global _GLOBAL_CONFIG

    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = get_adaptive_config()

    return _GLOBAL_CONFIG


def reset_global_adaptive_config() -> None:
    """重置全局配置（重新探测系统资源）。"""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None


# 便捷访问函数（向后兼容）
def get_duckdb_threads() -> int:
    """获取 DuckDB 推荐线程数。"""
    return get_global_adaptive_config().duckdb_threads


def get_duckdb_memory_limit() -> str:
    """获取 DuckDB 内存限制字符串（例如 "15GB"）。"""
    return get_global_adaptive_config().duckdb_memory_limit


def get_polars_threads() -> int:
    """获取 Polars 推荐线程数。"""
    return get_global_adaptive_config().polars_threads


def get_compile_chunk_size() -> int:
    """获取 compile_many_chunked 的 chunk_size。"""
    return get_global_adaptive_config().compile_chunk_size


def get_hard_memory_limit_bytes() -> int:
    """获取 ResourceBroker 硬内存限制（字节）。"""
    return get_global_adaptive_config().hard_memory_limit_bytes


def get_block_abs_max_bytes() -> int:
    """获取单个内存块最大大小（字节）。"""
    return get_global_adaptive_config().block_abs_max_bytes


def auto_configure(
    apply_env: bool = True,
    force_memory_gb: float | None = None,
    force_cpu_cores: int | None = None,
) -> dict[str, Any]:
    """启动时自动配置系统资源（推荐在 __init__.py 或 main 入口调用）。

    功能：
        1. 自动检测系统资源（内存、CPU）
        2. 计算最优配置参数
        3. 可选：自动设置环境变量（DuckDB/Polars threads 等）
        4. 返回配置字典供应用使用

    Args:
        apply_env: 是否自动设置环境变量（默认 True）
        force_memory_gb: 强制指定内存大小（用于测试/覆盖）
        force_cpu_cores: 强制指定 CPU 核心数（用于测试/覆盖）

    Returns:
        配置字典，包含所有自适应配置参数

    Example:
        >>> # 在应用启动时调用
        >>> from runtime.adaptive_config import auto_configure
        >>> config = auto_configure()
        >>> print(f"Auto-configured for {config['system_memory_gb']:.1f}GB memory")
        >>> print(f"DuckDB: {config['duckdb_threads']} threads, {config['duckdb_memory_limit']}")

    Environment Variables (尊重已有配置，优先级最高):
        - DUCKDB_THREADS: DuckDB 线程数
        - DUCKDB_MEMORY_LIMIT: DuckDB 内存限制（如 "16GB"）
        - POLARS_MAX_THREADS: Polars 线程数
        - FE_BATCH_SIZE: 批量处理大小
        - FE_DAG_CHUNK_SIZE: DAG 编译批量
        - FE_MAX_WORKERS: 最大并行工作数
    """
    # 获取自适应配置
    config_obj = get_adaptive_config(
        force_memory_gb=force_memory_gb,
        force_cpu_cores=force_cpu_cores,
    )

    # 转换为字典
    config = {
        # 系统资源
        "system_memory_gb": config_obj.system_memory_gb,
        "system_cpu_cores": config_obj.system_cpu_cores,

        # DuckDB 配置
        "duckdb_threads": config_obj.duckdb_threads,
        "duckdb_memory_limit": config_obj.duckdb_memory_limit,
        "duckdb_memory_limit_mb": config_obj.duckdb_memory_limit_mb,

        # Polars 配置
        "polars_threads": config_obj.polars_threads,
        "polars_streaming_chunk_size": config_obj.polars_streaming_chunk_size,

        # 批处理配置
        "compile_chunk_size": config_obj.compile_chunk_size,
        "dag_chunk_size": config_obj.dag_chunk_size,
        "batch_size": config_obj.batch_size,

        # 内存配置
        "block_abs_max_bytes": config_obj.block_abs_max_bytes,
        "cache_size_bytes": config_obj.cache_size_bytes,
        "streaming_threshold_bytes": config_obj.streaming_threshold_bytes,
        "hard_memory_limit_bytes": config_obj.hard_memory_limit_bytes,
        "safe_envelope_bytes": config_obj.safe_envelope_bytes,
        "compile_budget_bytes": config_obj.compile_budget_bytes,

        # 并发配置
        "max_workers_io": config_obj.max_workers_io,
        "max_workers_compute": config_obj.max_workers_compute,

        # 配置来源
        "config_source": config_obj.config_source,
    }

    # 应用环境变量（如果启用且未被用户显式设置）
    if apply_env:
        # DuckDB 配置
        if "DUCKDB_THREADS" not in os.environ:
            os.environ["DUCKDB_THREADS"] = str(config["duckdb_threads"])

        if "DUCKDB_MEMORY_LIMIT" not in os.environ:
            os.environ["DUCKDB_MEMORY_LIMIT"] = config["duckdb_memory_limit"]

        # Polars 配置
        if "POLARS_MAX_THREADS" not in os.environ:
            os.environ["POLARS_MAX_THREADS"] = str(config["polars_threads"])

        # 自定义环境变量（供其他模块读取）
        if "FE_BATCH_SIZE" not in os.environ:
            os.environ["FE_BATCH_SIZE"] = str(config["batch_size"])

        if "FE_DAG_CHUNK_SIZE" not in os.environ:
            os.environ["FE_DAG_CHUNK_SIZE"] = str(config["dag_chunk_size"])

        if "FE_MAX_WORKERS" not in os.environ:
            os.environ["FE_MAX_WORKERS"] = str(config["max_workers_compute"])

    # 日志输出（简洁版，详细日志在 get_adaptive_config 中）
    _logger.info(
        f"auto_configure: {config['system_memory_gb']:.1f}GB memory, "
        f"{config['system_cpu_cores']} CPU cores detected"
    )
    _logger.info(
        f"  DuckDB: {config['duckdb_threads']} threads, {config['duckdb_memory_limit']}"
    )
    _logger.info(
        f"  Batch size: {config['batch_size']}, DAG chunk: {config['dag_chunk_size']}"
    )
    _logger.info(
        f"  Workers: {config['max_workers_compute']} compute, {config['max_workers_io']} I/O"
    )

    return config
