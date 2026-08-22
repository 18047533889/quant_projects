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
    """获取系统总内存（GB），考虑 cgroup 限制。

    Returns:
        系统可用内存（GB），取 host 总内存和 cgroup 限制的较小值

    Notes:
        支持 cgroup v1 和 v2，确保容器内正确检测内存限制。
    """
    # 1. 获取 host 总内存
    host_memory_bytes = None
    try:
        import psutil
        host_memory_bytes = psutil.virtual_memory().total
    except ImportError:
        _logger.warning("psutil not available, assuming 16GB RAM")
        return 16.0

    # 2. 检查 cgroup 限制（容器环境）
    cgroup_limit_bytes = None

    # cgroup v2: /sys/fs/cgroup/memory.max
    try:
        with open("/sys/fs/cgroup/memory.max", "r") as f:
            content = f.read().strip()
            if content != "max":  # "max" = 无限制
                cgroup_limit_bytes = int(content)
                _logger.debug(f"cgroup v2 memory.max = {cgroup_limit_bytes / 1024**3:.2f}GB")
    except (FileNotFoundError, PermissionError, ValueError):
        pass

    # cgroup v1: /sys/fs/cgroup/memory/memory.limit_in_bytes
    if cgroup_limit_bytes is None:
        try:
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                limit = int(f.read().strip())
                # cgroup v1 使用巨大哨兵值表示无限制（如 9223372036854771712）
                if limit < (1 << 62):  # 小于 4 EiB 视为真实限制
                    cgroup_limit_bytes = limit
                    _logger.debug(f"cgroup v1 memory.limit_in_bytes = {cgroup_limit_bytes / 1024**3:.2f}GB")
        except (FileNotFoundError, PermissionError, ValueError):
            pass

    # 3. 取较小值（host vs cgroup）
    if cgroup_limit_bytes is not None:
        effective_bytes = min(host_memory_bytes, cgroup_limit_bytes)
        if effective_bytes < host_memory_bytes:
            _logger.info(
                f"Running in container: host={host_memory_bytes/1024**3:.1f}GB, "
                f"cgroup={cgroup_limit_bytes/1024**3:.1f}GB, using {effective_bytes/1024**3:.1f}GB"
            )
        return effective_bytes / (1024**3)

    return host_memory_bytes / (1024**3)


def _get_cpu_count() -> int:
    """获取可用 CPU 核心数（考虑 affinity 和 cgroup 配额）。

    Returns:
        可用的逻辑 CPU 核心数

    Notes:
        优先使用 sched_getaffinity（容器/affinity 感知），考虑 cgroup CPU 配额。
        返回**逻辑核心数**而非物理核心，因为：
        1. 容器和 cgroup 配额以逻辑核心计量
        2. DuckDB/Polars 均以逻辑核心优化调度
        3. 物理核心数在超线程环境下会浪费一半计算资源
    """
    available_cores = None

    # 1. 优先使用 sched_getaffinity（容器/CPU affinity 感知）
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
            available_cores = len(affinity)
            _logger.debug(f"sched_getaffinity reports {available_cores} cores")
        except (OSError, AttributeError):
            pass

    # 2. 回退到 os.cpu_count()（逻辑核心）
    if available_cores is None:
        available_cores = os.cpu_count()
        if available_cores:
            _logger.debug(f"os.cpu_count reports {available_cores} cores")

    # 3. 检查 cgroup CPU 配额（容器可能限制 CPU 使用）
    cgroup_quota_cores = None

    # cgroup v2: /sys/fs/cgroup/cpu.max (格式: "quota period" 或 "max period")
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            line = f.read().strip()
            parts = line.split()
            if parts[0] != "max":
                quota_us = int(parts[0])
                period_us = int(parts[1]) if len(parts) > 1 else 100000
                cgroup_quota_cores = max(1, quota_us // period_us)
                _logger.debug(f"cgroup v2 cpu.max = {cgroup_quota_cores} cores")
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass

    # cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us 和 cpu.cfs_period_us
    if cgroup_quota_cores is None:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                quota_us = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                period_us = int(f.read().strip())
            if quota_us > 0:  # -1 表示无限制
                cgroup_quota_cores = max(1, quota_us // period_us)
                _logger.debug(f"cgroup v1 cpu quota = {cgroup_quota_cores} cores")
        except (FileNotFoundError, PermissionError, ValueError):
            pass

    # 4. 取较小值（affinity/cpu_count vs cgroup 配额）
    if available_cores is None:
        return 4  # 最终默认

    if cgroup_quota_cores is not None:
        effective_cores = min(available_cores, cgroup_quota_cores)
        if effective_cores < available_cores:
            _logger.info(
                f"cgroup CPU quota: available={available_cores}, quota={cgroup_quota_cores}, "
                f"using {effective_cores}"
            )
        return max(1, effective_cores)

    return max(1, available_cores)


def _compute_host_fraction(memory_gb: float) -> float:
    """计算可用于 FE 的 host 内存比例（小机器保守，大机器激进）。

    Args:
        memory_gb: 系统总内存（GB）

    Returns:
        可用比例 [0.0, 1.0]

    Rationale:
        OS、Python 解释器、其他服务占用的绝对内存大致恒定（~1-2GB），
        因此在小机器上占比更高。我们必须留出更大的缓冲。
        大机器上这些开销占比很小，可以更激进地使用内存。

        比例设计：
        - 4GB:  55%  (留 1.8GB 给 OS/Python，FE 用 2.2GB)
        - 8GB:  60%  (留 3.2GB，FE 用 4.8GB)
        - 16GB: 65%  (留 5.6GB，FE 用 10.4GB)
        - 32GB: 72%  (留 9GB，FE 用 23GB)
        - 64GB: 78%  (留 14GB，FE 用 50GB)
        - 128GB+: 82-85%
    """
    if memory_gb <= 4:
        return 0.55
    elif memory_gb <= 8:
        return 0.60
    elif memory_gb <= 16:
        return 0.65
    elif memory_gb <= 32:
        return 0.72
    elif memory_gb <= 64:
        return 0.78
    elif memory_gb <= 128:
        return 0.82
    else:
        return 0.85


def _get_duckdb_thread_limit() -> int:
    """获取 DuckDB 线程数上限。

    Returns:
        DuckDB 最大线程数

    Notes:
        基于 B01-B09 benchmark，8 线程在 30GB/8 核机器上表现最佳。
        环境变量 DUCKDB_MAX_THREADS 可覆盖此上限（用于大规模服务器调优）。
    """
    env_limit = os.environ.get("DUCKDB_MAX_THREADS")
    if env_limit:
        try:
            return int(env_limit)
        except ValueError:
            _logger.warning(f"Invalid DUCKDB_MAX_THREADS={env_limit}, using default 8")
    return 8
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
        scale_power: 缩放幂次
            - 0.5 = 平方根缩放，适用于 COUNT（编译块数、批量大小等），
              因为 per-item 成本会复合（O(n²) 或更高）
            - 1.0 = 线性缩放，适用于 ABSOLUTE BYTE BUDGETS（内存限制、缓存大小等），
              因为它们应占系统内存的固定比例

    Returns:
        缩放后的整数值

    示例（scale_power=0.5 用于 COUNT）:
        30GB 基准 -> 15GB 环境：缩放 0.71x
        30GB 基准 -> 60GB 环境：缩放 1.41x
        30GB 基准 -> 500GB 环境：缩放 4.08x

    示例（scale_power=1.0 用于 BYTE BUDGET）:
        30GB 基准 8GB -> 15GB 环境：缩放 0.5x (4GB)
        30GB 基准 8GB -> 60GB 环境：缩放 2.0x (16GB)
        30GB 基准 8GB -> 512GB 环境：缩放 17.1x (137GB)
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

    # 2. 计算 host 可用比例（小机器保守，大机器激进）
    host_fraction = _compute_host_fraction(memory_gb)

    # 3. DuckDB 配置
    # - 线程数：基于 benchmark，默认上限 8（可通过 DUCKDB_MAX_THREADS 覆盖）
    # - 内存：40% 系统内存（保守，避免与其他组件竞争）
    duckdb_threads_env = os.environ.get("DUCKDB_THREADS")
    if duckdb_threads_env:
        duckdb_threads = int(duckdb_threads_env)
        config_source["duckdb_threads"] = "env"
    else:
        duckdb_threads = min(_get_duckdb_thread_limit(), cpu_cores)
        config_source["duckdb_threads"] = "adaptive"

    duckdb_memory_mb_env = os.environ.get("DUCKDB_MEMORY_LIMIT_MB")
    if duckdb_memory_mb_env:
        duckdb_memory_mb = int(duckdb_memory_mb_env)
        config_source["duckdb_memory_limit_mb"] = "env"
    else:
        # 40% 系统内存（DuckDB 是子系统，不应占据过半资源）
        duckdb_memory_mb = int(memory_gb * 1024 * 0.40)
        config_source["duckdb_memory_limit_mb"] = "adaptive"

    duckdb_memory_limit = f"{duckdb_memory_mb}MB"

    # 4. Polars 配置
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

    # 5. 批处理配置（COUNT 类型，使用 sqrt 缩放）
    compile_chunk_env = os.environ.get("COMPILE_CHUNK_SIZE")
    if compile_chunk_env:
        compile_chunk_size = int(compile_chunk_env)
        config_source["compile_chunk_size"] = "env"
    else:
        # 基准：30GB 系统 -> 500，sqrt 缩放（scale_power=0.5）
        compile_chunk_size = _adaptive_scale(
            500, memory_gb, min_value=100, max_value=2000, scale_power=0.5
        )
        config_source["compile_chunk_size"] = "adaptive"

    # DAG chunk size（较小，避免内存峰值）
    dag_chunk_size = _adaptive_scale(
        1000, memory_gb, min_value=500, max_value=5000, scale_power=0.5
    )
    config_source["dag_chunk_size"] = "adaptive"

    # 通用 batch size
    batch_size = _adaptive_scale(
        100_000, memory_gb, min_value=10_000, max_value=500_000, scale_power=0.5
    )
    config_source["batch_size"] = "adaptive"

    # 6. 内存预算配置（BYTE BUDGET 类型，使用线性缩放 scale_power=1.0）
    # 所有绝对字节预算应占系统内存的固定比例
    total_bytes = memory_gb * (1024**3)

    # hard_memory_limit: FE 总预算的主要限制（host_fraction 的 85%）
    hard_memory_limit_bytes = int(total_bytes * host_fraction * 0.85)
    hard_memory_limit_bytes = max(
        int(2 * 1024**3),    # 最小 2GB
        min(hard_memory_limit_bytes, int(768 * 1024**3))  # 最大 768GB
    )
    config_source["hard_memory_limit_bytes"] = "adaptive"

    # safe_envelope: OOM replan 安全阈值（hard_limit 的 25%）
    safe_envelope_bytes = int(hard_memory_limit_bytes * 0.25)
    safe_envelope_bytes = max(
        int(512 * 1024**2),  # 最小 512MB
        min(safe_envelope_bytes, int(64 * 1024**3))  # 最大 64GB
    )
    config_source["safe_envelope_bytes"] = "adaptive"

    # block_abs_max: 单块内存上限（hard_limit 的 6%，用于单个算子结果）
    block_abs_max_bytes = int(hard_memory_limit_bytes * 0.06)
    block_abs_max_bytes = max(
        int(512 * 1024**2),  # 最小 512MB
        min(block_abs_max_bytes, int(32 * 1024**3))  # 最大 32GB
    )
    config_source["block_abs_max_bytes"] = "adaptive"

    # cache_size: CSE/cache 总容量（hard_limit 的 20%）
    cache_size_bytes = int(hard_memory_limit_bytes * 0.20)
    cache_size_bytes = max(
        int(512 * 1024**2),  # 最小 512MB
        min(cache_size_bytes, int(128 * 1024**3))  # 最大 128GB
    )
    config_source["cache_size_bytes"] = "adaptive"

    # streaming_threshold: 流式处理阈值（hard_limit 的 8%）
    streaming_threshold_bytes = int(hard_memory_limit_bytes * 0.08)
    streaming_threshold_bytes = max(
        int(256 * 1024**2),  # 最小 256MB
        min(streaming_threshold_bytes, int(32 * 1024**3))  # 最大 32GB
    )
    config_source["streaming_threshold_bytes"] = "adaptive"

    # compile_budget: 编译阶段预算（hard_limit 的 15%）
    compile_budget_bytes = int(hard_memory_limit_bytes * 0.15)
    compile_budget_bytes = max(
        int(512 * 1024**2),  # 最小 512MB
        min(compile_budget_bytes, int(64 * 1024**3))  # 最大 64GB
    )
    config_source["compile_budget_bytes"] = "adaptive"

    # 7. 验证内存层级一致性（CRITICAL INVARIANT）
    # DuckDB 不应超过 hard_limit，safe_envelope 应在 hard_limit 内
    duckdb_memory_bytes = duckdb_memory_mb * (1024**2)
    if duckdb_memory_bytes > hard_memory_limit_bytes:
        _logger.warning(
            f"DuckDB memory ({duckdb_memory_bytes / 1024**3:.1f}GB) exceeds "
            f"hard_memory_limit ({hard_memory_limit_bytes / 1024**3:.1f}GB), "
            f"capping DuckDB to 80% of hard_limit"
        )
        duckdb_memory_mb = int(hard_memory_limit_bytes * 0.80 / (1024**2))
        duckdb_memory_limit = f"{duckdb_memory_mb}MB"
        duckdb_memory_bytes = duckdb_memory_mb * (1024**2)  # Recompute after cap

    # 断言：确保层级关系永不倒挂
    assert safe_envelope_bytes <= hard_memory_limit_bytes, \
        f"safe_envelope ({safe_envelope_bytes}) > hard_limit ({hard_memory_limit_bytes})"
    assert duckdb_memory_bytes <= hard_memory_limit_bytes, \
        f"duckdb_memory ({duckdb_memory_bytes}) > hard_limit ({hard_memory_limit_bytes})"
    assert block_abs_max_bytes <= hard_memory_limit_bytes, \
        f"block_abs_max ({block_abs_max_bytes}) > hard_limit ({hard_memory_limit_bytes})"

    # 8. 并发配置
    max_workers_io_env = os.environ.get("MAX_WORKERS_IO")
    if max_workers_io_env:
        max_workers_io = int(max_workers_io_env)
        config_source["max_workers_io"] = "env"
    else:
        # IO 密集型：允许超订（1.5x 逻辑核心）
        max_workers_io = min(cpu_cores * 3 // 2, 16)
        config_source["max_workers_io"] = "adaptive"

    max_workers_compute_env = os.environ.get("MAX_WORKERS_COMPUTE")
    if max_workers_compute_env:
        max_workers_compute = int(max_workers_compute_env)
        config_source["max_workers_compute"] = "env"
    else:
        # CPU 密集型：等于可用逻辑核心数
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
