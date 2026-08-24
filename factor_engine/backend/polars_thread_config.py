# -*- coding: utf-8 -*-
"""Polars 线程池配置与动态调优。

IMPORTANT - THREADING MODEL:
================================================================================
Polars thread pool size is FIXED at import time via POLARS_MAX_THREADS env var.
Runtime calls to pl.Config.set_thread_count() do NOT work reliably and can
cause undefined behavior.

CORRECT APPROACH:
- Set POLARS_MAX_THREADS environment variable BEFORE importing polars
- ResourceBroker should control the NUMBER OF CONCURRENT POLARS JOBS, not thread count
- Each Polars job uses the fixed thread pool; concurrency control limits how many
  jobs run simultaneously

Example:
    # At process startup (before any polars import)
    os.environ['POLARS_MAX_THREADS'] = str(physical_cores)

    # At runtime, ResourceBroker controls job concurrency:
    with resource_broker.acquire(cpu_tokens=4):
        # Run 1 Polars job using the fixed thread pool
        df = lf.collect()

    # To run fewer concurrent jobs, reduce semaphore count, not thread pool size

DEPRECATED: All pl.Config.set_thread_count() calls in this file are disabled.
================================================================================

根据 ResourceBroker CPU 预算和任务类型动态调整 Polars 线程数，目标：
- CPU 密集型：吞吐提升 30-50%（充分利用多核）
- IO 密集型：适度超订（1.5x），隐藏 IO 等待
- 资源隔离：多任务并发时各自独立线程预算
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Literal

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


TaskType = Literal["compute", "io", "mixed"]


def get_physical_cores() -> int:
    """获取物理核心数（非超线程）。

    Returns:
        物理核心数；获取失败时回退到逻辑核心数。
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return physical
    except ImportError:
        pass

    # 回退：逻辑核心数
    logical = os.cpu_count()
    if logical and logical > 0:
        # 保守估计：假设超线程 2x
        return max(1, logical // 2)

    return 4  # 最终默认


def optimal_polars_threads(
    cpu_budget: int,
    task_type: TaskType = "mixed",
    *,
    physical_cores: int | None = None,
) -> int:
    """根据 CPU 预算与任务类型计算最优 Polars 线程数。

    Args:
        cpu_budget: ResourceBroker 分配的 CPU token 数。
        task_type: 任务类型：
            - "compute": 纯 CPU 计算（严格遵守预算）
            - "io": IO 密集型（允许 1.5x 超订）
            - "mixed": 混合型（1.2x 适度超订）
        physical_cores: 可选物理核心数（None 时自动探测）。

    Returns:
        推荐的 Polars 线程数（1 到 physical_cores）。

    策略:
        - compute: min(budget, physical_cores)  # 严格
        - io: min(budget * 1.5, physical_cores)  # 超订
        - mixed: min(budget * 1.2, physical_cores)  # 适度
    """
    if physical_cores is None:
        physical_cores = get_physical_cores()

    if task_type == "io":
        # IO 任务：1.5x 超订（IO 等待时 CPU 可调度其他线程）
        target = int(cpu_budget * 1.5)
    elif task_type == "compute":
        # CPU 任务：严格遵守预算（避免过度竞争）
        target = cpu_budget
    else:  # mixed
        # 混合：1.2x 适度超订
        target = int(cpu_budget * 1.2)

    return max(1, min(target, physical_cores))


@contextmanager
def polars_thread_budget(threads: int):
    """DEPRECATED: Runtime thread control does not work reliably in Polars.

    用法:
        # WRONG (doesn't work):
        # with polars_thread_budget(4):
        #     df = lf.collect()

        # CORRECT: Control job concurrency instead
        with resource_broker.acquire(cpu_tokens=threads):
            df = lf.collect()  # Uses fixed thread pool

    注意:
        - Polars thread pool is fixed at import time via POLARS_MAX_THREADS
        - Runtime modifications via pl.Config.set_thread_count() are unreliable
        - ResourceBroker should limit CONCURRENT JOBS, not threads per job
        - This function is now a no-op and will be removed
    """
    # NO-OP: Runtime thread changes don't work reliably
    # The thread pool size was fixed when polars was imported
    yield


def configure_polars_for_execution(
    max_workers: int | None = None,
    memory_mb: float | None = None,
) -> dict[str, int]:
    """配置 Polars 全局执行参数（线程数、流式块大小）。

    DEPRECATED: Runtime thread configuration does not work.

    This function historically attempted to set thread count at runtime,
    but Polars thread pool is fixed at import time.

    Args:
        max_workers: IGNORED - thread count must be set via POLARS_MAX_THREADS before import
        memory_mb: Memory limit (MB); affects streaming chunk size.

    Returns:
        配置快照字典（用于日志/监控）。

    用法:
        # CORRECT: Set before importing polars
        import os
        os.environ['POLARS_MAX_THREADS'] = str(8)
        import polars as pl  # Thread pool is now fixed at 8

        # WRONG: Trying to change at runtime (doesn't work)
        # configure_polars_for_execution(max_workers=4)  # Has no effect
    """
    if pl is None:
        return {}

    # 1. Thread count - READ ONLY (cannot be changed at runtime)
    try:
        current_threads = pl.thread_pool_size() if hasattr(pl, 'thread_pool_size') else None
    except Exception:
        current_threads = None

    if current_threads is None:
        # Try to read from environment
        env_threads = os.environ.get("POLARS_MAX_THREADS")
        current_threads = int(env_threads) if env_threads else get_physical_cores()

    # 2. 流式块大小 - this CAN be configured at runtime
    if memory_mb is not None and memory_mb < 2000:
        chunk_size = 50_000
    else:
        # 使用自适应配置
        try:
            from factor_engine.runtime.adaptive_config import get_global_adaptive_config
            chunk_size = get_global_adaptive_config().polars_streaming_chunk_size
        except ImportError:
            chunk_size = 100_000  # 回退默认值

    # Apply streaming config only (thread count cannot be changed)
    if hasattr(pl, "Config"):
        try:
            pl.Config.set_streaming_chunk_size(chunk_size)
        except AttributeError:
            # 部分版本无此 API
            pass

    return {
        "thread_count": current_threads,  # Read-only, already fixed
        "streaming_chunk_size": chunk_size,
    }


def infer_task_type_from_plan(plan: Any) -> TaskType:
    """从逻辑计划推断任务类型（启发式）。

    Args:
        plan: PlanNode 逻辑计划。

    Returns:
        推断的任务类型（compute/io/mixed）。

    启发式规则:
        - 有 column 引用 → io（需要扫描）
        - 纯 literal/plan_ref/materialized_series → compute
        - 混合 → mixed
    """
    from factor_engine.planner.logical_plan import PlanNode

    if not isinstance(plan, PlanNode):
        return "mixed"

    def has_column_refs(node: PlanNode) -> bool:
        """递归检查是否有列引用。"""
        if node.op == "column":
            return True
        return any(has_column_refs(c) for c in node.inputs)

    def has_heavy_compute(node: PlanNode) -> bool:
        """检查是否有重计算算子。"""
        heavy_ops = {
            "ts_corr", "ts_cov", "ts_beta",  # 双序列窗口
            "ts_skew", "ts_kurt",  # 高阶矩
            "expanding_std", "quantile",  # map_groups
        }
        if node.op in heavy_ops:
            return True
        return any(has_heavy_compute(c) for c in node.inputs)

    has_io = has_column_refs(plan)
    has_compute = has_heavy_compute(plan)

    if has_io and not has_compute:
        return "io"
    elif has_compute and not has_io:
        return "compute"
    else:
        return "mixed"


def get_polars_thread_config_summary() -> dict[str, Any]:
    """获取当前 Polars 线程配置快照（调试/监控用）。

    Returns:
        包含 thread_count、version 等信息的字典。
    """
    if pl is None:
        return {"available": False}

    info: dict[str, Any] = {"available": True}

    try:
        info["version"] = pl.__version__
    except AttributeError:
        info["version"] = "unknown"

    try:
        info["thread_count"] = pl.thread_pool_size()
    except AttributeError:
        env = os.environ.get("POLARS_MAX_THREADS")
        info["thread_count"] = int(env) if env else None

    try:
        info["physical_cores"] = get_physical_cores()
    except Exception:
        info["physical_cores"] = None

    return info
