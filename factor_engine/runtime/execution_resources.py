# -*- coding: utf-8 -*-
"""执行资源预算（#34 → Phase 5 R1/R3）：协调 worker 数 / DuckDB threads / Polars
rayon threads / IO 并发，避免 oversubscription；worker 数同时受 CPU 与 RAM 约束。

背景
    如果 8 个 factor worker 各自跑一个 8-thread DuckDB 查询，在 16 核机器上会有
    64 个 runnable 线程——上下文切换反而拖慢。这里统一给
    ``n_jobs × duckdb_threads <= cpu_slots`` 的约束。

    Phase 5 起，真实的 CPU slot / 内存上限来自 :mod:`runtime.resource_governor`
    （cgroup / SLURM / RLIMIT / affinity 感知），worker 数额外受
    ``available_execution_memory / per_worker_peak`` 约束，避免 32 核 16GB 机器
    开 32 worker 各吃 3GB 直接 OOM。

用法
    plan = resource_plan(n_jobs=8)          # -> ResourcePlan（旧 API，兼容）
    eps  = build_execution_resource_plan()  # -> ExecutionResourcePlan（Phase 5 新 API）
    set_duckdb_max_threads(plan)            # 写入 DuckDBEngine 环境
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from runtime.resource_governor import (
    ExecutionResourcePlan,
    effective_cpu_slots,
    effective_memory_limit_bytes,
)

_cores: int | None = None


def physical_cores() -> int:
    """可用物理核数（缓存；Phase 5 起走 cgroup-aware ``effective_cpu_slots``）。

    R20-138：与 :func:`runtime.resource_governor.effective_cpu_slots` 完全对齐 ——
    env 显式值不再直接 return，而是被 hard CPU limit（cgroup quota / affinity）
    clamp。容器 2 CPU + ``FACTOR_ENGINE_CPU_BUDGET=16`` → 2（不是 16）。
    """
    global _cores
    if _cores is not None:
        return _cores
    _cores = effective_cpu_slots()
    return _cores


@dataclass(frozen=True)
class ResourcePlan:
    """一次并行执行的整体资源分配（旧 API，保留兼容）。"""

    n_jobs: int
    duckdb_threads: int
    polars_threads: int
    io_concurrency: int
    memory_budget_gb: float
    total_runnable: int
    memory_budget_bytes: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_jobs": self.n_jobs,
            "duckdb_threads": self.duckdb_threads,
            "polars_threads": self.polars_threads,
            "io_concurrency": self.io_concurrency,
            "memory_budget_gb": self.memory_budget_gb,
            "total_runnable": self.total_runnable,
            "physical_cores": physical_cores(),
        }


def _duckdb_base_threads() -> int:
    """DuckDB 单查询默认线程（DUCKDB_MAX_THREADS 优先）。"""
    env = os.environ.get("DUCKDB_MAX_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(2, min(physical_cores(), 8))


def _per_worker_peak_mb() -> float:
    """每 worker 峰值内存估计（MB）。存在时用于 RAM 约束 worker 数。"""
    env = os.environ.get("FACTOR_ENGINE_PER_WORKER_PEAK_MB", "").strip()
    if env.isdigit() and float(env) > 0:
        return float(env)
    # 默认乐观 3GB/worker；可由执行上下文覆盖（CSE/大分钟数据时调大）
    return 3.0


def resource_plan(n_jobs: int | None = None) -> ResourcePlan:
    """按可用核数与内存上限分配资源。

    约束：
        - ``n_jobs × duckdb_threads <= cpu_slots``
        - ``n_jobs <= floor(process_budget / per_worker_peak)``（RAM 硬约束）
    """
    eps = ExecutionResourcePlan.auto(max_workers=n_jobs)
    return ResourcePlan(
        n_jobs=eps.max_workers,
        duckdb_threads=eps.duckdb_threads,
        polars_threads=eps.polars_threads,
        io_concurrency=eps.io_concurrency,
        memory_budget_gb=round(eps.process_budget_bytes / 1024**3, 1),
        total_runnable=eps.max_workers * eps.duckdb_threads,
        memory_budget_bytes=eps.process_budget_bytes,
    )


def build_execution_resource_plan(
    config: dict | None = None,
    *,
    n_jobs: int | None = None,
) -> ExecutionResourcePlan:
    """Phase 5 新 API：构造完整 ``ExecutionResourcePlan``（可从 ``resources`` 配置）。"""
    if config:
        plan = ExecutionResourcePlan.from_dict(config)
        if n_jobs and n_jobs > 0:
            return _replan_workers(plan, n_jobs)
        return plan
    return ExecutionResourcePlan.auto(max_workers=n_jobs)


def _replan_workers(plan: ExecutionResourcePlan, n_jobs: int) -> ExecutionResourcePlan:
    """显式 n_jobs 覆盖：保持各预算不变，仅重算 worker/thread 组合。"""
    from dataclasses import replace

    workers = max(1, min(int(n_jobs), effective_cpu_slots()))
    cpu = effective_cpu_slots()
    duckdb_threads = max(1, min(plan.duckdb_threads, cpu // workers if workers else cpu))
    return replace(plan, max_workers=workers, duckdb_threads=duckdb_threads)


def set_duckdb_max_threads(plan: ResourcePlan | None = None, threads: int | None = None) -> int:
    """把 DuckDB 线程数写入环境变量（后续 DuckDBEngine 创建时读取），返回生效值。

    只改 env，**不**就地改已存在的共享引擎；需要就地调整时用
    ``ExecutionResourceScope`` 或 ``apply_live_duckdb_threads``。
    """
    value = threads if threads is not None else (plan.duckdb_threads if plan else None)
    if value is None:
        value = _duckdb_base_threads()
    os.environ["DUCKDB_MAX_THREADS"] = str(value)
    return value


def apply_live_duckdb_threads(threads: int | None = None) -> int:
    """就地调整当前共享 DuckDB 引擎的 ``PRAGMA threads``（batch 运行时用）。

    返回实际生效值；没有可用 store / 调整失败时返回 0（调用方忽略）。
    """
    if threads is None:
        threads = _duckdb_base_threads()
    try:
        from data_access import get_store

        engine = get_store()._engine
        if hasattr(engine, "_write_lock"):
            with engine._write_lock:
                engine._conn.execute(f"PRAGMA threads={int(threads)}")
        return int(threads)
    except Exception:
        return 0


def reset_resource_cache() -> None:
    """清空核数缓存（测试用）。"""
    global _cores
    _cores = None
