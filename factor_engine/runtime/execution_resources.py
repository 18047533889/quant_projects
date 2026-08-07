# -*- coding: utf-8 -*-
"""执行资源预算（#34）：协调 FactorEngine worker 数 / DuckDB threads / Polars
rayon threads / IO 并发，避免 oversubscription。

背景
    如果 8 个 factor worker 各自跑一个 8-thread DuckDB 查询，在 16 核机器上会
    有 64 个 runnable 线程——上下文切换开销反而拖慢。这里统一给
    ``n_jobs × duckdb_threads <= physical_cores`` 的约束，并暴露给调用方。

用法
    plan = resource_plan(n_jobs=8)          # -> {n_jobs, duckdb_threads, ...}
    plan.duckdb_threads                     # 该 batch 每个 worker 的 DuckDB 线程数
    set_duckdb_max_threads(plan)            # 写入 DuckDBEngine.config（如可调）
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_cores: int | None = None


def physical_cores() -> int:
    """可用物理核数（缓存；环境变量 FACTOR_ENGINE_CPU_BUDGET 可覆盖）。"""
    global _cores
    if _cores is not None:
        return _cores
    override = os.environ.get("FACTOR_ENGINE_CPU_BUDGET", "").strip()
    if override.isdigit() and int(override) > 0:
        _cores = int(override)
        return _cores
    try:
        import os as _os

        _cores = len(_os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            import multiprocessing

            _cores = multiprocessing.cpu_count()
        except Exception:  # pragma: no cover
            _cores = 4
    return _cores


@dataclass(frozen=True)
class ResourcePlan:
    """一次并行执行的整体资源分配。"""

    n_jobs: int
    duckdb_threads: int
    polars_threads: int
    io_concurrency: int
    memory_budget_gb: float
    total_runnable: int

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


def resource_plan(n_jobs: int | None = None) -> ResourcePlan:
    """按可用核数分配资源。

    约束：``n_jobs × duckdb_threads <= physical_cores``（DuckDB 是最大的
    线程池消费者）；polars rayon 线程 = duckdb_threads（同一 budget）。
    """
    cores = physical_cores()
    jobs = n_jobs if n_jobs and n_jobs > 0 else cores
    jobs = max(1, min(jobs, cores))
    base = _duckdb_base_threads()
    duckdb_threads = max(1, min(base, cores // jobs)) if jobs > 0 else base
    # IO 并发：每 worker 允许 1~2 个并发读
    io_concurrency = max(1, min(cores, jobs * 2))
    # 简单内存预算：总内存 × 0.7 / jobs（乐观估计单机总内存 32G，可覆盖）
    try:
        import psutil  # type: ignore[import-untyped]

        total_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        total_gb = 32.0
    memory_budget_gb = max(1.0, round(total_gb * 0.7 / jobs, 1))
    return ResourcePlan(
        n_jobs=jobs,
        duckdb_threads=duckdb_threads,
        polars_threads=duckdb_threads,
        io_concurrency=io_concurrency,
        memory_budget_gb=memory_budget_gb,
        total_runnable=jobs * duckdb_threads,
    )


def set_duckdb_max_threads(plan: ResourcePlan | None = None, threads: int | None = None) -> int:
    """把 DuckDB 线程数写入环境变量（DuckDBEngine 创建时读取），返回生效值。

    进程内已有 DuckDBEngine 时也尝试 ``PRAGMA threads`` 就地调整。
    """
    value = threads if threads is not None else (plan.duckdb_threads if plan else None)
    if value is None:
        value = _duckdb_base_threads()
    os.environ["DUCKDB_MAX_THREADS"] = str(value)
    try:
        from data_access import get_store

        engine = get_store()._engine
        if hasattr(engine, "_write_lock"):
            with engine._write_lock:
                engine._conn.execute(f"PRAGMA threads={int(value)}")
    except Exception:
        pass
    return value


def reset_resource_cache() -> None:
    """清空核数缓存（测试用）。"""
    global _cores
    _cores = None
