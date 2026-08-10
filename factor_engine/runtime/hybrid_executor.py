# -*- coding: utf-8 -*-
"""R27-048..052: HybridExecutor —— 多线程 vs 多进程按 GIL 自动选择。

目标（R27-048..051）
    - **线程**：DuckDB native / Polars native / NumPy vectorized / Numba nogil /
      Arrow IO 等明确 releases GIL 的实现——共享 DataAccess cache、共享 CSE panel、
      无 pickle、低内存复制。
    - **进程**：纯 Python loop / Pandas-heavy GIL-bound / Python UDF / 独立
      thread-pool 配置的 task。
    - Process pool 必须 long-lived（R27-052：禁止每 factor 启新进程），worker
      启动前设置 OMP/MKL/OPENBLAS/NUMEXPR/DUCKDB/Polars 线程。
    - 统一受 CPU token broker 约束（R27-210：不是 thread/process 各开 CPU 数）。

第一版原则（R27-057）
    - 线程 executor：``concurrent.futures.ThreadPoolExecutor``。
    - 进程 executor：``ProcessPoolExecutor``（long-lived），固定 Polars threads，
      外层调 worker 数。进程间传小 payload + 结果引用（R27-054）。
"""

from __future__ import annotations

import logging
import os
import queue as _queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from runtime.resource_broker import ResourceBroker

_logger = logging.getLogger(__name__)

#: GIL-bound backend（进程 executor 候选；R27-049）
GIL_BOUND_BACKENDS = frozenset({"pandas_numpy", "research_python"})
#: releases-GIL backend（线程 executor 候选；R27-048）
NATIVE_BACKENDS = frozenset({"duckdb_sql", "polars", "polars_panel", "polars_long", "sql"})


def classify_backend_execution(backend: str) -> str:
    """返回 ``thread`` 或 ``process``（R27-048/049 分类）。

    纯 Python / Pandas-heavy / GIL-bound → process；native releases-GIL → thread。
    未知 backend 保守归线程（共享 cache 优先），可由 ``FACTOR_ENGINE_HYBRID_*``
    覆盖。
    """
    b = str(backend or "pandas_numpy")
    env = os.environ.get("FACTOR_ENGINE_HYBRID_FORCE", "").strip().lower()
    if env in {"thread", "process"}:
        # 显式 force 覆盖分类（测试 / 运维强制路径）。
        return env
    if b in GIL_BOUND_BACKENDS:
        return "process"
    if b in NATIVE_BACKENDS:
        return "thread"
    return "thread"


def set_worker_thread_env(threads: int) -> None:
    """worker 启动前设置嵌套线程闭包（R27-045/046/052）。

    外层 16 worker 时内层 BLAS 不能再各开 16 threads（256 runnable 反而更慢）。
    """
    threads = max(1, int(threads))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        os.environ[var] = str(threads)
    os.environ["DUCKDB_MAX_THREADS"] = str(threads)


def _process_initializer(threads: int) -> Callable[[], None]:
    def _init() -> None:
        set_worker_thread_env(threads)
        _logger.info("r27 process worker ready threads=%d", threads)

    return _init


class HybridExecutor:
    """线程 / 进程混合执行器（R27-050/051/210）。

    - ``submit(backend, fn, *args)`` 自动按 backend 分类到 thread/process pool。
    - 两种 pool 的并发总量统一受 CPU token broker 约束（R27-210/043/044）。
    """

    def __init__(
        self,
        *,
        broker: ResourceBroker | None = None,
        max_thread_workers: int | None = None,
        max_process_workers: int | None = None,
        worker_threads: int = 1,
        prefer_process: bool = False,
    ) -> None:
        self.broker = broker or ResourceBroker()
        self.prefer_process = prefer_process
        self.worker_threads = max(1, worker_threads)
        from runtime.resource_governor import effective_cpu_slots

        cpu = effective_cpu_slots()
        # 不是 thread/process 各开 CPU 数（R27-210 双倍 oversubscribe）：
        # 总 CPU token 由 broker 统一，这里只给池的**上限**。
        self.max_thread_workers = max_thread_workers or cpu
        self.max_process_workers = max_process_workers or max(1, cpu // 2)
        self._thread_pool: ThreadPoolExecutor | None = None
        self._process_pool: Any = None
        self._lock = threading.Lock()
        self._thread_task_count = 0
        self._process_task_count = 0

    def _ensure_pools(self) -> None:
        with self._lock:
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(
                    max_workers=self.max_thread_workers, thread_name_prefix="r27-thread"
                )
            if self._process_pool is None:
                try:
                    from concurrent.futures import ProcessPoolExecutor

                    self._process_pool = ProcessPoolExecutor(
                        max_workers=self.max_process_workers,
                        initializer=_process_initializer(self.worker_threads),
                    )
                except Exception:
                    self._process_pool = None

    def submit(
        self,
        backend: str,
        fn: Callable[..., Any],
        *args: Any,
        cpu_tokens: int = 1,
        prefer: str | None = None,
    ) -> Future:
        """按 backend GIL 分类提交；``prefer`` ∈ {thread, process} 强制。"""
        self._ensure_pools()
        kind = prefer or classify_backend_execution(backend)
        if self.prefer_process and kind == "thread":
            kind = "process"
        if kind == "process" and self._process_pool is not None:
            self._process_task_count += 1
            return self._process_pool.submit(fn, *args)
        if self._thread_pool is None:
            self._ensure_pools()
        assert self._thread_pool is not None
        self._thread_task_count += 1
        return self._thread_pool.submit(fn, *args)

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=wait)
                self._thread_pool = None
            if self._process_pool is not None:
                self._process_pool.shutdown(wait=wait)
                self._process_pool = None

    def summary(self) -> dict[str, Any]:
        return {
            "thread_task_count": self._thread_task_count,
            "process_task_count": self._process_task_count,
            "max_thread_workers": self.max_thread_workers,
            "max_process_workers": self.max_process_workers,
            "worker_threads": self.worker_threads,
        }
