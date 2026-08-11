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


def _worker_initializer(threads: int) -> None:
    """R31-P0-008：顶层进程 worker 初始化函数（可 pickle，spawn 平台安全）。

    旧实现 ``_process_initializer`` 返回嵌套闭包；``ProcessPoolExecutor`` 在
    macOS/Windows spawn 下需要把 initializer 序列化传给子进程，嵌套函数不可
    pickle → 进程池创建即失败。顶层函数 + initargs 才是 spawn-safe。
    """
    set_worker_thread_env(threads)
    _logger.info("r31 process worker ready threads=%d", threads)


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
        # R31-007.4: 连续熔断计数（供调度器/运维观测 backend 健康）。
        self._process_breaker_hits = 0
        # R39-P1-PERF-082: 最近一次 runtime backend event（submit 时记录）。
        self._last_runtime_backend_event: dict[str, Any] | None = None

    def cohort_worker_threads(
        self,
        *,
        query_shape: Any = None,
        scan_bytes: int = 0,
        current_concurrency: int = 0,
    ) -> int:
        """R39-PERF-074：query-class cohort 选择内层 worker threads。

        默认 OFF（env ``FACTOR_ENGINE_COHORT_PROFILES=1`` 开启）。开启且 query
        shape 可用时按 {1,2,4,8} 固定 profile 池选择；shape 未知或 cohort 关闭时
        回退 ``self.worker_threads``（legacy 公式）。
        """
        from runtime.query_class_cohort import cohort_profiles_enabled, QueryClassCohort

        if not cohort_profiles_enabled():
            return int(self.worker_threads)
        selected = QueryClassCohort().select_cohort(
            query_shape, scan_bytes, current_concurrency
        )
        return int(self.worker_threads) if selected is None else int(selected)

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
                        initializer=_worker_initializer,
                        initargs=(self.worker_threads,),
                    )
                except Exception:
                    self._process_pool = None

    def record_runtime_backend_event(
        self,
        backend: str,
        kind: str,
        *,
        no_fallback: bool = True,
    ) -> dict[str, Any]:
        """R39-P1-PERF-082: 记录 runtime backend event（供 O(1) 校验消费）。"""
        event = {
            "backend": str(backend or "pandas_numpy"),
            "execution_kind": str(kind or "thread"),
            "no_fallback": bool(no_fallback),
        }
        self._last_runtime_backend_event = event
        return event

    def _validate_certificate_event(
        self,
        certificate: Any,
        event: dict[str, Any] | None,
        allowed_fallbacks: Any = (),
    ) -> bool:
        """O(1) 校验（不 import plan / 不 walk 树）；失败只计数 + 告警，不阻断执行。"""
        from runtime.perf_counters import get_global_counters
        from runtime.production_execution_certificate import validate

        ok = validate(certificate, event, allowed_fallbacks)
        if not ok:
            get_global_counters().incr("certificate_validation_failure_count")
            _logger.warning(
                "production execution certificate validation failed: event=%s",
                event,
            )
        return ok

    def validate_certificate(
        self,
        certificate: Any,
        allowed_fallbacks: Any = (),
    ) -> bool:
        """R39-P1-PERF-082: O(1) 校验最近一次 runtime backend event 与证书一致。"""
        return self._validate_certificate_event(
            certificate, self._last_runtime_backend_event, allowed_fallbacks
        )

    def submit(
        self,
        backend: str,
        fn: Callable[..., Any],
        *args: Any,
        cpu_tokens: int = 1,
        prefer: str | None = None,
        certificate: Any | None = None,
    ) -> Future:
        """按 backend GIL 分类提交；``prefer`` ∈ {thread, process} 强制。

        R31-008：``cpu_tokens`` 仅保留签名兼容，**不作为第二套资源治理**——唯一
        admission authority 是 ``ResourceBroker``（调度器在 submit 前已按 token
        预留）。此处忽略该参数，避免「Executor 自己也控制并发」的误导。

        R39-P1-PERF-082（additive）：每次 submit 记录 runtime backend event；
        若传入 ``certificate``，立即对本次 event 做 O(1) 校验。校验失败只计数
        + 告警，**绝不阻断执行**（默认行为不变）。
        """
        _ = cpu_tokens  # R31-008: admission 由 broker 统一，Executor 只执行
        self._ensure_pools()
        kind = prefer or classify_backend_execution(backend)
        if self.prefer_process and kind == "thread":
            kind = "process"
        if kind == "process" and self._process_pool is not None:
            try:
                self._process_task_count += 1
                event = self.record_runtime_backend_event(backend, "process", no_fallback=True)
                if certificate is not None:
                    self._validate_certificate_event(certificate, event)
                return self._process_pool.submit(fn, *args)
            except Exception:
                # R31-007.4 (BROKEN_WORKER_RECOVERY_PASS)：worker 崩溃后进程池不可用。
                # 重建池（尽快恢复），当前 task 立即回退线程执行，不让整个 Engine
                # 因一个坏 worker 永久不可用。
                self._recover_process_pool()
                self._process_breaker_hits += 1
                self._thread_task_count += 1
                event = self.record_runtime_backend_event(backend, "thread", no_fallback=False)
                if certificate is not None:
                    self._validate_certificate_event(certificate, event)
                return self._thread_pool.submit(fn, *args)
        if self._thread_pool is None:
            self._ensure_pools()
        assert self._thread_pool is not None
        self._thread_task_count += 1
        event = self.record_runtime_backend_event(backend, "thread", no_fallback=True)
        if certificate is not None:
            self._validate_certificate_event(certificate, event)
        return self._thread_pool.submit(fn, *args)

    def _recover_process_pool(self) -> None:
        """R31-007.4：重建损坏的 process pool（连续熔断计数可被调度器读取）。"""
        with self._lock:
            if self._process_pool is not None:
                try:
                    self._process_pool.shutdown(wait=False)
                except Exception:
                    pass
                self._process_pool = None
            try:
                from concurrent.futures import ProcessPoolExecutor

                self._process_pool = ProcessPoolExecutor(
                    max_workers=self.max_process_workers,
                    initializer=_worker_initializer,
                    initargs=(self.worker_threads,),
                )
            except Exception:
                self._process_pool = None

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
            "process_breaker_hits": self._process_breaker_hits,
            "max_thread_workers": self.max_thread_workers,
            "max_process_workers": self.max_process_workers,
            "worker_threads": self.worker_threads,
            "last_runtime_backend_event": self._last_runtime_backend_event,
        }
