# -*- coding: utf-8 -*-
"""R39-PERF-072：进程级 DeadlineManager —— 大量 deadline query 只用一个 timer thread。

Gate-08（R39 §28）：deadline thread 数 O(1)。无论注册多少 query，只创建**一个**
timer 线程（惰性启动），内部用 min-heap + 惰性删除管理超时。超时时在 manager
线程上调用 ``cancel_fn()``（对 DuckDB 即 ``conn.interrupt()``）。

测试环境约束
    - 进程级单例通过 :func:`get_deadline_manager` 获得；测试可用
      :func:`reset_deadline_manager` 复位。
    - 无生产调用方（当前 FactorEngine 没有 per-query watchdog）——本模块先交付
      ``DeadlineManager`` + ``GuardDeadline`` context manager，由单元测试证明
      O(1)-thread 行为（Gate-08）。
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable


class DeadlineManager:
    """进程级 deadline 管理器（单 timer thread + min-heap）。

    - ``register(query_id, deadline_s, cancel_fn)``：登记一个 query；``deadline_s``
      为相对秒，超时在 manager 的 timer 线程上调用 ``cancel_fn()``。
    - ``cancel(query_id)``：删除一个未到期的登记（返回是否删除成功）。
    - ``watchdog_thread_created_count``：**总共创建**的 timer 线程数——与 query
      数量无关，注册 N 个 query 仍为 1（Gate-08：O(1) 线程）。
    """

    _THREAD_NAME = "r39-deadline-manager"

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, str]] = []  # (deadline_ts, seq, query_id)
        self._active: dict[str, tuple[float, int, Callable[[], None]]] = {}
        self._seq = itertools.count()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._timer_thread: threading.Thread | None = None
        self._watchdog_thread_created_count = 0
        self._timeouts_fired = 0
        self._fired: set[str] = set()
        self._shutdown = False

    # -- 线程 & 生命周期 -------------------------------------------------

    @property
    def watchdog_thread_created_count(self) -> int:
        with self._lock:
            return self._watchdog_thread_created_count

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def timeouts_fired(self) -> int:
        with self._lock:
            return self._timeouts_fired

    @property
    def timer_thread(self) -> threading.Thread | None:
        with self._lock:
            return self._timer_thread

    def _start_locked(self) -> None:
        if self._timer_thread is not None and self._timer_thread.is_alive():
            return
        self._timer_thread = threading.Thread(
            target=self._run, daemon=True, name=self._THREAD_NAME
        )
        self._timer_thread.start()
        self._watchdog_thread_created_count += 1

    def shutdown(self) -> None:
        """停止 timer 线程（幂等）。"""
        with self._lock:
            self._shutdown = True
            self._cond.notify_all()
        thread = self._timer_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    # -- 对外 API ---------------------------------------------------------

    def register(self, query_id: str, deadline_s: float, cancel_fn: Callable[[], None]) -> None:
        """登记一个 query；``deadline_s<=0`` 视为不设 deadline（空操作）。"""
        deadline_s = max(0.0, float(deadline_s))
        if deadline_s <= 0:
            return
        with self._lock:
            self._start_locked()
            deadline_ts = time.monotonic() + deadline_s
            seq = next(self._seq)
            self._active[query_id] = (deadline_ts, seq, cancel_fn)
            heapq.heappush(self._heap, (deadline_ts, seq, query_id))
            self._cond.notify_all()

    def cancel(self, query_id: str) -> bool:
        """删除一个未到期登记；返回是否删除成功（已超时/不存在返回 False）。"""
        with self._lock:
            if query_id in self._active:
                del self._active[query_id]
                return True
            return False

    def query_due(self, query_id: str) -> bool:
        """该 query 是否已被触发过（测试/诊断用）。"""
        with self._lock:
            return query_id in self._fired

    # -- 内部 timer 循环 --------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._shutdown:
                    return
                if not self._heap:
                    self._cond.wait(timeout=1.0)
                    continue
                deadline_ts, seq, qid = self._heap[0]
                now = time.monotonic()
                if deadline_ts > now:
                    self._cond.wait(timeout=min(deadline_ts - now, 1.0))
                    continue
                # 到期：弹出堆顶（lazy 校验，避免已 cancel 的过期条目误触发）
                heapq.heappop(self._heap)
                entry = self._active.get(qid)
                if entry is None or entry[1] != seq:
                    continue  # 惰性删除：已 cancel 或已更新
                del self._active[qid]
                self._timeouts_fired += 1
                self._fired.add(qid)
                cancel_fn = entry[2]
            # 锁外触发 cancel_fn（可能回调 register/cancel / conn.interrupt）
            try:
                cancel_fn()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# 进程级默认单例 + GuardDeadline context manager
# ---------------------------------------------------------------------------

_DEFAULT_MANAGER: DeadlineManager | None = None
_DEFAULT_LOCK = threading.Lock()


def get_deadline_manager() -> DeadlineManager:
    """进程级默认 DeadlineManager（懒构造单例）。"""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_MANAGER is None:
                _DEFAULT_MANAGER = DeadlineManager()
    return _DEFAULT_MANAGER


def reset_deadline_manager() -> None:
    """复位进程级单例（测试用）。"""
    global _DEFAULT_MANAGER
    with _DEFAULT_LOCK:
        if _DEFAULT_MANAGER is not None:
            _DEFAULT_MANAGER.shutdown()
            _DEFAULT_MANAGER = None


_GID = itertools.count()


@contextmanager
def guard_deadline(
    manager: DeadlineManager,
    conn: Any,
    timeout_s: float,
    *,
    query_id: str | None = None,
):
    """为 ``conn.interrupt()``-可打断对象登记 deadline；退出时取消登记。

    ``timeout_s<=0`` 表示不设 deadline（直接 yield）。未来 deadline 代码应使用
    本 context manager，从而自动走进程级单线程 manager（不新建 per-query
    watchdog thread）。
    """
    qid = query_id or f"q-{next(_GID)}"
    if timeout_s is None or float(timeout_s) <= 0:
        yield qid
        return
    interrupt = getattr(conn, "interrupt", None)

    def _cancel() -> None:
        if callable(interrupt):
            interrupt()

    manager.register(qid, timeout_s, _cancel)
    try:
        yield qid
    finally:
        manager.cancel(qid)
