"""data_access.r30.query_trace —— R30-P0-010 QueryTrace + request 级观测。

纯 additive 包装层：**不修改** store.py / read/ / runtime/ 任何文件。R30 在既有
R24-R29 闭环之上新增观测能力：

    - ``QueryTrace``：一次读请求的结构化 trace（阶段计时 / 字节 / 行数 / 缓存态）。
    - ``QueryTracer``：request-scoped ContextVar 栈，嵌套 ``enter`` 压栈、exit
      恢复；线程安全（每个线程自己的 ContextVar）。
    - ``run_traced_read``：把 ``store.prepare_read + execute_prepared_read`` 包进
      trace；全部异常仍返回 trace（不丢观测）。

与 ``data_access.read.telemetry`` 的区别：telemetry 是 per-operator 计数
（OperatorCounters/record_query），这里是一次请求的完整生命周期 trace。

设计要点
    - stage 计时用 ``time.perf_counter``，统一记毫秒。
    - ``stage_timings`` 是累计 dict：同一 stage 多次进入会累加。
    - 对既有 Store / ReadResult / ReadHandle 一律防御性 getattr——拿不到就
      None/0，绝不抛异常。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

#: 规范阶段集合（R30-P0-010 §query_trace）。run_traced_read 只记其中能拿到的
#: 阶段；调用方可用 ``tracer.stage(name)`` 记任意阶段。
STAGES: tuple[str, ...] = (
    "auth",
    "contract",
    "resolution",
    "mirror",
    "remote_list",
    "remote_head",
    "schema",
    "snapshot",
    "calendar",
    "prepare",
    "governor_wait",
    "duckdb_wait",
    "duckdb_execute",
    "polars_execute",
    "normalize",
    "join",
    "serialize",
    "post_verify",
)


@dataclass
class QueryTrace:
    """一次读请求的结构化观测对象。"""

    request_id: str | None = None
    job_id: str | None = None
    dataset: str | None = None
    #: stage 名 → 累计毫秒（同名多次进入累加）。
    stage_timings: dict[str, float] = field(default_factory=dict)
    scan_bytes: int = 0
    result_bytes: int = 0
    rows: int = 0
    cache_status: str | None = None
    resolution_cache_hit: bool = False
    session_source_reuse: int = 0
    resource_wait_ms: float = 0.0
    backend: str | None = None
    source_snapshot: str | None = None
    columns: list[str] | None = None
    started_monotonic: float | None = None
    elapsed_ms: float | None = None
    #: 异常路径（run_traced_read 吞异常后写这里）——"全部异常仍返回 trace 不丢"。
    error: str | None = None

    def add_stage(self, name: str, ms: float) -> None:
        """累加一个阶段耗时（毫秒）。"""
        self.stage_timings[name] = self.stage_timings.get(name, 0.0) + float(ms)

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化的 dict 表示。"""
        return {
            "request_id": self.request_id,
            "job_id": self.job_id,
            "dataset": self.dataset,
            "stage_timings": dict(self.stage_timings),
            "scan_bytes": int(self.scan_bytes or 0),
            "result_bytes": int(self.result_bytes or 0),
            "rows": int(self.rows or 0),
            "cache_status": self.cache_status,
            "resolution_cache_hit": bool(self.resolution_cache_hit),
            "session_source_reuse": int(self.session_source_reuse or 0),
            "resource_wait_ms": float(self.resource_wait_ms or 0.0),
            "backend": self.backend,
            "source_snapshot": self.source_snapshot,
            "columns": list(self.columns) if self.columns else None,
            "started_monotonic": self.started_monotonic,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


#: request-scoped trace 栈（None = 无活跃 request）。所有 QueryTracer 实例共享
#: 同一 ContextVar → 嵌套 enter（跨实例）也正确压栈/恢复。
_trace_stack_var: ContextVar[list[QueryTrace] | None] = ContextVar(
    "data_access_query_trace_stack", default=None
)


class _StageGuard:
    """``guard.stage(name)`` 返回的计时 contextmanager。

    enter 时抓当前栈顶 trace 并开始计时；exit 时把毫秒累加进 trace。
    栈顶在 enter 时冻结——嵌套 enter 之间的 stage 不会串 trace。
    """

    __slots__ = ("_tracer", "_name", "_trace", "_started")

    def __init__(self, tracer: "QueryTracer", name: str) -> None:
        self._tracer = tracer
        self._name = name
        self._trace: QueryTrace | None = None
        self._started: float | None = None

    def __enter__(self) -> "_StageGuard":
        self._trace = self._tracer.current()
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._trace is not None and self._started is not None:
            self._trace.add_stage(self._name, (time.perf_counter() - self._started) * 1000.0)
        return False


class _TraceEnter:
    """``QueryTracer.enter(...)`` 返回的 contextmanager + 计时门面。

    既是 ``with tracer.enter(...) as g:`` 的 CM，又带 ``g.stage(name)``。
    """

    __slots__ = ("_tracer", "_trace", "_token")

    def __init__(self, tracer: "QueryTracer", trace: QueryTrace) -> None:
        self._tracer = tracer
        self._trace = trace
        self._token: Any = None

    def __enter__(self) -> "_TraceEnter":
        stack = self._tracer._stack_var.get()
        if stack is None:
            stack = []
        else:
            stack = list(stack)
        stack.append(self._trace)
        self._token = self._tracer._stack_var.set(stack)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._token is not None:
            self._tracer._stack_var.reset(self._token)
            self._token = None
        return False

    def stage(self, name: str) -> _StageGuard:
        """计时 contextmanager：``with g.stage("duckdb_execute"): ...``。"""
        return _StageGuard(self._tracer, name)


class QueryTracer:
    """request-scoped QueryTrace 栈。

    用法::

        tracer = QueryTracer()
        with tracer.enter(request_id="r1", dataset="ashare_daily") as g:
            with g.stage("prepare"):
                ...
            with g.stage("duckdb_execute"):
                ...
            trace = tracer.finish()   # elapsed 已算好

    嵌套 ``enter`` 压栈、exit 恢复；``current()`` 取当前栈顶。
    """

    def __init__(self) -> None:
        #: 共享进程级 ContextVar（嵌套跨实例也正确）。
        self._stack_var: ContextVar[list[QueryTrace] | None] = _trace_stack_var

    def enter(
        self,
        request_id: str | None = None,
        job_id: str | None = None,
        dataset: str | None = None,
    ) -> _TraceEnter:
        """压入一个全新 QueryTrace，返回 contextmanager（带 stage()）。"""
        trace = QueryTrace(
            request_id=request_id,
            job_id=job_id,
            dataset=dataset,
            started_monotonic=time.monotonic(),
        )
        return _TraceEnter(self, trace)

    def stage(self, name: str) -> _StageGuard:
        """对当前栈顶 trace 计时（无活跃 trace 则 no-op）。"""
        return _StageGuard(self, name)

    def current(self) -> QueryTrace | None:
        """当前 request 的 trace（无活跃 request → None）。"""
        stack = self._stack_var.get()
        if stack:
            return stack[-1]
        return None

    def finish(self) -> QueryTrace | None:
        """计算当前 trace 的 elapsed_ms 并返回（幂等；无活跃 trace → None）。"""
        trace = self.current()
        if trace is None:
            return None
        if trace.started_monotonic is not None:
            trace.elapsed_ms = (time.monotonic() - trace.started_monotonic) * 1000.0
        return trace


#: 进程级默认 tracer（run_traced_read / trace_request 用它）。
default_tracer = QueryTracer()


def trace_request(
    request_id: str | None = None,
    job_id: str | None = None,
    dataset: str | None = None,
) -> _TraceEnter:
    """顶层便捷：用默认 tracer 开启一个 request 上下文。"""
    return default_tracer.enter(
        request_id=request_id, job_id=job_id, dataset=dataset
    )


def reset_tracer() -> None:
    """清空当前线程的 trace 栈（测试 / 清理用）。

    注意：只在没有活跃 ``with`` 上下文时调用；嵌套活跃期间调用会破坏恢复链。
    """
    _trace_stack_var.set(None)


# ---- run_traced_read：把 store 读流程包进 trace ----

def _resolution_cache_has_dataset(dataset: str) -> bool:
    """探测当前 request 的 resolution cache 是否已含本 dataset（best-effort）。"""
    try:
        from data_access.runtime.read_session_context import get_resolution_cache
    except Exception:
        return False
    try:
        cache = get_resolution_cache()
    except Exception:
        return False
    if not cache:
        return False
    for key in cache:
        if isinstance(key, tuple) and key and key[0] == dataset:
            return True
        if key == dataset:
            return True
    return False


def _extract_rows(result: Any) -> int:
    if result is None:
        return 0
    for attr in ("num_rows", "rows"):
        v = getattr(result, attr, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    table = getattr(result, "table", None)
    if table is not None:
        v = getattr(table, "num_rows", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    stats = getattr(result, "stats", None)
    if stats is not None:
        v = getattr(stats, "rows", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _extract_columns(result: Any) -> list[str] | None:
    if result is None:
        return None
    cols = getattr(result, "columns", None)
    if isinstance(cols, (list, tuple)):
        return list(cols)
    table = getattr(result, "table", None)
    if table is not None:
        cols = getattr(table, "column_names", None)
        if cols is not None:
            return list(cols)
    return None


def _extract_bytes(result: Any) -> int:
    if result is None:
        return 0
    for attr in ("result_bytes", "bytes", "nbytes"):
        v = getattr(result, attr, None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    table = getattr(result, "table", None)
    if table is not None:
        v = getattr(table, "nbytes", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    stats = getattr(result, "stats", None)
    if stats is not None:
        v = getattr(stats, "bytes", None)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _extract_cache_status(result: Any, prepared: Any) -> str | None:
    for obj in (result, prepared):
        if obj is None:
            continue
        v = getattr(obj, "cache_status", None)
        if v is not None:
            return str(v)
    return None


def _extract_source_snapshot(prepared: Any, result: Any) -> str | None:
    snap = getattr(prepared, "resolved_source_snapshot", None)
    if snap is not None:
        for attr in ("snapshot_id", "content_digest", "identity"):
            v = getattr(snap, attr, None)
            if v is not None:
                return str(v)
    if result is not None:
        snap = getattr(result, "snapshot", None)
        if snap is not None:
            v = getattr(snap, "snapshot_id", None)
            if v is not None:
                return str(v)
    return None


def _extract_backend(prepared: Any) -> str:
    bp = getattr(prepared, "backend_plan", None)
    if isinstance(bp, dict):
        b = bp.get("backend")
        if b in ("duckdb", "polars"):
            return str(b)
    return "duckdb"


def _extract_resource_wait_ms(prepared: Any, result: Any) -> float:
    for obj in (prepared, result):
        if obj is None:
            continue
        for attr in ("resource_wait_ms", "governor_wait_ms", "wait_ms"):
            v = getattr(obj, attr, None)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return 0.0


def _record_metrics(trace: QueryTrace) -> None:
    """把 trace 落进默认 MetricsRegistry（best-effort，失败静默）。"""
    try:
        from data_access.r30.metrics import record_trace

        record_trace(trace)
    except Exception:  # noqa: BLE001 —— 观测失败不能影响主路径
        pass


def run_traced_read(
    store: Any,
    dataset: str,
    *,
    request_id: str | None = None,
    job_id: str | None = None,
    columns: Any = None,
    time_range: Any = None,
    params: Any = None,
    **kwargs: Any,
) -> tuple[QueryTrace, Any]:
    """用默认 tracer 包 ``store.prepare_read + execute_prepared_read``。

    返回 ``(trace, result)``。**全部异常仍返回 trace 不丢**：异常被记录到
    ``trace.error``，result 为 None（调用方读 trace.error / result is None 判断）。

    阶段计时（能拿到时）：``prepare``（整体 prepare_read）+ ``duckdb_execute`` /
    ``polars_execute``（整体 execute_prepared_read）；其余阶段由调用方用
    ``tracer.stage(...)`` 补充。
    """
    trace = QueryTrace(
        request_id=request_id,
        job_id=job_id,
        dataset=dataset,
        started_monotonic=time.monotonic(),
    )
    result: Any = None
    prepared: Any = None
    rc_hit_before = _resolution_cache_has_dataset(dataset)
    try:
        with default_tracer.enter(
            request_id=request_id, job_id=job_id, dataset=dataset
        ) as g:
            # ---- prepare 阶段（内部含 auth/contract/resolution/schema/snapshot/
            #      calendar/governor；整体计时为 prepare）----
            t0 = time.perf_counter()
            try:
                prepared = store.prepare_read(
                    dataset,
                    columns=columns,
                    time_range=time_range,
                    params=params,
                    **kwargs,
                )
            finally:
                trace.add_stage("prepare", (time.perf_counter() - t0) * 1000.0)

            trace.resolution_cache_hit = rc_hit_before
            trace.backend = _extract_backend(prepared)
            trace.source_snapshot = _extract_source_snapshot(prepared, None)
            trace.resource_wait_ms = _extract_resource_wait_ms(prepared, None)

            # ---- execute 阶段 ----
            exec_stage = (
                "polars_execute" if trace.backend == "polars" else "duckdb_execute"
            )
            t1 = time.perf_counter()
            try:
                result = store.execute_prepared_read(prepared)
            finally:
                trace.add_stage(exec_stage, (time.perf_counter() - t1) * 1000.0)

            # ---- 从 result 补元数据 ----
            trace.rows = _extract_rows(result)
            trace.columns = _extract_columns(result)
            trace.result_bytes = _extract_bytes(result)
            trace.cache_status = _extract_cache_status(result, prepared)
            if trace.source_snapshot is None:
                trace.source_snapshot = _extract_source_snapshot(prepared, result)
            wait = _extract_resource_wait_ms(prepared, result)
            if wait:
                trace.resource_wait_ms = wait
    except Exception as exc:  # noqa: BLE001 —— 全部异常仍返回 trace
        trace.error = f"{type(exc).__name__}: {exc}"
    finally:
        trace.elapsed_ms = (time.monotonic() - trace.started_monotonic) * 1000.0
        _record_metrics(trace)
    return trace, result


__all__ = [
    "STAGES",
    "QueryTrace",
    "QueryTracer",
    "default_tracer",
    "trace_request",
    "reset_tracer",
    "run_traced_read",
]
