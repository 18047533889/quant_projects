"""R21-048..060: bounded job queue, admission control, deadline/cancel/heartbeat.

A bounded FIFO queue (``queue.Queue(maxsize=...)``) + fixed worker threads.
A full queue returns 429/503 with ``Retry-After`` (R21-049); global and
per-principal running/queued limits are enforced (R21-050).  Every job has a
deadline, cancel flag, heartbeat and real phase so a job can always be
cancelled, timed out, or recovered (R21-055..060).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from service.errors import ServiceError, classify_exception, sanitize_message
from service.jobstore import JobPhase, JobRecord, JobStatus, _utc_now


class JobCancelledError(RuntimeError):
    pass


class JobDeadlineExceeded(RuntimeError):
    pass


#: R40 #101: job 的 request-scoped cancel event（worker 阶段间查询）。
#: ``_begin`` 把每个 job 的 event 设为当前线程 ContextVar；heartbeat monitor /
#: drain 通过 ``_set_cancel_event(run_id)`` set 它，worker 下一次
#: ``check_job_alive`` / ``set_phase`` 即抛 JobCancelledError —— 标 INTERRUPTED
#: 的同时真正停止仍在跑的计算（不再白烧 CPU / 发布旧代结果）。
_cancel_event_ctx_var: ContextVar[threading.Event | None] = ContextVar(
    "job_cancel_event", default=None
)


#: R31-P1-039：service 全局共享的 ResourceBroker（job admission + task admission
#: 统一）。worker 执行 job 时通过 **ContextVar** 传给内部 scheduler（§58），避免
#: 「4 个 HTTP job × 16 内部 workers = 64 runnable」的过载，也避免旧 module-global
#: ``_service_broker_ctx`` 的「A set → B set → A restore」竞态（R36 P0-018）。
_service_broker_ctx_var: ContextVar[Any] = ContextVar("service_broker", default=None)


def _get_service_broker() -> Any | None:
    return _service_broker_ctx_var.get()


#: QoS lane（R38 P0-028：压力升高时 BACKGROUND 停新 admission、STANDARD 收缩、
#: CRITICAL 保底）。
QOS_CRITICAL = "CRITICAL"
QOS_STANDARD = "STANDARD"
QOS_BACKGROUND = "BACKGROUND"
_QOS_ORDER = (QOS_CRITICAL, QOS_STANDARD, QOS_BACKGROUND)


@dataclass
class JobResourceEstimate:
    """R38 P0-027（§11）：job 的资源估计 → HostCoordinator JobLease admission。

    编译后 / enqueue 前提供；实际 worker 启动前申请 JobLease（不再只靠固定
    ``max_running`` 数并发）。
    """

    priority: str = QOS_STANDARD
    memory_p99: int = 0
    cpu_budget: int = 1
    io_class: str = "best_effort"
    estimated_runtime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "memory_p99": self.memory_p99,
            "cpu_budget": self.cpu_budget,
            "io_class": self.io_class,
            "estimated_runtime_s": self.estimated_runtime_s,
        }


@dataclass
class QueueSnapshot:
    queued: int
    running: int
    max_queue: int
    max_running: int
    longest_wait_seconds: float


class BoundedJobQueue:
    """Bounded FIFO + workers with per-job deadline/cancel/heartbeat support."""

    def __init__(
        self,
        *,
        max_queue: int | None = None,
        max_running: int | None = None,
        timeout_default: float | None = None,
        heartbeat_stale_seconds: float = 120.0,
        queue_full_status: int = 503,
        heartbeat_check_interval: float | None = None,
    ) -> None:
        self.max_queue = max_queue if max_queue is not None else int(
            os.environ.get("FACTOR_ENGINE_SERVICE_MAX_QUEUE", "64")
        )
        self.max_running = max_running if max_running is not None else int(
            os.environ.get("FACTOR_ENGINE_SERVICE_MAX_WORKERS", "4")
        )
        self.timeout_default = timeout_default if timeout_default is not None else float(
            os.environ.get("FACTOR_ENGINE_SERVICE_JOB_TIMEOUT", "300")
        )
        self.heartbeat_stale_seconds = float(heartbeat_stale_seconds)
        # R40 #100: heartbeat monitor 的轮询间隔（默认 15s；测试/调用方可调小）。
        self.heartbeat_check_interval = (
            float(heartbeat_check_interval)
            if heartbeat_check_interval is not None
            else float(os.environ.get("FACTOR_ENGINE_SERVICE_HEARTBEAT_INTERVAL", "15.0"))
        )
        self.queue_full_status = queue_full_status
        self._pending: "queue.Queue[JobRecord]" = queue.Queue(maxsize=self.max_queue)
        self._store: Any = None
        self._lock = threading.RLock()
        self._running: dict[str, JobRecord] = {}
        # R40 #101: run_id -> cancel event（worker 阶段间查询；heartbeat/drain
        # 标 INTERRUPTED 时 set，真正停止在跑计算）。
        self._cancel_events: dict[str, threading.Event] = {}
        self._per_principal: dict[str, int] = {}
        self._workers: list[threading.Thread] = []
        self._started = False
        # R40 #102: resource-reject 持久化失败（连 emergency journal 都写不进）
        # 的致命计数 —— 不再 ``except Exception: pass`` 静默吞掉。
        self.resource_reject_fatal_counter = 0
        # R32-P0-019: 显式生命周期状态机 ACCEPTING → DRAINING → STOPPED。
        # DRAINING 拒绝新任务但 worker 继续清空已有队列 —— 历史实现一开始就把
        # ``_stopping=True``，worker loop ``while not _stopping`` 立即不再消费
        # queued job，与 drain 语义冲突。
        self._state = "accepting"
        self._queue_put_times: dict[str, float] = {}
        self.jobs_submitted_total = 0
        self.jobs_rejected_total = 0
        # R38 P0-027（§11）：job 资源估计 + 实际 JobLease（worker 启动前申请）。
        self._job_estimates: dict[str, JobResourceEstimate] = {}
        self._job_leases: dict[str, Any] = {}
        self.jobs_lease_rejected_total = 0
        # R36 P0-016/§57：service 用 HostResourceCoordinator 的 broker——job
        # admission + task admission 与 FE 内部 scheduler / DA 共享同一资源权威
        # （不再各自建独立 broker）。
        try:
            from runtime.host_resource_coordinator import get_host_coordinator

            self.coordinator = get_host_coordinator()
            self.broker = self.coordinator.broker
        except Exception:
            self.coordinator = None
            self.broker = None

    # -- lifecycle ----------------------------------------------------------
    def start(self, store: Any) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._store = store
        for _ in range(self.max_running):
            t = threading.Thread(target=self._worker_loop, name="fe-job-worker", daemon=True)
            t.start()
            self._workers.append(t)
        monitor = threading.Thread(target=self._heartbeat_monitor, name="fe-heartbeat", daemon=True)
        monitor.start()

    def _set_cancel_event(self, run_id: str) -> None:
        """Set the job's request-scoped cancel event so the running worker stops."""
        ev = self._cancel_events.get(run_id)
        if ev is not None:
            ev.set()

    def _request_cancel_running(self) -> None:
        """Drain 阶段 2：对残余非终态 running job 请求取消（持久化 + set event）。

        R40 #99: 状态变更必须持久化（``_store.update``），崩溃不丢状态。
        """
        with self._lock:
            targets = [
                job
                for job in list(self._running.values())
                if not job.is_terminal and not job.cancel_requested_at
            ]
        for job in targets:
            job.cancel_requested_at = _utc_now()
            job.status = JobStatus.CANCELLING
            self._set_cancel_event(job.run_id)
            if self._store is not None:
                try:
                    self._store.update(job)
                except Exception:  # noqa: BLE001 - drain 不因单个持久化失败中断
                    pass

    def _interrupt_running(self) -> None:
        """Drain 阶段 3：force-timeout 后标记残余 straggler INTERRUPTED（CAS）。

        R40 #99: 状态变更走 CAS（terminal guard 防旧执行覆盖）；R40 #101: 同时
        set cancel event 真正停掉在跑 worker 线程。
        """
        with self._lock:
            targets = [job for job in list(self._running.values()) if not job.is_terminal]
        for job in targets:
            interrupted = JobRecord(**vars(job))
            interrupted.status = JobStatus.INTERRUPTED
            interrupted.error_code = "JOB_INTERRUPTED"
            interrupted.error = "job interrupted by service drain force-timeout"
            interrupted.finished_at = _utc_now()
            self._set_cancel_event(job.run_id)
            if self._store is not None:
                try:
                    ok = self._store.cas_transition(
                        interrupted,
                        expected_status=job.status,
                        write_manifest=True,
                    )
                except AttributeError:
                    self._store.update(interrupted)
                except Exception:  # noqa: BLE001
                    pass

    def drain(self, timeout: float = 30.0) -> None:
        """Graceful shutdown（R40 #97）：**先让 queued+running 自然完成**，再逐级
        升级 cancel / interrupt —— 绝不一开始就 cancel 所有 running。

        1) ``_state="draining"`` 停止新 admission；worker 仍消费 queued 直到空；
        2) 等 running==0 && queue 空（自然完成）直到 ``timeout``；
        3) 宽限期后对残余 running 请求 cancel（持久化 + set event）；
        4) 更短 force-timeout 后标记残余 INTERRUPTED（CAS + set event），
           绝不留下 phantom running。

        R32-P0-019 语义保留：draining 期间 worker ``while state != stopped``
        继续消费 queued。
        """
        self._state = "draining"
        deadline = time.monotonic() + timeout
        # Phase 1: let queued drain + running finish naturally.
        while time.monotonic() < deadline:
            with self._lock:
                running = len(self._running)
            if running == 0 and self._pending.empty():
                self._state = "stopped"
                return
            time.sleep(0.2)
        # Phase 2: request cancel for residual running.
        self._request_cancel_running()
        # Phase 3: shorter force-timeout, then mark stragglers interrupted.
        force_deadline = time.monotonic() + min(5.0, max(0.2, timeout / 2.0))
        while time.monotonic() < force_deadline:
            with self._lock:
                running = len(self._running)
            if running == 0 and self._pending.empty():
                self._state = "stopped"
                return
            time.sleep(0.2)
        self._interrupt_running()
        self._state = "stopped"

    def stop(self) -> None:
        """Immediate stop: stop consuming; running jobs are not drained."""
        self._state = "stopped"

    # -- admission (R21-049..051) -------------------------------------------
    def _retry_after(self) -> int:
        est_wait = float(os.environ.get("FACTOR_ENGINE_SERVICE_RETRY_AFTER", "10"))
        return max(1, int(est_wait))

    def submit(
        self,
        job: JobRecord,
        *,
        run_fn: Callable[[JobRecord], None],
        estimate: JobResourceEstimate | None = None,
    ) -> None:
        if self._state != "accepting":
            raise ServiceError(
                "JOB_REJECTED", "service is shutting down; not accepting new jobs",
                status=503, extra={"Retry-After": "5"},
            )
        principal = job.owner_principal or job.requested_by or "anonymous"
        with self._lock:
            running_now = len(self._running)
            queued_now = self._pending.qsize()
            per_p = self._per_principal.get(principal, 0)
        # R38 P0-028（§11）：QoS lane 接资源 controller——压力升高时 BACKGROUND
        # 停止新 admission，STANDARD 收缩，CRITICAL 保底（不能只靠 FIFO）。
        lane = (estimate.priority if estimate is not None else QOS_STANDARD)
        if self.broker is not None:
            try:
                stage = self.broker.pressure_stage()
                if lane == QOS_BACKGROUND and stage in {"PRESSURE_2", "PRESSURE_3", "PRESSURE_4", "CRITICAL"}:
                    self.jobs_rejected_total += 1
                    raise ServiceError(
                        "JOB_QOS_DEFERRED",
                        f"BACKGROUND lane paused under pressure_stage={stage} (R38-P0-028)",
                        status=429,
                        extra={"Retry-After": str(self._retry_after())},
                    )
            except ServiceError:
                raise
            except Exception:
                pass
        if running_now + queued_now >= self.max_queue + self.max_running:
            self.jobs_rejected_total += 1
            raise ServiceError(
                "JOB_QUEUE_FULL",
                f"job queue full (running={running_now}, queued={queued_now})",
                status=429,
                extra={"Retry-After": str(self._retry_after())},
            )
        if per_p >= int(os.environ.get("FACTOR_ENGINE_SERVICE_PER_PRINCIPAL_JOBS", "8")):
            self.jobs_rejected_total += 1
            raise ServiceError(
                "JOB_QUEUE_FULL",
                f"per-principal job limit exceeded for {principal!r}",
                status=429,
                extra={"Retry-After": str(self._retry_after())},
            )
        with self._lock:
            self._per_principal[principal] = self._per_principal.get(principal, 0) + 1
            self._queue_put_times[job.run_id] = time.monotonic()
            if estimate is not None:
                self._job_estimates[job.run_id] = estimate
        # R32-P0-020: 检查与 put 必须原子 —— 用 put_nowait，绝不阻塞请求线程。
        # 失败时 rollback per-principal counter。
        job.status = JobStatus.QUEUED
        job.queued_at = _utc_now()
        try:
            self._pending.put_nowait((job, run_fn))
        except queue.Full:
            with self._lock:
                self._per_principal[principal] = max(
                    0, self._per_principal.get(principal, 0) - 1
                )
                self._queue_put_times.pop(job.run_id, None)
                self._job_estimates.pop(job.run_id, None)
            self.jobs_rejected_total += 1
            raise ServiceError(
                "JOB_QUEUE_FULL",
                f"job queue full at enqueue (running={running_now}, queued={queued_now})",
                status=429,
                extra={"Retry-After": str(self._retry_after())},
            )
        self.jobs_submitted_total += 1

    def cancel(self, run_id: str) -> bool:
        """Request cancellation; returns True if the job was found non-terminal.

        R40 #101: 除了持久化 cancel 状态，还 set 该 job 的 cancel event —— 正在跑
        的 worker 线程在下一个阶段边界真正停止计算。
        """
        with self._lock:
            job = self._store.get(run_id) if self._store else None
            if job is None or job.is_terminal:
                return False
            job.cancel_requested_at = _utc_now()
            job.status = JobStatus.CANCELLING
            self._store.update(job)
            self._set_cancel_event(run_id)
            return True

    # -- worker loop ----------------------------------------------------------
    def _worker_loop(self) -> None:
        # R32-P0-019: DRAINING 期间仍消费 queued job，只有 STOPPED 才停止。
        while self._state != "stopped":
            try:
                job, run_fn = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._begin(job, run_fn)
            except BaseException:  # noqa: BLE001 - worker must never die
                # R32-P0-021: _begin 内已兜底；此处为最后防线 —— unexpected
                # exception 绝不能 kill worker thread（worker pool 容量永久减少）。
                try:
                    self._mark_unexpected_failure(job)
                except Exception:
                    pass
            finally:
                # R40 #98: 每出队一个 job 必须 task_done() —— 否则 queue.join()
                # 永不返回（pending 计数永远 > 0）。
                self._pending.task_done()

    def _mark_unexpected_failure(self, job: JobRecord) -> None:
        """R32-P0-021: unexpected run_fn exception → job FAILED + durable + 继续服务。"""
        if job.is_terminal or not self._store:
            return
        job.status = JobStatus.FAILED
        job.error_code = "JOB_UNEXPECTED_WORKER_EXCEPTION"
        job.error = "unexpected exception in run_fn; worker survived"
        job.finished_at = _utc_now()
        self._store.update(job)

    def _begin(self, job: JobRecord, run_fn: Callable[[JobRecord], None]) -> None:
        with self._lock:
            self._running[job.run_id] = job
            # R40 #101: 注册该 job 的 request-scoped cancel event。
            self._cancel_events[job.run_id] = threading.Event()
        # R36 P0-018（§58）：ContextVar——每个 worker 线程自己的 job 执行期间
        # 可见 shared broker；互不覆盖，A restore 不会影响 B 仍在运行的任务。
        token = _service_broker_ctx_var.set(self.broker)
        # R40 #101: cancel event 进 ContextVar —— worker 阶段间 check_job_alive /
        # set_phase 能读到并抛 JobCancelledError。
        cancel_token = _cancel_event_ctx_var.set(self._cancel_events[job.run_id])
        # R38 P0-027（§11）：worker 启动前真正申请 JobLease（不再只靠固定
        # max_running 数并发）。HostCoordinator 拒绝 → job FAILED，不执行。
        job_lease = self._request_job_lease(job)
        if job_lease is None and self.coordinator is not None:
            _service_broker_ctx_var.reset(token)
            _cancel_event_ctx_var.reset(cancel_token)
            with self._lock:
                self._running.pop(job.run_id, None)
                self._queue_put_times.pop(job.run_id, None)
                self._cancel_events.pop(job.run_id, None)
                principal = job.owner_principal or job.requested_by or "anonymous"
                self._per_principal[principal] = max(0, self._per_principal.get(principal, 0) - 1)
            self.jobs_lease_rejected_total += 1
            self._mark_resource_rejected(job)
            return
        # P0-018：job 执行期间把 JobLease 设为当前 context 的活跃 job lease——
        # FE scheduler / DA scan / writer 的 child lease 从同一棵 lease 树申请。
        if job_lease is not None and self.coordinator is not None:
            self.coordinator.set_active_job_lease(job_lease)
        try:
            run_fn(job)
        except BaseException:  # noqa: BLE001
            # R32-P0-021: run_fn 抛出的 unexpected exception 在此兜底 —— 标 FAILED、
            # 持久化、绝不向上传播到 worker loop 杀掉线程。若 run_fn 内部已把 job
            # 标成 terminal（SUCCEEDED/FAILED/…），store.update 的 terminal guard
            # 会拒绝覆盖，保持 terminal 状态权威。
            self._mark_unexpected_failure(job)
        finally:
            if self.coordinator is not None:
                try:
                    self.coordinator.set_active_job_lease(None)
                except Exception:
                    pass
            _service_broker_ctx_var.reset(token)
            _cancel_event_ctx_var.reset(cancel_token)
            if job_lease is not None:
                try:
                    job_lease.release()
                except Exception:
                    pass
            with self._lock:
                self._running.pop(job.run_id, None)
                self._queue_put_times.pop(job.run_id, None)
                self._job_estimates.pop(job.run_id, None)
                self._job_leases.pop(job.run_id, None)
                self._cancel_events.pop(job.run_id, None)
                principal = job.owner_principal or job.requested_by or "anonymous"
                self._per_principal[principal] = max(0, self._per_principal.get(principal, 0) - 1)

    def _request_job_lease(self, job: JobRecord) -> Any | None:
        """按 JobResourceEstimate 向 HostCoordinator 申请 JobLease。"""
        if self.coordinator is None:
            return None
        estimate = self._job_estimates.get(job.run_id)
        memory = max(0, int(estimate.memory_p99) if estimate is not None else 0)
        cpu = max(1, int(estimate.cpu_budget) if estimate is not None else 1)
        if memory <= 0:
            # 无显式估计：用 coordinator envelope 的合理默认（safe 的 25%）。
            try:
                env = self.coordinator.envelope()
                memory = max(1, int(env.safe_memory_bytes * 0.25))
            except Exception:
                memory = 1
        lease = self.coordinator.request_job_lease(
            owner=f"job:{job.run_id}", memory_bytes=memory, cpu_tokens=cpu,
        )
        if lease is not None:
            with self._lock:
                self._job_leases[job.run_id] = lease
        return lease

    def _mark_resource_rejected(self, job: JobRecord) -> None:
        """HostCoordinator JobLease 拒绝 → job FAILED（资源不可用，不静默）。

        R40 #102: 持久化失败绝不 ``except Exception: pass`` 静默吞掉 —— warning
        记日志 + 写 ``{store.root}/emergency_journal.jsonl`` 兜底；连兜底都失败则
        计数 ``resource_reject_fatal_counter``（zombie job 必须可见）。
        """
        if job.is_terminal or not self._store:
            return
        job.status = JobStatus.FAILED
        job.error_code = "JOB_RESOURCE_LEASE_REJECTED"
        job.error = "host resource lease rejected (memory/cpu over committed); job not started"
        job.finished_at = _utc_now()
        try:
            self._store.update(job)
        except Exception as exc:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "failed to persist resource-rejected job %s: %s", job.run_id, exc
            )
            self.resource_reject_fatal_counter += 1
            try:
                journal = self._store.root / "emergency_journal.jsonl"
                with journal.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "run_id": job.run_id,
                                "event": "resource_reject_persist_failed",
                                "error": str(exc),
                                "job_error": job.error,
                                "ts": _utc_now(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:  # noqa: BLE001 - 连兜底 journal 都失败，计数已暴露
                pass

    def _heartbeat_monitor(self) -> None:
        while self._state != "stopped":
            time.sleep(self.heartbeat_check_interval)
            if self._store is None:
                continue
            now = time.monotonic()
            for job in self._store.list_jobs():
                if job.status != JobStatus.RUNNING:
                    continue
                # R40 #100: heartbeat_at 为 None（从未心跳）的 RUNNING job 不能静默
                # 跳过 —— 用 started_at（没有则 submitted_at）算墙钟年龄。
                last_hb = job.heartbeat_at or job.started_at or job.submitted_at
                if not last_hb:
                    continue  # 连 submitted_at 都没有：跳过（极端异常）
                try:
                    hb_ts = datetime.fromisoformat(last_hb).timestamp()
                    # Wall-clock comparison is fine for staleness detection; the
                    # deadline itself uses monotonic-safe duration accounting.
                    wall_age = time.time() - hb_ts
                except ValueError:
                    wall_age = float("inf")
                if wall_age > self.heartbeat_stale_seconds:
                    # R32-P0-022: CAS 迁移 RUNNING → INTERRUPTED。真实线程随后
                    # 写 SUCCEEDED 会被 store 的 terminal guard 拒绝覆盖。
                    interrupted = JobRecord(**vars(job))
                    interrupted.status = JobStatus.INTERRUPTED
                    interrupted.error_code = "JOB_INTERRUPTED"
                    interrupted.error = (
                        f"heartbeat stale {wall_age:.0f}s; worker considered stuck"
                    )
                    interrupted.finished_at = _utc_now()
                    try:
                        ok = self._store.cas_transition(
                            interrupted,
                            expected_status=JobStatus.RUNNING,
                            write_manifest=True,
                        )
                    except AttributeError:
                        # 非 JobStore 兼容后端：fallback 到 update（terminal guard
                        # 仍防止后续覆盖）。
                        self._store.update(interrupted)
                        ok = True
                    if ok:
                        # R40 #101: 标 INTERRUPTED 的同时 set cancel event —— 真正
                        # 停止仍在跑的 worker 线程（不再白烧 CPU / 发布旧代结果）。
                        self._set_cancel_event(job.run_id)

    # -- metrics --------------------------------------------------------------
    def snapshot(self) -> QueueSnapshot:
        with self._lock:
            queued = self._pending.qsize()
            running = len(self._running)
        waits = [time.monotonic() - t for t in self._queue_put_times.values()]
        longest = max(waits) if waits else 0.0
        return QueueSnapshot(
            queued=queued, running=running,
            max_queue=self.max_queue, max_running=self.max_running,
            longest_wait_seconds=round(longest, 2),
        )


def deadline_remaining(job: JobRecord) -> Optional[float]:
    """R21-277: duration accounting uses the monotonic clock; the wall-clock
    ``deadline_at`` is display-only, so a system clock rollback cannot disable
    a timeout."""
    if job.deadline_monotonic is not None:
        return job.deadline_monotonic - time.monotonic()
    if not job.deadline_at:
        return None
    try:
        deadline_ts = datetime.fromisoformat(job.deadline_at).timestamp()
    except ValueError:
        return None
    return deadline_ts - time.time()


def check_job_alive(job: JobRecord) -> None:
    """Raise if the job is cancelled/timed-out — checked between phases.

    R40 #101: 先查 request-scoped cancel event（heartbeat/drain 标 INTERRUPTED
    时 set）—— 即使 job.cancel_requested_at 尚未写入，worker 也立即停止。
    """
    cancel_event = _cancel_event_ctx_var.get()
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("job cancel requested (cancel event set)")
    if job.cancel_requested_at and job.status in {JobStatus.CANCELLING, JobStatus.RUNNING}:
        raise JobCancelledError("job cancel requested")
    remaining = deadline_remaining(job)
    if remaining is not None and remaining <= 0:
        raise JobDeadlineExceeded(f"job deadline exceeded at {job.deadline_at}")


def set_phase(job: JobRecord, store: Any, phase: str) -> None:
    """R40 #101: 阶段切换也查 cancel event —— worker 在阶段边界停止在跑计算。"""
    cancel_event = _cancel_event_ctx_var.get()
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError("job cancel requested (cancel event set at phase boundary)")
    job.phase = phase
    job.touch_heartbeat()
    store.update(job)
