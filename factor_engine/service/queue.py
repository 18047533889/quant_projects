"""R21-048..060: bounded job queue, admission control, deadline/cancel/heartbeat.

A bounded FIFO queue (``queue.Queue(maxsize=...)``) + fixed worker threads.
A full queue returns 429/503 with ``Retry-After`` (R21-049); global and
per-principal running/queued limits are enforced (R21-050).  Every job has a
deadline, cancel flag, heartbeat and real phase so a job can always be
cancelled, timed out, or recovered (R21-055..060).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from service.errors import ServiceError, classify_exception, sanitize_message
from service.jobstore import JobPhase, JobRecord, JobStatus, _utc_now


class JobCancelledError(RuntimeError):
    pass


class JobDeadlineExceeded(RuntimeError):
    pass


#: R31-P1-039：service 全局共享的 ResourceBroker（job admission + task admission
#: 统一）。worker 执行 job 时通过 contextvar 传给内部 scheduler，避免「4 个 HTTP
#: job × 16 内部 workers = 64 runnable」的过载。
_service_broker_ctx: "Any | None" = None


def _get_service_broker() -> Any | None:
    return _service_broker_ctx


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
        self.queue_full_status = queue_full_status
        self._pending: "queue.Queue[JobRecord]" = queue.Queue(maxsize=self.max_queue)
        self._store: Any = None
        self._lock = threading.RLock()
        self._running: dict[str, JobRecord] = {}
        self._per_principal: dict[str, int] = {}
        self._workers: list[threading.Thread] = []
        self._started = False
        # R32-P0-019: 显式生命周期状态机 ACCEPTING → DRAINING → STOPPED。
        # DRAINING 拒绝新任务但 worker 继续清空已有队列 —— 历史实现一开始就把
        # ``_stopping=True``，worker loop ``while not _stopping`` 立即不再消费
        # queued job，与 drain 语义冲突。
        self._state = "accepting"
        self._queue_put_times: dict[str, float] = {}
        self.jobs_submitted_total = 0
        self.jobs_rejected_total = 0
        # R31-P1-039：service 全局共享 broker（跨 job 统一 admission）。
        try:
            from runtime.resource_broker import ResourceBroker

            self.broker = ResourceBroker()
        except Exception:
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

    def drain(self, timeout: float = 30.0) -> None:
        """Graceful shutdown: DRAINING (reject new, drain queued), then STOPPED.

        R32-P0-019: 先切到 ``draining`` —— worker 仍 ``while state != stopped``
        消费 queued job；等 running==0 且 queue 清空后置 ``stopped``。超时则
        force-mark stragglers interrupted（绝不留下 phantom running）。
        """
        self._state = "draining"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                running = len(self._running)
            if running == 0 and self._pending.empty():
                self._state = "stopped"
                return
            # Ask running jobs to stop.
            with self._lock:
                for job in list(self._running.values()):
                    if not job.is_terminal and not job.cancel_requested_at:
                        job.cancel_requested_at = _utc_now()
                        job.status = JobStatus.CANCELLING
            time.sleep(0.2)
        # Force-mark stragglers interrupted (never leave a phantom running).
        with self._lock:
            for job in list(self._running.values()):
                if not job.is_terminal:
                    job.status = JobStatus.INTERRUPTED
                    job.finished_at = _utc_now()
        self._state = "stopped"

    def stop(self) -> None:
        """Immediate stop: stop consuming; running jobs are not drained."""
        self._state = "stopped"

    # -- admission (R21-049..051) -------------------------------------------
    def _retry_after(self) -> int:
        est_wait = float(os.environ.get("FACTOR_ENGINE_SERVICE_RETRY_AFTER", "10"))
        return max(1, int(est_wait))

    def submit(self, job: JobRecord, *, run_fn: Callable[[JobRecord], None]) -> None:
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
            self.jobs_rejected_total += 1
            raise ServiceError(
                "JOB_QUEUE_FULL",
                f"job queue full at enqueue (running={running_now}, queued={queued_now})",
                status=429,
                extra={"Retry-After": str(self._retry_after())},
            )
        self.jobs_submitted_total += 1

    def cancel(self, run_id: str) -> bool:
        """Request cancellation; returns True if the job was found non-terminal."""
        with self._lock:
            job = self._store.get(run_id) if self._store else None
            if job is None or job.is_terminal:
                return False
            job.cancel_requested_at = _utc_now()
            job.status = JobStatus.CANCELLING
            self._store.update(job)
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
        global _service_broker_ctx
        with self._lock:
            self._running[job.run_id] = job
        prev_broker = _service_broker_ctx
        _service_broker_ctx = self.broker  # job 执行期间共享 broker
        try:
            run_fn(job)
        except BaseException:  # noqa: BLE001
            # R32-P0-021: run_fn 抛出的 unexpected exception 在此兜底 —— 标 FAILED、
            # 持久化、绝不向上传播到 worker loop 杀掉线程。若 run_fn 内部已把 job
            # 标成 terminal（SUCCEEDED/FAILED/…），store.update 的 terminal guard
            # 会拒绝覆盖，保持 terminal 状态权威。
            self._mark_unexpected_failure(job)
        finally:
            _service_broker_ctx = prev_broker
            with self._lock:
                self._running.pop(job.run_id, None)
                self._queue_put_times.pop(job.run_id, None)
                principal = job.owner_principal or job.requested_by or "anonymous"
                self._per_principal[principal] = max(0, self._per_principal.get(principal, 0) - 1)

    def _heartbeat_monitor(self) -> None:
        while self._state != "stopped":
            time.sleep(15.0)
            if self._store is None:
                continue
            now = time.monotonic()
            for job in self._store.list_jobs():
                if job.status != JobStatus.RUNNING:
                    continue
                last_hb = job.heartbeat_at
                if not last_hb:
                    continue
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
                        self._store.cas_transition(
                            interrupted,
                            expected_status=JobStatus.RUNNING,
                            write_manifest=True,
                        )
                    except AttributeError:
                        # 非 JobStore 兼容后端：fallback 到 update（terminal guard
                        # 仍防止后续覆盖）。
                        self._store.update(interrupted)

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
    """Raise if the job is cancelled/timed-out — checked between phases."""
    if job.cancel_requested_at and job.status in {JobStatus.CANCELLING, JobStatus.RUNNING}:
        raise JobCancelledError("job cancel requested")
    remaining = deadline_remaining(job)
    if remaining is not None and remaining <= 0:
        raise JobDeadlineExceeded(f"job deadline exceeded at {job.deadline_at}")


def set_phase(job: JobRecord, store: Any, phase: str) -> None:
    job.phase = phase
    job.touch_heartbeat()
    store.update(job)
