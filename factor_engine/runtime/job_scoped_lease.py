"""Pair per-job coordinator children with broker admission leases."""
from __future__ import annotations

import threading
import contextvars
from contextlib import contextmanager
from typing import Any, Callable


_SUPPRESS_BROKER_JOB_HOOK = contextvars.ContextVar(
    "fe_suppress_broker_job_hook", default=False,
)


def broker_job_hook_suppressed() -> bool:
    return bool(_SUPPRESS_BROKER_JOB_HOOK.get())


@contextmanager
def suppress_broker_job_hook():
    token = _SUPPRESS_BROKER_JOB_HOOK.set(True)
    try:
        yield
    finally:
        _SUPPRESS_BROKER_JOB_HOOK.reset(token)


class SharedChildRelease:
    """Release one job child after all associated physical leases release."""

    def __init__(self, job_lease: Any, child_lease: Any, count: int) -> None:
        self._job_lease = job_lease
        self._child_lease = child_lease
        self._remaining = int(count)
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            if self._remaining <= 0:
                return
            self._remaining -= 1
            final = self._remaining == 0
        if final:
            self._job_lease.release_child(self._child_lease)


class CombinedLease:
    """Release broker accounting first, then the job child, exactly once."""

    def __init__(self, broker_lease: Any, job_lease: Any, child_lease: Any) -> None:
        self._broker_lease = broker_lease
        self._job_lease = job_lease
        self._child_lease = child_lease
        self._released = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        try:
            self._broker_lease.release()
        finally:
            self._job_lease.release_child(self._child_lease)

    def transfer_memory_ownership(
        self, kind: Any, nbytes: int, *, lease_id: str,
    ) -> Any | None:
        """Atomically transfer broker accounting and keep the explicit child."""
        with self._lock:
            if self._released:
                return None
            lease = self._broker_lease.transfer_memory_ownership(
                kind, nbytes, lease_id=lease_id,
                on_release=lambda: self._job_lease.release_child(self._child_lease),
            )
            if lease is None:
                return None
            self._released = True
            return lease


def _paired(
    job_lease: Any | None,
    *,
    owner: str,
    kind: str,
    nbytes: int,
    acquire_broker: Callable[[], Any | None],
) -> Any | None:
    if job_lease is None:
        return acquire_broker()
    if bool(getattr(job_lease, "released", False)):
        return None
    child = job_lease.request_child(
        owner=owner, kind=kind, memory_bytes=max(0, int(nbytes))
    )
    if child is None:
        return None
    try:
        with suppress_broker_job_hook():
            broker_lease = acquire_broker()
    except BaseException:
        job_lease.release_child(child)
        raise
    if broker_lease is None:
        job_lease.release_child(child)
        return None
    return CombinedLease(broker_lease, job_lease, child)


def acquire_job_scoped_memory(
    broker: Any,
    job_lease: Any | None,
    kind: Any,
    nbytes: int,
    *,
    lease_id: str,
) -> Any | None:
    job_broker = getattr(job_lease, "broker", broker) if job_lease is not None else broker
    if job_broker is not broker:
        return None
    return _paired(
        job_lease,
        owner=lease_id,
        kind="read_wave",
        nbytes=nbytes,
        acquire_broker=lambda: broker.acquire_memory(
            kind, nbytes, lease_id=lease_id
        ),
    )


def reserve_job_scoped_task(
    broker: Any,
    job_lease: Any | None,
    contract: Any,
    *,
    task_id: str,
) -> Any | None:
    job_broker = getattr(job_lease, "broker", broker) if job_lease is not None else broker
    if job_broker is not broker:
        return None
    nbytes = int(getattr(contract, "admissible_peak_bytes", 0) or 0)
    return _paired(
        job_lease,
        owner=task_id,
        kind="compute",
        nbytes=nbytes,
        acquire_broker=lambda: broker.try_reserve(contract, task_id=task_id),
    )
