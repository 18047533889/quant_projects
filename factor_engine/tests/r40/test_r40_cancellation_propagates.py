"""R40 #96: request-scoped CancellationToken propagates to the engine entry /
scheduler stage boundaries and stops work deterministically."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("FACTOR_ENGINE_SERVICE_ROOT", "/tmp/r40_cancel_root")

from runtime.exceptions import (  # noqa: E402
    Cancellation,
    CancellationToken,
    DeadlineExceeded,
    get_active_cancellation_token,
    reset_active_cancellation_token,
    set_active_cancellation_token,
)
from service.app import _job_wrapper, _set_job_cancellation_token  # noqa: E402
from service.jobstore import JobRecord, JobStore, JobStatus  # noqa: E402


def test_token_raise_if_cancelled():
    token = CancellationToken(deadline_monotonic=None)
    token.cancel()
    with pytest.raises(Cancellation):
        token.raise_if_cancelled()


def test_token_expired_raises_deadline():
    import time

    token = CancellationToken(deadline_monotonic=time.monotonic() - 1)
    with pytest.raises(DeadlineExceeded):
        token.raise_if_cancelled()


def test_token_not_cancelled_passes():
    token = CancellationToken(deadline_monotonic=None)
    token.raise_if_cancelled()  # no raise


def test_job_wrapper_stops_when_token_cancelled(tmp_path):
    store = JobStore(tmp_path)
    token = CancellationToken(deadline_monotonic=None)
    _set_job_cancellation_token("cancel_job", token)
    token.cancel()
    job = JobRecord(
        run_id="cancel_job",
        status="running",
        endpoint_policy="research",
        request={"execution": {}},
    )
    store.create(job)
    _job_wrapper(job, phase_target="EXECUTING")
    # cancel token -> job is CANCELLED, not FAILED / RUNNING
    assert job.status == JobStatus.CANCELLED
    assert job.error_code == "JOB_CANCELLED"


def test_active_token_context_propagates():
    token = CancellationToken(deadline_monotonic=None)
    token.cancel()
    ctx = set_active_cancellation_token(token)
    try:
        assert get_active_cancellation_token() is token
        assert get_active_cancellation_token().is_cancelled is True
    finally:
        reset_active_cancellation_token(ctx)
    assert get_active_cancellation_token() is None


def test_scheduler_stops_admission_when_token_cancelled():
    from runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler

    sched = AdaptiveBatchScheduler()
    token = CancellationToken(deadline_monotonic=None)
    token.cancel()
    ctx = set_active_cancellation_token(token)
    try:
        task = SimpleNamespace(task_id="t1")
        future, lease = sched._admit_and_run(
            task,
            backend=None,
            ctx=None,
            execute_root=None,
            materialize_shared=None,
        )
        assert future is None and lease is None
        assert sched._cancelled is True
    finally:
        reset_active_cancellation_token(ctx)
