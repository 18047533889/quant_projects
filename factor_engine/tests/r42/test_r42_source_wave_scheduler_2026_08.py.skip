from __future__ import annotations

import errno
import types

import pytest

from planner.read_wave_planner import ReadWave, ReadWavePlan
from planner.source_representation import SourceRepresentation
from planner.wave_recovery import WaveRecoveryCategory, classify_wave_failure
from runtime.adaptive_batch_scheduler import (
    ERROR_PERMANENT,
    ERROR_TRANSIENT,
    ERROR_UNKNOWN,
    AdaptiveBatchScheduler,
    SchedulerPlan,
    _plan_cost_bytes,
    _should_retry_error,
    classify_error,
)
from runtime.buffer_ref import (
    BufferRef,
    CacheObjectRef,
    SourceBufferRef,
    SourceWaveExecutionError,
    SourceWaveExecutor,
    WaveExecutionFailed,
    WaveExecutionSuccess,
)


def _wave(**kwargs):
    values = dict(
        wave_id=7,
        source_scope="daily",
        dataset="d",
        snapshot_id="s",
        columns=frozenset({"close"}),
        time_range=None,
        source_tasks=("scan",),
        estimated_scan_bytes=8,
        estimated_memory_bytes=8,
    )
    values.update(kwargs)
    return ReadWave(**values)


def test_source_wave_failure_is_typed_and_never_none():
    class Source:
        def prefetch_columns(self, _columns):
            raise PermissionError(errno.EACCES, "denied")

    result = SourceWaveExecutor(Source()).execute_wave(_wave())
    assert isinstance(result, WaveExecutionFailed)
    assert result.failure.cause.errno == errno.EACCES


def test_legacy_source_certifies_pandas_not_requested_native_label():
    class Source:
        snapshot_token = "snap"

        def prefetch_columns(self, _columns):
            return None

    result = SourceWaveExecutor(Source()).execute_wave(
        _wave(preferred_representation=SourceRepresentation.DUCKDB_RELATION)
    )
    assert isinstance(result, WaveExecutionSuccess)
    assert result.buffer.representation is SourceRepresentation.PANDAS_COLUMNS
    assert isinstance(result.buffer.location, CacheObjectRef)


def test_typed_source_api_proves_actual_representation_and_location():
    expected = SourceBufferRef(
        representation=SourceRepresentation.ARROW_TABLE,
        schema=("close",),
        location=object(),
        wave_id=7,
    )

    class Source:
        def execute_read_wave(self, wave):
            assert wave.wave_id == 7
            return expected

    result = SourceWaveExecutor(Source()).execute_wave(_wave())
    assert isinstance(result, WaveExecutionSuccess)
    assert result.buffer is expected


def test_scheduler_failed_wave_does_not_commit_source_task():
    task = types.SimpleNamespace(task_id="scan", task_type="SOURCE_SCAN", inputs=())
    dag = types.SimpleNamespace(tasks={"scan": task})
    plan = SchedulerPlan(
        physical_dag=dag,
        read_waves=ReadWavePlan(waves=[_wave()]),
    )

    class Source:
        def prefetch_columns(self, _columns):
            raise ValueError("schema mismatch")

    scheduler = AdaptiveBatchScheduler()
    committed: set[str] = set()
    remaining = {"scan"}
    with pytest.raises(SourceWaveExecutionError):
        scheduler._execute_read_waves(
            plan, dag, committed, remaining, types.SimpleNamespace(data_source=Source())
        )
    assert committed == set()
    assert remaining == {"scan"}
    assert scheduler._buffer_results == {}


def test_buffer_meta_is_deep_frozen():
    original = {"axis": {"names": ["instrument", "time"]}}
    ref = BufferRef(
        representation=SourceRepresentation.NUMPY_BLOCK,
        meta=original,
    )
    original["axis"]["names"].append("mutated")
    assert ref.meta["axis"]["names"] == ("instrument", "time")
    with pytest.raises(TypeError):
        ref.meta["axis"] = {}


def test_unknown_cost_is_conservative(monkeypatch):
    import backend.operator_cost as operator_cost

    monkeypatch.setattr(operator_cost, "estimate_plan_cost", lambda _plan: (_ for _ in ()).throw(RuntimeError("bad")))
    cost = _plan_cost_bytes(object())
    assert cost["cost_unknown"] is True
    assert cost["peak_live_memory_bytes"] > 0
    assert cost["total_work"] > 0


def test_unknown_error_never_retries_by_default():
    assert classify_error(RuntimeError("programming bug")) == ERROR_UNKNOWN
    assert _should_retry_error(ERROR_UNKNOWN, 1) is False
    assert _should_retry_error(ERROR_TRANSIENT, 1) is True


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (errno.EAGAIN, ERROR_TRANSIENT),
        (errno.ETIMEDOUT, ERROR_TRANSIENT),
        (errno.ECONNRESET, ERROR_TRANSIENT),
        (errno.ENOSPC, ERROR_PERMANENT),
        (errno.EACCES, ERROR_PERMANENT),
        (errno.EROFS, ERROR_PERMANENT),
        (errno.EINVAL, ERROR_PERMANENT),
        (None, ERROR_UNKNOWN),
    ],
)
def test_oserror_classification_is_errno_aware(code, expected):
    exc = OSError("unknown") if code is None else OSError(code, "io")
    assert classify_error(exc) == expected
    if code == errno.EAGAIN:
        assert classify_wave_failure(exc) is WaveRecoveryCategory.TRANSIENT_IO
    elif code == errno.ENOSPC:
        assert classify_wave_failure(exc) is WaveRecoveryCategory.PERMANENT_SOURCE_ERROR
