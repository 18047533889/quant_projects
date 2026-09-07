"""Single-call, bounded-wave execution for streams of independent factor definitions."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from operator import index
import sys
from threading import Lock
from time import monotonic
from typing import Any

from factor_engine.api.factor import Factor
from factor_engine.planner.dag import DuplicateFactorNameError


@dataclass
class _QueuedResult:
    value: Any
    nbytes: int


class _PermanentSinkFailure(RuntimeError):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"permanent user sink failure: {cause}")
        self.cause = cause


class _SinkDeliveryRefused(RuntimeError):
    """Internal wake-up after the writer has already failed."""


def _transfer_result(value: Any) -> _QueuedResult:
    """Attach a conservative queue charge without weakening sink fail-closed rules."""
    from factor_engine.runtime.resource_governor import estimate_object_bytes
    from factor_engine.runtime.streaming_result_sink import ResultSizeUnknown

    try:
        size = int(estimate_object_bytes(value))
    except Exception as exc:
        raise ResultSizeUnknown("stream result size estimate failed") from exc
    if size <= 0 and isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
        size = sys.getsizeof(value)
    if size <= 0:
        raise ResultSizeUnknown("stream result size is unknown; ownership transfer refused")
    return _QueuedResult(value=value, nbytes=size)


def execute_run_many_stream(
    engine: Any,
    factors: Iterable[Factor],
    *,
    sink: Any,
    wave_size: int | None = None,
    sink_queue_bytes: int | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Compute bounded waves while transferring results to a bounded writer queue.

    All execution, resource admission, production checks and sink failure behavior
    delegate to run_many_parallel. Names are kept globally to reject duplicates;
    factor definitions, DAGs, paths and outputs are not retained across waves.
    """
    from factor_engine.runtime.batch_service import _validate_result_policy
    from factor_engine.planner.physical_lowerer import (
        _get_adaptive_chunk_size, _get_adaptive_dag_width_limit,
    )

    _validate_result_policy("sink", sink)
    width_limit = _get_adaptive_dag_width_limit()
    if wave_size is None:
        wave_size = min(_get_adaptive_chunk_size(), width_limit)
    if isinstance(wave_size, bool):
        raise ValueError("wave_size must be a positive integer")
    try:
        wave_size = index(wave_size)
    except TypeError as exc:
        raise ValueError("wave_size must be a positive integer") from exc
    if not 1 <= wave_size <= width_limit:
        raise ValueError(f"wave_size must be between 1 and host DAG width limit {width_limit}")

    if sink_queue_bytes is None:
        perf = run_kwargs.get("perf")
        sink_queue_bytes = getattr(perf, "result_budget_bytes", None) if perf is not None else None
    if isinstance(sink_queue_bytes, bool) or not isinstance(sink_queue_bytes, int) or sink_queue_bytes <= 0:
        raise ValueError("sink_queue_bytes or perf.result_budget_bytes must provide a positive explicit budget")

    from factor_engine.runtime.streaming_result_sink import StreamingResultSink

    pending = iter(factors)
    names: set[str] = set()
    requested = completed = waves = 0
    expected: set[str] = set()
    state_lock = Lock()
    started = monotonic()
    wave_execution_seconds = sink_submit_seconds = 0.0

    def write_batch(items):
        for item in items:
            try:
                accepted = sink(item.name, item.value.value)
                if accepted is False:
                    raise RuntimeError(f"sink rejected factor {item.name!r}")
            except BaseException as exc:
                # Arbitrary user callbacks have no receipt/idempotency proof.
                # Force permanent classification so timeout-like failures are
                # never replayed by StreamingResultSink.
                raise _PermanentSinkFailure(exc) from exc

    delivery = StreamingResultSink(
        writer=write_batch,
        queue_bytes=sink_queue_bytes,
        batch_size=1,
        writer_threads=1,
    )
    delivery.start()

    def write(name, result):
        nonlocal completed, sink_submit_seconds
        # Reserve delivery ownership atomically. Potentially blocking queue I/O
        # occurs after releasing this lock, so one slow sink does not serialize
        # concurrent compute completions.
        with state_lock:
            if name not in expected:
                raise RuntimeError(f"streaming wave delivered unexpected or duplicate factor {name!r}")
            expected.remove(name)
        try:
            owned = _transfer_result(result)
            submit_started = monotonic()
            accepted = delivery.submit(name, owned)
            submitted = monotonic() - submit_started
        except Exception:
            raise
        with state_lock:
            sink_submit_seconds += submitted
            if accepted:
                completed += 1
        if not accepted:
            raise _SinkDeliveryRefused(f"bounded streaming sink refused factor {name!r}")
        return True

    primary_error: BaseException | None = None
    try:
        while True:
            wave = list(islice(pending, wave_size))
            if not wave:
                break
            for factor in wave:
                if factor.name in names:
                    raise DuplicateFactorNameError(
                        f"run_many_stream requires globally unique factor names: {factor.name!r}"
                    )
                names.add(factor.name)
            requested += len(wave)
            expected = {factor.name for factor in wave}
            before = completed
            compute_started = monotonic()
            output = engine.run_many_parallel(
                wave, result_policy="sink", sink=write, **run_kwargs,
            )
            wave_execution_seconds += monotonic() - compute_started
            if output.get("results"):
                raise RuntimeError("streaming wave unexpectedly retained factor results")
            if expected or completed - before != len(wave):
                raise RuntimeError("streaming wave did not deliver every factor to the bounded sink")
            waves += 1
            del output, wave, factor
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        input_close_error: BaseException | None = None
        close_input = getattr(pending, "close", None)
        if callable(close_input):
            try:
                close_input()
            except BaseException as exc:
                input_close_error = exc
        finish_started = monotonic()
        try:
            delivery.finish()
        except Exception:
            fatal = delivery.fatal_error
            # Finalization must not replace an exception that already aborted
            # computation.  The sink is still fully closed/joined above, but
            # its concurrent failure is secondary to a genuine compute failure.
            # A delivery-refused error is only the producer wake-up caused by
            # that sink failure, so preserve the original user exception there.
            if primary_error is not None and not isinstance(primary_error, _SinkDeliveryRefused):
                raise primary_error
            if isinstance(fatal, _PermanentSinkFailure):
                raise fatal.cause
            if primary_error is not None:
                raise primary_error
            if fatal is not None:
                raise fatal
            raise
        if input_close_error is not None and primary_error is None:
            raise input_close_error
        sink_finish_wait_seconds = monotonic() - finish_started

    sink_summary = delivery.summary()
    wall_seconds = monotonic() - started

    return {
        "results": {},
        "result_policy": "sink",
        "executor": "streaming_waves",
        "requested_factors": requested,
        "completed_factors": completed,
        "completed_waves": waves,
        "wave_size": wave_size,
        "cse_scope": "per_wave",
        "seconds": wall_seconds,
        "cost_ledger": {
            "wall_seconds": wall_seconds,
            "wave_execution_wall_seconds": wave_execution_seconds,
            "sink_queue_wait_seconds": None,
            "sink_submit_wall_seconds": sink_submit_seconds,
            "sink_finish_wait_seconds": sink_finish_wait_seconds,
            "cse_scope": "per_wave",
            "global_cse": False,
            "name_metadata_entries": len(names),
            "name_utf8_payload_bytes": sum(len(name.encode("utf-8")) for name in names),
            "name_set_container_bytes": None,
            "sink_queue_capacity_bytes": sink_summary["total_queue_capacity_bytes"],
            "sink_accepted": sink_summary["accepted"],
            "sink_committed": sink_summary["committed"],
        },
    }
