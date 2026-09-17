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
from uuid import uuid4

from factor_engine.backend.operator_errors import OperatorParameterError
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


def _source_has_snapshot_proof(source: Any) -> bool:
    """Require an explicit immutable/versioned source identity for reuse.

    Dataset names and filesystem roots identify a source configuration, not the
    bytes observed by this run. They cannot authorize cross-wave reuse alone.
    """
    missing = object()
    token = getattr(source, "snapshot_token", missing)
    if token is not missing:
        # Sources exposing an authoritative token may have a stable data ID
        # while their manifest changes. A missing token cannot fall back to
        # that weaker identity.
        return type(token) is str and bool(token.strip())
    for name in (
        "data_snapshot_id", "snapshot_id", "generation_id",
        "content_hash",
    ):
        value = getattr(source, name, None)
        # A flag or an empty/mutable container is not a version identity.
        # Accept explicit text tokens and integer generations (including zero),
        # without interpreting arbitrary object truthiness as snapshot proof.
        if (type(value) is str and value.strip()) or type(value) is int:
            return True
    return False


def _bounded_stream_cache(engine: Any, run_kwargs: dict[str, Any]):
    """Return a per-call scoped L2 cache, broker lease and execution clone.

    Unknown/ephemeral source identity or unavailable broker budget deliberately
    keeps the established per-wave behavior.  The transient cache is never
    attached to the caller-owned engine.
    """
    if getattr(engine, "cache", None) is not None:
        return engine, None, None, "existing_engine_cache"
    lease = None
    try:
        from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
        from factor_engine.runtime.engine import FactorEngine
        from factor_engine.storage.cache import CacheManager
        from factor_engine.storage.data_scope import compute_data_scope

        if not isinstance(engine, FactorEngine):
            return engine, None, None, "unsupported_engine"
        refresh = getattr(engine.data_source, "refresh_snapshot", None)
        if callable(refresh):
            refresh()
        if not _source_has_snapshot_proof(engine.data_source):
            return engine, None, None, "no_snapshot_identity"
        source_scope = compute_data_scope(engine.data_source)
        if source_scope.startswith("ephemeral:"):
            return engine, None, None, "ephemeral_source"
        broker = run_kwargs.get("broker") or getattr(engine, "resource_broker", None)
        if broker is None:
            return engine, None, None, "no_resource_broker"
        budget = int(getattr(broker, "current_cse_budget")())
        perf = run_kwargs.get("perf")
        if perf is not None:
            plan = perf.build_resource_plan()
            configured = getattr(plan, "cse_budget_bytes", None) if plan is not None else None
            if configured is not None:
                budget = min(budget, int(configured))
        if budget <= 0:
            return engine, None, None, "no_cse_budget"
        lease = broker.acquire_memory(
            MemoryLeaseKind.CSE_CACHE, budget,
            lease_id=f"run-many-stream-cross-wave:{uuid4().hex}",
        )
        if lease is None:
            return engine, None, None, "cse_lease_refused"
        cache = CacheManager(data_scope=source_scope, budget_bytes=budget)
        clone = FactorEngine(
            engine.backend, engine.data_source, cache=cache,
            run_mode=engine.run_mode,
            production_fallback_policy=engine.production_fallback_policy,
            execution_scope=engine.execution_scope,
        )
        for name in ("default_execution_policy", "resource_broker", "execution_purpose"):
            if hasattr(engine, name):
                setattr(clone, name, getattr(engine, name))
        return clone, cache, lease, "transient_bounded_l2"
    except Exception:
        if lease is not None:
            lease.release()
        # Cache reuse is an optimization.  Production correctness remains the
        # existing per-wave path when identity/budget setup is unavailable.
        return engine, None, None, "cache_setup_unavailable"


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
    _wave_runner: str = "parallel",
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Compute bounded waves while transferring results to a bounded writer queue.

    All execution, resource admission, production checks and sink failure behavior
    delegate to the selected public batch runner. Names are kept globally to reject duplicates;
    factor definitions, DAGs, paths and outputs are not retained across waves.
    """
    from factor_engine.runtime.batch_service import _validate_result_policy
    from factor_engine.planner.physical_lowerer import (
        _get_adaptive_chunk_size, _get_adaptive_dag_width_limit,
    )

    _validate_result_policy("sink", sink)
    if _wave_runner not in {"parallel", "run_many"}:
        raise ValueError("_wave_runner must be 'parallel' or 'run_many'")
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
    if sink_queue_bytes is None:
        # v2 automatic path consumes the process/host ResourceBroker authority;
        # it never creates a private queue budget manager.
        broker = run_kwargs.get("broker") or getattr(engine, "resource_broker", None)
        if broker is not None:
            resolver = getattr(broker, "automatic_result_queue_budget", None)
            if callable(resolver):
                sink_queue_bytes = resolver()
            if sink_queue_bytes is None:
                resolver = getattr(broker, "current_sink_budget", None)
                if callable(resolver):
                    sink_queue_bytes = resolver()
    if isinstance(sink_queue_bytes, bool) or not isinstance(sink_queue_bytes, int) or sink_queue_bytes <= 0:
        raise ValueError(
            "an explicit budget from the shared ResourceBroker, sink_queue_bytes, "
            "or perf.result_budget_bytes must provide a positive result queue budget"
        )

    from factor_engine.runtime.streaming_result_sink import StreamingResultSink

    pending = iter(factors)
    names: set[str] = set()
    requested = completed = waves = 0
    failed_factors: dict[str, dict[str, str]] = {}
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

    execution_engine, cross_wave_cache, cross_wave_lease, cross_wave_cache_mode = (
        _bounded_stream_cache(engine, run_kwargs)
    )
    initial_source_scope = None
    cache_reuse_unmeasured = cross_wave_cache is not None or cross_wave_cache_mode == "existing_engine_cache"

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
        if cross_wave_cache is not None:
            # _bounded_stream_cache refreshed before admitting the cache.
            # Bind to the exact scope used at cache construction, not a later
            # unverified token read.
            initial_source_scope = cross_wave_cache.data_scope
        delivery.start()
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
            # Preflight the whole wave before starting it. The common success path
            # keeps batch CSE and avoids O(roots) compile calls. Only when the
            # analyzer reports an explicit unsupported expression do we compile
            # roots independently to identify safe per-root failures.
            compile_many = getattr(execution_engine, "_dag_from_factors", None)
            executable = wave
            compiled = None
            if callable(compile_many):
                compile_kwargs = {
                    key: run_kwargs[key]
                    for key in (
                        "enable_cse", "perf", "pit_enforce", "pit_forbid_forward_fill",
                    )
                    if key in run_kwargs
                }
                def isolatable(exc: BaseException) -> bool:
                    return (
                        type(exc) is OperatorParameterError
                        or isinstance(exc, NotImplementedError)
                        and str(exc).startswith("Unsupported expr:")
                    )
                try:
                    compiled = compile_many(wave, **compile_kwargs)
                except (NotImplementedError, OperatorParameterError) as wave_exc:
                    if not isolatable(wave_exc):
                        raise
                    executable = []
                    for root in wave:
                        try:
                            compile_many([root], **compile_kwargs)
                        except (NotImplementedError, OperatorParameterError) as exc:
                            if not isolatable(exc):
                                raise
                            failed_factors[root.name] = {
                                "phase": "compile_preflight",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        else:
                            executable.append(root)
                    if executable:
                        compiled = compile_many(executable, **compile_kwargs)
            if not executable:
                waves += 1
                del wave, factor
                continue
            if cross_wave_cache is not None:
                from factor_engine.storage.data_scope import compute_data_scope
                from factor_engine.planner.source_dependencies import (
                    build_source_dependency_manifest,
                )

                dag = compiled[0] if compiled is not None else None
                dependency_plans = [
                    root.root for root in (getattr(dag, "roots", ()) or ())
                ]
                dependency_plans.extend(
                    (getattr(dag, "shared_nodes", {}) or {}).values()
                )
                secondary_dependencies = {
                    dependency
                    for plan in dependency_plans
                    for dependency in build_source_dependency_manifest(plan)
                }
                if secondary_dependencies:
                    # The anchor snapshot does not prove the snapshot of a
                    # secondary SourceRef. Until a complete multi-source
                    # snapshot binding exists, fail closed to per-wave reuse.
                    cross_wave_cache.clear_memory()
                    cross_wave_cache = None
                    execution_engine.cache = None
                    if cross_wave_lease is not None:
                        cross_wave_lease.release()
                        cross_wave_lease = None
                    cross_wave_cache_mode = "secondary_source_dependencies"

                if cross_wave_cache is not None:
                    refresh = getattr(execution_engine.data_source, "refresh_snapshot", None)
                    if callable(refresh):
                        refresh()
                    observed_scope = compute_data_scope(execution_engine.data_source)
                    if observed_scope != initial_source_scope:
                        raise RuntimeError(
                            "streaming source identity changed between waves; refusing "
                            "mixed-snapshot cross-wave reuse"
                        )
                    factor_scopes = tuple(sorted({
                        getattr(root, "execution_scope").scope_key()
                        for root in (getattr(dag, "roots", ()) or ())
                    }))
                    if len(factor_scopes) != 1:
                        cross_wave_cache.clear_memory()
                        cross_wave_cache = None
                        execution_engine.cache = None
                        if cross_wave_lease is not None:
                            cross_wave_lease.release()
                            cross_wave_lease = None
                        cross_wave_cache_mode = "mixed_execution_scopes"
                        factor_scopes = ()
                if cross_wave_cache is not None:
                    backend = execution_engine.backend
                    cache_scope = repr((
                        observed_scope,
                        type(backend).__module__, type(backend).__qualname__,
                        factor_scopes,
                    ))
                    execution_engine.cache = cross_wave_cache.with_scope(cache_scope)
                    # Opt in only this transient stream cache to plan-ref key
                    # normalization. Persistent/caller-owned caches retain
                    # their established global key semantics.
                    execution_engine.cache.plan_key_shared_nodes = dict(
                        getattr(dag, "shared_nodes", {}) or {}
                    )
            runner_kwargs = dict(run_kwargs)
            if compiled is not None:
                runner_kwargs["_compiled"] = compiled
            expected = {factor.name for factor in executable}
            before = completed
            compute_started = monotonic()
            runner = (
                execution_engine.run_many
                if _wave_runner == "run_many"
                else execution_engine.run_many_parallel
            )
            output = runner(executable, result_policy="sink", sink=write, **runner_kwargs)
            wave_execution_seconds += monotonic() - compute_started
            if output.get("results"):
                raise RuntimeError("streaming wave unexpectedly retained factor results")
            if expected or completed - before != len(executable):
                raise RuntimeError("streaming wave did not deliver every factor to the bounded sink")
            waves += 1
            del output, wave, factor
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            if cross_wave_cache is not None:
                cross_wave_cache.clear_memory()
        except BaseException as exc:
            cleanup_error = exc
        try:
            if cross_wave_lease is not None:
                cross_wave_lease.release()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
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
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error
        sink_finish_wait_seconds = monotonic() - finish_started

    sink_summary = delivery.summary()
    wall_seconds = monotonic() - started

    return {
        "results": {},
        "result_policy": "sink",
        "executor": "streaming_waves",
        "requested_factors": requested,
        "completed_factors": completed,
        "failed_factors": failed_factors,
        "failed_factor_count": len(failed_factors),
        "completed_waves": waves,
        "wave_size": wave_size,
        "cse_scope": "per_wave",
        "cross_wave_cache_mode": cross_wave_cache_mode,
        "seconds": wall_seconds,
        "cost_ledger": {
            "wall_seconds": wall_seconds,
            "wave_execution_wall_seconds": wave_execution_seconds,
            "sink_queue_wait_seconds": None,
            "sink_submit_wall_seconds": sink_submit_seconds,
            "sink_finish_wait_seconds": sink_finish_wait_seconds,
            "cse_scope": "per_wave",
            "global_cse": False,
            "cross_wave_cache_enabled": cross_wave_cache is not None,
            # Cache eligibility is not a measured cross-wave hit. The native
            # cache currently has no wave-provenance hit counter.
            "cross_wave_value_reuse": None if cache_reuse_unmeasured else False,
            "name_metadata_entries": len(names),
            "name_utf8_payload_bytes": sum(len(name.encode("utf-8")) for name in names),
            "name_set_container_bytes": None,
            "sink_queue_capacity_bytes": sink_summary["total_queue_capacity_bytes"],
            "sink_accepted": sink_summary["accepted"],
            "sink_committed": sink_summary["committed"],
        },
    }
