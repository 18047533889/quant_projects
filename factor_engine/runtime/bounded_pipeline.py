"""Finite v2 orchestration over the existing FactorEngine execution path."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
import uuid
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from factor_engine.runtime.adaptive_batch_scheduler import (
    ERROR_OOM, ERROR_TRANSIENT, classify_error,
)
from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker


class ResultTransportBudgetExceeded(MemoryError):
    pass


def _apply_execution_scope(factor: Any, scope: dict[str, Any] | None) -> tuple[Any, str | None]:
    if not scope:
        return factor, None
    from factor_engine.api.factor import Factor, FactorExecutionScopeHint
    if not isinstance(factor, Factor):
        return factor, "INVALID_FACTOR_DEFINITION"
    hint = factor.semantic_identity or FactorExecutionScopeHint()
    values = {name: getattr(hint, name) for name in FactorExecutionScopeHint.__dataclass_fields__}
    for key in values:
        required = scope.get(key)
        if required is None:
            continue
        if values[key] is not None and values[key] != required:
            return factor, "EXECUTION_SCOPE_MISMATCH"
        values[key] = required
    frequency = scope.get("frequency")
    universe = scope.get("universe_id")
    if frequency is not None and factor.freq not in (None, frequency):
        return factor, "EXECUTION_SCOPE_MISMATCH"
    if universe is not None and factor.universe not in (None, universe):
        return factor, "EXECUTION_SCOPE_MISMATCH"
    return replace(factor, freq=frequency or factor.freq,
                   universe=universe or factor.universe,
                   semantic_identity=FactorExecutionScopeHint(**values)), None


def _policy_value(policy: Any, name: str, default: Any) -> Any:
    return getattr(policy, name)


def _compute_wave(engine: Any, factors: list[Any], run_kwargs: dict[str, Any],
                  transport_budget_bytes: int, broker_proxy: Any = None) -> bytes:
    from factor_engine.runtime.adaptive_batch_scheduler import disable_inner_retries_for_v2
    if broker_proxy is not None:
        engine.resource_broker = broker_proxy
        from factor_engine.runtime.resource_broker import install_v2_resource_broker_proxy
        install_v2_resource_broker_proxy(broker_proxy)
    with disable_inner_retries_for_v2():
        use_physical_isolation = False
        if str(getattr(engine, "run_mode", "")).lower() == "production":
            from factor_engine.storage.sources.data_access_source import DataAccessSource
            source = getattr(engine, "data_source", None)
            use_physical_isolation = bool(
                isinstance(source, DataAccessSource)
                and getattr(source, "production", False)
                and getattr(source, "strict_unknown_fields", False)
                and getattr(source, "pit_enforce", False)
            )
        if use_physical_isolation:
            from factor_engine.runtime.batch_service import execute_run_many_parallel
            output = execute_run_many_parallel(
                engine, factors, result_policy="return",
                isolate_physical_errors=True, **run_kwargs,
            )
        else:
            output = engine.run_many_parallel(factors, result_policy="return", **run_kwargs)
    factor_errors = dict(output.pop("physical_preflight_errors", {}) or {})
    results = output.pop("results", {})
    if not isinstance(results, dict):
        results = dict(results)
    del output
    result_blobs, transport_errors = {}, {}
    used = 1024
    sized = []
    for name, value in results.items():
        try:
            if hasattr(value, "memory_usage"):
                measured = int(value.memory_usage(index=True, deep=True))
            else:
                from factor_engine.runtime.resource_governor import estimate_object_bytes
                measured = int(estimate_object_bytes(value))
        except Exception:
            measured = 0
        if measured <= 0:
            transport_errors[name] = {"code": "RESULT_SIZE_UNKNOWN", "required_bytes": None,
                                      "available_bytes": max(0, transport_budget_bytes - used)}
        else:
            sized.append((measured, name))
    # Admit smaller peers first so one giant output cannot consume the whole
    # envelope and cause otherwise feasible results to be rejected.
    for measured, name in sorted(sized, key=lambda item: (item[0], item[1])):
        value = results.pop(name)
        preflight_charge = 4 * measured + 4096
        if preflight_charge > transport_budget_bytes - used:
            transport_errors[name] = {
                "code": "RESULT_TRANSPORT_EXCEEDS_BUDGET",
                "required_bytes": preflight_charge,
                "available_bytes": max(0, transport_budget_bytes - used),
            }
            del value
            continue
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        del value
        # value pickle, envelope pickle, worker response pickle and pipe frame.
        charge = max(preflight_charge, 4 * (len(blob) + len(name.encode("utf-8")) + 128))
        if charge > transport_budget_bytes - used:
            transport_errors[name] = {
                "code": "RESULT_TRANSPORT_EXCEEDS_BUDGET",
                "required_bytes": charge, "available_bytes": max(0, transport_budget_bytes - used),
            }
        else:
            result_blobs[name] = blob
            used += charge
    results.clear()
    payload = pickle.dumps(
        {"result_blobs": result_blobs, "transport_errors": transport_errors,
         "factor_errors": factor_errors},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    if len(payload) * 2 > transport_budget_bytes:
        raise ResultTransportBudgetExceeded("bounded transport envelope overhead exceeds budget")
    return payload


def _compile_factor(engine: Any, factor: Any) -> bool:
    engine.compile(factor)
    return True


def _compile_wave(engine: Any, factors: list[Any]) -> dict[str, str | None]:
    outcomes = {}
    for factor in factors:
        try:
            engine.compile(factor)
            outcomes[factor.name] = None
        except Exception as exc:
            outcomes[factor.name] = f"{type(exc).__name__}: {exc}"
    return outcomes


def _close_execution_core(engine: Any) -> None:
    close = getattr(engine, "close", None)
    if callable(close):
        close()
        return
    source = getattr(engine, "data_source", None)
    close = getattr(source, "close", None)
    if callable(close):
        close()


class _SpawnFactoryRuntime:
    """Picklable callable that builds exactly one core in each worker."""
    def __init__(self, factory: Any, config: Any, broker_proxy: Any, mode: str) -> None:
        self.factory, self.config = factory, config
        self.broker_proxy, self.mode = broker_proxy, mode
        self._engine = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_engine"] = None
        return state

    def _get_engine(self):
        if self._engine is None:
            from factor_engine.runtime.resource_broker import install_v2_resource_broker_proxy
            install_v2_resource_broker_proxy(self.broker_proxy)
            self._engine = self.factory(self.config)
        return self._engine

    def __call__(self, factors: list[Any], run_kwargs: dict[str, Any] | None = None,
                 transport_budget: int | None = None):
        engine = self._get_engine()
        if self.mode == "compile":
            return _compile_wave(engine, factors)
        return _compute_wave(engine, factors, run_kwargs or {}, int(transport_budget),
                             self.broker_proxy)

    def close(self) -> None:
        if self._engine is not None:
            _close_execution_core(self._engine)
            self._engine = None


def _default_worker_write(root: Path, run_id: str, policy: Any, ordinal: int,
                          name: str, value: Any, budget_bytes: int, generation: str):
    from factor_engine.runtime.durable_artifact_sink import write_verified_factor_artifact
    return write_verified_factor_artifact(
        root, run_id, ordinal, name, value, policy=policy, budget_bytes=budget_bytes,
        generation=generation,
    )


def _custom_worker_write_verify(sink: Any, root: Path, run_id: str, policy: Any,
                                ordinal: int, name: str, value: Any, budget_bytes: int):
    from factor_engine.runtime.durable_artifact_sink import verify_factor_artifact_receipt
    candidate = sink(ordinal, name, value)
    return verify_factor_artifact_receipt(
        candidate, root, run_id, ordinal, name, value,
        policy=policy, budget_bytes=budget_bytes,
    )


def _default_worker_reconcile(root: Path, run_id: str, policy: Any, ordinal: int,
                              name: str, value: Any, budget_bytes: int, generation: str):
    from factor_engine.runtime.durable_artifact_sink import reconcile_factor_artifact
    return reconcile_factor_artifact(
        root, run_id, ordinal, name, value, generation=generation,
        policy=policy, budget_bytes=budget_bytes,
    )


def _factory_compute_wave(factory: Any, config: Any, broker_proxy: Any,
                          factors: list[Any], run_kwargs: dict[str, Any],
                          transport_budget_bytes: int):
    from factor_engine.runtime.resource_broker import install_v2_resource_broker_proxy
    install_v2_resource_broker_proxy(broker_proxy)
    engine = factory(config)
    try:
        return _compute_wave(engine, factors, run_kwargs, transport_budget_bytes, broker_proxy)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def _factory_compile_factor(factory: Any, config: Any, broker_proxy: Any, factor: Any) -> bool:
    from factor_engine.runtime.resource_broker import install_v2_resource_broker_proxy
    install_v2_resource_broker_proxy(broker_proxy)
    engine = factory(config)
    try:
        return _compile_factor(engine, factor)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def execute_run_many_durable(
    engine: Any, factors: Iterable[Any], *, policy: Any, artifact_root: str | Path,
    run_kwargs: dict[str, Any] | None = None, manifest_path: str | Path | None = None,
    state_path: str | Path | None = None, sink: Any = None,
    execution_scope: dict[str, Any] | None = None,
    engine_factory: Any = None, engine_factory_config: Any = None,
) -> dict[str, Any]:
    """Execute finite input with ordinal terminals and one persisted retry budget.

    Numerical work always goes through ``engine.run_many_parallel``. This layer
    only bounds ingestion/residency, records outcomes, and owns durable delivery.
    """
    run_kwargs = dict(run_kwargs or {})
    from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy
    if not isinstance(policy, DefaultExecutionPolicy):
        raise TypeError("policy must be a validated DefaultExecutionPolicy")
    broker = run_kwargs.pop("broker", None) or getattr(engine, "resource_broker", None)
    if broker is None:
        try:
            from factor_engine.runtime.host_resource_coordinator import get_host_coordinator
            broker = get_host_coordinator().broker
        except Exception as exc:
            raise RuntimeError("v2 durable execution requires the shared ResourceBroker") from exc
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(manifest_path or run_dir / "manifest.sqlite3")
    state_path = Path(state_path or run_dir / "state.sqlite3")
    for database_path in (manifest_path, state_path):
        if database_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing durable state {database_path}; resume is not implemented"
            )
    max_attempts = min(3, int(_policy_value(policy, "work_item_max_attempts", 3)))
    lookahead = int(_policy_value(policy, "initial_lookahead_factors", 512))
    manifest = FiniteFactorManifest.ingest_supervised(
        factors, manifest_path,
        process_context="spawn" if engine_factory is not None else "fork",
        max_factors=int(_policy_value(policy, "max_manifest_factors", 1_000_000)),
        max_definition_bytes=int(_policy_value(policy, "max_definition_bytes", 131_072)),
        deadline_seconds=float(_policy_value(policy, "input_ingestion_deadline_seconds", 300)),
    )
    state = PersistentRunState(state_path, max_attempts=max_attempts)
    egress_leases: tuple[Any, Any] | None = None
    writer_bytes = 0
    queue_bytes = 0
    broker_ipc = compute_proxy = compile_proxy = None
    if engine_factory is not None:
        from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC
        broker_ipc = ParentBrokerIPC(broker, rpc_timeout_seconds=policy.connect_seconds)
        compute_proxy = broker_ipc.create_proxy()
        compile_proxy = broker_ipc.create_proxy()
    worker_context = "spawn" if engine_factory is not None else "fork"
    compute_callable = (_SpawnFactoryRuntime(engine_factory, engine_factory_config,
                                             compute_proxy, "compute")
                        if engine_factory is not None else partial(_compute_wave, engine))
    compile_callable = (_SpawnFactoryRuntime(engine_factory, engine_factory_config,
                                             compile_proxy, "compile")
                        if engine_factory is not None else partial(_compile_wave, engine))
    compute_worker = SupervisedReusableWorker(
        cancel_grace_seconds=float(_policy_value(policy, "cooperative_cancel_grace_seconds", 5)),
        exit_observation_seconds=float(_policy_value(policy, "worker_exit_observation_seconds", 10)),
        context=worker_context,
        function=compute_callable,
    )
    compile_worker = SupervisedReusableWorker(
        cancel_grace_seconds=float(policy.cooperative_cancel_grace_seconds),
        exit_observation_seconds=float(policy.worker_exit_observation_seconds),
        context=worker_context, function=compile_callable,
    )
    writer_callable = (partial(_custom_worker_write_verify, sink, root, run_id, policy)
                       if sink is not None else partial(_default_worker_write, root, run_id, policy))
    sink_worker = SupervisedReusableWorker(
        cancel_grace_seconds=float(_policy_value(policy, "cooperative_cancel_grace_seconds", 5)),
        exit_observation_seconds=float(_policy_value(policy, "worker_exit_observation_seconds", 10)),
        context="spawn",
        function=writer_callable,
    )
    reconcile_worker = None if sink is not None else SupervisedReusableWorker(
        context="spawn",
        function=partial(_default_worker_reconcile, root, run_id, policy),
        cancel_grace_seconds=float(policy.cooperative_cancel_grace_seconds),
        exit_observation_seconds=float(policy.worker_exit_observation_seconds),
    )
    error_groups: dict[str, list[dict[str, Any]]] = {}
    execution_batches = 0
    active_primary: BaseException | None = None
    try:
        for record in manifest.records(start=0, limit=len(manifest) or 1):
            state.register(record.ordinal, record.name)
        if broker_ipc is not None:
            compile_worker.start()
            broker_ipc.bind_client_process(compile_proxy.client_id, compile_worker._process.pid)
            compute_worker.start()
            broker_ipc.bind_client_process(compute_proxy.client_id, compute_worker._process.pid)
        cursor = 0
        while cursor < len(manifest):
            try:
                metadata_bytes = max(1, min(int(broker.current_read_budget()), 64 * 1024 * 1024))
            except Exception:
                metadata_bytes = max(1, policy.max_definition_bytes)
            wave = manifest.load_wave_bounded(
                start=cursor, max_items=max(1, lookahead),
                max_definition_bytes=metadata_bytes,
            )
            if not wave:
                break
            admitted = []
            for ordinal, factor, manifest_error, manifest_name in wave:
                factor, scope_error = _apply_execution_scope(factor, execution_scope)
                rejection = manifest_error or scope_error
                if rejection is not None:
                    state.terminal(ordinal, "REJECTED", error_code=rejection,
                                   detail=f"manifest/scope rejected ordinal {ordinal}", retryable=False)
                    group = error_groups.setdefault(rejection, {"count": 0, "examples": []})
                    group["count"] += 1
                    if len(group["examples"]) < int(_policy_value(policy, "max_examples_per_error_group", 3)):
                        group["examples"].append({"ordinal": ordinal, "name": manifest_name})
                    continue
                admitted.append((ordinal, factor))
            if admitted:
                try:
                    compile_results = compile_worker.execute(
                        [factor for _, factor in admitted],
                        timeout_seconds=float(policy.compute_unknown_seconds),
                    ).value
                except Exception as exc:
                    from factor_engine.runtime.supervised_worker import (
                        WorkerQuarantined, WorkerStaleResponse,
                        WorkerTimedOut, WorkerTransportFailed,
                    )
                    if isinstance(
                        exc, (WorkerQuarantined, WorkerStaleResponse,
                              WorkerTimedOut, WorkerTransportFailed)
                    ):
                        raise
                    compile_results = {factor.name: f"{type(exc).__name__}: {exc}"
                                       for _, factor in admitted}
                validated = []
                for ordinal, factor in admitted:
                    compile_error = compile_results.get(factor.name)
                    if compile_error is None:
                        validated.append((ordinal, factor))
                        continue
                    code = "INVALID_FACTOR_COMPILE"
                    state.terminal(ordinal, "REJECTED", error_code=code,
                                   detail=compile_error, retryable=False)
                    group = error_groups.setdefault(code, {"count": 0, "examples": []})
                    group["count"] += 1
                    if len(group["examples"]) < policy.max_examples_per_error_group:
                        group["examples"].append({"ordinal": ordinal, "name": factor.name})
                admitted = validated
            if admitted and egress_leases is None:
                    queue_bytes = int(broker.automatic_result_queue_budget())
                    writer_bytes = min(queue_bytes, int(_policy_value(
                        policy, "target_file_bytes", 134_217_728
                    )))
                    egress_leases = broker.acquire_protected_egress(
                        queue_bytes, writer_bytes, lease_id=f"v2-egress:{run_id}"
                    )
                    if egress_leases is None:
                        code = "RESOURCE_WAIT_EXHAUSTED"
                        for ordinal, factor in admitted:
                            state.terminal(ordinal, "FAILED", error_code=code,
                                           detail="protected result egress unavailable", retryable=False)
                        admitted = []
            if admitted:
                execution_batches += 1
                sink_started: set[int] = set()
                for ordinal, _factor in admitted:
                    state.consume_attempt(ordinal, "execution")
                try:
                    wave_decoded = False
                    results = compute_worker.execute(
                        [f for _, f in admitted], run_kwargs, queue_bytes,
                        timeout_seconds=float(_policy_value(policy, "compute_unknown_seconds", 900)),
                    ).value
                    envelope = pickle.loads(results)
                    result_blobs = envelope.pop("result_blobs")
                    transport_errors = envelope.pop("transport_errors")
                    factor_errors = envelope.pop("factor_errors", {})
                    del envelope, results
                    expected = {f.name for _, f in admitted}
                    if set(result_blobs) | set(transport_errors) | set(factor_errors) != expected:
                        raise RuntimeError("bounded wave returned an incomplete result set")
                    wave_decoded = True
                except Exception as exc:
                    from factor_engine.runtime.supervised_worker import (
                        WorkerQuarantined, WorkerStaleResponse,
                        WorkerTimedOut, WorkerTransportFailed,
                    )
                    if isinstance(
                        exc, (WorkerQuarantined, WorkerStaleResponse,
                              WorkerTimedOut, WorkerTransportFailed)
                    ):
                        raise
                    kind = classify_error(exc)
                    code = f"WAVE_{kind.upper()}"
                    group = error_groups.setdefault(code, {"count": 0, "examples": []})
                    for ordinal, factor in admitted:
                        state.terminal(ordinal, "FAILED", error_code=code,
                                       detail=f"{type(exc).__name__}: {exc}", retryable=False)
                        group["count"] += 1
                        if len(group["examples"]) < policy.max_examples_per_error_group:
                            group["examples"].append({"ordinal": ordinal, "name": factor.name})
                if wave_decoded:
                    for ordinal, factor in admitted:
                        result = None
                        try:
                            delivered = None
                            if factor.name in factor_errors:
                                problem = factor_errors[factor.name]
                                code = str(problem.get("code") or "INVALID_PHYSICAL_FACTOR")
                                state.terminal(
                                    ordinal, "REJECTED", error_code=code,
                                    detail=(f"{problem.get('error_type', 'Error')}: "
                                            f"{problem.get('message', '')}"),
                                    retryable=False,
                                )
                                group = error_groups.setdefault(
                                    code, {"count": 0, "examples": []}
                                )
                                group["count"] += 1
                                if len(group["examples"]) < policy.max_examples_per_error_group:
                                    group["examples"].append(
                                        {"ordinal": ordinal, "name": factor.name}
                                    )
                                continue
                            if factor.name in transport_errors:
                                problem = transport_errors[factor.name]
                                raise ResultTransportBudgetExceeded(
                                    f"result requires {problem['required_bytes']} bytes; "
                                    f"available {problem['available_bytes']}"
                                )
                            blob = result_blobs.pop(factor.name)
                            result = pickle.loads(blob)
                            del blob
                            if sink is None:
                                from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
                                from factor_engine.runtime.durable_artifact_sink import required_writer_workspace_bytes
                                required = required_writer_workspace_bytes(result)
                                extra_lease = None
                                available = writer_bytes
                                if required > available:
                                    extra_lease = broker.acquire_memory(
                                        MemoryLeaseKind.WRITER_BATCH, required - available,
                                        lease_id=f"v2-writer-extra:{run_id}:{ordinal}",
                                    )
                                    if extra_lease is None:
                                        raise MemoryError(
                                            f"writer workspace requires {required} bytes; admitted {available}"
                                        )
                                    available = required
                                sink_started.add(ordinal)
                                artifact_generation = uuid.uuid4().hex
                                state.record_commit_intent(ordinal, artifact_generation)
                                delivered = sink_worker.execute(
                                    ordinal, factor.name, result, available, artifact_generation,
                                    timeout_seconds=float(policy.sink_flush_seconds),
                                    lease=extra_lease,
                                ).value
                                if not delivered.get("committed") or not delivered.get("verified"):
                                    raise RuntimeError("artifact lacks committed verified receipt")
                            else:
                                from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
                                from factor_engine.runtime.durable_artifact_sink import required_writer_workspace_bytes
                                required = required_writer_workspace_bytes(result)
                                extra_lease = None
                                available = writer_bytes
                                if required > available:
                                    extra_lease = broker.acquire_memory(
                                        MemoryLeaseKind.WRITER_BATCH, required - available,
                                        lease_id=f"v2-custom-verify:{run_id}:{ordinal}",
                                    )
                                    if extra_lease is None:
                                        raise MemoryError(
                                            f"receipt verification requires {required} bytes; admitted {available}"
                                        )
                                    available = required
                                sink_started.add(ordinal)
                                delivered = sink_worker.execute(
                                    ordinal, factor.name, result, available,
                                    timeout_seconds=float(policy.sink_flush_seconds),
                                    lease=extra_lease,
                                ).value
                            state.terminal(ordinal, "SUCCEEDED", artifact=delivered,
                                           commit_state="VERIFIED")
                        except Exception as exc:
                            from factor_engine.runtime.supervised_worker import (
                                WorkerQuarantined, WorkerStaleResponse,
                                WorkerTimedOut, WorkerTransportFailed,
                            )
                            if isinstance(exc, WorkerQuarantined):
                                raise
                            if isinstance(
                                exc, (WorkerTimedOut, WorkerTransportFailed, WorkerStaleResponse)
                            ) and sink is None:
                                intent = state.get_commit_intent(ordinal)
                                if intent is not None:
                                    try:
                                        reconcile_lease = None
                                        if required > writer_bytes:
                                            reconcile_lease = broker.acquire_memory(
                                                MemoryLeaseKind.WRITER_BATCH,
                                                required - writer_bytes,
                                                lease_id=f"v2-reconcile:{run_id}:{ordinal}",
                                            )
                                            if reconcile_lease is None:
                                                raise MemoryError("reconcile workspace unavailable")
                                        delivered = reconcile_worker.execute(
                                            ordinal, factor.name, result,
                                            max(required, writer_bytes), intent["generation"],
                                            timeout_seconds=float(policy.sink_flush_seconds),
                                            lease=reconcile_lease,
                                        ).value
                                        state.terminal(
                                            ordinal, "SUCCEEDED", artifact=delivered,
                                            commit_state="VERIFIED",
                                        )
                                        continue
                                    except WorkerQuarantined:
                                        raise
                                    except Exception as reconcile_exc:
                                        exc = reconcile_exc
                            if ordinal in sink_started:
                                item_code = "UNKNOWN_COMMIT"
                            elif isinstance(exc, ResultTransportBudgetExceeded):
                                item_code = "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
                            else:
                                item_code = "OUTPUT_INVALID"
                            commit_state = "UNKNOWN" if ordinal in sink_started else "NOT_STARTED"
                            state.terminal(ordinal, "FAILED", error_code=item_code,
                                           detail=f"{type(exc).__name__}: {exc}", retryable=False,
                                           commit_state=commit_state)
                            group = error_groups.setdefault(item_code, {"count": 0, "examples": []})
                            group["count"] += 1
                            if len(group["examples"]) < policy.max_examples_per_error_group:
                                group["examples"].append({"ordinal": ordinal, "name": factor.name})
                        finally:
                            result = None
            cursor += len(wave)
            del wave
        counts = state.counts()
        ordinal_terminals = ("SUCCEEDED", "REUSED", "REJECTED", "FAILED", "BLOCKED", "CANCELLED")
        terminal_count = sum(counts.get(name, 0) for name in ordinal_terminals)
        all_terminal = terminal_count == len(manifest)
        all_requested_terminal = bool(manifest.input_complete and all_terminal)
        all_outputs_valid = bool(
            manifest.input_complete and all_terminal
            and counts.get("SUCCEEDED", 0) + counts.get("REUSED", 0) == len(manifest)
        )
        status = "SUCCEEDED" if all_outputs_valid else "COMPLETED_WITH_ERRORS"
        outcome_limit = min(512, int(policy.initial_lookahead_factors))
        outcome_page = state.outcomes_page(start=0, limit=outcome_limit)
        outcomes = [asdict(outcome) for outcome in outcome_page]
        outcomes_truncated = len(manifest) > len(outcomes)
        receipt = {
            "schema_version": "factor_engine.artifact_receipt.v2",
            "run_id": run_id, "status": status,
            "policy_id": _policy_value(policy, "policy_id", None),
            "policy_digest": getattr(policy, "digest", None),
            "requested_factors": len(manifest), "requested_total": len(manifest),
            "counts": counts, "outcomes": outcomes,
            "outcomes_truncated": outcomes_truncated,
            "outcomes_next_ordinal": len(outcomes) if outcomes_truncated else None,
            "input_complete": manifest.input_complete,
            "input_error": manifest.input_error,
            "all_terminal": all_terminal,
            "all_requested_terminal": all_requested_terminal,
            "all_outputs_valid": all_outputs_valid,
            "terminal_count": terminal_count,
            "execution_batches": execution_batches,
            "manifest_path": str(manifest_path), "state_path": str(state_path),
            "result_index": str(state_path),
            "error_groups": [
                {"code": code, "count": group["count"], "examples": group["examples"]}
                for code, group in list(error_groups.items())[
                    :int(_policy_value(policy, "max_error_groups_in_response", 20))
                ]
            ],
            "automatic_production_publish": False,
        }
        receipt_path = run_dir / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        receipt["receipt_path"] = str(receipt_path)
        return receipt
    except BaseException as exc:
        active_primary = exc
        try:
            start = 0
            while start < len(manifest):
                page = state.outcomes_page(
                    start=start, limit=min(1000, max(1, len(manifest) - start))
                )
                if not page:
                    break
                for outcome in page:
                    if outcome.state in {"SUCCEEDED", "REUSED", "REJECTED", "FAILED", "BLOCKED", "CANCELLED"}:
                        continue
                    intent = state.get_commit_intent(outcome.ordinal)
                    state.terminal(
                        outcome.ordinal, "CANCELLED", error_code="RUN_ABORTED",
                        detail=f"{type(exc).__name__}: {exc}", retryable=False,
                        commit_state="UNKNOWN" if intent is not None else "NOT_STARTED",
                    )
                start = page[-1].ordinal + 1
            aborted_receipt = {
                "schema_version": "factor_engine.artifact_receipt.v2",
                "run_id": run_id, "status": "ABORTED", "counts": state.counts(),
                "requested_factors": len(manifest), "requested_total": len(manifest),
                "input_complete": manifest.input_complete,
                "manifest_path": str(manifest_path), "state_path": str(state_path),
                "automatic_production_publish": False,
                "primary_error": f"{type(exc).__name__}: {exc}",
            }
            receipt_path = run_dir / "receipt.json"
            receipt_path.write_text(
                json.dumps(aborted_receipt, indent=2, sort_keys=True), encoding="utf-8"
            )
            exc.receipt_path = str(receipt_path)
        except BaseException as abort_exc:
            exc.abort_receipt_error = abort_exc
        raise
    finally:
        cleanup_errors = []
        for worker, proxy in (
            (compute_worker, compute_proxy), (compile_worker, compile_proxy),
            (sink_worker, None),
            (reconcile_worker, None),
        ):
            if worker is None:
                continue
            try:
                worker.close()
                if (broker_ipc is not None and proxy is not None
                        and worker._process is not None):
                    broker_ipc.reclaim_client(proxy.client_id)
            except BaseException as exc:
                cleanup_errors.append(exc)
        # Never release protected egress while any worker may still own result
        # memory or writer state. Attach authorities to the cleanup exception so
        # the facade/supervisor can retain them through PID cleanup.
        if not cleanup_errors:
            if egress_leases is not None:
                for lease in egress_leases:
                    try:
                        lease.release()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
            if broker_ipc is not None:
                try:
                    broker_ipc.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        for authority in (state, manifest):
            try:
                authority.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if active_primary is not None:
                active_primary.cleanup_errors = cleanup_errors
                active_primary.broker_ipc = broker_ipc
                active_primary.egress_leases = egress_leases
            else:
                failure = cleanup_errors[0]
                failure.broker_ipc = broker_ipc
                failure.egress_leases = egress_leases
                raise failure
