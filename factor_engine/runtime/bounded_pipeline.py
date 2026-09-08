"""Finite v2 orchestration over the existing FactorEngine execution path."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
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


class WorkerProtocolError(RuntimeError):
    reason_code = "WORKER_PROTOCOL_INTEGRITY"


@dataclass
class _DirectWaveContext:
    """Parent-owned state carried with one direct-artifact worker request."""

    wave: list[tuple[int, Any, str | None, str]]
    admitted: list[tuple[int, Any]]
    assignments: dict[str, tuple[int, str]]
    cursor: int
    manifest_length: int
    queue_bytes: int
    writer_bytes: int
    sink_started: set[int] = field(default_factory=set)

    @property
    def expected_names(self) -> set[str]:
        return {factor.name for _, factor in self.admitted}

    @property
    def has_later_wave(self) -> bool:
        return self.cursor + len(self.wave) < self.manifest_length


def _validate_compile_results(results, expected):
    if (type(results) is not dict or set(results) != set(expected)
            or any(value is not None and (type(value) is not str or not value)
                   for value in results.values())):
        raise WorkerProtocolError("compile protocol requires exact factor keys and typed outcomes")
    return results


def _validate_result_envelope(envelope, expected):
    keys = {"result_blobs", "transport_errors", "factor_errors"}
    if type(envelope) is not dict or set(envelope) != keys:
        raise WorkerProtocolError("result protocol has invalid envelope fields")
    groups = [envelope[name] for name in ("result_blobs", "transport_errors", "factor_errors")]
    if any(type(group) is not dict for group in groups):
        raise WorkerProtocolError("result protocol outcomes must be mappings")
    seen = set()
    for group in groups:
        if seen.intersection(group):
            raise WorkerProtocolError("result protocol has overlapping factor outcomes")
        seen.update(group)
    if seen != set(expected):
        raise WorkerProtocolError("result protocol requires exact factor coverage")
    if any(type(blob) is not bytes for blob in groups[0].values()):
        raise WorkerProtocolError("result protocol requires serialized bytes")
    for group in groups[1:]:
        if any(type(error) is not dict or type(error.get("code")) is not str
               or not error["code"] for error in group.values()):
            raise WorkerProtocolError("result protocol requires typed error records")
    return groups


def _validate_artifact_envelope(envelope, expected):
    keys = {"receipts", "artifact_errors", "factor_errors"}
    if type(envelope) is not dict or set(envelope) != keys:
        raise WorkerProtocolError("artifact protocol has invalid envelope fields")
    groups = [envelope[name] for name in ("receipts", "artifact_errors", "factor_errors")]
    if any(type(group) is not dict for group in groups):
        raise WorkerProtocolError("artifact protocol outcomes must be mappings")
    seen = set()
    for group in groups:
        if seen.intersection(group):
            raise WorkerProtocolError("artifact protocol has overlapping factor outcomes")
        seen.update(group)
    if seen != set(expected):
        raise WorkerProtocolError("artifact protocol requires exact factor coverage")
    for name, receipt in groups[0].items():
        if (type(receipt) is not dict or receipt.get("committed") is not True
                or receipt.get("verified") is not True
                or type(receipt.get("generation")) is not str):
            raise WorkerProtocolError(f"artifact receipt is not verified for {name}")
    for group in groups[1:]:
        if any(type(error) is not dict or type(error.get("code")) is not str
               or not error["code"] for error in group.values()):
            raise WorkerProtocolError("artifact protocol requires typed error records")
    return groups


class _DirectVerifiedArtifactSink:
    """Synchronous per-root sink owned by the supervised compute process."""

    def __init__(self, *, root, run_id, policy, assignments, writer_bytes, broker):
        self.root = Path(root)
        self.run_id = run_id
        self.policy = policy
        self.assignments = dict(assignments)
        self.writer_bytes = int(writer_bytes)
        self.broker = broker
        self.receipts = {}
        self.errors = {}

    def __call__(self, name, value):
        if name not in self.assignments or name in self.receipts or name in self.errors:
            raise WorkerProtocolError(f"unexpected or duplicate direct sink result {name!r}")
        ordinal, generation = self.assignments[name]
        extra_lease = None
        started = False
        try:
            from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
            from factor_engine.runtime.durable_artifact_sink import (
                required_writer_workspace_bytes, write_verified_factor_artifact,
            )
            required = required_writer_workspace_bytes(value)
            available = self.writer_bytes
            if required > available:
                extra_lease = self.broker.acquire_memory(
                    MemoryLeaseKind.WRITER_BATCH, required - available,
                    lease_id=f"v2-direct-writer:{self.run_id}:{ordinal}",
                )
                if extra_lease is None:
                    self.errors[name] = {
                        "code": "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET",
                        "started": False,
                        "required_bytes": required,
                        "available_bytes": available,
                    }
                    return True
                available = required
            started = True
            receipt = write_verified_factor_artifact(
                self.root, self.run_id, ordinal, name, value,
                policy=self.policy, budget_bytes=available, generation=generation,
            )
            if (receipt.get("committed") is not True or receipt.get("verified") is not True
                    or receipt.get("generation") != generation):
                raise WorkerProtocolError("direct sink returned an invalid receipt")
            self.receipts[name] = receipt
        except WorkerProtocolError:
            # Protocol integrity is wave-level authority failure, not a
            # per-factor OUTPUT_INVALID outcome that permits worker reuse.
            raise
        except Exception as exc:
            self.errors[name] = {
                "code": "UNKNOWN_COMMIT" if started else "OUTPUT_INVALID",
                "started": started,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            if extra_lease is not None:
                extra_lease.release()
        return True


def _compute_wave_to_artifacts(engine, factors, run_kwargs, artifact_plan, broker_proxy=None):
    """Compute roots into verified artifacts; never return factor values."""
    from factor_engine.runtime.adaptive_batch_scheduler import disable_inner_retries_for_v2
    broker = broker_proxy or getattr(engine, "resource_broker", None)
    if broker is None:
        raise RuntimeError("direct artifact compute requires the parent broker authority")
    if broker_proxy is not None:
        engine.resource_broker = broker_proxy
        from factor_engine.runtime.resource_broker import install_v2_resource_broker_proxy
        install_v2_resource_broker_proxy(broker_proxy)
    assignments = artifact_plan["assignments"]
    expected = {factor.name for factor in factors}
    if set(assignments) != expected:
        raise WorkerProtocolError("direct artifact assignments require exact factor coverage")
    target = _DirectVerifiedArtifactSink(
        root=artifact_plan["root"], run_id=artifact_plan["run_id"],
        policy=artifact_plan["policy"], assignments=assignments,
        writer_bytes=artifact_plan["writer_bytes"], broker=broker,
    )
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
                engine, factors, result_policy="sink", sink=target,
                isolate_physical_errors=True, **run_kwargs,
            )
        else:
            output = engine.run_many_parallel(
                factors, result_policy="sink", sink=target, **run_kwargs
            )
    factor_errors = dict(output.pop("physical_preflight_errors", {}) or {})
    returned = output.pop("results", {})
    if returned:
        raise WorkerProtocolError("direct artifact compute retained returned results")
    del output
    envelope = {
        "receipts": target.receipts,
        "artifact_errors": target.errors,
        "factor_errors": factor_errors,
    }
    _validate_artifact_envelope(envelope, expected)
    return envelope


def _write_control_receipt(path, receipt):
    """Atomic, synced control view; never use a missing view to rewrite values."""
    path = Path(path)
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    fd, staging = tempfile.mkstemp(prefix=".receipt-", suffix=".tmp", dir=path.parent)
    # Retain a failed staging file for diagnosis. Its random name cannot be
    # confused with the authoritative receipt.json or overwrite an artifact.
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    for parent in (path.parent, path.parent.parent):
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _retire_direct_compute_authority(worker, broker_ipc, proxy):
    """Prove compute exit, then reclaim only its bound broker client."""
    worker.close()
    broker_ipc.reclaim_client(proxy.client_id)


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
                 transport_budget: int | None = None, artifact_plan: dict[str, Any] | None = None):
        engine = self._get_engine()
        if self.mode == "compile":
            return _compile_wave(engine, factors)
        if artifact_plan is not None:
            return _compute_wave_to_artifacts(
                engine, factors, run_kwargs or {}, artifact_plan, self.broker_proxy
            )
        return _compute_wave(engine, factors, run_kwargs or {}, int(transport_budget),
                             self.broker_proxy)

    def close(self) -> None:
        if self._engine is not None:
            _close_execution_core(self._engine)
            self._engine = None


class _SpawnFactoryArtifactReconciler:
    """Recompute one factor, then exactly verify one known artifact generation."""

    def __init__(self, factory, config, broker_proxy):
        self.factory, self.config, self.broker_proxy = factory, config, broker_proxy
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
            self._engine.resource_broker = self.broker_proxy
        return self._engine

    def __call__(self, factor, run_kwargs, root, run_id, policy,
                 ordinal, name, budget_bytes, generation):
        from factor_engine.runtime.adaptive_batch_scheduler import disable_inner_retries_for_v2
        from factor_engine.runtime.durable_artifact_sink import (
            reconcile_factor_artifact, required_writer_workspace_bytes,
        )
        engine = self._get_engine()
        with disable_inner_retries_for_v2():
            output = engine.run_many_parallel(
                [factor], result_policy="return", **dict(run_kwargs or {})
            )
        results = output.pop("results", {})
        if set(results) != {name}:
            raise WorkerProtocolError("recompute reconciliation requires one exact result")
        value = results.pop(name)
        required = required_writer_workspace_bytes(value)
        extra_lease = None
        available = int(budget_bytes)
        try:
            if required > available:
                from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
                extra_lease = self.broker_proxy.acquire_memory(
                    MemoryLeaseKind.WRITER_BATCH, required - available,
                    lease_id=f"v2-direct-reconcile:{run_id}:{ordinal}",
                )
                if extra_lease is None:
                    raise ResultTransportBudgetExceeded(
                        f"reconcile requires {required} bytes; admitted {available}"
                    )
                available = required
            return reconcile_factor_artifact(
                root, run_id, ordinal, name, value, generation=generation,
                policy=policy, budget_bytes=available,
            )
        finally:
            if extra_lease is not None:
                extra_lease.release()

    def close(self):
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
    broker_ipc = compute_proxy = compile_proxy = direct_reconcile_proxy = None
    if engine_factory is not None:
        from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC
        broker_ipc = ParentBrokerIPC(broker, rpc_timeout_seconds=policy.connect_seconds)
        compute_proxy = broker_ipc.create_proxy()
        compile_proxy = broker_ipc.create_proxy()
        direct_reconcile_proxy = broker_ipc.create_proxy() if sink is None else None
    worker_context = "spawn" if engine_factory is not None else "fork"
    direct_artifacts = engine_factory is not None and sink is None
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
    direct_reconcile_worker = (
        SupervisedReusableWorker(
            context="spawn",
            function=_SpawnFactoryArtifactReconciler(
                engine_factory, engine_factory_config, direct_reconcile_proxy
            ),
            cancel_grace_seconds=float(policy.cooperative_cancel_grace_seconds),
            exit_observation_seconds=float(policy.worker_exit_observation_seconds),
        )
        if direct_artifacts else None
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
            if direct_reconcile_worker is not None:
                direct_reconcile_worker.start()
                broker_ipc.bind_client_process(
                    direct_reconcile_proxy.client_id, direct_reconcile_worker._process.pid
                )
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
                _validate_compile_results(compile_results, {factor.name for _, factor in admitted})
                validated = []
                for ordinal, factor in admitted:
                    compile_error = compile_results[factor.name]
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
                artifact_assignments = {}
                for ordinal, _factor in admitted:
                    state.consume_attempt(ordinal, "execution")
                    if direct_artifacts:
                        generation = uuid.uuid4().hex
                        state.record_commit_intent(ordinal, generation)
                        artifact_assignments[_factor.name] = (ordinal, generation)
                direct_context = (
                    _DirectWaveContext(
                        wave=wave, admitted=admitted,
                        assignments=artifact_assignments, cursor=cursor,
                        manifest_length=len(manifest), queue_bytes=queue_bytes,
                        writer_bytes=writer_bytes, sink_started=sink_started,
                    )
                    if direct_artifacts else None
                )
                try:
                    wave_decoded = False
                    lost_ack_reconciled = False
                    if direct_artifacts:
                        envelope = compute_worker.execute(
                            [f for _, f in admitted], run_kwargs, queue_bytes,
                            {
                                "root": root, "run_id": run_id, "policy": policy,
                                "assignments": direct_context.assignments,
                                "writer_bytes": writer_bytes,
                            },
                            timeout_seconds=float(_policy_value(
                                policy, "compute_unknown_seconds", 900
                            )),
                        ).value
                        receipts, artifact_errors, factor_errors = _validate_artifact_envelope(
                            envelope, direct_context.expected_names
                        )
                        result_blobs, transport_errors = {}, {}
                        del envelope
                    else:
                        results = compute_worker.execute(
                            [f for _, f in admitted], run_kwargs, queue_bytes,
                            timeout_seconds=float(_policy_value(
                                policy, "compute_unknown_seconds", 900
                            )),
                        ).value
                        envelope = pickle.loads(results)
                        result_blobs, transport_errors, factor_errors = _validate_result_envelope(
                            envelope, {f.name for _, f in admitted})
                        del envelope, results
                    wave_decoded = True
                except Exception as exc:
                    from factor_engine.runtime.supervised_worker import (
                        WorkerQuarantined, WorkerStaleResponse,
                        WorkerTimedOut, WorkerTransportFailed,
                    )
                    if isinstance(
                        exc, (WorkerQuarantined, WorkerStaleResponse,
                              WorkerTimedOut, WorkerTransportFailed, WorkerProtocolError)
                    ):
                        if not direct_artifacts or isinstance(exc, WorkerQuarantined):
                            raise
                        # Descriptor/protocol authority is gone.  Even a
                        # protocol-invalid response can come from a live,
                        # reusable process, unlike transport/stale/timeout
                        # failures whose supervisor already retires it.  Prove
                        # exit and reclaim that exact broker client before any
                        # recomputation reads its committed generation.
                        _retire_direct_compute_authority(
                            compute_worker, broker_ipc, compute_proxy
                        )
                        compute_worker = None
                        compute_proxy = None
                        # The compute process may have committed artifacts before
                        # losing its small descriptor ACK.  A manifest is not an
                        # oracle: consume one persisted attempt per ordinal,
                        # recompute under the same factory/snapshot, and exact-
                        # reconcile only the already-bound generation.
                        artifact_manifests = {
                            ordinal: (
                                root / run_id / "values"
                                / f"{ordinal:08d}-{direct_context.assignments[factor.name][1]}"
                                / "manifest.json"
                            )
                            for ordinal, factor in admitted
                        }
                        if (direct_context.has_later_wave
                                and not any(path.is_file() for path in artifact_manifests.values())):
                            # Preserve the established generation-fencing rule:
                            # a dead compute generation with no committed view
                            # aborts this and all later waves as CANCELLED.
                            raise
                        for ordinal, factor in admitted:
                            intent = state.get_commit_intent(ordinal)
                            if not artifact_manifests[ordinal].is_file():
                                state.terminal(
                                    ordinal, "FAILED", error_code="UNKNOWN_COMMIT",
                                    detail="descriptor ACK lost before a committed manifest was observable",
                                    retryable=False, commit_state="UNKNOWN",
                                )
                                continue
                            attempt = state.consume_attempt(ordinal, "recompute_reconcile")
                            if intent is None or attempt is None:
                                state.terminal(
                                    ordinal, "FAILED", error_code="UNKNOWN_COMMIT",
                                    detail="lost descriptor ACK and no reconciliation attempt remains",
                                    retryable=False, commit_state="UNKNOWN",
                                )
                                continue
                            try:
                                delivered = direct_reconcile_worker.execute(
                                    factor, run_kwargs, root, run_id, policy,
                                    ordinal, factor.name, writer_bytes, intent["generation"],
                                    timeout_seconds=float(policy.compute_unknown_seconds),
                                ).value
                                state.terminal(
                                    ordinal, "SUCCEEDED", artifact=delivered,
                                    commit_state="VERIFIED",
                                )
                            except WorkerQuarantined:
                                # Cleanup is incomplete and its broker leases
                                # remain owned.  This is a run-level quarantine,
                                # never a normal UNKNOWN_COMMIT factor result.
                                raise
                            except Exception as reconcile_exc:
                                state.terminal(
                                    ordinal, "FAILED", error_code="UNKNOWN_COMMIT",
                                    detail=(
                                        f"{type(exc).__name__}: {exc}; reconcile: "
                                        f"{type(reconcile_exc).__name__}: {reconcile_exc}"
                                    ),
                                    retryable=False, commit_state="UNKNOWN",
                                )
                        if direct_context.has_later_wave:
                            raise WorkerTransportFailed(
                                "compute generation retired after descriptor loss; "
                                "later waves cannot reuse its parent-broker identity"
                            ) from exc
                        wave_decoded = False
                        lost_ack_reconciled = True
                    if not lost_ack_reconciled:
                        kind = classify_error(exc)
                        code = f"WAVE_{kind.upper()}"
                        group = error_groups.setdefault(code, {"count": 0, "examples": []})
                        for ordinal, factor in admitted:
                            state.terminal(ordinal, "FAILED", error_code=code,
                                           detail=f"{type(exc).__name__}: {exc}", retryable=False)
                            group["count"] += 1
                            if len(group["examples"]) < policy.max_examples_per_error_group:
                                group["examples"].append(
                                    {"ordinal": ordinal, "name": factor.name}
                                )
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
                            if direct_artifacts:
                                if factor.name in artifact_errors:
                                    problem = artifact_errors[factor.name]
                                    if problem.get("started"):
                                        sink_started.add(ordinal)
                                    if problem["code"] == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET":
                                        raise ResultTransportBudgetExceeded(
                                            f"result requires {problem.get('required_bytes')} bytes; "
                                            f"available {problem.get('available_bytes')}"
                                        )
                                    raise RuntimeError(
                                        f"{problem.get('error_type', 'ArtifactError')}: "
                                        f"{problem.get('message', '')}"
                                    )
                                delivered = receipts.pop(factor.name)
                                if delivered.get("generation") != direct_context.assignments[factor.name][1]:
                                    raise WorkerProtocolError(
                                        "direct artifact generation differs from commit intent"
                                    )
                                state.terminal(
                                    ordinal, "SUCCEEDED", artifact=delivered,
                                    commit_state="VERIFIED",
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
                if (direct_artifacts and wave_decoded
                        and direct_context.has_later_wave):
                    # A completed native read wave may retain allocator-owned
                    # buffers or broker tokens beyond Python object lifetime.
                    # Observe PID exit first, then reclaim that exact client;
                    # a later wave always gets a fresh proxy and generation.
                    compute_worker.close()
                    broker_ipc.reclaim_client(compute_proxy.client_id)
                    compute_proxy = broker_ipc.create_proxy()
                    compute_callable = _SpawnFactoryRuntime(
                        engine_factory, engine_factory_config, compute_proxy, "compute"
                    )
                    compute_worker = SupervisedReusableWorker(
                        cancel_grace_seconds=float(
                            _policy_value(policy, "cooperative_cancel_grace_seconds", 5)
                        ),
                        exit_observation_seconds=float(
                            _policy_value(policy, "worker_exit_observation_seconds", 10)
                        ),
                        context="spawn", function=compute_callable,
                    )
                    compute_worker.start()
                    broker_ipc.bind_client_process(
                        compute_proxy.client_id, compute_worker._process.pid
                    )
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
        _write_control_receipt(receipt_path, receipt)
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
            _write_control_receipt(receipt_path, aborted_receipt)
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
            (direct_reconcile_worker, direct_reconcile_proxy),
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
