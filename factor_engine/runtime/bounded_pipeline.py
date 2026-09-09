"""Finite v2 orchestration over the existing FactorEngine execution path."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from factor_engine.runtime.adaptive_batch_scheduler import (
    ERROR_OOM, ERROR_TRANSIENT, classify_error,
)
from factor_engine.runtime.finite_manifest import FiniteFactorManifest, ManifestInputTimeout
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.runtime.supervised_worker import SupervisedReusableWorker


class ResultTransportBudgetExceeded(MemoryError):
    pass


class WorkerProtocolError(RuntimeError):
    reason_code = "WORKER_PROTOCOL_INTEGRITY"


class JobDeadlineExceeded(TimeoutError):
    reason_code = "JOB_DEADLINE_EXCEEDED"


class ResourceWaitExhausted(TimeoutError):
    reason_code = "RESOURCE_WAIT_EXHAUSTED"


_OWNERSHIP_STORE = "sqlite-v1"
_OWNERSHIP_COVERAGE = "all-durable-pipeline-os-processes-v1"


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


@dataclass
class _DirectSlot:
    """All parent-owned authority and progress for one admitted direct wave."""

    cursor: int
    wave: list[tuple[int, Any, str | None, str]]
    admitted: list[tuple[int, Any]]
    names: list[str]
    assignments: dict[str, tuple[int, str]]
    worker: Any = None
    proxy: Any = None
    handle: Any = None
    phase: str = "ADMITTED"
    evidence_id: str = ""

    # Transitional mapping access keeps the validation/state code mechanical
    # while slot scheduling is separated from manifest completion order.
    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def get(self, name, default=None):
        return getattr(self, name, default)


def _complete_direct_slot(slot, envelope, *, run_id, policy, state, error_groups):
    """Validate and durably terminalize one completed direct slot."""
    receipts, artifact_errors, factor_errors, evidence = _validate_artifact_envelope(
        envelope, slot.assignments, run_id=run_id, policy_digest=policy.digest,
        expected_evidence_id=slot.evidence_id, include_evidence=True,
    )
    state.record_fit_failure_evidence(
        slot.evidence_id,
        ({"ordinal": ordinal, "name": factor.name}
         for ordinal, factor in slot.admitted),
        None if evidence is None else evidence["snapshot"],
        availability="UNAVAILABLE" if evidence is None else "OBSERVED",
    )
    slot.phase = "EVIDENCE_PERSISTED"
    for ordinal, factor in slot.admitted:
        if factor.name in factor_errors:
            problem = factor_errors[factor.name]
            code = str(problem.get("code") or "INVALID_PHYSICAL_FACTOR")
            state.terminal(
                ordinal, "REJECTED", error_code=code,
                detail=(f"{problem.get('error_type', 'Error')}: "
                        f"{problem.get('message', '')}"),
                retryable=False,
            )
        elif factor.name in artifact_errors:
            problem = artifact_errors[factor.name]
            code = ("RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
                    if problem.get("code") == "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
                    else "OUTPUT_INVALID")
            state.terminal(
                ordinal, "FAILED", error_code=code,
                detail=(f"{problem.get('error_type', 'ArtifactError')}: "
                        f"{problem.get('message', '')}"),
                retryable=False,
                commit_state="UNKNOWN" if problem.get("started") else "NOT_STARTED",
            )
        else:
            state.terminal(
                ordinal, "SUCCEEDED", artifact=receipts.pop(factor.name),
                commit_state="VERIFIED",
            )
            continue
        group = error_groups.setdefault(code, {"count": 0, "examples": []})
        group["count"] += 1
        if len(group["examples"]) < policy.max_examples_per_error_group:
            group["examples"].append({"ordinal": ordinal, "name": factor.name})
    slot.phase = "VERIFIED"


def _fit_failure_evidence_reference(state, *, legacy_unavailable=False):
    reference = {
        "schema_version": "factor_engine.fit_failure_evidence_index.v1",
        "coverage": "instrumented_fit_producers_only",
        "store": "state.sqlite3",
        "index": "fit_failure_evidence",
    }
    if legacy_unavailable:
        return {**reference, "availability": "legacy_unavailable", "next_seq": None}
    summary = state.fit_failure_evidence_summary()
    return {
        **reference, "availability": "indexed",
        "observed_waves": summary["observed_waves"],
        "unavailable_waves": summary["unavailable_waves"],
        "truncated_waves": summary["truncated_waves"],
        "wave_count": summary["wave_count"],
        "last_seq": summary["last_seq"],
        "next_seq": 0 if summary["last_seq"] else None,
    }


def _validate_compile_results(results, expected):
    if (type(results) is not dict or set(results) != set(expected)
            or any(value is not None and (type(value) is not str or not value)
                   for value in results.values())):
        raise WorkerProtocolError("compile protocol requires exact factor keys and typed outcomes")
    return results


def _validated_fit_failure_evidence(
    envelope, expected_evidence_id, *, expected_run_id=None, expected_factor_names=(),
):
    if "fit_failure_evidence" not in envelope:
        return None
    evidence = envelope["fit_failure_evidence"]
    valid_id = (type(expected_evidence_id) is str and len(expected_evidence_id) == 32
                and all(char in "0123456789abcdef" for char in expected_evidence_id))
    if (type(evidence) is not dict
            or set(evidence) != {"evidence_id", "scope", "snapshot"}
            or not valid_id
            or evidence.get("evidence_id") != expected_evidence_id
            or evidence.get("scope") != "wave_sample"):
        raise WorkerProtocolError("fit failure evidence identity is invalid")
    from factor_engine.runtime.fit_failure_evidence import (
        encode_fit_failure_snapshot, validate_fit_failure_snapshot,
    )
    try:
        snapshot = validate_fit_failure_snapshot(evidence.get("snapshot"))
        encode_fit_failure_snapshot(snapshot)
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("fit failure snapshot is invalid") from exc
    admitted = set(expected_factor_names)
    for detail in snapshot["details"]:
        scope = detail["scope"]
        if scope["run_id"] is not None and scope["run_id"] != expected_run_id:
            raise WorkerProtocolError("fit failure snapshot run identity is invalid")
        if scope["factor_id"] is not None and scope["factor_id"] not in admitted:
            raise WorkerProtocolError("fit failure snapshot factor identity is invalid")
    return {**evidence, "snapshot": snapshot}


def _validate_result_envelope(
    envelope, expected, *, expected_evidence_id=None, expected_run_id=None,
    include_evidence=False,
):
    legacy_keys = {"result_blobs", "transport_errors", "factor_errors"}
    extended_keys = legacy_keys | {"fit_failure_evidence"}
    if type(envelope) is not dict or frozenset(envelope) not in {
            frozenset(legacy_keys), frozenset(extended_keys)}:
        raise WorkerProtocolError("result protocol has invalid envelope fields")
    evidence = _validated_fit_failure_evidence(
        envelope, expected_evidence_id, expected_run_id=expected_run_id,
        expected_factor_names=expected)
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
    return (*groups, evidence) if include_evidence else groups


def _validate_artifact_envelope(
    envelope, expected, *, run_id=None, policy_digest=None,
    expected_evidence_id=None, include_evidence=False,
):
    legacy_keys = {"receipts", "artifact_errors", "factor_errors"}
    extended_keys = legacy_keys | {"fit_failure_evidence"}
    if type(envelope) is not dict or frozenset(envelope) not in {
            frozenset(legacy_keys), frozenset(extended_keys)}:
        raise WorkerProtocolError("artifact protocol has invalid envelope fields")
    evidence = _validated_fit_failure_evidence(
        envelope, expected_evidence_id, expected_run_id=run_id,
        expected_factor_names=expected)
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
        if isinstance(expected, dict):
            ordinal, generation = expected[name]
            identity = {
                "schema_version": "factor_engine.factor_artifact.v2",
                "run_id": run_id,
                "ordinal": ordinal,
                "factor_id": name,
                "generation": generation,
                "policy_digest": policy_digest,
            }
            if (type(receipt.get("ordinal")) is not int
                    or any(receipt.get(key) != value for key, value in identity.items())):
                raise WorkerProtocolError(
                    f"artifact receipt identity differs from assignment for {name}"
                )
    for group in groups[1:]:
        for name, error in group.items():
            if (type(error) is not dict or type(error.get("code")) is not str
                    or not error["code"]):
                raise WorkerProtocolError("artifact protocol requires typed error records")
            if isinstance(expected, dict):
                ordinal, generation = expected[name]
                identity = {
                    "schema_version": "factor_engine.factor_artifact.v2",
                    "run_id": run_id,
                    "ordinal": ordinal,
                    "factor_id": name,
                    "generation": generation,
                    "policy_digest": policy_digest,
                }
                if (type(error.get("ordinal")) is not int
                        or any(error.get(key) != value for key, value in identity.items())):
                    raise WorkerProtocolError(
                        f"artifact error identity differs from assignment for {name}"
                    )
    return (*groups, evidence) if include_evidence else groups


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
        from factor_engine.runtime.supervised_worker import emit_worker_progress
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
            emit_worker_progress("WRITE", ordinal=ordinal)
            receipt = write_verified_factor_artifact(
                self.root, self.run_id, ordinal, name, value,
                policy=self.policy, budget_bytes=available, generation=generation,
            )
            if (receipt.get("committed") is not True or receipt.get("verified") is not True
                    or receipt.get("generation") != generation):
                raise WorkerProtocolError("direct sink returned an invalid receipt")
            artifact_identity = {
                "schema_version": "factor_engine.factor_artifact.v2",
                "run_id": self.run_id,
                "ordinal": ordinal,
                "factor_id": name,
                "policy_digest": self.policy.digest,
            }
            if (("ordinal" in receipt and type(receipt["ordinal"]) is not int)
                    or any(key in receipt and receipt[key] != value
                           for key, value in artifact_identity.items())):
                raise WorkerProtocolError("direct sink receipt identity differs from artifact")
            receipt = {**receipt, **artifact_identity}
            emit_worker_progress("VERIFY_DONE", ordinal=ordinal)
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


def _compute_wave_to_artifacts(
    engine, factors, run_kwargs, artifact_plan, broker_proxy=None,
    execution_owner=None, evidence_id=None, transport_budget_bytes=None,
):
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
    from factor_engine.runtime.supervised_worker import emit_worker_progress
    for name, (ordinal, _generation) in assignments.items():
        emit_worker_progress("COMPUTE", ordinal=ordinal)
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
                isolate_physical_errors=True,
                _execution_owner=execution_owner,
                _collect_fit_failure_snapshot=True, **run_kwargs,
            )
        else:
            output = engine.run_many_parallel(
                factors, result_policy="sink", sink=target,
                _execution_owner=execution_owner,
                _collect_fit_failure_snapshot=True, **run_kwargs
            )
    fit_failure_snapshot = output.pop("_fit_failure_snapshot", None)
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
    if fit_failure_snapshot is not None:
        envelope["fit_failure_evidence"] = {
            "evidence_id": evidence_id, "scope": "wave_sample",
            "snapshot": fit_failure_snapshot,
        }
    for group in (envelope["receipts"], envelope["artifact_errors"],
                  envelope["factor_errors"]):
        for name, record in tuple(group.items()):
            ordinal, generation = assignments[name]
            identity = {
                "schema_version": "factor_engine.factor_artifact.v2",
                "run_id": artifact_plan["run_id"],
                "ordinal": ordinal,
                "factor_id": name,
                "generation": generation,
                "policy_digest": artifact_plan["policy"].digest,
            }
            if (("ordinal" in record and type(record["ordinal"]) is not int)
                    or any(key in record and record[key] != value
                           for key, value in identity.items())):
                raise WorkerProtocolError(
                    f"worker outcome identity differs from assignment for {name}"
                )
            group[name] = {**record, **identity}
    descriptor_bytes = len(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    if (type(transport_budget_bytes) is not int or transport_budget_bytes <= 0
            or descriptor_bytes * 2 > transport_budget_bytes):
        raise ResultTransportBudgetExceeded(
            "direct artifact descriptor and fit evidence exceed transport budget")
    _validate_artifact_envelope(
        envelope, assignments, run_id=artifact_plan["run_id"],
        policy_digest=artifact_plan["policy"].digest,
        expected_evidence_id=evidence_id,
    )
    return envelope


def _validated_run_identity(value):
    """Freeze caller-supplied business identity before durable state is created."""
    if value is None:
        return {}
    if type(value) is not dict or any(type(key) is not str or not key for key in value):
        raise TypeError("run_identity must be a mapping with nonempty string keys")
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        frozen = json.loads(canonical)
    except (TypeError, ValueError):
        raise TypeError("run_identity must contain finite JSON-compatible values") from None
    if frozen != value:
        raise TypeError("run_identity must preserve exact JSON identity")
    return frozen


def _stage_timeout(job_deadline, stage_budget, stage):
    remaining = float(job_deadline) - time.monotonic()
    if remaining <= 0:
        raise JobDeadlineExceeded(f"job deadline exhausted before {stage}")
    return min(float(stage_budget), remaining)


def _check_cancellation(token):
    if token is not None:
        token.raise_if_cancelled()


class _DeadlineBoundMemoryBroker:
    """Retry one shared-broker admission without creating another pool."""

    def __init__(self, broker, *, job_deadline=None, input_deadline=None,
                 resource_deadline=None, deadline=None, cancellation_token):
        self._broker = broker
        if deadline is not None:
            job_deadline = input_deadline = resource_deadline = deadline
        if any(value is None for value in (
                job_deadline, input_deadline, resource_deadline)):
            raise ValueError("all serialization admission deadlines are required")
        self._job_deadline = float(job_deadline)
        self._input_deadline = float(input_deadline)
        self._resource_deadline = float(resource_deadline)
        self._deadline = min(
            self._job_deadline, self._input_deadline, self._resource_deadline,
        )
        self._cancellation_token = cancellation_token

    def _check_deadlines(self):
        now = time.monotonic()
        if now >= self._job_deadline:
            raise JobDeadlineExceeded("job deadline exhausted before manifest ingestion")
        if now >= self._input_deadline:
            raise ManifestInputTimeout("factor input ingestion deadline exceeded")
        if now >= self._resource_deadline:
            raise ResourceWaitExhausted(
                "serialization buffer resource wait exhausted"
            )

    def acquire_memory(self, kind, nbytes, *, lease_id=""):
        while True:
            _check_cancellation(self._cancellation_token)
            self._check_deadlines()
            lease = self._broker.acquire_memory(kind, nbytes, lease_id=lease_id)
            if lease is not None:
                try:
                    _check_cancellation(self._cancellation_token)
                    self._check_deadlines()
                except BaseException as primary:
                    try:
                        lease.release()
                    except BaseException as cleanup_exc:
                        primary.cleanup_pending = True
                        primary.serialization_buffer_lease = lease
                        primary.cleanup_errors = list(
                            getattr(primary, "cleanup_errors", [])
                        ) + [cleanup_exc]
                    raise
                return lease
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self._check_deadlines()
            # The current broker change-event API is scoped exclusively to its
            # protected-egress FIFO ticket. Generic memory admission must not
            # enroll there or block writer progress; bounded polling is used
            # until a separate generic notification API exists.
            time.sleep(min(0.05, remaining))


@contextmanager
def _active_cancellation(token):
    if token is None:
        yield
        return
    from factor_engine.runtime.exceptions import (
        get_active_cancellation_token, reset_active_cancellation_token,
        set_active_cancellation_token,
    )
    if get_active_cancellation_token() is token:
        yield
        return
    reset_token = set_active_cancellation_token(token)
    try:
        yield
    finally:
        reset_active_cancellation_token(reset_token)


def _await_direct_handles(current, prefetched, token):
    """Wait responsively; cancellation retires every admitted slot first."""
    while True:
        if current.done:
            return current.result()
        try:
            _check_cancellation(token)
        except BaseException as cancellation:
            errors = []
            phases = {}
            for handle in (current, prefetched):
                if handle is None:
                    continue
                try:
                    handle.cancel_and_retire()
                except BaseException as exc:
                    errors.append(exc)
                phases.update(getattr(handle, "cancelled_phases", {}))
            if errors:
                failure = errors[0]
                failure.cleanup_errors = list(
                    getattr(failure, "cleanup_errors", [])
                ) + errors[1:]
                raise failure
            cancellation.worker_phases = phases
            raise cancellation
        try:
            return current.result(timeout=.05)
        except TimeoutError:
            pass


def _await_any_direct_slot(slots, token):
    """Return the first completed slot; cancellation retires all live slots."""
    while True:
        # Prefer the newest admitted slot when completions race. This ensures a
        # just-finished refill is verified before an older gate is released.
        for slot in reversed(slots):
            if slot.handle.done:
                slot.phase = "DONE"
                try:
                    return slot, slot.handle.result()
                except BaseException as exc:
                    cleanup_errors = []
                    for live_slot in slots:
                        if live_slot is slot:
                            continue
                        try:
                            live_slot.worker.cancel_grace_seconds = 0.0
                            live_slot.handle.cancel_and_retire()
                            live_slot.phase = "RETIRED"
                        except BaseException as cleanup_exc:
                            cleanup_errors.append(cleanup_exc)
                    exc.direct_slot_fatal = True
                    exc.failing_slot = slot
                    exc.cleanup_errors = list(
                        getattr(exc, "cleanup_errors", [])
                    ) + cleanup_errors
                    raise
        try:
            _check_cancellation(token)
        except BaseException as cancellation:
            errors = []
            phases = {}
            for slot in slots:
                try:
                    slot.handle.cancel_and_retire()
                    slot.phase = "RETIRED"
                except BaseException as exc:
                    errors.append(exc)
                phases.update(getattr(slot.handle, "cancelled_phases", {}))
            if errors:
                failure = errors[0]
                failure.cleanup_errors = list(
                    getattr(failure, "cleanup_errors", [])
                ) + errors[1:]
                # Cleanup uncertainty is conservatively an unknown commit;
                # progress observed before quarantine cannot prove the final
                # write boundary after retirement reporting itself failed.
                failure.worker_phases = {}
                raise failure
            cancellation.worker_phases = phases
            raise cancellation
        time.sleep(.05)


def _await_worker_call(handle, token, *active_handles):
    """Wait for an auxiliary phase without creating a cancellation blind spot."""
    while True:
        if handle.done:
            return handle.result()
        try:
            _check_cancellation(token)
        except BaseException as cancellation:
            errors = []
            phases = {}
            for active in (handle,) + active_handles:
                if active is None:
                    continue
                try:
                    active.cancel_and_retire()
                except BaseException as exc:
                    errors.append(exc)
                phases.update(getattr(active, "cancelled_phases", {}))
            if errors:
                cancellation.cleanup_errors = errors
            cancellation.worker_phases = phases
            raise cancellation
        try:
            return handle.result(timeout=.05)
        except TimeoutError:
            pass


def _worker_thread_environment(proxy):
    quota = getattr(proxy, "cpu_quota", None)
    if type(quota) is not int or quota <= 0:
        return None
    value = str(quota)
    return {name: value for name in (
        "POLARS_MAX_THREADS", "DUCKDB_THREADS", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}


def _acquire_protected_egress_until(
    broker, queue_bytes, writer_bytes, *, lease_id, deadline, poll_seconds,
    cancellation_token=None,
):
    """Wait finitely for temporary pressure; reject provably impossible size."""
    if int(queue_bytes) <= 0 or int(writer_bytes) <= 0:
        return None, "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    required = int(queue_bytes) + int(writer_bytes)
    hard_limit = getattr(broker, "hard_memory_limit", None)
    if type(hard_limit) is int and hard_limit > 0 and required > hard_limit:
        return None, "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
    waiter_scope = getattr(broker, "protected_egress_waiter", None)
    scope = (waiter_scope(lease_id, deadline) if callable(waiter_scope)
             else nullcontext())
    with _active_cancellation(cancellation_token), scope:
        while True:
            _check_cancellation(cancellation_token)
            if float(deadline) - time.monotonic() <= 0:
                return None, "RESOURCE_WAIT_EXHAUSTED"
            leases = broker.acquire_protected_egress(
                queue_bytes, writer_bytes, lease_id=lease_id
            )
            if time.monotonic() >= float(deadline):
                if leases is not None:
                    for lease in leases:
                        lease.release()
                return None, "RESOURCE_WAIT_EXHAUSTED"
            if leases is not None:
                return leases, None
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                return None, "RESOURCE_WAIT_EXHAUSTED"
            interval = min(float(poll_seconds), remaining)
            waiter = getattr(broker, "wait_for_resource_change", None)
            if callable(waiter):
                wait_started = time.monotonic()
                changed = waiter(interval)
                # Event APIs may wake spuriously.  Only an explicit True denotes
                # a resource transition; otherwise retain the bounded cadence.
                if changed is not True:
                    residual = interval - (time.monotonic() - wait_started)
                    if residual > 0:
                        time.sleep(residual)
            else:
                time.sleep(interval)


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
    return hashlib.sha256(payload).hexdigest()


class _RunCoordinatorLock:
    """Crash-visible single coordinator ownership for one durable run."""

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / "coordinator.lock"
        self.token = uuid.uuid4().hex
        self.owned = False
        self._fd = None

    def acquire(self, *, allow_dead_owner: bool) -> None:
        import fcntl
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise RuntimeError("coordinator lock path is not safely openable") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError("prior run coordinator is still alive") from None
        except BaseException:
            os.close(fd)
            raise
        payload = {"pid": os.getpid(), "token": self.token}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        try:
            os.ftruncate(fd, 0)
            os.write(fd, encoded)
            os.fsync(fd)
            self._fd = fd
            self.owned = True
        except BaseException:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise

    def release(self) -> None:
        if not self.owned:
            return
        import fcntl
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
            self.owned = False


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
                  transport_budget_bytes: int, broker_proxy: Any = None,
                  execution_owner: dict[str, str] | None = None,
                  evidence_id: str | None = None) -> bytes:
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
                isolate_physical_errors=True,
                _execution_owner=execution_owner,
                _collect_fit_failure_snapshot=True, **run_kwargs,
            )
        else:
            output = engine.run_many_parallel(
                factors, result_policy="return",
                _execution_owner=execution_owner,
                _collect_fit_failure_snapshot=True, **run_kwargs
            )
    fit_failure_snapshot = output.pop("_fit_failure_snapshot", None)
    factor_errors = dict(output.pop("physical_preflight_errors", {}) or {})
    results = output.pop("results", {})
    if not isinstance(results, dict):
        results = dict(results)
    del output
    evidence = None
    snapshot_charge = 0
    if fit_failure_snapshot is not None:
        evidence = {"evidence_id": evidence_id, "scope": "wave_sample",
                    "snapshot": fit_failure_snapshot}
        from factor_engine.runtime.fit_failure_evidence import encode_fit_failure_snapshot
        snapshot_charge = 4 * (len(encode_fit_failure_snapshot(fit_failure_snapshot)) + 256)
    used = 1024 + snapshot_charge
    if used > transport_budget_bytes:
        raise ResultTransportBudgetExceeded("fit failure evidence exceeds transport budget")
    result_blobs, transport_errors = {}, {}
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
    envelope = {"result_blobs": result_blobs, "transport_errors": transport_errors,
                "factor_errors": factor_errors}
    if evidence is not None:
        envelope["fit_failure_evidence"] = evidence
    payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) * 2 > transport_budget_bytes:
        raise ResultTransportBudgetExceeded("bounded transport envelope overhead exceeds budget")
    return payload


def _compute_wave_worker(
    engine, factors, run_kwargs, transport_budget, artifact_plan=None, evidence_id=None,
):
    """Give fork and spawn compute callables one identical request shape."""
    if artifact_plan is not None:
        raise WorkerProtocolError("result worker received an artifact plan")
    return _compute_wave(
        engine, factors, run_kwargs, transport_budget, evidence_id=evidence_id)


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
                 transport_budget: int | None = None, artifact_plan: dict[str, Any] | None = None,
                 evidence_id: str | None = None):
        engine = self._get_engine()
        if self.mode == "compile":
            return _compile_wave(engine, factors)
        execution_owner: dict[str, str] = {}
        deployment = getattr(self.config, "deployment", None)
        if deployment is not None:
            profile_id = deployment.to_dict().get("profile_id")
            if type(profile_id) is str and profile_id:
                execution_owner["profile_id"] = profile_id
        if artifact_plan is not None:
            run_id = artifact_plan.get("run_id")
            if type(run_id) is str and run_id:
                execution_owner["run_id"] = run_id
            return _compute_wave_to_artifacts(
                engine, factors, run_kwargs or {}, artifact_plan, self.broker_proxy,
                execution_owner=execution_owner or None, evidence_id=evidence_id,
                transport_budget_bytes=int(transport_budget),
            )
        return _compute_wave(engine, factors, run_kwargs or {}, int(transport_budget),
                             self.broker_proxy, execution_owner=execution_owner or None,
                             evidence_id=evidence_id)

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
    run_identity: dict[str, Any] | None = None,
    resume_run_id: str | None = None,
    cancellation_token: Any = None,
) -> dict[str, Any]:
    """Execute finite input with ordinal terminals and one persisted retry budget.

    Numerical work always goes through ``engine.run_many_parallel``. This layer
    only bounds ingestion/residency, records outcomes, and owns durable delivery.
    """
    run_kwargs = dict(run_kwargs or {})
    if cancellation_token is None:
        from factor_engine.runtime.exceptions import get_active_cancellation_token
        cancellation_token = get_active_cancellation_token()
    _check_cancellation(cancellation_token)
    from factor_engine.runtime.default_execution_policy import DefaultExecutionPolicy
    if not isinstance(policy, DefaultExecutionPolicy):
        raise TypeError("policy must be a validated DefaultExecutionPolicy")
    run_identity = _validated_run_identity(run_identity)
    job_started_monotonic_ns = time.monotonic_ns()
    job_started_realtime_ns = time.time_ns()
    # Terminal-only revalidation has its own finite observation budget. New
    # computation persists this exact original epoch before any worker starts.
    job_deadline = job_started_monotonic_ns / 1_000_000_000 + min(
        float(policy.job_unknown_seconds), float(policy.job_max_seconds)
    )
    broker = run_kwargs.pop("broker", None) or getattr(engine, "resource_broker", None)
    if broker is None:
        try:
            from factor_engine.runtime.host_resource_coordinator import get_host_coordinator
            broker = get_host_coordinator().broker
        except Exception as exc:
            raise RuntimeError("v2 durable execution requires the shared ResourceBroker") from exc
    root = Path(artifact_root)
    if resume_run_id is None and manifest_path is not None and state_path is not None:
        supplied_manifest = Path(manifest_path)
        supplied_state = Path(state_path)
        if supplied_manifest.exists() and supplied_state.exists():
            try:
                resolved_root = root.resolve(strict=True)
                manifest_parent = supplied_manifest.resolve(strict=True).parent
                state_parent = supplied_state.resolve(strict=True).parent
            except OSError:
                pass
            else:
                if (supplied_manifest.name == "manifest.sqlite3"
                        and supplied_state.name == "state.sqlite3"
                        and manifest_parent == state_parent
                        and manifest_parent.parent == resolved_root):
                    resume_run_id = manifest_parent.name
                    manifest_path = state_path = None
    max_attempts = min(3, int(_policy_value(policy, "work_item_max_attempts", 3)))
    lookahead = int(_policy_value(policy, "initial_lookahead_factors", 512))
    coordinator_lock = None
    ownership_context = None
    resuming = resume_run_id is not None
    resume_pending_count = None
    resume_execute_new = False
    if resuming:
        if manifest_path is not None or state_path is not None:
            raise ValueError("resume_run_id owns canonical manifest/state paths")
        from factor_engine.runtime.resume_validation import (
            ResumeIdentityError, validate_resume_context,
        )
        if (type(resume_run_id) is not str or len(resume_run_id) != 32
                or any(char not in "0123456789abcdef" for char in resume_run_id)):
            raise ResumeIdentityError("resume_run_id must be a lowercase UUID hex")
        try:
            root = root.resolve(strict=True)
        except OSError as exc:
            raise ResumeIdentityError("resume artifact root does not exist") from exc
        run_id, run_dir = resume_run_id, root / resume_run_id
        if run_dir.is_symlink() or not run_dir.is_dir() or run_dir.resolve().parent != root:
            raise ResumeIdentityError("run directory is missing or outside artifact root")
        coordinator_lock = _RunCoordinatorLock(run_dir)
        coordinator_lock.acquire(allow_dead_owner=True)
        manifest = None
        try:
            resume_context = validate_resume_context(
                root, resume_run_id, policy=policy, run_identity=run_identity,
                deadline=job_deadline,
            )
            resume_pending_count = resume_context.pending_count
            if (resume_context.ownership_store == _OWNERSHIP_STORE
                    and resume_context.ownership_coverage == _OWNERSHIP_COVERAGE):
                from factor_engine.runtime.worker_ownership import RunOwnershipContext
                ownership_context = RunOwnershipContext(
                    run_id, resume_context.identity_sha256,
                )
            if resume_pending_count:
                if (ownership_context is None
                        or not resume_context.fit_failure_evidence_available):
                    raise ResumeIdentityError(
                        "pending resume requires full SQLite ownership and indexed evidence"
                    )
                from factor_engine.runtime.run_deadline import restore_run_deadline
                restored = restore_run_deadline(
                    resume_context.job_deadline_record,
                    expected_run_id=run_id, expected_policy_digest=policy.digest,
                )
                if restored.expired:
                    raise ResumeIdentityError("restored job deadline exhausted")
                job_deadline = restored.deadline_monotonic
                from factor_engine.runtime.worker_ownership_sqlite import (
                    validate_all_workers_exited_sqlite,
                )
                retirement = validate_all_workers_exited_sqlite(
                    run_dir, context=ownership_context,
                )
                if not retirement.all_exited:
                    raise ResumeIdentityError(
                        "pending resume requires complete persisted worker-exit proof"
                    )
                from factor_engine.runtime.resume_action_catalog import (
                    iter_resume_action_catalog,
                )
                allowed = {
                    "EXECUTE_NEW", "PRESERVE_TERMINAL",
                    "REVALIDATE_VERIFIED_ARTIFACT",
                }
                execute_count = 0
                for action in iter_resume_action_catalog(
                        resume_context, policy=policy,
                        restored_job_deadline=job_deadline):
                    if action.action not in allowed:
                        raise ResumeIdentityError(
                            "pending same-run execution contains dispatched or "
                            "reconcilable work without authorization"
                        )
                    execute_count += action.action == "EXECUTE_NEW"
                if execute_count != resume_pending_count:
                    raise ResumeIdentityError(
                        "pending resume is not exactly never-dispatched work"
                    )
                resume_execute_new = True
            manifest_path, state_path = resume_context.manifest_path, resume_context.state_path
            manifest = FiniteFactorManifest(manifest_path)
            state = PersistentRunState(state_path, max_attempts=max_attempts)
        except BaseException as exc:
            cleanup_errors = list(getattr(exc, "cleanup_errors", []))
            cleanup_pending = (
                getattr(exc, "cleanup_pending", False)
                or any(getattr(error, "cleanup_pending", False)
                       for error in cleanup_errors)
            )
            if not cleanup_pending and manifest is not None:
                try:
                    manifest.close()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
                    cleanup_pending = getattr(cleanup_exc, "cleanup_pending", False)
            if cleanup_pending:
                exc.cleanup_pending = True
                exc.coordinator_lock = coordinator_lock
                exc.broker = broker
                if manifest is not None:
                    exc.resume_manifest = manifest
            else:
                try:
                    coordinator_lock.release()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
                    exc.coordinator_lock = coordinator_lock
            if cleanup_errors:
                exc.cleanup_errors = cleanup_errors
            raise
    else:
        root.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        identity_path = run_dir / "identity.json"
        from factor_engine.runtime.run_deadline import (
            create_run_deadline_record, restore_run_deadline,
        )
        deadline_record = create_run_deadline_record(
            run_id=run_id, policy_digest=policy.digest,
            job_unknown_seconds=policy.job_unknown_seconds,
            job_max_seconds=policy.job_max_seconds,
            monotonic_ns=job_started_monotonic_ns,
            realtime_ns=job_started_realtime_ns,
        )
        job_deadline = restore_run_deadline(
            deadline_record, expected_run_id=run_id,
            expected_policy_digest=policy.digest,
        ).deadline_monotonic
        identity_payload = {
            "schema_version": "factor_engine.run_identity.v1",
            "run_id": run_id,
            "run_identity": run_identity,
            "deployment_digest": run_identity.get("deployment_digest"),
            "policy_id": _policy_value(policy, "policy_id", None),
            "policy_digest": getattr(policy, "digest", None),
            "job_deadline": deadline_record,
            "ownership_store": _OWNERSHIP_STORE,
            "ownership_coverage": _OWNERSHIP_COVERAGE,
        }
        identity_digest = _write_control_receipt(identity_path, identity_payload)
        from factor_engine.runtime.worker_ownership import RunOwnershipContext
        ownership_context = RunOwnershipContext(
            run_id, identity_digest,
        )
        manifest_path = Path(manifest_path or run_dir / "manifest.sqlite3")
        state_path = Path(state_path or run_dir / "state.sqlite3")
        for database_path in (manifest_path, state_path):
            if database_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing durable state {database_path}; "
                    "resume is not implemented through legacy path arguments"
                )
        if (manifest_path != run_dir / "manifest.sqlite3"
                or state_path != run_dir / "state.sqlite3"):
            raise ValueError("durable manifest/state must use canonical run-directory paths")
        # Ingestion is itself a database-writing worker. Exclusive run ownership
        # must precede it, not just the later compute/writer pipeline.
        coordinator_lock = _RunCoordinatorLock(run_dir)
        coordinator_lock.acquire(allow_dead_owner=False)
        manifest = None
        try:
            from factor_engine.runtime.worker_ownership_sqlite import initialize_worker_store
            initialize_worker_store(run_dir, context=ownership_context)
            ingestion_started = time.monotonic()
            ingestion_deadline = min(
                job_deadline,
                ingestion_started + float(_policy_value(
                    policy, "input_ingestion_deadline_seconds", 300,
                )),
            )
            input_deadline = ingestion_started + float(_policy_value(
                policy, "input_ingestion_deadline_seconds", 300,
            ))
            resource_deadline = (
                ingestion_started + float(policy.resource_wait_seconds)
            )
            serialization_broker = _DeadlineBoundMemoryBroker(
                broker, job_deadline=job_deadline,
                input_deadline=input_deadline,
                resource_deadline=resource_deadline,
                cancellation_token=cancellation_token,
            )
            remaining_ingestion = ingestion_deadline - time.monotonic()
            if remaining_ingestion <= 0:
                raise JobDeadlineExceeded(
                    "job deadline exhausted before manifest ingestion"
                )
            manifest = FiniteFactorManifest.ingest_supervised(
                factors, manifest_path,
                process_context="spawn" if engine_factory is not None else "fork",
                max_factors=int(_policy_value(policy, "max_manifest_factors", 1_000_000)),
                max_definition_bytes=int(_policy_value(policy, "max_definition_bytes", 131_072)),
                deadline_seconds=remaining_ingestion,
                deadline_monotonic=ingestion_deadline,
                serialization_broker=serialization_broker,
                ownership_run_dir=run_dir,
                ownership_context=ownership_context,
            )
            state = PersistentRunState(state_path, max_attempts=max_attempts)
            state.enable_fit_failure_evidence()
        except BaseException as exc:
            if (getattr(exc, "cleanup_pending", False) or any(
                    getattr(error, "cleanup_pending", False)
                    for error in getattr(exc, "cleanup_errors", []))):
                exc.cleanup_pending = True
                exc.coordinator_lock = coordinator_lock
                exc.broker = broker
                if manifest is not None:
                    exc.run_manifest = manifest
            else:
                cleanup_errors = list(getattr(exc, "cleanup_errors", []))
                if manifest is not None:
                    try:
                        manifest.close()
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                if any(getattr(error, "cleanup_pending", False) for error in cleanup_errors):
                    exc.cleanup_pending = True
                    exc.coordinator_lock = coordinator_lock
                    exc.broker = broker
                    if manifest is not None:
                        exc.run_manifest = manifest
                else:
                    try:
                        coordinator_lock.release()
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                        exc.coordinator_lock = coordinator_lock
                if cleanup_errors:
                    exc.cleanup_errors = cleanup_errors
            raise
    identity_path = run_dir / "identity.json"

    def ownership_kwargs(role):
        if ownership_context is None:
            return {}
        return {
            "ownership_run_dir": run_dir,
            "ownership_context": ownership_context,
            "ownership_role": role,
        }

    def resume_admission_allowed(ordinal):
        if not resume_execute_new:
            return True
        page = state.outcomes_page(start=ordinal, limit=1)
        return bool(
            page and page[0].ordinal == ordinal
            and page[0].state == "ACCEPTED"
            and page[0].attempts == 0
            and page[0].commit_state == "NOT_STARTED"
        )
    if resuming:
        from factor_engine.runtime.auto_memory_budget import MemoryLeaseKind
        from factor_engine.runtime.resume_validation import (
            ResumeIdentityError, iter_resume_artifacts_to_revalidate,
            verify_previously_committed_artifact,
        )
        verification_worker = None
        verification_close_started = False
        try:
            verification_worker = SupervisedReusableWorker(
                context="spawn",
                **ownership_kwargs("resume-artifact-verifier"),
                cancel_grace_seconds=float(policy.cooperative_cancel_grace_seconds),
                exit_observation_seconds=float(policy.worker_exit_observation_seconds),
            )
            for artifact_record in iter_resume_artifacts_to_revalidate(
                    resume_context, deadline=job_deadline):
                available = int(broker.current_sink_budget())
                expected_manifest = (
                    resume_context.run_dir / "values"
                    / f'{artifact_record["ordinal"]:08d}-{artifact_record["generation"]}'
                    / "manifest.json"
                )
                try:
                    manifest_bytes = expected_manifest.stat().st_size
                except OSError as exc:
                    raise ResumeIdentityError("persisted artifact manifest is missing") from exc
                required = max(65536, manifest_bytes * 4 + 1024 * 1024)
                verification_budget = min(required, available)
                if verification_budget < 65536:
                    raise ResumeIdentityError(
                        "insufficient parent workspace for resume artifact verification"
                    )
                verify_timeout = _stage_timeout(
                    job_deadline, policy.reconcile_seconds,
                    "resume artifact verification",
                )
                lease = broker.acquire_memory(
                    MemoryLeaseKind.WRITER_BATCH, verification_budget,
                    lease_id=f"v2-resume-verify:{run_id}:{artifact_record['ordinal']}",
                )
                if lease is None:
                    raise ResumeIdentityError(
                        "resume artifact verification workspace was not admitted"
                    )
                transferred = False
                try:
                    transferred = True
                    verification_worker.execute(
                        verify_previously_committed_artifact,
                        resume_context, artifact_record, policy=policy,
                        budget_bytes=verification_budget, deadline=job_deadline,
                        timeout_seconds=verify_timeout,
                        lease=lease,
                    )
                finally:
                    if not transferred:
                        lease.release()
            verification_close_started = True
            verification_worker.close()
        except BaseException as exc:
            from factor_engine.runtime.supervised_worker import WorkerQuarantined
            cleanup_errors = list(getattr(exc, "cleanup_errors", []))
            cleanup_pending = (
                isinstance(exc, WorkerQuarantined)
                or verification_close_started
                or getattr(exc, "cleanup_pending", False)
                or any(getattr(error, "cleanup_pending", False)
                       for error in cleanup_errors)
            )
            if not cleanup_pending and verification_worker is not None:
                try:
                    verification_worker.close()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
                    # Any failed close leaves retirement unproven here. Keep
                    # the original verification error and all run authorities.
                    cleanup_pending = True
            if cleanup_pending:
                exc.cleanup_pending = True
                exc.verification_worker = verification_worker
                exc.broker = broker
                exc.coordinator_lock = coordinator_lock
                exc.resume_state = state
                exc.resume_manifest = manifest
            else:
                for authority in (state, manifest):
                    try:
                        authority.close()
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                        if getattr(cleanup_exc, "cleanup_pending", False):
                            cleanup_pending = True
                            break
                if cleanup_pending:
                    exc.cleanup_pending = True
                    exc.verification_worker = verification_worker
                    exc.broker = broker
                    exc.coordinator_lock = coordinator_lock
                    exc.resume_state = state
                    exc.resume_manifest = manifest
                else:
                    try:
                        coordinator_lock.release()
                    except BaseException as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
                        exc.coordinator_lock = coordinator_lock
            if cleanup_errors:
                exc.cleanup_errors = cleanup_errors
            raise
    egress_leases: tuple[Any, Any] | None = None
    writer_bytes = 0
    queue_bytes = 0
    two_slot_egress = False
    direct_artifacts = engine_factory is not None and sink is None
    execution_cpu_budget = None
    broker_ipc = compute_proxy = spare_compute_proxy = compile_proxy = direct_reconcile_proxy = None
    if engine_factory is not None:
        from factor_engine.runtime.resource_broker_ipc import ParentBrokerIPC
        broker_ipc = ParentBrokerIPC(broker, rpc_timeout_seconds=policy.connect_seconds)
        execution_cpu_budget = int(broker.cpu_budget())
        partitions = broker_ipc.create_partitioned_proxies(2) if direct_artifacts else []
        if partitions:
            compute_proxy, spare_compute_proxy = partitions
        else:
            compute_proxy = broker_ipc.create_proxy()
        compile_proxy = broker_ipc.create_proxy()
        direct_reconcile_proxy = broker_ipc.create_proxy() if sink is None else None
    worker_context = "spawn" if engine_factory is not None else "fork"
    compute_callable = (_SpawnFactoryRuntime(engine_factory, engine_factory_config,
                                             compute_proxy, "compute")
                        if engine_factory is not None else partial(_compute_wave_worker, engine))
    compile_callable = (_SpawnFactoryRuntime(engine_factory, engine_factory_config,
                                             compile_proxy, "compile")
                        if engine_factory is not None else partial(_compile_wave, engine))
    compute_worker = SupervisedReusableWorker(
        **ownership_kwargs("compute-primary"),
        cancel_grace_seconds=float(_policy_value(policy, "cooperative_cancel_grace_seconds", 5)),
        exit_observation_seconds=float(_policy_value(policy, "worker_exit_observation_seconds", 10)),
        context=worker_context,
        function=compute_callable,
        process_environment=_worker_thread_environment(compute_proxy),
    )
    compile_worker = SupervisedReusableWorker(
        **ownership_kwargs("compiler"),
        cancel_grace_seconds=float(policy.cooperative_cancel_grace_seconds),
        exit_observation_seconds=float(policy.worker_exit_observation_seconds),
        context=worker_context, function=compile_callable,
    )
    writer_callable = (partial(_custom_worker_write_verify, sink, root, run_id, policy)
                       if sink is not None else partial(_default_worker_write, root, run_id, policy))
    sink_worker = SupervisedReusableWorker(
        **ownership_kwargs("artifact-writer"),
        cancel_grace_seconds=float(_policy_value(policy, "cooperative_cancel_grace_seconds", 5)),
        exit_observation_seconds=float(_policy_value(policy, "worker_exit_observation_seconds", 10)),
        context="spawn",
        function=writer_callable,
    )
    reconcile_worker = None if sink is not None else SupervisedReusableWorker(
        **ownership_kwargs("artifact-reconciler"),
        context="spawn",
        function=partial(_default_worker_reconcile, root, run_id, policy),
        cancel_grace_seconds=float(policy.cooperative_cancel_grace_seconds),
        exit_observation_seconds=float(policy.worker_exit_observation_seconds),
    )
    direct_reconcile_worker = (
        SupervisedReusableWorker(
            **ownership_kwargs("direct-artifact-reconciler"),
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
    prefetched_direct = None
    spare_compute_worker = None
    refill_compile_worker = None
    refill_compile_proxy = None
    try:
        _check_cancellation(cancellation_token)
        state.register_many(
            (record.ordinal, record.name)
            for record in manifest.records(start=0, limit=len(manifest) or 1)
        )
        # Sealing is a fallible, deadline-bound part of a new run. Register the
        # ingested ordinals first so seal failure uses the same durable abort
        # state machine, before any compiler/compute/writer child is started.
        if not resuming and manifest.input_complete:
            from factor_engine.runtime.resume_validation import manifest_seal_payload
            _write_control_receipt(
                run_dir / "manifest_identity.json",
                manifest_seal_payload(run_dir, policy=policy, deadline=job_deadline),
            )
        if (len(manifest) > 0 and resume_pending_count != 0 and broker_ipc is not None
                and execution_cpu_budget > 0):
            compile_worker.start()
            broker_ipc.bind_client_process(compile_proxy.client_id, compile_worker._process.pid)
            compute_worker.start()
            broker_ipc.bind_client_process(compute_proxy.client_id, compute_worker._process.pid)
            if direct_reconcile_worker is not None:
                direct_reconcile_worker.start()
                broker_ipc.bind_client_process(
                    direct_reconcile_proxy.client_id, direct_reconcile_worker._process.pid
                )
        cursor = len(manifest) if resume_pending_count == 0 else 0
        if broker_ipc is not None and execution_cpu_budget <= 0:
            code = "RESOURCE_REQUIREMENT_EXCEEDS_BUDGET"
            group = error_groups.setdefault(code, {"count": 0, "examples": []})
            for record in manifest.records(start=0, limit=len(manifest) or 1):
                state.terminal(
                    record.ordinal, "FAILED", error_code=code,
                    detail="no positive CPU execution authority", retryable=False,
                )
                group["count"] += 1
                if len(group["examples"]) < policy.max_examples_per_error_group:
                    group["examples"].append(
                        {"ordinal": record.ordinal, "name": record.name}
                    )
            cursor = len(manifest)
        while cursor < len(manifest):
            cursor_after_direct_slots = None
            _check_cancellation(cancellation_token)
            try:
                metadata_bytes = max(1, min(int(broker.current_read_budget()), 64 * 1024 * 1024))
            except Exception:
                metadata_bytes = max(1, policy.max_definition_bytes)
            prevalidated_wave = bool(
                prefetched_direct is not None
                and prefetched_direct["cursor"] == cursor
            )
            wave = (prefetched_direct["wave"] if prevalidated_wave else
                    manifest.load_wave_bounded(
                        start=cursor, max_items=max(1, lookahead),
                        max_definition_bytes=metadata_bytes,
                    ))
            if not wave:
                break
            if prevalidated_wave:
                admitted = list(prefetched_direct["admitted"])
            else:
                admitted = []
                for ordinal, factor, manifest_error, manifest_name in wave:
                    if not resume_admission_allowed(ordinal):
                        continue
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
            if admitted and not prevalidated_wave:
                _check_cancellation(cancellation_token)
                try:
                    compile_results = compile_worker.execute(
                        [factor for _, factor in admitted],
                        timeout_seconds=_stage_timeout(
                            job_deadline, policy.compute_unknown_seconds, "compile"
                        ),
                    ).value
                except Exception as exc:
                    from factor_engine.runtime.exceptions import Cancellation, DeadlineExceeded
                    from factor_engine.runtime.supervised_worker import (
                        WorkerQuarantined, WorkerStaleResponse,
                        WorkerTimedOut, WorkerTransportFailed,
                    )
                    if isinstance(exc, (Cancellation, DeadlineExceeded)):
                        raise
                    if getattr(exc, "direct_slot_fatal", False):
                        raise
                    if isinstance(
                        exc, (WorkerQuarantined, WorkerStaleResponse,
                              WorkerTimedOut, WorkerTransportFailed,
                              JobDeadlineExceeded)
                    ):
                        raise
                    compile_results = {factor.name: f"{type(exc).__name__}: {exc}"
                                       for _, factor in admitted}
                _validate_compile_results(compile_results, {factor.name for _, factor in admitted})
                _check_cancellation(cancellation_token)
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
                    total_queue_bytes = int(broker.automatic_result_queue_budget())
                    total_writer_bytes = min(total_queue_bytes, int(_policy_value(
                        policy, "target_file_bytes", 134_217_728
                    )))
                    egress_leases, egress_error = _acquire_protected_egress_until(
                        broker, total_queue_bytes, total_writer_bytes,
                        lease_id=f"v2-egress:{run_id}",
                        deadline=min(
                            job_deadline,
                            time.monotonic() + float(policy.resource_wait_seconds),
                        ),
                        poll_seconds=policy.sample_interval_seconds,
                        cancellation_token=cancellation_token,
                    )
                    two_slot_egress = bool(
                        spare_compute_proxy is not None
                        and total_queue_bytes >= 2 and total_writer_bytes >= 2
                    )
                    slot_count = 2 if two_slot_egress else 1
                    queue_bytes = total_queue_bytes // slot_count
                    writer_bytes = total_writer_bytes // slot_count
                    if egress_leases is None:
                        code = egress_error
                        group = error_groups.setdefault(
                            code, {"count": 0, "examples": []}
                        )
                        for ordinal, factor in admitted:
                            state.terminal(ordinal, "FAILED", error_code=code,
                                           detail="protected result egress unavailable", retryable=False)
                            group["count"] += 1
                            if len(group["examples"]) < policy.max_examples_per_error_group:
                                group["examples"].append(
                                    {"ordinal": ordinal, "name": factor.name}
                                )
                        admitted = []
            if admitted:
                if not prevalidated_wave:
                    execution_batches += 1
                sink_started: set[int] = set()
                artifact_assignments = {}
                using_prefetched = bool(
                    direct_artifacts and prefetched_direct is not None
                    and prefetched_direct["cursor"] == cursor
                    and prefetched_direct["names"] == [factor.name for _, factor in admitted]
                )
                for ordinal, _factor in admitted:
                    if using_prefetched:
                        artifact_assignments[_factor.name] = prefetched_direct["assignments"][_factor.name]
                    else:
                        if state.consume_attempt(ordinal, "execution") is None:
                            raise RuntimeError("admitted execution lost its attempt authority")
                    if direct_artifacts and not using_prefetched:
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
                wave_evidence_id = (
                    prefetched_direct.evidence_id if using_prefetched
                    else uuid.uuid4().hex
                )
                wave_dispatched = using_prefetched
                try:
                    wave_decoded = False
                    direct_slot_completed = False
                    lost_ack_reconciled = False
                    if direct_artifacts:
                        if using_prefetched:
                            consumed_prefetch = prefetched_direct
                            current_handle = consumed_prefetch["handle"]
                            current_evidence_id = consumed_prefetch.evidence_id
                            prefetched_direct = None
                        else:
                            current_handle = None
                            current_evidence_id = wave_evidence_id
                        pending_next = None
                        next_cursor = cursor + len(wave)
                        if (two_slot_egress
                                and spare_compute_proxy is not None
                                and next_cursor < len(manifest)
                                and prefetched_direct is None):
                            next_wave = manifest.load_wave_bounded(
                                start=next_cursor, max_items=max(1, lookahead),
                                max_definition_bytes=metadata_bytes,
                            )
                            next_admitted = []
                            for next_ordinal, next_factor, next_error, _next_name in next_wave:
                                if not resume_admission_allowed(next_ordinal):
                                    continue
                                next_factor, next_scope_error = _apply_execution_scope(
                                    next_factor, execution_scope
                                )
                                if next_error or next_scope_error:
                                    next_admitted = []
                                    break
                                next_admitted.append((next_ordinal, next_factor))
                            if next_admitted:
                                if using_prefetched:
                                    slot_cpu = spare_compute_proxy.cpu_quota
                                    slot_io = spare_compute_proxy.io_quota
                                    refill_compile_proxy = spare_compute_proxy
                                    refill_compile_worker = SupervisedReusableWorker(
                                        **ownership_kwargs("refill-compiler"),
                                        context="spawn",
                                        function=_SpawnFactoryRuntime(
                                            engine_factory, engine_factory_config,
                                            spare_compute_proxy, "compile",
                                        ),
                                        cancel_grace_seconds=float(
                                            policy.cooperative_cancel_grace_seconds
                                        ),
                                        exit_observation_seconds=float(
                                            policy.worker_exit_observation_seconds
                                        ),
                                        process_environment=_worker_thread_environment(
                                            spare_compute_proxy
                                        ),
                                    )
                                    try:
                                        refill_compile_worker.start()
                                        broker_ipc.bind_client_process(
                                            spare_compute_proxy.client_id,
                                            refill_compile_worker._process.pid,
                                        )
                                        refill_compile_handle = refill_compile_worker.execute_async(
                                            [factor for _, factor in next_admitted],
                                            timeout_seconds=_stage_timeout(
                                                job_deadline, policy.compute_unknown_seconds,
                                                "refill-slot compile",
                                            ),
                                        )
                                        next_compile = _await_worker_call(
                                            refill_compile_handle, cancellation_token,
                                            current_handle,
                                        ).value
                                        refill_compile_worker.close()
                                        broker_ipc.reclaim_client(
                                            spare_compute_proxy.client_id
                                        )
                                        refill_compile_worker = None
                                        refill_compile_proxy = None
                                        spare_compute_proxy = broker_ipc.create_proxy(
                                            cpu_quota=slot_cpu, io_quota=slot_io
                                        )
                                    except BaseException as exc:
                                        if refill_compile_worker is not None:
                                            try:
                                                refill_compile_worker.close()
                                            except BaseException as cleanup_exc:
                                                exc.cleanup_errors = list(
                                                    getattr(exc, "cleanup_errors", [])
                                                ) + [cleanup_exc]
                                        raise
                                else:
                                    next_compile = compile_worker.execute(
                                        [factor for _, factor in next_admitted],
                                        timeout_seconds=_stage_timeout(
                                            job_deadline, policy.compute_unknown_seconds,
                                            "next-slot compile",
                                        ),
                                    ).value
                                _validate_compile_results(
                                    next_compile, {factor.name for _, factor in next_admitted}
                                )
                                if all(value is None for value in next_compile.values()):
                                    next_assignments = {}
                                    for next_ordinal, next_factor in next_admitted:
                                        if state.consume_attempt(
                                                next_ordinal, "execution") is None:
                                            raise RuntimeError(
                                                "prefetched execution lost its attempt authority"
                                            )
                                        next_generation = uuid.uuid4().hex
                                        state.record_commit_intent(next_ordinal, next_generation)
                                        next_assignments[next_factor.name] = (
                                            next_ordinal, next_generation
                                        )
                                    next_callable = _SpawnFactoryRuntime(
                                        engine_factory, engine_factory_config,
                                        spare_compute_proxy, "compute",
                                    )
                                    spare_compute_worker = SupervisedReusableWorker(
                                        **ownership_kwargs("compute-spare"),
                                        cancel_grace_seconds=float(
                                            policy.cooperative_cancel_grace_seconds
                                        ),
                                        exit_observation_seconds=float(
                                            policy.worker_exit_observation_seconds
                                        ),
                                        context="spawn", function=next_callable,
                                        process_environment=_worker_thread_environment(
                                            spare_compute_proxy
                                        ),
                                    )
                                    spare_compute_worker.start()
                                    broker_ipc.bind_client_process(
                                        spare_compute_proxy.client_id,
                                        spare_compute_worker._process.pid,
                                    )
                                    pending_next = _DirectSlot(
                                        cursor=next_cursor, wave=next_wave,
                                        admitted=next_admitted,
                                        names=[factor.name for _, factor in next_admitted],
                                        assignments=next_assignments,
                                        evidence_id=uuid.uuid4().hex,
                                        worker=spare_compute_worker,
                                        proxy=spare_compute_proxy,
                                    )
                        if current_handle is None:
                            wave_dispatched = True
                            current_handle = compute_worker.execute_async(
                                [f for _, f in admitted], run_kwargs, queue_bytes,
                                {
                                    "root": root, "run_id": run_id, "policy": policy,
                                    "assignments": direct_context.assignments,
                                    "writer_bytes": writer_bytes,
                                },
                                current_evidence_id,
                                timeout_seconds=_stage_timeout(
                                    job_deadline,
                                    _policy_value(policy, "compute_unknown_seconds", 900),
                                    "direct compute/write",
                                ),
                                expected_progress_ordinals=frozenset(
                                    ordinal for ordinal, _factor in admitted
                                ),
                            )
                        if pending_next is not None:
                            next_handle = spare_compute_worker.execute_async(
                                        [factor for _, factor in next_admitted],
                                        run_kwargs, queue_bytes,
                                        {
                                            "root": root, "run_id": run_id,
                                            "policy": policy,
                                            "assignments": next_assignments,
                                            "writer_bytes": writer_bytes,
                                        },
                                        pending_next.evidence_id,
                                        timeout_seconds=_stage_timeout(
                                            job_deadline, policy.compute_unknown_seconds,
                                            "next-slot direct compute/write",
                                        ),
                                        expected_progress_ordinals=frozenset(
                                            ordinal for ordinal, _factor in next_admitted
                                        ),
                                    )
                            pending_next["handle"] = next_handle
                            execution_batches += 1
                            prefetched_direct = pending_next
                        completed_slot = _DirectSlot(
                            cursor=cursor, wave=wave, admitted=admitted,
                            names=[factor.name for _, factor in admitted],
                            assignments=direct_context.assignments,
                            evidence_id=current_evidence_id,
                            worker=compute_worker, proxy=compute_proxy,
                            handle=current_handle, phase="RUNNING",
                        )
                        active_slots = [completed_slot]
                        if pending_next is not None:
                            pending_next.phase = "RUNNING"
                            active_slots.append(pending_next)
                        claim_cursor = max(
                            slot.cursor + len(slot.wave) for slot in active_slots
                        )
                        while completed_slot.phase != "VERIFIED":
                            winner, worker_result = _await_any_direct_slot(
                                active_slots, cancellation_token
                            )
                            try:
                                _complete_direct_slot(
                                    winner, worker_result.value, run_id=run_id,
                                    policy=policy, state=state,
                                    error_groups=error_groups,
                                )
                            except BaseException as slot_failure:
                                slot_failure.failing_slot = winner
                                for live_slot in active_slots:
                                    if live_slot is winner:
                                        continue
                                    try:
                                        live_slot.worker.cancel_grace_seconds = 0.0
                                        live_slot.handle.cancel_and_retire()
                                    except BaseException as cleanup_exc:
                                        slot_failure.cleanup_errors = list(getattr(
                                            slot_failure, "cleanup_errors", []
                                        )) + [cleanup_exc]
                                try:
                                    winner.worker.close()
                                    broker_ipc.reclaim_client(winner.proxy.client_id)
                                    if winner.worker is spare_compute_worker:
                                        spare_compute_worker = None
                                        spare_compute_proxy = None
                                except BaseException as cleanup_exc:
                                    slot_failure.cleanup_errors = list(getattr(
                                        slot_failure, "cleanup_errors", []
                                    )) + [cleanup_exc]
                                slot_failure.direct_slot_fatal = True
                                raise
                            if winner is completed_slot:
                                break
                            active_slots.remove(winner)
                            winner.worker.close()
                            broker_ipc.reclaim_client(winner.proxy.client_id)
                            winner.phase = "RETIRED"
                            if winner.worker is spare_compute_worker:
                                spare_compute_worker = None
                                spare_compute_proxy = None
                            prefetched_direct = None
                            # Drain every completion already observable before
                            # claiming more work. A fatal older slot therefore
                            # closes the admission gate even when a newer slot
                            # won the preceding poll.
                            if any(slot.handle.done for slot in active_slots):
                                continue
                            try:
                                _check_cancellation(cancellation_token)
                            except BaseException:
                                for live_slot in active_slots:
                                    live_slot.handle.cancel_and_retire()
                                raise
                            if claim_cursor >= len(manifest):
                                continue
                            refill_admitted = []
                            while claim_cursor < len(manifest) and not refill_admitted:
                                refill_cursor = claim_cursor
                                refill_wave = manifest.load_wave_bounded(
                                    start=claim_cursor, max_items=max(1, lookahead),
                                    max_definition_bytes=metadata_bytes,
                                )
                                for (refill_ordinal, refill_factor, refill_error,
                                     refill_name) in refill_wave:
                                    if not resume_admission_allowed(refill_ordinal):
                                        continue
                                    refill_factor, refill_scope_error = _apply_execution_scope(
                                        refill_factor, execution_scope
                                    )
                                    rejection = refill_error or refill_scope_error
                                    if rejection is not None:
                                        state.terminal(
                                            refill_ordinal, "REJECTED",
                                            error_code=rejection,
                                            detail=("manifest/scope rejected ordinal "
                                                    f"{refill_ordinal}"),
                                            retryable=False,
                                        )
                                        group = error_groups.setdefault(
                                            rejection, {"count": 0, "examples": []}
                                        )
                                        group["count"] += 1
                                        if len(group["examples"]) < policy.max_examples_per_error_group:
                                            group["examples"].append({
                                                "ordinal": refill_ordinal,
                                                "name": refill_name,
                                            })
                                        continue
                                    refill_admitted.append((refill_ordinal, refill_factor))
                                claim_cursor += len(refill_wave)
                            if not refill_admitted:
                                continue
                            slot_cpu = winner.proxy.cpu_quota
                            slot_io = winner.proxy.io_quota
                            refill_compile_handle = compile_worker.execute_async(
                                [factor for _, factor in refill_admitted],
                                timeout_seconds=_stage_timeout(
                                    job_deadline, policy.compute_unknown_seconds,
                                    "await-any refill compile",
                                ),
                            )
                            refill_compile_result = _await_worker_call(
                                refill_compile_handle, cancellation_token,
                                *(slot.handle for slot in active_slots),
                            ).value
                            _validate_compile_results(
                                refill_compile_result,
                                {factor.name for _, factor in refill_admitted},
                            )
                            if not all(
                                value is None for value in refill_compile_result.values()
                            ):
                                for refill_ordinal, refill_factor in refill_admitted:
                                    state.terminal(
                                        refill_ordinal, "REJECTED",
                                        error_code="COMPILE_REJECTED",
                                        detail=str(refill_compile_result[refill_factor.name]),
                                        retryable=False,
                                    )
                                continue
                            refill_assignments = {}
                            for refill_ordinal, refill_factor in refill_admitted:
                                if state.consume_attempt(
                                        refill_ordinal, "execution") is None:
                                    raise RuntimeError(
                                        "refill execution lost its attempt authority"
                                    )
                                refill_generation = uuid.uuid4().hex
                                state.record_commit_intent(
                                    refill_ordinal, refill_generation
                                )
                                refill_assignments[refill_factor.name] = (
                                    refill_ordinal, refill_generation
                                )
                            spare_compute_proxy = broker_ipc.create_proxy(
                                cpu_quota=slot_cpu, io_quota=slot_io
                            )
                            spare_compute_worker = SupervisedReusableWorker(
                                **ownership_kwargs("compute-refill"),
                                context="spawn",
                                function=_SpawnFactoryRuntime(
                                    engine_factory, engine_factory_config,
                                    spare_compute_proxy, "compute",
                                ),
                                cancel_grace_seconds=float(
                                    policy.cooperative_cancel_grace_seconds
                                ),
                                exit_observation_seconds=float(
                                    policy.worker_exit_observation_seconds
                                ),
                                process_environment=_worker_thread_environment(
                                    spare_compute_proxy
                                ),
                            )
                            spare_compute_worker.start()
                            broker_ipc.bind_client_process(
                                spare_compute_proxy.client_id,
                                spare_compute_worker._process.pid,
                            )
                            refill_evidence_id = uuid.uuid4().hex
                            refill_handle = spare_compute_worker.execute_async(
                                [factor for _, factor in refill_admitted],
                                run_kwargs, queue_bytes,
                                {
                                    "root": root, "run_id": run_id,
                                    "policy": policy,
                                    "assignments": refill_assignments,
                                    "writer_bytes": writer_bytes,
                                },
                                refill_evidence_id,
                                timeout_seconds=_stage_timeout(
                                    job_deadline, policy.compute_unknown_seconds,
                                    "await-any refill compute/write",
                                ),
                                expected_progress_ordinals=frozenset(
                                    ordinal for ordinal, _factor in refill_admitted
                                ),
                            )
                            execution_batches += 1
                            refill_slot = _DirectSlot(
                                cursor=refill_cursor, wave=refill_wave,
                                admitted=refill_admitted,
                                names=[factor.name for _, factor in refill_admitted],
                                assignments=refill_assignments,
                                evidence_id=refill_evidence_id,
                                worker=spare_compute_worker,
                                proxy=spare_compute_proxy,
                                handle=refill_handle, phase="RUNNING",
                            )
                            active_slots.append(refill_slot)
                            prefetched_direct = refill_slot
                        direct_slot_completed = True
                        remaining_ahead = [
                            slot.cursor for slot in active_slots
                            if slot is not completed_slot
                        ]
                        cursor_after_direct_slots = (
                            min(remaining_ahead) if remaining_ahead else claim_cursor
                        )
                        receipts, artifact_errors, factor_errors = {}, {}, {}
                        result_blobs, transport_errors = {}, {}
                    else:
                        wave_dispatched = True
                        results = compute_worker.execute(
                            [f for _, f in admitted], run_kwargs, queue_bytes,
                            None, wave_evidence_id,
                            timeout_seconds=_stage_timeout(
                                job_deadline,
                                _policy_value(policy, "compute_unknown_seconds", 900),
                                "compute",
                            ),
                        ).value
                        envelope = pickle.loads(results)
                        (result_blobs, transport_errors, factor_errors,
                         fit_failure_evidence) = _validate_result_envelope(
                            envelope, {f.name for _, f in admitted},
                            expected_evidence_id=wave_evidence_id,
                            expected_run_id=run_id,
                            include_evidence=True,
                        )
                        state.record_fit_failure_evidence(
                            wave_evidence_id,
                            ({"ordinal": ordinal, "name": factor.name}
                             for ordinal, factor in admitted),
                            (None if fit_failure_evidence is None
                             else fit_failure_evidence["snapshot"]),
                            availability=("UNAVAILABLE" if fit_failure_evidence is None
                                          else "OBSERVED"),
                        )
                        del envelope, results
                    wave_decoded = True
                except Exception as exc:
                    from factor_engine.runtime.exceptions import Cancellation, DeadlineExceeded
                    from factor_engine.runtime.supervised_worker import (
                        WorkerQuarantined, WorkerStaleResponse,
                        WorkerTimedOut, WorkerTransportFailed,
                    )
                    failed_slot = getattr(exc, "failing_slot", None)
                    recoverable_retired_current_slot = bool(
                        isinstance(exc, (WorkerTimedOut, WorkerTransportFailed))
                        and failed_slot is not None
                        and direct_artifacts
                        and len(active_slots) == 1
                        and active_slots[0] is failed_slot
                        and failed_slot is completed_slot
                        and failed_slot.worker is compute_worker
                        and failed_slot.proxy is compute_proxy
                        and failed_slot.handle is current_handle
                        and failed_slot.wave is direct_context.wave
                        and failed_slot.admitted is direct_context.admitted
                        and failed_slot.assignments is direct_context.assignments
                        and failed_slot.cursor == direct_context.cursor == cursor
                        and failed_slot.evidence_id == wave_evidence_id
                        and prefetched_direct is None
                        and pending_next is None
                        and not direct_context.has_later_wave
                    )
                    evidence_already_persisted = bool(
                        failed_slot is not None
                        and failed_slot.phase == "EVIDENCE_PERSISTED"
                    )
                    if (wave_dispatched and not wave_decoded
                            and not evidence_already_persisted):
                        unavailable_id = (
                            failed_slot.evidence_id if failed_slot is not None
                            else wave_evidence_id
                        )
                        unavailable_admitted = (
                            failed_slot.admitted if failed_slot is not None else admitted
                        )
                        state.record_fit_failure_evidence(
                            unavailable_id,
                            ({"ordinal": ordinal, "name": factor.name}
                             for ordinal, factor in unavailable_admitted),
                            None, availability="UNAVAILABLE",
                        )
                    if failed_slot is not None and not recoverable_retired_current_slot:
                        for failed_ordinal, failed_factor in failed_slot.admitted:
                            page = state.outcomes_page(start=failed_ordinal, limit=1)
                            current_state = (
                                page[0].state if page and page[0].ordinal == failed_ordinal
                                else None
                            )
                            if current_state in {
                                "SUCCEEDED", "REUSED", "REJECTED", "FAILED",
                                "BLOCKED", "CANCELLED",
                            }:
                                continue
                            state.terminal(
                                failed_ordinal, "FAILED",
                                error_code="WORKER_PROTOCOL_INTEGRITY",
                                detail=f"{type(exc).__name__}: {exc}", retryable=False,
                                commit_state="UNKNOWN",
                            )
                        raise
                    if isinstance(exc, (Cancellation, DeadlineExceeded)):
                        raise
                    if isinstance(
                        exc, (WorkerQuarantined, WorkerStaleResponse,
                              WorkerTimedOut, WorkerTransportFailed, WorkerProtocolError,
                              JobDeadlineExceeded)
                    ):
                        if isinstance(exc, JobDeadlineExceeded):
                            raise
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
                                timeout_stage = getattr(exc, "phases", {}).get(
                                    ordinal, getattr(exc, "stage", "")
                                )
                                timeout_code = (
                                    "WRITE_TIMEOUT" if timeout_stage == "WRITE"
                                    else "COMPUTE_TIMEOUT" if timeout_stage == "COMPUTE"
                                    else "UNKNOWN_COMMIT"
                                )
                                state.terminal(
                                    ordinal, "FAILED", error_code=timeout_code,
                                    detail="descriptor ACK lost before a committed manifest was observable",
                                    retryable=False,
                                    commit_state=("UNKNOWN" if timeout_stage == "WRITE"
                                                  else "NOT_STARTED"),
                                )
                                group = error_groups.setdefault(
                                    timeout_code, {"count": 0, "examples": []}
                                )
                                group["count"] += 1
                                if len(group["examples"]) < policy.max_examples_per_error_group:
                                    group["examples"].append(
                                        {"ordinal": ordinal, "name": factor.name}
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
                                    timeout_seconds=_stage_timeout(
                                        job_deadline, policy.reconcile_seconds,
                                        "direct reconcile",
                                    ),
                                ).value
                                state.terminal(
                                    ordinal, "SUCCEEDED", artifact=delivered,
                                    commit_state="VERIFIED",
                                )
                            except (WorkerQuarantined, JobDeadlineExceeded):
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
                        if direct_artifacts and direct_slot_completed:
                            continue
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
                                    timeout_seconds=_stage_timeout(
                                        job_deadline, policy.sink_flush_seconds, "write"
                                    ),
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
                                    timeout_seconds=_stage_timeout(
                                        job_deadline, policy.sink_flush_seconds, "custom write"
                                    ),
                                    lease=extra_lease,
                                ).value
                            state.terminal(ordinal, "SUCCEEDED", artifact=delivered,
                                           commit_state="VERIFIED")
                        except Exception as exc:
                            from factor_engine.runtime.supervised_worker import (
                                WorkerQuarantined, WorkerStaleResponse,
                                WorkerTimedOut, WorkerTransportFailed,
                            )
                            if isinstance(exc, (WorkerQuarantined, JobDeadlineExceeded)):
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
                                            timeout_seconds=_stage_timeout(
                                                job_deadline, policy.reconcile_seconds,
                                                "write reconcile",
                                            ),
                                            lease=reconcile_lease,
                                        ).value
                                        state.terminal(
                                            ordinal, "SUCCEEDED", artifact=delivered,
                                            commit_state="VERIFIED",
                                        )
                                        continue
                                    except (WorkerQuarantined, JobDeadlineExceeded):
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
                if (direct_artifacts and wave_decoded and prefetched_direct is not None
                        and cancellation_token is not None
                        and (cancellation_token.is_cancelled or cancellation_token.expired)):
                    prefetched_handle = prefetched_direct.get("handle")
                    if prefetched_handle is not None:
                        prefetched_handle.cancel_and_retire()
                    _check_cancellation(cancellation_token)
                if (direct_artifacts and wave_decoded
                        and direct_context.has_later_wave):
                    # A completed native read wave may retain allocator-owned
                    # buffers or broker tokens beyond Python object lifetime.
                    # Observe PID exit first, then reclaim that exact client;
                    # a later wave always gets a fresh proxy and generation.
                    retired_proxy = compute_proxy
                    retired_cpu = getattr(retired_proxy, "cpu_quota", None)
                    retired_io = getattr(retired_proxy, "io_quota", None)
                    compute_worker.close()
                    broker_ipc.reclaim_client(retired_proxy.client_id)
                    if prefetched_direct is not None and spare_compute_worker is not None:
                        compute_worker = spare_compute_worker
                        compute_proxy = spare_compute_proxy
                        spare_compute_worker = None
                        spare_compute_proxy = broker_ipc.create_proxy(
                            cpu_quota=retired_cpu, io_quota=retired_io
                        )
                    else:
                        compute_proxy = broker_ipc.create_proxy(
                            cpu_quota=retired_cpu, io_quota=retired_io
                        )
                        compute_callable = _SpawnFactoryRuntime(
                            engine_factory, engine_factory_config, compute_proxy, "compute"
                        )
                        compute_worker = SupervisedReusableWorker(
                            **ownership_kwargs("compute-replacement"),
                            cancel_grace_seconds=float(
                                _policy_value(policy, "cooperative_cancel_grace_seconds", 5)
                            ),
                            exit_observation_seconds=float(
                                _policy_value(policy, "worker_exit_observation_seconds", 10)
                            ),
                            context="spawn", function=compute_callable,
                            process_environment=_worker_thread_environment(compute_proxy),
                        )
                        compute_worker.start()
                        broker_ipc.bind_client_process(
                            compute_proxy.client_id, compute_worker._process.pid
                        )
            cursor += len(wave)
            if cursor_after_direct_slots is not None:
                cursor = max(cursor, cursor_after_direct_slots)
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
            "run_identity": run_identity,
            "deployment_digest": run_identity.get("deployment_digest"),
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
            "identity_path": str(identity_path),
            "result_index": str(state_path),
            "error_groups": [
                {"code": code, "count": group["count"], "examples": group["examples"]}
                for code, group in list(error_groups.items())[
                    :int(_policy_value(policy, "max_error_groups_in_response", 20))
                ]
            ],
            "automatic_production_publish": False,
            "fit_failure_evidence": _fit_failure_evidence_reference(
                state,
                legacy_unavailable=bool(
                    resuming and not resume_context.fit_failure_evidence_available
                ),
            ),
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
                    cancelled_phase = getattr(exc, "worker_phases", {}).get(
                        outcome.ordinal, "UNKNOWN"
                    )
                    state.terminal(
                        outcome.ordinal, "CANCELLED", error_code="RUN_ABORTED",
                        detail=f"{type(exc).__name__}: {exc}", retryable=False,
                        commit_state=(
                            "UNKNOWN" if intent is not None
                            and cancelled_phase in {"WRITE", "VERIFY_DONE", "UNKNOWN"}
                            else "NOT_STARTED"
                        ),
                    )
                start = page[-1].ordinal + 1
            aborted_receipt = {
                "schema_version": "factor_engine.artifact_receipt.v2",
                "run_id": run_id, "status": "ABORTED", "counts": state.counts(),
                "run_identity": run_identity,
                "deployment_digest": run_identity.get("deployment_digest"),
                "policy_id": _policy_value(policy, "policy_id", None),
                "policy_digest": getattr(policy, "digest", None),
                "requested_factors": len(manifest), "requested_total": len(manifest),
                "input_complete": manifest.input_complete,
                "manifest_path": str(manifest_path), "state_path": str(state_path),
                "identity_path": str(identity_path),
                "automatic_production_publish": False,
                "fit_failure_evidence": _fit_failure_evidence_reference(
                    state,
                    legacy_unavailable=bool(
                        resuming and not resume_context.fit_failure_evidence_available
                    ),
                ),
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
        prior_cleanup_errors = list(
            getattr(active_primary, "cleanup_errors", [])
        ) if active_primary is not None else []
        worker_cleanup_failed = bool(
            active_primary is not None
            and getattr(active_primary, "cleanup_pending", False)
        ) or any(
            getattr(item, "cleanup_pending", False)
            or type(item).__name__ == "WorkerQuarantined"
            for item in prior_cleanup_errors
        )
        for worker, proxy in (
            (compute_worker, compute_proxy), (compile_worker, compile_proxy),
            (spare_compute_worker, spare_compute_proxy),
            (refill_compile_worker, refill_compile_proxy),
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
                worker_cleanup_failed = True
        # Never release protected egress while any worker may still own result
        # memory or writer state. Attach authorities to the cleanup exception so
        # the facade/supervisor can retain them through PID cleanup.
        if not cleanup_errors and not worker_cleanup_failed:
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
        if not worker_cleanup_failed:
            for authority in (state, manifest):
                try:
                    authority.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
                    if getattr(exc, "cleanup_pending", False):
                        worker_cleanup_failed = True
                        break
            if not worker_cleanup_failed and coordinator_lock is not None:
                try:
                    coordinator_lock.release()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if worker_cleanup_failed and active_primary is not None:
            active_primary.cleanup_pending = True
            active_primary.broker = broker
            active_primary.coordinator_lock = coordinator_lock
            active_primary.run_state = state
            active_primary.run_manifest = manifest
            active_primary.broker_ipc = broker_ipc
            active_primary.egress_leases = egress_leases
        if cleanup_errors:
            if active_primary is not None:
                active_primary.cleanup_errors = list(
                    getattr(active_primary, "cleanup_errors", [])
                ) + cleanup_errors
                active_primary.broker_ipc = broker_ipc
                active_primary.egress_leases = egress_leases
                if worker_cleanup_failed:
                    active_primary.cleanup_pending = True
                    active_primary.broker = broker
                    active_primary.coordinator_lock = coordinator_lock
                    active_primary.run_state = state
                    active_primary.run_manifest = manifest
            else:
                failure = cleanup_errors[0]
                failure.broker_ipc = broker_ipc
                failure.egress_leases = egress_leases
                if worker_cleanup_failed:
                    failure.cleanup_pending = True
                    failure.broker = broker
                    failure.coordinator_lock = coordinator_lock
                    failure.run_state = state
                    failure.run_manifest = manifest
                raise failure
