"""Outer E2E-H composition over existing FE/FP and durable publication APIs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from factor_engine.backend.cleaned_bridge import execute_operator_recipe
from factor_engine.ir.analyzer import Analyzer
from factor_engine.runtime.stateful_checkpoint_store import StatefulCheckpointStore
from factor_engine.runtime.stateful_incremental import (
    _source_snapshot_scope,
    segmented_incremental_available,
    try_stateful_segmented_incremental,
)
from factor_engine.storage.catalog import compute_ir_hash
from factor_engine.storage.factory import build_data_source
from factor_preprocess.contracts.treatment_recipe import TreatmentRecipe
from factor_preprocess.adapters.fitted_recipe import apply_frozen_fitted_recipe
from factor_preprocess.contracts.state import FittedState
from factor_engine.stateful_contract import StateCheckpoint
from quant_platform.app.contracts import ArtifactRef, ARTIFACT_TYPE_FACTOR_VALUE


@dataclass(frozen=True)
class FixedUpdateEvidence:
    factor_definition_hash: str
    recipe_hash: str
    snapshot_ref: str
    generation_id: str
    content_hash: str
    bootstrap: bool
    row_count: int
    fitted_state_refs: tuple[str, ...] = ()


def _source_content_hash(source_spec: dict[str, Any]) -> str:
    """Bind a rebuildable Parquet source to its actual immutable input bytes."""
    if str(source_spec.get("type", "")).strip().lower() != "parquet":
        raise ValueError("fixed update requires a content-addressable Parquet source")
    root = Path(str(source_spec.get("root", ""))).expanduser().resolve()
    if root.is_file():
        files = (root,)
        base = root.parent
    elif root.is_dir():
        files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
        base = root
    else:
        raise ValueError("fixed update Parquet source root does not exist")
    if not files:
        raise ValueError("fixed update Parquet source contains no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class _DeferredCheckpointStore:
    """Expose reads immediately while holding one runtime checkpoint batch."""

    def __init__(self, store: StatefulCheckpointStore) -> None:
        self.store = store
        self.pending: tuple[str, list[StateCheckpoint], str | None] | None = None

    def load_latest(self, *args: Any, **kwargs: Any) -> StateCheckpoint | None:
        return self.store.load_latest(*args, **kwargs)

    def commit_batch(
        self, factor_id: str, checkpoints: list[StateCheckpoint],
        *, execution_id: str | None = None,
    ) -> None:
        if self.pending is not None:
            raise RuntimeError("fixed update attempted more than one checkpoint batch")
        self.pending = (factor_id, list(checkpoints), execution_id)


def _pending_path(store: StatefulCheckpointStore, artifact_id: str) -> Path:
    key = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
    return Path(store.root) / ".fixed_update_pending" / f"{key}.json"


def _serialized_update(function):
    """Serialize all update/recovery work sharing one checkpoint root."""
    @wraps(function)
    def locked(*args: Any, **kwargs: Any):
        store = kwargs.get("checkpoint_store")
        if not isinstance(store, StatefulCheckpointStore):
            raise TypeError("checkpoint_store must be supplied by keyword")
        root = Path(store.root)
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / ".fixed_update.lock"
        with lock_path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another fixed update or recovery owns this checkpoint root"
                ) from exc
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return locked


def _artifact_payload(artifact: ArtifactRef) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id, "artifact_type": artifact.artifact_type,
        "schema_version": artifact.schema_version, "semantic_hash": artifact.semantic_hash,
        "content_hash": artifact.content_hash, "storage_uri": artifact.storage_uri,
        "size_bytes": artifact.size_bytes, "created_at": artifact.created_at.isoformat(),
        "producer_type": artifact.producer_type, "producer_version": artifact.producer_version,
        "media_type": artifact.media_type, "producer_source_ref": artifact.producer_source_ref,
        "snapshot_ref": artifact.snapshot_ref, "universe_ref": artifact.universe_ref,
        "security_classification": artifact.security_classification,
    }


def _write_pending(
    path: Path, *, artifact: ArtifactRef, payload: bytes,
    checkpoint_batch: tuple[str, list[StateCheckpoint], str | None],
    evidence: dict[str, Any],
) -> None:
    factor_id, checkpoints, execution_id = checkpoint_batch
    document = {
        "artifact": _artifact_payload(artifact), "payload_hex": payload.hex(),
        "factor_id": factor_id, "execution_id": execution_id,
        "checkpoints": [checkpoint.to_json() for checkpoint in checkpoints],
        "evidence": evidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    encoded = json.dumps(document, sort_keys=True).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _remove_pending(path: Path) -> None:
    path.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@_serialized_update
def recover_fixed_daily_update(
    *, checkpoint_store: StatefulCheckpointStore, generation_coordinator: Any,
    artifact_id: str,
) -> FixedUpdateEvidence:
    """Publish a durably saved segment without rerunning FE or FP execution."""
    path = _pending_path(checkpoint_store, artifact_id)
    if not path.is_file():
        raise RuntimeError("no persisted fixed-update segment is available for recovery")
    document = json.loads(path.read_text(encoding="utf-8"))
    artifact_data = dict(document["artifact"])
    artifact_data["created_at"] = datetime.fromisoformat(artifact_data["created_at"])
    artifact = ArtifactRef(**artifact_data)
    if artifact.artifact_id != artifact_id:
        raise RuntimeError("persisted fixed-update artifact identity mismatch")
    payload = bytes.fromhex(document["payload_hex"])
    if hashlib.sha256(payload).hexdigest() != artifact.content_hash:
        raise RuntimeError("persisted fixed-update payload hash mismatch")
    checkpoints = [StateCheckpoint.from_json(value) for value in document["checkpoints"]]
    checkpoint_store.commit_batch(
        str(document["factor_id"]), checkpoints,
        execution_id=document.get("execution_id"),
    )
    generation_id = generation_coordinator.stage(artifact, payload)
    generation_coordinator.outbox.publish_pending(
        idempotency_key=f"publish:{generation_id}"
    )
    active = generation_coordinator.resolve_active(artifact_id)
    if active is None or active["content_hash"] != artifact.content_hash:
        raise RuntimeError("recovered fixed update was not atomically activated")
    _remove_pending(path)
    evidence = document["evidence"]
    return FixedUpdateEvidence(
        evidence["factor_definition_hash"], evidence["recipe_hash"],
        evidence["snapshot_ref"], generation_id, artifact.content_hash,
        bool(evidence["bootstrap"]), int(evidence["row_count"]),
        tuple(evidence.get("fitted_state_refs", ())),
    )


@_serialized_update
def run_fixed_daily_update(
    *, factor: Any, recipe: TreatmentRecipe, source_spec: dict[str, Any],
    start: Any, end: Any, bootstrap: bool,
    checkpoint_store: StatefulCheckpointStore, generation_coordinator: Any,
    artifact_id: str, fitted_states: Mapping[str, FittedState] | None = None,
) -> FixedUpdateEvidence:
    """Execute and publish exactly one frozen checkpoint-backed update segment."""
    pending_path = _pending_path(checkpoint_store, artifact_id)
    if pending_path.exists():
        raise RuntimeError("a persisted segment requires recover_fixed_daily_update")
    if factor.source_expr is None or not factor.source_expr.strip():
        raise ValueError("fixed update requires the executable DSL source identity")
    analysis = Analyzer().lower(factor.expr)
    factor_definition_hash = compute_ir_hash(analysis.ir)
    if recipe.source_factor_definition_ref != factor_definition_hash:
        raise ValueError("frozen recipe does not match executable factor definition")
    if not segmented_incremental_available(ir=analysis.ir):
        raise ValueError("factor IR is not supported by the checkpointed incremental runtime")
    implementations = tuple(step.implementation_ref for step in recipe.ordered_steps)
    fitted_state_refs: tuple[str, ...] = ()
    if implementations == ("rank",):
        if any(step.requires_fit or step.state_ref for step in recipe.ordered_steps):
            raise ValueError("stateless rank recipe cannot bind fitted state")
    else:
        # Validate every state/cutoff/feature binding before source execution or
        # checkpoint mutation. The adapter has no fitting entry point.
        _, fitted_state_refs = apply_frozen_fitted_recipe(
            pd.DataFrame([[0.0]], index=[pd.Timestamp(start)], columns=["probe"]),
            recipe=recipe, fitted_states=dict(fitted_states or {}),
            factor_definition_hash=factor_definition_hash, decision_start=start,
        )

    spec = dict(source_spec)
    spec["start_date"], spec["end_date"] = str(start), str(end)
    source = build_data_source(spec)
    # ``compute_data_scope`` already treats ``content_hash`` as a supported
    # stable source identity.  ParquetSource itself only identifies its path,
    # so bind this invocation to the concrete bytes before checkpoint lookup.
    source_content_hash = _source_content_hash(source_spec)
    source.content_hash = source_content_hash
    snapshot_ref = _source_snapshot_scope(source, mode="production")
    deferred_store = _DeferredCheckpointStore(checkpoint_store)
    incremental = try_stateful_segmented_incremental(
        factor_id=factor.name, ir=analysis.ir, source=source,
        store=deferred_store, start=start, end=end, bootstrap=bootstrap,
        mode="production",
    )
    if incremental is None:
        raise RuntimeError("checkpointed incremental segment unavailable")
    segment, mode = incremental
    if bool(mode["bootstrap"]) != bool(bootstrap):
        raise RuntimeError("incremental runtime returned the wrong checkpoint mode")
    panel = segment.unstack(level="instrument")
    if implementations == ("rank",):
        steps = tuple((step.implementation_ref, dict(step.parameters))
                      for step in recipe.ordered_steps)
        treated = execute_operator_recipe(panel, steps)
    else:
        treated, applied_refs = apply_frozen_fitted_recipe(
            panel, recipe=recipe, fitted_states=dict(fitted_states or {}),
            factor_definition_hash=factor_definition_hash, decision_start=start,
        )
        if applied_refs != fitted_state_refs:
            raise RuntimeError("fitted state identity changed during execution")
    if _source_content_hash(source_spec) != source_content_hash:
        raise RuntimeError("source changed during fixed update; refusing checkpoint and publication")
    payload = json.dumps({
        "factor_definition_hash": factor_definition_hash,
        "recipe_hash": recipe.content_hash,
        "snapshot_ref": snapshot_ref,
        "dates": [value.isoformat() for value in treated.index],
        "assets": list(treated.columns),
        "values": treated.to_numpy().tolist(),
        "bootstrap": bool(bootstrap),
        "fitted_state_refs": list(fitted_state_refs),
    }, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    content_hash = hashlib.sha256(payload).hexdigest()
    artifact = ArtifactRef(
        artifact_id=artifact_id, artifact_type=ARTIFACT_TYPE_FACTOR_VALUE,
        schema_version="1.0", semantic_hash=recipe.content_hash,
        content_hash=content_hash, storage_uri=f"memory://{artifact_id}/{content_hash}",
        size_bytes=len(payload), created_at=datetime.now(timezone.utc),
        producer_type="e2e-h-fixed-update", producer_version="1",
        producer_source_ref=factor_definition_hash, snapshot_ref=snapshot_ref,
    )
    if deferred_store.pending is None:
        raise RuntimeError("checkpointed runtime produced no checkpoint batch")
    evidence_data = {
        "factor_definition_hash": factor_definition_hash,
        "recipe_hash": recipe.content_hash, "snapshot_ref": snapshot_ref,
        "bootstrap": bool(bootstrap), "row_count": int(treated.size),
        "fitted_state_refs": list(fitted_state_refs),
    }
    _write_pending(
        pending_path, artifact=artifact, payload=payload,
        checkpoint_batch=deferred_store.pending, evidence=evidence_data,
    )
    factor_id, checkpoints, execution_id = deferred_store.pending
    checkpoint_store.commit_batch(factor_id, checkpoints, execution_id=execution_id)
    generation_id = generation_coordinator.stage(artifact, payload)
    generation_coordinator.outbox.publish_pending(
        idempotency_key=f"publish:{generation_id}"
    )
    active = generation_coordinator.resolve_active(artifact_id)
    if active is None or active["content_hash"] != content_hash:
        raise RuntimeError("fixed update generation was not atomically activated")
    _remove_pending(pending_path)
    return FixedUpdateEvidence(
        factor_definition_hash, recipe.content_hash, snapshot_ref, generation_id,
        content_hash, bool(bootstrap), int(treated.size), fitted_state_refs,
    )


__all__ = ["FixedUpdateEvidence", "recover_fixed_daily_update", "run_fixed_daily_update"]
