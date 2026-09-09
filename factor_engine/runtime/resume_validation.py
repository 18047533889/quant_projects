"""Bounded, non-deserializing validation of an existing durable run.

This module does not authorize worker lease reclamation or prove coordinator
exclusivity. The caller must hold the run lock and prove old workers exited
before acting on pending intents. No supplied iterable is consumed here.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

from factor_engine.runtime.persistent_run_state import TERMINAL_STATES


class ResumeIdentityError(ValueError):
    pass


class ResumeValidationTimeout(ResumeIdentityError, TimeoutError):
    """Preserve the validation API while exposing an exhausted time budget."""


def _typed_resume_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ResumeIdentityError:
            raise
        except (KeyError, TypeError, OSError, sqlite3.Error, ValueError) as exc:
            raise ResumeIdentityError(f"resume validation failed: {type(exc).__name__}") from exc
    return checked


@dataclass(frozen=True)
class ValidatedResumeContext:
    run_id: str
    run_dir: Path
    manifest_path: Path
    state_path: Path
    factor_count: int
    pending_count: int
    manifest_digest: str
    identity_sha256: str | None = None
    ownership_store: str | None = None
    ownership_coverage: str | None = None
    artifact_revalidation_required: bool = True
    old_worker_exit_proof_required: bool = True
    fit_failure_evidence_available: bool = False


def _read_json(path: Path, *, maximum_bytes=1048576):
    # Validate the opened object, not a pathname checked before opening it.
    # NONBLOCK prevents a concurrently substituted FIFO from hanging resume.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
                raise ResumeIdentityError(f"nonregular or oversized control record: {path.name}")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                payload = stream.read(maximum_bytes + 1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ResumeIdentityError(f"missing, linked or unreadable control record: {path.name}") from exc
    if len(payload) > maximum_bytes:
        raise ResumeIdentityError("control record grew beyond its size bound")
    return _decode_json(payload), hashlib.sha256(payload).hexdigest()


def _decode_json(payload):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ResumeIdentityError("duplicate identity JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(payload, object_pairs_hook=unique_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ResumeIdentityError("nonfinite identity JSON")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeIdentityError("invalid control JSON") from exc
    if type(value) is not dict:
        raise ResumeIdentityError("control record must be an object")
    return value


def _open_readonly(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ResumeIdentityError(f"missing or linked database: {path.name}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    return connection


def _hash_record(digest, record):
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _exact_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _validation_deadline(value, policy):
    if value is None:
        return time.monotonic() + policy.input_ingestion_deadline_seconds
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ResumeIdentityError("validation deadline must be finite")
    if value <= time.monotonic():
        raise ResumeValidationTimeout("resume validation deadline exhausted")
    return value


def _manifest_fingerprint(db, *, max_definition_bytes, max_factors, deadline):
    meta = dict(db.execute("SELECT key,value FROM meta"))
    if meta.get("sealed") != "1" or meta.get("input_complete") != "1":
        raise ResumeIdentityError("resume requires a complete sealed manifest")
    try:
        count = int(meta["count"])
    except (KeyError, ValueError) as exc:
        raise ResumeIdentityError("invalid manifest count") from exc
    if not 0 <= count <= max_factors:
        raise ResumeIdentityError("manifest exceeds factor admission bound")
    digest = hashlib.sha256(b"factor_engine.resume_manifest.v1\0")
    _hash_record(digest, [count, meta.get("input_error", "")])
    seen = 0
    rows = db.execute(
        "SELECT ordinal,name,definition_bytes,definition_digest,valid,error_code,length(definition) "
        "FROM factors ORDER BY ordinal")
    for ordinal, name, size, expected, valid, error, stored_size in rows:
        if time.monotonic() >= deadline:
            raise ResumeValidationTimeout("resume manifest validation deadline exhausted")
        if ordinal != seen or not isinstance(name, str) or not name:
            raise ResumeIdentityError("noncontiguous or malformed manifest ordinal")
        if type(size) is not int or size < 0 or valid not in (0, 1):
            raise ResumeIdentityError("invalid manifest definition metadata")
        if stored_size is not None:
            if size != stored_size or stored_size > max_definition_bytes:
                raise ResumeIdentityError("stored definition violates admission bound")
            actual = hashlib.sha256()
            for offset in range(0, stored_size, 65536):
                if time.monotonic() >= deadline:
                    raise ResumeValidationTimeout("resume definition validation deadline exhausted")
                chunk = db.execute(
                    "SELECT substr(definition,?,65536) FROM factors WHERE ordinal=?",
                    (offset + 1, ordinal)).fetchone()[0]
                actual.update(chunk)
            if actual.hexdigest() != expected:
                raise ResumeIdentityError("manifest definition digest mismatch")
        elif valid or expected:
            raise ResumeIdentityError("missing valid factor definition")
        _hash_record(digest, [ordinal, name, size, expected, valid, error])
        seen += 1
    if seen != count:
        raise ResumeIdentityError("manifest count mismatch")
    for ordinal, dependency in db.execute(
            "SELECT ordinal,dependency_name FROM dependencies ORDER BY ordinal,dependency_name"):
        if time.monotonic() >= deadline:
            raise ResumeValidationTimeout("resume dependency validation deadline exhausted")
        if type(ordinal) is not int or not 0 <= ordinal < count:
            raise ResumeIdentityError("dependency references unknown ordinal")
        _hash_record(digest, [ordinal, dependency])
    return count, digest.hexdigest()


@_typed_resume_errors
def manifest_seal_payload(run_dir, *, policy, deadline=None):
    """Build the record to atomically persist ONCE before first execution.

    Never call this to repair a missing seal during resume: that would approve
    whatever definition bytes happen to be present at restart time.
    """
    directory = Path(run_dir)
    deadline = _validation_deadline(deadline, policy)
    identity, identity_digest = _read_json(directory / "identity.json")
    if (identity.get("run_id") != directory.name
            or identity.get("schema_version") != "factor_engine.run_identity.v1"
            or identity.get("policy_digest") != policy.digest):
        raise ResumeIdentityError("cannot seal a manifest under another run identity")
    with closing(_open_readonly(directory / "manifest.sqlite3")) as manifest:
        count, digest = _manifest_fingerprint(
            manifest, max_definition_bytes=policy.max_definition_bytes,
            max_factors=policy.max_manifest_factors, deadline=deadline)
    return dict(schema_version="factor_engine.manifest_identity.v1",
                run_id=identity["run_id"], identity_sha256=identity_digest,
                manifest_digest=digest, factor_count=count)


@_typed_resume_errors
def validate_resume_context(artifact_root, run_id, *, policy, run_identity, deadline=None):
    """Validate identities only; pending work still requires PID/lock proof."""
    if type(run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ResumeIdentityError("resume_run_id must be a lowercase UUID hex")
    if type(run_identity) is not dict:
        raise ResumeIdentityError("resume requires the exact business identity mapping")
    deadline = _validation_deadline(deadline, policy)
    try:
        root = Path(artifact_root).resolve(strict=True)
    except OSError as exc:
        raise ResumeIdentityError("artifact root is unavailable") from exc
    directory = root / run_id
    if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != root:
        raise ResumeIdentityError("run directory is missing or outside artifact root")
    identity, identity_digest = _read_json(directory / "identity.json")
    if (identity.get("schema_version") != "factor_engine.run_identity.v1"
            or identity.get("run_id") != run_id
            or identity.get("policy_digest") != policy.digest
            or identity.get("policy_id") != policy.policy_id
            or _exact_json(identity.get("run_identity")) != _exact_json(run_identity)
            or identity.get("deployment_digest") != run_identity.get("deployment_digest")):
        raise ResumeIdentityError("resume business identity or policy differs")
    has_store = "ownership_store" in identity
    has_coverage = "ownership_coverage" in identity
    if has_store or has_coverage:
        if (not has_store or not has_coverage
                or identity.get("ownership_store") != "sqlite-v1"
                or identity.get("ownership_coverage")
                != "all-durable-pipeline-os-processes-v1"):
            raise ResumeIdentityError("resume ownership identity markers are invalid")
    receipt_path = directory / "receipt.json"
    persisted_receipt = _read_json(receipt_path)[0] if receipt_path.is_file() else None
    evidence_reference = (
        persisted_receipt.get("fit_failure_evidence")
        if isinstance(persisted_receipt, dict) else None
    )
    reference_base = {
        "schema_version", "coverage", "store", "index", "availability", "next_seq",
    }
    indexed_keys = reference_base | {
        "observed_waves", "unavailable_waves", "truncated_waves",
        "wave_count", "last_seq",
    }
    reference_fixed = bool(
        isinstance(evidence_reference, dict)
        and evidence_reference.get("schema_version")
        == "factor_engine.fit_failure_evidence_index.v1"
        and evidence_reference.get("coverage") == "instrumented_fit_producers_only"
        and evidence_reference.get("store") == "state.sqlite3"
        and evidence_reference.get("index") == "fit_failure_evidence"
    )
    indexed_evidence = bool(
        reference_fixed and set(evidence_reference) == indexed_keys
        and evidence_reference.get("availability") == "indexed"
        and all(type(evidence_reference.get(key)) is int
                and evidence_reference[key] >= 0
                for key in ("observed_waves", "unavailable_waves", "truncated_waves",
                            "wave_count", "last_seq"))
        and (evidence_reference.get("next_seq") is None
             or type(evidence_reference.get("next_seq")) is int
             and evidence_reference["next_seq"] == 0)
    )
    legacy_evidence = bool(
        reference_fixed and set(evidence_reference) == reference_base
        and evidence_reference.get("availability") == "legacy_unavailable"
        and evidence_reference.get("next_seq") is None
    )
    if evidence_reference is not None and not (indexed_evidence or legacy_evidence):
        raise ResumeIdentityError("unknown fit failure evidence receipt reference")
    seal, _ = _read_json(directory / "manifest_identity.json")
    manifest_path, state_path = directory / "manifest.sqlite3", directory / "state.sqlite3"
    with closing(_open_readonly(manifest_path)) as manifest, closing(_open_readonly(state_path)) as state:
        count, digest = _manifest_fingerprint(
            manifest, max_definition_bytes=policy.max_definition_bytes,
            max_factors=policy.max_manifest_factors, deadline=deadline)
        expected_seal = dict(schema_version="factor_engine.manifest_identity.v1", run_id=run_id,
                             identity_sha256=identity_digest, manifest_digest=digest, factor_count=count)
        if _exact_json(seal) != _exact_json(expected_seal):
            raise ResumeIdentityError("sealed manifest identity differs")
        retry = state.execute("SELECT value FROM state_policy WHERE key='max_attempts'").fetchone()
        if retry is None or retry[0] != str(policy.work_item_max_attempts):
            raise ResumeIdentityError("persisted attempt budget differs")
        evidence_table = state.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fit_failure_evidence'"
        ).fetchone() is not None
        evidence_assignment_table = state.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='fit_failure_evidence_assignments'"
        ).fetchone() is not None
        evidence_marker = state.execute(
            "SELECT value FROM state_policy WHERE key='fit_failure_evidence_schema'"
        ).fetchone()
        if evidence_marker not in (None, ("v1",)):
            raise ResumeIdentityError("unknown fit failure evidence state marker")
        evidence_required = evidence_marker == ("v1",)
        if persisted_receipt is not None and indexed_evidence != evidence_required:
            raise ResumeIdentityError("fit failure evidence receipt/state marker differs")
        if evidence_required and (not evidence_table or not evidence_assignment_table):
            raise ResumeIdentityError("fit failure evidence tables are missing")
        if evidence_table and evidence_assignment_table and state.execute(
                "SELECT 1 FROM fit_failure_evidence_assignments AS assignment "
                "LEFT JOIN fit_failure_evidence AS evidence "
                "ON evidence.evidence_id=assignment.evidence_id "
                "WHERE evidence.evidence_id IS NULL LIMIT 1").fetchone() is not None:
            raise ResumeIdentityError("orphan fit failure evidence assignment")
        if evidence_required:
            linked_total = state.execute(
                "SELECT COUNT(*) FROM fit_failure_evidence_assignments"
            ).fetchone()[0]
            declared_total = state.execute(
                "SELECT COALESCE(SUM(assignment_count),0) FROM fit_failure_evidence"
            ).fetchone()[0]
            if linked_total != declared_total:
                raise ResumeIdentityError("fit failure evidence assignment count differs")
        evidence_row_count = 0
        evidence_last_seq = 0
        evidence_observed = 0
        evidence_unavailable = 0
        evidence_truncated = 0
        if evidence_required:
            from factor_engine.runtime.fit_failure_evidence import validate_fit_failure_snapshot

            metadata_rows = state.execute(
                "SELECT seq,evidence_id,scope,assignment_count,assignments_sha256,"
                "availability,evidence_sha256,payload_bytes,"
                "length(CAST(payload_json AS BLOB)) "
                "FROM fit_failure_evidence ORDER BY seq")
            for (seq, evidence_id, scope, assignment_count, assignments_sha256,
                 availability, evidence_sha256, payload_bytes,
                 stored_payload_bytes) in metadata_rows:
                if time.monotonic() >= deadline:
                    raise ResumeIdentityError("fit failure evidence validation deadline exhausted")
                if (type(seq) is not int or seq < 1 or type(evidence_id) is not str
                        or re.fullmatch(r"[0-9a-f]{32}", evidence_id) is None
                        or scope != "wave_sample" or type(assignment_count) is not int
                        or not 1 <= assignment_count <= count
                        or type(assignments_sha256) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", assignments_sha256) is None
                        or type(evidence_sha256) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None):
                    raise ResumeIdentityError("invalid fit failure evidence identity")
                if availability == "UNAVAILABLE":
                    if stored_payload_bytes is not None or payload_bytes != 0:
                        raise ResumeIdentityError("unavailable evidence contains counts")
                elif availability == "OBSERVED":
                    if (type(stored_payload_bytes) is not int
                            or not 1 <= stored_payload_bytes <= 262144
                            or type(payload_bytes) is not int
                            or payload_bytes != stored_payload_bytes):
                        raise ResumeIdentityError("fit failure evidence payload integrity failed")
                else:
                    raise ResumeIdentityError("unknown fit failure evidence availability")
                payload_json = state.execute(
                    "SELECT payload_json FROM fit_failure_evidence WHERE seq=?",
                    (seq,),
                ).fetchone()[0]
                assignment_digest = hashlib.sha256(
                    b"factor_engine.fit_failure_assignments.v1\0")
                linked_count = 0
                for (ordinal,) in state.execute(
                    "SELECT ordinal FROM fit_failure_evidence_assignments "
                    "WHERE evidence_id=? ORDER BY ordinal", (evidence_id,)):
                    if time.monotonic() >= deadline:
                        raise ResumeIdentityError(
                            "fit failure assignment validation deadline exhausted")
                    if type(ordinal) is not int or not 0 <= ordinal <= 9223372036854775807:
                        raise ResumeIdentityError("invalid fit failure evidence ordinal")
                    manifest_name = manifest.execute(
                        "SELECT name FROM factors WHERE ordinal=?", (ordinal,)
                    ).fetchone()
                    if manifest_name is None or type(manifest_name[0]) is not str:
                        raise ResumeIdentityError(
                            "fit failure evidence assignment differs from manifest")
                    encoded_name = manifest_name[0].encode("utf-8")
                    assignment_digest.update(ordinal.to_bytes(8, "big"))
                    assignment_digest.update(len(encoded_name).to_bytes(8, "big"))
                    assignment_digest.update(encoded_name)
                    linked_count += 1
                if (linked_count != assignment_count
                        or assignment_digest.hexdigest() != assignments_sha256):
                    raise ResumeIdentityError("fit failure evidence assignments differ")
                if availability == "UNAVAILABLE":
                    if payload_json is not None:
                        raise ResumeIdentityError("unavailable evidence contains counts")
                    snapshot = None
                elif availability == "OBSERVED":
                    if (type(payload_json) is not str
                            or len(payload_json.encode("utf-8")) != payload_bytes
                            ):
                        raise ResumeIdentityError("fit failure evidence payload integrity failed")
                    snapshot = validate_fit_failure_snapshot(_decode_json(payload_json.encode()))
                digest_payload = json.dumps({
                    "evidence_id": evidence_id, "scope": scope,
                    "assignment_count": assignment_count,
                    "assignments_sha256": assignments_sha256,
                    "availability": availability,
                    "snapshot": snapshot,
                }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
                if hashlib.sha256(digest_payload).hexdigest() != evidence_sha256:
                    raise ResumeIdentityError("fit failure evidence digest mismatch")
                evidence_row_count += 1
                evidence_last_seq = seq
                evidence_observed += availability == "OBSERVED"
                evidence_unavailable += availability == "UNAVAILABLE"
                evidence_truncated += bool(snapshot and snapshot["truncated"])
        if not evidence_required:
            stray_count = (state.execute(
                "SELECT COUNT(*) FROM fit_failure_evidence"
            ).fetchone()[0] if evidence_table else 0)
            stray_assignments = (state.execute(
                "SELECT COUNT(*) FROM fit_failure_evidence_assignments"
            ).fetchone()[0] if evidence_assignment_table else 0)
            if stray_count or stray_assignments:
                raise ResumeIdentityError("legacy receipt unexpectedly has fit failure evidence")
            evidence_table = False
        elif indexed_evidence:
            expected_summary = {
                "wave_count": evidence_row_count,
                "last_seq": evidence_last_seq,
                "observed_waves": evidence_observed,
                "unavailable_waves": evidence_unavailable,
                "truncated_waves": evidence_truncated,
                "next_seq": 0 if evidence_last_seq else None,
            }
            if any(evidence_reference.get(key) != value
                   for key, value in expected_summary.items()):
                raise ResumeIdentityError("fit failure evidence index differs from receipt")
        completed = 0
        for ordinal, name, status, attempts, generation, commit_state, artifact_size in state.execute(
                "SELECT ordinal,name,state,attempts,artifact_generation,commit_state,length(artifact_json) FROM outcomes ORDER BY ordinal"):
            if time.monotonic() >= deadline:
                raise ResumeIdentityError("resume state validation deadline exhausted")
            expected = manifest.execute("SELECT name FROM factors WHERE ordinal=?", (ordinal,)).fetchone()
            if expected is None or expected[0] != name:
                raise ResumeIdentityError("state ordinal differs from sealed manifest")
            if type(attempts) is not int or not 0 <= attempts <= policy.work_item_max_attempts:
                raise ResumeIdentityError("invalid persisted attempt count")
            if status not in TERMINAL_STATES | {"ACCEPTED", "RUNNING"}:
                raise ResumeIdentityError("unknown persisted outcome state")
            if commit_state not in {"NOT_STARTED", "INTENT", "UNKNOWN", "VERIFIED"}:
                raise ResumeIdentityError("unknown persisted commit state")
            if status == "ACCEPTED" and (attempts != 0 or generation is not None or commit_state != "NOT_STARTED"):
                raise ResumeIdentityError("accepted state has execution history")
            if status in {"RUNNING", "SUCCEEDED"} and attempts == 0:
                raise ResumeIdentityError("running or succeeded state requires an attempt")
            if generation is not None and (type(generation) is not str or re.fullmatch(r"[0-9a-f]{32}", generation) is None):
                raise ResumeIdentityError("invalid persisted commit generation")
            if commit_state in {"INTENT", "UNKNOWN"} and generation is None:
                raise ResumeIdentityError("commit intent lost its generation")
            if status in {"SUCCEEDED", "REUSED"}:
                if commit_state != "VERIFIED" or generation is None or artifact_size is None:
                    raise ResumeIdentityError("successful state lacks verified artifact identity")
                if artifact_size > 1048576:
                    raise ResumeIdentityError("oversized persisted artifact receipt")
                artifact_json = state.execute("SELECT artifact_json FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()[0]
                artifact = _decode_json(artifact_json)
                if (artifact.get("committed") is not True or artifact.get("verified") is not True
                        or artifact.get("generation") != generation):
                    raise ResumeIdentityError("successful artifact differs from commit intent")
            elif commit_state == "VERIFIED":
                raise ResumeIdentityError("non-success state claims verified commitment")
            completed += status in TERMINAL_STATES and commit_state not in {"INTENT", "UNKNOWN"}
    return ValidatedResumeContext(
        run_id, directory, manifest_path, state_path, count, count - completed, digest,
        identity_sha256=identity_digest,
        ownership_store=identity.get("ownership_store"),
        ownership_coverage=identity.get("ownership_coverage"),
        fit_failure_evidence_available=evidence_table,
    )


def iter_resume_pending(context, *, deadline=None):
    """Stream pending intents; caller still owes coordinator/PID ownership proof.

    No worker identity exists in the legacy state schema, so this iterator must
    never be interpreted as evidence that an old process has exited.
    """
    with closing(_open_readonly(context.state_path)) as state, closing(_open_readonly(context.manifest_path)) as manifest:
        for ordinal, name in manifest.execute("SELECT ordinal,name FROM factors ORDER BY ordinal"):
            if deadline is not None and time.monotonic() >= deadline:
                raise ResumeIdentityError("resume pending iteration deadline exhausted")
            row = state.execute(
                "SELECT state,attempts,artifact_generation,commit_state FROM outcomes WHERE ordinal=?",
                (ordinal,)).fetchone()
            status, attempts, generation, commit_state = row or ("UNREGISTERED", 0, None, "NOT_STARTED")
            if status in TERMINAL_STATES and commit_state not in {"INTENT", "UNKNOWN"}:
                continue
            yield dict(ordinal=ordinal, name=name, state=status, attempts=attempts,
                       generation=generation, commit_state=commit_state,
                       worker_exit_proof="REQUIRED_NOT_AVAILABLE_IN_LEGACY_STATE")


def iter_resume_artifacts_to_revalidate(context, *, deadline=None):
    """Every prior success requires exact on-disk verification under the lock."""
    with closing(_open_readonly(context.state_path)) as state:
        for ordinal, name, generation, artifact_size, commit_state in state.execute(
                "SELECT ordinal,name,artifact_generation,length(artifact_json),commit_state FROM outcomes "
                "WHERE state IN ('SUCCEEDED','REUSED') ORDER BY ordinal"):
            if deadline is not None and time.monotonic() >= deadline:
                raise ResumeIdentityError("resume artifact iteration deadline exhausted")
            if commit_state != "VERIFIED":
                raise ResumeIdentityError("successful outcome no longer has verified commitment")
            if artifact_size is None or artifact_size > 1048576:
                raise ResumeIdentityError("missing or oversized persisted artifact receipt")
            artifact_json = state.execute("SELECT artifact_json FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()[0]
            yield dict(ordinal=ordinal, name=name, generation=generation,
                       artifact=_decode_json(artifact_json), action="VERIFY_EXACT_ARTIFACT")


@_typed_resume_errors
def verify_previously_committed_artifact(context, obligation, *, policy, budget_bytes, deadline):
    """Verify unchanged bytes of a PREVIOUSLY VERIFIED receipt, not new math.

    This must never be used to approve a pending manifest lacking a prior
    verified receipt hash. The caller holds admitted workspace and supervises
    I/O with a killable deadline, in addition to the checks between file reads.
    """
    deadline = _validation_deadline(deadline, policy)
    if type(budget_bytes) is not int or budget_bytes < 65536:
        raise ResumeIdentityError("insufficient admitted artifact verification workspace")
    if type(obligation) is not dict:
        raise ResumeIdentityError("invalid artifact verification obligation")
    ordinal, name, generation = (obligation[key] for key in ("ordinal", "name", "generation"))
    receipt = obligation["artifact"]
    if (obligation.get("action") != "VERIFY_EXACT_ARTIFACT"
            or type(ordinal) is not int or ordinal < 0
            or type(generation) is not str or re.fullmatch(r"[0-9a-f]{32}", generation) is None
            or type(receipt) is not dict
            or type(receipt.get("path")) is not str
            or receipt.get("committed") is not True or receipt.get("verified") is not True
            or receipt.get("generation") != generation):
        raise ResumeIdentityError("artifact lacks a prior verified commitment")
    # The dictionary is not a capability: rebind to persisted prior success so
    # an existing pending manifest cannot manufacture its own trusted receipt.
    with closing(_open_readonly(context.state_path)) as state, closing(_open_readonly(context.manifest_path)) as definitions:
        row = state.execute(
            "SELECT name,state,artifact_generation,commit_state,length(artifact_json) FROM outcomes WHERE ordinal=?",
            (ordinal,)).fetchone()
        factor = definitions.execute("SELECT name FROM factors WHERE ordinal=?", (ordinal,)).fetchone()
        if (row is None or factor is None or factor[0] != name or row[0] != name
                or row[1] not in {"SUCCEEDED", "REUSED"} or row[2] != generation
                or row[3] != "VERIFIED" or row[4] is None or row[4] > 1048576):
            raise ResumeIdentityError("artifact obligation lacks persisted verified authority")
        stored = state.execute("SELECT artifact_json FROM outcomes WHERE ordinal=?", (ordinal,)).fetchone()[0]
        if _exact_json(_decode_json(stored)) != _exact_json(receipt):
            raise ResumeIdentityError("artifact obligation differs from persisted receipt")
    directory = context.run_dir / "values" / f"{ordinal:08d}-{generation}"
    path = directory / "manifest.json"
    if (directory.is_symlink() or directory.parent.is_symlink()
            or directory.resolve().parent != (context.run_dir / "values").resolve()
            or Path(receipt.get("path", "")).resolve() != path.resolve()):
        raise ResumeIdentityError("artifact receipt path differs from its exact generation")
    manifest, manifest_digest = _read_json(path, maximum_bytes=min(16 * 1024 * 1024, budget_bytes // 4))
    if manifest_digest != receipt.get("sha256"):
        raise ResumeIdentityError("previously verified artifact manifest hash changed")
    expected = dict(schema_version="factor_engine.factor_artifact.v2", run_id=context.run_id,
                    ordinal=ordinal, factor_id=name, policy_digest=policy.digest, generation=generation)
    if (any(_exact_json(manifest.get(key)) != _exact_json(value) for key, value in expected.items())
            or manifest.get("committed") is not True
            or manifest.get("production_published") is not False):
        raise ResumeIdentityError("artifact manifest identity differs from persisted outcome")
    chunks = manifest.get("chunks")
    if type(chunks) is not list or type(manifest.get("rows")) is not int or manifest["rows"] < 0:
        raise ResumeIdentityError("invalid artifact coverage declaration")
    offset = total_bytes = 0
    seen = set()
    for chunk in chunks:
        if time.monotonic() >= deadline:
            raise ResumeIdentityError("artifact verification deadline exhausted")
        if type(chunk) is not dict:
            raise ResumeIdentityError("invalid artifact chunk")
        filename, rows, size = (chunk.get(key) for key in ("path", "rows", "bytes"))
        if (type(filename) is not str or Path(filename).name != filename or filename in {".", ".."}
                or filename in seen or type(rows) is not int or rows <= 0
                or type(size) is not int or not 0 < size <= policy.max_file_bytes
                or type(chunk.get("offset")) is not int or chunk["offset"] != offset):
            raise ResumeIdentityError("invalid or noncontiguous artifact chunk identity")
        seen.add(filename)
        chunk_path = directory / filename
        if chunk_path.is_symlink() or not chunk_path.is_file() or chunk_path.stat().st_size != size:
            raise ResumeIdentityError("artifact chunk path or size changed")
        digest = hashlib.sha256()
        with chunk_path.open("rb") as stream:
            while True:
                if time.monotonic() >= deadline:
                    raise ResumeIdentityError("artifact file verification deadline exhausted")
                block = stream.read(min(1048576, budget_bytes // 16))
                if not block:
                    break
                digest.update(block)
        if digest.hexdigest() != chunk.get("sha256"):
            raise ResumeIdentityError("artifact chunk hash changed")
        offset += rows
        total_bytes += size
    if (offset != manifest["rows"] or type(receipt.get("rows")) is not int
            or receipt["rows"] != offset or type(receipt.get("bytes")) is not int
            or receipt["bytes"] != total_bytes):
        raise ResumeIdentityError("artifact receipt rows or bytes differ")
    return dict(receipt, resume_bytes_reverified=True)
