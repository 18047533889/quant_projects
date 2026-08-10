"""R21-055..060 + R21-070..079: durable job store and lifecycle.

- ``JobStatus`` extended to the full lifecycle (submitted/queued/running/
  cancelling/cancelled/timed_out/rejected/interrupted/succeeded/failed).
- ``JobRecord`` carries deadline_at, timeout_seconds, cancel_requested_at,
  attempt, worker_id, heartbeat_at, phase, owner principal/tenant/project.
- Manifests are written atomically (tmp + fsync + replace + checksum +
  schema_version); a corrupt manifest is quarantined, not silently skipped.
- Startup reconciliation marks stale non-terminal jobs ``INTERRUPTED`` so a
  restart never leaves a phantom ``running`` (R21-073).
- Multi-worker detection: production refuses ``workers > 1`` with the default
  process-local store (R21-078).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from service.errors import ServiceError

MANIFEST_SCHEMA_VERSION = 3

_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "timed_out", "rejected", "interrupted"}
_NONTERMINAL = {"submitted", "queued", "running", "cancelling"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monotonic_ms() -> float:
    return time.monotonic()


class JobStatus:
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobPhase:
    VALIDATING = "VALIDATING"
    COMPILING = "COMPILING"
    PREFLIGHT = "PREFLIGHT"
    READING = "READING"
    EXECUTING = "EXECUTING"
    DQ = "DQ"
    MATERIALIZING = "MATERIALIZING"
    PUBLISHING = "PUBLISHING"
    FINALIZING = "FINALIZING"


@dataclass
class JobRecord:
    run_id: str
    service: str = "factor_engine"
    requested_by: str = "anonymous"
    owner_principal: str = "anonymous"
    tenant: str | None = None
    project: str | None = None
    job_type: str = "compute"
    endpoint_policy: str = "research"
    idempotency_key: Optional[str] = None
    request_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = JobStatus.SUBMITTED
    submitted_at: str = field(default_factory=_utc_now)
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    deadline_at: Optional[str] = None
    deadline_monotonic: Optional[float] = None  # R21-277: monotonic duration clock
    timeout_seconds: Optional[float] = None
    cancel_requested_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    attempt: int = 1
    worker_id: Optional[str] = None
    phase: Optional[str] = None
    error_code: Optional[str] = None
    error: Optional[str] = None
    error_id: Optional[str] = None
    request: dict[str, Any] = field(default_factory=dict)
    request_digest: Optional[str] = None
    execution_policy_digest: Optional[str] = None
    cost_estimate: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    # --- lifecycle helpers -------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    @property
    def is_nonterminal(self) -> bool:
        return self.status in _NONTERMINAL

    def touch_heartbeat(self) -> None:
        self.heartbeat_at = _utc_now()

    def to_public(self) -> dict[str, Any]:
        from service.errors import sanitize_message

        public_artifacts = {
            key: value
            for key, value in self.artifacts.items()
            if key not in {"traceback", "manifest_path", "lake_root", "config_path", "secret_redacted"}
        }
        return {
            "run_id": self.run_id,
            "service": self.service,
            "requested_by": self.requested_by,
            "owner_principal": self.owner_principal,
            "tenant": self.tenant,
            "project": self.project,
            "job_type": self.job_type,
            "endpoint_policy": self.endpoint_policy,
            "idempotency_key": self.idempotency_key,
            "request_metadata": self.request_metadata,
            "status": self.status,
            "phase": self.phase,
            "submitted_at": self.submitted_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "deadline_at": self.deadline_at,
            "attempt": self.attempt,
            "error_code": self.error_code,
            # R21-089: credentials are redacted at the public boundary, so an
            # error carrying a password/DSN can never leak to a client even if
            # it was stored raw.
            "error": sanitize_message(self.error) if self.error else None,
            "error_id": self.error_id,
            "cost_estimate": self.cost_estimate,
            "result_summary": self.result_summary,
            "artifacts": public_artifacts,
        }


def _manifest_checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class JobStore:
    """Durable job registry: in-memory + atomic JSON manifests (default) or
    SQLite-backed (``FACTOR_ENGINE_SERVICE_DURABLE_STORE=sqlite``).

    The manifest write is ``tmp -> fsync -> os.replace`` with a checksum; a
    corrupt/partial manifest is moved to a ``quarantine/`` dir and surfaced
    via ``corruption_count`` instead of being silently ignored (R21-070/071).
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(
            root
            or os.environ.get("FACTOR_ENGINE_SERVICE_ROOT")
            or (Path.cwd() / ".factor_engine_service")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_root = self.root / "manifests"
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        (self.root / "quarantine").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_index: dict[str, str] = {}
        self.corruption_count = 0
        self.corruptions: list[str] = []
        self._use_sqlite = (
            os.environ.get("FACTOR_ENGINE_SERVICE_DURABLE_STORE", "").lower() == "sqlite"
        )
        self._sqlite_conn: Any = None
        if self._use_sqlite:
            self._init_sqlite()
        self._restore()

    # -- SQLite backing (R21-076..079) --------------------------------------
    def _init_sqlite(self) -> None:
        import sqlite3

        self._sqlite_path = self.root / "jobs.sqlite3"
        self._sqlite_conn = sqlite3.connect(str(self._sqlite_path), check_same_thread=False)
        self._sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self._sqlite_conn.execute("PRAGMA busy_timeout=5000")
        self._sqlite_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                idempotency_key TEXT,
                owner_principal TEXT,
                job_type TEXT,
                status TEXT,
                submitted_at TEXT,
                UNIQUE(idempotency_key, owner_principal, job_type)
            )
            """
        )
        self._sqlite_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
        )
        self._sqlite_conn.commit()

    # -- restore / reconciliation (R21-070..075) -----------------------------
    def _restore(self) -> None:
        if self._use_sqlite:
            self._restore_from_sqlite()
        else:
            self._restore_from_manifests()
        self._reconcile_stale_running()

    def _restore_from_sqlite(self) -> None:
        cur = self._sqlite_conn.execute("SELECT payload FROM jobs")
        for (payload_text,) in cur.fetchall():
            try:
                raw = json.loads(payload_text)
            except json.JSONDecodeError:
                self._quarantine_text(payload_text, run_id="unknown")
                continue
            job = self._job_from_raw(raw)
            self._jobs[job.run_id] = job
            if job.idempotency_key:
                self._idempotency_index[job.idempotency_key] = job.run_id

    def _restore_from_manifests(self) -> None:
        for path in sorted(self.manifest_root.glob("*.json")):
            try:
                raw_text = path.read_text(encoding="utf-8")
                payload = json.loads(raw_text)
            except (OSError, ValueError, json.JSONDecodeError):
                self._quarantine_path(path)
                continue
            checksum = payload.get("_checksum")
            body = dict(payload)
            body.pop("_checksum", None)
            if checksum and _manifest_checksum(body) != checksum:
                self._quarantine_path(path)
                continue
            try:
                job = self._job_from_raw(payload)
            except (ValueError, TypeError, KeyError):
                self._quarantine_path(path)
                continue
            self._jobs[job.run_id] = job
            if job.idempotency_key:
                self._idempotency_index[job.idempotency_key] = job.run_id

    def _job_from_raw(self, raw: dict[str, Any]) -> JobRecord:
        if int(raw.get("schema_version", MANIFEST_SCHEMA_VERSION)) > MANIFEST_SCHEMA_VERSION:
            raise ValueError("manifest schema version newer than supported")
        return JobRecord(
            run_id=str(raw["run_id"]),
            service=str(raw.get("service") or "factor_engine"),
            requested_by=str(raw.get("requested_by") or "anonymous"),
            owner_principal=str(raw.get("owner_principal") or raw.get("requested_by") or "anonymous"),
            tenant=raw.get("tenant"),
            project=raw.get("project"),
            job_type=str(raw.get("job_type") or "compute"),
            endpoint_policy=str(raw.get("endpoint_policy") or "research"),
            idempotency_key=raw.get("idempotency_key"),
            request_metadata=dict(raw.get("request_metadata") or {}),
            status=str(raw.get("status") or JobStatus.FAILED),
            submitted_at=str(raw.get("submitted_at") or _utc_now()),
            queued_at=raw.get("queued_at"),
            started_at=raw.get("started_at"),
            finished_at=raw.get("finished_at"),
            deadline_at=raw.get("deadline_at"),
            deadline_monotonic=raw.get("deadline_monotonic"),
            timeout_seconds=raw.get("timeout_seconds"),
            cancel_requested_at=raw.get("cancel_requested_at"),
            heartbeat_at=raw.get("heartbeat_at"),
            attempt=int(raw.get("attempt") or 1),
            worker_id=raw.get("worker_id"),
            phase=raw.get("phase"),
            error_code=raw.get("error_code"),
            error=raw.get("error"),
            error_id=raw.get("error_id"),
            request=dict(raw.get("request") or {}),
            request_digest=raw.get("request_digest"),
            execution_policy_digest=raw.get("execution_policy_digest"),
            cost_estimate=dict(raw.get("cost_estimate") or {}),
            result_summary=dict(raw.get("result_summary") or {}),
            artifacts=dict(raw.get("artifacts") or {}),
        )

    def _reconcile_stale_running(self) -> None:
        """R21-073/074: a restart must never leave a phantom ``running`` job."""
        for job in list(self._jobs.values()):
            if job.is_nonterminal:
                job.status = JobStatus.INTERRUPTED
                job.error_code = "JOB_INTERRUPTED"
                job.error = "job interrupted by service restart (non-terminal at startup)"
                job.finished_at = _utc_now()
                self.update(job, write_manifest=False)

    def _quarantine_path(self, path: Path) -> None:
        self.corruption_count += 1
        self.corruptions.append(str(path))
        try:
            shutil.move(str(path), str(self.root / "quarantine" / f"{path.stem}-corrupt.json"))
        except OSError:
            pass

    def _quarantine_text(self, text: str, *, run_id: str) -> None:
        self.corruption_count += 1
        target = self.root / "quarantine" / f"{run_id}-corrupt.json"
        try:
            target.write_text(text, encoding="utf-8")
        except OSError:
            pass

    # -- CRUD ---------------------------------------------------------------
    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            if job.idempotency_key:
                existing = self._idempotency_index.get(job.idempotency_key)
                if existing and existing in self._jobs:
                    return self._jobs[existing]
                self._idempotency_index[job.idempotency_key] = job.run_id
            self._jobs[job.run_id] = job
            self._persist(job)
        return job

    def get_by_idempotency_key(self, key: str) -> Optional[JobRecord]:
        with self._lock:
            run_id = self._idempotency_index.get(str(key))
            return self._jobs.get(run_id) if run_id else None

    def get(self, run_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(run_id)

    def update(self, job: JobRecord, *, write_manifest: bool = True) -> None:
        with self._lock:
            self._jobs[job.run_id] = job
            if write_manifest:
                self._persist(job)

    def _persist(self, job: JobRecord) -> None:
        if self._use_sqlite:
            payload = self._job_to_raw(job)
            self._sqlite_conn.execute(
                "INSERT OR REPLACE INTO jobs(run_id,payload,idempotency_key,owner_principal,job_type,status,submitted_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    job.run_id,
                    json.dumps(payload, ensure_ascii=False),
                    job.idempotency_key,
                    job.owner_principal,
                    job.job_type,
                    job.status,
                    job.submitted_at,
                ),
            )
            self._sqlite_conn.commit()
        else:
            self._write_manifest(job)

    def _job_to_raw(self, job: JobRecord) -> dict[str, Any]:
        raw = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": job.run_id,
            "service": job.service,
            "requested_by": job.requested_by,
            "owner_principal": job.owner_principal,
            "tenant": job.tenant,
            "project": job.project,
            "job_type": job.job_type,
            "endpoint_policy": job.endpoint_policy,
            "idempotency_key": job.idempotency_key,
            "request_metadata": job.request_metadata,
            "status": job.status,
            "submitted_at": job.submitted_at,
            "queued_at": job.queued_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "deadline_at": job.deadline_at,
            "deadline_monotonic": job.deadline_monotonic,
            "timeout_seconds": job.timeout_seconds,
            "cancel_requested_at": job.cancel_requested_at,
            "heartbeat_at": job.heartbeat_at,
            "attempt": job.attempt,
            "worker_id": job.worker_id,
            "phase": job.phase,
            "error_code": job.error_code,
            "error": job.error,
            "error_id": job.error_id,
            "request": job.request,
            "request_digest": job.request_digest,
            "execution_policy_digest": job.execution_policy_digest,
            "cost_estimate": job.cost_estimate,
            "result_summary": job.result_summary,
            "artifacts": job.artifacts,
        }
        return raw

    def _write_manifest(self, job: JobRecord) -> None:
        raw = self._job_to_raw(job)
        raw["_checksum"] = _manifest_checksum(raw)
        path = self.manifest_root / f"{job.run_id}.json"
        tmp = path.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(raw, ensure_ascii=False, indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(str(tmp), str(path))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        job.artifacts["manifest_path"] = str(path)

    # -- introspection ------------------------------------------------------
    def list_jobs(self, *, status: str | None = None) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def count_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    def close(self) -> None:
        if self._sqlite_conn is not None:
            try:
                self._sqlite_conn.close()
            except Exception:
                pass


def check_single_process_workers() -> None:
    """R21-078: refuse production with the process-local store if workers>1.

    Called from create_app() when the store is not SQLite-backed and uvicorn
    reports multiple workers.
    """
    workers = int(os.environ.get("UVICORN_WORKERS", "") or os.environ.get("WEB_CONCURRENCY", "1"))
    if workers > 1 and os.environ.get("FACTOR_ENGINE_SERVICE_DURABLE_STORE", "").lower() != "sqlite":
        raise ServiceError(
            "INTERNAL_ERROR",
            "process-local JobStore is unsafe with multiple workers; set "
            "FACTOR_ENGINE_SERVICE_DURABLE_STORE=sqlite or run single-process",
            status=500,
        )
