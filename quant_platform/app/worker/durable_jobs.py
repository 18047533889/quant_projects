"""Optional durable persistence for the existing JobRunner authority."""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone

from quant_platform.app.contracts.jobs import ErrorClass, JobAttempt, JobResult, JobStatus


def _dt(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


class DurableJobStore:
    """Persist runner snapshots in the canonical jobs tables via the Db seam."""

    def __init__(self, db, *, now_fn=None):
        self.db = db
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _attempt_id(job_id, token):
        # Portable across SQLite/PostgreSQL and independent of backend-specific
        # autoincrement syntax. The token is monotonic within the job; the hash
        # makes the integer globally collision-resistant across jobs.
        digest = hashlib.sha256(f"{job_id}:{token}".encode("utf-8")).hexdigest()
        return int(digest[:15], 16)

    def load(self, idempotency_key):
        rows = self.db.query("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
        if not rows:
            return None
        row = rows[0]
        attempts = self.db.query(
            "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY heartbeat, attempt_id",
            (row["job_id"],)
        )
        result_rows = self.db.query("SELECT * FROM job_results WHERE job_id = ?", (row["job_id"],))
        from quant_platform.app.worker.jobs import JobExecutionRecord
        result = None
        if result_rows:
            result = JobResult(tuple(json.loads(result_rows[0]["output_artifact_refs_json"] or "[]")))
        typed_attempts = tuple(JobAttempt(
            worker_id=a["worker_id"], heartbeat=_dt(a["heartbeat"]),
            error_class=ErrorClass(a["error_class"]) if a["error_class"] else None,
            logs_ref=a["logs_ref"],
        ) for a in attempts)
        error = typed_attempts[-1].error_class if typed_attempts else None
        return JobExecutionRecord(
            job_type=row["job_type"], idempotency_key=row["idempotency_key"],
            status=JobStatus(row["status"]), current_stage=row["current_stage"],
            progress=row["progress"], created_at=_dt(row["created_at"]),
            started_at=_dt(row["started_at"]), finished_at=_dt(row["finished_at"]),
            attempt_count=row["attempt_count"], attempts=typed_attempts,
            error_class=error, result=result,
        )

    def claim(self, spec, worker_id):
        """Atomically claim work and return its monotonic fencing token."""
        job_id = "job:" + spec.idempotency_key
        now = self._now_fn()
        if now.tzinfo is None:
            raise ValueError("DurableJobStore clock must return a timezone-aware datetime")
        with self.db.transaction() as conn:
            inserted = conn.execute(
                "INSERT INTO jobs(job_id,job_type,idempotency_key,status,current_stage,progress,created_at,started_at,attempt_count,timeout_seconds,resource_class) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
                (job_id, spec.job_type, spec.idempotency_key, "RUNNING", "handling", 0.0,
                 _iso(now), _iso(now), 1, spec.timeout_seconds, spec.resource_class),
            )
            if inserted.rowcount == 1:
                token = 1
                conn.execute(
                    "INSERT INTO job_attempts(attempt_id,job_id,worker_id,heartbeat,error_class,logs_ref) VALUES(?,?,?,?,?,?)",
                    (self._attempt_id(job_id, token), job_id, worker_id, _iso(now), None, None),
                )
                return token
            row = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (spec.idempotency_key,)
            ).fetchone()
            if row is None:
                return None
            status = row["status"]
            stale = False
            if status == "RUNNING" and row["timeout_seconds"] and row["started_at"]:
                started = _dt(row["started_at"])
                if started.tzinfo is None: started = started.replace(tzinfo=timezone.utc)
                stale = (now-started).total_seconds() > row["timeout_seconds"]
            if status != "FAILED_RETRYABLE" and not stale:
                return None
            prior_token = int(row["attempt_count"])
            token = prior_token + 1
            claimed = conn.execute(
                "UPDATE jobs SET status='RUNNING',current_stage='handling',started_at=?,"
                "finished_at=NULL,attempt_count=? WHERE job_id=? AND attempt_count=? AND status=?",
                (_iso(now), token, job_id, prior_token, status),
            )
            if claimed.rowcount != 1:
                return None
            conn.execute(
                "INSERT INTO job_attempts(attempt_id,job_id,worker_id,heartbeat,error_class,logs_ref) VALUES(?,?,?,?,?,?)",
                (self._attempt_id(job_id, token), job_id, worker_id, _iso(now), None, None),
            )
            return token

    def save(self, record, *, fencing_token, worker_id):
        """CAS a terminal snapshot; stale owners can never publish results."""
        job_id = "job:" + record.idempotency_key
        with self.db.transaction() as conn:
            saved = conn.execute(
                "UPDATE jobs SET status=?,current_stage=?,progress=?,finished_at=? "
                "WHERE job_id=? AND status='RUNNING' AND attempt_count=? "
                "AND EXISTS (SELECT 1 FROM job_attempts WHERE job_id=? AND attempt_id=? AND worker_id=?)",
                (record.status.value, record.current_stage, record.progress,
                 _iso(record.finished_at), job_id, fencing_token, job_id,
                 self._attempt_id(job_id, fencing_token), worker_id),
            )
            if saved.rowcount != 1:
                return False
            attempt = record.attempts[-1]
            conn.execute(
                "UPDATE job_attempts SET heartbeat=?,error_class=?,logs_ref=? WHERE attempt_id=?",
                (_iso(attempt.heartbeat),
                 attempt.error_class.value if attempt.error_class else None,
                 attempt.logs_ref, self._attempt_id(job_id, fencing_token)),
            )
            if record.result is not None:
                refs = json.dumps(list(record.result.output_artifact_refs), separators=(",", ":"))
                conn.execute(
                    "INSERT INTO job_results(job_id,output_artifact_refs_json) VALUES(?,?) ON CONFLICT(job_id) DO UPDATE SET output_artifact_refs_json=excluded.output_artifact_refs_json",
                    (job_id, refs),
                )
            return True


__all__ = ["DurableJobStore"]
