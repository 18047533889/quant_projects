"""Read-only, bounded planning catalog for a future pending-resume executor."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import math
import re
import time
from typing import Iterator

from factor_engine.runtime.resume_validation import (
    ResumeIdentityError,
    ResumeValidationTimeout,
    ValidatedResumeContext,
    _open_readonly,
)


_TERMINAL_IMMUTABLE = frozenset({"FAILED", "CANCELLED", "BLOCKED", "REJECTED"})
_KNOWN_STATES = frozenset({"ACCEPTED", "RUNNING", "SUCCEEDED", "REUSED"}) | _TERMINAL_IMMUTABLE
_GENERATION = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class ResumeAction:
    ordinal: int
    name: str
    action: str
    persisted_state: str
    attempts: int
    max_attempts: int
    remaining_attempts: int
    generation: str | None
    commit_state: str
    terminal_immutable: bool


def _action_for_row(ordinal, name, row, max_attempts) -> ResumeAction:
    if row is None:
        return ResumeAction(
            ordinal, name, "REGISTER_AND_EXECUTE", "MISSING", 0,
            max_attempts, max_attempts, None, "NOT_STARTED", False,
        )
    stored_name, state, attempts, generation, commit_state = row
    if stored_name != name:
        raise ResumeIdentityError("resume action row differs from manifest identity")
    if type(attempts) is not int or not 0 <= attempts <= max_attempts:
        raise ResumeIdentityError("resume action has invalid persisted attempt count")
    if state not in _KNOWN_STATES:
        raise ResumeIdentityError("resume action has unknown persisted state")
    if commit_state not in {"NOT_STARTED", "INTENT", "UNKNOWN", "VERIFIED"}:
        raise ResumeIdentityError("resume action has unknown commit state")
    if generation is not None and (
        type(generation) is not str or _GENERATION.fullmatch(generation) is None
    ):
        raise ResumeIdentityError("resume action has invalid artifact generation")
    if state == "ACCEPTED" and (
        attempts != 0 or generation is not None or commit_state != "NOT_STARTED"
    ):
        raise ResumeIdentityError("accepted resume action has execution history")
    if state == "RUNNING" and attempts == 0:
        raise ResumeIdentityError("running resume action requires a persisted attempt")
    if state == "SUCCEEDED" and attempts == 0:
        raise ResumeIdentityError("succeeded resume action requires a persisted attempt")
    if commit_state == "NOT_STARTED" and generation is not None:
        raise ResumeIdentityError("uncommitted resume action has artifact generation")
    if commit_state in {"INTENT", "UNKNOWN"} and (
        generation is None or state in {"ACCEPTED", "SUCCEEDED", "REUSED"}
    ):
        raise ResumeIdentityError("persisted state cannot reconcile commit authority")
    if commit_state == "VERIFIED" and state not in {"SUCCEEDED", "REUSED"}:
        raise ResumeIdentityError("non-success state claims verified commitment")
    if state in {"SUCCEEDED", "REUSED"} and commit_state != "VERIFIED":
        raise ResumeIdentityError("successful state lacks verified commitment")
    if commit_state in {"INTENT", "UNKNOWN"}:
        action = "RECONCILE_EXACT_GENERATION"
    elif state in {"SUCCEEDED", "REUSED"}:
        if generation is None:
            raise ResumeIdentityError(
                "resume revalidation lacks a verified artifact generation"
            )
        action = "REVALIDATE_VERIFIED_ARTIFACT"
    elif state in _TERMINAL_IMMUTABLE:
        action = "PRESERVE_TERMINAL"
    elif state == "ACCEPTED":
        action = "EXECUTE_NEW"
    elif state == "RUNNING":
        if generation is not None or commit_state != "NOT_STARTED":
            raise ResumeIdentityError("running resume action is malformed")
        action = (
            "RETRY_WITH_REMAINING_BUDGET"
            if attempts < max_attempts else "NO_EXECUTION_BUDGET"
        )
    terminal_immutable = state in {
        "SUCCEEDED", "REUSED", "FAILED", "CANCELLED", "BLOCKED", "REJECTED",
    }
    return ResumeAction(
        ordinal, name, action, state, attempts, max_attempts,
        max_attempts - attempts, generation, commit_state, terminal_immutable,
    )


def iter_resume_action_catalog(
    context: ValidatedResumeContext, *, policy, restored_job_deadline,
    page_size=512,
) -> Iterator[ResumeAction]:
    """Yield a finite action directory; never mutate or execute persisted work.

    ``restored_job_deadline`` must be the absolute monotonic deadline restored
    from the original durable run identity. This helper never creates a fresh
    timeout window during resume. The caller must already hold the coordinator
    lock and independently prove all old workers exited. A validated context is
    read authority only; neither it nor a yielded action authorizes execution or
    mutation of a terminal outcome.
    """
    if type(context) is not ValidatedResumeContext:
        raise ResumeIdentityError("validated resume context is required")
    if type(page_size) is not int or not 1 <= page_size <= 512:
        raise ValueError("resume action page size must be an integer in [1, 512]")
    if (type(restored_job_deadline) not in (int, float)
            or not math.isfinite(restored_job_deadline)):
        raise ResumeIdentityError("restored job deadline must be finite")
    deadline = restored_job_deadline
    if deadline <= time.monotonic():
        raise ResumeValidationTimeout("restored job deadline exhausted")
    with closing(_open_readonly(context.manifest_path)) as manifest, closing(
        _open_readonly(context.state_path)
    ) as state:
        retry_rows = state.execute(
            "SELECT value FROM state_policy WHERE key='max_attempts' LIMIT 2"
        ).fetchall()
        expected_max = policy.work_item_max_attempts
        if (len(retry_rows) != 1 or retry_rows[0][0] != str(expected_max)
                or type(expected_max) is not int or not 1 <= expected_max <= 3):
            raise ResumeIdentityError("persisted attempt budget differs")
        cursor = manifest.execute(
            "SELECT ordinal,name FROM factors ORDER BY ordinal"
        )
        yielded = 0
        while True:
            if time.monotonic() >= deadline:
                raise ResumeValidationTimeout(
                    "resume action catalog deadline exhausted"
                )
            manifest_rows = cursor.fetchmany(page_size)
            if not manifest_rows:
                break
            ordinals = [row[0] for row in manifest_rows]
            placeholders = ",".join("?" for _ in ordinals)
            outcome_rows = state.execute(
                "SELECT ordinal,name,state,attempts,artifact_generation,commit_state "
                f"FROM outcomes WHERE ordinal IN ({placeholders}) ORDER BY ordinal",
                ordinals,
            ).fetchall()
            by_ordinal = {row[0]: row[1:] for row in outcome_rows}
            if len(by_ordinal) != len(outcome_rows):
                raise ResumeIdentityError("duplicate persisted resume ordinal")
            for ordinal, name in manifest_rows:
                if time.monotonic() >= deadline:
                    raise ResumeValidationTimeout(
                        "resume action catalog deadline exhausted"
                    )
                if (type(ordinal) is not int or ordinal != yielded
                        or type(name) is not str or not name):
                    raise ResumeIdentityError("resume manifest action identity differs")
                yield _action_for_row(
                    ordinal, name, by_ordinal.get(ordinal), expected_max
                )
                yielded += 1
        if yielded != context.factor_count:
            raise ResumeIdentityError("resume action count differs from validated context")
