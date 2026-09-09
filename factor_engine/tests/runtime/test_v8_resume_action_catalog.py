import hashlib
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import factor_engine.runtime.resume_action_catalog as catalog
from factor_engine.runtime.resume_validation import (
    ResumeIdentityError,
    ResumeValidationTimeout,
    ValidatedResumeContext,
)


def _fixture(tmp_path, rows, *, count=None, max_attempts=3):
    manifest_path = tmp_path / "manifest.sqlite3"
    state_path = tmp_path / "state.sqlite3"
    count = len(rows) if count is None else count
    with sqlite3.connect(manifest_path) as db:
        db.execute("create table factors(ordinal integer primary key,name text not null)")
        db.executemany(
            "insert into factors values(?,?)",
            ((ordinal, f"factor-{ordinal}") for ordinal in range(count)),
        )
    with sqlite3.connect(state_path) as db:
        db.execute(
            "create table outcomes(ordinal integer primary key,name text,state text,"
            "attempts integer,artifact_generation text,commit_state text)"
        )
        db.execute("create table state_policy(key text primary key,value text)")
        db.execute("insert into state_policy values('max_attempts',?)", (str(max_attempts),))
        db.executemany("insert into outcomes values(?,?,?,?,?,?)", rows)
    context = ValidatedResumeContext(
        "a" * 32, tmp_path, manifest_path, state_path, count, count,
        hashlib.sha256(b"manifest").hexdigest(),
    )
    return context, SimpleNamespace(work_item_max_attempts=max_attempts)


def _row(ordinal, state, attempts, generation=None, commit_state="NOT_STARTED"):
    return (ordinal, f"factor-{ordinal}", state, attempts, generation, commit_state)


def test_matrix_preserves_attempts_generations_and_terminal_immutability(tmp_path):
    generation = "b" * 32
    rows = [
        _row(1, "ACCEPTED", 0),
        _row(2, "RUNNING", 1),
        _row(3, "RUNNING", 3),
        _row(4, "FAILED", 2, generation, "INTENT"),
        _row(5, "CANCELLED", 1, generation, "UNKNOWN"),
        _row(6, "SUCCEEDED", 1, generation, "VERIFIED"),
        _row(7, "REUSED", 0, generation, "VERIFIED"),
        _row(8, "FAILED", 3),
        _row(9, "REJECTED", 0),
    ]
    context, policy = _fixture(tmp_path, rows, count=10)
    before = context.state_path.read_bytes()

    actions = list(catalog.iter_resume_action_catalog(
        context, policy=policy,
        restored_job_deadline=time.monotonic() + 60, page_size=2,
    ))

    assert [item.action for item in actions] == [
        "REGISTER_AND_EXECUTE", "EXECUTE_NEW",
        "RETRY_WITH_REMAINING_BUDGET", "NO_EXECUTION_BUDGET",
        "RECONCILE_EXACT_GENERATION", "RECONCILE_EXACT_GENERATION",
        "REVALIDATE_VERIFIED_ARTIFACT", "REVALIDATE_VERIFIED_ARTIFACT",
        "PRESERVE_TERMINAL", "PRESERVE_TERMINAL",
    ]
    assert [item.attempts for item in actions] == [0, 0, 1, 3, 2, 1, 1, 0, 3, 0]
    assert [item.remaining_attempts for item in actions] == [3, 3, 2, 0, 1, 2, 2, 3, 0, 3]
    assert [item.terminal_immutable for item in actions] == [
        False, False, False, False, True, True, True, True, True, True,
    ]
    assert actions[4].generation == generation
    assert actions[5].generation == generation
    assert context.state_path.read_bytes() == before


def test_catalog_pages_state_queries_without_loading_all_outcomes(tmp_path, monkeypatch):
    context, policy = _fixture(tmp_path, [], count=513)
    statements = []
    real_open = catalog._open_readonly

    def traced_open(path):
        connection = real_open(path)
        if Path(path) == context.state_path:
            connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(catalog, "_open_readonly", traced_open)
    count = sum(1 for _ in catalog.iter_resume_action_catalog(
        context, policy=policy,
        restored_job_deadline=time.monotonic() + 60, page_size=17,
    ))
    outcome_queries = [sql for sql in statements if "FROM outcomes WHERE ordinal IN" in sql]
    assert count == 513
    assert len(outcome_queries) == 31
    assert max(
        sql.split("ordinal IN (", 1)[1].split(")", 1)[0].count(",") + 1
        for sql in outcome_queries
    ) <= 17


def test_restored_deadline_is_required_and_never_refreshed(tmp_path, monkeypatch):
    context, policy = _fixture(tmp_path, [], count=1)
    monkeypatch.setattr(catalog.time, "monotonic", lambda: 100.0)
    with pytest.raises(TypeError):
        catalog.iter_resume_action_catalog(context, policy=policy)
    for _ in range(2):
        with pytest.raises(ResumeValidationTimeout, match="restored job deadline exhausted"):
            list(catalog.iter_resume_action_catalog(
                context, policy=policy, restored_job_deadline=99.0,
            ))


@pytest.mark.parametrize("row", [
    _row(0, "ACCEPTED", 1),
    _row(0, "UNKNOWN_STATE", 1, "b" * 32, "INTENT"),
    _row(0, "ACCEPTED", 0, "b" * 32, "INTENT"),
    _row(0, "SUCCEEDED", 1, "b" * 32, "INTENT"),
    _row(0, "REUSED", 0, "b" * 32, "UNKNOWN"),
    _row(0, "RUNNING", 0, "b" * 32, "INTENT"),
    _row(0, "RUNNING", 1, "not-a-generation", "INTENT"),
    _row(0, "FAILED", 1, "not-a-generation", "NOT_STARTED"),
    _row(0, "SUCCEEDED", 1, "b" * 32, "NOT_STARTED"),
])
def test_invalid_persisted_matrix_fails_closed(tmp_path, row):
    context, policy = _fixture(tmp_path, [row], count=1)
    with pytest.raises(ResumeIdentityError):
        list(catalog.iter_resume_action_catalog(
            context, policy=policy,
            restored_job_deadline=time.monotonic() + 60,
        ))


def test_terminal_unknown_reconciliation_is_readonly_and_immutable(tmp_path):
    generation = "c" * 32
    context, policy = _fixture(
        tmp_path, [_row(0, "FAILED", 2, generation, "UNKNOWN")], count=1,
    )
    before = context.state_path.read_bytes()
    [action] = list(catalog.iter_resume_action_catalog(
        context, policy=policy,
        restored_job_deadline=time.monotonic() + 60,
    ))
    assert action.action == "RECONCILE_EXACT_GENERATION"
    assert action.persisted_state == "FAILED"
    assert action.terminal_immutable is True
    assert action.generation == generation
    assert context.state_path.read_bytes() == before


def test_requires_exact_typed_validated_context(tmp_path):
    _context, policy = _fixture(tmp_path, [], count=0)
    with pytest.raises(ResumeIdentityError, match="validated resume context"):
        list(catalog.iter_resume_action_catalog(
            "not-a-context", policy=policy,
            restored_job_deadline=time.monotonic() + 60,
        ))
