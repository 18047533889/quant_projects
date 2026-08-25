"""FO-P0-04 adversarial regression tests for the append-only TrialLedger.

Every proposal attempt must leave a record in the append-only ledger with the
burned-budget outcomes (raised proposal, non-Trial proposal) DISTINCT from a
true duplicate.  Also pins checkpoint/restore integrity: the ledger survives a
roundtrip, a dropped/reordered/duplicated ledger is rejected, and the ledger is
never sealed against legitimate mid-run writes.
"""

import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.contracts.trial_ledger import (
    LedgerEntry,
    TrialLedger,
)
from factor_optimizer.search.runner import (
    SearchConfig,
    SearchRunner,
    SearchSession,
)


def _protocol(fn):
    return EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}), fn
    )


def _config(max_trials=10, max_eval=10, **kw):
    return SearchConfig(
        budget=SearchBudget(
            max_trials=max_trials, max_evaluations=max_eval, max_cost_units=float(max_eval)
        ),
        enable_multifidelity=False,
        **kw,
    )


def _trial(tid, mid="m"):
    return Trial(trial_id=tid, mutation_id=mid, status=TrialStatus.PROPOSED)


def _eval_ok(trial, fidelity):
    return {"evaluation_id": f"e-{trial.trial_id}", "score": 0.5, "cost": 1.0}


# ---------------------------------------------------------------------------
# Proposal-attempt outcomes are distinct, not silent, not duplicates
# ---------------------------------------------------------------------------


def test_raised_proposal_is_proposal_failed_not_duplicate():
    def boom():
        raise RuntimeError("generator exploded")

    session = SearchRunner(
        _config(max_trials=1, max_eval=1), boom, _protocol(_eval_ok)
    ).run("raised-proposal")
    counts = session.ledger.status_counts()
    assert counts.get("PROPOSAL_FAILED") == 1
    assert counts.get("DUPLICATE", 0) == 0
    # The burned trial budget is reflected in the tracker.
    assert session.budget_tracker.trials_used == 1
    assert "generator exploded" in session.ledger.entries[-1].failure_reason


def test_non_trial_proposal_is_invalid_proposal_not_silent():
    calls = [0]

    def proposal():
        calls[0] += 1
        if calls[0] == 1:
            return None
        if calls[0] == 2:
            return {"not": "a trial"}
        return _trial("t3")

    session = SearchRunner(
        _config(3, 3), proposal, _protocol(_eval_ok)
    ).run("invalid-proposals")
    counts = session.ledger.status_counts()
    # 2 invalid proposals + 1 real trial.
    assert counts.get("INVALID_PROPOSAL") == 2
    assert counts.get("PROPOSED") == 1
    assert counts.get("EVALUATED") == 1
    assert len(session.ledger) >= 4


def test_duplicate_trial_still_duplicate_but_distinct():
    proposals = iter([
        _trial("same"),
        _trial("same"),
        _trial("other"),
    ])
    session = SearchRunner(
        _config(3, 3),
        lambda: next(proposals, _trial("stop")),
        _protocol(_eval_ok),
    ).run("dup-ledger")
    counts = session.ledger.status_counts()
    assert counts.get("DUPLICATE") == 1
    assert counts.get("PROPOSAL_FAILED", 0) == 0
    assert counts.get("INVALID_PROPOSAL", 0) == 0


# ---------------------------------------------------------------------------
# The ledger is append-only and order-validated on checkpoint restore
# ---------------------------------------------------------------------------


def test_ledger_is_append_only_no_removal():
    ledger = TrialLedger()
    a = ledger.append("PROPOSED", trial_id="t1")
    b = ledger.append("EVALUATED", trial_id="t1")
    # The backing store is an immutable tuple: no remove/clear/pop methods.
    assert not hasattr(ledger._entries, "remove")
    assert not hasattr(ledger._entries, "append")
    assert ledger.entries == (a, b)
    # The returned entries are immutable snapshots (mutating the returned
    # tuple cannot reach the ledger's internal tuple).
    with pytest.raises((TypeError, AttributeError)):
        ledger.entries[0] = b  # type: ignore[index]
    assert ledger._entries == (a, b)


def test_ledger_entries_are_immutable_frozen():
    ledger = TrialLedger()
    entry = ledger.append("PROPOSAL_FAILED", failure_reason="boom")
    with pytest.raises(Exception):
        entry.status = "DUPLICATE"  # type: ignore[misc]


def test_ledger_roundtrip_through_session_checkpoint(tmp_path):
    config = _config(2, 2)
    runner = SearchRunner(config, lambda: _trial("t"), _protocol(_eval_ok))
    session = runner.run("ckpt")
    path = str(tmp_path / "sess.json")
    session.checkpoint(path)
    restored = SearchSession.resume(path)
    assert restored.ledger.to_dict() == session.ledger.to_dict()
    assert restored.ledger.status_counts() == session.ledger.status_counts()


def test_ledger_rejects_dropped_entry():
    session = SearchRunner(
        _config(2, 2),
        lambda: _trial("t"),
        _protocol(_eval_ok),
    ).run("drop-ckpt")
    data = session.to_dict()
    entries = list(data["ledger"]["entries"])
    # Drop an INTERIOR record (sequence 2) so the remaining sequences are no
    # longer contiguous — a genuinely lost burned-budget record.
    del entries[1]
    data["ledger"]["entries"] = entries
    with pytest.raises(ValueError, match="non-contiguous"):
        SearchSession.from_dict(data)


def test_ledger_rejects_duplicated_entry():
    session = SearchRunner(
        _config(2, 2),
        lambda: _trial("t"),
        _protocol(_eval_ok),
    ).run("dup-ckpt")
    data = session.to_dict()
    entries = list(data["ledger"]["entries"])
    entries.append(dict(entries[-1]))  # duplicate the last record
    data["ledger"]["entries"] = entries
    # from_dict rejects the duplicate (either as non-contiguous sequence or as
    # a reordered timestamp).
    with pytest.raises(ValueError):
        SearchSession.from_dict(data)


def test_ledger_rejects_reordered_entries():
    ledger = TrialLedger()
    ledger.append("PROPOSAL_FAILED", failure_reason="boom")
    ledger.append("PROPOSED", trial_id="t")
    entries = [e.to_dict() for e in ledger.entries]
    entries.reverse()
    with pytest.raises(ValueError, match="non-contiguous"):
        TrialLedger.from_dict({"entries": entries})


def test_ledger_rejects_missing_ledger_in_checkpoint():
    session = SearchRunner(
        _config(1, 1),
        lambda: _trial("t"),
        _protocol(_eval_ok),
    ).run("no-ledger")
    data = session.to_dict()
    del data["ledger"]
    with pytest.raises(ValueError, match="ledger"):
        SearchSession.from_dict(data)


def test_ledger_unknown_status_rejected_at_append():
    ledger = TrialLedger()
    with pytest.raises(ValueError, match="unknown trial outcome status"):
        ledger.append("MYSTERY")


def test_ledger_entry_sequence_validation():
    with pytest.raises(ValueError, match="sequence"):
        LedgerEntry(sequence=0, status="PROPOSED")
    with pytest.raises(ValueError, match="sequence"):
        LedgerEntry(sequence="x", status="PROPOSED")
