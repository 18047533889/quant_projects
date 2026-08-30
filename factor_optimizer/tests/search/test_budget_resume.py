"""FO-P1-24 adversarial regression tests for budget resume hardening.

The historical ``SearchRunner.resume`` re-issued the whole ``BudgetTracker`` on
every resume, so a session could exceed its budget by repeatedly resuming.
These tests pin the two opt-in modes:

- ``resume_same_budget=True`` preserves an unfinished session's budget.
- ``BudgetExtensionAuthorization`` extends a finished session's budget and is
  rejected when tampered / mismatched / shrinking.
"""

from datetime import datetime, timezone

import pytest

from factor_optimizer.contracts.budget_extension import BudgetExtensionAuthorization
from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.search.runner import SearchConfig, SearchRunner, SearchSession
import numpy as np


def _protocol(fn):
    return EvaluationProtocol(
        SplitPlan("search", [True], [False], [False], {}), fn
    )


def _budget(max_trials=3, max_eval=3, max_cost=30.0):
    return SearchBudget(max_trials=max_trials, max_evaluations=max_eval, max_cost_units=max_cost)


def _config(max_trials=3, max_eval=3, max_cost=30.0, **kw):
    return SearchConfig(
        budget=_budget(max_trials, max_eval, max_cost),
        enable_multifidelity=False,
        **kw,
    )


def _trial(tid):
    return Trial(trial_id=tid, mutation_id="m", status=TrialStatus.PROPOSED)


def _eval_ok(trial, fidelity):
    return {"evaluation_id": f"e-{trial.trial_id}", "score": 0.5, "cost": 1.0, "treatment_integrity_evidence": _integrity_evidence(trial.trial_id)}


def _extension(
    session_id,
    old,
    new,
    actor="operator",
    reason="round 2",
    timestamp=None,
    signature="<auto>",
):
    ts = timestamp or datetime.now(timezone.utc)
    ext = BudgetExtensionAuthorization(
        search_session_id=session_id,
        reason=reason,
        old_budget=old,
        new_budget=new,
        actor=actor,
        timestamp=ts,
        signature=signature if signature != "<auto>" else None,
    )
    # If the caller supplied a real signature, re-sign over the correct hash.
    if signature != "<auto>":
        return BudgetExtensionAuthorization(
            search_session_id=session_id,
            reason=reason,
            old_budget=old,
            new_budget=new,
            actor=actor,
            timestamp=ts,
            signature=ext.content_hash if signature is None else signature,
        )
    return ext


# ---------------------------------------------------------------------------
# resume_same_budget: an unfinished session keeps its budget
# ---------------------------------------------------------------------------


def test_resume_same_budget_preserves_consumed_budget():
    config = _config(2, 2, 20.0)
    session = SearchSession(
        session_id="s1", config=config, budget_tracker=BudgetTracker(config.budget)
    )
    # One trial/eval consumed.
    session.budget_tracker.record_trial()
    session.budget_tracker.record_evaluation(5.0)
    trial = _trial("t1")
    trial.update_status(TrialStatus.EVALUATED, evaluation_ref="e1", metadata={"score": 0.5})
    session.add_trial(trial)
    session.best_score = 0.5
    session.best_trial_id = "t1"

    runner = SearchRunner(
        config,
        lambda: _trial("t2"),
        _protocol(_eval_ok),
    )
    result = runner.resume(session, resume_same_budget=True)
    # The budget was NOT re-armed: consumed counters survive.
    assert result.budget_tracker.trials_used >= 1
    assert result.budget_tracker.evaluations_used >= 1
    # The session did not silently exceed its budget.
    assert result.budget_tracker.trials_used <= config.budget.max_trials
    assert result.budget_tracker.evaluations_used <= config.budget.max_evaluations


def test_resume_same_budget_rejects_finished_session():
    config = _config()
    session = SearchSession(
        session_id="s1", config=config, budget_tracker=BudgetTracker(config.budget)
    )
    session.finish("budget_exhausted")
    runner = SearchRunner(
        config, lambda: _trial("t"), _protocol(_eval_ok)
    )
    with pytest.raises(ValueError, match="cannot resume a finished session"):
        runner.resume(session, resume_same_budget=True)


# ---------------------------------------------------------------------------
# BudgetExtensionAuthorization: content hash + growth
# ---------------------------------------------------------------------------


def test_extension_is_signed_and_verifies():
    old = _budget(2)
    new = _budget(5)
    ext = _extension("sess-1", old, new)
    assert ext.content_hash
    ext.verify()  # genuine extension verifies


def test_extension_rejects_tampered_content():
    old = _budget(2)
    new = _budget(5)
    ext = _extension("sess-1", old, new)
    # Tamper with a field the hash covers.
    object.__setattr__(ext, "actor", "attacker")
    with pytest.raises(ValueError, match="tampered"):
        ext.verify()


def test_extension_from_dict_rejects_tampered_hash():
    old = _budget(2)
    new = _budget(5)
    ext = _extension("sess-1", old, new)
    data = ext.to_dict()
    data["new_budget"] = _budget(100).to_dict()  # tamper
    with pytest.raises(ValueError, match="content_hash"):
        BudgetExtensionAuthorization.from_dict(data)


def test_resume_with_extension_grows_budget():
    config = _config(1, 1, 10.0)
    session = SearchSession(
        session_id="sess-ext",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
    )
    session.budget_tracker.record_trial()
    session.budget_tracker.record_evaluation(5.0)
    t1 = Trial(trial_id="t1", mutation_id="m", status=TrialStatus.EVALUATED)
    t1.update_status(TrialStatus.EVALUATED, evaluation_ref="e", metadata={"score": 0.5})
    session.add_trial(t1)
    session.finish("budget_exhausted")

    new_budget = _budget(3, 3, 30.0)
    ext = _extension("sess-ext", config.budget, new_budget)

    runner = SearchRunner(config, lambda: _trial("t2"), _protocol(_eval_ok))
    result = runner.resume(session, budget_extension=ext)
    assert result.budget_tracker.budget.max_trials == 3
    assert result.budget_tracker.budget.max_evaluations == 3


def test_resume_with_extension_rejects_mismatched_session():
    config = _config(1, 1, 10.0)
    session = SearchSession(
        session_id="sess-other",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
    )
    ext = _extension("sess-wrong", _budget(2), _budget(5))
    runner = SearchRunner(_config(3, 3, 30.0), lambda: _trial("t"), _protocol(_eval_ok))
    with pytest.raises(ValueError, match="does not match"):
        runner.resume(session, budget_extension=ext)


def test_resume_rejects_shrinking_extension():
    config = _config(3, 3, 30.0)
    session = SearchSession(
        session_id="sess-shrink",
        config=config,
        budget_tracker=BudgetTracker(config.budget),
    )
    # New budget is SMALLER than old.
    ext = _extension("sess-shrink", config.budget, _budget(2))
    runner = SearchRunner(config, lambda: _trial("t"), _protocol(_eval_ok))
    with pytest.raises(ValueError, match="must not shrink"):
        runner.resume(session, budget_extension=ext)


def test_resume_same_budget_and_extension_mutually_exclusive():
    config = _config()
    session = SearchSession(
        session_id="s1", config=config, budget_tracker=BudgetTracker(config.budget)
    )
    ext = _extension("s1", _budget(2), _budget(5))
    runner = SearchRunner(config, lambda: _trial("t"), _protocol(_eval_ok))
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.resume(session, resume_same_budget=True, budget_extension=ext)


def _pro_ok(trial, fidelity):
    return _eval_ok(trial, fidelity)


def _integrity_evidence(trial_id="t1", kind=None):
    """Passing TreatmentIntegrityEvidence measured from arrays (R55 P0-9)."""
    from factor_optimizer.contracts.treatment_integrity import (
        build_integrity_evidence,
    )

    rng = np.random.default_rng(abs(hash(trial_id)) % (2 ** 32))
    before = rng.normal(size=32)
    treated_kind = kind if kind else f"treatment::{trial_id}"
    if treated_kind == "raw":
        after = before
    else:
        after = before * 0.5 + 0.01
    return build_integrity_evidence(
        trial_id,
        treated_kind,
        {} if treated_kind == "raw" else {"window": 3},
        before,
        after,
    )

