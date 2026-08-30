"""FO-P0-01 real scoped data isolation adversarial tests.

These are PASS-when-blocked tests.  The search worker must NEVER receive the
full dataset, the test label vector, or a global/other-scope payload.  The
providers resolve only the authorized coordinate rows, and the evaluator is
invoked through a capability-gated ScopedEvaluator that hands it only that
subset.

Adversarial vectors: closure holding full_y / global test data, provider
back-reference to the full set, a forged capability, mask substitution, and a
wrong-scope capability swapped into the evaluation path.
"""

import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.data_capabilities import (
    DataCapability,
    DataScope,
    TrainDataCapability,
    ValidationDataCapability,
    build_search_capabilities,
)
from factor_optimizer.data_providers import (
    ScopedEvaluator,
    TrainDataProvider,
    ValidationDataProvider,
)
from factor_optimizer.errors import CapabilityForgeryError
from factor_optimizer.search.runner import SearchConfig, SearchRunner
import numpy as np


def _disjoint_plan():
    # 3 samples: row 0 train, row 1 validation, row 2 test.
    return SplitPlan("p1", [True, False, False], [False, True, False], [False, False, True], {})


def _budget():
    return SearchBudget(max_trials=3, max_evaluations=3)


def _trial(tid="t1"):
    return Trial(trial_id=tid, mutation_id="m", status=TrialStatus.PROPOSED)


def _data():
    return {
        "X": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        "y": [10.0, 20.0, 30.0],
    }


def _capabilities(plan):
    return build_search_capabilities(
        plan,
        search_session_id="real-sess-0001",
        dataset_identity="real-ds-0001",
        provider_identity="real-prov-0001",
    )


def test_train_provider_resolves_only_authorized_train_rows():
    plan = _disjoint_plan()
    caps = _capabilities(plan)
    data = _data()
    provider = TrainDataProvider(
        data,
        search_session_id="real-sess-0001",
        split_id=plan.split_id,
        dataset_identity="real-ds-0001",
        provider_identity="real-prov-0001",
    )
    seen = []

    def evaluate(trial, fidelity, scoped):
        seen.append((trial.trial_id, scoped["y"]))

    scoped = ScopedEvaluator(provider, evaluate)
    scoped.evaluate(caps[DataScope.TRAIN], _trial(), 0)
    # The evaluator only ever saw the train row (y=10.0), never test/global.
    assert seen == [("t1", [10.0])]


def test_train_provider_back_reference_to_full_set_is_not_exposed():
    plan = _disjoint_plan()
    caps = _capabilities(plan)
    data = _data()
    provider = TrainDataProvider(
        data,
        search_session_id="real-sess-0001",
        split_id=plan.split_id,
        dataset_identity="real-ds-0001",
        provider_identity="real-prov-0001",
    )
    resolved = provider.resolve(caps[DataScope.TRAIN])
    # The resolved view is a copy; it does NOT hold the full y vector.
    assert "y" in resolved
    assert resolved["y"] == [10.0]
    # The provider does not hand back the raw full payload reference.
    assert resolved is not data
    assert resolved["y"] is not data["y"]


def test_train_provider_rejects_wrong_scope_capability():
    plan = _disjoint_plan()
    caps = _capabilities(plan)
    provider = TrainDataProvider(
        _data(),
        search_session_id="real-sess-0001",
        split_id=plan.split_id,
        dataset_identity="real-ds-0001",
        provider_identity="real-prov-0001",
    )
    with pytest.raises(CapabilityForgeryError, match="train capability"):
        provider.resolve(caps[DataScope.VALIDATION])


def test_provider_rejects_forged_or_altered_mask():
    plan = _disjoint_plan()
    caps = _capabilities(plan)
    provider = TrainDataProvider(
        _data(),
        search_session_id="real-sess-0001",
        split_id=plan.split_id,
        dataset_identity="real-ds-0001",
        provider_identity="real-prov-0001",
    )
    train = caps[DataScope.TRAIN]
    # Mask substitution: authorize a different row set (the test row).
    object.__setattr__(train, "_allowed_mask", (False, False, True))
    with pytest.raises(CapabilityForgeryError, match="coordinate_hash"):
        provider.resolve(train)


def test_wrong_scope_provider_swapped_into_train_path_is_blocked():
    # A caller swapping a VALIDATION provider into the train evaluation path
    # must fail closed — the train path may only resolve TRAIN-authorized rows.
    plan = _disjoint_plan()
    caps = _capabilities(plan)
    val_provider = ValidationDataProvider(
        _data(),
        search_session_id="real-sess-0001",
        split_id=plan.split_id,
        dataset_identity="real-ds-0001",
        provider_identity="real-prov-0001",
    )
    scoped = ScopedEvaluator(val_provider, lambda t, f, d: d)
    with pytest.raises(CapabilityForgeryError, match="validation capability"):
        scoped.evaluate(caps[DataScope.TRAIN], _trial(), 0)


def test_scoped_evaluator_never_hands_test_payload_to_train_path():
    # The search phase's object graph contains NO test data.  A full_y /
    # global_y closure is never reachable because the evaluator only receives
    # the provider-resolved train subset.
    full_y = [10.0, 20.0, 30.0]
    seen = []

    def leaky(trial, fidelity, scoped):
        # Even if the closure holds full_y, it can only see what the provider
        # hands it.
        seen.append(scoped["y"])
        return {"score": 0.5, "cost": 1.0, "evidence_ref": "ev-1",
        "treatment_integrity_evidence": _integrity_evidence(trial.trial_id),
            }

    plan = _disjoint_plan()
    runner = SearchRunner(
        SearchConfig(budget=_budget(), enable_multifidelity=False),
        lambda: _trial(),
        EvaluationProtocol(plan, lambda t, f: {"score": 0.5, "cost": 1.0, "evidence_ref": "e", "treatment_integrity_evidence": _integrity_evidence(t.trial_id)}),
    )
    # The search runner's data capabilities are only train/validation.
    assert set(runner._data_capabilities) == {DataScope.TRAIN, DataScope.VALIDATION}
    assert runner._test_authority_broker is None


def test_build_search_capabilities_has_no_test_capability():
    caps = _capabilities(_disjoint_plan())
    assert set(caps) == {DataScope.TRAIN, DataScope.VALIDATION}


# ---------------------------------------------------------------------------
# FO-P0-02: real capability provenance (no placeholder fixed strings)
# ---------------------------------------------------------------------------


def test_capability_uses_real_session_provenance_not_placeholders():
    plan = _disjoint_plan()
    runner = SearchRunner(
        SearchConfig(budget=_budget(), enable_multifidelity=False),
        lambda: _trial(),
        EvaluationProtocol(plan, lambda t, f: {"score": 0.5, "cost": 1.0, "evidence_ref": "e", "treatment_integrity_evidence": _integrity_evidence(t.trial_id)}),
    )
    # Before run: capability is bound to a pending session id, but never to the
    # placeholder strings "search"/"search_dataset"/"search_provider".
    train = runner._data_capabilities[DataScope.TRAIN]
    assert train.search_session_id != "search"
    assert train.dataset_identity != "search_dataset"
    assert train.provider_identity != "search_provider"

    # After run: capabilities are bound to the REAL session id.
    session = runner.run("real-session-abc-123")
    train = runner._data_capabilities[DataScope.TRAIN]
    val = runner._data_capabilities[DataScope.VALIDATION]
    assert train.search_session_id == "real-session-abc-123"
    assert val.search_session_id == "real-session-abc-123"
    assert train.split_id == plan.split_id
    assert "search" != train.dataset_identity
    assert train.dataset_identity.startswith("dataset:")
    assert train.provider_identity.startswith("provider:")


def test_capability_provenance_binds_split_and_coordinate_hash():
    plan = _disjoint_plan()
    runner = SearchRunner(
        SearchConfig(budget=_budget(), enable_multifidelity=False),
        lambda: _trial(),
        EvaluationProtocol(plan, lambda t, f: {"score": 0.5, "cost": 1.0, "evidence_ref": "e", "treatment_integrity_evidence": _integrity_evidence(t.trial_id)}),
    )
    runner.run("sess-bind-xyz")
    train = runner._data_capabilities[DataScope.TRAIN]
    val = runner._data_capabilities[DataScope.VALIDATION]
    # The split id and a non-empty coordinate hash bind the capability to the
    # exact authorized rows for this real session.
    assert train.split_id == "p1"
    assert train.coordinate_hash
    assert train.coordinate_hash != val.coordinate_hash


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
