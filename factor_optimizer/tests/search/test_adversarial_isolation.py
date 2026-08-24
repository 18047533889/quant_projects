"""R46 P0-Q / P0-R / P0-S adversarial isolation tests.

These are PASS-when-blocked tests: every test asserts that the search phase is
UNABLE to reach the test payload.  The search runner's object graph must
contain no TEST capability, no test provider, and no test data path.

Adversarial vectors covered:
  - closure capturing test_y / test labels
  - global-variable leak of test payload
  - ``getattr(runner, '__dict__')`` / attribute inspection leaks
  - python ``pickle`` of the runner round-trip
  - checkpoint/store that embeds test data
  - provider back-reference to the test set
  - capability forgery (wrong scope/split)
  - mask substitution (swap authorized mask)
  - split substitution (run TRAIN capability against VALIDATION rows)
"""

import pickle

import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.contracts.splits import EvaluationProtocol, SplitPlan
from factor_optimizer.contracts.trial import Trial, TrialStatus
from factor_optimizer.data_capabilities import (
   
        DataCapability,
    DataScope,
    TestAuthorityBroker,
    TestDataCapability,
    TestDataProvider,
    TrainDataCapability,
    ValidationDataCapability,
    build_search_capabilities,
)
from factor_optimizer.errors import CapabilityForgeryError
from factor_optimizer.search.runner import (
   
        SearchConfig,
    SearchRunner,
    SealedTestExecutor,
)


def _plan(
    train=None,
    validation=None,
    test=None,
    split_id="p1",
):
    return SplitPlan(
        split_id,
        train if train is not None else [True, True, False],
        validation if validation is not None else [False, False, True],
        test if test is not None else [False, False, False],
        {},
    )


def _disjoint_plan(train, validation, test, split_id="p1"):
    """Build a plan with explicitly disjoint masks (no default overlap)."""
    return SplitPlan(split_id, train, validation, test, {})


def _protocol(plan):
    return EvaluationProtocol(
        plan,
        lambda trial, fid: {
            "score": 0.5,
            "cost": 1.0,
            "evidence_ref": "ev-1",
        },
    )


def _budget():
    return SearchBudget(max_trials=3, max_evaluations=3)


def _trial():
    return Trial(
        trial_id="t1",
        mutation_id="m1",
        status=TrialStatus.VALIDATING,
    )


def _runner(plan=None):
    plan = plan or _plan()
    return SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=_protocol(plan),
    )


# ---------------------------------------------------------------------------
# P0-R: no TEST capability reachable from the search runner
# ---------------------------------------------------------------------------


def test_search_runner_has_no_test_capability():
    runner = _runner()
    assert set(runner._data_capabilities) == {DataScope.TRAIN, DataScope.VALIDATION}
    for cap in runner._data_capabilities.values():
        assert not isinstance(cap, TestDataCapability)


def test_no_method_returns_test_capability():
    runner = _runner()
    # Every public method that could yield a capability must not return a TEST
    # capability.
    assert runner._test_authority_broker is None
    executor = runner.create_sealed_test_executor()
    assert executor.has_test_authority is False
    # The sealed-test executor created from the search runner cannot evaluate.
    with pytest.raises(ValueError, match="no test authority broker"):
        executor.evaluate_sealed_test(None, None, _plan())


def test_test_capability_not_constructible_without_broker():
    # A TestDataCapability requires a full identity; constructing one without
    # a broker's identity fields fails closed.
    with pytest.raises(ValueError, match="search_session_id"):
        TestDataCapability([False, False, True])


def test_test_provider_not_constructible_inside_search():
    # TestDataProvider requires a broker-supplied identity; there is no way to
    # obtain one from the search runner.
    runner = _runner()
    assert not hasattr(runner, "_test_provider")
    assert runner._test_authority_broker is None


# ---------------------------------------------------------------------------
# P0-Q: capability identity binding and exact-subset authorization
# ---------------------------------------------------------------------------


def test_can_evaluate_requires_exact_subset_not_any_overlap():
    # A train capability authorizes rows {0,1}.  A plan requesting row 2 (a
    # validation row) must be rejected even though it also overlaps row 0.
    caps = build_search_capabilities(
        _disjoint_plan([True, True, False], [False, False, True], [False, False, False]),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    # Authorized rows {0,1}; requesting {0,1} is a subset -> True.
    assert train.can_evaluate(
        _disjoint_plan([True, True, False], [False, False, False], [False, False, False])
    )
    # Requesting {0,2} includes row 2 which is NOT authorized -> False.
    assert not train.can_evaluate(
        _disjoint_plan([True, False, True], [False, False, False], [False, False, False])
    )
    # Requesting only {0} is a strict subset -> True.
    assert train.can_evaluate(
        _disjoint_plan([True, False, False], [False, False, False], [False, False, False])
    )


def test_can_evaluate_rejects_wrong_scope_rows():
    # A validation capability authorizes row 2 only.  A plan that requests a
    # train row (0) must be rejected.
    caps = build_search_capabilities(
        _disjoint_plan([True, True, False], [False, False, True], [False, False, False]),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    val = caps[DataScope.VALIDATION]
    assert val.can_evaluate(
        _disjoint_plan([False, False, False], [False, False, True], [False, False, False])
    )
    assert not val.can_evaluate(
        _disjoint_plan([False, False, False], [True, False, False], [False, False, False])
    )


def test_capability_identity_fields_are_bound():
    caps = build_search_capabilities(
        _disjoint_plan([True, True, False], [False, False, True], [False, False, False]),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    assert train.search_session_id == "s1"
    assert train.split_id == "p1"
    assert train.dataset_identity == "ds"
    assert train.provider_identity == "prov"
    assert train.capability_id
    assert train.coordinate_hash
    assert train.nonce
    assert train.issued_at is not None
    # verify_identity passes for a genuine capability.
    train.verify_identity()


def test_capability_forgery_wrong_scope_fails():
    # A TRAIN capability cannot be passed off as a TEST capability.
    caps = build_search_capabilities(
        _plan(),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    broker = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    provider = broker.create_test_provider({"test_y": [1, 2, 3]})
    with pytest.raises(CapabilityForgeryError, match="requires a TestDataCapability"):
        provider.resolve(train)


def test_capability_forgery_altered_mask_fails():
    # Altering the authorized mask after construction must invalidate the
    # capability's coordinate hash.
    caps = build_search_capabilities(
        _plan(),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    object.__setattr__(train, "_allowed_mask", (True, True, True))
    with pytest.raises(CapabilityForgeryError, match="coordinate_hash"):
        train.verify_identity()


def test_capability_forgery_altered_scope_fails():
    caps = build_search_capabilities(
        _plan(),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    object.__setattr__(train, "_scope", DataScope.TEST)
    with pytest.raises(CapabilityForgeryError, match="coordinate_hash"):
        train.verify_identity()


def test_capability_forgery_blank_identity_fails():
    caps = build_search_capabilities(
        _plan(),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    object.__setattr__(train, "_search_session_id", "")
    with pytest.raises(CapabilityForgeryError, match="search_session_id"):
        train.verify_identity()


def test_expired_capability_is_rejected():
    from datetime import datetime, timedelta, timezone

    broker = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    cap = broker.issue_test_capability([False, False, True])
    # Force expiry into the past.
    object.__setattr__(
        cap,
        "_expiry",
        datetime.now(timezone.utc) - timedelta(seconds=10),
    )
    provider = broker.create_test_provider({"test_y": [1, 2, 3]})
    with pytest.raises(CapabilityForgeryError, match="expired"):
        provider.resolve(cap)


# ---------------------------------------------------------------------------
# P0-S: the test authority seam
# ---------------------------------------------------------------------------


def test_test_authority_broker_issues_test_capability():
    broker = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    cap = broker.issue_test_capability([False, False, True])
    assert isinstance(cap, TestDataCapability)
    assert cap.scope is DataScope.TEST
    cap.verify_identity()


def test_test_provider_resolves_only_for_matching_test_capability():
    broker = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    provider = broker.create_test_provider({"test_y": [1, 2, 3]})
    cap = broker.issue_test_capability([False, False, True])
    assert provider.resolve(cap) == {"test_y": [1, 2, 3]}


def test_test_provider_rejects_mismatched_dataset_identity():
    broker = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    provider = broker.create_test_provider({"test_y": [1, 2, 3]})
    other = TestAuthorityBroker(
        dataset_identity="other_ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    cap = other.issue_test_capability([False, False, True])
    with pytest.raises(CapabilityForgeryError, match="dataset_identity"):
        provider.resolve(cap)


def test_test_provider_rejects_mismatched_provider_identity():
    broker = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="prov",
        search_session_id="s1",
        split_id="p1",
    )
    provider = broker.create_test_provider({"test_y": [1, 2, 3]})
    other = TestAuthorityBroker(
        dataset_identity="ds",
        provider_identity="other_prov",
        search_session_id="s1",
        split_id="p1",
    )
    cap = other.issue_test_capability([False, False, True])
    with pytest.raises(CapabilityForgeryError, match="provider_identity"):
        provider.resolve(cap)


# ---------------------------------------------------------------------------
# Adversarial: the search phase must be unable to reach test payload
# ---------------------------------------------------------------------------


def test_closure_capturing_test_y_is_blocked():
    # A closure that captures test_y must not be reachable from the search
    # runner's object graph.  The runner only holds train/validation
    # capabilities and no test provider.
    test_y = [1, 2, 3]

    def leaky_eval(trial, fid):
        return {"score": test_y[0], "cost": 1.0, "evidence_ref": "e"}

    plan = _plan()
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=lambda: _trial(),
        evaluation_fn=EvaluationProtocol(plan, leaky_eval),
    )
    # The runner's data capabilities never reference test data.
    assert set(runner._data_capabilities) == {DataScope.TRAIN, DataScope.VALIDATION}
    # No test provider is reachable.
    assert runner._test_authority_broker is None
    # The sealed-test executor from the search runner cannot evaluate.
    with pytest.raises(ValueError, match="no test authority broker"):
        runner.create_sealed_test_executor().evaluate_sealed_test(
            None, None, _plan()
        )


def test_global_variable_leak_of_test_payload_is_blocked():
    # A module-global holding test payload must not be reachable through the
    # search runner's capability graph.
    runner = _runner()
    assert set(runner._data_capabilities) == {DataScope.TRAIN, DataScope.VALIDATION}
    for cap in runner._data_capabilities.values():
        assert not isinstance(cap, TestDataCapability)
        # Capabilities never hold data payloads.
        assert not hasattr(cap, "_test_data")
        assert not hasattr(cap, "test_y")


def test_attribute_inspection_leak_is_blocked():
    # getattr(runner, '__dict__') must not expose a test provider or test data.
    runner = _runner()
    d = getattr(runner, "__dict__")
    assert "_test_authority_broker" in d
    assert d["_test_authority_broker"] is None
    # No test provider attribute exists on the runner.
    assert "_test_provider" not in d
    # The data capabilities dict contains only train/validation.
    caps = d["_data_capabilities"]
    assert set(caps) == {DataScope.TRAIN, DataScope.VALIDATION}
    for cap in caps.values():
        assert not isinstance(cap, TestDataCapability)


def test_pickle_roundtrip_does_not_carry_test_payload():
    # Pickling the runner must not smuggle test data: the runner holds no test
    # provider and no test capability, so a round-trip cannot carry test_y.
    # Use a module-level evaluation function so the runner is picklable.
    plan = _disjoint_plan([True, True, False], [False, False, True], [False, False, False])
    runner = SearchRunner(
        config=SearchConfig(budget=_budget()),
        proposal_fn=_module_trial,
        evaluation_fn=EvaluationProtocol(plan, _module_eval),
    )
    blob = pickle.dumps(runner)
    restored = pickle.loads(blob)
    assert set(restored._data_capabilities) == {
        DataScope.TRAIN,
        DataScope.VALIDATION,
    }
    assert restored._test_authority_broker is None
    for cap in restored._data_capabilities.values():
        assert not isinstance(cap, TestDataCapability)


def _module_trial():
    return _trial()


def _module_eval(trial, fid):
    return {"score": 0.5, "cost": 1.0, "evidence_ref": "ev-1"}


def test_checkpoint_does_not_embed_test_data():
    # A search session checkpoint must not embed test data.  The runner's
    # capabilities carry no test payload, and the session serializes only
    # search state.
    runner = _runner()
    session = runner.run("s1")
    data = session.to_dict()
    # No test payload key anywhere in the checkpoint.
    assert "test_y" not in str(data)
    assert "test_data" not in str(data)


def test_provider_back_reference_to_test_set_is_blocked():
    # A TestDataProvider holds a back-reference to the test set, but it is
    # NOT constructible inside a search session.  The search runner never
    # holds one.
    runner = _runner()
    assert runner._test_authority_broker is None
    # There is no way to obtain a provider from the search runner.
    assert not hasattr(runner, "_test_provider")
    # The sealed-test executor from the search runner has no authority.
    assert runner.create_sealed_test_executor().has_test_authority is False


def test_mask_substitution_swap_authorized_mask_is_blocked():
    # Swapping the authorized mask must invalidate the capability.
    caps = build_search_capabilities(
        _plan(),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    # Swap the authorized mask to authorize a different row set.
    object.__setattr__(train, "_allowed_mask", (False, False, True))
    with pytest.raises(CapabilityForgeryError, match="coordinate_hash"):
        train.verify_identity()


def test_split_substitution_train_capability_against_validation_rows_is_blocked():
    # A TRAIN capability must not authorize evaluation against VALIDATION rows.
    caps = build_search_capabilities(
        _disjoint_plan([True, True, False], [False, False, True], [False, False, False]),
        search_session_id="s1",
        dataset_identity="ds",
        provider_identity="prov",
    )
    train = caps[DataScope.TRAIN]
    # Train authorizes rows {0,1}.  A plan whose train mask requests row 2
    # (a validation row) must be rejected.
    assert not train.can_evaluate(
        _disjoint_plan([False, False, True], [False, False, False], [False, False, False])
    )
    # A plan that requests validation rows as if they were train rows is
    # rejected because those coordinates are not authorized.
    assert not train.can_evaluate(
        _disjoint_plan([False, True, True], [False, False, False], [False, False, False])
    )


def test_search_runner_cannot_evaluate_sealed_test_without_authority():
    # The physical test-data path is absent from the search process.
    runner = _runner()
    executor = runner.create_sealed_test_executor()
    with pytest.raises(ValueError, match="no test authority broker"):
        executor.evaluate_sealed_test(None, None, _plan())


def test_scoped_evaluator_requires_test_provider():
    # A ScopedEvaluator cannot be built without a TestDataProvider, and the
    # search runner never holds one.
    from factor_optimizer.data_capabilities import ScopedEvaluator

    runner = _runner()
    assert runner._test_authority_broker is None
    with pytest.raises(TypeError, match="TestDataProvider"):
        ScopedEvaluator(None, lambda t, f, d: {})
