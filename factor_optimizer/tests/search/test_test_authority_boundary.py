"""FO-P0-03 test-authority store-boundary regression tests.

The test authority's storage path must be behind an opaque ``TestStoreRef`` and
``TestDatasetIdentity`` — the search worker must never hold a raw object/dict
that reaches test data.  These are PASS-when-blocked tests: the store reference
and the physical test-data path exist only inside the test authority; a search
worker cannot mint one, and a caller's raw payload cannot be treated as the
authoritative store without going through the opaque identity.
"""

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
)
from factor_optimizer.errors import CapabilityForgeryError
from factor_optimizer.search.runner import SearchConfig, SearchRunner


def _plan():
    return SplitPlan("p1", [True, False, False], [False, True, False], [False, False, True], {})


def _budget():
    return SearchBudget(max_trials=3, max_evaluations=3)


def _trial(tid="t1"):
    return Trial(trial_id=tid, mutation_id="m", status=TrialStatus.PROPOSED)


def _broker(session_id="sess-abc", split_id="p1"):
    return TestAuthorityBroker(
        dataset_identity="ds-123",
        provider_identity="prov-456",
        search_session_id=session_id,
        split_id=split_id,
    )


def test_broker_holds_opaque_store_ref_not_raw_payload():
    broker = _broker()
    # The broker's store ref is opaque: it does not expose a raw y/X payload.
    assert hasattr(broker, "store_ref")
    assert not hasattr(broker.store_ref, "y")
    assert not hasattr(broker.store_ref, "X")


def test_provider_resolves_only_via_opaque_store_identity():
    broker = _broker()
    store = TestStoreRef(_OpaqueTestStore({"test_y": [1, 2, 3]}), dataset_identity="ds-identity")
    broker.attach_store_ref(store)
    provider = broker.create_test_provider()
    cap = broker.issue_test_capability([False, False, True])
    data = provider.resolve(cap)
    assert data == {"test_y": [1, 2, 3]}
    # The provider resolved through the store's opaque identity, not a caller
    # dict smuggled directly.
    assert provider.store_ref is store


def test_search_runner_never_holds_test_credential():
    runner = SearchRunner(
        SearchConfig(budget=_budget(), enable_multifidelity=False),
        lambda: _trial(),
        EvaluationProtocol(_plan(), lambda t, f: {"score": 0.5, "cost": 1.0, "evidence_ref": "e"}),
    )
    # The search runner's object graph contains no test provider, no store ref,
    # and no raw test credential.
    assert runner._test_authority_broker is None
    assert not hasattr(runner, "_test_provider")
    assert not hasattr(runner, "_test_store_ref")


def test_provider_without_attached_store_fails_closed():
    # A test provider with no opaque store attached cannot resolve — it must
    # never fall back to a caller-supplied raw dict.
    broker = _broker()
    provider = broker.create_test_provider()
    cap = broker.issue_test_capability([False, False, True])
    with pytest.raises(CapabilityForgeryError):
        provider.resolve(cap)


class _OpaqueTestStore:
    """A minimal opaque store the authority owns."""

    def __init__(self, data):
        object.__setattr__(self, "_data", data)

    def read(self):
        return object.__getattribute__(self, "_data")


from factor_optimizer.data_capabilities import TestStoreRef  # noqa: E402


def test_store_ref_is_opaque_and_identity_bound():
    store = _OpaqueTestStore({"test_y": [1, 2, 3]})
    ref = TestStoreRef(store, dataset_identity="ds-identity")
    assert ref.dataset_identity == "ds-identity"
    # The ref cannot expose raw columns; reading goes through read().
    assert not hasattr(ref, "y")
    assert ref.read() == {"test_y": [1, 2, 3]}
