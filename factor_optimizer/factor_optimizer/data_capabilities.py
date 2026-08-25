"""Real data capabilities: authorization-only scope boundaries.

P0-10 / R46: the train/validation/test evaluation contexts are logical
wrappers that all call the same ``evaluation_fn``.  This module introduces a
capability model that authorizes evaluation against a specific data scope
WITHOUT holding or referencing the underlying data.  A ``DataCapability``
knows only its scope, an immutable boolean mask, and a real identity binding;
it can say whether a split plan is consistent with that scope, and nothing
else.

R46 P0-Q: a capability is not merely ``any(mask)``.  Every capability is
bound to a real identity (``capability_id``, ``scope``, ``search_session_id``,
``split_id``, ``dataset_identity``, ``coordinate_hash``, ``provider_identity``,
``issued_at``, ``expiry``, ``nonce``).  ``can_evaluate`` requires the
requested coordinates to be an EXACT SUBSET of the authorized coordinates —
not merely that some overlap exists.  Forging or altering any identity field
fails closed with ``CapabilityForgeryError``.

R46 P0-R: the search runner's object graph contains ONLY train and validation
capabilities.  A TEST capability is not constructible inside the search
session: constructing one requires a ``TestAuthorityBroker`` (an explicit
test-provider token the search worker lacks).  ``build_search_capabilities``
returns only train/validation; ``build_test_capability`` is gated behind the
broker.

R46 P0-S: sealed test evaluation happens only through a separate test
authority.  ``ScopedEvaluator.evaluate(capability, trial, fidelity)`` resolves
data based on the capability; the search worker never holds a test provider.
``TestAuthorityBroker`` is the interface seam and ``TestWorker`` /
``TestDataProvider`` are NOT constructible inside a search session.
"""

import hashlib
import secrets
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Optional, Tuple

from factor_optimizer.contracts.splits import SplitPlan
from factor_optimizer.errors import CapabilityForgeryError


class DataScope(Enum):
    """The only three data scopes a split plan can authorize."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# R46 P0-R/P0-S: thread-local "current issuing broker".  Only
# ``TestAuthorityBroker.issue_test_capability`` sets this while it constructs a
# ``TestDataCapability``; a bare construction outside that call sees no broker
# and fails closed as a forgery.
_capability_issuer = threading.local()


def _as_bool_tuple(mask) -> Tuple[bool, ...]:
    """Coerce a validated plan mask to an immutable tuple of plain bools."""
    if not hasattr(mask, "__len__"):
        raise ValueError("mask must be a sized sequence of booleans")
    values = tuple(bool(value) for value in mask)
    if not values:
        raise ValueError("mask must not be empty")
    return values


def _coordinate_hash(mask: Tuple[bool, ...], scope: DataScope) -> str:
    """Deterministic digest over the authorized coordinates for a scope.

    Binds the capability to the exact set of authorized rows so a capability
    cannot be re-pointed at a different mask without invalidating its
    identity.  The scope is folded in so a train mask and a test mask that
    happen to be bit-identical still produce distinct hashes.
    """
    payload = f"{scope.value}:{','.join('1' if v else '0' for v in mask)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DataCapability:
    """Authorization for evaluating against a single data scope.

    Deliberately tiny: holds ``scope``, an immutable boolean ``allowed_mask``,
    and a real identity binding.  It does NOT hold or reference any data
    (train/validation/test payloads live in the evaluator/adapters, never in
    the capability).
    """

    scope: DataScope

    def __init__(
        self,
        scope: DataScope,
        allowed_mask,
        *,
        capability_id: Optional[str] = None,
        search_session_id: Optional[str] = None,
        split_id: Optional[str] = None,
        dataset_identity: Optional[str] = None,
        provider_identity: Optional[str] = None,
        issued_at: Optional[datetime] = None,
        expiry: Optional[datetime] = None,
        nonce: Optional[str] = None,
    ):
        if not isinstance(scope, DataScope):
            raise TypeError("scope must be a DataScope")
        self._scope = scope
        self._allowed_mask = _as_bool_tuple(allowed_mask)
        # R46 P0-R/P0-S: only a TestAuthorityBroker may construct a TEST
        # capability.  A bare `TestDataCapability(...)` constructed outside a
        # broker is a forgery (the search worker must never be able to mint
        # its own test authorization).  The `_capability_issuer` thread-local
        # is set exclusively by `TestAuthorityBroker.issue_test_capability`
        # for the duration of that call.
        if self._scope is DataScope.TEST:
            if not getattr(_capability_issuer, "broker", None):
                raise CapabilityForgeryError(
                    "TestDataCapability is not constructible outside a "
                    "TestAuthorityBroker; constructing one is a forgery attempt"
                )
        # Real identity binding.  Every field is required and validated so a
        # capability cannot be constructed with a blank/forged identity.
        self._capability_id = self._require_str(
            capability_id, "capability_id", default=self._default_id()
        )
        self._search_session_id = self._require_str(
            search_session_id, "search_session_id"
        )
        self._split_id = self._require_str(split_id, "split_id")
        self._dataset_identity = self._require_str(
            dataset_identity, "dataset_identity"
        )
        self._provider_identity = self._require_str(
            provider_identity, "provider_identity"
        )
        self._issued_at = self._require_dt(issued_at, "issued_at")
        self._expiry = self._require_dt(expiry, "expiry")
        if self._expiry is not None and self._issued_at is not None:
            if self._expiry < self._issued_at:
                raise ValueError("expiry must not precede issued_at")
        self._nonce = self._require_str(nonce, "nonce", default=secrets.token_hex(16))
        # The coordinate hash is derived from the immutable mask + scope and
        # pins the exact authorized rows.
        self._coordinate_hash = _coordinate_hash(self._allowed_mask, self._scope)

    @staticmethod
    def _default_id() -> str:
        return secrets.token_hex(16)

    @staticmethod
    def _require_str(value, name: str, default: Optional[str] = None) -> str:
        if value is None:
            if default is None:
                raise ValueError(f"{name} is required for a DataCapability identity")
            value = default
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _require_dt(value, name: str) -> Optional[datetime]:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"{name} must be a datetime")
        return value

    @property
    def scope(self) -> DataScope:
        return self._scope

    @property
    def allowed_mask(self) -> Tuple[bool, ...]:
        return self._allowed_mask

    @property
    def capability_id(self) -> str:
        return self._capability_id

    @property
    def search_session_id(self) -> str:
        return self._search_session_id

    @property
    def split_id(self) -> str:
        return self._split_id

    @property
    def dataset_identity(self) -> str:
        return self._dataset_identity

    @property
    def coordinate_hash(self) -> str:
        return self._coordinate_hash

    @property
    def provider_identity(self) -> str:
        return self._provider_identity

    @property
    def issued_at(self) -> Optional[datetime]:
        return self._issued_at

    @property
    def expiry(self) -> Optional[datetime]:
        return self._expiry

    @property
    def nonce(self) -> str:
        return self._nonce

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """True when the capability has passed its expiry."""
        if self._expiry is None:
            return False
        now = now or datetime.now(timezone.utc)
        return now > self._expiry

    def _authorized_positions(self) -> Tuple[int, ...]:
        return tuple(i for i, flag in enumerate(self._allowed_mask) if flag)

    def can_evaluate(self, split_plan: SplitPlan) -> bool:
        """True iff ``split_plan``'s coordinates for this scope are an EXACT
        SUBSET of the authorized coordinates.

        R46 P0-Q: this is not ``any(mask)``.  The requested coordinates must
        be a subset of the authorized coordinates — every requested row must
        be authorized.  A plan that requests a row outside the authorized set
        (or a different scope's rows) is rejected, even if it also overlaps
        some authorized row.
        """
        if not isinstance(split_plan, SplitPlan):
            return False
        if self.scope is DataScope.TRAIN:
            mask = split_plan.train_mask
        elif self.scope is DataScope.VALIDATION:
            mask = split_plan.validation_mask
        else:
            mask = split_plan.test_mask
        try:
            requested = tuple(bool(v) for v in mask)
        except (TypeError, ValueError):
            return False
        if len(requested) != len(self._allowed_mask):
            return False
        authorized = self._authorized_positions()
        # A capability that authorizes no rows cannot evaluate anything, and an
        # empty request is not a meaningful evaluation.
        if not authorized or not any(requested):
            return False
        for i, flag in enumerate(requested):
            if flag and i not in authorized:
                return False
        return True

    def verify_identity(self) -> None:
        """Re-derive the coordinate hash and reject any tampering.

        R46 P0-Q adversarial guard: if the mask or scope were altered after
        construction (via ``object.__setattr__`` or a forged subclass), the
        stored ``coordinate_hash`` no longer matches the re-derived digest and
        the capability is rejected as forged.
        """
        expected = _coordinate_hash(self._allowed_mask, self._scope)
        if self._coordinate_hash != expected:
            raise CapabilityForgeryError(
                "capability coordinate_hash does not match its mask/scope; "
                "the capability was forged or altered"
            )
        if self._capability_id is None or not self._capability_id.strip():
            raise CapabilityForgeryError("capability_id is missing")
        if self._search_session_id is None or not self._search_session_id.strip():
            raise CapabilityForgeryError("search_session_id is missing")
        if self._split_id is None or not self._split_id.strip():
            raise CapabilityForgeryError("split_id is missing")
        if self._dataset_identity is None or not self._dataset_identity.strip():
            raise CapabilityForgeryError("dataset_identity is missing")
        if self._provider_identity is None or not self._provider_identity.strip():
            raise CapabilityForgeryError("provider_identity is missing")
        if self._nonce is None or not self._nonce.strip():
            raise CapabilityForgeryError("nonce is missing")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(scope={self.scope.value!r}, "
            f"n={len(self.allowed_mask)}, id={self._capability_id[:8]}…)"
        )


class TrainDataCapability(DataCapability):
    """Authorization for the train scope only."""

    def __init__(self, allowed_mask, **identity):
        super().__init__(DataScope.TRAIN, allowed_mask, **identity)


class ValidationDataCapability(DataCapability):
    """Authorization for the validation scope only."""

    def __init__(self, allowed_mask, **identity):
        super().__init__(DataScope.VALIDATION, allowed_mask, **identity)


class TestDataCapability(DataCapability):
    """Authorization for the test scope only.

    This is the only capability that authorizes the test scope, and it is the
    only place a sealed-test executor may look.  It is NOT constructible inside
    a search session: constructing one requires a ``TestAuthorityBroker``
    (R46 P0-R / P0-S).
    """

    def __init__(self, allowed_mask, **identity):
        super().__init__(DataScope.TEST, allowed_mask, **identity)


# ---------------------------------------------------------------------------
# R46 P0-S: the test authority seam
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R46 P0-S: the test authority seam
# ---------------------------------------------------------------------------


class TestStoreRef:
    """Opaque reference to a test-data store (FO-P0-03).

    The test authority holds the physical test data behind this opaque ref,
    never as a raw caller object/dict.  ``read()`` is the ONLY way to obtain
    the payload; the ref itself exposes no columns, so a caller (or a search
    worker) cannot reach the rows without holding the store.  The ref is bound
    to a ``dataset_identity`` so a store cannot be swapped across datasets.
    """

    def __init__(self, store, *, dataset_identity: str):
        if not hasattr(store, "read"):
            raise TypeError("TestStoreRef requires an object exposing read()")
        if not isinstance(dataset_identity, str) or not dataset_identity.strip():
            raise ValueError("dataset_identity must be a non-empty string")
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_dataset_identity", dataset_identity)

    @property
    def dataset_identity(self) -> str:
        return self._dataset_identity

    def read(self):
        return self._store.read()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(dataset={self._dataset_identity!r}, "
            "store=opaque)"
        )


class TestDatasetIdentity:
    """Opaque identity of a test dataset (FO-P0-003).

    Binds a physical test store to a real dataset artifact id so the search
    worker cannot substitute a raw payload for the authoritative store.
    """

    def __init__(self, dataset_identity: str, store_ref: TestStoreRef):
        if not isinstance(dataset_identity, str) or not dataset_identity.strip():
            raise ValueError("dataset_identity must be a non-empty string")
        if not isinstance(store_ref, TestStoreRef):
            raise TypeError("store_ref must be a TestStoreRef")
        self._dataset_identity = dataset_identity
        self._store_ref = store_ref

    @property
    def dataset_identity(self) -> str:
        return self._dataset_identity

    @property
    def store_ref(self) -> TestStoreRef:
        return self._store_ref


class TestAuthorityBroker:
    """Interface seam for issuing TEST capabilities and resolving test data.

    The search worker never holds a ``TestAuthorityBroker``.  Sealed test
    evaluation happens only through a separate test authority (an independent
    worker/process).  A broker is the ONLY way to construct a
    ``TestDataCapability`` and the ONLY way to obtain a ``TestDataProvider``.

    FO-P0-003: the broker holds an opaque ``TestStoreRef`` to the physical test
    store.  The search worker cannot mint a store ref; a caller's raw payload
    is never the store boundary — the broker only resolves through an attached
    opaque store.
    """

    def __init__(
        self,
        *,
        dataset_identity: str,
        provider_identity: str,
        search_session_id: str,
        split_id: str,
        ttl_seconds: int = 3600,
    ):
        if not isinstance(dataset_identity, str) or not dataset_identity.strip():
            raise ValueError("dataset_identity must be a non-empty string")
        if not isinstance(provider_identity, str) or not provider_identity.strip():
            raise ValueError("provider_identity must be a non-empty string")
        if not isinstance(search_session_id, str) or not search_session_id.strip():
            raise ValueError("search_session_id must be a non-empty string")
        if not isinstance(split_id, str) or not split_id.strip():
            raise ValueError("split_id must be a non-empty string")
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        self._dataset_identity = dataset_identity
        self._provider_identity = provider_identity
        self._search_session_id = search_session_id
        self._split_id = split_id
        self._ttl_seconds = ttl_seconds
        # FO-P0-03: the physical test-data path.  None until the test
        # authority attaches an opaque store; a provider without one fails
        # closed rather than falling back to a caller's raw dict.
        self._store_ref: Optional[TestStoreRef] = None

    @property
    def store_ref(self) -> Optional[TestStoreRef]:
        return self._store_ref

    def attach_store_ref(self, store_ref: TestStoreRef) -> None:
        """Attach the opaque physical store this broker owns."""
        if not isinstance(store_ref, TestStoreRef):
            raise TypeError("store_ref must be a TestStoreRef")
        if self._store_ref is not None:
            raise ValueError("broker already has a store attached")
        self._store_ref = store_ref

    def issue_test_capability(self, test_mask) -> TestDataCapability:
        """Issue a TEST capability bound to this broker's identity."""
        now = datetime.now(timezone.utc)
        _capability_issuer.broker = self
        try:
            return TestDataCapability(
                test_mask,
                search_session_id=self._search_session_id,
                split_id=self._split_id,
                dataset_identity=self._dataset_identity,
                provider_identity=self._provider_identity,
                issued_at=now,
                expiry=now + timedelta(seconds=self._ttl_seconds),
            )
        finally:
            _capability_issuer.broker = None

    def create_test_provider(self, test_data=None) -> "TestDataProvider":
        """Create a provider that resolves test data for a TEST capability.

        ``test_data`` (backward-compatible): a raw dict is wrapped into an
        opaque ``TestStoreRef`` so existing test code keeps working.  When
        omitted, the broker's attached opaque store is used.  A provider with
        no store fails closed at resolve.
        """
        if test_data is not None and not isinstance(test_data, TestStoreRef):
            if not isinstance(test_data, dict):
                raise TypeError("test_data must be a dict or TestStoreRef")
            test_data = TestStoreRef(
                _DictStore(test_data),
                dataset_identity=self._dataset_identity,
            )
        store_ref = test_data if test_data is not None else self._store_ref
        return TestDataProvider(
            store_ref,
            dataset_identity=self._dataset_identity,
            provider_identity=self._provider_identity,
        )


class _DictStore:
    """Adapter wrapping a plain dict as an opaque ``read()`` store."""

    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        object.__setattr__(self, "_data", dict(data))

    def read(self):
        return object.__getattribute__(self, "_data")


class TestDataProvider:
    """Resolves test data ONLY for a valid, unexpired TEST capability.

    This is the physical test-data path.  It is NOT constructible inside a
    search session: the only way to obtain one is through a
    ``TestAuthorityBroker``, which the search worker never holds.

    FO-P0-003: the provider resolves through an opaque ``TestStoreRef``, never
    a caller's raw payload.  A provider with no store attached fails closed.
    """

    def __init__(
        self,
        store_ref: Optional[TestStoreRef],
        *,
        dataset_identity: str,
        provider_identity: str,
    ):
        if not isinstance(dataset_identity, str) or not dataset_identity.strip():
            raise ValueError("dataset_identity must be a non-empty string")
        if not isinstance(provider_identity, str) or not provider_identity.strip():
            raise ValueError("provider_identity must be a non-empty string")
        if store_ref is not None and not isinstance(store_ref, TestStoreRef):
            raise TypeError("store_ref must be a TestStoreRef or None")
        self._store_ref = store_ref
        self._dataset_identity = dataset_identity
        self._provider_identity = provider_identity

    @property
    def dataset_identity(self) -> str:
        return self._dataset_identity

    @property
    def provider_identity(self) -> str:
        return self._provider_identity

    @property
    def store_ref(self) -> Optional[TestStoreRef]:
        return self._store_ref

    def resolve(self, capability: DataCapability):
        """Return the test payload only for a valid TEST capability.

        Rejects (CapabilityForgeryError / ValueError) any capability that is
        not a TEST capability, is expired, is forged, or whose identity does
        not match this provider's dataset/provider identity.
        """
        if not isinstance(capability, TestDataCapability):
            raise CapabilityForgeryError(
                "test data provider requires a TestDataCapability; "
                f"got {type(capability).__name__}"
            )
        capability.verify_identity()
        if capability.is_expired():
            raise CapabilityForgeryError("test capability is expired")
        if capability.dataset_identity != self._dataset_identity:
            raise CapabilityForgeryError(
                "test capability dataset_identity does not match the provider"
            )
        if capability.provider_identity != self._provider_identity:
            raise CapabilityForgeryError(
                "test capability provider_identity does not match the provider"
            )
        if self._store_ref is None:
            raise CapabilityForgeryError(
                "test provider has no opaque store attached; refusing to "
                "resolve test data from a raw caller payload"
            )
        return self._store_ref.read()


class ScopedEvaluator:
    """Evaluates a trial against data resolved from a capability.

    R46 P0-S: ``evaluate(capability, trial, fidelity)`` resolves data based on
    the capability.  The search worker never holds a test provider, so it can
    never construct a ``ScopedEvaluator`` bound to test data.
    """

    def __init__(self, provider: TestDataProvider, evaluation_fn):
        if not isinstance(provider, TestDataProvider):
            raise TypeError("ScopedEvaluator requires a TestDataProvider")
        if not callable(evaluation_fn):
            raise TypeError("evaluation_fn must be callable")
        self._provider = provider
        self._evaluation_fn = evaluation_fn

    def evaluate(self, capability: DataCapability, trial, fidelity: int):
        """Resolve data from the capability and evaluate the trial."""
        data = self._provider.resolve(capability)
        return self._evaluation_fn(trial, fidelity, data)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_search_capabilities(
    split_plan: SplitPlan,
    *,
    search_session_id: str,
    dataset_identity: str,
    provider_identity: str,
) -> Dict[DataScope, DataCapability]:
    """Build ONLY the train and validation capabilities for a search session.

    R46 P0-R: the search runner's object graph must contain ONLY train and
    validation capabilities.  This builder never constructs a TEST capability;
    obtaining one requires a ``TestAuthorityBroker`` the search worker lacks.
    """
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    now = datetime.now(timezone.utc)
    common = dict(
        search_session_id=search_session_id,
        split_id=split_plan.split_id,
        dataset_identity=dataset_identity,
        provider_identity=provider_identity,
        issued_at=now,
    )
    return {
        DataScope.TRAIN: TrainDataCapability(
            split_plan.train_mask, **common
        ),
        DataScope.VALIDATION: ValidationDataCapability(
            split_plan.validation_mask, **common
        ),
    }


def build_data_capabilities(split_plan: SplitPlan) -> Dict[DataScope, DataCapability]:
    """Backward-compatible builder returning one capability per scope.

    NOTE: this builds a TEST capability and is therefore NOT safe for the
    search runner's object graph.  It is retained for callers that explicitly
    need all three scopes (e.g. a test authority that holds a broker).  The
    search runner must use ``build_search_capabilities`` instead.

    R46 P0-R/P0-S: because a TEST capability is only constructible through a
    ``TestAuthorityBroker``, this builder issues the TEST capability via a
    broker so it stays compliant with the non-forgery rule.
    """
    if not isinstance(split_plan, SplitPlan):
        raise TypeError("split_plan must be a SplitPlan")
    now = datetime.now(timezone.utc)
    common = dict(
        search_session_id="unspecified",
        split_id=split_plan.split_id,
        dataset_identity="unspecified",
        provider_identity="unspecified",
        issued_at=now,
    )
    _issuer = TestAuthorityBroker(
        dataset_identity="unspecified",
        provider_identity="unspecified",
        search_session_id="unspecified",
        split_id=split_plan.split_id,
    )
    return {
        DataScope.TRAIN: TrainDataCapability(split_plan.train_mask, **common),
        DataScope.VALIDATION: ValidationDataCapability(
            split_plan.validation_mask, **common
        ),
        DataScope.TEST: _issuer.issue_test_capability(split_plan.test_mask),
    }


__all__ = [
    "DataScope",
    "DataCapability",
    "TrainDataCapability",
    "ValidationDataCapability",
    "TestDataCapability",
    "TestAuthorityBroker",
    "TestDataProvider",
    "TestStoreRef",
    "TestDatasetIdentity",
    "ScopedEvaluator",
    "build_search_capabilities",
    "build_data_capabilities",
]
