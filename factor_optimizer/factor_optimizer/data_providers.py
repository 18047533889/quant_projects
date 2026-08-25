"""Real scoped data isolation (FO-P0-01).

The historical ``_evaluate_with_capability`` verified the capability/split
subset but then invoked the UNRESTRICTED evaluation callback, so an evaluator
closure holding ``full_X``/``full_y``/``test_y`` could still see the whole
dataset despite passing the capability check.  This module closes that gap:

- ``DataProvider`` (``TrainDataProvider`` / ``ValidationDataProvider`` /
  ``TestDataProvider``): resolves ONLY the coordinate rows a capability
  authorizes (a copy, never the live full payload).
- ``ScopedEvaluator``: evaluates a trial against the provider-resolved subset
  only.
- The search worker's ``TrainEvaluationContext`` / ``ValidationEvaluationContext``
  go through a provider + ``ScopedEvaluator``; the search worker never holds a
  test credential.

A forged / altered / wrong-scope / expired / mismatched-identity capability
fails closed before any row is resolved.
"""

from typing import Any, Callable, Dict

from factor_optimizer.contracts.splits import SplitPlan
from factor_optimizer.data_capabilities import (
    DataCapability,
    DataScope,
)
from factor_optimizer.errors import CapabilityForgeryError


def _authorized_rows(mask) -> tuple:
    """Return the 0-based indices a boolean mask authorizes."""
    return tuple(i for i, flag in enumerate(mask) if flag)


def _copy_rows(data: Dict[str, Any], authorized: tuple) -> Dict[str, Any]:
    """Return a shallow copy of ``data`` holding only the authorized rows."""
    result: Dict[str, Any] = {}
    for key, values in data.items():
        if not hasattr(values, "__getitem__") or isinstance(values, (str, bytes)):
            # Scalar / non-indexed metadata: copy by value.
            result[key] = values
            continue
        try:
            result[key] = [values[i] for i in authorized]
        except (IndexError, TypeError) as exc:
            raise CapabilityForgeryError(
                f"authorized coordinate {exc} is outside the provided payload"
            ) from exc
    return result


class DataProvider:
    """Resolves only the coordinate rows a capability authorizes.

    The provider owns a physical payload (dict of column arrays).  On
    ``resolve`` it verifies the capability fail-closed (type, scope, identity,
    expiry, forgery) and returns a COPY of only the authorized rows — never the
    live full payload, so the evaluator cannot reach out-of-scope data through
    aliasing.
    """

    scope: DataScope

    def __init__(
        self,
        data: Dict[str, Any],
        *,
        search_session_id: str,
        split_id: str,
        dataset_identity: str,
        provider_identity: str,
    ):
        if not isinstance(data, dict):
            raise TypeError("provider data must be a dict of column arrays")
        for name, value in (
            ("search_session_id", search_session_id),
            ("split_id", split_id),
            ("dataset_identity", dataset_identity),
            ("provider_identity", provider_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        self._data = data
        self._search_session_id = search_session_id
        self._split_id = split_id
        self._dataset_identity = dataset_identity
        self._provider_identity = provider_identity

    @property
    def scope(self) -> DataScope:
        return self._scope

    @property
    def dataset_identity(self) -> str:
        return self._dataset_identity

    @property
    def provider_identity(self) -> str:
        return self._provider_identity

    def _verify_capability(self, capability: DataCapability) -> None:
        if not isinstance(capability, DataCapability):
            raise CapabilityForgeryError(
                f"{self._scope.value} data provider requires a DataCapability; "
                f"got {type(capability).__name__}"
            )
        if capability.scope is not self._scope:
            raise CapabilityForgeryError(
                f"{self._scope.value} data provider requires a "
                f"{self._scope.value} capability; got scope="
                f"{capability.scope.value!r}"
            )
        capability.verify_identity()
        if capability.is_expired():
            raise CapabilityForgeryError(f"{self._scope.value} capability is expired")
        if capability.search_session_id != self._search_session_id:
            raise CapabilityForgeryError(
                "capability search_session_id does not match the provider"
            )
        if capability.split_id != self._split_id:
            raise CapabilityForgeryError(
                "capability split_id does not match the provider"
            )
        if capability.dataset_identity != self._dataset_identity:
            raise CapabilityForgeryError(
                "capability dataset_identity does not match the provider"
            )
        if capability.provider_identity != self._provider_identity:
            raise CapabilityForgeryError(
                "capability provider_identity does not match the provider"
            )

    def resolve(self, capability: DataCapability) -> Dict[str, Any]:
        """Return a COPY of the authorized rows, or raise fail-closed."""
        self._verify_capability(capability)
        authorized = _authorized_rows(capability.allowed_mask)
        return _copy_rows(self._data, authorized)


class TrainDataProvider(DataProvider):
    """Resolves only the authorized TRAIN rows."""

    _scope = DataScope.TRAIN


class ValidationDataProvider(DataProvider):
    """Resolves only the authorized VALIDATION rows."""

    _scope = DataScope.VALIDATION


class TestDataProvider(DataProvider):
    """Resolves only the authorized TEST rows (test authority only)."""

    _scope = DataScope.TEST


class ScopedEvaluator:
    """Evaluates a trial against ONLY the data a provider resolves.

    The evaluation callback receives ``(trial, fidelity, scoped_data)`` where
    ``scoped_data`` is the provider-resolved subset — never the full payload.
    """

    def __init__(self, provider: DataProvider, evaluation_fn: Callable):
        if not isinstance(provider, DataProvider):
            raise TypeError("ScopedEvaluator requires a DataProvider")
        if not callable(evaluation_fn):
            raise TypeError("evaluation_fn must be callable")
        self._provider = provider
        self._evaluation_fn = evaluation_fn

    def evaluate(self, capability: DataCapability, trial, fidelity: int):
        scoped = self._provider.resolve(capability)
        return self._evaluation_fn(trial, fidelity, scoped)


__all__ = [
    "DataProvider",
    "TrainDataProvider",
    "ValidationDataProvider",
    "TestDataProvider",
    "ScopedEvaluator",
]
