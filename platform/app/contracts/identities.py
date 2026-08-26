"""Identity canonicalization + hashing helpers.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.9 (spec §8.1–
§8.4). PURE stdlib. All identities are canonicalize-then-hash (sha256 of a
sorted, length-prefixed tuple). Display names never enter the canonical input.
``[RECONCILE]`` against ``factor_assets/identity/identity.py`` and
``quant_evaluator/contracts/`` before freeze.
"""

from __future__ import annotations

from typing import Any, Mapping

from ._contenthash import content_hash

__all__ = [
    "Identity",
    "FactorDefinitionIdentity",
    "FactorValueIdentity",
    "EvaluationIdentity",
    "TreatmentIdentity",
    "canonicalize",
]


def canonicalize(fields: Mapping[str, Any]) -> str:
    """Deterministic canonical string of a field mapping (sorted keys)."""
    return content_hash(fields)


class Identity:
    """Base for canonicalize-then-hash identities (spec §8)."""

    __slots__ = ("_hash",)

    def __init__(self, fields: Mapping[str, Any]) -> None:
        self._hash = canonicalize(fields)

    @property
    def hash(self) -> str:
        """sha256 hex digest of the canonicalized fields."""
        return self._hash

    def __str__(self) -> str:
        return self._hash

    def __repr__(self) -> str:
        return f"{type(self).__name__}(hash={self._hash})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Identity) and self._hash == other._hash

    def __hash__(self) -> int:
        return hash(self._hash)


class FactorDefinitionIdentity(Identity):
    """spec §8.1: formula/AST, operator semantics version, frequency, market,
    input schema requirements, parameter values, calculation semantics."""


class FactorValueIdentity(Identity):
    """spec §8.2: FactorDefinitionIdentity + DataSnapshotIdentity +
    UniverseIdentity + CalculationSpecIdentity + TimingSemanticsIdentity."""


class EvaluationIdentity(Identity):
    """spec §8.3: FactorValueIdentity + EvaluationPolicyIdentity +
    LabelDefinitionIdentity + EvaluationProfileIdentity."""


class TreatmentIdentity(Identity):
    """spec §8.4: source_factor_value_id + ordered preprocessing recipe +
    fit boundary + fit state content hash + neutralization schema."""
