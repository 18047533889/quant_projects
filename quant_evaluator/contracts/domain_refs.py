"""Typed cross-domain artifact references (R55 audit #24).

Before this module, every field of :class:`EvaluationArtifact` that pointed at
another artifact was a loosely-typed ``Any`` — a bare ``str``, an ad-hoc dict,
or a ref object, with no way to tell them apart and no validation.  A
downstream consumer could not ask "which domain does this ref belong to?",
"what kind of artifact is it?", or "what content was it pinned to?" without
guessing at the dict's shape.

:class:`DomainArtifactRef` is the single typed reference value for every
cross-artifact edge:

- ``domain``        — :class:`ArtifactDomain` enum (which subsystem owns the
  referenced artifact: factor value, label definition, evaluation policy /
  profile, split, snapshot, universe, metric evidence, diagnostic);
- ``artifact_kind`` — the kind of artifact *within* that domain
  (``"factor_value"``, ``"sealed_split"``, ``"rank_ic_evidence"``, ...);
- ``identity``      — the stable identity string of the referenced artifact
  (its id / uri), never empty;
- ``content_hash``  — the content hash the reference is pinned to (empty means
  the reference is identity-only; a non-empty value must be a real hash
  string).

It is a frozen dataclass (fail-closed validation in ``__post_init__``),
JSON-serializable via ``to_dict`` / ``from_dict``, and carries its own
canonical SHA-256 (:attr:`ref_hash`) so the *reference itself* — not just the
artifact it points at — is addressable and hashable across processes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from quant_evaluator.contracts._hashutil import stable_content_hex, stable_hash

__all__ = ["ArtifactDomain", "DomainArtifactRef"]


class ArtifactDomain(enum.Enum):
    """Which subsystem owns the artifact a :class:`DomainArtifactRef` points at."""

    FACTOR_VALUE = "factor_value"
    LABEL_DEFINITION = "label_definition"
    EVALUATION_POLICY = "evaluation_policy"
    EVALUATION_PROFILE = "evaluation_profile"
    SPLIT = "split"
    SNAPSHOT = "snapshot"
    UNIVERSE = "universe"
    METRIC_EVIDENCE = "metric_evidence"
    DIAGNOSTIC = "diagnostic"

    @classmethod
    def from_value(cls, value: Any) -> "ArtifactDomain":
        """Parse a serialized ``domain`` value back to the enum (fail-closed)."""
        if isinstance(value, ArtifactDomain):
            return value
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            "ArtifactDomain.from_value: unknown domain "
            f"{value!r} (type {type(value).__name__}); expected one of "
            f"{[m.value for m in cls]}"
        )


@dataclass(frozen=True)
class DomainArtifactRef:
    """A typed, validated reference to an artifact in another domain.

    Attributes:
        domain:        :class:`ArtifactDomain` owning the referenced artifact.
            A bare string is rejected — use :meth:`of` or
            :meth:`from_dict` for the explicit coercion entry points.
        artifact_kind: Kind of the artifact within ``domain`` (non-empty).
        identity:      Stable identity string of the referenced artifact
            (non-empty).  This is the addressable id / uri.
        content_hash:  Content hash the reference is pinned to.  Empty means
            the reference is identity-only; a non-empty value must be a real
            hash string (whitespace-only is rejected).
    """

    domain: ArtifactDomain
    artifact_kind: str
    identity: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ArtifactDomain):
            raise TypeError(
                "DomainArtifactRef.domain must be an ArtifactDomain, got "
                f"{type(self.domain).__name__} ({self.domain!r}); a bare string "
                "is rejected (fail-closed) -- use ArtifactDomain.from_value(...) "
                "or DomainArtifactRef.from_dict(...)"
            )
        for name in ("artifact_kind", "identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"DomainArtifactRef.{name} must be a non-empty string, got "
                    f"{value!r}"
                )
        if not isinstance(self.content_hash, str):
            raise TypeError(
                f"DomainArtifactRef.content_hash must be a str or '', got "
                f"{type(self.content_hash).__name__}"
            )
        if self.content_hash and not self.content_hash.strip():
            raise ValueError(
                "DomainArtifactRef.content_hash must be a real hash string or "
                f"'', got whitespace-only {self.content_hash!r}"
            )

    # -- canonical identity of the reference itself -------------------------

    @property
    def ref_hash(self) -> str:
        """Canonical SHA-256 of the reference tuple (domain/kind/id/hash).

        This is the identity of the *reference*, distinct from
        :attr:`content_hash`, which is the content identity of the artifact
        being referenced.  Deterministic across processes.
        """
        return stable_content_hex(
            tag="DomainArtifactRef",
            fields={
                "domain": self.domain.value,
                "artifact_kind": self.artifact_kind,
                "identity": self.identity,
                "content_hash": self.content_hash,
            },
        )

    def __hash__(self) -> int:
        # Stable across processes (PYTHONHASHSEED-independent), per QE-P0-03.
        return stable_hash(self.ref_hash)

    @property
    def is_content_addressed(self) -> bool:
        """True when the reference pins a non-empty content hash."""
        return bool(self.content_hash)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict."""
        payload: Dict[str, Any] = {
            "domain": self.domain.value,
            "artifact_kind": self.artifact_kind,
            "identity": self.identity,
        }
        if self.content_hash:
            payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainArtifactRef":
        """Rebuild from a :meth:`to_dict` payload (fail-closed on malformed)."""
        if not isinstance(data, Mapping):
            raise TypeError(
                f"DomainArtifactRef.from_dict requires a Mapping, got "
                f"{type(data).__name__}"
            )
        missing = [
            key for key in ("domain", "artifact_kind", "identity") if key not in data
        ]
        if missing:
            raise ValueError(
                "DomainArtifactRef.from_dict requires the keys 'domain', "
                f"'artifact_kind' and 'identity'; missing {missing} from keys "
                f"{sorted(str(k) for k in data)} -- a legacy loose ref must be "
                "migrated to a typed DomainArtifactRef"
            )
        return cls(
            domain=ArtifactDomain.from_value(data["domain"]),
            artifact_kind=data["artifact_kind"],
            identity=data["identity"],
            content_hash=data.get("content_hash", ""),
        )

    # -- construction convenience -------------------------------------------

    @classmethod
    def of(
        cls,
        domain: ArtifactDomain,
        identity: str,
        content_hash: str = "",
        artifact_kind: str = "",
    ) -> "DomainArtifactRef":
        """Build a ref, defaulting ``artifact_kind`` to the domain value."""
        domain = ArtifactDomain.from_value(domain)
        return cls(
            domain=domain,
            artifact_kind=artifact_kind or domain.value,
            identity=identity,
            content_hash=content_hash,
        )

    def as_tuple(self) -> Tuple[str, str, str, str]:
        """The (domain, kind, identity, content_hash) tuple form."""
        return (
            self.domain.value,
            self.artifact_kind,
            self.identity,
            self.content_hash,
        )

    def identity_only(self) -> "DomainArtifactRef":
        """The same reference with its content hash stripped.

        Used by :class:`EvaluationSpecIdentity` to reference WHAT was evaluated
        by identity alone, so a re-run over re-computed data (same logical
        artifact identity, new content) keeps the spec identity stable while
        the result content hash moves.
        """
        if not self.content_hash:
            return self
        return DomainArtifactRef(
            domain=self.domain,
            artifact_kind=self.artifact_kind,
            identity=self.identity,
            content_hash="",
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DomainArtifactRef):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()