"""Carried identity *references* — the platform never mints domain identity.

R55 P0-5 (task #95): ``quant_platform`` is a thin integration / DTO layer. The
spec's identity axes (spec §8.1–§8.4 — factor definition, factor value,
evaluation, treatment) are **domain-owned**:

* factor definition / factor value — ``factor_assets/identity/``
  (+ ``factor_engine/runtime/factor_value_identity.py`` for the value axis);
* evaluation — ``quant_evaluator`` / ``factor_assets``;
* treatment — ``factor_assets`` treatment selection.

The platform therefore **carries** those identities as opaque, format-validated
references (the domain's own digest string + the slot values the domain hashed).
It never recomputes, re-negotiates or semantically judges them: this module
contains **no hash function at all** — only structural validation of the ref
the domain produced. Two platform actors cannot disagree on an identity because
neither of them computes it; the domain package that produced the digest is the
single authority.

What the platform DOES still own (and keeps hashing, see ``_contenthash.py``):
platform-governance artifacts over *carried* refs — ``ArtifactRef`` /
``FeatureSetVersion`` / report-export envelopes — plus transport codecs. Those
are platform objects, not domain semantics.

Contract: every ``*Ref`` here is a frozen value object that accepts the
domain's own digest verbatim (fail-closed on absent/malformed FORMAT only).
``ref.hash`` is the carried digest — never a recomputation.
"""

from __future__ import annotations

import re as _re
from typing import Any, Mapping

__all__ = [
    "sha256_hex",
    "IdentityRef",
    "FactorDefinitionRef",
    "FactorValueRef",
    "EvaluationRef",
    "TreatmentRef",
    "require_non_empty",
]

# 64-char lowercase hex sha256 — the *format* any domain digest must have to be
# carried in a platform DTO. Validating the format is transport duty; the digest
# itself is opaque to the platform.
_SHA256_RE = _re.compile(r"^[0-9a-f]{64}$")


def sha256_hex(value: Any, label: str) -> str:
    """Fail-closed format check: ``value`` must be a 64-char lowercase hex sha256.

    FORMAT ONLY — the digest is never recomputed and never semantically
    interpreted. This is the entire identity validation surface the platform
    is allowed to apply to a domain-owned identity.
    """
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a str sha256 hex digest, got {type(value).__name__}")
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{label} must be a 64-char lowercase hex sha256 minted by the owning "
            f"domain package — the platform carries it, never recomputes it "
            f"(got {value!r})"
        )
    return value


def require_non_empty(value: Any, label: str) -> str:
    """Non-empty string check (format only)."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


class IdentityRef:
    """Base for carried domain-identity references (spec §8 identity axes).

    A ref holds the domain package's own digest **verbatim**. ``hash`` returns
    that carried digest — there is no hashing anywhere in this
    module, so a platform process can never mint a domain identity of its own.
    """

    __slots__ = ("_hash", "_descriptor")

    _hash: str

    def __init__(self, digest: str, *, label: str = "digest") -> None:
        object.__setattr__(self, "_hash", sha256_hex(digest, label))
        object.__setattr__(self, "_descriptor", {})

    @property
    def hash(self) -> str:
        """The domain package's own digest, carried verbatim (opaque)."""
        return self._hash

    @property
    def descriptor(self) -> Mapping[str, Any]:
        """Opaque domain-provided descriptor (carried, never judged)."""
        return dict(self._descriptor)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._hash

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(hash={self._hash})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IdentityRef) and self._hash == other._hash

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._hash))


class FactorDefinitionRef(IdentityRef):
    """Carried spec §8.1 factor-DEFINITION identity (domain-owned).

    Minted by ``factor_assets/identity`` (its own factor-definition identity);
    the platform carries the digest and the opaque descriptor the domain hashed.
    No recomputation, no parameter-dict hashing, no 口径 judgment here.
    """

    __slots__ = ("factor_version",)

    def __init__(
        self,
        digest: str,
        *,
        factor_version: str = "",
        descriptor: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(digest, label="factor_definition_ref digest")
        object.__setattr__(
            self, "factor_version", require_non_empty(factor_version, "factor_version")
        )
        object.__setattr__(self, "_descriptor", dict(descriptor or {}))


class FactorValueRef(IdentityRef):
    """Carried spec §8.2 factor-VALUE identity (domain-owned)."""

    __slots__ = ("factor_definition_ref",)

    def __init__(self, digest: str, *, factor_definition_ref: str = "") -> None:
        super().__init__(digest, label="factor_value_ref digest")
        object.__setattr__(
            self,
            "factor_definition_ref",
            require_non_empty(factor_definition_ref, "factor_definition_ref"),
        )


class EvaluationRef(IdentityRef):
    """Carried spec §8.3 evaluation identity (domain-owned, quant_evaluator)."""

    __slots__ = ("factor_value_ref",)

    def __init__(self, digest: str, *, factor_value_ref: str = "") -> None:
        super().__init__(digest, label="evaluation_ref digest")
        object.__setattr__(
            self, "factor_value_ref", require_non_empty(factor_value_ref, "factor_value_ref")
        )


class TreatmentRef(IdentityRef):
    """Carried spec §8.4 treatment identity (domain-owned, factor_assets)."""

    __slots__ = ("source_factor_value_ref",)

    def __init__(self, digest: str, *, source_factor_value_ref: str = "") -> None:
        super().__init__(digest, label="treatment_ref digest")
        object.__setattr__(
            self,
            "source_factor_value_ref",
            require_non_empty(source_factor_value_ref, "source_factor_value_ref"),
        )