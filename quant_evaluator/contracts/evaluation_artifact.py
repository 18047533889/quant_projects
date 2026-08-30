"""Canonical durable evaluation domain artifact (DLIB-QE-002, R55 audit #23/#24).

Splits the mutable, API-shaped :class:`EvaluationBundle` (in
``quant_evaluator.api.requests``) from a *canonical durable evaluation domain
artifact*.  :class:`EvaluationArtifact` is the single source of truth for a
completed evaluation: it references the inputs it was evaluated against
(factor value / label definition / policy / profile / split / snapshot /
universe), the evidence and diagnostics it produced, and the timing /
provenance of the run.

Design notes:

- **Three-part identity (R55 audit #23).**  One hash over the whole artifact
  conflates three things that must be separately addressable:

  1. :class:`EvaluationSpecIdentity` (``artifact.evaluation_spec_identity``)
     -- hash of the evaluation *spec only* (what was asked: metric set,
     universe, window/split, snapshot, policy/profile, treatment refs,
     parameters, version pins).  Stable even if the result changes, so
     "same evaluation?" checks survive a re-run.
  2. :attr:`evaluation_result_content_hash` -- hash of the *results only*
     (metric values, series digests, timing).  Changes with every re-run.
  3. :class:`ArtifactEnvelopeIdentity` (``artifact.evaluation_envelope_identity``)
     -- identity of the *envelope* (evaluation id + spec identity + result
     content hash + timestamps).  The cross-reference handle: this is what a
     downstream artifact stores when it says "produced by evaluation X".

  ``content_hash`` keeps its legacy DLIB-QE-002 meaning: the derived hash over
  all semantic fields, fail-closed on a forged value.

- **Typed cross-domain references (R55 audit #24).**  Every field that points
  at another artifact is a :class:`DomainArtifactRef` (domain enum + artifact
  kind + identity string + content hash), validated fail-closed: a bare
  string or ad-hoc dict is rejected at construction.

- **Deep immutability** -- every semantic mapping is stored as a recursively
  frozen :class:`FrozenMapping` (from ``metric_artifacts``), so mutating the
  caller's original dict after construction never changes the artifact, and
  nested mutation raises ``TypeError``.
- **References, not payloads** -- the artifact carries stable references
  rather than raw arrays, so it is small, durable, and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.domain_refs import (
    ArtifactDomain,
    DomainArtifactRef,
)
from quant_evaluator.contracts.metric_artifacts import FrozenMapping

__all__ = [
    "ArtifactDomain",
    "DomainArtifactRef",
    "EvaluationArtifact",
    "EvaluationSpecIdentity",
    "EvaluationResultContentHash",
    "ArtifactEnvelopeIdentity",
]


def _coerce_domain_ref(value: Any, name: str) -> Optional[DomainArtifactRef]:
    """Coerce ``value`` to a :class:`DomainArtifactRef` (fail-closed).

    Accepts ``None`` or an existing :class:`DomainArtifactRef`.  Anything else
    -- a bare string, an ad-hoc dict, a foreign ref object -- is rejected, so
    a loosely-typed reference can never reach the durable artifact.
    """
    if value is None:
        return None
    if isinstance(value, DomainArtifactRef):
        return value
    raise TypeError(
        f"EvaluationArtifact.{name} must be a DomainArtifactRef or None, got "
        f"{type(value).__name__} ({value!r}). Plain-string / dict references "
        "are rejected fail-closed (R55 audit #24) -- build one with "
        "DomainArtifactRef.of(ArtifactDomain.<...>, identity, content_hash=...)"
    )


def _coerce_domain_ref_tuple(
    value: Any, name: str
) -> Tuple[DomainArtifactRef, ...]:
    """Coerce ``value`` to a tuple of :class:`DomainArtifactRef` (fail-closed)."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(
            f"EvaluationArtifact.{name} must be an iterable of "
            f"DomainArtifactRef, got a {type(value).__name__} (fail-closed)"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"EvaluationArtifact.{name} must be an iterable of "
            f"DomainArtifactRef, got {type(value).__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, DomainArtifactRef):
            raise TypeError(
                f"EvaluationArtifact.{name}[{index}] must be a "
                f"DomainArtifactRef, got {type(item).__name__} ({item!r}) -- "
                "plain-string refs are rejected fail-closed (R55 audit #24)"
            )
    return items


def _freeze_mapping(value: Any, name: str) -> FrozenMapping:
    """Coerce ``value`` to a recursively-immutable FrozenMapping (fail-closed)."""
    if value is None:
        return FrozenMapping({})
    if not isinstance(value, Mapping):
        raise TypeError(
            f"EvaluationArtifact.{name} must be a Mapping or None, got "
            f"{type(value).__name__}"
        )
    return FrozenMapping(value)


def _freeze_str_tuple(value: Any, name: str) -> Tuple[str, ...]:
    """Coerce ``value`` to a tuple of non-empty strings (fail-closed)."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be an iterable of strings, got a "
            f"{type(value).__name__}"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an iterable of strings, got {type(value).__name__}"
        ) from exc
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string, got {item!r}")
    return items


def _ref_dict(ref: Optional[DomainArtifactRef]) -> Optional[Dict[str, Any]]:
    """Serialize a ref (or None) for the canonical / dict forms."""
    return None if ref is None else ref.to_dict()


def _ref_from_dict(data: Any) -> Optional[DomainArtifactRef]:
    """Rebuild a ref (or None) from its serialized form (fail-closed)."""
    if data is None:
        return None
    if isinstance(data, DomainArtifactRef):
        return data
    if isinstance(data, Mapping):
        return DomainArtifactRef.from_dict(data)
    raise TypeError(
        "expected a DomainArtifactRef, a ref dict, or None, got "
        f"{type(data).__name__} ({data!r})"
    )


@dataclass(frozen=True)
class EvaluationSpecIdentity:
    """Identity of the evaluation SPEC only -- what was asked (R55 audit #23).

    Covers everything that defines the *request* for an evaluation: the metric
    set, the universe / snapshot / split / policy / profile references, the
    treatment references, the subject references, and the free parameters.
    Two evaluations with the same spec identity asked the same question, even
    when their results differ because the underlying data changed.

    Reference fields are included by *identity* (``DomainArtifactRef.identity_only``)
    for the subject, so a re-run over re-computed data under the same logical
    artifact identity keeps the spec stable; the context refs (universe /
    split / snapshot / policy / profile) are carried as passed, since the
    caller owns whether their content version is part of the question.

    Attributes:
        metric_set:       Ordered tuple of metric ids requested.
        universe_ref:     :class:`DomainArtifactRef` to the universe.
        split_ref:        :class:`DomainArtifactRef` to the sealed split window.
        snapshot_ref:     :class:`DomainArtifactRef` to the data snapshot.
        policy_ref:       :class:`DomainArtifactRef` to the evaluation policy.
        profile_ref:      :class:`DomainArtifactRef` to the evaluation profile.
        treatment_refs:   Ordered tuple of treatment / recipe references.
        subject_refs:     Ordered tuple of identity-only refs naming WHAT was
            evaluated (factor value, label definition).
        parameters:       Free-form evaluation parameters (deep-frozen).
        producer_version / schema_version: version pins.
    """

    metric_set: Tuple[str, ...] = ()
    universe_ref: Optional[DomainArtifactRef] = None
    split_ref: Optional[DomainArtifactRef] = None
    snapshot_ref: Optional[DomainArtifactRef] = None
    policy_ref: Optional[DomainArtifactRef] = None
    profile_ref: Optional[DomainArtifactRef] = None
    treatment_refs: Tuple[DomainArtifactRef, ...] = ()
    subject_refs: Tuple[DomainArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    producer_version: str = "0.1"
    schema_version: str = "0.1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_set", _freeze_str_tuple(self.metric_set, "metric_set")
        )
        for name in (
            "universe_ref",
            "split_ref",
            "snapshot_ref",
            "policy_ref",
            "profile_ref",
        ):
            object.__setattr__(
                self, name, _coerce_domain_ref(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "treatment_refs",
            _coerce_domain_ref_tuple(self.treatment_refs, "treatment_refs"),
        )
        object.__setattr__(
            self,
            "subject_refs",
            _coerce_domain_ref_tuple(self.subject_refs, "subject_refs"),
        )
        object.__setattr__(
            self, "parameters", _freeze_mapping(self.parameters, "parameters")
        )
        for name in ("producer_version", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"EvaluationSpecIdentity.{name} must be a non-empty string, "
                    f"got {value!r}"
                )
        object.__setattr__(self, "_identity_hash", self._derive_identity_hash())

    def _canonical(self) -> Dict[str, Any]:
        """Canonical, hash-stable form of the spec."""
        return {
            "metric_set": list(self.metric_set),
            "universe_ref": _ref_dict(self.universe_ref),
            "split_ref": _ref_dict(self.split_ref),
            "snapshot_ref": _ref_dict(self.snapshot_ref),
            "policy_ref": _ref_dict(self.policy_ref),
            "profile_ref": _ref_dict(self.profile_ref),
            "treatment_refs": [_ref_dict(r) for r in self.treatment_refs],
            "subject_refs": [_ref_dict(r) for r in self.subject_refs],
            "parameters": dict(self.parameters),
            "producer_version": self.producer_version,
            "schema_version": self.schema_version,
        }

    def _derive_identity_hash(self) -> str:
        return stable_content_hex(tag="EvaluationSpecIdentity", fields=self._canonical())

    @property
    def identity_hash(self) -> str:
        """Canonical SHA-256 of the spec (derived-only, computed once)."""
        return self._identity_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "metric_set": list(self.metric_set),
            "universe_ref": _ref_dict(self.universe_ref),
            "split_ref": _ref_dict(self.split_ref),
            "snapshot_ref": _ref_dict(self.snapshot_ref),
            "policy_ref": _ref_dict(self.policy_ref),
            "profile_ref": _ref_dict(self.profile_ref),
            "treatment_refs": [_ref_dict(r) for r in self.treatment_refs],
            "subject_refs": [_ref_dict(r) for r in self.subject_refs],
            "parameters": dict(self.parameters),
            "producer_version": self.producer_version,
            "schema_version": self.schema_version,
            "identity_hash": self.identity_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationSpecIdentity":
        """Rebuild from a :meth:`to_dict` payload."""
        if not isinstance(data, Mapping):
            raise TypeError("EvaluationSpecIdentity.from_dict requires a mapping")
        return cls(
            metric_set=tuple(data.get("metric_set", ())),
            universe_ref=_ref_from_dict(data.get("universe_ref")),
            split_ref=_ref_from_dict(data.get("split_ref")),
            snapshot_ref=_ref_from_dict(data.get("snapshot_ref")),
            policy_ref=_ref_from_dict(data.get("policy_ref")),
            profile_ref=_ref_from_dict(data.get("profile_ref")),
            treatment_refs=tuple(
                _ref_from_dict(r) for r in data.get("treatment_refs", ())
            ),
            subject_refs=tuple(
                _ref_from_dict(r) for r in data.get("subject_refs", ())
            ),
            parameters=data.get("parameters", {}),
            producer_version=data.get("producer_version", "0.1"),
            schema_version=data.get("schema_version", "0.1"),
        )


@dataclass(frozen=True)
class EvaluationResultContentHash:
    """Content identity of the evaluation RESULTS only (R55 audit #23).

    Hashes what the run *produced* -- the metric values, the per-series
    digests, and the run's timing/provenance -- and deliberately NOT the spec.
    Two re-runs of the same spec over different data produce different result
    content hashes; two artifacts with identical results but different specs
    never collide on this field alone.

    Attributes:
        metric_values:  metric name -> value (deep-frozen).
        series_digests: series name -> canonical digest string (deep-frozen).
        timing:         timing / provenance mapping (deep-frozen).
        content_hash:   derived SHA-256 over the above (derived-only; a
            caller-supplied value must match exactly or ``ValueError``).
    """

    metric_values: Mapping[str, Any] = field(default_factory=dict)
    series_digests: Mapping[str, str] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric_values", _freeze_mapping(self.metric_values, "metric_values")
        )
        object.__setattr__(
            self,
            "series_digests",
            _freeze_mapping(self.series_digests, "series_digests"),
        )
        object.__setattr__(self, "timing", _freeze_mapping(self.timing, "timing"))
        derived = self._derive_content_hash()
        if self.content_hash and self.content_hash != derived:
            raise ValueError(
                "EvaluationResultContentHash.content_hash is DERIVED-ONLY and "
                f"does not match the computed value (got {self.content_hash!r}, "
                f"expected {derived!r})"
            )
        object.__setattr__(self, "content_hash", derived)

    def _canonical(self) -> Dict[str, Any]:
        return {
            "metric_values": dict(self.metric_values),
            "series_digests": dict(self.series_digests),
            "timing": dict(self.timing),
        }

    def _derive_content_hash(self) -> str:
        return stable_content_hex(
            tag="EvaluationResultContentHash", fields=self._canonical()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "metric_values": dict(self.metric_values),
            "series_digests": dict(self.series_digests),
            "timing": dict(self.timing),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResultContentHash":
        """Rebuild from a :meth:`to_dict` payload."""
        if not isinstance(data, Mapping):
            raise TypeError(
                "EvaluationResultContentHash.from_dict requires a mapping"
            )
        return cls(
            metric_values=data.get("metric_values", {}),
            series_digests=data.get("series_digests", {}),
            timing=data.get("timing", {}),
            content_hash=data.get("content_hash", ""),
        )


@dataclass(frozen=True)
class ArtifactEnvelopeIdentity:
    """Identity of the evaluation ENVELOPE (R55 audit #23).

    The addressable cross-reference handle for a completed evaluation: the
    evaluation id, the spec identity, the result content hash and the
    timestamps, combined into one canonical digest.  Downstream artifacts
    store this when they say "this factor set member was produced by
    evaluation X".
    """

    evaluation_id: str
    spec_identity: EvaluationSpecIdentity
    result_content_hash: str
    created_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip():
            raise ValueError(
                "ArtifactEnvelopeIdentity.evaluation_id must be a non-empty "
                f"string, got {self.evaluation_id!r}"
            )
        if not isinstance(self.spec_identity, EvaluationSpecIdentity):
            raise TypeError(
                "ArtifactEnvelopeIdentity.spec_identity must be an "
                f"EvaluationSpecIdentity, got {type(self.spec_identity).__name__}"
            )
        if not isinstance(self.result_content_hash, str):
            raise TypeError(
                "ArtifactEnvelopeIdentity.result_content_hash must be a str, got "
                f"{type(self.result_content_hash).__name__}"
            )
        if not isinstance(self.created_at, str):
            raise ValueError(
                "ArtifactEnvelopeIdentity.created_at must be a string, got "
                f"{type(self.created_at).__name__}"
            )
        object.__setattr__(self, "_identity_hash", self._derive_identity_hash())

    def _canonical(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "spec_identity": self.spec_identity._canonical(),
            "result_content_hash": self.result_content_hash,
            "created_at": self.created_at,
        }

    def _derive_identity_hash(self) -> str:
        return stable_content_hex(
            tag="ArtifactEnvelopeIdentity", fields=self._canonical()
        )

    @property
    def identity_hash(self) -> str:
        """Canonical SHA-256 over the whole envelope (derived-only)."""
        return self._identity_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "evaluation_id": self.evaluation_id,
            "spec_identity": self.spec_identity.to_dict(),
            "result_content_hash": self.result_content_hash,
            "created_at": self.created_at,
            "identity_hash": self.identity_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactEnvelopeIdentity":
        """Rebuild from a :meth:`to_dict` payload."""
        if not isinstance(data, Mapping):
            raise TypeError("ArtifactEnvelopeIdentity.from_dict requires a mapping")
        return cls(
            evaluation_id=data["evaluation_id"],
            spec_identity=EvaluationSpecIdentity.from_dict(
                data.get("spec_identity", {})
            ),
            result_content_hash=data.get("result_content_hash", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True)
class EvaluationArtifact:
    """Canonical durable evaluation domain artifact.

    Attributes:
        evaluation_id:          Stable identity of this evaluation.
        evaluation_identity:    Free-form identity / tag for the evaluation
            (e.g. a human-readable name or a run label).
        factor_value_ref:       Reference to the factor-value artifact
            evaluated (:class:`DomainArtifactRef`).
        label_definition_ref:   Reference to the label definition evaluated
            (:class:`DomainArtifactRef`).
        evaluation_policy_ref:  Reference to the evaluation policy applied
            (:class:`DomainArtifactRef`).
        evaluation_profile_ref: Reference to the evaluation profile applied
            (:class:`DomainArtifactRef`).
        split_ref:              Reference to the sealed split the evaluation
            was bound to (may be None).
        snapshot_ref:           Reference to the data snapshot evaluated.
        universe_ref:           Reference to the universe evaluated.
        metric_evidence_refs:   Tuple of metric-evidence
            :class:`DomainArtifactRef`.
        diagnostic_refs:        Tuple of diagnostic :class:`DomainArtifactRef`.
        spec_identity:          :class:`EvaluationSpecIdentity` -- identity of
            what was asked.  Derived from the artifact's own spec-relevant
            fields when omitted.
        result_content:         :class:`EvaluationResultContentHash` -- identity
            of the results (metric values / series digests / timing).
        envelope_identity:      :class:`ArtifactEnvelopeIdentity` -- identity of
            the envelope (id + spec + result hash + timestamps).
        timing:                 Timing / provenance mapping (deep-frozen).
        producer_version:       Version of the producer that created this.
        schema_version:         Schema version of this artifact.
        content_hash:           DERIVED-ONLY: computed in ``__post_init__``
            over the semantic fields.  A caller-supplied value must match
            exactly or ``ValueError`` is raised.
        created_at:             Free-form creation timestamp / tag.
    """

    evaluation_id: str
    evaluation_identity: str
    factor_value_ref: Optional[DomainArtifactRef] = None
    label_definition_ref: Optional[DomainArtifactRef] = None
    evaluation_policy_ref: Optional[DomainArtifactRef] = None
    evaluation_profile_ref: Optional[DomainArtifactRef] = None
    split_ref: Optional[DomainArtifactRef] = None
    snapshot_ref: Optional[DomainArtifactRef] = None
    universe_ref: Optional[DomainArtifactRef] = None
    metric_evidence_refs: Tuple[DomainArtifactRef, ...] = ()
    diagnostic_refs: Tuple[DomainArtifactRef, ...] = ()
    spec_identity: Optional[EvaluationSpecIdentity] = None
    result_content: Optional[EvaluationResultContentHash] = None
    envelope_identity: Optional[ArtifactEnvelopeIdentity] = None
    timing: Mapping[str, Any] = field(default_factory=dict)
    producer_version: str = "0.1"
    schema_version: str = "0.1"
    content_hash: str = ""
    created_at: str = ""

    _SEMANTIC_FIELDS = (
        "evaluation_id",
        "evaluation_identity",
        "factor_value_ref",
        "label_definition_ref",
        "evaluation_policy_ref",
        "evaluation_profile_ref",
        "split_ref",
        "snapshot_ref",
        "universe_ref",
        "metric_evidence_refs",
        "diagnostic_refs",
        "timing",
        "producer_version",
        "schema_version",
        "created_at",
    )

    def __post_init__(self) -> None:
        for name in (
            "evaluation_id",
            "evaluation_identity",
            "producer_version",
            "schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"EvaluationArtifact.{name} must be a non-empty string, got "
                    f"{value!r}"
                )
        if not isinstance(self.created_at, str):
            raise ValueError(
                "EvaluationArtifact.created_at must be a string, got "
                f"{type(self.created_at).__name__}"
            )
        # Typed cross-domain references (R55 audit #24): every reference field
        # is a validated DomainArtifactRef (or tuple of them).  Bare strings /
        # ad-hoc dicts are rejected fail-closed at construction.
        for name in (
            "factor_value_ref",
            "label_definition_ref",
            "evaluation_policy_ref",
            "evaluation_profile_ref",
            "split_ref",
            "snapshot_ref",
            "universe_ref",
        ):
            object.__setattr__(self, name, _coerce_domain_ref(getattr(self, name), name))
        object.__setattr__(
            self,
            "metric_evidence_refs",
            _coerce_domain_ref_tuple(self.metric_evidence_refs, "metric_evidence_refs"),
        )
        object.__setattr__(
            self,
            "diagnostic_refs",
            _coerce_domain_ref_tuple(self.diagnostic_refs, "diagnostic_refs"),
        )
        object.__setattr__(self, "timing", _freeze_mapping(self.timing, "timing"))
        # R55 #23: three-part identity.  The spec identity defaults to one
        # derived from the artifact's own spec-relevant fields; the result
        # content defaults to the run timing; the envelope combines them.
        spec = self.spec_identity
        if spec is None:
            spec = self._derive_default_spec_identity()
        elif not isinstance(spec, EvaluationSpecIdentity):
            raise TypeError(
                "EvaluationArtifact.spec_identity must be an "
                f"EvaluationSpecIdentity or None, got {type(spec).__name__}"
            )
        object.__setattr__(self, "spec_identity", spec)
        result = self.result_content
        if result is None:
            result = EvaluationResultContentHash(timing=dict(self.timing))
        elif not isinstance(result, EvaluationResultContentHash):
            raise TypeError(
                "EvaluationArtifact.result_content must be an "
                f"EvaluationResultContentHash or None, got {type(result).__name__}"
            )
        object.__setattr__(self, "result_content", result)
        envelope = self.envelope_identity
        if envelope is None:
            envelope = ArtifactEnvelopeIdentity(
                evaluation_id=self.evaluation_id,
                spec_identity=spec,
                result_content_hash=result.content_hash,
                created_at=self.created_at,
            )
        elif not isinstance(envelope, ArtifactEnvelopeIdentity):
            raise TypeError(
                "EvaluationArtifact.envelope_identity must be an "
                f"ArtifactEnvelopeIdentity or None, got {type(envelope).__name__}"
            )
        object.__setattr__(self, "envelope_identity", envelope)
        # Derived-only content hash: computed over the semantic fields; a
        # caller-supplied value that disagrees is a forged identity and raises.
        derived = self._derive_content_hash()
        if self.content_hash and self.content_hash != derived:
            raise ValueError(
                "EvaluationArtifact.content_hash is DERIVED-ONLY and does not "
                f"match the computed value (got {self.content_hash!r}, expected "
                f"{derived!r})"
            )
        object.__setattr__(self, "content_hash", derived)

    def _derive_default_spec_identity(self) -> EvaluationSpecIdentity:
        """Derive the spec identity from this artifact's spec-relevant fields.

        The subject refs (factor value / label definition) enter as
        identity-only projections, so a re-run over re-computed data under the
        same logical artifact identity yields the SAME spec identity while the
        result content hash moves.
        """
        subject_refs = tuple(
            ref.identity_only()
            for ref in (self.factor_value_ref, self.label_definition_ref)
            if ref is not None
        )
        return EvaluationSpecIdentity(
            universe_ref=self.universe_ref,
            split_ref=self.split_ref,
            snapshot_ref=self.snapshot_ref,
            policy_ref=self.evaluation_policy_ref,
            profile_ref=self.evaluation_profile_ref,
            subject_refs=subject_refs,
            producer_version=self.producer_version,
            schema_version=self.schema_version,
        )

    # -- R55 #23: the three identities --------------------------------------

    @property
    def evaluation_spec_identity(self) -> EvaluationSpecIdentity:
        """Identity of the evaluation SPEC only (what was asked).

        Stable across re-runs: two evaluations of the same spec over different
        data share this identity.
        """
        return self.spec_identity

    @property
    def evaluation_result_content_hash(self) -> str:
        """Content identity of the RESULTS only (metric values / digests).

        Changes with every re-run whose data changed.
        """
        return self.result_content.content_hash

    @property
    def evaluation_envelope_identity(self) -> ArtifactEnvelopeIdentity:
        """Identity of the envelope (id + spec identity + result hash + ts)."""
        return self.envelope_identity

    @property
    def artifact_hash(self) -> str:
        """LEGACY-COMPAT: the envelope identity hash.

        Kept so existing callers keep working; the three identities above are
        the primary API.  Exactly
        ``self.evaluation_envelope_identity.identity_hash``.
        """
        return self.envelope_identity.identity_hash

    # -- canonical forms / hashing ------------------------------------------

    def _canonical_semantic(self) -> Dict[str, Any]:
        """Canonical, hash-stable form of the semantic fields."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_identity": self.evaluation_identity,
            "factor_value_ref": _ref_dict(self.factor_value_ref),
            "label_definition_ref": _ref_dict(self.label_definition_ref),
            "evaluation_policy_ref": _ref_dict(self.evaluation_policy_ref),
            "evaluation_profile_ref": _ref_dict(self.evaluation_profile_ref),
            "split_ref": _ref_dict(self.split_ref),
            "snapshot_ref": _ref_dict(self.snapshot_ref),
            "universe_ref": _ref_dict(self.universe_ref),
            "metric_evidence_refs": [_ref_dict(r) for r in self.metric_evidence_refs],
            "diagnostic_refs": [_ref_dict(r) for r in self.diagnostic_refs],
            "timing": dict(self.timing),
            "producer_version": self.producer_version,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    def _derive_content_hash(self) -> str:
        """Derive the content hash over the semantic fields."""
        return stable_content_hex(
            tag="EvaluationArtifact", fields=self._canonical_semantic()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (all identities included)."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_identity": self.evaluation_identity,
            "factor_value_ref": _ref_dict(self.factor_value_ref),
            "label_definition_ref": _ref_dict(self.label_definition_ref),
            "evaluation_policy_ref": _ref_dict(self.evaluation_policy_ref),
            "evaluation_profile_ref": _ref_dict(self.evaluation_profile_ref),
            "split_ref": _ref_dict(self.split_ref),
            "snapshot_ref": _ref_dict(self.snapshot_ref),
            "universe_ref": _ref_dict(self.universe_ref),
            "metric_evidence_refs": [_ref_dict(r) for r in self.metric_evidence_refs],
            "diagnostic_refs": [_ref_dict(r) for r in self.diagnostic_refs],
            "spec_identity": self.spec_identity.to_dict(),
            "result_content": self.result_content.to_dict(),
            "envelope_identity": self.envelope_identity.to_dict(),
            "timing": dict(self.timing),
            "producer_version": self.producer_version,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationArtifact":
        """Rebuild from a :meth:`to_dict` payload.

        The persisted identities are restored faithfully when they agree with
        the payload; a persisted envelope whose ``evaluation_id`` disagrees
        with the payload's own ``evaluation_id`` is STALE (the caller mutated
        the identity after serializing) and is discarded so the envelope is
        recomputed from the actual fields rather than silently binding a wrong
        cross-reference.
        """
        if not isinstance(data, Mapping):
            raise TypeError("EvaluationArtifact.from_dict requires a mapping")
        persisted_envelope = data.get("envelope_identity")
        if isinstance(persisted_envelope, Mapping) and persisted_envelope.get(
            "evaluation_id"
        ) != data.get("evaluation_id"):
            persisted_envelope = None
        return cls(
            evaluation_id=data["evaluation_id"],
            evaluation_identity=data["evaluation_identity"],
            factor_value_ref=_ref_from_dict(data.get("factor_value_ref")),
            label_definition_ref=_ref_from_dict(data.get("label_definition_ref")),
            evaluation_policy_ref=_ref_from_dict(data.get("evaluation_policy_ref")),
            evaluation_profile_ref=_ref_from_dict(data.get("evaluation_profile_ref")),
            split_ref=_ref_from_dict(data.get("split_ref")),
            snapshot_ref=_ref_from_dict(data.get("snapshot_ref")),
            universe_ref=_ref_from_dict(data.get("universe_ref")),
            metric_evidence_refs=tuple(
                _ref_from_dict(r) for r in data.get("metric_evidence_refs", ())
            ),
            diagnostic_refs=tuple(
                _ref_from_dict(r) for r in data.get("diagnostic_refs", ())
            ),
            spec_identity=(
                EvaluationSpecIdentity.from_dict(data["spec_identity"])
                if isinstance(data.get("spec_identity"), Mapping)
                else None
            ),
            result_content=(
                EvaluationResultContentHash.from_dict(data["result_content"])
                if isinstance(data.get("result_content"), Mapping)
                else None
            ),
            envelope_identity=(
                ArtifactEnvelopeIdentity.from_dict(persisted_envelope)
                if persisted_envelope is not None
                   and not isinstance(persisted_envelope, (str, bytes))
                else None
            ),
            timing=data.get("timing", {}),
            producer_version=data.get("producer_version", "0.1"),
            schema_version=data.get("schema_version", "0.1"),
            content_hash=data.get("content_hash", ""),
            created_at=data.get("created_at", ""),
        )