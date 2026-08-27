"""Canonical durable evaluation domain artifact (DLIB-QE-002).

Splits the mutable, API-shaped :class:`EvaluationBundle` (in
``quant_evaluator.api.requests``) from a *canonical durable evaluation domain
artifact*.  :class:`EvaluationArtifact` is the single source of truth for a
completed evaluation: it references the inputs it was evaluated against
(factor value / label definition / policy / profile / split / snapshot /
universe), the evidence and diagnostics it produced, and the timing /
provenance of the run.

Design notes:

- **Deep immutability** — every semantic mapping is stored as a recursively
  frozen :class:`FrozenMapping` (from ``metric_artifacts``), so mutating the
  caller's original dict after construction never changes the artifact, and
  nested mutation raises ``TypeError``.
- **Derived-only content hash** — ``content_hash`` is computed in
  ``__post_init__`` over the semantic fields.  A caller-supplied value that
  disagrees with the derived hash raises ``ValueError`` (fail-closed): the
  caller cannot forge the content identity.
- **References, not payloads** — the artifact carries stable references
  (``factor_value_ref``, ``label_definition_ref``, ...) rather than raw
  arrays, so it is small, durable, and serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.metric_artifacts import FrozenMapping

__all__ = ["EvaluationArtifact"]


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


def _freeze_tuple(value: Any, name: str) -> tuple:
    """Coerce ``value`` to a tuple (fail-closed on non-iterable)."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"EvaluationArtifact.{name} must be an iterable of refs, got a "
            f"{type(value).__name__}"
        )
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"EvaluationArtifact.{name} must be an iterable, got "
            f"{type(value).__name__}"
        ) from exc


@dataclass(frozen=True)
class EvaluationArtifact:
    """Canonical durable evaluation domain artifact.

    Attributes:
        evaluation_id:          Stable identity of this evaluation.
        evaluation_identity:    Free-form identity / tag for the evaluation
            (e.g. a human-readable name or a run label).
        factor_value_ref:       Reference to the factor-value artifact
            evaluated (a :class:`FactorValueRef`-shaped dict or the ref).
        label_definition_ref:   Reference to the label definition evaluated
            (a :class:`LabelBundleRef`-shaped dict or the ref).
        evaluation_policy_ref:  Reference to the evaluation policy applied.
        evaluation_profile_ref: Reference to the evaluation profile applied.
        split_ref:              Reference to the sealed split the evaluation
            was bound to (may be None).
        snapshot_ref:           Reference to the data snapshot evaluated.
        universe_ref:           Reference to the universe evaluated.
        metric_evidence_refs:   Iterable of metric-evidence references.
        diagnostic_refs:        Iterable of diagnostic references.
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
    factor_value_ref: Any
    label_definition_ref: Any
    evaluation_policy_ref: Any
    evaluation_profile_ref: Any
    split_ref: Any = None
    snapshot_ref: Any = None
    universe_ref: Any = None
    metric_evidence_refs: tuple = ()
    diagnostic_refs: tuple = ()
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
        # Deep-freeze all semantic mappings / iterables so the artifact owns
        # its content and the caller's original structures cannot mutate it.
        object.__setattr__(
            self, "factor_value_ref", _freeze_mapping(self.factor_value_ref, "factor_value_ref")
        )
        object.__setattr__(
            self, "label_definition_ref", _freeze_mapping(self.label_definition_ref, "label_definition_ref")
        )
        object.__setattr__(
            self, "evaluation_policy_ref", _freeze_mapping(self.evaluation_policy_ref, "evaluation_policy_ref")
        )
        object.__setattr__(
            self, "evaluation_profile_ref", _freeze_mapping(self.evaluation_profile_ref, "evaluation_profile_ref")
        )
        object.__setattr__(self, "split_ref", _freeze_mapping(self.split_ref, "split_ref"))
        object.__setattr__(self, "snapshot_ref", _freeze_mapping(self.snapshot_ref, "snapshot_ref"))
        object.__setattr__(self, "universe_ref", _freeze_mapping(self.universe_ref, "universe_ref"))
        object.__setattr__(
            self, "metric_evidence_refs", _freeze_tuple(self.metric_evidence_refs, "metric_evidence_refs")
        )
        object.__setattr__(
            self, "diagnostic_refs", _freeze_tuple(self.diagnostic_refs, "diagnostic_refs")
        )
        object.__setattr__(self, "timing", _freeze_mapping(self.timing, "timing"))
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

    def _canonical_semantic(self) -> Dict[str, Any]:
        """Canonical, hash-stable form of the semantic fields."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_identity": self.evaluation_identity,
            "factor_value_ref": dict(self.factor_value_ref),
            "label_definition_ref": dict(self.label_definition_ref),
            "evaluation_policy_ref": dict(self.evaluation_policy_ref),
            "evaluation_profile_ref": dict(self.evaluation_profile_ref),
            "split_ref": dict(self.split_ref),
            "snapshot_ref": dict(self.snapshot_ref),
            "universe_ref": dict(self.universe_ref),
            "metric_evidence_refs": list(self.metric_evidence_refs),
            "diagnostic_refs": list(self.diagnostic_refs),
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
        """Serialize to a JSON-friendly plain dict (content_hash included)."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_identity": self.evaluation_identity,
            "factor_value_ref": dict(self.factor_value_ref),
            "label_definition_ref": dict(self.label_definition_ref),
            "evaluation_policy_ref": dict(self.evaluation_policy_ref),
            "evaluation_profile_ref": dict(self.evaluation_profile_ref),
            "split_ref": dict(self.split_ref),
            "snapshot_ref": dict(self.snapshot_ref),
            "universe_ref": dict(self.universe_ref),
            "metric_evidence_refs": list(self.metric_evidence_refs),
            "diagnostic_refs": list(self.diagnostic_refs),
            "timing": dict(self.timing),
            "producer_version": self.producer_version,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationArtifact":
        """Rebuild from a :meth:`to_dict` payload."""
        if not isinstance(data, Mapping):
            raise TypeError("EvaluationArtifact.from_dict requires a mapping")
        return cls(
            evaluation_id=data["evaluation_id"],
            evaluation_identity=data["evaluation_identity"],
            factor_value_ref=data.get("factor_value_ref", {}),
            label_definition_ref=data.get("label_definition_ref", {}),
            evaluation_policy_ref=data.get("evaluation_policy_ref", {}),
            evaluation_profile_ref=data.get("evaluation_profile_ref", {}),
            split_ref=data.get("split_ref"),
            snapshot_ref=data.get("snapshot_ref"),
            universe_ref=data.get("universe_ref"),
            metric_evidence_refs=tuple(data.get("metric_evidence_refs", ())),
            diagnostic_refs=tuple(data.get("diagnostic_refs", ())),
            timing=data.get("timing", {}),
            producer_version=data.get("producer_version", "0.1"),
            schema_version=data.get("schema_version", "0.1"),
            content_hash=data.get("content_hash", ""),
            created_at=data.get("created_at", ""),
        )
