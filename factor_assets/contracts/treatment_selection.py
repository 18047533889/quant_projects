"""Treatment-selection artifact contract for factor_assets (FA track).

A :class:`TreatmentSelectionArtifact` is a complete, immutable, hash-addressed
record of the *auto-treatment optimizer* decision for one factor: the raw
baseline evidence, the factor profile, the eligibility / search-space policies,
every trial that was evaluated, the Pareto-optimal candidates, the winning
treatment recipe, the metric/delta/dimension scores that justified it, the hard
gate and soft-floor results, robustness and complexity signals, and the full
snapshot/universe/split provenance.

``content_hash`` is sha256 over every semantic field and is *derived-only*: a
caller may not self-report an arbitrary hash — any supplied hash that does not
equal the recomputed value fails closed (``ValueError``).  This matches the
established immutable-artifact style of ``similarity.py`` and ``admission.py``.

Two hardening properties are built into the contract:

- **Recursive deep-freeze** (``_freeze``): mapping fields are frozen into a
  recursive immutable snapshot, so a nested ``dict``/``list`` inside e.g.
  ``winner_recipe`` cannot be mutated after construction.
- **Canonical structural hash** (``_canonical``): mapping fields are hashed via
  a deterministic, type-distinguishing, order-independent byte encoding rather
  than ``str(...)`` reprs, so structurally equal nested recipes hash identically
  and equal-looking values of different types hash differently.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Optional

__all__ = ["TreatmentSelectionArtifact"]


def _length_prefixed(digest: "hashlib._Hash", field_value: object) -> None:
    """Hash a field with length-prefixing so delimiters cannot collide."""
    encoded = str(field_value).encode("utf-8")
    digest.update(str(len(encoded)).encode("ascii"))
    digest.update(b":")
    digest.update(encoded)


def _freeze(value: object) -> object:
    """Recursively deep-freeze arbitrary nested structures into immutables.

    - dict / Mapping   -> MappingProxyType of deep-frozen values
    - list             -> tuple of deep-frozen elements
    - tuple            -> tuple of deep-frozen elements
    - set / frozenset  -> frozenset of deep-frozen elements
    - str/int/float/bool/None (and other scalars) pass through unchanged

    The snapshot is a private copy: mutating a caller-supplied nested list or
    dict after construction can no longer reach the frozen artifact.  Any other
    object type is rejected so the frozen artifact is provably immutable.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    raise TypeError(
        f"cannot deep-freeze value of unsupported type {type(value).__name__}"
    )


def _frozen(mapping: Mapping[str, object]) -> Mapping[str, object]:
    """Deep-freeze a mapping into an immutable recursive snapshot."""
    frozen = _freeze(mapping)
    if not isinstance(frozen, Mapping):
        raise TypeError("_frozen requires a mapping")
    return frozen


def _validate_str_map(values: Mapping[str, object], label: str) -> dict:
    """Validate a str->non-empty-str map. Returns a normalized copy."""
    normalized: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}[{key!r}] value must be a non-empty string")
        normalized[key] = value
    return normalized


def _validate_number_map(values: Mapping[str, object], label: str) -> dict:
    """Validate a str->finite non-bool number map. Returns a normalized copy."""
    normalized: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label}[{key!r}] must be a non-boolean number")
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{label}[{key!r}] must be finite")
        normalized[key] = number
    return normalized


def _canonical(value: object) -> bytes:
    """Canonical structural byte encoding of a nested object for hashing.

    Guarantees:
      - Structurally equal nested objects encode identically regardless of
        mapping insertion order or ``str(dict)`` repr quirks.
      - Types are distinguished (``1`` vs ``1.0`` vs ``"1"``; dict vs list vs
        tuple vs set), so equal-looking values of different types differ.
      - Every atomic and aggregate is length-prefixed, so delimiters cannot
        collide and no two distinct structures share a prefix/suffix encoding.
      - Containers are recursively canonicalized; no depth guard is needed for
        the bounded, validated artifact fields this module hashes.
    """
    if value is None:
        return b"N1:\x00"
    if isinstance(value, bool):
        return b"B1:" + (b"1" if value else b"0")
    if isinstance(value, str):
        body = value.encode("utf-8")
        return b"S" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, int):
        body = str(value).encode("ascii")
        return b"I" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, float):
        body = repr(value).encode("ascii")
        return b"F" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, Mapping):
        children = sorted(_canonical(k) + _canonical(v) for k, v in value.items())
        body = b"".join(children)
        return b"M" + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, (list, tuple)):
        body = b"".join(_canonical(v) for v in value)
        tag = b"L" if isinstance(value, list) else b"T"
        return tag + str(len(body)).encode("ascii") + b":" + body
    if isinstance(value, (set, frozenset)):
        body = b"".join(sorted(_canonical(v) for v in value))
        return b"E" + str(len(body)).encode("ascii") + b":" + body
    raise TypeError(
        f"cannot canonicalize value of unsupported type {type(value).__name__}"
    )


def _content_hash(
    factor_id: str,
    factor_version: str,
    raw_baseline_evidence_ref: str,
    factor_profile_ref: str,
    eligibility_policy_ref: str,
    search_space_ref: str,
    all_trial_refs: tuple[str, ...],
    pareto_candidate_refs: tuple[str, ...],
    winner_recipe: Mapping[str, object],
    winner_policy_identity: str,
    absolute_metric_refs: Mapping[str, object],
    delta_metric_refs: Mapping[str, object],
    dimension_scores: Mapping[str, object],
    hard_gate_results: Mapping[str, object],
    soft_floor_results: Mapping[str, object],
    robustness_evidence: Optional[str],
    complexity_score: Optional[float],
    snapshot_ref: str,
    universe_ref: str,
    split_ref: str,
) -> str:
    """sha256 over every semantic field of the treatment-selection artifact.

    Provenance bookkeeping (``created_at``, the derived ``content_hash`` itself)
    is intentionally excluded: two artifacts with identical decision content but
    different timestamps share a hash.  Mapping fields are hashed via
    ``_canonical`` (structural, order-independent, type-distinguishing); scalar
    and string-sequence fields remain length-prefixed.
    """
    digest = hashlib.sha256()
    for item in (
        factor_id,
        factor_version,
        raw_baseline_evidence_ref,
        factor_profile_ref,
        eligibility_policy_ref,
        search_space_ref,
        "#",
        *all_trial_refs,
        "#",
        *pareto_candidate_refs,
        "#",
    ):
        _length_prefixed(digest, item)
    for mapping in (
        winner_recipe,
        absolute_metric_refs,
        delta_metric_refs,
        dimension_scores,
        hard_gate_results,
        soft_floor_results,
    ):
        _length_prefixed(digest, _canonical(mapping))
    for item in (
        winner_policy_identity,
        robustness_evidence or "",
        complexity_score,
        snapshot_ref,
        universe_ref,
        split_ref,
    ):
        _length_prefixed(digest, item)
    return digest.hexdigest()


@dataclass(frozen=True)
class TreatmentSelectionArtifact:
    """Complete, hash-addressed record of the auto-treatment decision.

    Fields:
        factor_id: Factor being treated.
        factor_version: Version of the factor definition that was treated.
        raw_baseline_evidence_ref: Reference to the un-treated baseline evidence.
        factor_profile_ref: Reference to the factor profile used by the optimizer.
        eligibility_policy_ref: Reference to the eligibility policy consulted.
        search_space_ref: Reference to the treatment search space definition.
        all_trial_refs: References to every treatment trial evaluated.
        pareto_candidate_refs: References to the Pareto-optimal candidate trials.
        winner_recipe: The chosen treatment recipe (dict describing it).
        winner_policy_identity: Identity of the winning treatment policy.
        absolute_metric_refs: metric_name -> evidence_ref for absolute metrics.
        delta_metric_refs: delta-metric_name -> evidence_ref for improvement.
        dimension_scores: dimension -> desirability score (finite float).
        hard_gate_results: gate -> PASS/FAIL verdict.
        soft_floor_results: floor/gate -> verdict (str) or numeric score (float).
        robustness_evidence: Optional reference to robustness evidence.
        complexity_score: Optional complexity/simplicity score.
        snapshot_ref, universe_ref, split_ref: data provenance.
        created_at: ISO 8601 creation timestamp.
        content_hash: sha256 over every semantic field (derived-only).
    """

    factor_id: str
    factor_version: str
    raw_baseline_evidence_ref: str
    factor_profile_ref: str
    eligibility_policy_ref: str
    search_space_ref: str
    all_trial_refs: tuple[str, ...] = ()
    pareto_candidate_refs: tuple[str, ...] = ()
    winner_recipe: Mapping[str, object] = field(default_factory=dict)
    winner_policy_identity: str = ""
    absolute_metric_refs: Mapping[str, object] = field(default_factory=dict)
    delta_metric_refs: Mapping[str, object] = field(default_factory=dict)
    dimension_scores: Mapping[str, object] = field(default_factory=dict)
    hard_gate_results: Mapping[str, object] = field(default_factory=dict)
    soft_floor_results: Mapping[str, object] = field(default_factory=dict)
    robustness_evidence: Optional[str] = None
    complexity_score: Optional[float] = None
    snapshot_ref: str = ""
    universe_ref: str = ""
    split_ref: str = ""
    created_at: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.factor_version:
            raise ValueError("factor_version is required")

        # tuples of non-empty strings
        if not isinstance(self.all_trial_refs, (tuple, list)):
            raise TypeError("all_trial_refs must be a tuple/list of str")
        trials = tuple(self.all_trial_refs)
        if any(not isinstance(r, str) or not r for r in trials):
            raise ValueError("all_trial_refs must contain only non-empty strings")
        object.__setattr__(self, "all_trial_refs", trials)

        if not isinstance(self.pareto_candidate_refs, (tuple, list)):
            raise TypeError("pareto_candidate_refs must be a tuple/list of str")
        pareto = tuple(self.pareto_candidate_refs)
        if any(not isinstance(r, str) or not r for r in pareto):
            raise ValueError("pareto_candidate_refs must contain only non-empty strings")
        object.__setattr__(self, "pareto_candidate_refs", pareto)

        # str->str maps
        object.__setattr__(
            self,
            "absolute_metric_refs",
            _frozen(_validate_str_map(self.absolute_metric_refs, "absolute_metric_refs")),
        )
        object.__setattr__(
            self,
            "delta_metric_refs",
            _frozen(_validate_str_map(self.delta_metric_refs, "delta_metric_refs")),
        )
        object.__setattr__(
            self,
            "hard_gate_results",
            _frozen(_validate_str_map(self.hard_gate_results, "hard_gate_results")),
        )

        # soft_floor_results: str (verdict) or finite non-bool number (score)
        sf: dict[str, object] = {}
        for key, value in self.soft_floor_results.items():
            if not isinstance(key, str) or not key:
                raise ValueError("soft_floor_results keys must be non-empty strings")
            if isinstance(value, str):
                if not value:
                    raise ValueError("soft_floor_results str values must be non-empty")
                sf[key] = value
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"soft_floor_results[{key!r}] must be a non-empty string or number"
                )
            else:
                number = float(value)
                if number != number or number in (float("inf"), float("-inf")):
                    raise ValueError(f"soft_floor_results[{key!r}] must be finite")
                sf[key] = number
        object.__setattr__(self, "soft_floor_results", _frozen(sf))

        # dimension_scores: finite non-bool floats
        dims = _validate_number_map(self.dimension_scores, "dimension_scores")
        object.__setattr__(self, "dimension_scores", _frozen(dims))

        # winner_recipe: free-form mapping, frozen snapshot
        if not isinstance(self.winner_recipe, Mapping):
            raise TypeError("winner_recipe must be a mapping")
        object.__setattr__(self, "winner_recipe", _frozen(self.winner_recipe))

        if not self.winner_policy_identity:
            raise ValueError("winner_policy_identity is required")

        if not self.snapshot_ref or not self.universe_ref or not self.split_ref:
            raise ValueError("snapshot_ref / universe_ref / split_ref are required")

        if self.complexity_score is not None:
            if isinstance(self.complexity_score, bool) or not isinstance(
                self.complexity_score, (int, float)
            ):
                raise TypeError(
                    "complexity_score must be a non-boolean number or None"
                )
            cs = float(self.complexity_score)
            if cs != cs or cs in (float("inf"), float("-inf")):
                raise ValueError("complexity_score must be finite")
            object.__setattr__(self, "complexity_score", cs)

        computed_hash = _content_hash(
            self.factor_id,
            self.factor_version,
            self.raw_baseline_evidence_ref,
            self.factor_profile_ref,
            self.eligibility_policy_ref,
            self.search_space_ref,
            self.all_trial_refs,
            self.pareto_candidate_refs,
            self.winner_recipe,
            self.winner_policy_identity,
            self.absolute_metric_refs,
            self.delta_metric_refs,
            self.dimension_scores,
            self.hard_gate_results,
            self.soft_floor_results,
            self.robustness_evidence,
            self.complexity_score,
            self.snapshot_ref,
            self.universe_ref,
            self.split_ref,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed_hash)
        elif self.content_hash != computed_hash:
            raise ValueError(
                "content_hash does not match the recomputed treatment-selection "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )

    def to_dict(self) -> dict:
        """Serializable dict form (content_hash + created_at for audit)."""
        return {
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "raw_baseline_evidence_ref": self.raw_baseline_evidence_ref,
            "factor_profile_ref": self.factor_profile_ref,
            "eligibility_policy_ref": self.eligibility_policy_ref,
            "search_space_ref": self.search_space_ref,
            "all_trial_refs": list(self.all_trial_refs),
            "pareto_candidate_refs": list(self.pareto_candidate_refs),
            "winner_recipe": dict(self.winner_recipe),
            "winner_policy_identity": self.winner_policy_identity,
            "absolute_metric_refs": dict(self.absolute_metric_refs),
            "delta_metric_refs": dict(self.delta_metric_refs),
            "dimension_scores": dict(self.dimension_scores),
            "hard_gate_results": dict(self.hard_gate_results),
            "soft_floor_results": dict(self.soft_floor_results),
            "robustness_evidence": self.robustness_evidence,
            "complexity_score": self.complexity_score,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "split_ref": self.split_ref,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TreatmentSelectionArtifact":
        """Rebuild an artifact from its ``to_dict`` form.

        The recorded content_hash is preserved and re-verified against the
        recomputed hash during construction — an inconsistent hash fails closed.
        """
        return cls(
            factor_id=data["factor_id"],
            factor_version=data["factor_version"],
            raw_baseline_evidence_ref=data["raw_baseline_evidence_ref"],
            factor_profile_ref=data["factor_profile_ref"],
            eligibility_policy_ref=data["eligibility_policy_ref"],
            search_space_ref=data["search_space_ref"],
            all_trial_refs=tuple(data["all_trial_refs"]),
            pareto_candidate_refs=tuple(data["pareto_candidate_refs"]),
            winner_recipe=data["winner_recipe"],
            winner_policy_identity=data["winner_policy_identity"],
            absolute_metric_refs=data["absolute_metric_refs"],
            delta_metric_refs=data["delta_metric_refs"],
            dimension_scores=data["dimension_scores"],
            hard_gate_results=data["hard_gate_results"],
            soft_floor_results=data["soft_floor_results"],
            robustness_evidence=data.get("robustness_evidence"),
            complexity_score=data.get("complexity_score"),
            snapshot_ref=data["snapshot_ref"],
            universe_ref=data["universe_ref"],
            split_ref=data["split_ref"],
            created_at=data.get("created_at"),
            content_hash=data.get("content_hash", ""),
        )
