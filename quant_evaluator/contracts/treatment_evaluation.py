"""Factor auto-treatment evaluation contracts (delta-vs-raw).

The factor auto-treatment optimizer evaluates candidate *treatments* (e.g.
preprocessing / factor transforms) against an un-treated *raw* baseline and
records how each candidate moved the metrics relative to that baseline.  Two
typed containers live here:

- :class:`DeltaMetrics` — the per-metric difference ``treatment - raw`` for a
  fixed set of evaluation fields.  A delta is ``None`` when the *raw* baseline
  metric is unavailable (fail-closed: a missing raw metric means that delta is
  unknown, never fabricated as ``0``).
- :class:`TreatmentEvaluationArtifact` — a frozen, self-describing evaluation
  of one treatment candidate against its raw baseline, with a derived-only
  ``content_hash`` so identical evaluations produce identical hashes.

Invariant (RAW is candidate 0): the raw / no-op treatment is ALWAYS the first
candidate in any treatment candidate list, and every treatment candidate must
be comparable to the raw baseline.  Preprocessing must never be assumed better
than raw; the raw baseline is the reference against which all deltas are
measured, so the evaluation artifact always references the RAW candidate's
evidence as ``raw_baseline_evidence_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex

__all__ = [
    "DeltaMetrics",
    "TreatmentEvaluationArtifact",
]

#: Field names of :class:`DeltaMetrics` in canonical order (single authority
#: for iteration / hashing / serialization).
_DELTA_FIELDS = (
    "delta_rank_ic",
    "delta_icir",
    "delta_turnover",
    "delta_cost_adjusted_alpha",
    "delta_worst_slice",
    "delta_exposure",
    "delta_coverage",
)


def _validate_finite_metric(name: str, value: Any, where: str) -> None:
    """Validate a single optional metric value as finite, non-boolean number.

    ``None`` is allowed (metric unavailable).  A present value must be a real
    (non-boolean) number and finite — ``NaN`` / ``inf`` are rejected
    fail-closed rather than silently propagating.
    """
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{where}.{name} must be a finite non-boolean number, got bool "
            f"{value!r} (True/False would silently coerce to 1.0/0.0)"
        )
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}.{name} must be a finite non-boolean number or None, got "
            f"{type(value).__name__}"
        )
    if not np.isfinite(value):
        raise ValueError(
            f"{where}.{name} must be finite, got {value!r} "
            "(NaN/inf is rejected fail-closed)"
        )


def _delta_as_dict(delta: "DeltaMetrics") -> Dict[str, Any]:
    """Return the delta's semantic fields as a plain dict (None-safe)."""
    return {name: getattr(delta, name) for name in _DELTA_FIELDS}


@dataclass(frozen=True)
class DeltaMetrics:
    """Per-metric delta of a treatment candidate relative to the raw baseline.

    Each field is ``treatment_metric - raw_metric`` for that metric.  ``None``
    means the *raw* baseline metric was unavailable, so the delta is unknown
    (fail-closed: never fabricated as ``0``).

    Attributes:
        delta_rank_ic:            change in mean rank IC
        delta_icir:               change in ICIR
        delta_turnover:           change in portfolio turnover
        delta_cost_adjusted_alpha: change in cost-adjusted alpha
        delta_worst_slice:        change in worst-quantile-slice performance
        delta_exposure:           change in factor exposure / concentration
        delta_coverage:           change in data coverage
    """

    delta_rank_ic: Optional[float] = None
    delta_icir: Optional[float] = None
    delta_turnover: Optional[float] = None
    delta_cost_adjusted_alpha: Optional[float] = None
    delta_worst_slice: Optional[float] = None
    delta_exposure: Optional[float] = None
    delta_coverage: Optional[float] = None

    def __post_init__(self) -> None:
        for name in _DELTA_FIELDS:
            _validate_finite_metric(name, getattr(self, name), "DeltaMetrics")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return _delta_as_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeltaMetrics":
        """Rebuild from a :meth:`to_dict` payload."""
        return cls(**{name: data.get(name) for name in _DELTA_FIELDS})


@dataclass(frozen=True)
class TreatmentEvaluationArtifact:
    """Evaluation of one treatment candidate against its raw baseline.

    Attributes:
        factor_id:                 Identifier of the factor being treated.
        treatment_id:              Identifier of the treatment candidate
            (``"NO_OP"`` / ``"RAW"`` denotes the raw baseline itself).
        raw_baseline_evidence_ref: Reference to the RAW / NO_OP candidate's
            evidence.  RAW is ALWAYS candidate 0, so every treatment artifact
            points back at the raw baseline it is measured against.
        treatment_evidence_ref:    Reference to this treatment candidate's
            evidence.
        absolute_metrics:          Raw metric_name -> value for this candidate
            (the candidate's own evaluation, before any delta).
        delta_vs_raw:              :class:`DeltaMetrics` relative to the raw
            baseline (``None`` fields where the raw metric is unavailable).
        snapshot_ref:              Snapshot reference the evaluation ran on.
        universe_ref:              Universe reference the evaluation ran on.
        split_ref:                 Split reference the evaluation ran on.
        created_at:                Free-form creation timestamp / tag.
        content_hash:              DERIVED-ONLY: computed in ``__post_init__``
            over the semantic fields.  A caller-supplied value must match the
            derived hash exactly or ``ValueError`` is raised; the caller cannot
            forge the content identity.
    """

    factor_id: str
    treatment_id: str
    raw_baseline_evidence_ref: str
    treatment_evidence_ref: str
    absolute_metrics: Mapping[str, float]
    delta_vs_raw: DeltaMetrics
    snapshot_ref: str
    universe_ref: str
    split_ref: str
    created_at: str
    content_hash: str = ""

    _SEMANTIC_FIELDS = (
        "factor_id",
        "treatment_id",
        "raw_baseline_evidence_ref",
        "treatment_evidence_ref",
        "absolute_metrics",
        "delta_vs_raw",
        "snapshot_ref",
        "universe_ref",
        "split_ref",
        "created_at",
    )

    def __post_init__(self) -> None:
        for name in (
            "factor_id",
            "treatment_id",
            "raw_baseline_evidence_ref",
            "treatment_evidence_ref",
            "snapshot_ref",
            "universe_ref",
            "split_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"TreatmentEvaluationArtifact.{name} must be a non-empty string, "
                    f"got {value!r}"
                )
        if not isinstance(self.created_at, str):
            raise ValueError(
                "TreatmentEvaluationArtifact.created_at must be a string, got "
                f"{type(self.created_at).__name__}"
            )
        if not isinstance(self.absolute_metrics, Mapping):
            raise ValueError(
                "TreatmentEvaluationArtifact.absolute_metrics must be a Mapping, got "
                f"{type(self.absolute_metrics).__name__}"
            )
        for m_name, m_value in self.absolute_metrics.items():
            if not isinstance(m_name, str) or not m_name.strip():
                raise ValueError(
                    "TreatmentEvaluationArtifact.absolute_metrics keys must be "
                    "non-empty strings"
                )
            _validate_finite_metric(m_name, m_value, "absolute_metrics")
        if not isinstance(self.delta_vs_raw, DeltaMetrics):
            raise ValueError(
                "TreatmentEvaluationArtifact.delta_vs_raw must be a DeltaMetrics, got "
                f"{type(self.delta_vs_raw).__name__}"
            )
        object.__setattr__(self, "absolute_metrics", dict(self.absolute_metrics))
        # Derived-only content hash: computed over the semantic fields; a
        # caller-supplied value that disagrees is a forged identity and raises.
        derived = self._derive_content_hash()
        if self.content_hash and self.content_hash != derived:
            raise ValueError(
                "TreatmentEvaluationArtifact.content_hash is DERIVED-ONLY and does "
                f"not match the computed value (got {self.content_hash!r}, expected "
                f"{derived!r})"
            )
        object.__setattr__(self, "content_hash", derived)

    def _canonical_semantic(self) -> Dict[str, Any]:
        """Canonical, hash-stable form of the semantic fields."""
        return {
            "factor_id": self.factor_id,
            "treatment_id": self.treatment_id,
            "raw_baseline_evidence_ref": self.raw_baseline_evidence_ref,
            "treatment_evidence_ref": self.treatment_evidence_ref,
            "absolute_metrics": dict(self.absolute_metrics),
            "delta_vs_raw": _delta_as_dict(self.delta_vs_raw),
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "split_ref": self.split_ref,
            "created_at": self.created_at,
        }

    def _derive_content_hash(self) -> str:
        """Derive the 16-hex content hash over the semantic fields."""
        return stable_content_hex(
            tag="TreatmentEvaluationArtifact", fields=self._canonical_semantic()
        )

    @property
    def is_raw_baseline(self) -> bool:
        """True when this artifact IS the raw baseline (NO_OP / RAW treatment)."""
        tid = self.treatment_id.strip().upper()
        return tid in ("NO_OP", "RAW")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (content_hash included)."""
        payload = {
            "factor_id": self.factor_id,
            "treatment_id": self.treatment_id,
            "raw_baseline_evidence_ref": self.raw_baseline_evidence_ref,
            "treatment_evidence_ref": self.treatment_evidence_ref,
            "absolute_metrics": dict(self.absolute_metrics),
            "delta_vs_raw": self.delta_vs_raw.to_dict(),
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "split_ref": self.split_ref,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TreatmentEvaluationArtifact":
        """Rebuild from a :meth:`to_dict` payload."""
        return cls(
            factor_id=data["factor_id"],
            treatment_id=data["treatment_id"],
            raw_baseline_evidence_ref=data["raw_baseline_evidence_ref"],
            treatment_evidence_ref=data["treatment_evidence_ref"],
            absolute_metrics=dict(data.get("absolute_metrics", {})),
            delta_vs_raw=DeltaMetrics.from_dict(data.get("delta_vs_raw", {})),
            snapshot_ref=data["snapshot_ref"],
            universe_ref=data["universe_ref"],
            split_ref=data["split_ref"],
            created_at=data["created_at"],
            content_hash=data.get("content_hash", ""),
        )
