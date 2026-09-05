"""Evidence-bound metric values: missing evidence is never a fabricated 0.0 (R61-FI-030).

Plan §1.5 / matrix E2 / E-TDD-001: the historical ``TreatmentMetrics`` carried
every metric as a bare ``float`` defaulting to ``0.0``, so a candidate whose
RankIC / coverage / bootstrap was MISSING scored exactly like a candidate whose
value was genuinely zero.  A pipeline cannot tell "no evidence" from "evidence
says zero", and ``select_winner`` / STEP-6 would happily rank a missing
bootstrap series as full confidence.

This module supplies the evidence-bound vocabulary the pipeline consumes:

- :class:`EvidenceStatus` — the canonical 8-token *consumer* status vocabulary,
  mapping 1:1 onto QE's ``quant_evaluator.contracts.evidence_status``
  ``EvidenceStatus`` (COMPUTED / NOT_COMPUTED / UNAVAILABLE / UNSUPPORTED /
  INSUFFICIENT_DATA / LABEL_NOT_MATURE / INVALID_EVIDENCE / FAILED) without
  importing quant_evaluator (QE is an optional dependency here).  FA-side
  graded evidence (a 9-dim / 14-dim health-card metric) is additionally exposed
  as the typed string ``SEALED_TEST_CONFIRMED`` on the evidence tier —
  see below.
- :class:`EvidenceTier` — the evidence-quality ladder the winner policy
  declares a minimum on (plan §21 / E2): POINT_ESTIMATE_ONLY /
  VALIDATION_SERIES / BOOTSTRAP_CONFIDENCE / MULTIPLE_TESTING_ADJUSTED /
  SEALED_TEST_CONFIRMED.  ``EvidenceStatus.COMPUTED`` alone never implies a
  tier; a tier is a *declaration* about how much uncertainty work stands
  behind a value.
- :class:`EvidenceValue` — a frozen ``value: float | None`` + ``status`` pair.
  ``None`` means "no numeric evidence exists"; it is never coerced to 0.0, and
  every aggregation below treats a missing value as a missing *observation*
  (the aggregate itself becomes ``None``), not as a zero contribution.
- :class:`EvidenceSeries` — the bootstrap/validation-sample container whose
  empty state is explicit (``None`` series = no uncertainty work was done).

No grading rule, threshold or magic confidence number lives here.  These
vocabulary types only *annotate* evidence produced elsewhere (QE metrics /
FA health); FO policy decides what the statuses mean for admission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

__all__ = [
    "EvidenceStatus",
    "EvidenceTier",
    "EvidenceValue",
    "EvidenceSeries",
    "status_of",
    "QE_STATUS_TOKENS",
    "EVIDENCE_TIER_ORDER",
]


#: QE EvidenceStatus string tokens (reference-only; no quant_evaluator import).
QE_STATUS_TOKENS = (
    "computed",
    "not_computed",
    "unavailable",
    "unsupported",
    "insufficient_data",
    "label_not_mature",
    "invalid_evidence",
    "failed",
)


class EvidenceStatus(str, Enum):
    """Whether numeric evidence for a metric exists and can be trusted.

    String-typed (values compare equal to their token) so serialized FO
    artifacts match QE's ``EvidenceStatus`` values verbatim.  ``COMPUTED`` is
    the only token that implies a numeric value may be read off the artifact.
    """

    COMPUTED = "computed"
    NOT_COMPUTED = "not_computed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_DATA = "insufficient_data"
    LABEL_NOT_MATURE = "label_not_mature"
    INVALID_EVIDENCE = "invalid_evidence"
    FAILED = "failed"

    @classmethod
    def from_value(cls, value: object) -> "EvidenceStatus":
        """Resolve a str/enum value to an EvidenceStatus (fail closed)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown EvidenceStatus value {value!r}; expected one of "
                    f"{[m.value for m in cls]}"
                ) from None
        raise TypeError(
            f"EvidenceStatus.from_value expects str or EvidenceStatus, got "
            f"{type(value).__name__}"
        )

    @property
    def computed(self) -> bool:
        """True only when the status is exactly COMPUTED."""
        return self is EvidenceStatus.COMPUTED


#: Ordered evidence tiers from weakest to strongest.  A winner policy declares
#: ``minimum_evidence_tier``; any candidate below that tier is not a valid
#: winner.
EVIDENCE_TIER_ORDER = (
    "POINT_ESTIMATE_ONLY",
    "VALIDATION_SERIES",
    "BOOTSTRAP_CONFIDENCE",
    "MULTIPLE_TESTING_ADJUSTED",
    "SEALED_TEST_CONFIRMED",
)


class EvidenceTier(str, Enum):
    """The evidence-quality ladder a candidate's value stands on (plan §21/E2).

    Tiers are ordered weakest -> strongest.  A candidate carries the tier of
    its *weakest* load-bearing evidence, and a policy's
    ``minimum_evidence_tier`` hard-rejects anything below it.

    POINT_ESTIMATE_ONLY        one number, no uncertainty work.
    VALIDATION_SERIES          an out-of-sample/validation series exists.
    BOOTSTRAP_CONFIDENCE       bootstrap CI / resampled distribution computed.
    MULTIPLE_TESTING_ADJUSTED  adjusted for the hypothesis multiplicity.
    SEALED_TEST_CONFIRMED      survived a one-shot, sealed test (highest).
    """

    POINT_ESTIMATE_ONLY = "POINT_ESTIMATE_ONLY"
    VALIDATION_SERIES = "VALIDATION_SERIES"
    BOOTSTRAP_CONFIDENCE = "BOOTSTRAP_CONFIDENCE"
    MULTIPLE_TESTING_ADJUSTED = "MULTIPLE_TESTING_ADJUSTED"
    SEALED_TEST_CONFIRMED = "SEALED_TEST_CONFIRMED"

    @classmethod
    def from_value(cls, value: object) -> "EvidenceTier":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown EvidenceTier value {value!r}; expected one of "
                    f"{[m.value for m in cls]}"
                ) from None
        raise TypeError(
            f"EvidenceTier.from_value expects str or EvidenceTier, got "
            f"{type(value).__name__}"
        )

    @property
    def rank(self) -> int:
        """Ordinal rank (0 weakest .. 4 strongest)."""
        return EVIDENCE_TIER_ORDER.index(self.value)

    def meets(self, minimum: "EvidenceTier") -> bool:
        """True when this tier is at least as strong as ``minimum``."""
        return self.rank >= EvidenceTier.from_value(minimum).rank


@dataclass(frozen=True)
class EvidenceValue:
    """A single metric value bound to its evidence status.

    ``value`` is ``None`` exactly when no numeric evidence exists — a missing
    value is never fabricated as ``0.0``.  A non-``None`` value is only legal
    when ``status`` is ``COMPUTED`` (the status that implies a numeric payload
    may be read); a caller that wants to record *why* a value is absent keeps
    ``value=None`` and reads ``status`` / ``reason``.

    Attributes:
        metric_id: Metric/dimension id this value describes (e.g. ``rank_ic``,
            or a FA health dimension id).
        value: The numeric value when computed, else ``None``.
        status: :class:`EvidenceStatus` — ``COMPUTED`` iff a value exists.
        tier: :class:`EvidenceTier` declaration of the uncertainty work behind
            this value.
        source_ref: Optional provenance ref (QE evaluation_ref / evidence
            bundle ref / FA artifact ref).
        reason: Optional human-readable reason when the value is absent.
    """

    metric_id: str
    value: Optional[float] = None
    status: EvidenceStatus = EvidenceStatus.NOT_COMPUTED
    tier: EvidenceTier = EvidenceTier.POINT_ESTIMATE_ONLY
    source_ref: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.metric_id, str) or not self.metric_id.strip():
            raise ValueError("metric_id must be a non-empty string")
        object.__setattr__(self, "status", EvidenceStatus.from_value(self.status))
        object.__setattr__(self, "tier", EvidenceTier.from_value(self.tier))
        if not isinstance(self.source_ref, str):
            raise TypeError("source_ref must be a string")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if self.value is None:
            if self.status is EvidenceStatus.COMPUTED:
                raise ValueError(
                    "value=None is illegal with status=COMPUTED — a computed "
                    "value must carry its numeric payload; use a non-COMPUTED "
                    "status (e.g. NOT_COMPUTED / INSUFFICIENT_DATA) for an "
                    "absent value (fail closed, missing != 0)"
                )
            return
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("value must be a non-boolean number or None")
        if not math.isfinite(float(self.value)):
            raise ValueError("value must be finite (NaN/Inf is a failed value)")
        object.__setattr__(self, "value", float(self.value))
        if self.status is not EvidenceStatus.COMPUTED:
            raise ValueError(
                f"a numeric value with status={self.status.value!r} is "
                "illegal — only status=COMPUTED may carry a numeric payload "
                "(missing evidence must stay value=None, never a fabricated "
                "number)"
            )

    @property
    def present(self) -> bool:
        """True when a numeric value is available (status COMPUTED)."""
        return self.value is not None

    def to_dict(self) -> dict:
        return {
            "metric_id": self.metric_id,
            "value": self.value,
            "status": self.status.value,
            "tier": self.tier.value,
            "source_ref": self.source_ref,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "EvidenceValue":
        return cls(
            metric_id=data["metric_id"],
            value=data.get("value"),
            status=EvidenceStatus.from_value(data.get("status", "not_computed")),
            tier=EvidenceTier.from_value(
                data.get("tier", "POINT_ESTIMATE_ONLY")
            ),
            source_ref=data.get("source_ref", ""),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class EvidenceSeries:
    """Bootstrap / validation resampled series with an explicit empty state.

    The historical bootstrap store was ``Dict[str, Sequence[float]]`` defaulting
    to ``{}`` — an empty dict meant "no uncertainty work", which STEP 6 could
    not distinguish from "full confidence".  Here the *absence* is typed: a
    candidate with ``bootstrap=None`` has no uncertainty evidence at all (its
    values are POINT_ESTIMATE_ONLY), while a candidate whose bootstrap work
    failed records an explicit non-COMPUTED status instead of an empty list.

    Attributes:
        samples: Resampled values (one per bootstrap iteration).  Empty when
            ``status`` is non-COMPUTED; non-empty only when COMPUTED.
        status: COMPUTED iff the resampling genuinely ran and produced samples.
        n_iterations: Number of resampled draws (len of samples when COMPUTED).
    """

    samples: Tuple[float, ...] = ()
    status: EvidenceStatus = EvidenceStatus.NOT_COMPUTED

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(float(s) for s in self.samples))
        object.__setattr__(self, "status", EvidenceStatus.from_value(self.status))
        if self.samples and self.status is not EvidenceStatus.COMPUTED:
            raise ValueError(
                "a non-empty sample series with a non-COMPUTED status is "
                "illegal — samples are only present when the resampling ran"
            )
        if not self.samples and self.status is EvidenceStatus.COMPUTED:
            raise ValueError(
                "status=COMPUTED requires at least one bootstrap sample "
                "(an empty COMPUTED series would be a fabricated pass)"
            )
        for value in self.samples:
            if not math.isfinite(value):
                raise ValueError(
                    "bootstrap samples must be finite (NaN/Inf is a failed "
                    "resample, not evidence)"
                )

    @property
    def present(self) -> bool:
        return self.status is EvidenceStatus.COMPUTED and bool(self.samples)

    def to_dict(self) -> dict:
        return {
            "samples": list(self.samples),
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "EvidenceSeries":
        return cls(
            samples=data.get("samples", ()),
            status=EvidenceStatus.from_value(data.get("status", "not_computed")),
        )


def status_of(value: object) -> EvidenceStatus:
    """Coerce a QE ``EvidenceStatus`` / FA status string into our enum.

    Accepts :class:`EvidenceStatus`, the QE enum (duck-typed by ``.value``)
    and any string token.  Unknown tokens fail closed — FO never invents a
    status vocabulary.
    """
    if isinstance(value, EvidenceStatus):
        return value
    token = getattr(value, "value", value)
    return EvidenceStatus.from_value(token)
