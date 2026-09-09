"""Library promotion / rollback decision gate (QRP-P7).

A :class:`PromotionGate` and :class:`RollbackGate` are **decision pure
functions** — they read candidate evidence + library lineage and produce an
immutable, content-hashed decision artifact. They never write the library,
never touch the DB / release tooling, and never create a second Authority: the
``FactorLibraryVersionArtifact`` remains the single source of truth for the
library body, and promotion / rollback (upstream, by the library operator)
still only change the active pointer.

Fail-closed promotion rules (matching Memory: vwap-to-vwap 收益口径 — the
global hard return basis is ``Vwap.pct_change().shift(-1)``):

* label not mature (computed=False / evidence not mature) -> REJECT
  (reason ``label_not_mature``);
* return basis is not vwap->vwap -> REJECT (defensive, ``return_basis_wrong``);
* rank_ic / performance below threshold -> REJECT (``rank_ic_below_threshold``);
* candidate similar to an existing library member above the threshold -> REVIEW
  with ``merge_suggested`` (or hard REJECT with
  ``duplicate_of_existing_member`` when ``reject_duplicates=True``, the
  default) — a duplicate must be **merged**, not promoted;
* a candidate whose uniqueness cannot be measured (no similarity_fn, or an
  unmeasured ``None`` similarity) is forced to REVIEW, never silent APPROVE.

``RollbackGate`` validates lineage reachability: a library version may only
roll back to ancestors along **its own lineage chain** (parent refs), never
across to an unrelated version. No lineage link -> REJECT.

The module is pure-python with no runtime dependencies on ``quant_evaluator``
/ ``factor_optimizer`` — the EvidenceStatus values and the
``library_snapshot_ref`` semantics are only *referenced* (string / optional-ref
shapes), so this module imports cleanly in ``factor_assets`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional, Sequence

from factor_assets.contracts._canonical import canonical_digest

__all__ = [
    "PromotionDecision",
    "PromotionReasonCode",
    "VWAP_TO_VWAP_BASIS",
    "EVIDENCE_NOT_COMPUTED",
    "DEFAULT_MIN_RANK_IC",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_REJECT_DUPLICATES",
    "CandidateEvaluationRef",
    "PromotionDecisionArtifact",
    "RollbackDecisionArtifact",
    "PromotionGate",
    "RollbackGate",
]


#: Canonical vwap->vwap return-basis marker. Mirrors the convention carried on
#: ``quant_evaluator.contracts.label_bundle.LabelBundle.price_convention``
#: ("vwap_to_vwap") — see Memory: vwap-to-vwap 收益口径.
VWAP_TO_VWAP_BASIS = "vwap_to_vwap"

#: QE ``EvidenceStatus`` values that mean the label is NOT mature / evidence is
#: NOT computed (reference-only string values — do not import quant_evaluator).
#: Any of these fail the label-maturity gate closed.
EVIDENCE_NOT_COMPUTED: tuple[str, ...] = (
    "not_computed",
    "label_not_mature",
    "insufficient_data",
    "unavailable",
    "unsupported",
    "invalid_evidence",
    "failed",
)

#: Default minimum abs rank_ic for promotion. Matches the FA admission
#: ``MinimumICGate`` default (0.02) — promotion is at least as strict as
#: admission, so a candidate that cannot pass the admission IC floor may not
#: be promoted.
DEFAULT_MIN_RANK_IC = 0.02

#: Default similarity threshold above which a candidate is deemed a duplicate
#: of an existing library member and must be merged, not promoted. Matches the
#: FA admission ``similarity_threshold`` default.
DEFAULT_SIMILARITY_THRESHOLD = 0.7

#: Default: enforce the duplicate gate hard (REJECT). A caller that wants the
#: soft "suggest merge" REVIEW can opt out with ``reject_duplicates=False``.
DEFAULT_REJECT_DUPLICATES = True


class PromotionDecision(Enum):
    """Decision of the promotion / rollback gate (fail-closed)."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class PromotionReasonCode(str, Enum):
    """Machine-readable reason codes recorded on a decision.

    The REJECT codes are fail-closed: any single one makes the promotion
    negative regardless of every other input.
    """

    QUALIFIES = "qualifies"
    LABEL_NOT_MATURE = "label_not_mature"
    RETURN_BASIS_WRONG = "return_basis_wrong"
    RANK_IC_BELOW_THRESHOLD = "rank_ic_below_threshold"
    DUPLICATE_OF_EXISTING_MEMBER = "duplicate_of_existing_member"
    MERGE_SUGGESTED = "merge_suggested"
    LINEAGE_NOT_REACHABLE = "lineage_not_reachable"
    UNCERTIFIED_EVIDENCE = "uncertified_evidence"


def _finite_float(value: object, label: str, *, allow_none: bool) -> Optional[float]:
    """Coerce a finite non-boolean number (or None with ``allow_none``)."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{label} must be a non-boolean number" + (" or None" if allow_none else "")
        )
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class CandidateEvaluationRef:
    """Explicit, provenance-bearing evaluation of a candidate factor.

    Fields are explicit (never a bare ``object``): the gate inspects this
    surface (``rank_ic`` / label maturity / evidence status / return basis)
    plus the wrapped provenance refs it binds onto the decision artifact.
    """

    candidate_ref: str
    #: Forward-return rank IC on the candidate's own evaluation. A missing
    #: (None) IC is never a pass — the gate rejects it.
    rank_ic: Optional[float] = None
    #: True once the forward label window elapsed and evidence was computed.
    #: False (or a not-computed evidence_status) -> REJECT.
    label_maturity: bool = True
    #: Optional machine-readable evidence status (QE EvidenceStatus value,
    #: reference-only string). Not computed / label-not-mature -> REJECT.
    evidence_status: Optional[str] = None
    #: The return basis the candidate was evaluated on (default vwap->vwap).
    #: Any basis other than the gate's required basis -> REJECT (defensive).
    return_basis: str = VWAP_TO_VWAP_BASIS
    evidence_ref: Optional[str] = None
    #: Reference to the quant_evaluator evaluation that produced this evidence.
    evaluation_ref: Optional[str] = None
    #: Optional reference to the factor_optimizer treatment result
    #: (``TreatmentOptimizationResultArtifact`` / ``library_snapshot_ref``)
    #: that produced the candidate's winning treatment.
    treatment_optimization_ref: Optional[str] = None
    factor_definition_id: Optional[str] = None
    factor_value_ref: Optional[str] = None
    config_hash: Optional[str] = None
    health_card_ref: Optional[str] = None
    metric_versions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_ref:
            raise ValueError("candidate_ref is required")
        object.__setattr__(
            self,
            "rank_ic",
            _finite_float(self.rank_ic, "rank_ic", allow_none=True),
        )
        if not isinstance(self.label_maturity, bool):
            raise TypeError("label_maturity must be a bool")
        if self.evidence_status is not None and (
            not isinstance(self.evidence_status, str) or not self.evidence_status
        ):
            raise ValueError("evidence_status must be a non-empty string or None")
        if not isinstance(self.return_basis, str) or not self.return_basis:
            raise ValueError("return_basis must be a non-empty string")
        for name in ("evidence_ref", "evaluation_ref", "treatment_optimization_ref", "factor_definition_id", "factor_value_ref", "config_hash", "health_card_ref"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        versions = tuple((str(metric), str(version)) for metric, version in self.metric_versions)
        if len({metric for metric, _ in versions}) != len(versions):
            raise ValueError("metric_versions must not contain duplicate metric ids")
        object.__setattr__(self, "metric_versions", versions)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CandidateEvaluationRef":
        """Rebuild a candidate evaluation from a plain mapping."""

        def _optional_str(key: str) -> Optional[str]:
            value = data.get(key)
            return None if value is None else str(value)

        return cls(
            candidate_ref=str(data["candidate_ref"]),
            rank_ic=data.get("rank_ic"),
            label_maturity=data.get("label_maturity", False),
            evidence_status=_optional_str("evidence_status"),
            return_basis=data.get("return_basis", VWAP_TO_VWAP_BASIS),
            evidence_ref=_optional_str("evidence_ref"),
            evaluation_ref=_optional_str("evaluation_ref"),
            treatment_optimization_ref=_optional_str("treatment_optimization_ref"),
            factor_definition_id=_optional_str("factor_definition_id"),
            factor_value_ref=_optional_str("factor_value_ref"),
            config_hash=_optional_str("config_hash"),
            health_card_ref=_optional_str("health_card_ref"),
            metric_versions=tuple(data.get("metric_versions", ())),
        )


@dataclass(frozen=True)
class PromotionDecisionArtifact:
    """Immutable output of the promotion gate (QRP-P7).

    Binds the decision, the reason codes, the evaluated candidate reference,
    the library version it was evaluated against and the bound provenance
    (``evaluation_ref`` / ``treatment_optimization_ref``). ``content_hash``
    is derived-only: a caller may not self-report an arbitrary hash — any
    supplied hash that disagrees with the recomputed value fails closed.
    """

    decision: PromotionDecision
    reason_codes: tuple[PromotionReasonCode, ...]
    candidate_ref: str
    library_version_ref: str
    observation_metadata: Mapping[str, object]
    evaluation_ref: Optional[str] = None
    treatment_optimization_ref: Optional[str] = None
    created_at: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PromotionDecision):
            raise TypeError("decision must be a PromotionDecision")
        if not self.candidate_ref:
            raise ValueError("candidate_ref is required")
        if not self.library_version_ref:
            raise ValueError("library_version_ref is required")
        if not self.reason_codes:
            raise ValueError("reason_codes cannot be empty")
        if any(not isinstance(r, PromotionReasonCode) for r in self.reason_codes):
            raise TypeError("reason_codes must contain PromotionReasonCode values")
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(
            self,
            "observation_metadata",
            dict(self.observation_metadata) if self.observation_metadata else {},
        )
        for name in ("evaluation_ref", "treatment_optimization_ref"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        self._assert_decision_consistency()
        computed = canonical_digest(
            self.decision.value,
            tuple(r.value for r in self.reason_codes),
            self.candidate_ref,
            self.library_version_ref,
            dict(self.observation_metadata),
            self.evaluation_ref or "",
            self.treatment_optimization_ref or "",
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed promotion-decision "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )

    def _assert_decision_consistency(self) -> None:
        """Cross-check decision <-> reason codes (fail-closed)."""
        reject_codes = (
            PromotionReasonCode.LABEL_NOT_MATURE,
            PromotionReasonCode.RETURN_BASIS_WRONG,
            PromotionReasonCode.RANK_IC_BELOW_THRESHOLD,
            PromotionReasonCode.DUPLICATE_OF_EXISTING_MEMBER,
            PromotionReasonCode.UNCERTIFIED_EVIDENCE,
        )
        if self.decision is PromotionDecision.REJECT:
            if not any(code in reject_codes for code in self.reason_codes):
                raise ValueError(
                    "REJECT requires at least one fail-closed reject reason code"
                )
            if PromotionReasonCode.MERGE_SUGGESTED in self.reason_codes:
                raise ValueError("REJECT must not carry MERGE_SUGGESTED")
        elif self.decision is PromotionDecision.APPROVE:
            if PromotionReasonCode.QUALIFIES not in self.reason_codes:
                raise ValueError("APPROVE requires the QUALIFIES reason code")
            wrong = [c.value for c in self.reason_codes if c in reject_codes]
            if wrong:
                raise ValueError(
                    "APPROVE must not carry reject reason codes: " + ", ".join(wrong)
                )
            if PromotionReasonCode.MERGE_SUGGESTED in self.reason_codes:
                raise ValueError("APPROVE must not carry MERGE_SUGGESTED")
        else:  # REVIEW
            if PromotionReasonCode.MERGE_SUGGESTED not in self.reason_codes:
                raise ValueError("REVIEW requires the MERGE_SUGGESTED reason code")
            wrong = [c.value for c in self.reason_codes if c in reject_codes]
            if wrong:
                raise ValueError(
                    "REVIEW must not carry fail-closed reject reason codes: "
                    + ", ".join(wrong)
                )

    @property
    def is_approved(self) -> bool:
        return self.decision is PromotionDecision.APPROVE

    @property
    def is_rejected(self) -> bool:
        return self.decision is PromotionDecision.REJECT

    @property
    def is_review(self) -> bool:
        return self.decision is PromotionDecision.REVIEW

    def to_dict(self) -> dict:
        """Serializable, hash-verified output form."""
        return {
            "decision": self.decision.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "candidate_ref": self.candidate_ref,
            "library_version_ref": self.library_version_ref,
            "observation_metadata": dict(self.observation_metadata),
            "evaluation_ref": self.evaluation_ref,
            "treatment_optimization_ref": self.treatment_optimization_ref,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "PromotionDecisionArtifact":
        """Rebuild an artifact from its ``to_dict`` form (hash re-verified)."""
        return cls(
            decision=PromotionDecision(str(data["decision"])),
            reason_codes=[PromotionReasonCode(code) for code in data["reason_codes"]],
            candidate_ref=str(data["candidate_ref"]),
            library_version_ref=str(data["library_version_ref"]),
            observation_metadata=dict(data.get("observation_metadata") or {}),
            evaluation_ref=data.get("evaluation_ref"),
            treatment_optimization_ref=data.get("treatment_optimization_ref"),
            created_at=data.get("created_at"),
            content_hash=str(data.get("content_hash", "")),
        )


@dataclass(frozen=True)
class RollbackDecisionArtifact:
    """Immutable output of the rollback gate (QRP-P7).

    Validates that ``target_library_version_ref`` is reachable from
    ``current_library_version_ref`` along the library's own lineage chain
    (parent refs only), and records the concrete reachable ``lineage_path``.
    """

    decision: PromotionDecision
    reason_codes: tuple[PromotionReasonCode, ...]
    current_library_version_ref: str
    target_library_version_ref: str
    lineage_path: tuple[str, ...] = ()
    content_hash: str = ""
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PromotionDecision):
            raise TypeError("decision must be a PromotionDecision")
        if self.decision not in (PromotionDecision.APPROVE, PromotionDecision.REJECT):
            raise ValueError("a rollback decision must be APPROVE or REJECT")
        if not self.current_library_version_ref:
            raise ValueError("current_library_version_ref is required")
        if not self.target_library_version_ref:
            raise ValueError("target_library_version_ref is required")
        if self.current_library_version_ref == self.target_library_version_ref:
            raise ValueError(
                "current and target library version must differ (a version "
                "cannot roll back to itself)"
            )
        codes = tuple(self.reason_codes)
        if not codes:
            raise ValueError("reason_codes cannot be empty")
        expected = (
            (PromotionReasonCode.QUALIFIES,)
            if self.decision is PromotionDecision.APPROVE
            else (PromotionReasonCode.LINEAGE_NOT_REACHABLE,)
        )
        if set(codes) != set(expected):
            raise ValueError(
                f"a rollback {self.decision.value} requires exactly the reason "
                f"codes {[c.value for c in expected]} — FAIL CLOSED"
            )
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "lineage_path", tuple(self.lineage_path))
        if self.decision is PromotionDecision.APPROVE and not self.lineage_path:
            raise ValueError(
                "an APPROVED rollback must carry a concrete non-empty lineage_path"
            )
        computed = canonical_digest(
            self.decision.value,
            tuple(c.value for c in self.reason_codes),
            self.current_library_version_ref,
            self.target_library_version_ref,
            self.lineage_path,
        )
        if not self.content_hash:
            object.__setattr__(self, "content_hash", computed)
        elif self.content_hash != computed:
            raise ValueError(
                "content_hash does not match the recomputed rollback-decision "
                "content hash; a caller may not self-report an arbitrary hash — "
                "FAIL CLOSED"
            )
        if not self.created_at:
            object.__setattr__(
                self, "created_at", datetime.now(timezone.utc).isoformat()
            )

    @property
    def is_approved(self) -> bool:
        """Whether the rollback is lineage-permitted."""
        return self.decision is PromotionDecision.APPROVE

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "current_library_version_ref": self.current_library_version_ref,
            "target_library_version_ref": self.target_library_version_ref,
            "lineage_path": list(self.lineage_path),
            "content_hash": self.content_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RollbackDecisionArtifact":
        return cls(
            decision=PromotionDecision(str(data["decision"])),
            reason_codes=[PromotionReasonCode(code) for code in data["reason_codes"]],
            current_library_version_ref=str(data["current_library_version_ref"]),
            target_library_version_ref=str(data["target_library_version_ref"]),
            lineage_path=tuple(data.get("lineage_path") or ()),
            content_hash=str(data.get("content_hash", "")),
            created_at=data.get("created_at"),
        )


class PromotionGate:
    """Fail-closed promotion decision for a candidate library member.

    ``evaluate`` is a pure decision: it reads explicit candidate evaluation
    fields, applies the threshold configuration, and returns an immutable
    :class:`PromotionDecisionArtifact`. It never writes the library, never
    fabricates a missing metric, and never downgrades a fail to a pass.
    """

    def __init__(
        self,
        *,
        min_rank_ic: float = DEFAULT_MIN_RANK_IC,
        duplicate_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        reject_duplicates: bool = DEFAULT_REJECT_DUPLICATES,
        return_basis: str = VWAP_TO_VWAP_BASIS,
    ) -> None:
        self._min_rank_ic = _finite_float(min_rank_ic, "min_rank_ic", allow_none=False)
        if self._min_rank_ic < 0:
            raise ValueError("min_rank_ic must be non-negative")
        self._duplicate_similarity_threshold = _finite_float(
            duplicate_similarity_threshold, "duplicate_similarity_threshold",
            allow_none=False,
        )
        if not 0.0 <= self._duplicate_similarity_threshold <= 1.0:
            raise ValueError("duplicate_similarity_threshold must be in [0, 1]")
        if not isinstance(reject_duplicates, bool):
            raise TypeError("reject_duplicates must be a bool")
        self._reject_duplicates = reject_duplicates
        if not isinstance(return_basis, str) or not return_basis:
            raise ValueError("return_basis must be a non-empty string")
        self._return_basis = return_basis

    @property
    def min_rank_ic(self) -> float:
        return self._min_rank_ic

    @property
    def duplicate_similarity_threshold(self) -> float:
        return self._duplicate_similarity_threshold

    @property
    def reject_duplicates(self) -> bool:
        return self._reject_duplicates

    @property
    def return_basis(self) -> str:
        return self._return_basis

    def evaluate(
        self,
        library_version_ref: str,
        evaluation: CandidateEvaluationRef,
        *,
        library_member_refs: Sequence[str] = (),
        similarity_fn: Optional[Callable[[str, str], Optional[float]]] = None,
        certified_release: bool = False,
        required_metric_versions: Optional[Mapping[str, str]] = None,
    ) -> PromotionDecisionArtifact:
        """Evaluate one candidate against the promotion gates (fail-closed).

        ``similarity_fn`` (when provided) computes the similarity between the
        candidate and one existing library member; a value strictly above
        ``duplicate_similarity_threshold`` triggers the duplicate gate. If the
        library has members but no similarity_fn is provided, or any similarity
        measurement returns ``None`` (unmeasured / UNKNOWN) — uniqueness
        against *every* member is a precondition of promotion — the candidate
        is forced to REVIEW (merge suggested) instead of a silent APPROVE.
        """
        if not isinstance(evaluation, CandidateEvaluationRef):
            raise TypeError("evaluation must be a CandidateEvaluationRef")
        if not library_version_ref:
            raise ValueError("library_version_ref is required")
        if not isinstance(library_member_refs, Sequence):
            raise TypeError("library_member_refs must be a sequence of str refs")
        member_refs = tuple(library_member_refs)
        if similarity_fn is not None and not callable(similarity_fn):
            raise TypeError("similarity_fn must be callable or None")

        reject_codes: list[PromotionReasonCode] = []

        if not isinstance(certified_release, bool):
            raise TypeError("certified_release must be a bool")
        if certified_release and not all(
            (
                evaluation.evidence_ref,
                evaluation.evaluation_ref,
                evaluation.factor_definition_id,
                evaluation.factor_value_ref,
                evaluation.config_hash,
                evaluation.health_card_ref,
            )
        ):
            reject_codes.append(PromotionReasonCode.UNCERTIFIED_EVIDENCE)
        if certified_release and required_metric_versions:
            observed_versions = dict(evaluation.metric_versions)
            if any(
                observed_versions.get(metric_id) != version
                for metric_id, version in required_metric_versions.items()
            ):
                reject_codes.append(PromotionReasonCode.UNCERTIFIED_EVIDENCE)

        # 1) Label maturity (fail-closed).
        if not evaluation.label_maturity or (
            evaluation.evidence_status is not None
            and evaluation.evidence_status in EVIDENCE_NOT_COMPUTED
        ):
            reject_codes.append(PromotionReasonCode.LABEL_NOT_MATURE)

        # 2) Return basis (defensive — the whole warehouse is vwap->vwap).
        if evaluation.return_basis != self._return_basis:
            reject_codes.append(PromotionReasonCode.RETURN_BASIS_WRONG)

        # 3) Rank IC / performance (an absent IC is never a pass).
        if evaluation.rank_ic is None or abs(evaluation.rank_ic) < self._min_rank_ic:
            reject_codes.append(PromotionReasonCode.RANK_IC_BELOW_THRESHOLD)

        # 4) Duplicate vs existing library members (merge, not promote).
        observed_similarities: dict[str, float] = {}
        max_similarity: Optional[float] = None
        measured_count = 0
        unmeasured_count = 0
        unmeasured = False
        no_similarity_measurement = False
        is_duplicate = False
        if member_refs:
            if similarity_fn is None:
                # We cannot assess uniqueness without a measurement — never
                # approve a promotion we cannot prove unique (fail-closed).
                no_similarity_measurement = True
            else:
                observed_similarities, unmeasured_count = self._similarity_scan(
                    evaluation.candidate_ref, member_refs, similarity_fn
                )
                measured_count = len(observed_similarities)
                max_similarity = (
                    max(observed_similarities.values()) if observed_similarities else None
                )
                # Uniqueness must be proven against EVERY member — a partial
                # measurement (some matched member returned None / UNKNOWN) is
                # as unprovable as no measurement at all, so it must fail
                # closed to REVIEW, never a silent APPROVE.
                unmeasured = max_similarity is None or unmeasured_count > 0
                is_duplicate = (
                    max_similarity is not None
                    and max_similarity > self._duplicate_similarity_threshold
                )

        if is_duplicate:
            if self._reject_duplicates:
                reject_codes.append(
                    PromotionReasonCode.DUPLICATE_OF_EXISTING_MEMBER
                )
            elif not reject_codes:
                return self._artifact(
                    PromotionDecision.REVIEW,
                    (PromotionReasonCode.MERGE_SUGGESTED,),
                    library_version_ref,
                    evaluation,
                    observed_similarities,
                    max_similarity,
                    measured_count,
                )

        if reject_codes:
            return self._artifact(
                PromotionDecision.REJECT,
                tuple(reject_codes),
                library_version_ref,
                evaluation,
                observed_similarities,
                max_similarity,
                measured_count,
            )

        # An unmeasured similarity (or library members we could not assess)
        # must force REVIEW — we cannot prove the candidate is unique, so we
        # must not APPROVE it (fail-closed).
        if unmeasured or no_similarity_measurement:
            return self._artifact(
                PromotionDecision.REVIEW,
                (PromotionReasonCode.MERGE_SUGGESTED,),
                library_version_ref,
                evaluation,
                observed_similarities,
                max_similarity,
                measured_count,
            )

        return self._artifact(
            PromotionDecision.APPROVE,
            (PromotionReasonCode.QUALIFIES,),
            library_version_ref,
            evaluation,
            observed_similarities,
            max_similarity,
            measured_count,
        )

    @staticmethod
    def _similarity_scan(
        candidate_ref: str,
        member_refs: Sequence[str],
        similarity_fn: Callable[[str, str], Optional[float]],
    ) -> tuple[dict[str, float], int]:
        """Measure candidate->member similarities; unmeasured pairs are counted.

        A similarity measurement that is genuinely ``None`` (unmeasured /
        UNKNOWN) must not be conflated with a computed zero — the caller
        records it and forces REVIEW, never a silent APPROVE. The number of
        unmeasured member pairs is returned so the caller can fail closed on a
        *partial* measurement too (uniqueness against every member must be
        proven, not just against the measured ones).
        """
        observed: dict[str, float] = {}
        unmeasured_count = 0
        for member_ref in member_refs:
            value = similarity_fn(candidate_ref, member_ref)
            if value is None:
                unmeasured_count += 1
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"similarity_fn returned a non-number for ({candidate_ref!r}, "
                    f"{member_ref!r}): {value!r}"
                )
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                raise ValueError(
                    f"similarity_fn returned a non-finite value for "
                    f"({candidate_ref!r}, {member_ref!r}) — fail closed"
                )
            observed[member_ref] = number
        return observed, unmeasured_count

    def _artifact(
        self,
        decision: PromotionDecision,
        reason_codes: tuple[PromotionReasonCode, ...],
        library_version_ref: str,
        evaluation: CandidateEvaluationRef,
        observed_similarities: Mapping[str, float],
        max_similarity: Optional[float],
        measured_count: int,
    ) -> PromotionDecisionArtifact:
        observation_metadata: dict[str, object] = {}
        observation_metadata["candidate_ref"] = evaluation.candidate_ref
        for key in ("rank_ic", "label_maturity", "evidence_status", "return_basis"):
            value = getattr(evaluation, key)
            if value is not None:
                observation_metadata[key] = value
        if observed_similarities:
            observation_metadata["max_similarity_to_member"] = max_similarity
            observation_metadata["similarity_measured_count"] = measured_count
            observation_metadata["similarity_above_threshold"] = bool(
                max_similarity is not None
                and max_similarity > self._duplicate_similarity_threshold
            )
        return PromotionDecisionArtifact(
            decision=decision,
            reason_codes=reason_codes,
            candidate_ref=evaluation.candidate_ref,
            library_version_ref=library_version_ref,
            observation_metadata=observation_metadata,
            evaluation_ref=evaluation.evaluation_ref,
            treatment_optimization_ref=evaluation.treatment_optimization_ref,
        )


class RollbackGate:
    """Lineage-only rollback permission for a factor library version.

    A version may only roll back to an ancestor along **its own lineage
    chain** (parent version refs). Jumping across to an unrelated version — or
    to a version with no lineage link — is rejected fail-closed.
    """

    def validate(
        self,
        current_library_version_ref: str,
        target_library_version_ref: str,
        lineage: Mapping[str, Iterable[str]],
    ) -> RollbackDecisionArtifact:
        """Validate that the target is reachable along the lineage chain.

        Args:
            current_library_version_ref: The active library version being
                rolled back.
            target_library_version_ref: The version to roll back *to*.
            lineage: ``version_ref -> (parent_version_ref, ...)``. Parents may
                be absent (a root version is its own chain end).

        Returns:
            :class:`RollbackDecisionArtifact` — APPROVE with the concrete
            ``lineage_path`` when the target is reachable, REJECT with
            ``lineage_not_reachable`` otherwise (or when the walk terminates
            without ever reaching the target).
        """
        if not current_library_version_ref:
            raise ValueError("current_library_version_ref is required")
        if not target_library_version_ref:
            raise ValueError("target_library_version_ref is required")
        if current_library_version_ref == target_library_version_ref:
            raise ValueError(
                "current and target library version must differ (a version "
                "cannot roll back to itself)"
            )
        if not isinstance(lineage, Mapping):
            raise TypeError("lineage must be a Mapping of version -> parent refs")
        parents_of = {
            str(version): tuple(str(p) for p in parents)
            for version, parents in lineage.items()
        }

        path = self._find_lineage_path(
            current_library_version_ref, target_library_version_ref, parents_of
        )
        if path is None:
            return RollbackDecisionArtifact(
                decision=PromotionDecision.REJECT,
                reason_codes=(PromotionReasonCode.LINEAGE_NOT_REACHABLE,),
                current_library_version_ref=current_library_version_ref,
                target_library_version_ref=target_library_version_ref,
                lineage_path=(),
            )
        return RollbackDecisionArtifact(
            decision=PromotionDecision.APPROVE,
            reason_codes=(PromotionReasonCode.QUALIFIES,),
            current_library_version_ref=current_library_version_ref,
            target_library_version_ref=target_library_version_ref,
            lineage_path=path,
        )

    @staticmethod
    def _find_lineage_path(
        current: str,
        target: str,
        parents_of: Mapping[str, Sequence[str]],
    ) -> Optional[tuple[str, ...]]:
        """Depth-first walk along parents only; ancestor chain, else None.

        ``current`` -> ``target`` is valid only when ``target`` is an ancestor
        on this version's own chain. A version with no recorded parents is a
        terminal (a root): the walk ends, never crosses.
        """
        visited: set[str] = set()

        def walk(node: str, trail: tuple[str, ...]) -> Optional[tuple[str, ...]]:
            if node == target:
                return trail + (node,)
            if node in visited:
                return None
            visited.add(node)
            for parent in parents_of.get(node, ()):
                found = walk(parent, trail + (node,))
                if found is not None:
                    return found
            return None

        found = walk(current, ())
        if found is None or len(found) < 2:
            return None
        # ``found`` is [current, ..., target]. Return the ancestor chain
        # excluding ``current`` itself — the concrete hops the rollback takes.
        return found[1:]
