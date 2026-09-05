"""RawTreatmentOutcome + extended TreatmentOptimizationResultArtifact (R61-FI-033).

Plan §22 / matrix E4: the historical
:class:`TreatmentOptimizationResultArtifact` recorded only a
``selected_trial_ref`` — a RAW fallback winner and a failed search were
indistinguishable, and the result carried none of the per-candidate refs
(RAW evaluation / RAW health / winner health / candidate trials / Pareto /
multiplicity family / selection reason).  FA's ``TreatmentSelectionArtifact``
consumes this result, so every ref must be present and attributable.

This module extends the artifact WITHOUT deleting any existing field:

- :class:`RawTreatmentOutcome` — the 8 canonical RAW outcomes (plan §22).
- The artifact gains optional outcome/ref fields (all defaulted so every
  existing construction site keeps working):
  ``raw_factor_ref`` / ``raw_evaluation_ref`` / ``raw_health_ref`` /
  ``winner_factor_ref`` / ``winner_evaluation_ref`` / ``winner_health_ref`` /
  ``candidate_trial_refs`` / ``pareto_refs`` / ``multiplicity_family_ref`` /
  ``selection_reason`` / ``failure_summary``.
- :func:`treatment_result_from_selection` builds the fully-ref-attributed
  artifact from a selection summary (outcome + refs).  The IRON RULE is
  enforced here: when RAW wins, the returned production candidate is the RAW
  factor body — the outcome is one of ``RAW_SELECTED_*`` and
  ``winner_factor_ref == raw_factor_ref`` (never ``None``).

The content hash covers the new fields (a tampered outcome or ref is
detected), and ``from_dict`` keeps rejecting payloads whose hash disagrees.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from factor_optimizer.contracts.library_snapshot_ref import LibrarySnapshotRef

__all__ = [
    "RawTreatmentOutcome",
    "RAW_OUTCOME_VALUES",
    "OUTCOME_IS_RAW_WIN",
    "OUTCOME_IS_SUCCESS",
    "TreatmentOptimizationResultArtifact",
    "treatment_result_from_selection",
    "validate_outcome_refs",
]

#: Canonical outcome values (plan §22, 8 statuses).
RAW_OUTCOME_VALUES = (
    "IMPROVED",
    "RAW_SELECTED_NO_IMPROVEMENT",
    "RAW_SELECTED_NEAR_EQUIVALENT",
    "RAW_SELECTED_CANDIDATES_FAILED",
    "NO_ELIGIBLE_REPAIR",
    "SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED",
    "FAILED_RAW_VALID",
    "FAILED_RAW_INVALID",
)


class RawTreatmentOutcome(str, Enum):
    """Canonical outcome of a treatment-optimization run (plan §22).

    The first four describe a run that ENDED WITH A WINNER:

    IMPROVED                          a treatment beat RAW.
    RAW_SELECTED_NO_IMPROVEMENT       RAW is the winner: no treatment improved
                                      on it (treated-got-worse discoverable).
    RAW_SELECTED_NEAR_EQUIVALENT      a treatment was statistically
                                      indistinguishable, so simpler RAW won.
    RAW_SELECTED_CANDIDATES_FAILED    every treatment candidate failed hard
                                      gates/validation; RAW survives.

    The rest describe runs that ended WITHOUT a winner candidate:

    NO_ELIGIBLE_REPAIR                the source factor had no eligible
                                      repair (RAW itself is not a valid
                                      alternative).
    SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED  budget ran out and RAW was selected
                                      as the safe fallback.
    FAILED_RAW_VALID                  the run failed but RAW remains a valid
                                      factor (no production candidate is a
                                      treatment; RAW stays).
    FAILED_RAW_INVALID                the run failed AND RAW is itself invalid
                                      (nothing may be produced).
    """

    IMPROVED = "IMPROVED"
    RAW_SELECTED_NO_IMPROVEMENT = "RAW_SELECTED_NO_IMPROVEMENT"
    RAW_SELECTED_NEAR_EQUIVALENT = "RAW_SELECTED_NEAR_EQUIVALENT"
    RAW_SELECTED_CANDIDATES_FAILED = "RAW_SELECTED_CANDIDATES_FAILED"
    NO_ELIGIBLE_REPAIR = "NO_ELIGIBLE_REPAIR"
    SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED = "SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED"
    FAILED_RAW_VALID = "FAILED_RAW_VALID"
    FAILED_RAW_INVALID = "FAILED_RAW_INVALID"

    @classmethod
    def from_value(cls, value: object) -> "RawTreatmentOutcome":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown RawTreatmentOutcome value {value!r}; expected one "
                    f"of {RAW_OUTCOME_VALUES}"
                ) from None
        raise TypeError(
            f"RawTreatmentOutcome.from_value expects str or "
            f"RawTreatmentOutcome, got {type(value).__name__}"
        )

    @property
    def raw_won(self) -> bool:
        """True when RAW is the production candidate returned by the run."""
        return self in _RAW_WIN_OUTCOMES

    @property
    def produced_winner(self) -> bool:
        """True when the run ended with a winner candidate to produce."""
        return self in _SUCCESS_OUTCOMES


#: Outcomes where RAW is the returned production candidate (RAW body, not None).
_RAW_WIN_OUTCOMES = frozenset(
    {
        RawTreatmentOutcome.RAW_SELECTED_NO_IMPROVEMENT,
        RawTreatmentOutcome.RAW_SELECTED_NEAR_EQUIVALENT,
        RawTreatmentOutcome.RAW_SELECTED_CANDIDATES_FAILED,
    }
)

#: Outcomes where the run ended with a winner candidate to produce (a
#: treatment that beat RAW, or RAW itself).  SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED
#: is a RAW fallback selected without a full decision run — the RAW body is
#: returned, but the outcome is not a "winner produced by the decision
#: pipeline" success.
_SUCCESS_OUTCOMES = frozenset(
    {
        RawTreatmentOutcome.IMPROVED,
        RawTreatmentOutcome.RAW_SELECTED_NO_IMPROVEMENT,
        RawTreatmentOutcome.RAW_SELECTED_NEAR_EQUIVALENT,
        RawTreatmentOutcome.RAW_SELECTED_CANDIDATES_FAILED,
    }
)

#: Function-style predicates for callers that prefer functions over properties.
OUTCOME_IS_RAW_WIN = lambda outcome: RawTreatmentOutcome.from_value(outcome).raw_won  # noqa: E731
OUTCOME_IS_SUCCESS = lambda outcome: RawTreatmentOutcome.from_value(outcome).produced_winner  # noqa: E731


def _ref_map(value: Any, label: str) -> Dict[str, str]:
    """Normalize a mapping of refs into {field: ref}; fail on empty refs."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping of field -> ref string")
    out: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{key!r}] must be a non-empty ref string")
        out[key] = item
    return out


def validate_outcome_refs(
    outcome: RawTreatmentOutcome,
    *,
    raw_factor_ref: Optional[str],
    raw_evaluation_ref: Optional[str],
    raw_health_ref: Optional[str],
    winner_factor_ref: Optional[str] = None,
    winner_evaluation_ref: Optional[str] = None,
    winner_health_ref: Optional[str] = None,
    candidate_trial_refs: Tuple[str, ...] = (),
    pareto_refs: Tuple[str, ...] = (),
    multiplicity_family_ref: str = "",
    selection_reason: str = "",
    failure_summary: str = "",
) -> None:
    """Fail-closed ref validation for a RAW outcome (plan §22 iron law).

    Rules:
    - RAW win outcomes require raw_factor_ref + raw_evaluation_ref +
      raw_health_ref and set winner_*_ref to the RAW refs (never None).
    - IMPROVED requires a treatment winner ref that differs from the RAW ref.
    - FAILED_RAW_VALID keeps RAW refs present (RAW stays valid) but produces
      no treatment winner.
    - FAILED_RAW_INVALID requires the raw refs to be ABSENT (nothing is valid).
    """
    outcome = RawTreatmentOutcome.from_value(outcome)

    def _nonempty(value: Optional[str]) -> bool:
        return isinstance(value, str) and bool(value.strip())

    # Legacy constructions (pre-R61-FI-033) never set the plan §22 refs and
    # keep the default outcome IMPROVED with an empty winner ref; their
    # attribution is via selected_trial_ref / trial_ledger_ref only.  When the
    # caller supplied NO plan-§22 ref at all, skip the iron-rule validation —
    # it only applies once the caller opts into §22 attribution.
    if (
        not _nonempty(raw_factor_ref)
        and not _nonempty(raw_evaluation_ref)
        and not _nonempty(raw_health_ref)
        and not _nonempty(winner_factor_ref)
        and not _nonempty(raw_health_ref)
        and outcome is RawTreatmentOutcome.IMPROVED
    ):
        return

    if outcome.raw_won:
        # SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED is a RAW fallback selected
        # without a full decision run; the RAW body is returned but the run
        # never declared a winner.  Require the raw refs but do not demand the
        # winner refs equal RAW (there was no winner selection).
        if outcome is RawTreatmentOutcome.SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED:
            missing = []
            if not _nonempty(raw_factor_ref):
                missing.append("raw_factor_ref")
            if not _nonempty(raw_evaluation_ref):
                missing.append("raw_evaluation_ref")
            if missing:
                raise ValueError(
                    f"outcome {outcome.value} returns RAW as the fallback but "
                    f"{', '.join(missing)} is/are missing — the RAW factor "
                    "body must be attributable (never None)"
                )
            return
        missing = []
        if not _nonempty(raw_factor_ref):
            missing.append("raw_factor_ref")
        if not _nonempty(raw_evaluation_ref):
            missing.append("raw_evaluation_ref")
        if not _nonempty(raw_health_ref):
            missing.append("raw_health_ref")
        if missing:
            raise ValueError(
                f"outcome {outcome.value} is a RAW win but {', '.join(missing)} "
                "is/are missing — the RAW factor body must be returned when RAW "
                "wins (never None)"
            )
        # RAW is the winner: winner refs must equal RAW refs (the production
        # candidate IS the RAW factor body).
        if not _nonempty(winner_factor_ref) or winner_factor_ref != raw_factor_ref:
            raise ValueError(
                f"outcome {outcome.value} wins with RAW — winner_factor_ref "
                "must equal raw_factor_ref (the returned candidate is the RAW "
                "factor body, never None or a different factor)"
            )
        return
    if outcome is RawTreatmentOutcome.IMPROVED:
        if not _nonempty(raw_factor_ref) or not _nonempty(raw_evaluation_ref):
            raise ValueError(
                "outcome IMPROVED still requires raw_factor_ref and "
                "raw_evaluation_ref (the RAW baseline evidence must be present "
                "to demonstrate the improvement)"
            )
        if not _nonempty(winner_factor_ref):
            raise ValueError(
                "outcome IMPROVED requires a non-empty winner_factor_ref (the "
                "treatment that beat RAW)"
            )
        if winner_factor_ref == raw_factor_ref:
            raise ValueError(
                "outcome IMPROVED requires winner_factor_ref to differ from "
                "raw_factor_ref (a RAW win must be declared as a RAW_SELECTED_* "
                "outcome, not IMPROVED)"
            )
        return
    if outcome in (
        RawTreatmentOutcome.NO_ELIGIBLE_REPAIR,
        RawTreatmentOutcome.SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED,
    ):
        # No repair / budget exhausted before a real winner: RAW must still be
        # a valid attributable baseline.
        if not _nonempty(raw_factor_ref) or not _nonempty(raw_evaluation_ref):
            raise ValueError(
                f"outcome {outcome.value} requires raw_factor_ref and "
                "raw_evaluation_ref (the run still has a valid RAW baseline)"
            )
        return
    if outcome is RawTreatmentOutcome.FAILED_RAW_VALID:
        if not _nonempty(raw_factor_ref):
            raise ValueError(
                "outcome FAILED_RAW_VALID requires raw_factor_ref — RAW is "
                "still a valid factor even though the run failed"
            )
        return
    if outcome is RawTreatmentOutcome.FAILED_RAW_INVALID:
        if _nonempty(raw_factor_ref):
            raise ValueError(
                "outcome FAILED_RAW_INVALID requires raw_factor_ref to be "
                "absent — RAW is invalid, nothing may be produced"
            )
        return
    raise ValueError(f"unhandled outcome {outcome.value}")  # pragma: no cover


@dataclass(frozen=True)
class TreatmentOptimizationResultArtifact:
    """Deep-immutable, content-hashed result of a treatment-optimization search.

    Extends the DLIB-FO-007 artifact with the plan §22 RAW-outcome + full-ref
    fields.  All new fields are optional-with-default so every existing
    construction site (and test) continues to work byte-for-byte; callers that
    need FA-side attribution construct through
    :func:`treatment_result_from_selection`.

    When ``raw_outcome`` is a RAW win (``RAW_SELECTED_*`` /
    ``SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED``), the production candidate IS the
    RAW factor body: ``winner_factor_ref == raw_factor_ref``, never ``None``.
    """

    # -- pre-existing required refs (DLIB-FO-007) ---------------------------
    search_session_id: str
    source_factor_value_ref: str
    raw_baseline_evidence_ref: str
    factor_profile_ref: str
    treatment_search_space_ref: str
    transform_registry_snapshot_ref: str
    desirability_policy_ref: str
    winner_policy_ref: str
    split_plan_ref: str
    trial_ledger_ref: str
    all_trial_refs: Tuple[str, ...]
    pareto_trial_refs: Tuple[str, ...]
    multiplicity_ref: str
    # -- pre-existing optional fields (DLIB-FO-007 / DLIB-FO-008) ------------
    library_snapshot_ref: Optional[Any] = None
    require_library_snapshot_ref: bool = False
    selected_trial_ref: str = ""
    uncertainty_evidence_ref: str = ""
    sealed_test_ref: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # -- R61-FI-033 plan §22 fields -------------------------------------------
    raw_outcome: str = "IMPROVED"
    raw_factor_ref: str = ""
    raw_evaluation_ref: str = ""
    raw_health_ref: str = ""
    winner_factor_ref: str = ""
    winner_evaluation_ref: str = ""
    winner_health_ref: str = ""
    candidate_trial_refs: Tuple[str, ...] = ()
    pareto_refs: Tuple[str, ...] = ()
    multiplicity_family_ref: str = ""
    selection_reason: str = ""
    failure_summary: str = ""
    per_field_refs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "search_session_id",
            "source_factor_value_ref",
            "raw_baseline_evidence_ref",
            "factor_profile_ref",
            "treatment_search_space_ref",
            "transform_registry_snapshot_ref",
            "desirability_policy_ref",
            "winner_policy_ref",
            "split_plan_ref",
            "trial_ledger_ref",
            "multiplicity_ref",
            "selected_trial_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.uncertainty_evidence_ref, str):
            raise TypeError("uncertainty_evidence_ref must be a string")
        if self.sealed_test_ref is not None and (
            not isinstance(self.sealed_test_ref, str)
            or not self.sealed_test_ref.strip()
        ):
            raise ValueError("sealed_test_ref must be a non-empty string or None")
        for name in ("all_trial_refs", "pareto_trial_refs"):
            refs = getattr(self, name)
            if not isinstance(refs, tuple) or not all(
                isinstance(r, str) and r.strip() for r in refs
            ):
                raise ValueError(f"{name} must be a tuple of non-empty strings")
        if not self.all_trial_refs:
            raise ValueError("all_trial_refs must be non-empty")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")
        # The selected trial must be among the evaluated trials (pre-existing).
        if self.selected_trial_ref and self.selected_trial_ref not in self.all_trial_refs:
            raise ValueError(
                "selected_trial_ref must reference one of all_trial_refs"
            )
        # Pareto trials must be a subset of all trials (pre-existing).
        if not set(self.pareto_trial_refs).issubset(set(self.all_trial_refs)):
            raise ValueError("pareto_trial_refs must be a subset of all_trial_refs")
        # Candidate trials are a subset of all trials too.
        if not set(self.candidate_trial_refs).issubset(set(self.all_trial_refs)):
            raise ValueError(
                "candidate_trial_refs must be a subset of all_trial_refs"
            )
        # Normalize the outcome and validate the plan §22 ref iron rules.
        object.__setattr__(
            self, "raw_outcome", RawTreatmentOutcome.from_value(self.raw_outcome)
        )
        validate_outcome_refs(
            self.raw_outcome,
            raw_factor_ref=self.raw_factor_ref,
            raw_evaluation_ref=self.raw_evaluation_ref,
            raw_health_ref=self.raw_health_ref,
            winner_factor_ref=self.winner_factor_ref,
            winner_evaluation_ref=self.winner_evaluation_ref,
            winner_health_ref=self.winner_health_ref,
            candidate_trial_refs=tuple(self.candidate_trial_refs),
            pareto_refs=tuple(self.pareto_refs),
            multiplicity_family_ref=self.multiplicity_family_ref,
            selection_reason=self.selection_reason,
            failure_summary=self.failure_summary,
        )
        # Normalize candidate_trial_refs / pareto_refs / per_field_refs.
        object.__setattr__(
            self, "candidate_trial_refs", tuple(self.candidate_trial_refs)
        )
        object.__setattr__(self, "pareto_refs", tuple(self.pareto_refs))
        object.__setattr__(self, "per_field_refs", _ref_map(self.per_field_refs, "per_field_refs"))
        # DLIB-FO-008: normalize the optional library-snapshot reference.
        if self.library_snapshot_ref is None:
            if self.require_library_snapshot_ref:
                raise ValueError(
                    "library_snapshot_ref is required when "
                    "require_library_snapshot_ref=True — the winner must be "
                    "attributable to the FA library snapshot it searched against"
                )
            object.__setattr__(self, "library_snapshot_ref", None)
        elif isinstance(self.library_snapshot_ref, LibrarySnapshotRef):
            pass  # already the canonical FO form
        elif isinstance(self.library_snapshot_ref, str):
            if not self.library_snapshot_ref.strip():
                raise ValueError(
                    "library_snapshot_ref must be a non-empty string or a "
                    "LibrarySnapshotRef"
                )
            object.__setattr__(
                self,
                "library_snapshot_ref",
                LibrarySnapshotRef(library_version_ref=self.library_snapshot_ref),
            )
        else:
            raise TypeError(
                "library_snapshot_ref must be a LibrarySnapshotRef, a non-empty "
                f"str, or None; got {type(self.library_snapshot_ref).__name__}"
            )
        if not isinstance(self.require_library_snapshot_ref, bool):
            raise TypeError("require_library_snapshot_ref must be a bool")
        # R61-FI-033: pre-existing callers never set the outcome refs — the
        # old selected_trial_ref keeps working and the artifact defaults to
        # IMPROVED with no extra refs (validate_outcome_refs passed because
        # raw win rules are only enforced for raw_won outcomes, and IMPROVED
        # with empty winner ref is tolerated for legacy constructions whose
        # selected_trial_ref points at a trial).  New callers use
        # treatment_result_from_selection which always supplies the refs.
        object.__setattr__(self, "_canonical", self._canonical_payload())

    @property
    def raw_won(self) -> bool:
        """True when RAW is the production candidate of this run."""
        return RawTreatmentOutcome.from_value(self.raw_outcome).raw_won

    @property
    def produced_winner(self) -> bool:
        """True when the run produced a winner candidate."""
        return RawTreatmentOutcome.from_value(self.raw_outcome).produced_winner

    def _canonical_payload(self) -> str:
        snap = self.library_snapshot_ref
        if snap is None:
            snap_dict = None
        elif isinstance(snap, LibrarySnapshotRef):
            snap_dict = snap.to_dict()
        else:  # pragma: no cover - normalized at construction
            snap_dict = str(snap)
        return json.dumps(
            {
                "search_session_id": self.search_session_id,
                "source_factor_value_ref": self.source_factor_value_ref,
                "raw_baseline_evidence_ref": self.raw_baseline_evidence_ref,
                "factor_profile_ref": self.factor_profile_ref,
                "treatment_search_space_ref": self.treatment_search_space_ref,
                "transform_registry_snapshot_ref": self.transform_registry_snapshot_ref,
                "desirability_policy_ref": self.desirability_policy_ref,
                "winner_policy_ref": self.winner_policy_ref,
                "split_plan_ref": self.split_plan_ref,
                "trial_ledger_ref": self.trial_ledger_ref,
                "all_trial_refs": list(self.all_trial_refs),
                "pareto_trial_refs": list(self.pareto_trial_refs),
                "multiplicity_ref": self.multiplicity_ref,
                "library_snapshot_ref": snap_dict,
                "require_library_snapshot_ref": self.require_library_snapshot_ref,
                "selected_trial_ref": self.selected_trial_ref,
                "uncertainty_evidence_ref": self.uncertainty_evidence_ref,
                "sealed_test_ref": self.sealed_test_ref,
                "created_at": self.created_at.isoformat(),
                "raw_outcome": self.raw_outcome.value
                if isinstance(self.raw_outcome, RawTreatmentOutcome)
                else self.raw_outcome,
                "raw_factor_ref": self.raw_factor_ref,
                "raw_evaluation_ref": self.raw_evaluation_ref,
                "raw_health_ref": self.raw_health_ref,
                "winner_factor_ref": self.winner_factor_ref,
                "winner_evaluation_ref": self.winner_evaluation_ref,
                "winner_health_ref": self.winner_health_ref,
                "candidate_trial_refs": list(self.candidate_trial_refs),
                "pareto_refs": list(self.pareto_refs),
                "multiplicity_family_ref": self.multiplicity_family_ref,
                "selection_reason": self.selection_reason,
                "failure_summary": self.failure_summary,
                "per_field_refs": self.per_field_refs,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()

    def verify(self) -> None:
        """Fail closed if the artifact was altered after construction."""
        if self._canonical_payload() != self._canonical:
            raise ValueError(
                "treatment optimization result artifact content was tampered"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_session_id": self.search_session_id,
            "source_factor_value_ref": self.source_factor_value_ref,
            "raw_baseline_evidence_ref": self.raw_baseline_evidence_ref,
            "factor_profile_ref": self.factor_profile_ref,
            "treatment_search_space_ref": self.treatment_search_space_ref,
            "transform_registry_snapshot_ref": self.transform_registry_snapshot_ref,
            "desirability_policy_ref": self.desirability_policy_ref,
            "winner_policy_ref": self.winner_policy_ref,
            "split_plan_ref": self.split_plan_ref,
            "trial_ledger_ref": self.trial_ledger_ref,
            "all_trial_refs": list(self.all_trial_refs),
            "pareto_trial_refs": list(self.pareto_trial_refs),
            "multiplicity_ref": self.multiplicity_ref,
            "library_snapshot_ref": (
                self.library_snapshot_ref.to_dict()
                if self.library_snapshot_ref is not None
                else None
            ),
            "require_library_snapshot_ref": self.require_library_snapshot_ref,
            "selected_trial_ref": self.selected_trial_ref,
            "uncertainty_evidence_ref": self.uncertainty_evidence_ref,
            "sealed_test_ref": self.sealed_test_ref,
            "created_at": self.created_at.isoformat(),
            "raw_outcome": (
                self.raw_outcome.value
                if isinstance(self.raw_outcome, RawTreatmentOutcome)
                else self.raw_outcome
            ),
            "raw_factor_ref": self.raw_factor_ref,
            "raw_evaluation_ref": self.raw_evaluation_ref,
            "raw_health_ref": self.raw_health_ref,
            "winner_factor_ref": self.winner_factor_ref,
            "winner_evaluation_ref": self.winner_evaluation_ref,
            "winner_health_ref": self.winner_health_ref,
            "candidate_trial_refs": list(self.candidate_trial_refs),
            "pareto_refs": list(self.pareto_refs),
            "multiplicity_family_ref": self.multiplicity_family_ref,
            "selection_reason": self.selection_reason,
            "failure_summary": self.failure_summary,
            "per_field_refs": dict(self.per_field_refs),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreatmentOptimizationResultArtifact":
        if not isinstance(data, dict):
            raise TypeError(
                "TreatmentOptimizationResultArtifact.from_dict requires a dict"
            )
        created = data["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        raw_snap = data.get("library_snapshot_ref")
        if raw_snap is None:
            library_snapshot_ref = None
        elif isinstance(raw_snap, dict):
            library_snapshot_ref = LibrarySnapshotRef.from_dict(raw_snap)
        else:
            library_snapshot_ref = raw_snap  # normalized at construction
        obj = cls(
            search_session_id=data["search_session_id"],
            source_factor_value_ref=data["source_factor_value_ref"],
            raw_baseline_evidence_ref=data["raw_baseline_evidence_ref"],
            factor_profile_ref=data["factor_profile_ref"],
            treatment_search_space_ref=data["treatment_search_space_ref"],
            transform_registry_snapshot_ref=data["transform_registry_snapshot_ref"],
            desirability_policy_ref=data["desirability_policy_ref"],
            winner_policy_ref=data["winner_policy_ref"],
            split_plan_ref=data["split_plan_ref"],
            trial_ledger_ref=data["trial_ledger_ref"],
            all_trial_refs=tuple(data["all_trial_refs"]),
            pareto_trial_refs=tuple(data["pareto_trial_refs"]),
            multiplicity_ref=data["multiplicity_ref"],
            library_snapshot_ref=library_snapshot_ref,
            require_library_snapshot_ref=data.get("require_library_snapshot_ref", False),
            selected_trial_ref=data["selected_trial_ref"],
            uncertainty_evidence_ref=data["uncertainty_evidence_ref"],
            sealed_test_ref=data.get("sealed_test_ref"),
            created_at=created,
            raw_outcome=data.get("raw_outcome", "IMPROVED"),
            raw_factor_ref=data.get("raw_factor_ref", ""),
            raw_evaluation_ref=data.get("raw_evaluation_ref", ""),
            raw_health_ref=data.get("raw_health_ref", ""),
            winner_factor_ref=data.get("winner_factor_ref", ""),
            winner_evaluation_ref=data.get("winner_evaluation_ref", ""),
            winner_health_ref=data.get("winner_health_ref", ""),
            candidate_trial_refs=tuple(data.get("candidate_trial_refs", ())),
            pareto_refs=tuple(data.get("pareto_refs", ())),
            multiplicity_family_ref=data.get("multiplicity_family_ref", ""),
            selection_reason=data.get("selection_reason", ""),
            failure_summary=data.get("failure_summary", ""),
            per_field_refs=dict(data.get("per_field_refs", {})),
        )
        if data.get("content_hash") not in (None, obj.content_hash):
            raise ValueError(
                "treatment optimization result artifact content_hash does not "
                "match its payload"
            )
        return obj


def treatment_result_from_selection(
    *,
    search_session_id: str,
    source_factor_value_ref: str,
    raw_baseline_evidence_ref: str,
    factor_profile_ref: str,
    treatment_search_space_ref: str,
    transform_registry_snapshot_ref: str,
    desirability_policy_ref: str,
    winner_policy_ref: str,
    split_plan_ref: str,
    trial_ledger_ref: str,
    all_trial_refs: Tuple[str, ...],
    pareto_trial_refs: Tuple[str, ...],
    multiplicity_ref: str,
    outcome: RawTreatmentOutcome,
    raw_factor_ref: str,
    raw_evaluation_ref: str,
    raw_health_ref: str,
    winner_factor_ref: str = "",
    winner_evaluation_ref: str = "",
    winner_health_ref: str = "",
    candidate_trial_refs: Tuple[str, ...] = (),
    pareto_refs: Tuple[str, ...] = (),
    multiplicity_family_ref: str = "",
    selection_reason: str = "",
    failure_summary: str = "",
    selected_trial_ref: str = "",
    library_snapshot_ref: Optional[Any] = None,
    require_library_snapshot_ref: bool = False,
    uncertainty_evidence_ref: str = "",
    sealed_test_ref: Optional[str] = None,
    per_field_refs: Optional[Mapping[str, str]] = None,
) -> TreatmentOptimizationResultArtifact:
    """Build a fully-ref-attributed result artifact from a selection summary.

    The iron rule (plan §22) is enforced by :func:`validate_outcome_refs`: a
    RAW-win outcome (``RAW_SELECTED_*`` / ``SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED``)
    requires ``winner_factor_ref == raw_factor_ref`` — the production candidate
    is the RAW factor body, never ``None``.
    """
    outcome_enum = RawTreatmentOutcome.from_value(outcome)
    if not selected_trial_ref:
        # Default a winner-candidate outcome's selected trial to the winner
        # factor's trial id when the trial set contains it (RAW wins select
        # RAW; IMPROVED selects the winner).  When the trial refs do not name
        # the factor ref, fall back to the pre-existing required-trial list.
        candidate = winner_factor_ref or "RAW"
        if candidate in tuple(all_trial_refs):
            selected_trial_ref = candidate
        else:
            selected_trial_ref = tuple(all_trial_refs)[0] if all_trial_refs else "RAW"
    if outcome_enum.raw_won:
        # RAW is the production candidate: winner refs ARE the RAW refs.
        winner_factor_ref = winner_factor_ref or raw_factor_ref
        winner_evaluation_ref = winner_evaluation_ref or raw_evaluation_ref
        winner_health_ref = winner_health_ref or raw_health_ref
        if "RAW" in tuple(all_trial_refs):
            selected_trial_ref = "RAW"
    if (
        outcome_enum in _SUCCESS_OUTCOMES
        and not outcome_enum.raw_won
        and selected_trial_ref == "RAW"
        and winner_factor_ref
        and winner_factor_ref in tuple(all_trial_refs)
    ):
        # IMPROVED with a treatment winner: the selected trial is the winner
        # factor's trial, never the RAW placeholder.
        selected_trial_ref = winner_factor_ref
    return TreatmentOptimizationResultArtifact(
        search_session_id=search_session_id,
        source_factor_value_ref=source_factor_value_ref,
        raw_baseline_evidence_ref=raw_baseline_evidence_ref,
        factor_profile_ref=factor_profile_ref,
        treatment_search_space_ref=treatment_search_space_ref,
        transform_registry_snapshot_ref=transform_registry_snapshot_ref,
        desirability_policy_ref=desirability_policy_ref,
        winner_policy_ref=winner_policy_ref,
        split_plan_ref=split_plan_ref,
        trial_ledger_ref=trial_ledger_ref,
        all_trial_refs=tuple(all_trial_refs),
        pareto_trial_refs=tuple(pareto_trial_refs),
        multiplicity_ref=multiplicity_ref,
        library_snapshot_ref=library_snapshot_ref,
        require_library_snapshot_ref=require_library_snapshot_ref,
        selected_trial_ref=selected_trial_ref,
        uncertainty_evidence_ref=uncertainty_evidence_ref,
        sealed_test_ref=sealed_test_ref,
        raw_outcome=outcome_enum,
        raw_factor_ref=raw_factor_ref,
        raw_evaluation_ref=raw_evaluation_ref,
        raw_health_ref=raw_health_ref,
        winner_factor_ref=winner_factor_ref,
        winner_evaluation_ref=winner_evaluation_ref,
        winner_health_ref=winner_health_ref,
        candidate_trial_refs=tuple(candidate_trial_refs),
        pareto_refs=tuple(pareto_refs),
        multiplicity_family_ref=multiplicity_family_ref,
        selection_reason=selection_reason,
        failure_summary=failure_summary,
        per_field_refs=dict(per_field_refs or {}),
    )
