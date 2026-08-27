"""TreatmentOptimizationResultArtifact (DLIB-FO-007).

Binds every reference a treatment-optimization search produced into a single
deep-immutable, content-hashed artifact.  The FA ``TreatmentSelectionArtifact``
will consume this result, so every ref must be present and attributable.

The artifact is deep-immutable: all nested collections are frozen, and a
derived ``content_hash`` is computed at construction.  ``verify()`` re-derives
the hash and fails closed if the payload was altered after construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from factor_optimizer.contracts.library_snapshot_ref import LibrarySnapshotRef


@dataclass(frozen=True)
class TreatmentOptimizationResultArtifact:
    """Deep-immutable, content-hashed result of a treatment-optimization search.

    Attributes:
        search_session_id: The search session that produced this result.
        source_factor_value_ref: Reference to the source factor's value.
        raw_baseline_evidence_ref: Reference to the RAW baseline evidence.
        factor_profile_ref: Reference to the factor profile.
        treatment_search_space_ref: Reference to the treatment search space.
        transform_registry_snapshot_ref: Reference to the transform registry
            snapshot used during the search.
        desirability_policy_ref: Reference to the frozen desirability policy.
        winner_policy_ref: Reference to the winner policy.
        split_plan_ref: Reference to the split plan.
        trial_ledger_ref: Reference to the trial ledger.
        all_trial_refs: References to every trial.
        pareto_trial_refs: References to the Pareto-frontier trials.
        multiplicity_ref: Reference to the multiplicity artifact.
        library_snapshot_ref: Reference to the FA library snapshot the search
            operated against (DLIB-FO-008).  FA is the authority for the
            library body (``FactorLibraryVersionArtifact`` /
            ``FactorSetArtifact``); FO records only the ref so the winner is
            attributable to the exact library snapshot.  Optional: the
            reference may be provided as a :class:`LibrarySnapshotRef` or as a
            bare ``str`` ref (converted at construction); ``None`` is allowed
            for backward compatibility but the artifact can opt into
            ``require_library_snapshot_ref`` to fail closed when missing.
        require_library_snapshot_ref: When True, construction fails closed if
            ``library_snapshot_ref`` is absent.  Defaults to False so existing
            callers (and tests) continue to construct the artifact without the
            new binding.
        selected_trial_ref: Reference to the selected (winning) trial.
        uncertainty_evidence_ref: Reference to the uncertainty evidence.
        sealed_test_ref: Optional reference to the sealed test result (if a
            one-shot sealed test was run).
        created_at: Creation timestamp.
    """

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
    library_snapshot_ref: Optional[Any] = None
    require_library_snapshot_ref: bool = False
    selected_trial_ref: str = ""
    uncertainty_evidence_ref: str = ""
    sealed_test_ref: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

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
            "uncertainty_evidence_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
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
        # The selected trial must be among the evaluated trials.
        if self.selected_trial_ref not in self.all_trial_refs:
            raise ValueError(
                "selected_trial_ref must reference one of all_trial_refs"
            )
        # Pareto trials must be a subset of all trials.
        if not set(self.pareto_trial_refs).issubset(set(self.all_trial_refs)):
            raise ValueError("pareto_trial_refs must be a subset of all_trial_refs")
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
            # Bare-str shorthand: wrap into the canonical FO ref form.
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
        object.__setattr__(self, "_canonical", self._canonical_payload())

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
        )
        if data.get("content_hash") not in (None, obj.content_hash):
            raise ValueError(
                "treatment optimization result artifact content_hash does not "
                "match its payload"
            )
        return obj


__all__ = [
    "TreatmentOptimizationResultArtifact",
]
