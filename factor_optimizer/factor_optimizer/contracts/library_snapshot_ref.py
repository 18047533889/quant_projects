"""Library-snapshot reference binding for FO (DLIB-FO-008).

FO's ``TreatmentOptimizationResultArtifact`` (and the winner-selection pipeline
behind it) records every provenance ref of a treatment search, but it never
binds the *FA library snapshot* the search operated against.  When the winner
is later consumed by FA as a FactorSet member, FA's ``FactorMembership`` /
``FactorLibraryMembership`` record ``treatment_selection_ref`` /
``selected_treatment_ref`` (the treatment artifact's ``content_hash``) but the
FO side has nothing tying the result to the library snapshot it referenced —
so a stale/unrelated library version could be claimed after the fact.

FA is the authority for ``FactorSetArtifact`` / ``FactorLibraryVersionArtifact``
(``snapshot_ref`` / ``universe_ref`` / ``cluster_set_version_ref`` are REQUIRED
and enforced there).  FO must NOT create a parallel FA authority.  This module
provides only a minimal *reference-typed* snapshot binding (a ref, not a
snapshot body), using FO's own lightweight form because ``factor_assets`` is an
optional dependency here (FO's pyproject declares only ``factor-engine`` /
``quant-evaluator`` extras).

FA-side wiring (NOT done here, FA is green — report-only):
  - ``FactorSetSpec`` could add ``treatment_optimization_ref`` to bind the
    treatment-optimization run that produced the winner, next to its existing
    ``data_snapshot_ref``;
  - ``FactorLibraryMembership`` could add ``treatment_optimization_ref`` next
    to ``selected_treatment_ref`` so each library member points back at the
    optimization result, not just the treatment artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LibrarySnapshotRef:
    """Minimal, immutable reference to the FA library snapshot a search ran on.

    This is a *reference* to an FA-side snapshot (library version, cluster-set
    version, factor-set version), not a parallel copy of the snapshot content.
    FA's own artifacts (``FactorLibraryVersionArtifact``,
    ``FactorSetArtifact``) remain the single source of truth for the snapshot
    body; FO only records which snapshot it referenced, so the winner→library
    link is attributable and the ref can be validated against FA at assembly
    time.

    Attributes:
        library_version_ref: FA FactorLibraryVersionArtifact ref (its
            ``library_version_id`` / ``content_hash``) — REQUIRED.
        cluster_set_version_ref: Optional FA ClusterSetVersionArtifact ref the
            search space was drawn from.
        factor_set_version_ref: Optional FA FactorSetArtifact ref the search
            consumed as its candidate universe.
        snapshot_ref: Data snapshot ref the search evaluated on.
        universe_ref: Data universe ref.
    """

    library_version_ref: str
    cluster_set_version_ref: Optional[str] = None
    factor_set_version_ref: Optional[str] = None
    snapshot_ref: Optional[str] = None
    universe_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.library_version_ref, str) or not self.library_version_ref.strip():
            raise ValueError("library_version_ref must be a non-empty string")
        for name in (
            "cluster_set_version_ref",
            "factor_set_version_ref",
            "snapshot_ref",
            "universe_ref",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")

    @property
    def content_hash(self) -> str:
        """sha256 over the ref fields — stable, order-independent."""
        payload = json.dumps(
            {
                "library_version_ref": self.library_version_ref,
                "cluster_set_version_ref": self.cluster_set_version_ref,
                "factor_set_version_ref": self.factor_set_version_ref,
                "snapshot_ref": self.snapshot_ref,
                "universe_ref": self.universe_ref,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "library_version_ref": self.library_version_ref,
            "cluster_set_version_ref": self.cluster_set_version_ref,
            "factor_set_version_ref": self.factor_set_version_ref,
            "snapshot_ref": self.snapshot_ref,
            "universe_ref": self.universe_ref,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LibrarySnapshotRef":
        return cls(
            library_version_ref=data["library_version_ref"],
            cluster_set_version_ref=data.get("cluster_set_version_ref"),
            factor_set_version_ref=data.get("factor_set_version_ref"),
            snapshot_ref=data.get("snapshot_ref"),
            universe_ref=data.get("universe_ref"),
        )


__all__ = ["LibrarySnapshotRef"]
