"""Treatment policy reference contract for factor_assets.

A :class:`TreatmentPolicyRef` is the typed identifier FA carries for the
selected treatment policy once a treatment decision is made.  It captures
which policy, which version of that policy, and the implementation hash of
the executable treatment recipe, plus an optional link to the
:class:`~factor_assets.contracts.treatment_selection.TreatmentSelectionArtifact`
that produced the selection.

The sibling owner of ``factor_set.py`` puts ``treatment_selection_ref`` /
``preprocess_policy_ref`` on FactorSet membership; we only define the contract
types they import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["TreatmentPolicyRef"]


@dataclass(frozen=True)
class TreatmentPolicyRef:
    """Immutable reference to a concrete treatment policy.

    Fields:
        policy_id: Identifier of the treatment policy.
        policy_version: Version string of the policy definition.
        implementation_hash: sha256 over the executable treatment recipe /
            implementation, pinning exactly which code produced the result.
        treatment_selection_ref: Optional reference to the
            TreatmentSelectionArtifact that selected this policy (when the
            selection provenance is known).
    """

    policy_id: str
    policy_version: str
    implementation_hash: str
    treatment_selection_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if not self.policy_version:
            raise ValueError("policy_version is required")
        if not self.implementation_hash:
            raise ValueError("implementation_hash is required")
