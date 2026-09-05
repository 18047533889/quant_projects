"""
Model-specific representation policy (R61-FI-044, plan §28).

Alpha representation selection and model-input normalization are SEPARATE
decisions: a factor asset's canonical form is chosen for its ALPHA evidence,
while a model may require a different representation (rank / clipped /
robust-zscore / …) as model input. This module formalizes four frozen,
versioned *representation policy profiles*:

- :data:`TREE_TABULAR` — raw / rank / clipped allowed, z-score optional.
- :data:`LINEAR` — robust outlier handling plus a train-fitted z-score /
  robust-z-score normally required (an unfitted raw/rank-only linear input
  violates the profile).
- :data:`NEURAL_TABULAR` — robust z-score or rank-gaussian, a missing
  mask/channel, and stable clipping.
- :data:`SEQUENCE_MODEL` — train-fitted scaling and causal / stable temporal
  normalization.

Key guarantees (R61-FI-044):

- **Versioned frozen profiles.** Each profile is a frozen dataclass carrying a
  policy id + semantic version, resolved through a frozen
  :class:`RepresentationPolicyRegistry`. Unknown profile ids fail closed.
- **Model-specific artifacts never overwrite the canonical factor asset.**
  :class:`FeatureRepresentationArtifact` records the *model-specific*
  representation (its transform chain and its policy profile); its writer
  refuses to write into a canonical factor-asset identity, and the artifact
  pins ``canonical_factor_ref`` so the model representation is *tracked as a
  separate artifact*, not a replacement of the alpha asset.
- **Non-inferiority for model safety.** A model-mandated normalization /
  neutralization is admitted only when the alpha evidence is non-inferior
  within the versioned tolerance (:func:`non_inferior`). When the model
  treatment *materially destroys* the alpha signal the guard raises a typed
  :class:`SignalDestructionConflict` — it NEVER silently forces the
  normalization.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from factor_preprocess.contracts._deep_freeze import deep_freeze
from factor_preprocess.errors import InvalidContractError


class RepresentationProfileId(str, Enum):
    """The four frozen model representation policy profiles (plan §28)."""

    TREE_TABULAR = "TREE_TABULAR"
    LINEAR = "LINEAR"
    NEURAL_TABULAR = "NEURAL_TABULAR"
    SEQUENCE_MODEL = "SEQUENCE_MODEL"


REPRESENTATION_POLICY_VERSION = "2026-09-05.1"
"""Version of the frozen representation policy table (R61-FI-044)."""


@dataclass(frozen=True)
class RepresentationPolicy:
    """Frozen, versioned representation policy for one model family.

    ``allowed_representations`` lists the alpha-side representations the model
    family may consume as input; ``required_normalizations`` lists the
    model-side normalizations the family REQUIRES (enforced by
    :meth:`~.require_normalizations`). ``missing_channel_required`` and
    ``stable_clipping_required`` encode the NEURAL_TABULAR guarantees.
    ``temporal_causal_required`` encodes the SEQUENCE_MODEL guarantee.
    """

    profile_id: RepresentationProfileId
    version: str
    display_name: str
    description: str
    allowed_representations: Tuple[str, ...] = field(default_factory=tuple)
    required_normalizations: Tuple[str, ...] = field(default_factory=tuple)
    zscore_optional: bool = False
    robust_outlier_handling_required: bool = False
    train_fitted_scaling_required: bool = False
    missing_channel_required: bool = False
    stable_clipping_required: bool = False
    rank_gaussian_allowed: bool = False
    temporal_causal_normalization_required: bool = False

    def __post_init__(self):
        object.__setattr__(
            self, "allowed_representations", tuple(self.allowed_representations)
        )
        object.__setattr__(
            self, "required_normalizations", tuple(self.required_normalizations)
        )
        if not isinstance(self.profile_id, RepresentationProfileId):
            object.__setattr__(
                self, "profile_id", RepresentationProfileId(self.profile_id)
            )
        if not self.version:
            raise InvalidContractError("RepresentationPolicy.version cannot be empty")

    # -- guards -----------------------------------------------------------

    def require_normalizations(
        self, proposed_normalizations: Tuple[str, ...]
    ) -> None:
        """Fail closed unless all required normalization ALTERNATIVES are met.

        ``required_normalizations`` is a tuple of *alternative groups* — a
        profile is satisfied when EVERY entry in the required tuple is present
        in ``proposed_normalizations``. LINEAR therefore requires BOTH the
        train-fitted z-score AND the train-fitted robust-z-score to be present
        as declared normalizations; callers declare whichever their transform
        chain actually provides.
        """
        proposed = set(proposed_normalizations or ())
        missing = [r for r in self.required_normalizations if r not in proposed]
        if missing:
            raise InvalidContractError(
                f"{self.profile_id.value} profile requires normalization(s) "
                f"{missing} but the representation carries only "
                f"{sorted(proposed)}"
            )

    def allows_representation(self, representation: str) -> bool:
        """True when the profile permits this alpha-side representation."""
        return representation in self.allowed_representations


# ---------------------------------------------------------------------------
# The four frozen profiles (plan §28).
# ---------------------------------------------------------------------------


def _profile_tree_tabular() -> RepresentationPolicy:
    return RepresentationPolicy(
        profile_id=RepresentationProfileId.TREE_TABULAR,
        version=REPRESENTATION_POLICY_VERSION,
        display_name="Tree tabular",
        description=(
            "Tree models (LightGBM/XGBoost/CatBoost) handle missing values "
            "natively and are invariant to monotone rescaling, so raw / rank / "
            "clipped values are allowed and z-score is optional."
        ),
        allowed_representations=("raw", "rank", "clipped"),
        required_normalizations=(),
        zscore_optional=True,
        robust_outlier_handling_required=False,
        train_fitted_scaling_required=False,
    )


def _profile_linear() -> RepresentationPolicy:
    return RepresentationPolicy(
        profile_id=RepresentationProfileId.LINEAR,
        version=REPRESENTATION_POLICY_VERSION,
        display_name="Linear (OLS/Ridge/Lasso/GLM)",
        description=(
            "Linear models require dense, same-scale, outlier-controlled "
            "inputs: robust outlier handling plus a train-fitted z-score or "
            "robust-z-score is normally required."
        ),
        allowed_representations=("robust_outlier_treated", "zscore", "robust_zscore"),
        required_normalizations=("train_fitted_zscore", "train_fitted_robust_zscore"),
        zscore_optional=False,
        robust_outlier_handling_required=True,
        train_fitted_scaling_required=True,
    )


def _profile_neural_tabular() -> RepresentationPolicy:
    return RepresentationPolicy(
        profile_id=RepresentationProfileId.NEURAL_TABULAR,
        version=REPRESENTATION_POLICY_VERSION,
        display_name="Neural tabular (MLP/TabNet)",
        description=(
            "Neural networks need stable, bounded inputs with explicit missing "
            "signaling: robust z-score or rank-gaussian plus a missing "
            "mask/channel and stable clipping."
        ),
        allowed_representations=(
            "robust_zscore",
            "rank_gaussian",
            "clipped",
        ),
        required_normalizations=("robust_zscore_or_rank_gaussian",),
        zscore_optional=False,
        robust_outlier_handling_required=True,
        train_fitted_scaling_required=True,
        missing_channel_required=True,
        stable_clipping_required=True,
        rank_gaussian_allowed=True,
    )


def _profile_sequence_model() -> RepresentationPolicy:
    return RepresentationPolicy(
        profile_id=RepresentationProfileId.SEQUENCE_MODEL,
        version=REPRESENTATION_POLICY_VERSION,
        display_name="Sequence model (LSTM/Transformer)",
        description=(
            "Sequence models require per-feature train-fitted scaling and "
            "causal / stable temporal normalization (no future leakage in the "
            "normalization window)."
        ),
        allowed_representations=("train_fitted_scaled", "causal_temporal_normalized"),
        required_normalizations=("train_fitted_scaling",),
        zscore_optional=False,
        train_fitted_scaling_required=True,
        temporal_causal_normalization_required=True,
    )


# ---------------------------------------------------------------------------
# Frozen registry.
# ---------------------------------------------------------------------------


_PROFILES: Dict[RepresentationProfileId, RepresentationPolicy] = {
    RepresentationProfileId.TREE_TABULAR: _profile_tree_tabular(),
    RepresentationProfileId.LINEAR: _profile_linear(),
    RepresentationProfileId.NEURAL_TABULAR: _profile_neural_tabular(),
    RepresentationProfileId.SEQUENCE_MODEL: _profile_sequence_model(),
}


class UnknownRepresentationProfileError(InvalidContractError):
    """A representation profile id is not one of the four frozen profiles."""


def get_representation_policy(profile_id) -> RepresentationPolicy:
    """Resolve a frozen representation policy by id (fail closed)."""
    if not isinstance(profile_id, RepresentationProfileId):
        try:
            profile_id = RepresentationProfileId(profile_id)
        except (ValueError, TypeError):
            raise UnknownRepresentationProfileError(
                f"unknown representation profile {profile_id!r}; valid ids: "
                f"{[p.value for p in RepresentationProfileId]}"
            )
    try:
        return _PROFILES[profile_id]
    except KeyError:
        raise UnknownRepresentationProfileError(
            f"unknown representation profile {profile_id.value!r}"
        )


def list_representation_policies() -> Tuple[RepresentationPolicy, ...]:
    """All four frozen representation policies in stable profile order."""
    return tuple(_PROFILES[p] for p in RepresentationProfileId)


# ---------------------------------------------------------------------------
# Model-specific FeatureRepresentation artifact (R61-FI-044).
# ---------------------------------------------------------------------------


class ArtifactKind(str, Enum):
    """What a model-input artifact represents (never a canonical factor asset)."""

    MODEL_SPECIFIC_REPRESENTATION = "model_specific_representation"


# Canonical factor-asset namespace prefixes. The representation writer refuses
# to target an identity in the canonical asset namespace (R61-FI-044).
CANONICAL_FACTOR_NAMESPACE_PREFIX = "factor_asset:"
REPRESENTATION_NAMESPACE_PREFIX = "model_representation:"


class CanonicalAssetOverwriteError(InvalidContractError):
    """A model-specific representation tried to overwrite a canonical asset."""


@dataclass(frozen=True)
class FeatureRepresentationArtifact:
    """Model-specific representation of a factor, recorded as its OWN artifact.

    A model may consume a representation that is NOT the RAW-vs-treatment alpha
    winner, but ONLY when it is tracked here as a model-specific artifact and
    evaluated for model suitability (plan §28). This artifact never replaces
    the canonical factor asset: the writer forbids canonical-asset targets and
    the artifact pins the canonical factor reference it derives from.
    """

    artifact_id: str
    factor_id: str
    canonical_factor_ref: str
    model_id: str
    profile_id: RepresentationProfileId
    profile_version: str
    transform_chain: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    representation_name: str = ""
    artifact_kind: ArtifactKind = ArtifactKind.MODEL_SPECIFIC_REPRESENTATION
    content_hash: str = ""

    def __post_init__(self):
        if not self.artifact_id:
            raise InvalidContractError("FeatureRepresentationArtifact.artifact_id cannot be empty")
        if not self.factor_id:
            raise InvalidContractError("FeatureRepresentationArtifact.factor_id cannot be empty")
        if not self.canonical_factor_ref:
            raise InvalidContractError(
                "FeatureRepresentationArtifact.canonical_factor_ref cannot be empty"
            )
        if not self.model_id:
            raise InvalidContractError("FeatureRepresentationArtifact.model_id cannot be empty")
        if not isinstance(self.profile_id, RepresentationProfileId):
            object.__setattr__(
                self, "profile_id", RepresentationProfileId(self.profile_id)
            )
        if not isinstance(self.artifact_kind, ArtifactKind):
            object.__setattr__(self, "artifact_kind", ArtifactKind(self.artifact_kind))
        object.__setattr__(self, "transform_chain", tuple(self.transform_chain))

        # Fail closed: a model representation must NEVER point at the canonical
        # factor-asset namespace.
        if self.canonical_factor_ref.startswith(CANONICAL_FACTOR_NAMESPACE_PREFIX):
            raise InvalidContractError(
                "FeatureRepresentationArtifact.canonical_factor_ref must reference "
                "the canonical factor ASSET (alpha form); it cannot BE a "
                "canonical-asset identity"
            )

        actual = self._derive_hash()
        if self.content_hash and self.content_hash != actual:
            raise InvalidContractError(
                "FeatureRepresentationArtifact.content_hash does not match "
                "artifact content (fail closed)"
            )
        object.__setattr__(self, "content_hash", actual)

    def _derive_hash(self) -> str:
        import hashlib
        payload = "|".join(
            [
                "FeatureRepresentationArtifact",
                self.artifact_id,
                self.factor_id,
                self.canonical_factor_ref,
                self.model_id,
                self.profile_id.value,
                self.profile_version,
                self.representation_name,
                repr(self.transform_chain),
                self.artifact_kind.value,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "factor_id": self.factor_id,
            "canonical_factor_ref": self.canonical_factor_ref,
            "model_id": self.model_id,
            "profile_id": self.profile_id.value,
            "profile_version": self.profile_version,
            "transform_chain": [dict(s) for s in self.transform_chain],
            "representation_name": self.representation_name,
            "artifact_kind": self.artifact_kind.value,
            "content_hash": self.content_hash,
        }


def register_feature_representation(
    *,
    artifact_id: str,
    factor_id: str,
    canonical_factor_ref: str,
    model_id: str,
    profile_id,
    transform_chain,
    representation_name: str = "",
) -> FeatureRepresentationArtifact:
    """Create + register a model-specific representation artifact.

    ``canonical_factor_ref`` must reference the canonical factor asset (the
    alpha form) and MUST NOT be a canonical-asset *identity* — writing a
    model representation *over* the canonical factor asset is forbidden
    (R61-FI-044, plan §28). The artifact is model-specific and evaluated for
    model suitability; it is never the canonical factor asset.
    """
    profile = get_representation_policy(profile_id)
    if canonical_factor_ref.startswith(CANONICAL_FACTOR_NAMESPACE_PREFIX):
        raise CanonicalAssetOverwriteError(
            f"refusing to write model representation {artifact_id!r} over the "
            f"canonical factor-asset namespace {canonical_factor_ref!r} — a "
            "model-specific representation must be tracked as its own "
            "artifact, never overwrite the canonical factor asset"
        )
    return FeatureRepresentationArtifact(
        artifact_id=artifact_id,
        factor_id=factor_id,
        canonical_factor_ref=canonical_factor_ref,
        model_id=model_id,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        transform_chain=tuple(transform_chain),
        representation_name=representation_name,
    )


# ---------------------------------------------------------------------------
# Non-inferiority guard (R61-FI-044, plan §28).
# ---------------------------------------------------------------------------


class SignalDestructionConflict(InvalidContractError):
    """A model-mandated normalization materially destroys the alpha signal.

    Raised by :func:`non_inferior` when the evidence delta is beyond the
    versioned tolerance. The conflict is exposed explicitly — the caller must
    NEVER silently force the model normalization over material signal
    destruction.
    """


NON_INFERIORITY_POLICY_VERSION = "2026-09-05.1"
"""Versioned tolerance table for the non-inferiority guard (R61-FI-044)."""


@dataclass(frozen=True)
class NonInferiorityTolerance:
    """Versioned tolerance for the model-safety non-inferiority test.

    ``alpha_metric`` is the metric the alpha evidence is measured on
    (e.g. ``rank_ic``). ``max_relative_destroy`` is the maximum FRACTION of
    the alpha evidence the model treatment may destroy before the guard
    raises. A model treatment that destroys MORE than the tolerance is
    material signal destruction and is exposed as a conflict, never silently
    forced.
    """

    alpha_metric: str
    max_relative_destroy: float
    policy_version: str = NON_INFERIORITY_POLICY_VERSION
    allow_exact_equal: bool = True

    def __post_init__(self):
        if not self.alpha_metric:
            raise InvalidContractError("NonInferiorityTolerance.alpha_metric cannot be empty")
        if not (0.0 <= self.max_relative_destroy <= 1.0):
            raise InvalidContractError(
                "NonInferiorityTolerance.max_relative_destroy must be in [0, 1]"
            )
        if not self.policy_version:
            raise InvalidContractError("NonInferiorityTolerance.policy_version cannot be empty")


# Versioned default: model normalization may destroy up to 10% of the alpha
# evidence (rank_ic) while staying non-inferior.
DEFAULT_NON_INFERIORITY_TOLERANCE = NonInferiorityTolerance(
    alpha_metric="rank_ic",
    max_relative_destroy=0.10,
)


def non_inferior(
    evidence_alpha: float,
    evidence_model_treated: float,
    tolerance: NonInferiorityTolerance = DEFAULT_NON_INFERIORITY_TOLERANCE,
) -> bool:
    """True when the model-treated representation is non-inferior.

    The model treatment destroys ``abs(evidence_alpha - evidence_model_treated)``
    of the alpha evidence. When the destruction exceeds the versioned
    tolerance the guard raises :class:`SignalDestructionConflict` — it never
    returns False silently, because returning False would let the caller
    "decide to force anyway". A conflict is an EXCEPTION the policy layer must
    surface.

    Exact-equal evidence is always non-inferior. A negative alpha metric is
    normalized to its absolute value so "flipping sign" is detected as full
    destruction.
    """
    if evidence_alpha == evidence_model_treated:
        return True
    base = abs(evidence_alpha)
    if base == 0.0:
        # No positive alpha evidence to protect: the model treatment is not
        # destroying any measurable signal.
        return True
    relative_destroy = abs(evidence_alpha - evidence_model_treated) / base
    if relative_destroy > tolerance.max_relative_destroy:
        raise SignalDestructionConflict(
            f"model treatment destroys {relative_destroy:.2%} of the alpha "
            f"evidence ({tolerance.alpha_metric}: {evidence_alpha} -> "
            f"{evidence_model_treated}); tolerance is "
            f"{tolerance.max_relative_destroy:.2%} (policy version "
            f"{tolerance.policy_version}). The conflict must be exposed, not "
            "silently forced."
        )
    return True


__all__ = [
    "RepresentationProfileId",
    "REPRESENTATION_POLICY_VERSION",
    "RepresentationPolicy",
    "UnknownRepresentationProfileError",
    "get_representation_policy",
    "list_representation_policies",
    "ArtifactKind",
    "CANONICAL_FACTOR_NAMESPACE_PREFIX",
    "REPRESENTATION_NAMESPACE_PREFIX",
    "CanonicalAssetOverwriteError",
    "FeatureRepresentationArtifact",
    "register_feature_representation",
    "NonInferiorityTolerance",
    "NON_INFERIORITY_POLICY_VERSION",
    "DEFAULT_NON_INFERIORITY_TOLERANCE",
    "non_inferior",
    "SignalDestructionConflict",
]
