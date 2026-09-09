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
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from factor_preprocess.contracts._deep_freeze import as_plain, deep_freeze
from factor_preprocess.contracts.treatment_recipe import _stable_repr
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
    required_normalization_any_of: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
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
        object.__setattr__(
            self, "required_normalization_any_of",
            tuple(tuple(group) for group in self.required_normalization_any_of),
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
        missing_groups = [
            group for group in self.required_normalization_any_of
            if not proposed.intersection(group)
        ]
        if missing or missing_groups:
            raise InvalidContractError(
                f"{self.profile_id.value} profile requires all_of={missing} "
                f"and one_of_each={missing_groups} but the representation carries only "
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
        required_normalizations=(),
        required_normalization_any_of=((
            "train_fitted_zscore", "train_fitted_robust_zscore"
        ),),
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
    factor_version: str
    canonical_storage_ref: str
    destination_storage_ref: str
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
        if not self.factor_version:
            raise InvalidContractError("FeatureRepresentationArtifact.factor_version cannot be empty")
        if not self.destination_storage_ref:
            raise InvalidContractError("destination_storage_ref cannot be empty")
        if not self.canonical_storage_ref:
            raise InvalidContractError("canonical_storage_ref cannot be empty")
        if not isinstance(self.profile_id, RepresentationProfileId):
            object.__setattr__(
                self, "profile_id", RepresentationProfileId(self.profile_id)
            )
        if not isinstance(self.artifact_kind, ArtifactKind):
            object.__setattr__(self, "artifact_kind", ArtifactKind(self.artifact_kind))
        object.__setattr__(self, "transform_chain", deep_freeze(tuple(self.transform_chain)))

        expected_source = (
            f"{CANONICAL_FACTOR_NAMESPACE_PREFIX}{self.factor_id}@{self.factor_version}"
        )
        if self.canonical_factor_ref != expected_source:
            raise InvalidContractError(
                "canonical_factor_ref must identify the matching canonical "
                f"factor/version {expected_source!r}"
            )
        if not self.artifact_id.startswith(REPRESENTATION_NAMESPACE_PREFIX):
            raise CanonicalAssetOverwriteError(
                "FeatureRepresentationArtifact.artifact_id must be in the "
                "model_representation namespace"
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
                self.factor_version,
                self.canonical_storage_ref,
                self.destination_storage_ref,
                self.representation_name,
                _stable_repr(self.transform_chain),
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
            "factor_version": self.factor_version,
            "canonical_storage_ref": self.canonical_storage_ref,
            "destination_storage_ref": self.destination_storage_ref,
            "transform_chain": [as_plain(s) for s in self.transform_chain],
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
    factor_version: str,
    canonical_storage_ref: str,
    destination_storage_ref: str,
    profile_id,
    transform_chain,
    representation_name: str = "",
) -> FeatureRepresentationArtifact:
    """Create + register a model-specific representation artifact.

    ``canonical_factor_ref`` is the typed source identity and must be in the
    canonical ``factor_asset`` domain. ``artifact_id`` is the distinct write
    identity and must be in the ``model_representation`` domain. Resolved
    source and destination locations are bound into the content hash and are
    checked again by :func:`write_feature_representation`.
    """
    profile = get_representation_policy(profile_id)
    if not artifact_id.startswith(REPRESENTATION_NAMESPACE_PREFIX):
        raise CanonicalAssetOverwriteError(
            f"representation destination {artifact_id!r} is outside the "
            "model_representation namespace"
        )
    return FeatureRepresentationArtifact(
        artifact_id=artifact_id,
        factor_id=factor_id,
        canonical_factor_ref=canonical_factor_ref,
        model_id=model_id,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        factor_version=factor_version,
        canonical_storage_ref=canonical_storage_ref,
        destination_storage_ref=destination_storage_ref,
        transform_chain=tuple(transform_chain),
        representation_name=representation_name,
    )


def write_feature_representation(
    artifact: FeatureRepresentationArtifact,
    payload: bytes,
    *,
    canonical_storage_path,
    representation_storage_root,
) -> Path:
    """Create a representation file without overwriting any resolved object."""
    if not isinstance(artifact, FeatureRepresentationArtifact):
        raise TypeError("artifact must be a FeatureRepresentationArtifact")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if artifact.content_hash != artifact._derive_hash():
        raise InvalidContractError("representation artifact content hash is stale or tampered")
    source = Path(canonical_storage_path).resolve(strict=True)
    registered_source = Path(artifact.canonical_storage_ref).resolve(strict=True)
    if source != registered_source:
        raise InvalidContractError(
            "canonical storage path does not resolve to the registered parent asset"
        )
    root = Path(representation_storage_root).resolve(strict=True)
    destination = Path(artifact.destination_storage_ref).resolve(strict=False)
    if destination == source:
        raise CanonicalAssetOverwriteError(
            "resolved representation destination aliases the canonical source"
        )
    if destination != root and root not in destination.parents:
        raise CanonicalAssetOverwriteError(
            "resolved representation destination is outside its storage domain"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"representation destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


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


def decide_representation(provider, request, treated_candidate_id: str):
    """Delegate representation admission to the shared DecisionProvider.

    FP intentionally owns no score table here.  The same immutable request can
    therefore be replayed through FO and FP and yields the provider's canonical
    DecisionArtifact/content hash and comparison relationship.
    """
    if provider is None or not callable(getattr(provider, "decide", None)):
        raise InvalidContractError("provider must implement DecisionProvider.decide(request)")
    if not treated_candidate_id:
        raise InvalidContractError("treated_candidate_id is required")
    candidate_ids = {c.candidate_id for c in request.candidates}
    if treated_candidate_id not in candidate_ids:
        raise InvalidContractError("treated candidate is not in the canonical request")
    artifact = provider.decide(request)
    # Admission/status semantics belong to the provider's versioned policy;
    # this adapter deliberately returns its artifact unchanged.
    return artifact


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

    The model treatment destroys only a signed deterioration. Improvement is
    never converted into destruction by an absolute value.
    of the alpha evidence. When the destruction exceeds the versioned
    tolerance the guard raises :class:`SignalDestructionConflict` — it never
    returns False silently, because returning False would let the caller
    "decide to force anyway". A conflict is an EXCEPTION the policy layer must
    surface.

    Exact-equal evidence is always non-inferior. A negative alpha metric is
    normalized to its absolute value so "flipping sign" is detected as full
    destruction.
    """
    if isinstance(evidence_alpha, bool) or isinstance(evidence_model_treated, bool):
        raise InvalidContractError("non-inferiority evidence must be finite numeric values")
    try:
        finite = math.isfinite(float(evidence_alpha)) and math.isfinite(
            float(evidence_model_treated)
        )
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        raise InvalidContractError("non-inferiority evidence must be finite numeric values")
    if evidence_alpha == evidence_model_treated:
        return True
    base = abs(evidence_alpha)
    if base == 0.0:
        if evidence_model_treated >= 0.0:
            return True
        raise SignalDestructionConflict(
            "model treatment degrades a zero alpha baseline; relative "
            "non-inferiority is undefined and cannot be auto-approved"
        )
    signed_destroy = evidence_alpha - evidence_model_treated
    if signed_destroy <= 0.0:
        return True
    relative_destroy = signed_destroy / base
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
    "write_feature_representation",
    "NonInferiorityTolerance",
    "NON_INFERIORITY_POLICY_VERSION",
    "DEFAULT_NON_INFERIORITY_TOLERANCE",
    "non_inferior",
    "SignalDestructionConflict",
    "decide_representation",
]
