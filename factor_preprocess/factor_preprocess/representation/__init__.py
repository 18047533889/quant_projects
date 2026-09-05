"""Representation package - multi-channel representations for model input."""

from factor_preprocess.representation.multichannel import (
    build_multichannel,
    MultichannelConfig,
    MultichannelResult,
)
from factor_preprocess.representation.linear_ready import (
    build_linear_ready,
    assess_collinearity,
    LinearReadyConfig,
    LinearReadyResult,
)
from factor_preprocess.representation.tree_ready import (
    build_tree_ready,
    suggest_tree_params,
    TreeReadyConfig,
    TreeReadyResult,
)
from factor_preprocess.representation.neural_ready import (
    build_neural_ready,
    prepare_embeddings,
    NeuralReadyConfig,
    NeuralReadyResult,
)
from factor_preprocess.representation.policy import (
    RepresentationProfileId,
    REPRESENTATION_POLICY_VERSION,
    RepresentationPolicy,
    UnknownRepresentationProfileError,
    get_representation_policy,
    list_representation_policies,
    ArtifactKind,
    CANONICAL_FACTOR_NAMESPACE_PREFIX,
    REPRESENTATION_NAMESPACE_PREFIX,
    CanonicalAssetOverwriteError,
    FeatureRepresentationArtifact,
    register_feature_representation,
    NonInferiorityTolerance,
    NON_INFERIORITY_POLICY_VERSION,
    DEFAULT_NON_INFERIORITY_TOLERANCE,
    non_inferior,
    SignalDestructionConflict,
)

__all__ = [
    "build_multichannel",
    "MultichannelConfig",
    "MultichannelResult",
    "build_linear_ready",
    "assess_collinearity",
    "LinearReadyConfig",
    "LinearReadyResult",
    "build_tree_ready",
    "suggest_tree_params",
    "TreeReadyConfig",
    "TreeReadyResult",
    "build_neural_ready",
    "prepare_embeddings",
    "NeuralReadyConfig",
    "NeuralReadyResult",
    # R61-FI-044 model-specific representation policy profiles
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
