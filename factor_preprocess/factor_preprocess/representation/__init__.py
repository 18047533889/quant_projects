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
]
