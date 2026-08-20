"""Registry package - transform and policy catalogs."""

from factor_preprocess.registry.transforms import (
    TransformRegistry,
    TransformMetadata,
    TransformCategory,
    create_default_registry,
    get_default_registry,
)
from factor_preprocess.registry.policies import (
    PolicyRegistry,
    PolicyPreset,
    PolicyLevel,
    TransformStep,
    create_default_policies,
    get_default_policy_registry,
)

# Deprecated compatibility view retained from the old contracts policy
# system. New code should use the registry-based PolicyPreset directly.
from factor_preprocess.contracts.policy import PreprocessingPolicy, TransformSpec, TransformKind, TransformMode

__all__ = [
    # Canonical policy authority
    "PolicyRegistry",
    "PolicyPreset",
    "PolicyLevel",
    "TransformStep",
    "create_default_policies",
    "get_default_policy_registry",
    # Transform registry
    "TransformRegistry",
    "TransformMetadata",
    "TransformCategory",
    "create_default_registry",
    "get_default_registry",
    # Deprecated compatibility view
    "PreprocessingPolicy",
    "TransformSpec",
    "TransformKind",
    "TransformMode",
]
