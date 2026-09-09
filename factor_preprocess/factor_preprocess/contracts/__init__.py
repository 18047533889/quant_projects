"""Contracts package."""
from factor_preprocess.contracts.policy import (
    PreprocessingPolicy,
    TransformSpec,
    TransformKind,
    TransformMode,
)
from factor_preprocess.contracts.state import FittedState
from factor_preprocess.contracts.feature_bundle import (
    FeatureBundle,
    AxisRef,
    ChannelRef,
    FeatureManifest,
)
from factor_preprocess.contracts.treatment_lineage import OutputProperties, derive_output_properties
from factor_preprocess.contracts.treatment_spec import (
    PriceBasis,
    ValueUnit,
    MaterializationSplit,
    TreatmentSpecIdentity,
    TreatmentMaterializationIdentity,
    neutralization_spec_identity,
    neutralization_binding,
    ensure_materialization_split_valid,
    materialize_identity,
    spec_to_materialization_identity,
    ASHARE_INDUSTRY_SCHEMA,
    ASHARE_SIZE_DEFINITION,
)

__all__ = [
    "PreprocessingPolicy",
    "TransformSpec",
    "TransformKind",
    "TransformMode",
    "FittedState",
    "FeatureBundle",
    "AxisRef",
    "ChannelRef",
    "FeatureManifest",
    "OutputProperties",
    "derive_output_properties",
    # Treatment identity split (P0-FP #103)
    "PriceBasis",
    "ValueUnit",
    "MaterializationSplit",
    "TreatmentSpecIdentity",
    "TreatmentMaterializationIdentity",
    "neutralization_spec_identity",
    "neutralization_binding",
    "ensure_materialization_split_valid",
    "materialize_identity",
    "spec_to_materialization_identity",
    "ASHARE_INDUSTRY_SCHEMA",
    "ASHARE_SIZE_DEFINITION",
]
