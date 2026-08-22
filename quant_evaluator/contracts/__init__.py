"""
Core contracts for quant_evaluator.
"""

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.artifact_types import (
    ICSeriesArtifact,
    QuantileReturnArtifact,
    ProbePortfolioArtifact,
    ExposureArtifact,
)
from quant_evaluator.contracts.errors import (
    QuantEvaluatorError,
    ContractError,
    SchemaVersionError,
    MissingInputError,
    InvalidContractError,
    TimingContractError,
    SnapshotMismatchError,
    SealedSplitOverlapError,
    CapabilityError,
    UnsupportedMetricError,
    OptionalDependencyMissing,
    DataError,
    InsufficientObservations,
    InvalidValidityMask,
    MissingLabelError,
    EvidenceUnavailableError,
    NumericalFailure,
    OverflowOrNonFiniteError,
)
from quant_evaluator.contracts.sealed_split import SealedSplitRef, check_sealed_split_overlap

__all__ = [
    "FactorBatch",
    "AxisRef",
    "LabelBundle",
    "ICSeriesArtifact",
    "QuantileReturnArtifact",
    "ProbePortfolioArtifact",
    "ExposureArtifact",
    "QuantEvaluatorError",
    "ContractError",
    "SchemaVersionError",
    "MissingInputError",
    "InvalidContractError",
    "TimingContractError",
    "SnapshotMismatchError",
    "SealedSplitOverlapError",
    "CapabilityError",
    "UnsupportedMetricError",
    "OptionalDependencyMissing",
    "DataError",
    "InsufficientObservations",
    "InvalidValidityMask",
    "MissingLabelError",
    "EvidenceUnavailableError",
    "NumericalFailure",
    "OverflowOrNonFiniteError",
    "SealedSplitRef",
    "check_sealed_split_overlap",
]
