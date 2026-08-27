"""ModelVersion — immutable training result (spec §2, §17, §18).

DRAFT. PURE stdlib frozen dataclasses. The FOURTH layer of the versioning chain:

    ClusterVersion -> FactorLibraryVersion -> FeatureSetVersion -> ModelVersion

A ``ModelVersion`` is the *training result*: model_architecture, training_params,
weights, a reference to the FeatureSetVersion it was trained on, plus the
LabelDefinition, DataSnapshot and SplitPlan. ``weight_in_model`` lives HERE (not
in ``FactorLibraryVersion`` / ``LibraryMembership``).

``ModelVersion`` is immutable. A retrain triggered by a ``FeatureSetDiff``
produces a NEW ``ModelVersion``; the old one is preserved in history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "LabelDefinition",
    "DataSnapshot",
    "SplitPlan",
    "ModelWeight",
    "ModelVersion",
]


@dataclass(frozen=True)
class LabelDefinition:
    """The label / target definition a model is trained against (spec §18)."""

    label_definition_id: str
    label_name: str
    horizon: str
    return_basis: str = "vwap"  # global caliber: Vwap.pct_change().shift(-1)
    aggregation: str = ""
    universe_id: str = ""

    def __post_init__(self) -> None:
        if not self.label_definition_id:
            raise ValueError("label_definition_id is required")
        if not self.label_name:
            raise ValueError("label_name is required")
        if not self.horizon:
            raise ValueError("horizon is required")


@dataclass(frozen=True)
class DataSnapshot:
    """The data snapshot a model was trained on (spec §2)."""

    snapshot_id: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    universe_id: str = ""
    data_revision_ref: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")


@dataclass(frozen=True)
class SplitPlan:
    """Train/validation/test split plan (spec §2)."""

    split_plan_id: str
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.split_plan_id:
            raise ValueError("split_plan_id is required")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ModelWeight:
    """One feature's weight within a model (spec §17).

    ``weight_in_model`` belongs to the MODEL, not the library. The feature is
    identified by its FeatureSet member identity (feature_name + position).
    """

    feature_name: str
    position: int
    weight: float

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("feature_name is required")
        if self.position < 0:
            raise ValueError("position must be >= 0")


@dataclass(frozen=True)
class ModelVersion:
    """Immutable training result (spec §2, §17, §18)."""

    model_id: str
    model_version: str
    model_architecture: str
    feature_set_version_ref: str
    label_definition: LabelDefinition
    data_snapshot: DataSnapshot
    split_plan: SplitPlan
    training_params: dict[str, Any] = field(default_factory=dict)
    weights: tuple[ModelWeight, ...] = ()
    training_run_ref: str = ""
    eval_metrics_uri: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.model_version:
            raise ValueError("model_version is required")
        if not self.model_architecture:
            raise ValueError("model_architecture is required")
        if not self.feature_set_version_ref:
            raise ValueError("feature_set_version_ref is required")
