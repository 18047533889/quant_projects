"""BacktestProvider Protocol + Backtest/Model DTOs.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.12 (spec §41,
§2). PURE stdlib frozen dataclasses + ``typing.Protocol``. Cheap ``qe.probe.*``
is strictly distinct from formal ``bt.*`` execution backtests. ``[RECONCILE]``
against ``vectorbt_qs/contracts/backtest.py`` (``BacktestArtifact``) before
freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from .artifact_ref import ArtifactRef
from .timing import EvidenceStatus

__all__ = [
    "BacktestProvider",
    "BacktestRequest",
    "BacktestArtifactRef",
    "ModelDatasetRequest",
    "ModelTrainingRequest",
    "ModelArtifactRef",
]


@dataclass(frozen=True)
class BacktestRequest:
    """Backtest request (spec §41)."""

    request_id: str
    idempotency_key: str
    signal_artifact_ref: ArtifactRef
    execution_policy_ref: str
    universe_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")
        if not self.execution_policy_ref:
            raise ValueError("execution_policy_ref is required")


@dataclass(frozen=True)
class BacktestArtifactRef:
    """Backtest artifact ref (spec §41). ``artifact_type=BACKTEST``."""

    artifact: ArtifactRef
    backtest_id: str
    strategy_ref: str
    request_ref: str
    result_summary_uri: str
    qe_reevaluation_status: EvidenceStatus = EvidenceStatus.NOT_COMPUTED

    def __post_init__(self) -> None:
        if self.artifact.artifact_type != "BACKTEST":
            raise ValueError("BacktestArtifactRef requires artifact_type=BACKTEST")
        if not self.backtest_id:
            raise ValueError("backtest_id is required")


@dataclass(frozen=True)
class ModelDatasetRequest:
    """Model dataset request (spec §2)."""

    dataset_request_id: str
    feature_set_artifacts: tuple[str, ...] = ()
    label_policy_ref: str = ""
    output_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_request_id:
            raise ValueError("dataset_request_id is required")


@dataclass(frozen=True)
class ModelTrainingRequest:
    """Model training request (spec §2)."""

    training_request_id: str
    idempotency_key: str
    model_dataset_ref: ArtifactRef
    model_spec_ref: str
    training_params: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.training_request_id:
            raise ValueError("training_request_id is required")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")
        if not self.model_spec_ref:
            raise ValueError("model_spec_ref is required")
        if self.training_params is None:
            object.__setattr__(self, "training_params", {})


@dataclass(frozen=True)
class ModelArtifactRef:
    """Model artifact ref (spec §2). ``artifact_type=MODEL``."""

    artifact: ArtifactRef
    model_id: str
    model_version: str
    feature_set_ref: str
    train_dataset_ref: str
    eval_metrics_uri: str = ""

    def __post_init__(self) -> None:
        if self.artifact.artifact_type != "MODEL":
            raise ValueError("ModelArtifactRef requires artifact_type=MODEL")
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.model_version:
            raise ValueError("model_version is required")


@runtime_checkable
class BacktestProvider(Protocol):
    """Backtest execution abstraction (spec §41)."""

    def run(
        self,
        signal_artifact: ArtifactRef,
        request: BacktestRequest,
        execution_policy: str,
    ) -> BacktestArtifactRef:
        """Run a formal execution backtest; return a BacktestArtifactRef."""
        ...
