"""JobSpec / JobStatus / ErrorClass — platform job contract.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.3 (spec §9,
§12.2, §42, §43). PURE stdlib frozen dataclass + enums. ``JobStatus`` is the
execution status and is distinct from ``LifecycleState`` and ``HealthState``
(spec §9 invariant).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from ._contenthash import content_hash

__all__ = [
    "JobStatus",
    "ErrorClass",
    "JobSpec",
    "idempotency_key",
    "materialization_idempotency_key",
    "qe_idempotency_key",
    "treatment_idempotency_key",
]


class JobStatus(enum.Enum):
    """Execution status (spec §9). Distinct from LifecycleState / HealthState."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    BLOCKED_DATA = "BLOCKED_DATA"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
    TIMED_OUT = "TIMED_OUT"


class ErrorClass(enum.Enum):
    """Worker error taxonomy (spec §43)."""

    RETRYABLE_INFRASTRUCTURE = "RetryableInfrastructureError"
    RETRYABLE_STORAGE = "RetryableStorageError"
    RETRYABLE_DATABASE = "RetryableDatabaseError"
    DATA_UNAVAILABLE = "DataUnavailableError"
    INVALID_INPUT = "InvalidInputError"
    SEMANTIC_CONTRACT = "SemanticContractError"
    CAPABILITY = "CapabilityError"
    NUMERICAL_FAILURE = "NumericalFailure"
    RESOURCE_EXCEEDED = "ResourceExceededError"
    CANCELLATION = "CancellationError"

    @property
    def retryable(self) -> bool:
        """Whether this error class is retried (spec §43)."""
        return self in {
            ErrorClass.RETRYABLE_INFRASTRUCTURE,
            ErrorClass.RETRYABLE_STORAGE,
            ErrorClass.RETRYABLE_DATABASE,
            ErrorClass.DATA_UNAVAILABLE,
        }


@dataclass(frozen=True)
class JobSpec:
    """Declared activity contract (spec §12.2)."""

    job_id: str
    job_type: str
    idempotency_key: str
    activity_kind: str | None = None
    priority: int = 0
    max_retries: int = 0
    timeout_seconds: float | None = None
    heartbeat_seconds: float | None = None
    resource_class: str = "light"
    estimated_factor_count: int = 0
    estimated_row_count: int = 0
    input_artifact_refs: tuple[str, ...] = ()
    output_artifact_refs: tuple[str, ...] = ()
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id is required")
        if not self.job_type:
            raise ValueError("job_type is required")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")
        if self.priority < 0:
            raise ValueError("priority must be >= 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


def idempotency_key(*fields: object) -> str:
    """sha256 of a canonical tuple of fields (spec §42)."""
    return content_hash(*fields)


def materialization_idempotency_key(
    factor_definition_id: str,
    data_snapshot_id: str,
    universe_id: str,
    calculation_spec_id: str,
) -> str:
    """Materialization idempotency key (spec §42)."""
    return idempotency_key(
        factor_definition_id, data_snapshot_id, universe_id, calculation_spec_id
    )


def qe_idempotency_key(
    factor_value_id: str,
    evaluation_policy_id: str,
    label_id: str,
    profile_id: str,
) -> str:
    """QE evaluation idempotency key (spec §42)."""
    return idempotency_key(factor_value_id, evaluation_policy_id, label_id, profile_id)


def treatment_idempotency_key(
    source_evidence: str, search_policy: str, split_plan: str
) -> str:
    """Treatment idempotency key (spec §42)."""
    return idempotency_key(source_evidence, search_policy, split_plan)
