"""
Versioned contract envelope — repeated across packages without runtime dependency.

Every cross-package record uses this envelope structure per CONTRACT_FREEZE_DRAFT §4.
Large values are Arrow/NumPy/Polars or external value_ref; never embedded in JSON metadata.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ContractEnvelope:
    """
    Versioned envelope for cross-package contracts.

    This structure is repeated (not imported) across packages to avoid
    a shared runtime dependency. Each package projects this into its
    local types.
    """
    schema_version: str
    producer: str
    producer_version: str
    created_at: str  # ISO 8601 timestamp
    run_id: str
    factor_ids: tuple[str, ...]
    market: Optional[str] = None
    frequency: Optional[str] = None
    universe_ref: Optional[str] = None
    source_snapshot_ref: Optional[str] = None
    parent_refs: tuple[str, ...] = ()
    config_hash: Optional[str] = None
    code_hash: Optional[str] = None

    def __post_init__(self):
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if not self.producer:
            raise ValueError("producer is required")
        if not self.producer_version:
            raise ValueError("producer_version is required")
        if not self.run_id:
            raise ValueError("run_id is required")
