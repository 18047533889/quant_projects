"""
Fitted transform state contract.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from factor_preprocess.errors import (
    MissingInputError,
    InvalidContractError,
    TimingContractError,
)


@dataclass(frozen=True)
class FittedState:
    """
    Immutable state from fitting a transform.

    Records fit window, learned parameters, and metadata for reproducibility.
    """
    state_id: str
    transform_name: str
    transform_version: str

    # Fit window metadata (required for causality)
    fit_start_time: datetime
    fit_end_time: datetime
    fit_universe_ref: Optional[str] = None

    # Feature contract
    feature_ids: List[str] = field(default_factory=list)
    feature_order: List[str] = field(default_factory=list)

    # Learned parameters
    learned_params: Dict[str, Any] = field(default_factory=dict)
    learned_params_hash: Optional[str] = None

    # Provenance
    config_hash: Optional[str] = None
    created_at: Optional[datetime] = None
    producer: str = "factor_preprocess"
    producer_version: str = "0.1.0"

    def __post_init__(self):
        """Validate state on construction."""
        if not self.state_id:
            raise MissingInputError("FittedState.state_id cannot be empty")
        if not self.transform_name:
            raise MissingInputError("FittedState.transform_name cannot be empty")

        # Fit window validation
        if self.fit_start_time >= self.fit_end_time:
            raise TimingContractError("fit_start_time must be before fit_end_time")

        # Feature contract validation
        if self.feature_ids and not self.feature_order:
            raise InvalidContractError("feature_order required when feature_ids provided")
        if len(self.feature_ids) != len(set(self.feature_ids)):
            raise InvalidContractError("Duplicate feature IDs")
        if len(self.feature_order) != len(set(self.feature_order)):
            raise InvalidContractError("Duplicate features in feature_order")

    def is_compatible_with(self, factor_ids: List[str]) -> bool:
        """Check if this state can transform the given factors."""
        if not self.feature_ids:
            # No feature contract, assume compatible
            return True
        return set(factor_ids) == set(self.feature_ids)
