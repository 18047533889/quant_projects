"""
Label bundle contract for evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class LabelBundle:
    """
    Explicit forward labels with strict timing.

    QE consumes this contract and never infers labels, shifts, or fills. All timing
    is supplied by caller and validated, not guessed from file timestamps.
    """
    target_id: str
    values: np.ndarray
    horizon: int
    execution_delay: int = 0
    decision_time: Tuple[Any, ...] = field(default_factory=tuple)
    execution_time: Tuple[Any, ...] = field(default_factory=tuple)
    label_start_time: Tuple[Any, ...] = field(default_factory=tuple)
    label_end_time: Tuple[Any, ...] = field(default_factory=tuple)
    validity: Optional[np.ndarray] = None
    source_ref: Optional[str] = None
    calendar_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        if self.values is None:
            raise ValueError("values cannot be None")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.execution_delay < 0:
            raise ValueError(f"execution_delay cannot be negative, got {self.execution_delay}")

        if not self.decision_time:
            raise ValueError("decision_time (explicit timing) is required, never inferred")
        if not self.label_start_time:
            raise ValueError("label_start_time is required")
        if not self.label_end_time:
            raise ValueError("label_end_time is required")

        if self.validity is not None and self.validity.shape != self.values.shape:
            raise ValueError(
                f"Validity shape {self.validity.shape} must match values shape {self.values.shape}"
            )

    def num_observations(self) -> int:
        return len(self.values) if hasattr(self.values, "__len__") else 1
