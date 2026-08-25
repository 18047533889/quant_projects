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

    Return convention: the default forward-return measure is vwap-to-vwap
    (VWAP_{t+H} / VWAP_t - 1).  External callers provide ``values`` in this
    convention; the field is recorded as provenance only and is NOT enforced
    or validated against ``target_id`` / content.
    """
    target_id: str
    values: np.ndarray
    horizon: int
    execution_delay: int = 0
    decision_time: Tuple[Any, ...] = field(default_factory=tuple)
    execution_time: Tuple[Any, ...] = field(default_factory=tuple)
    signal_available_time: Tuple[Any, ...] = field(default_factory=tuple)
    label_start_time: Tuple[Any, ...] = field(default_factory=tuple)
    label_end_time: Tuple[Any, ...] = field(default_factory=tuple)
    validity: Optional[np.ndarray] = None
    source_ref: Optional[str] = None
    calendar_ref: Optional[str] = None
    price_convention: str = "vwap_to_vwap"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.target_id:
            raise ValueError("target_id cannot be empty")
        if not self.price_convention:
            raise ValueError("price_convention cannot be empty")
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

        if self.values.ndim not in (1, 2):
            raise ValueError("LabelBundle values must be 1D or 2D")
        n_times = self.values.shape[0]
        for name, timing in (("decision_time", self.decision_time), ("label_start_time", self.label_start_time), ("label_end_time", self.label_end_time)):
            if len(timing) != n_times:
                raise ValueError(f"{name} length {len(timing)} must match label time length {n_times}")
        if self.execution_time and len(self.execution_time) != n_times:
            raise ValueError(f"execution_time length {len(self.execution_time)} must match label time length {n_times}")
        try:
            if any(self.decision_time[i] >= self.decision_time[i + 1] for i in range(n_times - 1)):
                raise ValueError("decision_time must be strictly increasing")
        except TypeError as exc:
            raise ValueError("decision_time must be orderable") from exc

        # Causal chain (elementwise, where supplied):
        #   decision_time <= signal_available_time (default = decision_time)
        #   signal_available_time <= execution_time (where execution_time supplied)
        #   execution_time <= label_start_time (where supplied; else
        #       signal_available_time <= label_start_time)
        #   label_start_time < label_end_time (strict, positive label window)
        if self.signal_available_time and len(self.signal_available_time) != n_times:
            raise ValueError(
                f"signal_available_time length {len(self.signal_available_time)} must match "
                f"label time length {n_times}"
            )
        signal_available = (
            self.signal_available_time if self.signal_available_time else self.decision_time
        )
        try:
            for i in range(n_times):
                decision_i = self.decision_time[i]
                signal_i = signal_available[i]
                if decision_i > signal_i:
                    raise ValueError(
                        f"decision_time must be <= signal_available_time at position {i} "
                        f"(got {decision_i!r} > {signal_i!r})"
                    )
                if self.execution_time:
                    execution_i = self.execution_time[i]
                    if signal_i > execution_i:
                        raise ValueError(
                            f"signal_available_time must be <= execution_time at position {i} "
                            f"(got {signal_i!r} > {execution_i!r})"
                        )
                else:
                    execution_i = signal_i
                if execution_i > self.label_start_time[i]:
                    raise ValueError(
                        f"execution_time must be <= label_start_time at position {i} "
                        f"(got {execution_i!r} > {self.label_start_time[i]!r})"
                    )
                if self.label_start_time[i] >= self.label_end_time[i]:
                    raise ValueError(
                        f"label_start_time must be < label_end_time at position {i} "
                        f"(got {self.label_start_time[i]!r} >= {self.label_end_time[i]!r})"
                    )
        except TypeError as exc:
            raise ValueError("timing fields must be orderable") from exc

        if self.validity is not None and self.validity.shape != self.values.shape:
            raise ValueError(
                f"Validity shape {self.validity.shape} must match values shape {self.values.shape}"
            )

    def num_observations(self) -> int:
        return len(self.values) if hasattr(self.values, "__len__") else 1
