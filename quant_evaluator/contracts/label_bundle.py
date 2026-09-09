"""
Label bundle contract for evaluation.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple
import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.metric_artifacts import FrozenMapping, _freeze_array
from quant_evaluator.contracts._hashutil import stable_content_hex


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
    asset_axis: Optional[AxisRef] = None
    observation_time: Tuple[Any, ...] = field(default_factory=tuple)
    schema_version: str = "0.2"
    content_hash: str = field(init=False, default="")

    def __post_init__(self):
        for name in ("decision_time", "execution_time", "signal_available_time",
                     "label_start_time", "label_end_time", "observation_time"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
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
        if self.asset_axis is not None:
            if not isinstance(self.asset_axis, AxisRef):
                raise ValueError("asset_axis must be an AxisRef")
            if self.values.ndim != 2 or self.asset_axis.size != self.values.shape[1]:
                raise ValueError("asset_axis must match the label asset dimension")
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
        #   observation_time <= signal_available_time <= decision_time
        #   decision_time <= execution_time (where execution_time supplied)
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
        if self.observation_time and len(self.observation_time) != n_times:
            raise ValueError("observation_time must match label time length")
        try:
            for i in range(n_times):
                decision_i = self.decision_time[i]
                signal_i = signal_available[i]
                if signal_i > decision_i:
                    raise ValueError(
                        f"signal_available_time must be <= decision_time at position {i} "
                        f"(got {signal_i!r} > {decision_i!r})"
                    )
                if self.observation_time and self.observation_time[i] > signal_i:
                    raise ValueError("observation_time must be <= signal_available_time")
                if self.execution_time:
                    execution_i = self.execution_time[i]
                    if decision_i > execution_i:
                        raise ValueError(
                            f"decision_time must be <= execution_time at position {i} "
                            f"(got {decision_i!r} > {execution_i!r})"
                        )
                else:
                    execution_i = decision_i
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
        if self.validity is not None and self.validity.dtype != np.bool_:
            raise ValueError("Label validity must have boolean dtype")
        object.__setattr__(self, "values", _freeze_array(self.values, "label values"))
        if self.validity is not None:
            object.__setattr__(self, "validity", _freeze_array(self.validity, "label validity"))
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata))
        identity = {name: getattr(self, name) for name in (
            "target_id", "values", "validity", "horizon", "execution_delay",
            "decision_time", "signal_available_time", "execution_time",
            "label_start_time", "label_end_time", "observation_time",
            "source_ref", "calendar_ref", "price_convention", "metadata", "schema_version")}
        identity["asset_coordinates"] = (
            tuple(self.asset_axis.values.tolist())
            if self.asset_axis is not None and self.asset_axis.values is not None else None)
        object.__setattr__(self, "content_hash", stable_content_hex(tag="LabelBundle.v2", fields=identity))

    def slice(self, time_slice: slice, asset_slice: slice = slice(None)) -> "LabelBundle":
        """Slice data and all semantic coordinates together; never re-default fields."""
        values = self.values[time_slice, asset_slice] if self.values.ndim == 2 else self.values[time_slice]
        validity = None if self.validity is None else (
            self.validity[time_slice, asset_slice] if self.validity.ndim == 2 else self.validity[time_slice])
        axis = self.asset_axis
        if axis is not None:
            coordinates = None if axis.values is None else axis.values[asset_slice]
            axis = replace(axis, size=values.shape[1], values=coordinates)
        timing = {name: getattr(self, name)[time_slice] for name in (
            "decision_time", "execution_time", "signal_available_time",
            "label_start_time", "label_end_time", "observation_time")}
        return replace(self, values=values, validity=validity, asset_axis=axis, **timing)

    def num_observations(self) -> int:
        return len(self.values) if hasattr(self.values, "__len__") else 1
