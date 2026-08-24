"""Execution policy contract for the vectorbt_qs backtest layer.

The execution policy names how orders are generated and filled in A-share
accurate mode.  Defaults are read from the mvp engine's ``ExecutionCosts``
(A-share fees) and the runner's ``DEFAULT_CONFIG`` (slippage, lot size).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CostParams:
    """A-share transaction-cost parameters (mirror of mvp ExecutionCosts)."""

    commission: float = 0.00025
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    minimum_commission: float = 5.0

    @classmethod
    def from_mvp(cls) -> "CostParams":
        """Read A-share defaults from the mvp engine (falls back to built-ins)."""
        try:
            from vectorbt_qs.mvp.engine.execution import ExecutionCosts

            costs = ExecutionCosts()
            return cls(
                commission=float(costs.commission),
                stamp_tax=float(costs.stamp_tax),
                transfer_fee=float(costs.transfer_fee),
                minimum_commission=float(costs.minimum_commission),
            )
        except Exception:  # pragma: no cover - fallback path
            return cls()


@dataclass(frozen=True)
class ExecutionPolicy:
    """Frozen description of how orders execute.

    Attributes
    ----------
    execution_type:
        ``"open"`` | ``"vwap"`` | ``"close"`` — the price used to fill orders.
    order_side_rules:
        Tuple of A-share execution rules applied by the simulator.
    lot_size:
        100-share board lot for A-shares.
    slippage:
        Fractional one-way slippage applied to the fill price.
    costs:
        A-share fee schedule (direction + minimum commission).
    """

    execution_type: str = "open"
    order_side_rules: tuple = ("long_only", "100_share_lot", "t_next_day_open")
    lot_size: int = 100
    slippage: float = 0.001
    costs: CostParams = field(default_factory=CostParams)

    def __post_init__(self) -> None:
        if self.execution_type not in {"open", "vwap", "close"}:
            raise ValueError(
                f"execution_type must be 'open'|'vwap'|'close', got {self.execution_type!r}"
            )
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.slippage < 0.0 or self.slippage >= 1.0:
            raise ValueError("slippage must be in [0, 1)")
        if not isinstance(self.order_side_rules, tuple):
            raise TypeError("order_side_rules must be a tuple")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_type": self.execution_type,
            "order_side_rules": list(self.order_side_rules),
            "lot_size": self.lot_size,
            "slippage": self.slippage,
            "costs": {
                "commission": self.costs.commission,
                "stamp_tax": self.costs.stamp_tax,
                "transfer_fee": self.costs.transfer_fee,
                "minimum_commission": self.costs.minimum_commission,
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExecutionPolicy":
        return cls(
            execution_type=payload.get("execution_type", "open"),
            order_side_rules=tuple(payload.get("order_side_rules", ())),
            lot_size=int(payload.get("lot_size", 100)),
            slippage=float(payload.get("slippage", 0.001)),
            costs=CostParams(
                commission=float(payload.get("costs", {}).get("commission", 0.00025)),
                stamp_tax=float(payload.get("costs", {}).get("stamp_tax", 0.0005)),
                transfer_fee=float(payload.get("costs", {}).get("transfer_fee", 0.00001)),
                minimum_commission=float(
                    payload.get("costs", {}).get("minimum_commission", 5.0)
                ),
            ),
        )


@dataclass(frozen=True)
class CapacityPolicy:
    """Optional per-order capacity limit (participation-rate capped volume)."""

    max_participation_rate: float = 1.0
    volume_ref: str = "volume"

    def __post_init__(self) -> None:
        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be in (0, 1]")


__all__ = ["ExecutionPolicy", "CostParams", "CapacityPolicy"]
