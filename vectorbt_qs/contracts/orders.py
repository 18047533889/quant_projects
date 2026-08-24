"""Order artifacts for the vectorbt_qs backtest contract layer.

Frozen, stdlib-only records describing planned and filled orders.  ``OrderArtifact``
is the canonical contract between the mvp simulators and any downstream consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OrderArtifact:
    """One order (intent plus fill) at an execution timestamp.

    Attributes
    ----------
    order_id:
        Stable order identifier.
    timestamp:
        Execution timestamp (ISO string).
    asset_id:
        Ticker.
    side:
        ``"buy"`` | ``"sell"``.
    requested_size:
        Requested quantity in shares.
    filled_size:
        Filled quantity in shares.
    execution_price:
        Fill price (real currency / real share).
    fees:
        Total fee for this order (commission + tax + transfer, incl. minimum).
    status:
        ``"filled"`` | ``"partial"`` | ``"blocked"``.
    reason:
        Free-text reason when not fully filled (e.g. ``"suspend"``).
    """

    order_id: int
    timestamp: str
    asset_id: str
    side: str
    requested_size: float
    filled_size: float
    execution_price: float
    fees: float = 0.0
    status: str = "filled"
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id required")
        if self.side not in {"buy", "sell"}:
            raise ValueError(f"side must be 'buy'|'sell', got {self.side!r}")
        if self.requested_size < 0 or self.filled_size < 0:
            raise ValueError("sizes cannot be negative")
        if self.execution_price < 0:
            raise ValueError("execution_price cannot be negative")
        if self.fees < 0:
            raise ValueError("fees cannot be negative")

    @property
    def notional(self) -> float:
        """Gross notional of the filled portion."""
        return self.filled_size * self.execution_price

    @property
    def net_cash_flow(self) -> float:
        """Signed cash movement from this order (negative on buy)."""
        gross = self.filled_size * self.execution_price
        return -gross - self.fees if self.side == "buy" else gross - self.fees

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "side": self.side,
            "requested_size": self.requested_size,
            "filled_size": self.filled_size,
            "execution_price": self.execution_price,
            "fees": self.fees,
            "status": self.status,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OrderArtifact":
        return cls(
            order_id=payload["order_id"],
            timestamp=payload["timestamp"],
            asset_id=payload["asset_id"],
            side=payload["side"],
            requested_size=float(payload["requested_size"]),
            filled_size=float(payload["filled_size"]),
            execution_price=float(payload["execution_price"]),
            fees=float(payload.get("fees", 0.0)),
            status=payload.get("status", "filled"),
            reason=payload.get("reason", ""),
        )


@dataclass(frozen=True)
class OrdersArtifact:
    """A hash-addressed container of orders for one backtest."""

    orders: tuple
    content_hash: Optional[str] = None

    @classmethod
    def from_orders(cls, orders: List[OrderArtifact]) -> "OrdersArtifact":
        from .signal import content_hash_of

        payload = {"orders": [o.to_dict() for o in orders]}
        return cls(tuple(orders), content_hash_of(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {"orders": [o.to_dict() for o in self.orders]}


__all__ = ["OrderArtifact", "OrdersArtifact"]
