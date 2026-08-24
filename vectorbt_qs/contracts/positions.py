"""Position artifacts for the vectorbt_qs backtest contract layer.

A position artifact records the per-asset share balance at a valuation
timestamp.  It is the simulator's exposure ledger, distinct from the cash
ledger (cash + NAV) and from trades (round-trips).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PositionArtifact:
    """Share balance and mark for one asset at one timestamp.

    Attributes
    ----------
    timestamp:
        Valuation timestamp.
    asset_id:
        Ticker.
    shares:
        Real share balance (board-lot-aligned for A-shares).
    price:
        Mark price in real currency / real share.
    """

    timestamp: str
    asset_id: str
    shares: float
    price: float

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id required")
        if self.shares < 0:
            raise ValueError("shares cannot be negative (long-only A-share)")
        if self.price < 0:
            raise ValueError("price cannot be negative")

    @property
    def market_value(self) -> float:
        return self.shares * self.price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "shares": self.shares,
            "price": self.price,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PositionArtifact":
        return cls(
            timestamp=payload["timestamp"],
            asset_id=payload["asset_id"],
            shares=float(payload["shares"]),
            price=float(payload["price"]),
        )


@dataclass(frozen=True)
class PositionsArtifact:
    """Hash-addressed container of positions for one backtest."""

    positions: tuple
    content_hash: Optional[str] = None

    @classmethod
    def from_positions(cls, positions: List[PositionArtifact]) -> "PositionsArtifact":
        from .signal import content_hash_of

        payload = {"positions": [p.to_dict() for p in positions]}
        return cls(tuple(positions), content_hash_of(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {"positions": [p.to_dict() for p in self.positions]}


__all__ = ["PositionArtifact", "PositionsArtifact"]
