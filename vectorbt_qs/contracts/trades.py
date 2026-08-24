"""Trade artifacts for the vectorbt_qs backtest contract layer.

A trade is a round-trip: an entry order into an asset and a matching exit.
Trades aggregate orders into position-level round-trips for PnL analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TradeArtifact:
    """A completed or open round-trip in a single asset.

    Attributes
    ----------
    trade_id:
        Stable trade identifier.
    asset_id:
        Ticker.
    entry_timestamp:
        Entry execution timestamp.
    exit_timestamp:
        Exit execution timestamp (``None`` for open trades).
    entry_price:
        Volume-weighted entry fill price.
    exit_price:
        Volume-weighted exit fill price (``None`` for open trades).
    size:
        Traded quantity in shares.
    entry_fees:
        Fees paid at entry.
    exit_fees:
        Fees paid at exit.
    """

    trade_id: int
    asset_id: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    entry_fees: float = 0.0
    exit_fees: float = 0.0

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id required")
        if self.size <= 0:
            raise ValueError("trade size must be positive")

    @property
    def gross_pnl(self) -> float:
        """Gross PnL for a long round-trip."""
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.size

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.entry_fees - self.exit_fees

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "asset_id": self.asset_id,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "entry_fees": self.entry_fees,
            "exit_fees": self.exit_fees,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TradeArtifact":
        return cls(
            trade_id=payload["trade_id"],
            asset_id=payload["asset_id"],
            entry_time=payload["entry_time"],
            exit_time=payload["exit_time"],
            entry_price=float(payload["entry_price"]),
            exit_price=float(payload["exit_price"]),
            size=float(payload["size"]),
            entry_fees=float(payload.get("entry_fees", 0.0)),
            exit_fees=float(payload.get("exit_fees", 0.0)),
        )


@dataclass(frozen=True)
class TradesArtifact:
    """Hash-addressed container of trades for one backtest."""

    trades: tuple
    content_hash: Optional[str] = None

    @classmethod
    def from_trades(cls, trades: List[TradeArtifact]) -> "TradesArtifact":
        from .signal import content_hash_of

        payload = {"trades": [t.to_dict() for t in trades]}
        return cls(tuple(trades), content_hash_of(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {"trades": [t.to_dict() for t in self.trades]}


__all__ = ["TradeArtifact", "TradesArtifact"]
