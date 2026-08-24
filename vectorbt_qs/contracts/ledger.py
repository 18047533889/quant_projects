"""Cash-ledger artifacts for the vectorbt_qs backtest contract layer.

``CashLedgerArtifact`` is the valuation spine of a backtest: for each trading
day it records cash, per-asset shares, and the resulting NAV.  The Accounting
Invariant Gate consumes this ledger to verify
``cash_t + sum(shares_t * price_t) == NAV_t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CashLedgerRow:
    """One day's cash, positions, and NAV."""

    timestamp: str
    cash: float
    shares: List[float]  # per-asset, aligned with asset_ids
    nav: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cash": self.cash,
            "shares": list(self.shares),
            "nav": self.nav,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CashLedgerRow":
        return cls(
            timestamp=payload["timestamp"],
            cash=float(payload["cash"]),
            shares=list(payload["shares"]),
            nav=float(payload["nav"]),
        )


@dataclass(frozen=True)
class CashLedgerArtifact:
    """Frozen daily cash / shares / NAV series for one backtest.

    Attributes
    ----------
    timestamps:
        Trading-day timestamps (ISO).
    cash:
        Daily cash balance.
    shares:
        ``[n_days, n_assets]`` share balances.
    asset_ids:
        Asset ordering matching the trailing share axis.
    nav:
        Daily net asset value.
    mark_prices:
        Optional per-day mark price used for NAV; ``None`` if shares already
        carry price.  When present it must equal NAV/cash math for the gate.
    content_hash:
        SHA-256 over the canonical payload.
    """

    timestamps: tuple
    cash: tuple
    shares: tuple
    asset_ids: tuple
    nav: tuple
    mark_prices: Optional[tuple] = None
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if not n:
            raise ValueError("ledger timestamps cannot be empty")
        if len(self.cash) != n:
            raise ValueError("cash must align with timestamps")
        if len(self.shares) != n:
            raise ValueError("shares must align with timestamps")
        if len(self.nav) != n:
            raise ValueError("nav must align with timestamps")
        for row in self.shares:
            if len(row) != len(self.asset_ids):
                raise ValueError(
                    "each shares row must match asset_ids width "
                    f"({len(row)} != {len(self.asset_ids)})"
                )
        if self.mark_prices is not None and len(self.mark_prices) != n:
            raise ValueError("mark_prices must align with timestamps")

    def recompute_hash(self) -> str:
        from .signal import content_hash_of

        payload = {
            "timestamps": list(self.timestamps),
            "cash": list(self.cash),
            "shares": [list(r) for r in self.shares],
            "asset_ids": list(self.asset_ids),
            "nav": list(self.nav),
            "mark_prices": None
            if self.mark_prices is None
            else [list(r) for r in self.mark_prices],
        }
        return content_hash_of(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamps": list(self.timestamps),
            "cash": list(self.cash),
            "shares": [list(r) for r in self.shares],
            "asset_ids": list(self.asset_ids),
            "nav": list(self.nav),
            "mark_prices": None
            if self.mark_prices is None
            else [list(r) for r in self.mark_prices],
            "content_hash": self.content_hash or self.recompute_hash(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CashLedgerArtifact":
        obj = cls(
            timestamps=tuple(payload["timestamps"]),
            cash=tuple(float(x) for x in payload["cash"]),
            shares=tuple(tuple(float(x) for x in r) for r in payload["shares"]),
            asset_ids=tuple(payload["asset_ids"]),
            nav=tuple(float(x) for x in payload["nav"]),
            mark_prices=None
            if payload.get("mark_prices") is None
            else tuple(tuple(float(x) for x in r) for r in payload["mark_prices"]),
            content_hash=payload.get("content_hash"),
        )
        expected = obj.recompute_hash()
        supplied = payload.get("content_hash")
        if supplied and supplied != expected:
            raise ValueError(
                f"content_hash mismatch for CashLedgerArtifact: "
                f"supplied {supplied} != recomputed {expected}"
            )
        return obj


__all__ = ["CashLedgerArtifact", "CashLedgerRow"]
