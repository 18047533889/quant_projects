"""Instrument identity must be namespaced by market.

``instrument = "A"`` is not globally unique: ``ashare:000001.SZ`` (Ping An Bank)
and ``us:AAPL`` are different instruments.  Every cache / factor-lake / lineage
key uses ``(market, instrument, trade_date)`` so same-named tickers never collide.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class InstrumentKey:
    """A market-namespaced instrument identity."""

    market: str
    instrument: str

    def __post_init__(self) -> None:
        if not isinstance(self.market, str) or not self.market.strip():
            raise ValueError(f"market must be a non-empty string, got {self.market!r}")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise ValueError(
                f"instrument must be a non-empty string, got {self.instrument!r}"
            )
        object.__setattr__(self, "market", self.market.strip().lower())
        object.__setattr__(self, "instrument", self.instrument.strip())

    def __str__(self) -> str:
        return f"{self.market}:{self.instrument}"

    @classmethod
    def parse(cls, qualified: str) -> "InstrumentKey":
        if ":" not in str(qualified):
            raise ValueError(
                f"instrument key must be 'market:instrument', got {qualified!r}"
            )
        market, instrument = str(qualified).split(":", 1)
        return cls(market=market, instrument=instrument)

    def as_tuple(self) -> tuple[str, str]:
        return (self.market, self.instrument)


def qualified_instrument(market: str, instrument: str) -> str:
    """Return the canonical ``market:instrument`` spelling."""
    return str(InstrumentKey(market=market, instrument=instrument))


__all__ = ["InstrumentKey", "qualified_instrument"]
