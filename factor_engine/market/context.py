"""MarketContext — every compile/run must carry an explicit market.

The market is never inferred from a symbol (``if ".SH" in symbol`` is forbidden).
Callers pass ``ASHARE_CONTEXT`` or ``US_CONTEXT`` (or a derived context) so the
field provider, financial-period adapter, universe policy, and capability gate
all resolve against the same market contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore[assignment]

Market = Literal["ashare", "us"]

_MARKETS = ("ashare", "us")
_CURRENCIES = {"ashare": "CNY", "us": "USD"}
_TIMEZONES = {"ashare": "Asia/Shanghai", "us": "America/New_York"}
_CALENDARS = {"ashare": "SSE_SZSE", "us": "US_EQUITY"}
_SESSIONS = {"ashare": "ASHARE_CONTINUOUS", "us": "US_REGULAR"}


@dataclass(frozen=True)
class MarketContext:
    """The market contract under which fields resolve and operators run.

    Parameters
    ----------
    market :
        ``"ashare"`` or ``"us"``.  Never inferred from symbols.
    currency :
        Reporting currency (``CNY`` / ``USD``).
    timezone :
        Local market timezone (``Asia/Shanghai`` / ``America/New_York``).
    calendar_id :
        Trading-calendar identifier used by the runtime.
    session_id :
        Session identifier (A-share continuous 09:31-11:30/13:01-15:00,
        US regular 09:30-16:00 Eastern with DST/early-close awareness).
    decision_timestamp :
        The absolute decision instant.  Used for cross-market PIT reasoning:
        a signal at TradeDate T is only usable once the local market has closed.
    strict_pit :
        Reject effective-time-only / sparse-asof / proxy fallbacks by default.
    universe_policy :
        ``"default"`` — universe construction policy.
    provider_profile :
        ``"production"`` (fail-closed) or ``"research"`` (lenient, warns).
    allow_proxy :
        Permit ``PROXY_RESEARCH`` providers (research only).
    allow_sparse :
        Permit sparse providers / sparse as-of backfill (research only).
    allow_effective_time_only :
        Permit effective-time (e.g. ex-date) semantics without announcement PIT.
    """

    market: str
    currency: str
    timezone: str
    calendar_id: str
    session_id: str
    decision_timestamp: Any | None = None
    strict_pit: bool = True
    universe_policy: str = "default"
    provider_profile: str = "production"
    allow_proxy: bool = False
    allow_sparse: bool = False
    allow_effective_time_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.market not in _MARKETS:
            raise ValueError(
                f"unknown market {self.market!r}; expected one of {_MARKETS}"
            )
        if self.provider_profile not in ("production", "research"):
            raise ValueError(
                f"provider_profile must be 'production' or 'research', got {self.provider_profile!r}"
            )

    def with_profile(self, profile: str) -> "MarketContext":
        """Return a copy with a different provider profile (production/research)."""
        return replace(self, provider_profile=profile)

    def as_research(self) -> "MarketContext":
        """Production -> research convenience (opens proxy/sparse gates)."""
        return replace(
            self,
            provider_profile="research",
            allow_proxy=True,
            allow_sparse=True,
        )

    @property
    def trading_days_per_year(self) -> int:
        # Both markets trade ~242-252 days/year.  A/US annualization factors are
        # kept here (not hardcoded sqrt(252) inside operators) per the market
        # contract; exact value comes from the calendar when available.
        return 252

    @property
    def annualization_factor(self) -> float:
        return float(self.trading_days_per_year)

    def to_dict(self) -> dict[str, Any]:
        result = dict(
            market=self.market,
            currency=self.currency,
            timezone=self.timezone,
            calendar_id=self.calendar_id,
            session_id=self.session_id,
            strict_pit=self.strict_pit,
            universe_policy=self.universe_policy,
            provider_profile=self.provider_profile,
            allow_proxy=self.allow_proxy,
            allow_sparse=self.allow_sparse,
            allow_effective_time_only=self.allow_effective_time_only,
        )
        if self.decision_timestamp is not None:
            result["decision_timestamp"] = str(self.decision_timestamp)
        return result


ASHARE_CONTEXT = MarketContext(
    market="ashare",
    currency=_CURRENCIES["ashare"],
    timezone=_TIMEZONES["ashare"],
    calendar_id=_CALENDARS["ashare"],
    session_id=_SESSIONS["ashare"],
    strict_pit=True,
)

US_CONTEXT = MarketContext(
    market="us",
    currency=_CURRENCIES["us"],
    timezone=_TIMEZONES["us"],
    calendar_id=_CALENDARS["us"],
    session_id=_SESSIONS["us"],
    strict_pit=True,
)

_CONTEXT_BY_MARKET = {"ashare": ASHARE_CONTEXT, "us": US_CONTEXT}


def market_context(market: str) -> MarketContext:
    """Return the canonical ``MarketContext`` for ``market`` (ashare/us)."""
    ctx = _CONTEXT_BY_MARKET.get(str(market).strip().lower())
    if ctx is None:
        raise KeyError(f"no MarketContext for {market!r}; expected ashare|us")
    return ctx


def known_markets() -> tuple[str, ...]:
    return _MARKETS


__all__ = [
    "ASHARE_CONTEXT",
    "Market",
    "MarketContext",
    "US_CONTEXT",
    "known_markets",
    "market_context",
]
