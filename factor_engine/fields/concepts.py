"""Canonical field concepts — the semantic layer above physical fields.

Per the multi-market plan (spec §5-§6, §10-§13): math operators must never see
``Return`` vs ``Ret`` vs ``CNY`` vs ``USD``.  They operate on canonical concepts
such as ``return_decimal``, ``raw_close``, ``continuous_close``,
``market_cap_local``.  Each concept is bound to a *market provider* by
``fields/providers.py``.

Legacy DSL spellings (``close``, ``ret``, ``volume``, ``roe``, ...) keep working:
the resolver treats them as aliases into the canonical concepts below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .units_v2 import (
    BOOLEAN,
    DATE,
    DIMENSIONLESS,
    IDENTIFIER,
    LOCAL_MONEY,
    LOCAL_PRICE_PER_SHARE,
    RATIO,
    SHARES,
    TEXT,
    UnitSpec,
)

# price basis vocabulary
PRICE_BASIS_RAW = "RAW"
PRICE_BASIS_CONTINUOUS = "CONTINUOUS"
PRICE_BASIS_RAW_OFFICIAL_LIMIT = "RAW_OFFICIAL_LIMIT"
PRICE_BASIS_EITHER = "EITHER"

# role vocabulary
ROLE_FEATURE = "feature"
ROLE_GROUP_KEY = "group_key"
ROLE_STATUS = "status"
ROLE_TIME = "time"
ROLE_IDENTIFIER = "identifier"


@dataclass(frozen=True)
class FieldConceptSpec:
    """One canonical market-independent concept."""

    concept_id: str
    domain: str  # price_volume | fundamental | classification | corporate_action | ...
    value_kind: str  # return | price | volume | amount | ratio | ...
    canonical_unit: UnitSpec
    frequency: str = "daily"
    grain: str = "instrument_trade_date"
    role: str = ROLE_FEATURE
    price_basis: str | None = None  # RAW / CONTINUOUS / RAW_OFFICIAL_LIMIT
    cross_market_comparable: bool = False
    market_local_only: bool = False
    allowed_operator_families: tuple[str, ...] = ()
    description: str = ""
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "domain": self.domain,
            "value_kind": self.value_kind,
            "canonical_unit": self.canonical_unit.to_dict(),
            "frequency": self.frequency,
            "grain": self.grain,
            "role": self.role,
            "price_basis": self.price_basis,
            "cross_market_comparable": self.cross_market_comparable,
            "market_local_only": self.market_local_only,
            "allowed_operator_families": list(self.allowed_operator_families),
            "aliases": list(self.aliases),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Concept registry.
# ---------------------------------------------------------------------------
_CONCEPTS: dict[str, FieldConceptSpec] = {}


def _reg(spec: FieldConceptSpec) -> FieldConceptSpec:
    if spec.concept_id in _CONCEPTS:
        raise ValueError(f"duplicate concept {spec.concept_id!r}")
    _CONCEPTS[spec.concept_id] = spec
    return spec


def _c(
    concept_id,
    domain,
    value_kind,
    unit,
    *,
    price_basis=None,
    cross_market_comparable=False,
    market_local_only=False,
    role=ROLE_FEATURE,
    aliases=(),
    description="",
):
    return _reg(
        FieldConceptSpec(
            concept_id=concept_id,
            domain=domain,
            value_kind=value_kind,
            canonical_unit=unit,
            price_basis=price_basis,
            cross_market_comparable=cross_market_comparable,
            market_local_only=market_local_only,
            role=role,
            aliases=tuple(aliases),
            description=description,
        )
    )


# --- price/volume ---------------------------------------------------------
_c("return_decimal", "price_volume", "return", RATIO,
  cross_market_comparable=True, aliases=("ret", "return", "returns"),
  description="Daily return as a decimal (0.02 == 2%). A: Return/10000; US: Ret identity.")

_c("raw_open", "price_volume", "price", LOCAL_PRICE_PER_SHARE, price_basis=PRICE_BASIS_RAW,
  aliases=("open",), cross_market_comparable=False)
_c("raw_high", "price_volume", "price", LOCAL_PRICE_PER_SHARE, price_basis=PRICE_BASIS_RAW,
  aliases=("high",), cross_market_comparable=False)
_c("raw_low", "price_volume", "price", LOCAL_PRICE_PER_SHARE, price_basis=PRICE_BASIS_RAW,
  aliases=("low",), cross_market_comparable=False)
_c("raw_close", "price_volume", "price", LOCAL_PRICE_PER_SHARE, price_basis=PRICE_BASIS_RAW,
  aliases=("close",), cross_market_comparable=False)
_c("raw_pre_close", "price_volume", "price", LOCAL_PRICE_PER_SHARE, price_basis=PRICE_BASIS_RAW,
  aliases=("pre_close", "prev_close"), cross_market_comparable=False)
_c("raw_vwap", "price_volume", "price", LOCAL_PRICE_PER_SHARE, price_basis=PRICE_BASIS_RAW,
  aliases=("vwap",), cross_market_comparable=False)

_c("continuous_open", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_CONTINUOUS, cross_market_comparable=False,
  description="Backward-adjusted open (Close*Factor family). A: Open*Factor; US: Open*AdjFactor.")
_c("continuous_high", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_CONTINUOUS, cross_market_comparable=False)
_c("continuous_low", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_CONTINUOUS, cross_market_comparable=False)
_c("continuous_close", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_CONTINUOUS, cross_market_comparable=False,
  description="Backward-adjusted close. A: Close*Factor; US: Close*AdjFactor.")
_c("continuous_vwap", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_CONTINUOUS, cross_market_comparable=False)

_c("raw_volume_shares", "price_volume", "volume", SHARES,
  aliases=("volume",), cross_market_comparable=False,
  description="Trade volume in shares. NEVER divided by any adjustment factor.")

_c("amount_local", "price_volume", "amount", LOCAL_MONEY, aliases=("amount", "turnover_value"),
  market_local_only=True, cross_market_comparable=False,
  description="Turnover amount in local currency (CNY/USD). NOT cross-market comparable as a raw number.")

# --- capital / valuation --------------------------------------------------
_c("market_cap_local", "valuation", "amount", LOCAL_MONEY, market_local_only=True,
  aliases=("market_cap", "mkt_cap"),
  description="Market cap in local currency. A: MarketCap (exact). US: Close*weighted_shares (derived, ~42% coverage).")
_c("free_float_market_cap_local", "valuation", "amount", LOCAL_MONEY, market_local_only=True,
  aliases=("free_market_cap", "float_market_cap"))
_c("total_shares", "capital", "count", SHARES, aliases=("total_capital", "total_shares"))
_c("free_float_shares", "capital", "count", SHARES, aliases=("free_cap", "free_float_shares"))
_c("turnover_ratio_decimal", "valuation", "ratio", RATIO, aliases=("turnover_ratio", "turnover"),
  cross_market_comparable=True, description="Turnover as decimal. A: TurnoverRatio/100.")

# --- price limits (A-share mechanism) -------------------------------------
_c("upper_price_limit", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_RAW_OFFICIAL_LIMIT, market_local_only=True,
  aliases=("high_limit",), description="Official daily upper limit price (raw). A-share only.")
_c("lower_price_limit", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_RAW_OFFICIAL_LIMIT, market_local_only=True,
  aliases=("low_limit",), description="Official daily lower limit price (raw). A-share only.")

# --- fundamental ratios ---------------------------------------------------
_c("roe_decimal", "fundamental", "ratio", RATIO, aliases=("roe",),
  cross_market_comparable=True, description="ROE as decimal. A: Roe/100; US: return_on_equity identity.")
_c("roa_decimal", "fundamental", "ratio", RATIO, aliases=("roa",),
  cross_market_comparable=True)
_c("gross_margin_decimal", "fundamental", "ratio", RATIO, aliases=("gross_profit_margin",),
  cross_market_comparable=True)
_c("net_profit_margin_decimal", "fundamental", "ratio", RATIO,
  aliases=("net_profit_margin",), cross_market_comparable=True)
_c("dividend_yield_decimal", "fundamental", "ratio", RATIO,
  aliases=("dividend_yield", "dividend_ratio"),
  cross_market_comparable=True, description="Dividend yield as decimal. A: DividendRatio/100; US: dividend_yield identity.")

# --- financial statements (flow / stock amounts, local currency) ----------
_c("operating_revenue", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  aliases=("revenue",), description="Revenue for the period in local currency.")
_c("net_profit", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  aliases=("net_income",))
_c("operating_cash_flow", "fundamental", "amount", LOCAL_MONEY, market_local_only=True, aliases=("ocf",))
_c("total_assets", "fundamental", "amount", LOCAL_MONEY, market_local_only=True)
_c("total_liabilities", "fundamental", "amount", LOCAL_MONEY, market_local_only=True)
_c("equity", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  aliases=("shareholders_equity", "total_equity"))
_c("earnings_per_share", "fundamental", "price", LOCAL_PRICE_PER_SHARE, aliases=("eps",))

# --- classification / status ----------------------------------------------
_c("industry_group", "classification", "group", IDENTIFIER, role=ROLE_GROUP_KEY,
  market_local_only=True, description="Industry classification. A: StockIndustry (sw_l1); US: PROVIDER_REQUIRED.")
_c("tradability_state", "classification", "status", BOOLEAN, role=ROLE_STATUS,
  market_local_only=True, description="Tradable mask. A: not IsSuspend; US: CS universe + valid price/volume.")
_c("index_member", "index", "boolean", BOOLEAN, role=ROLE_STATUS, aliases=("is_index_member",))
_c("index_weight", "index", "ratio", RATIO, role=ROLE_GROUP_KEY,
  market_local_only=True, description="Index constituent weight as decimal. A: Weight/100; US: no weight.")

# --- dividends / corporate actions ----------------------------------------
_c("cash_dividend_per_share", "corporate_action", "price", LOCAL_PRICE_PER_SHARE,
  market_local_only=True, description="Per-share cash dividend in local currency.")
_c("dividend_ex_date", "corporate_action", "date", DATE, role=ROLE_TIME)

# --- news / holders --------------------------------------------------------
_c("news_sentiment", "alternative", "ratio", RATIO, market_local_only=True,
  description="News sentiment. US-only in current data.")
_c("holder_concentration", "alternative", "ratio", RATIO, market_local_only=True,
  description="Top-holder concentration. A-share only in current data.")


def get_concept(concept_id: str) -> FieldConceptSpec | None:
    return _CONCEPTS.get(str(concept_id).strip())


def require_concept(concept_id: str) -> FieldConceptSpec:
    spec = get_concept(concept_id)
    if spec is None:
        raise KeyError(f"unknown canonical concept {concept_id!r}")
    return spec


def list_concepts() -> tuple[FieldConceptSpec, ...]:
    return tuple(sorted(_CONCEPTS.values(), key=lambda item: item.concept_id))


def concept_alias_map() -> dict[str, str]:
    """Legacy DSL spelling -> canonical concept id."""
    out: dict[str, str] = {}
    for spec in _CONCEPTS.values():
        out[spec.concept_id] = spec.concept_id
        for alias in spec.aliases:
            out.setdefault(alias, spec.concept_id)
    return out


__all__ = [
    "PRICE_BASIS_CONTINUOUS",
    "PRICE_BASIS_EITHER",
    "PRICE_BASIS_RAW",
    "PRICE_BASIS_RAW_OFFICIAL_LIMIT",
    "FieldConceptSpec",
    "concept_alias_map",
    "get_concept",
    "list_concepts",
    "require_concept",
]
