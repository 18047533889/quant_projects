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
# R17-011: the official/reference pre-close (ex-dividend / corporate-action
# adjusted reference price), distinct from ``lag(raw_close, 1)`` — the exchange's
# PreClose already removes company-action jumps, so return/gap/limit operators
# must choose the correct basis instead of mixing them.
PRICE_BASIS_OFFICIAL_REFERENCE_PRE_CLOSE = "OFFICIAL_REFERENCE_PRE_CLOSE"
PRICE_BASIS_RETURN = "RETURN"
PRICE_BASIS_EITHER = "EITHER"


class PriceBasis:
    """R37-P0-014：price basis 语义 identity（不同 basis 不共 cache/CSE/checkpoint）。

    与既有字符串词汇表对齐；``canonical`` 返回规范化字符串，进入
    ``FactorSemanticIdentity.price_basis`` / ``DataKnowledgeIdentity``。
    """

    RAW = "RAW"                      # 未调整原始价格
    CONTINUOUS = "CONTINUOUS"        # 连续复权（前复权/后复权统一视为 continuous）
    FORWARD_ADJUSTED = "FORWARD_ADJUSTED"   # 前复权
    BACKWARD_ADJUSTED = "BACKWARD_ADJUSTED"  # 后复权
    TOTAL_RETURN = "TOTAL_RETURN"    # 全收益（含再投资）
    RAW_OFFICIAL_LIMIT = "RAW_OFFICIAL_LIMIT"
    OFFICIAL_REFERENCE_PRE_CLOSE = "OFFICIAL_REFERENCE_PRE_CLOSE"
    RETURN = "RETURN"
    EITHER = "EITHER"

    _KNOWN = frozenset({
        RAW, CONTINUOUS, FORWARD_ADJUSTED, BACKWARD_ADJUSTED, TOTAL_RETURN,
        RAW_OFFICIAL_LIMIT, OFFICIAL_REFERENCE_PRE_CLOSE, RETURN, EITHER,
    })

    @classmethod
    def canonical(cls, value: str | None) -> str:
        """规范化为标准值；未知值保留原样（向后兼容），但空值归一化为 RAW。"""
        if not value:
            return cls.RAW
        return value if value in cls._KNOWN else value

# reporting-flow semantics vocabulary
FLOW_SEMANTICS_STOCK = "stock"
FLOW_SEMANTICS_SINGLE_PERIOD = "single_period_flow"
FLOW_SEMANTICS_CUMULATIVE_YTD = "cumulative_ytd_flow"
FLOW_SEMANTICS_TTM = "ttm_flow"

# role vocabulary
ROLE_FEATURE = "feature"
ROLE_GROUP_KEY = "group_key"
ROLE_STATUS = "status"
ROLE_TIME = "time"
ROLE_IDENTIFIER = "identifier"
# R17-067: continuous weight (index weight, portfolio weight) — NOT a group key.
ROLE_WEIGHT = "weight"


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
    price_basis: str | None = None  # RAW / CONTINUOUS / RAW_OFFICIAL_LIMIT / RETURN
    flow_semantics: str | None = None  # stock / single_period_flow / cumulative_ytd_flow / ttm_flow
    # R17-065: ``cross_market_comparable`` was overloaded.  Unit comparability is
    # NOT accounting-definition comparability (A ROE % vs US X0 ROE have the same
    # decimal unit but differ in period/annualization/attribution).  The single
    # bool is deprecated in favour of the four explicit fields.
    cross_market_comparable: bool = False
    unit_comparable: bool | None = None          # same canonical unit/dimension
    definition_comparable: bool | None = None    # same accounting/economic definition
    provider_available_by_market: dict[str, bool] = field(default_factory=dict)
    cross_market_rank_allowed: bool = False      # only True when BOTH unit+definition
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
            "flow_semantics": self.flow_semantics,
            "cross_market_comparable": self.cross_market_comparable,
            "unit_comparable": self.unit_comparable,
            "definition_comparable": self.definition_comparable,
            "provider_available_by_market": dict(self.provider_available_by_market),
            "cross_market_rank_allowed": self.cross_market_rank_allowed,
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
    flow_semantics=None,
    cross_market_comparable=False,
    unit_comparable=None,
    definition_comparable=None,
    provider_available_by_market=None,
    cross_market_rank_allowed=None,
    market_local_only=False,
    role=ROLE_FEATURE,
    aliases=(),
    description="",
):
    # R17-065: when the new structured comparability fields are given they win;
    # cross_market_comparable (deprecated) back-fills unit_comparable only.
    eff_unit = unit_comparable if unit_comparable is not None else cross_market_comparable
    eff_definition = definition_comparable if definition_comparable is not None else cross_market_comparable
    eff_rank = (
        cross_market_rank_allowed
        if cross_market_rank_allowed is not None
        else (bool(eff_unit and eff_definition))
    )
    return _reg(
        FieldConceptSpec(
            concept_id=concept_id,
            domain=domain,
            value_kind=value_kind,
            canonical_unit=unit,
            price_basis=price_basis,
            flow_semantics=flow_semantics,
            cross_market_comparable=cross_market_comparable,
            unit_comparable=eff_unit,
            definition_comparable=eff_definition,
            provider_available_by_market=dict(provider_available_by_market or {}),
            cross_market_rank_allowed=eff_rank,
            market_local_only=market_local_only,
            role=role,
            aliases=tuple(aliases),
            description=description,
        )
    )


# --- price/volume ---------------------------------------------------------
_c("return_decimal", "price_volume", "return", RATIO,
  price_basis=PRICE_BASIS_RETURN, cross_market_comparable=True,
  aliases=("ret", "return", "returns"),
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
  aliases=("prev_close",), cross_market_comparable=False,
  description="physically lagged previous raw close (lag(raw_close,1)).  NOT the "
              "exchange's official reference pre-close — see reference_pre_close (R17-011).")
# R17-011: the exchange's official/reference pre-close (A-share PreClose already
# removes corporate-action jumps; US PreClose is the official reference).  This is
# a DIFFERENT economic concept from lag(raw_close,1): return/gap/limit operators
# that consume the official basis must reference this concept, never the lagged
# raw close.
_c("reference_pre_close", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_OFFICIAL_REFERENCE_PRE_CLOSE,
  aliases=("pre_close",), cross_market_comparable=False,
  description="official reference pre-close (company-action adjusted); distinct "
              "from lag(raw_close,1) (R17-011)")
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
# R17-065: turnover is UNIT-comparable (decimal ratio) but NOT cross-market
# rankable — US has no isomorphic D1 turnover provider, so a joint A+US rank
# would silently rank US names as constant/absent.
_c("turnover_ratio_decimal", "valuation", "ratio", RATIO, aliases=("turnover_ratio", "turnover"),
  cross_market_comparable=False, unit_comparable=True, definition_comparable=False,
  provider_available_by_market={"ashare": True, "us": False},
  cross_market_rank_allowed=False,
  description="Turnover as decimal. A: TurnoverRatio/100; US: no provider (R17-065).")

# --- price limits (A-share mechanism) -------------------------------------
_c("upper_price_limit", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_RAW_OFFICIAL_LIMIT, market_local_only=True,
  aliases=("high_limit",), description="Official daily upper limit price (raw). A-share only.")
_c("lower_price_limit", "price_volume", "price", LOCAL_PRICE_PER_SHARE,
  price_basis=PRICE_BASIS_RAW_OFFICIAL_LIMIT, market_local_only=True,
  aliases=("low_limit",), description="Official daily lower limit price (raw). A-share only.")

# --- fundamental ratios ---------------------------------------------------
# R17-066: ROE has the same DECIMAL UNIT in A/US but different accounting
# definitions (A Roe indicator often unannualized parent-equity basis; US X0
# return_on_equity sparse valuation basis).  unit_comparable but NOT
# definition_comparable -> no joint A+US cross-sectional rank without an
# explicit comparator policy.
_c("roe_decimal", "fundamental", "ratio", RATIO, aliases=("roe",),
  cross_market_comparable=False, unit_comparable=True, definition_comparable=False,
  provider_available_by_market={"ashare": True, "us": True},
  cross_market_rank_allowed=False,
  description="ROE as decimal. A: Roe/100; US: return_on_equity identity. "
              "Unit-comparable, definition-NOT-comparable (R17-066).")
_c("roa_decimal", "fundamental", "ratio", RATIO, aliases=("roa",),
  cross_market_comparable=False, unit_comparable=True, definition_comparable=False,
  provider_available_by_market={"ashare": False, "us": True},
  cross_market_rank_allowed=False)
_c("gross_margin_decimal", "fundamental", "ratio", RATIO, aliases=("gross_profit_margin",),
  cross_market_comparable=False, unit_comparable=True, definition_comparable=False,
  provider_available_by_market={"ashare": False, "us": False},
  cross_market_rank_allowed=False)
_c("net_profit_margin_decimal", "fundamental", "ratio", RATIO,
  aliases=("net_profit_margin",), cross_market_comparable=False,
  unit_comparable=True, definition_comparable=False,
  provider_available_by_market={"ashare": False, "us": False},
  cross_market_rank_allowed=False)
_c("dividend_yield_decimal", "fundamental", "ratio", RATIO,
  aliases=("dividend_yield", "dividend_ratio"),
  cross_market_comparable=False, unit_comparable=True, definition_comparable=False,
  provider_available_by_market={"ashare": True, "us": True},
  cross_market_rank_allowed=False,
  description="Dividend yield as decimal. A: DividendRatio/100; US: dividend_yield identity.")

# --- financial statements (flow / stock amounts, local currency) ----------
# The canonical statement concepts default to the A-share reporting-flow
# semantics (fiscal-year-to-date cumulative for Income/CashFlow, point-in-time
# stock for Balance); the US market binding overrides per ``timeframe``
# (quarterly/annual -> single_period_flow, trailing_twelve_months -> ttm_flow).
_c("operating_revenue", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_CUMULATIVE_YTD,
  aliases=("revenue",), description="Revenue for the period in local currency.")
# R17-078: ``net_profit`` (A NetProfit includes minority interest; US
# net_income_loss_attributable_common_shareholders) and the parent/common-attributable
# line are DIFFERENT economic concepts.  Cross-market profitability/ROE/PE-derived
# factors must choose the matching attribution basis.
_c("net_profit", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_CUMULATIVE_YTD,
  aliases=("net_income",),
  description="Net profit TOTAL (A includes minority interest).  For parent/common-"
              "attributable use net_income_attributable (R17-078).")
_c("net_income_attributable", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_CUMULATIVE_YTD,
  aliases=("net_income_parent", "parent_net_profit", "net_income_common"),
  description="Net income attributable to parent (A) / common shareholders (US). "
              "The attribution-aligned profitability basis (R17-078).")
_c("operating_cash_flow", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_CUMULATIVE_YTD, aliases=("ocf",))
_c("total_assets", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_STOCK)
_c("total_liabilities", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_STOCK)
# R17-079: equity attribution must align with the net-income attribution basis.
# A uses parent equity; US should use total_equity_attributable_to_parent when the
# numerator is parent/common net income — not total_equity (which includes
# noncontrolling interest).
_c("equity", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_STOCK,
  aliases=("shareholders_equity", "total_equity"),
  description="Equity.  A: parent-company owners' equity.  US: prefer "
              "total_equity_attributable_to_parent for attribution-aligned "
              "ROE (R17-079).")
_c("equity_attributable", "fundamental", "amount", LOCAL_MONEY, market_local_only=True,
  flow_semantics=FLOW_SEMANTICS_STOCK,
  aliases=("equity_parent", "parent_equity", "equity_common"),
  description="Equity attributable to parent (A) / common shareholders (US); the "
              "attribution-aligned denominator (R17-079).")
_c("earnings_per_share", "fundamental", "price", LOCAL_PRICE_PER_SHARE, aliases=("eps",))

# --- classification / status ----------------------------------------------
_c("industry_group", "classification", "group", IDENTIFIER, role=ROLE_GROUP_KEY,
  market_local_only=True, description="Industry classification. A: StockIndustry (sw_l1); US: PROVIDER_REQUIRED.")
_c("tradability_state", "classification", "status", BOOLEAN, role=ROLE_STATUS,
  market_local_only=True, description="Tradable mask. A: not IsSuspend; US: CS universe + valid price/volume.")
_c("index_member", "index", "boolean", BOOLEAN, role=ROLE_STATUS, aliases=("is_index_member",))
# R17-067: index_weight is a CONTINUOUS ratio, not a categorical group id — it
# must never be routed through a group-by/categorical path.  It carries the new
# ROLE_WEIGHT (a continuous weight semantic kind).
_c("index_weight", "index", "ratio", RATIO, role=ROLE_WEIGHT,
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
    "FLOW_SEMANTICS_CUMULATIVE_YTD",
    "FLOW_SEMANTICS_SINGLE_PERIOD",
    "FLOW_SEMANTICS_STOCK",
    "FLOW_SEMANTICS_TTM",
    "PRICE_BASIS_CONTINUOUS",
    "PRICE_BASIS_EITHER",
    "PRICE_BASIS_RAW",
    "PRICE_BASIS_RAW_OFFICIAL_LIMIT",
    "PRICE_BASIS_RETURN",
    "FieldConceptSpec",
    "concept_alias_map",
    "get_concept",
    "list_concepts",
    "require_concept",
]
