"""Per-operator market capability contracts (additive to OperatorMetadata).

The base ``OperatorMetadata`` stays untouched.  Market-mechanism operators
(price-limit family, industry convenience ops, holder/pledge, index weights,
news) declare a small, explicit contract here; generic math/TS/CS operators get
"both, input-dependent" by default from the capability resolver.  This keeps
the 1293-canonical manifest machine-generated instead of hand-written
(spec §119-§120).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market.capabilities import MarketCapability as _Cap

PRICE_BASIS_EITHER = "EITHER"
PRICE_BASIS_RAW = "RAW"
PRICE_BASIS_CONTINUOUS = "CONTINUOUS"
PRICE_BASIS_RAW_OFFICIAL_LIMIT = "RAW_OFFICIAL_LIMIT"


@dataclass(frozen=True)
class OperatorMarketContract:
    """Declared market/capability requirements for one canonical operator."""

    canonical: str
    intrinsic_markets: tuple[str, ...] = ("ashare", "us")
    required_capabilities: tuple[str, ...] = ()
    price_basis: str = PRICE_BASIS_EITHER
    cross_market_comparable: bool = True
    depends_on_inputs: bool = False
    # P0-035: input grain the operator needs to be meaningful — "daily" /
    # "minute" / "either" (default).  Minute-mechanism families declare "minute".
    required_grain: str = "either"
    notes: str = ""


class OperatorMarketContractRegistry:
    """Deterministic canonical -> OperatorMarketContract registry."""

    def __init__(self) -> None:
        self._contracts: dict[str, OperatorMarketContract] = {}

    def register(self, contract: OperatorMarketContract, *, replace: bool = False) -> None:
        key = contract.canonical.strip()
        if key in self._contracts and not replace:
            raise ValueError(f"market contract already registered: {key}")
        self._contracts[key] = contract

    def register_many(
        self,
        canonicals: Any,
        *,
        intrinsic_markets: tuple[str, ...] = ("ashare", "us"),
        required_capabilities: tuple[str, ...] = (),
        price_basis: str = PRICE_BASIS_EITHER,
        cross_market_comparable: bool = True,
        notes: str = "",
    ) -> int:
        count = 0
        for name in canonicals:
            if name in self._contracts:
                continue
            self.register(
                OperatorMarketContract(
                    canonical=name,
                    intrinsic_markets=tuple(intrinsic_markets),
                    required_capabilities=tuple(required_capabilities),
                    price_basis=price_basis,
                    cross_market_comparable=cross_market_comparable,
                    depends_on_inputs=not required_capabilities,
                    notes=notes,
                )
            )
            count += 1
        return count

    def get(self, canonical: str) -> OperatorMarketContract | None:
        return self._contracts.get(str(canonical).strip())

    def canonicals(self) -> tuple[str, ...]:
        return tuple(sorted(self._contracts))


OPERATOR_MARKET_CONTRACTS = OperatorMarketContractRegistry()


# ---------------------------------------------------------------------------
# Bulk declarations.
# ---------------------------------------------------------------------------

# A-share daily price-limit family (spec §29, §30).  US has no equivalent daily
# static price limit; LULD/halt is a different mechanism and must NOT be mapped
# onto high/low limit.
_PRICE_LIMIT_FAMILY = frozenset(
    {
        "ashare_days_since_limit_down", "ashare_days_since_limit_up",
        "ashare_failed_limit_count", "ashare_limit_asymmetry",
        "ashare_limit_distance", "ashare_limit_down_streak",
        "ashare_limit_down_touch", "ashare_limit_down_volume_ratio",
        "ashare_limit_event_density", "ashare_limit_failed",
        "ashare_limit_one_price", "ashare_limit_open_down_streak",
        "ashare_limit_open_failed", "ashare_limit_open_up_streak",
        "ashare_limit_touch_count", "ashare_limit_up_streak",
        "ashare_limit_up_touch", "ashare_limit_up_volume_ratio",
        "ashare_one_price_limit_streak", "ashare_open_at_upper_limit",
        "limit_down_close", "limit_up_close",
    }
)

# Industry convenience ops: they implicitly resolve ``industry_group``.
# A-share native; US = PROVIDER_REQUIRED until a GICS/SIC/NAICS provider exists.
_INDUSTRY_FAMILY = frozenset(
    {
        "industry_size_neutralize", "industry_rolling_pca_loading",
        "intra_industry_lead_lag_ex_self", "ts_industry_liquidity_beta",
        "industry_neutralize", "industry_rank", "industry_peer_mean",
        "industry_peer_median", "industry_peer_std", "industry_zscore",
        "industry_size_residual",
    }
)

# Holder / pledge family: A-share top-holder feeds.  US = PROVIDER_REQUIRED
# (no same-structure top-holder feed; do NOT substitute institutional ownership).
_HOLDER_FAMILY = frozenset(
    {
        "holder_class_entropy", "holder_class_js_shift",
        "holder_common_holding_peer_return", "holder_concentration",
        "holder_concentration_acceleration", "holder_concentration_change",
        "holder_concentration_slope", "holder_count_change_rate",
        "holder_entry_share", "holder_exit_share",
        "holder_float_concentration_gap", "holder_freeze_concentration",
        "holder_freeze_ratio", "holder_id_matched_churn",
        "holder_id_matched_entry_share", "holder_id_matched_exit_share",
        "holder_id_overlap_ratio", "holder_locked_share_ratio",
        "holder_nature_entropy", "holder_net_entry_share",
        "holder_peer_return_breadth", "holder_pledge_change",
        "holder_pledge_churn", "holder_pledge_concentration",
        "holder_pledge_ratio", "holder_pledged_holder_count",
        "holder_rank_stability", "holder_share_weighted_rank_migration",
        "holder_shareholder_network_centrality",
        "holder_shareholder_overlap_ratio", "holder_weighted_churn",
    }
)

# Index-weight ops: A-share has Weight (%) in IndexConstituent; US Components
# have NO weight (equal-weight is a different operator semantics).
_INDEX_WEIGHT_FAMILY = frozenset(
    {
        "index_weight", "index_weight_change", "index_weight_gap_to_free_float",
        "index_weighted_peer_mean", "index_weighted_neutralize",
        "index_weighted_zscore",
    }
)

# News family: US FactNews available; A-share clean has no Chinese news.
_NEWS_FAMILY = frozenset(
    {
        "news_sentiment", "news_volume", "news_event_age", "news_surprise",
        "news_sentiment_momentum", "news_coverage_ratio", "news_negative_ratio",
    }
)

# P0-016: the holder family is SPLIT.  Base holder analysis (concentration /
# entropy / rank / churn / entry-exit / class / overlap / network) only needs the
# top-holder feed.  Only pledge / freeze / locked ops additionally need the
# pledge capability.  A-share TopTen has SharePledge/ShareFreeze (in shares) and
# ShareRatio (%), but FloatTopTen usually does not — forcing every holder op to
# require pledge made ordinary holder concentration US-blocked twice over.
_HOLDER_FAMILY = frozenset(
    {
        "holder_class_entropy", "holder_class_js_shift",
        "holder_common_holding_peer_return", "holder_concentration",
        "holder_concentration_acceleration", "holder_concentration_change",
        "holder_concentration_slope", "holder_count_change_rate",
        "holder_entry_share", "holder_exit_share",
        "holder_float_concentration_gap", "holder_id_matched_churn",
        "holder_id_matched_entry_share", "holder_id_matched_exit_share",
        "holder_id_overlap_ratio", "holder_nature_entropy",
        "holder_net_entry_share", "holder_peer_return_breadth",
        "holder_rank_stability", "holder_share_weighted_rank_migration",
        "holder_shareholder_network_centrality",
        "holder_shareholder_overlap_ratio", "holder_weighted_churn",
    }
)
_HOLDER_PLEDGE_FAMILY = frozenset(
    {
        "holder_freeze_concentration", "holder_freeze_ratio",
        "holder_locked_share_ratio", "holder_pledge_change",
        "holder_pledge_churn", "holder_pledge_concentration",
        "holder_pledge_ratio", "holder_pledged_holder_count",
    }
)

# P0-015: market-mechanism families with no explicit per-op contracts defaulted
# to "both" before.  These need declared capabilities so US is not silently
# granted capabilities it does not have (FULL_MINUTE_OHLCV / DAILY_TURNOVER /
# FREE_FLOAT_SHARES are A-share-only in the current capability snapshot).
# R17-057: ``minute_`` was missing — the capability resolver itself classifies
# ``minute_`` as a mechanism prefix, so the operator-market fallback must match it
# too or minute_* ops silently fell to INPUT_DEPENDENT.
_MINUTE_PREFIXES = ("intraday_", "intra_", "session_", "micro_", "minute_")

# intra_lunch_gap_return depends on the A-share lunch-break mechanism itself.
_LUNCH_GAP_ONLY = frozenset({"intra_lunch_gap_return"})
# intraday_barrier_approach_acceleration needs minute bars AND daily price limits.
_MINUTE_WITH_LIMITS = frozenset({"intraday_barrier_approach_acceleration"})
# R17-058: minute-level limit behavior (duration / first-hit-time / reopen-count)
# cannot be derived from daily HighLimit/LowLimit alone — it needs full minute
# OHLCV on top of the daily price-limit mechanism.
_MINUTE_LIMIT_FAMILY = frozenset(
    {"intra_limit_duration", "intra_limit_first_hit_time", "intra_limit_reopen_count"}
)

_TURNOVER_FAMILY = frozenset(
    {
        "average_turnover", "abnormal_turnover", "turnover_volatility",
        "turnover_autocorr", "return_per_turnover", "turnover_shock",
        "turnover_acceleration", "price_turnover_divergence",
        "return_turnover_beta", "turnover_adjusted_volatility",
        "turnover_momentum", "turnover_zscore",
    }
)
# TurnoverSurvivalKernel chip family (u_t = free-float turnover).
_CHIP_FAMILY = frozenset(
    {
        "ts_turnover_reference_price", "ts_turnover_cost_dispersion",
        "ts_turnover_profit_share", "ts_turnover_holding_age",
        "ts_turnover_near_cost_mass", "ts_turnover_cost_quantile_distance",
        "ts_turnover_cost_entropy", "ts_turnover_cost_mode_distance",
        "ts_turnover_cost_skew", "ts_turnover_age_dispersion",
    }
)
_FREE_FLOAT_FAMILY = frozenset(
    {
        "free_float_turnover", "free_float_ratio", "free_float_share_ratio",
        "free_to_circulating_ratio", "market_cap_free_cap_gap",
        "float_share_ratio", "true_turnover_rate", "real_turnover_rate",
    }
)


def _register_default_contracts() -> None:
    OPERATOR_MARKET_CONTRACTS.register_many(
        _PRICE_LIMIT_FAMILY,
        intrinsic_markets=("ashare",),
        required_capabilities=(_Cap.DAILY_PRICE_LIMITS.value,),
        price_basis=PRICE_BASIS_RAW_OFFICIAL_LIMIT,
        cross_market_comparable=False,
        notes="A-share daily static price limit mechanism; US LULD/halt is not an equivalent",
    )
    OPERATOR_MARKET_CONTRACTS.register_many(
        _INDUSTRY_FAMILY,
        required_capabilities=(_Cap.INDUSTRY_CLASSIFICATION.value,),
        price_basis=PRICE_BASIS_EITHER,
        cross_market_comparable=True,
        notes="implicitly resolves industry_group; US requires a GICS/SIC/NAICS provider",
    )
    OPERATOR_MARKET_CONTRACTS.register_many(
        _HOLDER_FAMILY,
        required_capabilities=(_Cap.TOP_HOLDERS.value,),
        cross_market_comparable=False,
        notes="A-share top-holder feeds; US has no same-structure holder feed (a 13F/institutional provider would unblock US)",
    )
    OPERATOR_MARKET_CONTRACTS.register_many(
        _HOLDER_PLEDGE_FAMILY,
        required_capabilities=(_Cap.TOP_HOLDERS.value, _Cap.HOLDER_PLEDGE.value),
        cross_market_comparable=False,
        notes="pledge/freeze/locked holder ops additionally need the pledge capability; A-share TopTen only",
    )
    OPERATOR_MARKET_CONTRACTS.register_many(
        _INDEX_WEIGHT_FAMILY,
        required_capabilities=(_Cap.INDEX_WEIGHTS.value,),
        cross_market_comparable=False,
        notes="US index components have no Weight field",
    )
    OPERATOR_MARKET_CONTRACTS.register_many(
        _NEWS_FAMILY,
        required_capabilities=(_Cap.NEWS.value,),
        cross_market_comparable=False,
        notes="US FactNews available; A-share clean has no Chinese news feed (a CN news provider would unblock A)",
    )


def _fallback_contract(canonical: str) -> OperatorMarketContract | None:
    """Synthesized contract for prefix/name families (P0-015).

    Keeps the 1311-canonical manifest machine-generated: the minute / turnover /
    chip / free-float families get an explicit capability contract without a
    hand-maintained per-op list of every intraday name.  Explicitly-registered
    contracts always win (they are checked first by ``contract_for``).
    """
    name = str(canonical).strip()
    if name in _LUNCH_GAP_ONLY:
        return OperatorMarketContract(
            canonical=name,
            intrinsic_markets=("ashare",),
            cross_market_comparable=False,
            required_grain="minute",
            notes="intra_lunch_gap_return measures the A-share lunch-break mechanism itself; US has no lunch break",
        )
    if name in _MINUTE_WITH_LIMITS:
        return OperatorMarketContract(
            canonical=name,
            intrinsic_markets=("ashare",),
            required_capabilities=(
                _Cap.FULL_MINUTE_OHLCV.value,
                _Cap.DAILY_PRICE_LIMITS.value,
            ),
            cross_market_comparable=False,
            required_grain="minute",
            notes="minute bars + daily price limits; US has neither as a daily mechanism",
        )
    if name in _MINUTE_LIMIT_FAMILY:
        # R17-058: intra_limit_* are MINUTE-level limit behavior — daily
        # HighLimit/LowLimit alone cannot compute duration / first-hit-time /
        # reopen-count.  FULL_MINUTE_OHLCV is a hard requirement.
        return OperatorMarketContract(
            canonical=name,
            intrinsic_markets=("ashare",),
            required_capabilities=(
                _Cap.DAILY_PRICE_LIMITS.value,
                _Cap.FULL_MINUTE_OHLCV.value,
            ),
            price_basis=PRICE_BASIS_RAW_OFFICIAL_LIMIT,
            cross_market_comparable=False,
            required_grain="minute",
            notes="minute-level limit behavior needs daily limit prices AND full minute OHLCV (R17-058)",
        )
    if name in _TURNOVER_FAMILY or name in _CHIP_FAMILY:
        return OperatorMarketContract(
            canonical=name,
            required_capabilities=(_Cap.DAILY_TURNOVER.value,),
            cross_market_comparable=False,
            notes="A-share daily turnover feed; US has no isomorphic D1 turnover (do not fake it with volume)",
        )
    if name in _FREE_FLOAT_FAMILY:
        return OperatorMarketContract(
            canonical=name,
            required_capabilities=(_Cap.FREE_FLOAT_SHARES.value,),
            cross_market_comparable=False,
            notes="A-share free-float shares/turnover; US lacks an isomorphic free-float feed",
        )
    if name.startswith(_MINUTE_PREFIXES):
        return OperatorMarketContract(
            canonical=name,
            required_capabilities=(_Cap.FULL_MINUTE_OHLCV.value,),
            cross_market_comparable=False,
            required_grain="minute",
            notes="minute-microstructure family requires FULL_MINUTE_OHLCV (A supported; US provider_required)",
        )
    return None


_register_default_contracts()


def contract_for(canonical: str) -> OperatorMarketContract | None:
    explicit = OPERATOR_MARKET_CONTRACTS.get(canonical)
    if explicit is not None:
        return explicit
    return _fallback_contract(canonical)


def contract_set() -> frozenset[str]:
    return frozenset(
        set(OPERATOR_MARKET_CONTRACTS.canonicals())
        | set(_LUNCH_GAP_ONLY)
        | set(_MINUTE_WITH_LIMITS)
        | set(_MINUTE_LIMIT_FAMILY)
        | set(_TURNOVER_FAMILY)
        | set(_CHIP_FAMILY)
        | set(_FREE_FLOAT_FAMILY)
    )


__all__ = [
    "OPERATOR_MARKET_CONTRACTS",
    "OperatorMarketContract",
    "OperatorMarketContractRegistry",
    "PRICE_BASIS_CONTINUOUS",
    "PRICE_BASIS_EITHER",
    "PRICE_BASIS_RAW",
    "PRICE_BASIS_RAW_OFFICIAL_LIMIT",
    "contract_for",
    "contract_set",
]
