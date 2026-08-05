"""Deterministic generator for the FactorEngine-native cold-start catalogs."""
from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .model import ColdStartFactor, ensure_unique, formula_hash, formula_tree_key

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent

HORIZONS = (3, 5, 10, 20, 40, 60, 120, 252)
SHORT_HORIZONS = (3, 5, 10, 20)
MEDIUM_HORIZONS = (20, 40, 60)
LONG_HORIZONS = (60, 120, 252)

MARKET_FIELDS: dict[str, dict[str, set[str]]] = {
    "ashare": {
        "core": {"open", "high", "low", "close", "pre_close", "volume", "amount", "ret", "vwap", "factor"},
        "enriched": {"turnover_ratio", "market_cap", "circulating_cap", "circulating_market_cap", "free_market_cap", "exchange", "average_volume"},
    },
    "us": {
        "core": {"open", "high", "low", "close", "pre_close", "volume", "amount", "ret", "vwap", "adj_factor"},
        "derived": {"ret__intra", "ret__overnight", "high__low__ratio", "upper__shadow__ratio", "vwap__close__dist"},
        "enriched": {"turnover_ratio", "market_cap", "exchange", "average_volume", "avg_daily_volume", "short_volume_ratio", "short_interest", "days_to_cover"},
        "microstructure": {"bid_price", "ask_price", "bid_size", "ask_size", "total_volume"},
    },
}

EXCLUDED_OPERATOR_REASONS: dict[str, str] = {
    "period_average": "fundamental fiscal-period operator; excluded from price-volume cold start",
    "period_cagr": "fundamental fiscal-period operator; excluded from price-volume cold start",
    "period_change": "fundamental fiscal-period operator; excluded from price-volume cold start",
    "period_lag": "fundamental fiscal-period operator; excluded from price-volume cold start",
    "period_stability": "fundamental fiscal-period operator; excluded from price-volume cold start",
    "quarter_from_cumulative": "fundamental statement transformation",
    "ttm_from_cumulative": "fundamental statement transformation",
    "ttm_from_quarterly": "fundamental statement transformation",
    "yoy_by_period": "fundamental statement transformation",
    "fundamental_staleness": "fundamental availability diagnostic",
    "revision_delta": "fundamental revision diagnostic",
    "arg": "complex-valued phase is not meaningful for real daily price-volume inputs",
    "cot": "singular trigonometric transform; unstable and economically unmotivated",
    "csc": "singular trigonometric transform; unstable and economically unmotivated",
    "sec": "singular trigonometric transform; unstable and economically unmotivated",
    "tan": "unbounded periodic transform; unstable around singularities",
    "cosh": "explosive transform; unsuitable for unbounded production signals",
    "sinh": "explosive transform; unsuitable for unbounded production signals",
}


def _expr_key(formula: str) -> str:
    return formula_tree_key(formula)


def _strip_dsl(path: Path) -> str:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return " ".join(lines)


def existing_formula_hashes(repo_root: Path = ROOT) -> set[str]:
    """Return structural formula hashes of published GTJA/Week2 packs.

    The committed snapshot keeps generation deterministic when this package is
    installed without the sibling historical factor projects.
    """
    hashes: set[str] = set()
    snapshot = PACKAGE_ROOT / "source" / "existing_pack_formula_hashes.json"
    if snapshot.is_file():
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        hashes.update(str(value) for value in data.get("formula_hashes", []))
    for path in (repo_root / "gtja191" / "formulas").glob("*.dsl"):
        formula = _strip_dsl(path)
        if formula:
            try:
                hashes.add(formula_hash(formula))
            except SyntaxError:
                pass
    week2 = repo_root / "week2_pv_factors" / "source" / "week2_factors_catalog.json"
    if week2.is_file():
        for row in json.loads(week2.read_text(encoding="utf-8")).get("factors", []):
            formula = str(row.get("formula") or "").strip()
            if formula:
                try:
                    hashes.add(formula_hash(formula))
                except SyntaxError:
                    pass
    return hashes


def existing_formula_keys(repo_root: Path = ROOT) -> set[str]:
    """Compatibility helper returning live structural keys when source packs exist."""
    keys: set[str] = set()
    for path in (repo_root / "gtja191" / "formulas").glob("*.dsl"):
        formula = _strip_dsl(path)
        if formula:
            try:
                keys.add(_expr_key(formula))
            except SyntaxError:
                pass
    week2 = repo_root / "week2_pv_factors" / "source" / "week2_factors_catalog.json"
    if week2.is_file():
        for row in json.loads(week2.read_text(encoding="utf-8")).get("factors", []):
            formula = str(row.get("formula") or "").strip()
            if formula:
                try:
                    keys.add(_expr_key(formula))
                except SyntaxError:
                    pass
    return keys



class Builder:
    def __init__(self, market: str, surface: str, *, excluded_keys: set[str]) -> None:
        self.market = market
        self.surface = surface
        self.excluded_keys = set(excluded_keys)
        self.rows: list[ColdStartFactor] = []
        self.keys: set[str] = set()

    def add(
        self,
        formula: str,
        *,
        family: str,
        subfamily: str,
        horizon: int | None,
        complexity: str = "moderate",
        availability_tier: str = "core",
        rationale: str,
        direction_hint: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        formula = formula.strip()
        key = formula_hash(formula)
        if key in self.excluded_keys or key in self.keys:
            return
        seq = len(self.rows) + 1
        row = ColdStartFactor(
            factor_id=f"{self.market}_{self.surface}_{seq:04d}",
            market=self.market,
            surface=self.surface,
            formula=formula,
            family=family,
            subfamily=subfamily,
            horizon=horizon,
            complexity=complexity,
            availability_tier=availability_tier,
            rationale=rationale,
            direction_hint=direction_hint,
            metadata={"causal": True, "frequency": "1d", **(metadata or {})},
        )
        self.rows.append(row)
        self.keys.add(key)

    def add_wrapped(
        self,
        base: str,
        *,
        family: str,
        subfamily: str,
        horizon: int | None,
        availability_tier: str = "core",
        rationale: str,
        direction_hint: str = "unknown",
        wrappers: tuple[str, ...] = ("rank", "zscore", "mad"),
        complexity: str = "moderate",
    ) -> None:
        variants: list[tuple[str, str]] = [(base, "raw")]
        if "rank" in wrappers:
            variants.append((f"rank({base})", "rank"))
        if "pct_rank" in wrappers:
            variants.append((f"cs_pct_rank({base})", "pct_rank"))
        if "zscore" in wrappers:
            variants.append((f"zscore({base})", "zscore"))
        if "mad" in wrappers:
            variants.append((f"cs_mad_zscore({base})", "mad_zscore"))
        if "winsor" in wrappers:
            variants.append((f"zscore(winsorize({base}, 0.01, 0.99))", "winsor_zscore"))
        if "normalize" in wrappers:
            variants.append((f"normalize({base})", "normalize"))
        for formula, suffix in variants:
            self.add(
                formula,
                family=family,
                subfamily=f"{subfamily}_{suffix}",
                horizon=horizon,
                complexity=complexity if suffix == "raw" else "composite",
                availability_tier=availability_tier,
                rationale=rationale,
                direction_hint=direction_hint,
            )


def _base_daily(builder: Builder) -> None:
    ret1 = "coalesce(ret, subtract(safe_div_null(close, pre_close), 1.0))"
    intraday = "subtract(safe_div_null(close, open), 1.0)"
    gap = "subtract(safe_div_null(open, pre_close), 1.0)"
    range_pct = "safe_div_null(subtract(high, low), pre_close)"
    body_pct = "safe_div_null(subtract(close, open), pre_close)"
    close_loc = "safe_div_null(subtract(close, low), subtract(high, low))"
    vwap_gap = "subtract(safe_div_null(close, vwap), 1.0)"

    # Return, momentum and reversal across heterogeneous horizons.
    for h in HORIZONS:
        builder.add_wrapped(
            f"ts_pct(close, {h})",
            family="return_momentum",
            subfamily="simple_return",
            horizon=h,
            rationale=f"{h}-day price momentum captures persistent repricing.",
            direction_hint="positive",
            wrappers=("rank", "zscore", "mad", "winsor" if h in MEDIUM_HORIZONS else "rank"),
        )
        builder.add_wrapped(
            f"ts_log_return(close, {h})",
            family="return_momentum",
            subfamily="log_return",
            horizon=h,
            rationale=f"{h}-day log return provides an additive momentum representation.",
            direction_hint="positive",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_sum({ret1}, {h})",
            family="return_momentum",
            subfamily="summed_return",
            horizon=h,
            rationale=f"Cumulative normalized daily return over {h} sessions.",
            direction_hint="positive",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"neg(ts_zscore(close, {h}))",
            family="reversal",
            subfamily="price_z_reversal",
            horizon=h,
            rationale=f"Mean reversion after an extreme {h}-day price deviation.",
            direction_hint="positive",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            f"subtract(safe_div_null(close, ts_mean(close, {h})), 1.0)",
            family="trend_location",
            subfamily="moving_average_distance",
            horizon=h,
            rationale=f"Distance from the {h}-day moving average measures trend extension.",
            direction_hint="positive",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"safe_div_null(subtract(close, ts_min(close, {h})), subtract(ts_max(close, {h}), ts_min(close, {h})))",
            family="trend_location",
            subfamily="range_position",
            horizon=h,
            rationale=f"Position inside the trailing {h}-day high-low channel.",
            direction_hint="positive",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"ts_sharpe({ret1}, {h}, 252.0)",
            family="trend_quality",
            subfamily="return_sharpe",
            horizon=h,
            rationale=f"Risk-adjusted return persistence over {h} sessions.",
            direction_hint="positive",
            wrappers=("rank", "mad"),
        )

    # Daily candle, gap and VWAP structure.
    structures = [
        (intraday, "intraday_return", "Close-to-open return captures same-day pressure."),
        (gap, "overnight_gap", "Open-to-previous-close gap captures overnight repricing."),
        (range_pct, "normalized_range", "Daily range normalized by previous close."),
        (body_pct, "normalized_body", "Candle body normalized by previous close."),
        (close_loc, "close_location", "Close location within the daily range captures auction pressure."),
        (vwap_gap, "vwap_distance", "Close versus VWAP measures late-session directional pressure."),
        (f"safe_div_null(subtract(high, maximum(open, close)), subtract(high, low))", "upper_shadow", "Upper shadow ratio measures intraday rejection."),
        (f"safe_div_null(subtract(minimum(open, close), low), subtract(high, low))", "lower_shadow", "Lower shadow ratio measures intraday support."),
    ]
    for base, sub, rationale in structures:
        builder.add_wrapped(base, family="candle_vwap", subfamily=sub, horizon=1, rationale=rationale, wrappers=("rank", "zscore", "mad", "winsor"))

    # Volatility, range and downside asymmetry.
    for h in HORIZONS:
        builder.add_wrapped(
            f"ts_std({ret1}, {h})",
            family="volatility",
            subfamily="return_volatility",
            horizon=h,
            rationale=f"Realized return volatility over {h} sessions.",
            direction_hint="negative",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_mean({range_pct}, {h})",
            family="volatility",
            subfamily="average_range",
            horizon=h,
            rationale=f"Average normalized trading range over {h} sessions.",
            direction_hint="negative",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"safe_div_null(ts_std({ret1}, {h}), ts_mean(abs({ret1}), {h}))",
            family="volatility",
            subfamily="volatility_to_abs_return",
            horizon=h,
            rationale=f"Volatility relative to average absolute movement over {h} days.",
            direction_hint="negative",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            f"ts_autocorr({ret1}, {h}, 1)",
            family="serial_dependence",
            subfamily="return_autocorrelation",
            horizon=h,
            rationale=f"First-order return serial dependence over {h} sessions.",
            wrappers=("rank", "zscore"),
        )

    # Volume, amount and liquidity activity.
    for field in ("volume", "amount"):
        for h in HORIZONS:
            builder.add_wrapped(
                f"subtract(safe_div_null({field}, ts_mean({field}, {h})), 1.0)",
                family="liquidity_activity",
                subfamily=f"relative_{field}",
                horizon=h,
                rationale=f"Current {field} relative to its {h}-day baseline.",
                wrappers=("rank", "zscore", "mad"),
            )
            builder.add_wrapped(
                f"ts_zscore(log(add({field}, 1.0)), {h})",
                family="liquidity_activity",
                subfamily=f"log_{field}_zscore",
                horizon=h,
                rationale=f"Log-transformed {field} surprise over {h} sessions.",
                wrappers=("rank", "mad"),
            )
    for h in HORIZONS:
        builder.add_wrapped(
            f"safe_div_null(ts_mean(abs({ret1}), {h}), ts_mean(amount, {h}))",
            family="liquidity_impact",
            subfamily="amihud_average",
            horizon=h,
            rationale=f"Average absolute return per unit traded amount over {h} sessions.",
            direction_hint="negative",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"safe_div_null(abs(ts_sum({ret1}, {h})), ts_sum(amount, {h}))",
            family="liquidity_impact",
            subfamily="cumulative_price_impact",
            horizon=h,
            rationale=f"Cumulative price movement per unit amount over {h} sessions.",
            direction_hint="negative",
            wrappers=("rank", "mad"),
        )

    # Price-volume interaction and lead/lag structure.
    vol_change = "ts_pct(add(volume, 1.0), 1)"
    amount_change = "ts_pct(add(amount, 1.0), 1)"
    for h in HORIZONS:
        for other, label in ((vol_change, "volume"), (amount_change, "amount"), (range_pct, "range"), (intraday, "intraday")):
            builder.add_wrapped(
                f"ts_corr({ret1}, {other}, {h})",
                family="price_volume_interaction",
                subfamily=f"return_{label}_correlation",
                horizon=h,
                rationale=f"Rolling dependence between return and {label} over {h} sessions.",
                wrappers=("rank", "zscore"),
            )
        builder.add_wrapped(
            f"ts_beta({ret1}, {vol_change}, {h})",
            family="price_volume_interaction",
            subfamily="return_volume_beta",
            horizon=h,
            rationale=f"Return sensitivity to volume changes over {h} sessions.",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            f"ts_cov({ret1}, {vol_change}, {h})",
            family="price_volume_interaction",
            subfamily="return_volume_covariance",
            horizon=h,
            rationale=f"Return-volume covariance over {h} sessions.",
            wrappers=("rank", "zscore"),
        )

    # Conditional regimes, boolean primitives and robust nonlinear transforms.
    for h in MEDIUM_HORIZONS:
        vol_z = f"ts_zscore(log(add(volume, 1.0)), {h})"
        ret_z = f"ts_zscore({ret1}, {h})"
        builder.add_wrapped(
            f"where(and_(gt({vol_z}, 0.0), gt({ret1}, 0.0)), {ret1}, 0.0)",
            family="conditional_regime",
            subfamily="positive_return_high_volume",
            horizon=h,
            rationale="Positive return retained only under above-normal volume.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"where(and_(gt({vol_z}, 0.0), lt({ret1}, 0.0)), neg({ret1}), 0.0)",
            family="conditional_regime",
            subfamily="selloff_high_volume",
            horizon=h,
            rationale="Downside pressure amplified when volume is above normal.",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            f"tanh(clip({ret_z}, -3.0, 3.0))",
            family="nonlinear_robust",
            subfamily="bounded_return_zscore",
            horizon=h,
            rationale="Bounded nonlinear mapping limits outlier leverage while retaining sign.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"signed_log({ret_z})",
            family="nonlinear_robust",
            subfamily="signed_log_return_zscore",
            horizon=h,
            rationale="Signed logarithm compresses extreme standardized returns.",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            f"signed_sqrt({ret_z})",
            family="nonlinear_robust",
            subfamily="signed_sqrt_return_zscore",
            horizon=h,
            rationale="Signed square root stabilizes heavy-tailed return signals.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"exp(neg(abs({ret_z})))",
            family="nonlinear_robust",
            subfamily="calm_regime_score",
            horizon=h,
            rationale="Exponential decay scores observations close to the local return regime.",
            wrappers=("rank",),
        )

    # Cross-sectional breadth and market-state interactions use daily primitives.
    up = f"gt({ret1}, 0.0)"
    breadth = f"safe_div_null(cs_sum({up}), cs_count({ret1}))"
    dispersion = f"cs_std({ret1})"
    builder.add_wrapped(
        f"multiply(subtract({breadth}, 0.5), {ret1})",
        family="cross_sectional_state",
        subfamily="breadth_conditioned_return",
        horizon=1,
        rationale="Stock return conditioned on contemporaneous market breadth.",
        wrappers=("rank", "zscore"),
    )
    builder.add_wrapped(
        f"safe_div_null(subtract({ret1}, cs_mean({ret1})), {dispersion})",
        family="cross_sectional_state",
        subfamily="relative_return_dispersion",
        horizon=1,
        rationale="Return standardized by same-day cross-sectional dispersion.",
        wrappers=("rank", "mad"),
    )
    builder.add(
        f"where(is_finite({vwap_gap}), {vwap_gap}, fillna_const({vwap_gap}, 0.0))",
        family="data_quality",
        subfamily="finite_vwap_distance",
        horizon=1,
        complexity="composite",
        rationale="Explicit finite-value guard for VWAP distance.",
    )
    builder.add(
        f"add(is_null(vwap), is_infinite({vwap_gap}))",
        family="data_quality",
        subfamily="vwap_invalid_indicator",
        horizon=1,
        complexity="basic",
        rationale="Missing or infinite VWAP-derived input indicator.",
        direction_hint="negative",
    )
    builder.add(
        f"where(or_(eq(volume, 0.0), not_(is_not_null(volume))), 1.0, 0.0)",
        family="data_quality",
        subfamily="invalid_volume_indicator",
        horizon=1,
        complexity="composite",
        rationale="Explicit zero or missing volume indicator.",
        direction_hint="negative",
    )
    builder.add(
        f"subtract(ceil(clip({close_loc}, 0.0, 1.0)), floor(clip({close_loc}, 0.0, 1.0)))",
        family="discrete_regime",
        subfamily="non_boundary_close_location",
        horizon=1,
        complexity="composite",
        rationale="Discretized close-location regime using strict bounded arithmetic.",
    )
    builder.add(
        f"inverse(add(abs({ret1}), 0.0001))",
        family="risk_scaling",
        subfamily="inverse_absolute_return",
        horizon=1,
        complexity="basic",
        rationale="Inverse realized movement as a local risk-scaling signal.",
    )
    builder.add(
        f"power(add(abs({ret1}), 0.0001), -0.5)",
        family="risk_scaling",
        subfamily="inverse_sqrt_absolute_return",
        horizon=1,
        complexity="basic",
        rationale="Inverse square-root movement provides less aggressive risk scaling.",
    )

    # Explicitly cover the remaining production-safe daily primitives with
    # economically interpretable price-volume constructions.
    completion = [
        (f"safe_div_null(cs_demean({ret1}), cs_mad({ret1}))", "cross_sectional_state", "demeaned_return_over_mad", 1, "Return demeaned cross-sectionally and scaled by robust dispersion."),
        (f"cs_pct_rank(ts_pct(close, 20))", "cross_sectional_state", "momentum_percentile_rank", 20, "Cross-sectional percentile rank of twenty-day momentum."),
        (f"normalize(ts_pct(close, 20))", "cross_sectional_state", "momentum_minmax_normalized", 20, "Twenty-day momentum normalized to the cross-sectional [0,1] range."),
        (f"divide(ts_delta(close, 5), ts_delay(close, 5))", "return_momentum", "five_day_delta_over_lagged_price", 5, "Five-day price change scaled by the lagged price."),
        (f"log_abs(ts_zscore({ret1}, 20))", "nonlinear_robust", "log_absolute_return_zscore", 20, "Log absolute standardized return compresses heavy tails."),
        (f"multiply(sign({ret1}), sqrt(abs({ret1})))", "nonlinear_robust", "signed_sqrt_return_explicit", 1, "Explicit sign times square-root magnitude transform."),
        (f"subtract(close, ts_median(close, 20))", "reversal", "price_minus_rolling_median", 20, "Distance from the rolling median captures robust trend extension."),
        (f"ts_rank({ret1}, 20)", "trend_location", "return_time_series_rank", 20, "Current return rank within its own twenty-day history."),
        (f"ts_var({ret1}, 20)", "volatility", "return_variance", 20, "Twenty-day realized return variance."),
        (f"where(and_(ge({ret1}, 0.0), le(abs({ret1}), ts_std({ret1}, 20))), {ret1}, 0.0)", "conditional_regime", "moderate_positive_return", 20, "Positive returns retained only when movement is not beyond local volatility."),
        (f"where(ne(volume, 0.0), safe_div_null(amount, volume), 0.0)", "liquidity_activity", "nonzero_volume_average_price", 1, "Average traded price guarded by an explicit nonzero-volume condition."),
    ]
    for formula, family, subfamily, horizon, rationale in completion:
        builder.add_wrapped(
            formula,
            family=family,
            subfamily=subfamily,
            horizon=horizon,
            rationale=rationale,
            wrappers=("rank", "zscore", "mad"),
        )


def _market_daily(builder: Builder) -> None:
    market = builder.market
    if market == "ashare":
        adjusted = "multiply(close, factor)"
        builder.add_wrapped(
            "ts_pct(factor, 1)",
            family="adjustment_event",
            subfamily="factor_change",
            horizon=1,
            rationale="Vendor adjustment-factor changes identify corporate-action dates.",
            wrappers=("rank", "mad"),
        )
        for h in HORIZONS:
            builder.add_wrapped(
                f"ts_pct({adjusted}, {h})",
                family="adjusted_price",
                subfamily="adjusted_momentum",
                horizon=h,
                rationale=f"Corporate-action-adjusted A-share momentum over {h} sessions.",
                wrappers=("rank", "zscore", "mad"),
            )
    else:
        adjusted = "multiply(close, adj_factor)"
        builder.add_wrapped(
            "ts_pct(adj_factor, 1)",
            family="adjustment_event",
            subfamily="adj_factor_change",
            horizon=1,
            rationale="US adjustment-factor changes identify splits and distributions.",
            wrappers=("rank", "mad"),
        )
        for h in HORIZONS:
            builder.add_wrapped(
                f"ts_pct({adjusted}, {h})",
                family="adjusted_price",
                subfamily="adjusted_momentum",
                horizon=h,
                rationale=f"Corporate-action-adjusted US momentum over {h} sessions.",
                wrappers=("rank", "zscore", "mad"),
            )
        derived = [
            ("ret__intra", "intraday_return", "US close-to-open return supplied by DataAccess."),
            ("ret__overnight", "overnight_return", "US open-to-previous-close return supplied by DataAccess."),
            ("high__low__ratio", "high_low_ratio", "Vendor-derived high-low ratio."),
            ("upper__shadow__ratio", "upper_shadow_ratio", "Vendor-derived upper-shadow ratio."),
            ("vwap__close__dist", "vwap_close_distance", "Vendor-derived VWAP-close distance."),
        ]
        for field, sub, rationale in derived:
            builder.add_wrapped(
                field,
                family="us_derived",
                subfamily=sub,
                horizon=1,
                availability_tier="derived",
                rationale=rationale,
                wrappers=("rank", "zscore", "mad", "winsor"),
            )
            for h in MEDIUM_HORIZONS:
                builder.add_wrapped(
                    f"ts_zscore({field}, {h})",
                    family="us_derived",
                    subfamily=f"{sub}_timeseries",
                    horizon=h,
                    availability_tier="derived",
                    rationale=f"Standardized {sub.replace('_', ' ')} over {h} sessions.",
                    wrappers=("rank", "mad"),
                )
        for h in HORIZONS:
            builder.add_wrapped(
                f"subtract(ts_sum(ret__overnight, {h}), ts_sum(ret__intra, {h}))",
                family="overnight_intraday",
                subfamily="overnight_minus_intraday",
                horizon=h,
                availability_tier="derived",
                rationale=f"Decomposes {h}-day return into overnight versus regular-session pressure.",
                wrappers=("rank", "zscore", "mad"),
            )
            builder.add_wrapped(
                f"ts_corr(ret__overnight, ret__intra, {h})",
                family="overnight_intraday",
                subfamily="overnight_intraday_correlation",
                horizon=h,
                availability_tier="derived",
                rationale=f"Interaction between overnight and intraday returns over {h} sessions.",
                wrappers=("rank", "zscore"),
            )
        builder.add(
            "abs(subtract(ret, subtract(multiply(add(1.0, ret__overnight), add(1.0, ret__intra)), 1.0)))",
            family="data_consistency",
            subfamily="return_decomposition_error",
            horizon=1,
            availability_tier="derived",
            complexity="composite",
            rationale="Absolute mismatch between total return and overnight/intraday compounding identity.",
            direction_hint="negative",
        )

    # Optional market-cap, turnover and exchange-aware factors.
    for h in MEDIUM_HORIZONS:
        builder.add_wrapped(
            f"ts_zscore(log(add(turnover_ratio, 0.000001)), {h})",
            family="turnover",
            subfamily="turnover_surprise",
            horizon=h,
            availability_tier="enriched",
            rationale=f"Turnover surprise relative to a {h}-day baseline.",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            f"safe_div_null(ts_sum(coalesce(ret, 0.0), {h}), ts_mean(turnover_ratio, {h}))",
            family="turnover",
            subfamily="return_per_turnover",
            horizon=h,
            availability_tier="enriched",
            rationale=f"Cumulative return scaled by average turnover over {h} sessions.",
            wrappers=("rank", "zscore"),
        )
    size = "log(add(market_cap, 1.0))"
    builder.add(
        f"group_rank(ts_pct(close, 20), exchange)",
        family="group_relative",
        subfamily="exchange_momentum_rank",
        horizon=20,
        availability_tier="enriched",
        complexity="moderate",
        rationale="Twenty-day momentum ranked within listing exchange.",
    )
    builder.add(
        f"group_zscore(ts_pct(close, 20), exchange)",
        family="group_relative",
        subfamily="exchange_momentum_zscore",
        horizon=20,
        availability_tier="enriched",
        complexity="moderate",
        rationale="Twenty-day momentum standardized within listing exchange.",
    )
    builder.add(
        f"group_neutralize(ts_pct(close, 20), exchange)",
        family="group_relative",
        subfamily="exchange_neutral_momentum",
        horizon=20,
        availability_tier="enriched",
        complexity="moderate",
        rationale="Exchange-neutral twenty-day momentum.",
    )
    builder.add(
        f"group_normalize(ts_std(ret, 20), exchange)",
        family="group_relative",
        subfamily="exchange_relative_volatility",
        horizon=20,
        availability_tier="enriched",
        complexity="moderate",
        rationale="Volatility normalized to [0,1] within listing exchange.",
    )
    builder.add(
        f"group_winsorize(safe_div_null(ret, add(abs({size}), 1.0)), exchange, 0.05)",
        family="group_relative",
        subfamily="exchange_winsorized_size_scaled_return",
        horizon=1,
        availability_tier="enriched",
        complexity="composite",
        rationale="Size-scaled return winsorized within exchange.",
    )
    builder.add(
        f"safe_div_null(subtract(ts_pct(close, 20), group_mean(ts_pct(close, 20), exchange)), group_std(ts_pct(close, 20), exchange))",
        family="group_relative",
        subfamily="exchange_relative_momentum",
        horizon=20,
        availability_tier="enriched",
        complexity="composite",
        rationale="Momentum relative to exchange-group mean and dispersion.",
    )
    builder.add(
        f"safe_div_null(subtract(market_cap, group_min(market_cap, exchange)), subtract(group_max(market_cap, exchange), group_min(market_cap, exchange)))",
        family="group_relative",
        subfamily="exchange_size_location",
        horizon=None,
        availability_tier="enriched",
        complexity="composite",
        rationale="Market-cap location inside the listing-exchange range.",
    )
    builder.add(
        "safe_div_null(group_sum(gt(ret, 0.0), exchange), group_count(ret, exchange))",
        family="group_relative",
        subfamily="exchange_breadth",
        horizon=1,
        availability_tier="enriched",
        complexity="composite",
        rationale="Positive-return breadth within listing exchange.",
    )

    if market == "us":
        for h in MEDIUM_HORIZONS:
            builder.add_wrapped(
                f"ts_zscore(short_volume_ratio, {h})",
                family="short_flow",
                subfamily="short_volume_surprise",
                horizon=h,
                availability_tier="enriched",
                rationale=f"Short-volume ratio surprise over {h} sessions.",
                direction_hint="negative",
                wrappers=("rank", "zscore", "mad"),
            )
            builder.add_wrapped(
                f"ts_corr(ret, short_volume_ratio, {h})",
                family="short_flow",
                subfamily="return_short_volume_correlation",
                horizon=h,
                availability_tier="enriched",
                rationale=f"Return interaction with short selling over {h} sessions.",
                wrappers=("rank", "zscore"),
            )
        builder.add_wrapped(
            "safe_div_null(short_interest, average_volume)",
            family="short_flow",
            subfamily="short_interest_to_volume",
            horizon=None,
            availability_tier="enriched",
            rationale="Short interest scaled by average volume approximates crowding and cover time.",
            direction_hint="negative",
            wrappers=("rank", "mad"),
        )
        builder.add_wrapped(
            "days_to_cover",
            family="short_flow",
            subfamily="days_to_cover",
            horizon=None,
            availability_tier="enriched",
            rationale="Reported days-to-cover measure of short-position liquidity risk.",
            direction_hint="negative",
            wrappers=("rank", "zscore", "mad"),
        )
        spread = "safe_div_null(subtract(ask_price, bid_price), multiply(0.5, add(ask_price, bid_price)))"
        imbalance = "safe_div_null(subtract(bid_size, ask_size), add(bid_size, ask_size))"
        builder.add_wrapped(
            spread,
            family="microstructure",
            subfamily="relative_bid_ask_spread",
            horizon=1,
            availability_tier="microstructure",
            rationale="Relative quoted spread measures immediate transaction cost and liquidity.",
            direction_hint="negative",
            wrappers=("rank", "zscore", "mad", "winsor"),
        )
        builder.add_wrapped(
            imbalance,
            family="microstructure",
            subfamily="quoted_size_imbalance",
            horizon=1,
            availability_tier="microstructure",
            rationale="Bid versus ask size imbalance measures displayed order-book pressure.",
            wrappers=("rank", "zscore", "mad"),
        )
        for h in SHORT_HORIZONS:
            builder.add_wrapped(
                f"ts_mean({spread}, {h})",
                family="microstructure",
                subfamily="average_relative_spread",
                horizon=h,
                availability_tier="microstructure",
                rationale=f"Average quoted spread over {h} sessions.",
                direction_hint="negative",
                wrappers=("rank", "zscore"),
            )
            builder.add_wrapped(
                f"ts_mean({imbalance}, {h})",
                family="microstructure",
                subfamily="average_size_imbalance",
                horizon=h,
                availability_tier="microstructure",
                rationale=f"Persistent displayed order imbalance over {h} sessions.",
                wrappers=("rank", "zscore"),
            )


def _extended(builder: Builder) -> None:
    ret1 = "coalesce(ret, subtract(safe_div_null(close, pre_close), 1.0))"
    range_pct = "safe_div_null(subtract(high, low), pre_close)"
    vol_change = "ts_pct(add(volume, 1.0), 1)"

    # Explicit ts_average_volume for collision-free operator naming
    builder.add_wrapped(
        "ts_average_volume(volume,20)",
        family="liquidity_activity",
        subfamily="average_volume_explicit",
        horizon=20,
        rationale="Trailing 20-day average volume using collision-free operator name.",
        wrappers=("rank", "zscore", "mad"),
    )

    # Technical indicators and Wilder stateful signals.
    for h in (7, 14, 21, 28):
        builder.add_wrapped(
            f"RSI_WILDER(close, {h})",
            family="technical",
            subfamily="rsi_wilder",
            horizon=h,
            rationale=f"Wilder RSI over {h} sessions captures bounded trend strength.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"safe_div_null(ATR_WILDER(high, low, close, {h}), close)",
            family="technical",
            subfamily="atr_wilder_normalized",
            horizon=h,
            rationale=f"Wilder ATR normalized by price over {h} sessions.",
            direction_hint="negative",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ADX(high, low, close, {h})",
            family="technical",
            subfamily="adx",
            horizon=h,
            rationale=f"ADX over {h} sessions measures trend strength independent of direction.",
            wrappers=("rank", "zscore"),
        )
    for fast, slow, signal in ((6, 19, 5), (12, 26, 9), (20, 50, 9)):
        builder.add_wrapped(
            f"safe_div_null(MACD_line(close, {fast}, {slow}), close)",
            family="technical",
            subfamily="macd_line_normalized",
            horizon=slow,
            rationale=f"MACD line ({fast},{slow}) normalized by price.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"safe_div_null(MACD_signal(close, {fast}, {slow}, {signal}), close)",
            family="technical",
            subfamily="macd_signal_normalized",
            horizon=slow,
            rationale=f"MACD signal ({fast},{slow},{signal}) normalized by price.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"safe_div_null(MACD_hist(close, {fast}, {slow}, {signal}), close)",
            family="technical",
            subfamily="macd_hist_normalized",
            horizon=slow,
            rationale=f"MACD histogram ({fast},{slow},{signal}) normalized by price.",
            wrappers=("rank", "zscore", "mad"),
        )
    builder.add_wrapped(
        "safe_div_null(true_range(high, low, close), close)",
        family="technical",
        subfamily="true_range_normalized",
        horizon=1,
        rationale="True range captures overnight gaps in addition to intraday range.",
        direction_hint="negative",
        wrappers=("rank", "zscore", "mad"),
    )

    # Distribution shape, tails and extreme observations.
    for h in HORIZONS:
        for op, sub, rationale in (
            (f"ts_skew({ret1}, {h})", "return_skew", "Return asymmetry"),
            (f"ts_kurt({ret1}, {h})", "return_kurtosis", "Return tail thickness"),
            (f"ts_mad({ret1}, {h})", "return_mad", "Mean absolute return deviation"),
            (f"ts_max_drawdown(close, {h}, {max(3, h // 2)})", "max_drawdown", "Trailing maximum drawdown"),
            (f"ts_quantile({ret1}, {h}, 0.1)", "lower_return_quantile", "Lower return quantile"),
            (f"ts_quantile({ret1}, {h}, 0.9)", "upper_return_quantile", "Upper return quantile"),
            (f"ts_tail_mean({ret1}, {h}, 0.2, \"lower\", {max(3, h // 2)})", "lower_tail_mean", "Average lower-tail return"),
            (f"ts_tail_mean({ret1}, {h}, 0.2, \"upper\", {max(3, h // 2)})", "upper_tail_mean", "Average upper-tail return"),
        ):
            builder.add_wrapped(
                op,
                family="distribution_tail",
                subfamily=sub,
                horizon=h,
                rationale=f"{rationale} over {h} sessions.",
                wrappers=("rank", "zscore", "mad"),
            )
        k = max(2, min(5, h // 5))
        mp = max(k, h // 2)
        for op, sub in (
            (f"ts_topk_mean({ret1}, {h}, {k}, {mp})", "topk_return_mean"),
            (f"ts_bottomk_mean({ret1}, {h}, {k}, {mp})", "bottomk_return_mean"),
            (f"ts_topk_std({ret1}, {h}, {k}, {mp})", "topk_return_std"),
            (f"ts_bottomk_std({ret1}, {h}, {k}, {mp})", "bottomk_return_std"),
            (f"ts_topk_sum({ret1}, {h}, {k})", "topk_return_sum"),
            (f"ts_bottomk_sum({ret1}, {h}, {k}, {mp})", "bottomk_return_sum"),
        ):
            builder.add_wrapped(
                op,
                family="distribution_tail",
                subfamily=sub,
                horizon=h,
                rationale=f"Extreme-observation statistic over {h} sessions.",
                wrappers=("rank", "zscore"),
            )
        builder.add_wrapped(
            f"ts_nth_value({ret1}, {h}, {k}, \"largest\", {mp})",
            family="distribution_tail",
            subfamily="nth_largest_return",
            horizon=h,
            rationale=f"The {k}-th largest return in a {h}-day window.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"ts_nth_value({ret1}, {h}, {k}, \"smallest\", {mp})",
            family="distribution_tail",
            subfamily="nth_smallest_return",
            horizon=h,
            rationale=f"The {k}-th smallest return in a {h}-day window.",
            wrappers=("rank", "zscore"),
        )

    # Conditional, streak and time-since signals.
    for h in HORIZONS:
        mp = max(3, h // 2)
        high_volume = f"gt(volume, ts_mean(volume, {h}))"
        positive = f"gt({ret1}, 0.0)"
        negative = f"lt({ret1}, 0.0)"
        for formula, sub, rationale in (
            (f"ts_count_if({positive}, {h}, {mp})", "positive_day_count", "Number of positive-return sessions"),
            (f"ts_count_if({high_volume}, {h}, {mp})", "high_volume_day_count", "Number of above-average-volume sessions"),
            (f"ts_mean_if({ret1}, {high_volume}, {h}, {mp})", "return_on_high_volume", "Mean return on high-volume sessions"),
            (f"ts_sum_if({ret1}, {positive}, {h}, {mp})", "positive_return_sum", "Sum of positive returns"),
            (f"ts_std_if({ret1}, {negative}, {h}, {mp}, 1)", "downside_return_std", "Downside return dispersion"),
            (f"ts_last_if({ret1}, {high_volume}, {h})", "last_high_volume_return", "Most recent return observed on a high-volume day"),
            (f"ts_days_since({positive}, {h})", "days_since_positive_return", "Days since the last positive return"),
        ):
            builder.add_wrapped(
                formula,
                family="conditional_history",
                subfamily=sub,
                horizon=h,
                rationale=f"{rationale} over a {h}-day lookback.",
                wrappers=("rank", "zscore"),
            )
    builder.add_wrapped(
        f"ts_true_streak(gt({ret1}, 0.0))",
        family="conditional_history",
        subfamily="positive_return_streak",
        horizon=None,
        rationale="Current consecutive positive-return streak length.",
        wrappers=("rank", "zscore"),
    )
    builder.add_wrapped(
        f"ts_true_streak(gt(volume, ts_mean(volume, 20)))",
        family="conditional_history",
        subfamily="high_volume_streak",
        horizon=20,
        rationale="Current consecutive above-average-volume streak length.",
        wrappers=("rank", "zscore"),
    )

    # Decay, EWM and trend diagnostics.
    for h in HORIZONS:
        builder.add_wrapped(
            f"ts_decay_linear({ret1}, {h})",
            family="decay_trend",
            subfamily="linear_decay_return",
            horizon=h,
            rationale=f"Linearly decayed return over {h} sessions emphasizes recent information.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_decay_exp_window({ret1}, {h}, 0.2)",
            family="decay_trend",
            subfamily="exponential_window_return",
            horizon=h,
            rationale=f"Exponentially weighted return over a bounded {h}-day window.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"ts_ema({ret1}, {h})",
            family="decay_trend",
            subfamily="ema_return",
            horizon=h,
            rationale=f"Exponentially weighted return level with span {h}.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_time_slope(log(close), {h})",
            family="decay_trend",
            subfamily="log_price_time_slope",
            horizon=h,
            rationale=f"OLS slope of log price against time over {h} sessions.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_trend_tstat(log(close), {h}, {max(3, h // 2)})",
            family="decay_trend",
            subfamily="log_price_trend_tstat",
            horizon=h,
            rationale=f"Statistical strength of the log-price trend over {h} sessions.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"safe_div_null(ts_argmax(close, {h}), {float(h)})",
            family="decay_trend",
            subfamily="days_since_high_scaled",
            horizon=h,
            rationale=f"Recency of the trailing {h}-day high, scaled by window length.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"safe_div_null(ts_argmin(close, {h}), {float(h)})",
            family="decay_trend",
            subfamily="days_since_low_scaled",
            horizon=h,
            rationale=f"Recency of the trailing {h}-day low, scaled by window length.",
            wrappers=("rank", "zscore"),
        )

    # Rolling regression and conditional dependence.
    for h in MEDIUM_HORIZONS + LONG_HORIZONS:
        mp = max(10, h // 2)
        for x, label in ((vol_change, "volume_change"), (range_pct, "range"), ("ts_pct(add(amount, 1.0), 1)", "amount_change")):
            for op, sub in (
                ("ts_regression_slope", "slope"),
                ("ts_regression_r2", "r2"),
                ("ts_regression_tstat", "tstat"),
                ("ts_regression_resid", "residual"),
                ("ts_regression_intercept", "intercept"),
            ):
                builder.add_wrapped(
                    f"{op}({ret1}, {x}, {h}, {mp}, True)",
                    family="rolling_regression",
                    subfamily=f"return_{label}_{sub}",
                    horizon=h,
                    rationale=f"Rolling return versus {label} regression {sub} over {h} sessions.",
                    wrappers=("rank", "zscore"),
                )
        builder.add_wrapped(
            f"ts_partial_corr({ret1}, {vol_change}, {range_pct}, {h}, {mp})",
            family="rolling_regression",
            subfamily="return_volume_partial_corr_range",
            horizon=h,
            rationale=f"Return-volume dependence controlling for range over {h} sessions.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"ts_ewm_corr({ret1}, {vol_change}, {h})",
            family="rolling_regression",
            subfamily="ewm_return_volume_corr",
            horizon=h,
            rationale=f"Exponentially weighted return-volume correlation with span {h}.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"ts_ewm_cov({ret1}, {vol_change}, {h})",
            family="rolling_regression",
            subfamily="ewm_return_volume_cov",
            horizon=h,
            rationale=f"Exponentially weighted return-volume covariance with span {h}.",
            wrappers=("rank", "zscore"),
        )
        builder.add_wrapped(
            f"ts_ewm_std({ret1}, {h})",
            family="rolling_regression",
            subfamily="ewm_return_std",
            horizon=h,
            rationale=f"Exponentially weighted return volatility with span {h}.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_ewm_var({ret1}, {h})",
            family="rolling_regression",
            subfamily="ewm_return_var",
            horizon=h,
            rationale=f"Exponentially weighted return variance with span {h}.",
            wrappers=("rank", "zscore"),
        )

    # Extended robust cross-sectional operations on optional enriched fields.
    size = "log(add(market_cap, 1.0))"
    base_signal = "ts_pct(close, 20)"
    for formula, sub, rationale in (
        (f"cs_bucket({base_signal}, 10, True)", "momentum_decile", "Cross-sectional momentum decile."),
        (f"cs_rank_gaussian({base_signal}, \"average\")", "momentum_rank_gaussian", "Gaussianized cross-sectional momentum rank."),
        (f"cs_quantile({base_signal}, 0.5)", "momentum_cross_section_median", "Cross-sectional median momentum benchmark."),
        (f"cs_weighted_mean({base_signal}, market_cap)", "cap_weighted_momentum_mean", "Market-cap-weighted cross-sectional momentum mean."),
        (f"cs_weighted_demean({base_signal}, market_cap)", "cap_weighted_demeaned_momentum", "Momentum demeaned by market-cap weights."),
        (f"cs_weighted_zscore({base_signal}, market_cap)", "cap_weighted_momentum_zscore", "Market-cap-weighted momentum z-score."),
        (f"group_weighted_mean({base_signal}, exchange, market_cap)", "exchange_cap_weighted_momentum", "Exchange-level cap-weighted momentum mean."),
        (f"group_weighted_zscore({base_signal}, exchange, market_cap)", "exchange_cap_weighted_momentum_zscore", "Momentum weighted z-score within exchange."),
        (f"group_percentile({base_signal}, exchange, 0.8)", "exchange_top_momentum_indicator", "Top momentum quintile within exchange."),
        (f"cs_resid({base_signal}, {size})", "size_neutral_momentum_residual", "Cross-sectional momentum residual after size regression."),
        (f"cs_regression({base_signal}, {size}, 0)", "size_neutral_momentum_regression", "Cross-sectional regression residual of momentum on size."),
        (f"cs_wls_resid({base_signal}, {size}, market_cap, True, 10)", "size_wls_momentum_residual", "Market-cap-weighted cross-sectional size-neutral momentum."),
        (f"cs_multi_resid({base_signal}, {size}, ts_std(ret, 20), True, 10)", "size_vol_neutral_momentum", "Momentum neutralized against size and volatility."),
        (f"cs_neutralize({base_signal}, {size}, exchange, market_cap, True, 10)", "joint_neutral_momentum", "Momentum jointly neutralized against size and exchange with cap weights."),
    ):
        builder.add(
            formula,
            family="advanced_cross_sectional",
            subfamily=sub,
            horizon=20,
            availability_tier="enriched",
            complexity="composite",
            rationale=rationale,
        )

    # Useful price-volume transforms not present on daily surface.
    for h in HORIZONS:
        builder.add_wrapped(
            f"price_spread_deviation(close, {h})",
            family="price_deviation",
            subfamily="price_spread_deviation",
            horizon=h,
            rationale=f"Close relative to its {h}-day mean using the native operator.",
            wrappers=("rank", "zscore", "mad"),
        )
        builder.add_wrapped(
            f"ts_product(add(1.0, clip({ret1}, -0.99, 10.0)), {h})",
            family="compounded_return",
            subfamily="gross_return_product",
            horizon=h,
            rationale=f"Compounded gross return over {h} sessions.",
            wrappers=("rank", "zscore"),
        )
    if builder.market == "ashare":
        builder.add_wrapped(
            "real_turnover_rate(volume, circulating_cap)",
            family="turnover",
            subfamily="real_turnover_rate",
            horizon=1,
            availability_tier="enriched",
            rationale="Volume divided by free-float shares using the native turnover operator.",
            wrappers=("rank", "zscore", "mad"),
        )

    # Stable nonlinear and discretization transforms on standardized signals.
    z = "clip(ts_zscore(ret, 20), -3.0, 3.0)"
    nonlinear = [
        (f"sigmoid({z})", "sigmoid", "Smooth bounded probability-like mapping."),
        (f"unitize({z})", "unitize", "Maps signal to [-1,1]."),
        (f"saturate(add(multiply({z}, 0.2), 0.5))", "saturate", "Clamps an affine standardized signal to [0,1]."),
        (f"signed_power({z}, 0.5)", "signed_power_half", "Sign-preserving square-root power."),
        (f"exp_neg(abs({z}))", "exp_negative_abs", "Exponential decay in absolute standardized movement."),
        (f"sqrt_abs({z})", "sqrt_abs", "Square-root magnitude transform."),
        (f"cbrt({z})", "cbrt", "Cube-root heavy-tail compression."),
        (f"log10(add(abs({z}), 1.0))", "log10_abs", "Base-10 logarithmic compression."),
        (f"log2(add(abs({z}), 1.0))", "log2_abs", "Base-2 logarithmic compression."),
        (f"square({z})", "square", "Squared standardized movement magnitude."),
        (f"atan({z})", "atan", "Bounded arctangent transform."),
        (f"asin(clip(multiply({z}, 0.3), -0.99, 0.99))", "asin", "Bounded inverse-sine transform on scaled signal."),
        (f"acos(clip(multiply({z}, 0.3), -0.99, 0.99))", "acos", "Inverse-cosine transform on a valid bounded domain."),
        (f"sin({z})", "sin", "Bounded local sinusoidal transform for standardized values."),
        (f"cos({z})", "cos", "Bounded even transform of standardized values."),
        (f"atan2({z}, add(abs(ts_zscore(volume, 20)), 0.01))", "atan2", "Joint angular representation of return and volume surprise."),
        (f"lerp(neg(abs({z})), abs({z}), sigmoid(ts_zscore(volume, 20)))", "lerp", "Volume-adaptive interpolation between negative and positive magnitude."),
        (f"round({z}, 1)", "round", "One-decimal discretization of standardized return."),
        (f"truncate({z}, 1)", "truncate", "Toward-zero discretization of standardized return."),
        (f"fix({z})", "fix", "Integer-part regime encoding."),
        (f"flex_max({z}, -1.0)", "flex_max", "Lower-bounded standardized return."),
        (f"flex_min({z}, 1.0)", "flex_min", "Upper-bounded standardized return."),
    ]
    for formula, sub, rationale in nonlinear:
        builder.add_wrapped(
            formula,
            family="nonlinear_experimental",
            subfamily=sub,
            horizon=20,
            availability_tier="core",
            rationale=rationale,
            wrappers=("rank", "zscore"),
        )

    # Missing-data-aware variants.
    builder.add_wrapped(
        "ffill_limit(vwap, 3)",
        family="data_quality",
        subfamily="limited_vwap_forward_fill",
        horizon=3,
        rationale="Limited forward fill provides an explicitly bounded missing-value recovery candidate.",
        wrappers=("rank", "zscore"),
    )
    builder.add_wrapped(
        "cs_fill_mean(ts_pct(close, 20))",
        family="data_quality",
        subfamily="cross_section_mean_fill_momentum",
        horizon=20,
        rationale="Cross-sectional mean fill for otherwise missing momentum values.",
        wrappers=("rank", "zscore"),
    )
    builder.add_wrapped(
        "cs_fill_median(ts_pct(close, 20))",
        family="data_quality",
        subfamily="cross_section_median_fill_momentum",
        horizon=20,
        rationale="Robust cross-sectional median fill for missing momentum values.",
        wrappers=("rank", "zscore"),
    )
    builder.add_wrapped(
        "winsorize_mean(ret, 0.1)",
        family="cross_sectional_state",
        subfamily="trimmed_cross_section_return_mean",
        horizon=1,
        rationale="Cross-sectional trimmed mean return as a robust market-state measure.",
        wrappers=("rank",),
    )
    builder.add_wrapped(
        "ts_ratio(add(volume, 1.0))",
        family="liquidity_activity",
        subfamily="one_day_volume_ratio",
        horizon=1,
        rationale="Current volume relative to the prior session using the native ratio operator.",
        wrappers=("rank", "zscore", "mad"),
    )
    builder.add_wrapped(
        "scale(ts_pct(close, 20), 1.0)",
        family="cross_sectional_state",
        subfamily="l1_scaled_momentum",
        horizon=20,
        rationale="Cross-sectional momentum scaled to unit absolute exposure.",
        wrappers=("rank", "zscore"),
    )
    builder.add(
        "is_nan(ts_pct(close, 20))",
        family="data_quality",
        subfamily="momentum_nan_indicator",
        horizon=20,
        complexity="basic",
        rationale="Explicit NaN indicator for the twenty-day momentum input.",
        direction_hint="negative",
    )

    if builder.market == "us":
        # Extended US-specific overnight/intraday and microstructure diagnostics.
        for h in HORIZONS:
            mp = max(3, h // 2)
            builder.add_wrapped(
                f"ts_regression_slope(ret__intra, ret__overnight, {h}, {mp}, True)",
                family="overnight_intraday",
                subfamily="intraday_on_overnight_slope",
                horizon=h,
                availability_tier="derived",
                rationale=f"Regular-session response to overnight return over {h} sessions.",
                wrappers=("rank", "zscore"),
            )
            builder.add_wrapped(
                f"ts_partial_corr(ret__overnight, ret__intra, high__low__ratio, {h}, {mp})",
                family="overnight_intraday",
                subfamily="overnight_intraday_partial_corr_range",
                horizon=h,
                availability_tier="derived",
                rationale=f"Overnight-intraday relation controlling for range over {h} sessions.",
                wrappers=("rank", "zscore"),
            )
            builder.add_wrapped(
                f"ts_tail_mean(ret__overnight, {h}, 0.2, \"lower\", {mp})",
                family="overnight_intraday",
                subfamily="overnight_lower_tail",
                horizon=h,
                availability_tier="derived",
                rationale=f"Average adverse overnight tail over {h} sessions.",
                wrappers=("rank", "zscore", "mad"),
            )


def build_catalogs(repo_root: Path = ROOT) -> dict[tuple[str, str], tuple[ColdStartFactor, ...]]:
    # Exact formula reuse across markets is intentional: A-share and US factors
    # execute against different field contracts and must remain separately selectable.
    existing = existing_formula_hashes(repo_root)
    output: dict[tuple[str, str], tuple[ColdStartFactor, ...]] = {}
    for market in ("ashare", "us"):
        market_keys = set(existing)
        daily = Builder(market, "daily", excluded_keys=market_keys)
        _base_daily(daily)
        _market_daily(daily)
        daily_rows = ensure_unique(daily.rows)
        output[(market, "daily")] = daily_rows
        market_keys.update(row.formula_hash for row in daily_rows)

        extended = Builder(market, "extended", excluded_keys=market_keys)
        _extended(extended)
        extended_rows = ensure_unique(extended.rows)
        output[(market, "extended")] = extended_rows
    return output


def write_catalogs(repo_root: Path = ROOT) -> dict[str, int]:
    catalogs = build_catalogs(repo_root)
    PACKAGE_ROOT.joinpath("catalogs").mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for (market, surface), rows in catalogs.items():
        payload = {
            "schema_version": "factor_cold_start.v1",
            "market": market,
            "surface": surface,
            "dsl_surface": "daily" if surface == "daily" else "compat",
            "factor_count": len(rows),
            "factors": [row.to_dict() for row in rows],
        }
        path = PACKAGE_ROOT / "catalogs" / f"{market}_{surface}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts[f"{market}_{surface}"] = len(rows)
    return counts


if __name__ == "__main__":
    print(json.dumps(write_catalogs(), ensure_ascii=False, indent=2))
