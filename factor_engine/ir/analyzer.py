"""Lower expression trees to IR and derive causal history requirements."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from factor_engine.expr.base import Expr
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.ir.nodes import IRNode
from factor_engine.ir.schema import DEFAULT_COLUMN_SCHEMA, Schema
from factor_engine.ir.types import (
    MIXED,
    OPERATOR_INPUT_TYPE_CONTRACTS,
    SemanticLattice,
    axis_effect_contract_for,
    check_axis_effect_declared,
    lattice_join_semantic_attrs,
    semantic_type_of,
)

_LAG_PARAM_NAMES = {
    "ts_delay": ("n", "d", "lag", "periods", "window"),
    "prev": (),
    "ts_delta": ("n", "d", "lag", "periods", "window"),
    "ts_pct": ("d", "n", "lag", "periods", "window"),
    "ts_log_return": ("d", "n", "lag", "periods", "window"),
    "ts_ratio": (),
    "MOM": ("window", "d", "n"),
    "ROC": ("window", "d", "n"),
}

# review #4 R4-03: canonical-authoritative history requirements.  A canonical
# registers its own ``history_requirement(params) -> int`` instead of the
# analyzer guessing from parameter NAMES, because a composite lookback is almost
# never ``max(...)`` — ``rolling_corr(x, shift(y, lag), window)`` needs
# ``(window-1) + lag`` rows, and ``volume_autocorr(volume, window, lag)`` needs
# ``window-1+lag``.  The analyzer prefers this registry, then a
# ``history_requirement`` method on the implementation, then its own rules.
_HISTORY_REQUIREMENT_FUNCS: dict[str, Any] = {}


def register_history_requirement(canonical: str, fn: Any) -> None:
    """Register a deterministic ``fn(params: dict) -> int`` for a canonical."""
    _HISTORY_REQUIREMENT_FUNCS[canonical] = fn


class HistoryRequirementError(ValueError):
    """A registered ``history_requirement(params)`` failed to evaluate.

    P1-13: a history-requirement evaluation error is a PLANNING error — it must
    not silently downgrade to 0 (which would under-allocate causal lookback and
    silently corrupt the factor).  Only an explicit "not applicable" history
    function returning 0 may yield zero lookback.
    """


_FIXED_LAGS = {
    "prev": 1,
    "ts_ratio": 1,
    "candle_gap": 1,
    "candle_gap_pct": 1,
    "cdl_engulfing": 1,
    "cdl_inside_bar": 1,
    "cdl_outside_bar": 1,
    "candle_overlap_ratio": 1,
    "candle_inside_ratio": 1,
    "fin_revision_delta": 1,
    "fin_revision_pct": 1,
    "fin_revision_direction": 1,
    "fin_expectation_revision": 1,
    "fin_expectation_revision_pct": 1,
}
_WINDOW_PARAM_NAMES = (
    "window",
    "d",
    "span",
    "period",
    "periods",
    "lookback",
    "max_lookback",
    "fast",
    "slow",
    "fast_period",
    "slow_period",
    "signal_span",
    "signal_period",
    "fast_window",
    "slow_window",
    "signal_window",
    "short_window",
    "medium_window",
    "long_window",
    "ema_window",
    "atr_window",
    "tenkan_window",
    "kijun_window",
    "senkou_b_window",
    "er_window",
    "vol_window",
    "baseline_window",
    "price_window",
    "volume_window",
    "turnover_window",
    "adl_window",
    "impulse_window",
    "flag_window",
    "cup_window",
    "handle_window",
    "pennant_window",
    "max_wait",
    "max_periods",
    "window_days",
    "max_days",
    "body_window",
    "shadow_window",
)
_WINDOW_PLUS_ONE_CANONICALS = frozenset({
    "ts_prev_high",
    "ts_prev_low",
    "ts_distance_to_high",
    "ts_distance_to_low",
    "ts_breakout_high",
    "ts_breakdown_low",
    "ts_new_high",
    "ts_new_low",
    "ts_channel_position",
    "ts_days_since_high",
    "ts_days_since_low",
    "ts_range_expansion",
    "donchian_upper",
    "donchian_lower",
    "donchian_mid",
    "donchian_position",
    "relative_volume",
    "volume_zscore",
    "dollar_volume_zscore",
    "volume_momentum",
    "turnover_momentum",
    "turnover_zscore",
    "rolling_obv",
    "rolling_pvt",
    "MFI",
    "choppiness_index",
    "yang_zhang_vol",
    "overnight_volatility",
    "candle_range_atr",
    "abnormal_volume",
    "abnormal_turnover",
    "volume_shock",
    "turnover_shock",
    "candle_body_zscore",
    "candle_range_zscore",
    "candle_upper_shadow_zscore",
    "candle_lower_shadow_zscore",
    "candle_body_percentile",
    "candle_range_percentile",
    "candle_gap_atr",
})
_PIVOT_CONFIRM_CANONICALS = frozenset({
    "ts_confirmed_pivot_high",
    "ts_confirmed_pivot_low",
})
_BOUNDED_STRUCTURE_CANONICALS = frozenset({
    "ts_last_pivot_high",
    "ts_last_pivot_low",
    "ts_pivot_high_age",
    "ts_pivot_low_age",
    "ts_resistance_level",
    "ts_support_level",
    "ts_resistance_slope",
    "ts_support_slope",
    "ts_distance_to_resistance",
    "ts_distance_to_support",
    "ts_resistance_break",
    "ts_support_break",
})
_STRUCTURE_PREFIXES = (
    "ts_nth_pivot_",
    "ts_pivot_",
    "ts_swing_",
    "ts_channel_",
    "ts_line_",
    "ts_resistance_fit_",
    "ts_support_fit_",
    "ts_pattern_",
)
_STRUCTURE_PATTERNS = frozenset({
    "pattern_double_top",
    "pattern_double_bottom",
    "pattern_triple_top",
    "pattern_triple_bottom",
    "pattern_head_shoulders",
    "pattern_inverse_head_shoulders",
    "pattern_123_bull",
    "pattern_123_bear",
    "pattern_sym_triangle",
    "pattern_ascending_triangle",
    "pattern_descending_triangle",
    "pattern_rising_wedge",
    "pattern_falling_wedge",
    "pattern_rectangle",
    "pattern_rising_channel",
    "pattern_falling_channel",
    "pattern_broadening",
})
_FLAG_PATTERNS = frozenset({"pattern_bull_flag", "pattern_bear_flag"})
_PENNANT_PATTERNS = frozenset({"pattern_bull_pennant", "pattern_bear_pennant"})
_RETEST_PATTERNS = frozenset({"pattern_breakout_retest", "pattern_breakdown_retest"})
# review #4 R4-89: structural-level operators confirm a pivot `confirmation`
# bars past the bar, so the causal history is window-1 + confirmation (the
# pivot at the window's left edge must already be confirmed).
_STRUCTURAL_LEVEL_CANONICALS = frozenset({
    "ts_structural_level_density",
    "ts_nearest_structural_level_distance",
    "ts_structural_level_strength",
})

# Report-period operators need a conservative conversion from report count to
# pre-start trading rows. Daily-window and elementwise fundamental operators do
# not use this budget.
_FIN_REPORT_PERIOD_CANONICALS = frozenset({
    "fin_lag",
    "fin_diff",
    "fin_pct_change",
    "fin_log_change",
    "fin_qoq",
    "fin_yoy",
    "fin_ttm",
    "fin_ttm_quarterly",
    "fin_quarter_from_cumulative",
    "fin_ttm_cumulative",
    "fin_average_balance",
    "fin_growth",
    "fin_cagr",
    "fin_growth_acceleration",
    "fin_growth_change",
    "fin_growth_volatility",
    "fin_growth_stability",
    "fin_growth_persistence",
    "fin_std",
    "fin_mad",
    "fin_cv",
    "fin_stability",
    "fin_range",
    "fin_zscore_history",
    "fin_percentile_history",
    "fin_trend_slope",
    "fin_trend_r2",
    "fin_trend_tstat",
    "fin_trend_acceleration",
    "fin_monotonicity",
    "fin_positive_streak",
    "fin_negative_streak",
    "fin_sign_change_count",
    "fin_turnover",
    "fin_divergence",
    "fin_working_capital_change",
    "fin_beat_streak",
    "fin_miss_streak",
})
_FIN_DAILY_WINDOW_CANONICALS = frozenset({
    "fin_revision_count",
    "fin_revision_magnitude",
    "fin_restated_flag",
    "fin_days_since_update",
    "fin_staleness",
    "fin_surprise_zscore",
    "fin_expectation_revision_speed",
})

# R4-03: explicit canonical-registered history requirements for the compound
# lookback families the audit called out.  These override the name-guessing
# rules with the exact formula the kernel uses (addition, never max).
def _compound_history(params: dict[str, Any], window_name: str = "window", *extra: str) -> int:
    w = _positive_int(params.get(window_name)) or 1
    total = max(0, w - 1)
    for name in extra:
        total += _positive_int(params.get(name)) or 0
    return total


register_history_requirement(
    "volume_autocorr", lambda p: _compound_history(p, "window", "lag")
)
register_history_requirement(
    "turnover_autocorr", lambda p: _compound_history(p, "window", "lag")
)
# event-response: an event at s needs its ENTIRE response path s+1..s+H, so the
# lookback is history_window-1 + horizon (the horizon is NOT a warm-up, it is a
# forward reach the historical window must cover).
register_history_requirement(
    "event_historical_response_mean",
    lambda p: _compound_history(p, "history_window", "horizon"),
)
register_history_requirement(
    "event_historical_response_sign_balance",
    lambda p: _compound_history(p, "history_window", "horizon"),
)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _literal_value(expr: Expr) -> Any | None:
    return expr.value if isinstance(expr, Literal) else None


def _operator_param_values(node: CleanedCall, op_impl: Any) -> dict[str, Any]:
    values = dict(node.kwargs_dict())
    param_names = tuple(
        getattr(getattr(op_impl, "metadata", None), "param_names", ()) or ()
    )
    for index, argument in enumerate(node.args):
        if index >= len(param_names):
            break
        literal = _literal_value(argument)
        if literal is not None or isinstance(argument, Literal):
            values.setdefault(param_names[index], literal)
    return values


def _report_period_count(canonical: str, params: dict[str, Any]) -> int:
    def value(name: str, default: int = 0) -> int:
        return _positive_int(params.get(name)) or default

    if canonical == "fin_growth_change":
        return value("growth_periods", 4) + value("compare_periods", 1)
    if canonical in {
        "fin_growth_volatility",
        "fin_growth_stability",
        "fin_growth_persistence",
    }:
        return value("growth_periods", 1) + value("window_periods", 8)
    if canonical == "fin_trend_acceleration":
        return max(value("short_periods", 4), value("long_periods", 8))
    if canonical == "fin_turnover":
        return value("average_periods", 2)
    if canonical in {"fin_positive_streak", "fin_negative_streak", "fin_beat_streak", "fin_miss_streak"}:
        return value("max_periods", 8)
    if canonical in {"fin_yoy", "fin_ttm", "fin_ttm_quarterly", "fin_ttm_cumulative"}:
        return value("periods_per_year", 4)
    if canonical == "fin_quarter_from_cumulative":
        return 2
    return max(
        1,
        value("periods"),
        value("window_periods"),
        value("average_periods"),
        value("long_periods"),
        value("max_periods"),
    )


def _financial_lookback(
    canonical: str, params: dict[str, Any], *, production: bool = False
) -> int:
    """P0-03 compatibility wrapper — the single history authority is
    ``runtime.execution_contract`` (``history_requirement`` + the ported
    ``_financial_report_extension``).  Production: an unavailable data-driven
    fiscal calendar raises ``HistoryRequirementError`` (fail-closed planning).
    Research: the explicit 80-rows/period heuristic.  Non-report-period
    canonicals contribute 0.
    """
    if canonical not in _FIN_REPORT_PERIOD_CANONICALS:
        return 0
    from factor_engine.runtime.execution_contract import (
        _financial_research_heuristic,
        history_requirement,
    )

    req = history_requirement(canonical, params, production=production)
    if production and req.is_full_history:
        # R11 P1-12: production must NOT silently fall back to the fixed 80-rows
        # heuristic on ANY failure.  A missing/errored financial source, a bad
        # period calendar or a broken data-source contract would be masked as a
        # "conservative heuristic" and under-allocate causal lookback, silently
        # corrupting the factor.  Fail the planning instead.
        raise HistoryRequirementError(
            f"{canonical}: data-driven fiscal history is unavailable for "
            f"{_report_period_count(canonical, params)} report periods but mode "
            "is production — no silent fallback to the fixed 80-rows heuristic "
            "(R11 P1-12 fail-closed)"
        )
    if req.is_full_history:
        return _financial_research_heuristic(canonical, params)
    return max(0, int(req.rows))




@dataclass
class AnalysisResult:
    ir: IRNode
    lookback: int
    has_ts_op: bool
    has_cs_op: bool
    referenced_columns: set[str]
    requires_full_history: bool = False
    referenced_fields: dict[str, Any] = field(default_factory=dict)
    referenced_field_ids: set[str] = field(default_factory=set)
    column_schemas: dict[str, Schema] = field(default_factory=dict)
    # P0-03: the authoritative composed history requirement (single authority
    # from ``runtime.execution_contract.factor_history_requirement``).  The
    # legacy integer ``lookback`` is kept only as a derived compatibility value.
    history_requirement: Any = None


# Audit §4.4 flow-grain contracts.  A-share statement flow fields are
# fiscal-year-to-date cumulative (年初至报告期累计), never one-period flows and
# never point-in-time balances.  An operator whose semantics re-read its primary
# input at a specific economic grain must reject a field carrying the opposite
# grain at compile time instead of silently mis-reading it.
#
# Each value is the set of allowed economic grain tuples on the primary input.
# An empty set means "no declared economic grain is accepted" — the operator only
# takes a derived/synthetic period reading (default grain), so feeding it a raw
# YTD-cumulative or balance field fails closed.
OPERATOR_INPUT_GRAIN_CONTRACTS: dict[str, tuple[tuple[str, ...], ...]] = {
    # Cumulative kernels require the YTD-cumulative reading of a flow statement.
    "fin_quarter_from_cumulative": (("flow", "ytd"),),
    "fin_ttm_cumulative": (("flow", "ytd"),),
    # One-period-flow TTM must not receive a YTD-cumulative raw field — that is
    # the "treating a cumulative value as a quarter" hazard.  A balance reading
    # is not a period flow either.
    "fin_ttm_quarterly": (),
    # Point-in-time averaging requires a balance (stock) reading.
    "fin_average_balance": (("balance",),),
}

# Economic grains that carry a real flow-vs-balance meaning.  Default
# ("instrument", "time") grain marks synthetic/research columns, which are
# exempt from grain contracts.
_ECONOMIC_GRAINS = frozenset({("flow", "ytd"), ("balance",)})


class FieldGrainContractError(ValueError):
    """An operator received a field whose economic grain it cannot consume.

    Audit §4.4: a YTD-cumulative flow field must not be fed to an operator that
    re-reads its input as a one-period flow or point-in-time balance, and a
    balance field must not be fed to a flow-cumulative kernel.
    """


def validate_field_grain_contracts(ir: IRNode) -> list[str]:
    """Walk the lowered IR and reject grain-contract violations.

    Only columns carrying a declared economic grain (``flow_ytd`` / ``balance``)
    are checked; synthetic columns with the default instrument/time grain are
    skipped (research formulas may feed derived period flows).
    """
    errors: list[str] = []

    def walk(node: IRNode, lineage: tuple[str, ...] = ()) -> None:
        contract = OPERATOR_INPUT_GRAIN_CONTRACTS.get(node.op)
        if contract is not None and node.inputs:
            first = node.inputs[0]
            if first.op == "column":
                grain = tuple(first.semantic_attrs.get("grain") or ())
                if grain in _ECONOMIC_GRAINS and grain not in contract:
                    name = first.attrs.get("name") or first.attrs.get("field")
                    errors.append(
                        f"operator {node.op!r} requires input grain "
                        f"{contract or 'one-period flow (derived)'}, got {grain!r} on "
                        f"field {name!r} (audit §4.4)"
                    )
        for child in node.inputs:
            walk(child, lineage + (node.op,))

    walk(ir)
    return errors


class FundamentalZeroImputationError(ValueError):
    """Forbidden zero/mean-imputation of undisclosed financial data (P1-003).

    A financial NaN means "not disclosed / not applicable" — it is NOT zero
    revenue / zero debt / zero cash-flow.  Auto-generated formulas must never
    reach ``nan_to_num`` / ``fillna_const(0)`` / ``cs_fill_mean`` /
    ``cs_fill_median`` on a fundamental-domain input unless an explicit research
    override opts in.
    """


# Zero/mean imputation operators that are forbidden on fundamental-domain inputs.
_FUNDAMENTAL_ZERO_IMPUTERS = frozenset(
    {
        "nan_to_num",
        "cs_fill_mean",
        "cs_fill_median",
        "cs_impute_mean",
        "cs_impute_median",
        "group_impute_median",
    }
)


class FieldCatalogMismatchError(ValueError):
    """A persisted FieldRef was built against a different field catalog."""


# ---------------------------------------------------------------------------
# P1-T typed input contracts: an operator family that requires a specific
# price basis / level type must reject an illegal input BEFORE execution.
#
# P0-31: the price-basis decisions consult the field concept's ``price_basis``
# metadata (via ``fields.registry.resolve_field`` -> ``FieldSpec.price_basis``,
# then the canonical concept registry for legacy DSL spellings) instead of the
# hardcoded name-sets.  The name-sets remain a fast fallback for legacy columns
# that carry no typed basis.
# ---------------------------------------------------------------------------
_RETURN_NAME_MARKERS = ("ret", "return", "pct_chg", "pct_change", "momentum", "mom")
_RAW_PRICE_FIELDS = frozenset({"close", "open", "high", "low", "pre_close", "vwap"})
_CONTINUOUS_PRICE_FIELDS = frozenset({
    "continuous_close", "continuous_open", "continuous_high", "continuous_low",
    "continuous_vwap",
})
_DRAWDOWN_FAMILY = frozenset({
    "drawdown", "drawdown_area", "drawdown_depth", "drawdown_duration",
    "drawdown_recovery_half_life", "ts_max_drawdown", "ts_max_drawdown_activity_cost",
    "ts_current_drawdown_area", "ts_current_drawdown_duration",
})
_SPECTRAL_FAMILY = frozenset({
    "spectral_concentration", "spectral_gap", "spectral_entropy",
    "ts_spectral_centroid", "ts_spectral_entropy", "ts_spectral_flatness",
    "ts_spectral_low_frequency_ratio", "ts_periodogram_energy",
})


class TypedInputContractError(ValueError):
    pass


def _leaf_fields(node: IRNode, out: set[str]) -> None:
    if node.op == "column":
        name = str((node.attrs or {}).get("field") or (node.attrs or {}).get("name") or "")
        if name:
            out.add(name)
    for child in node.inputs:
        _leaf_fields(child, out)


def _direct_child_attrs(node: IRNode) -> list[tuple[str, dict[str, Any]]]:
    """Collect ``(child_name, semantic_attrs)`` for a node's DIRECT children.

    R11 contract: each AST node carries its OWN propagated ``OutputSemanticType``
    (in ``semantic_attrs``), so a parent validates only its direct children's
    semantic attributes — never a scan of descendant leaves.  ``child_name`` is
    the child's column name when it is a leaf field, otherwise a positional
    ``input[N]`` label used only for the legacy name-based fallbacks.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for index, child in enumerate(node.inputs):
        name = str(
            (child.attrs or {}).get("field")
            or (child.attrs or {}).get("name")
            or f"input[{index}]"
        )
        out.append((name, child.semantic_attrs or {}))
    return out


def _is_return_field(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in _RETURN_NAME_MARKERS)


def _price_basis_of_field(name: str, *, market: str | None = None) -> str | None:
    """Resolve a leaf field's canonical ``price_basis`` metadata.

    Prefers the registered ``FieldSpec.price_basis`` (populated by the catalog),
    then the canonical-concept registry (legacy DSL spellings such as ``close``
    -> ``raw_close``).  ``None`` means neither carries a price basis, so callers
    keep the legacy name-set as a fallback.

    R17-037: when ``market`` is given the field is resolved through the
    per-market registry — a US ``close`` must never be tagged with an A-share
    price-basis/alias.  ``market=None`` keeps the legacy A-share-bound resolver
    for pre-market IR paths that cannot prove a market.
    """
    if market is not None:
        try:
            from factor_engine.fields.resolver import resolve_market_field

            resolved = resolve_market_field(name, market)
            if resolved is not None:
                basis = getattr(resolved.spec, "price_basis", None)
                if basis:
                    return basis
        except Exception:
            pass
    else:
        from factor_engine.fields import resolve_field

        try:
            spec = resolve_field(name)
            basis = getattr(spec, "price_basis", None)
            if basis:
                return basis
        except Exception:
            pass
    try:
        from factor_engine.fields.concepts import concept_alias_map, get_concept

        concept_id = concept_alias_map().get(str(name).lower())
        if concept_id:
            concept = get_concept(concept_id)
            if concept is not None and concept.price_basis:
                return concept.price_basis
    except Exception:
        pass
    return None


def _resolve_operator_price_basis(
    canonical: str,
    implementation: Any,
    visited_inputs: list[tuple[Any, int]],
) -> str | None:
    """P0-60: resolve the single authoritative ``price_basis`` for a
    return-decomposition operator, or ``None`` when it cannot be positively
    established.

    The operator's ``input_units`` declares which positional params are prices;
    each price input's basis comes from its TYPED child (semantic ``price_basis``
    or field-concept resolution) — never from runtime instrument-column names,
    which are unverifiable in a cross-sectional panel and would fail the
    runtime gate on every call.  A basis is returned only when EVERY price input
    resolves to the SAME basis; a raw-open / adjusted-close mix (or an
    unresolvable price input) stays ``None`` so the runtime
    ``_assert_shared_price_basis`` gate fails closed as designed.
    """
    meta = getattr(implementation, "metadata", None)
    if getattr(meta, "category", None) != "return_decomposition":
        return None
    units = getattr(meta, "input_units", None) or {}
    price_params = {p for p, u in units.items() if u == "price"}
    if not price_params:
        return None
    param_names = list(getattr(meta, "param_names", None) or ())
    bases: set[str] = set()
    for index, (child, _lookback) in enumerate(visited_inputs):
        if index >= len(param_names) or param_names[index] not in price_params:
            continue
        basis = (child.semantic_attrs or {}).get("price_basis")
        if basis is None and str(child.op) == "column":
            basis = _price_basis_of_field(str(child.attrs.get("name", "")))
        if basis is None:
            return None  # a price input without a resolvable basis -> fail closed
        bases.add(basis)
    return bases.pop() if len(bases) == 1 else None


def _flow_semantics_of_field(name: str, *, market: str | None = None) -> str | None:
    """Resolve a leaf field's reporting-flow semantics from its ``FieldSpec``.

    R17-037: market-aware when ``market`` is provided (US quarterly/TTM flow
    semantics must not be read from the A-share legacy FieldSpec).
    """
    if market is not None:
        try:
            from factor_engine.fields.resolver import resolve_market_field

            resolved = resolve_market_field(name, market)
            if resolved is not None:
                return getattr(resolved.spec, "flow_semantics", None)
        except Exception:
            return None
    from factor_engine.fields import resolve_field

    try:
        spec = resolve_field(name)
    except Exception:
        spec = None
    if spec is None:
        return None
    return getattr(spec, "flow_semantics", None)


def validate_typed_input_contracts(ir: IRNode, *, market: str | None = None) -> list[str]:
    """Reject operator families whose typed input contract is violated.

    Audit P1-T: ``drawdown(return_series)`` is rejected (drawdown needs a
    level/wealth-index); spectral analysis on a raw split-sensitive close is
    rejected (continuous/return required); A-share limit operators on a
    continuous price are rejected (they need RAW official limit prices).

    Price-basis decisions consult the field concept's ``price_basis`` (via
    ``fields.registry.resolve_field`` -> ``FieldSpec.price_basis``, then the
    canonical concept registry), falling back to the legacy hardcoded name-sets
    when a field carries no typed basis.

    R11 P0-22: each validator checks only DIRECT children's own
    ``semantic_attrs`` (never a descendant-leaf scan) — the parent reads each
    child's propagated ``price_basis`` / name instead of sweeping the subtree.

    R17-037: ``market`` threads the per-market registry into the leaf-field
    price-basis fallback so a US formula is never tagged by the A-share legacy
    resolver.
    """
    errors: list[str] = []

    def walk(node: IRNode) -> None:
        children = _direct_child_attrs(node)
        if node.op in _DRAWDOWN_FAMILY:
            for name, sem in children:
                basis = sem.get("price_basis") or _price_basis_of_field(name, market=market)
                if _is_return_field(name) or basis in {"RETURN"}:
                    errors.append(
                        f"{node.op} on return input {name}: drawdown requires a "
                        "level / wealth-index input (audit P1-T)"
                    )
        elif node.op in _SPECTRAL_FAMILY:
            for name, sem in children:
                basis = sem.get("price_basis") or _price_basis_of_field(name, market=market)
                if name in _RAW_PRICE_FIELDS or basis in {"RAW"}:
                    errors.append(
                        f"{node.op} on raw split-sensitive price {name}: spectral "
                        "analysis requires continuous/return input (audit P1-T)"
                    )
        elif str(node.op).startswith("ashare_limit_"):
            for name, sem in children:
                basis = sem.get("price_basis") or _price_basis_of_field(name, market=market)
                if name in _CONTINUOUS_PRICE_FIELDS or basis in {"CONTINUOUS", "RETURN"}:
                    errors.append(
                        f"{node.op} on continuous price {name}: A-share limit "
                        "operators require RAW official limit prices (audit P1-T)"
                    )
        for child in node.inputs:
            walk(child)

    walk(ir)
    return errors


def validate_semantic_kind_contracts(ir: IRNode) -> list[str]:
    """Reject operator/input combos that violate typed-IR semantic kinds (P0-30).

    This is the semantic-kind enforcement pass.  It uses the semantic kinds
    propagated onto ``IRNode.semantic_attrs`` (``PriceRaw`` / ``PriceContinuous``
    / ``ReturnDecimal`` / ...) rather than field-name guessing, and is additive
    to :func:`validate_typed_input_contracts` (which keeps the name-set fallback
    for unresolvable legacy columns).

    R11 P0-22: only DIRECT children's own ``semantic_attrs`` are validated (never
    a descendant-leaf scan) — each child's propagated ``semantic_kind`` is the
    single authority for its input type.
    """
    errors: list[str] = []

    def walk(node: IRNode) -> None:
        children = _direct_child_attrs(node)
        if node.op in _DRAWDOWN_FAMILY:
            for name, sem in children:
                if sem.get("semantic_kind") == "ReturnDecimal" or _is_return_field(name):
                    errors.append(
                        f"{node.op} on return input {name}: drawdown requires a "
                        "level / wealth-index input (typed-IR P0-30)"
                    )
        elif node.op in _SPECTRAL_FAMILY:
            for name, sem in children:
                if sem.get("semantic_kind") == "PriceRaw" or name in _RAW_PRICE_FIELDS:
                    errors.append(
                        f"{node.op} on raw split-sensitive price {name}: spectral "
                        "analysis requires continuous/return input (typed-IR P0-30)"
                    )
        elif str(node.op).startswith("ashare_limit_"):
            for name, sem in children:
                if sem.get("semantic_kind") in {"PriceContinuous", "ReturnDecimal"}:
                    errors.append(
                        f"{node.op} on continuous price {name}: A-share limit "
                        "operators require RAW official limit prices (typed-IR P0-30)"
                    )
        for child in node.inputs:
            walk(child)

    walk(ir)
    return errors


class UnknownRawColumnError(ValueError):
    """A raw ``ColumnRef`` with no registered ``FieldSpec`` in production mode.

    Round-7 WS-C (#273): production formulas must reference catalog fields via
    ``field(...)``; an unresolvable raw column would otherwise fall back to a
    generic float panel and silently carry no semantic contract.  Research
    formulas keep the explicit opt-in (``col(...)`` / default ``production=False``).
    """


def validate_input_type_contracts(
    ir: IRNode, *, strict_unknown: bool = False
) -> list[str]:
    """Reject operator calls whose argument violates the typed-input gate.

    Round-7 WS-C (#269-#273): an operator that declares per-parameter
    ``input_types`` (see ``ir.types.OPERATOR_INPUT_TYPE_CONTRACTS``) rejects an
    argument whose ``semantic_kind`` is present but not in the declared set —
    e.g. swapping Volume into a close slot, or Return into a volume slot.
    Arguments with no declared semantic kind (untyped research columns) pass in
    research mode: the gate cannot prove a mismatch and stays permissive for raw
    columns.

    R11 P1-01: ``strict_unknown=True`` (production / default mining) flips the
    fail-open — a CONSTRAINED slot (``allowed_semantic_kinds`` declared) that
    receives an UNKNOWN ``semantic_kind`` is REJECTED, because the operator has
    stated it can only be fed a specific kind and a caller that cannot prove the
    argument is that kind must not proceed.  Only the explicit research override
    (``strict_unknown=False``) keeps the permissive "cannot prove a mismatch"
    behaviour.
    """
    errors: list[str] = []

    def walk(node: IRNode) -> None:
        contract = OPERATOR_INPUT_TYPE_CONTRACTS.get(node.op)
        if contract is not None:
            for index, arg_contract in enumerate(contract):
                if index >= len(node.inputs):
                    break
                child = node.inputs[index]
                if child.op in {"literal", "materialized_series", "plan_ref"}:
                    continue
                child_kind = (child.semantic_attrs or {}).get("semantic_kind")
                name = str(
                    (child.attrs or {}).get("name")
                    or (child.attrs or {}).get("field")
                    or f"input[{index}]"
                )
                if child_kind is None and strict_unknown:
                    if arg_contract.allowed_semantic_kinds is not None:
                        errors.append(
                            f"operator {node.op!r} parameter {arg_contract.parameter!r} "
                            f"declares input_types={sorted(arg_contract.allowed_semantic_kinds)} "
                            f"but got an UNKNOWN semantic_kind on {name!r} — a constrained "
                            "slot cannot be fed an untyped column in production "
                            "(R11 P1-01 fail-closed)"
                        )
                    continue
                if child_kind is not None and not arg_contract.accepts(child_kind):
                    errors.append(
                        f"operator {node.op!r} parameter {arg_contract.parameter!r} "
                        f"declares input_types={sorted(arg_contract.allowed_semantic_kinds)}, "
                        f"got {child_kind!r} on {name!r} (typed-IR #269)"
                    )
        for child in node.inputs:
            walk(child)

    walk(ir)
    return errors


def validate_fundamental_zero_imputation(ir: IRNode) -> list[str]:
    """Return errors for zero/mean imputation applied to fundamental data.

    ``ir.semantic_attrs["domain"]`` is propagated up from the leaf column, so an
    operator whose first input is a fundamental statement column carries
    ``domain == "fundamental"`` all the way to the root.
    """
    errors: list[str] = []

    def walk(node: IRNode) -> None:
        domain = (node.semantic_attrs or {}).get("domain")
        if node.op in _FUNDAMENTAL_ZERO_IMPUTERS:
            if domain == "fundamental":
                errors.append(
                    f"{node.op} on fundamental-domain input: an undisclosed financial "
                    "figure is NOT zero/mean (P1-003); financial NaN = undisclosed, "
                    "never impute in an auto-generated formula"
                )
        elif node.op == "fillna_const":
            # the fill value is a positional literal input, not a kwarg attr
            value = None
            if len(node.inputs) > 1:
                val_node = node.inputs[1]
                if val_node.op == "literal":
                    value = val_node.attrs.get("value")
            if domain == "fundamental" and value == 0:
                errors.append(
                    "fillna_const(0) on fundamental-domain input: undisclosed financial "
                    "data must not be read as zero (P1-003)"
                )
        for child in node.inputs:
            walk(child)

    walk(ir)
    return errors


# A-share Income/CashFlow statements are fiscal-YTD cumulative.  Single-period
# comparisons (QoQ / YoY / growth / pct_change) applied directly to a cumulative
# total are accounting errors: they must first be converted with
# fin_quarter_from_cumulative (or fin_ttm_cumulative).
_YTD_NEEDS_QUARTER_OPS = frozenset({"fin_quarter_from_cumulative", "fin_ttm_cumulative"})
_FLOW_COMPARISON_OPS = frozenset(
    {"fin_qoq", "fin_yoy", "fin_growth", "fin_pct_change", "fin_log_change"}
)


def _flow_grains(node: IRNode, converted: bool) -> Iterable[tuple[str, bool]]:
    """Yield ("flow_ytd", already_converted) for fundamental leaf columns."""
    if node.op in _YTD_NEEDS_QUARTER_OPS:
        converted = True
    if node.op == "column":
        grain = (node.semantic_attrs or {}).get("grain") or ()
        if "_".join(str(g) for g in grain) == "flow_ytd":
            yield ("flow_ytd", converted)
        return
    for child in node.inputs:
        yield from _flow_grains(child, converted)


def validate_flow_semantics(ir: IRNode) -> list[str]:
    """P1-004: a single-period flow comparison must not see a YTD cumulative total.

    A-share ``operating_revenue``/``net_profit``/``operating_cash_flow`` are
    fiscal-YTD cumulative; ``fin_qoq`` (or fin_yoy / growth / pct_change) on them
    computes quarter-over-quarter on a running total.  The guard requires the
    conversion op (``fin_quarter_from_cumulative`` / ``fin_ttm_cumulative``) on the
    path before the comparison.  US statements are already period-scoped by the
    ``timeframe`` filter on their provider bindings.
    """
    errors: list[str] = []

    def walk(node: IRNode) -> None:
        if node.op in _FLOW_COMPARISON_OPS:
            for child in node.inputs:
                for _grain, converted in _flow_grains(child, converted=False):
                    if not converted:
                        errors.append(
                            f"{node.op} applied to A-share fiscal-YTD cumulative flow: "
                            "convert with fin_quarter_from_cumulative (or fin_ttm_cumulative) "
                            "first (P1-004); quarter-over-quarter on a cumulative total is an "
                            "accounting error"
                        )
        for child in node.inputs:
            walk(child)

    walk(ir)
    return errors


class PeriodSelectionContractError(ValueError):
    """A referenced financial statement field has no valid period-selection contract.

    Audit §2.10: every ``financial_pit`` field (StockBalance/StockIncome/
    StockCashFlow/StockIndicator statement columns) must declare how its visible
    report period is selected.  A formula that references such a field without a
    registered contract — or with an invalid selector — fails at compile time
    instead of silently mixing report periods.
    """


def validate_fundamental_period_contracts(referenced_fields: dict[str, Any]) -> list[str]:
    """Fail-closed period-selection contract check for referenced fields.

    Returns a list of human-readable errors (empty when every fundamental
    reference is contracted).  The check runs inside :meth:`Analyzer.lower`, so a
    formula reaching a fundamental statement column must carry a registered
    period-selection contract.
    """
    from factor_engine.storage.sources.financial import FUNDAMENTAL_FIELD_CONTRACTS
    from factor_engine.storage.sources.logical_tables import logical_table_contract

    errors: list[str] = []
    for name, spec in referenced_fields.items():
        # Structural metadata columns (pub_date / update_time / report_period_end_date
        # are reused by name across all four statements) carry no period-selection
        # semantics and are exempt, exactly as in the contract registry.  The
        # exemption keys on the STRUCTURAL ROLE, not on ``mining_allowed``
        # (R10-P0-014): a fundamental field whose mining eligibility is UNKNOWN
        # must still fail closed on a missing period contract — the old guard
        # ``if not mining_allowed`` silently exempted every None-declared field.
        role = str(getattr(spec, "role", "") or "").lower()
        if role in {
            "time", "instrument", "identifier", "label", "group_key",
            "knowledge_time", "effective_time", "period_id", "ingestion_time",
        }:
            continue
        table = str(getattr(spec, "table", "") or "")
        contract = logical_table_contract(table) if table else None
        if contract is None or contract.join_policy != "financial_pit":
            continue
        # The raw name may be an encoded source ref; the contract registry is
        # keyed by canonical field name from the catalog spec.
        canonical = str(getattr(spec, "name", "") or name)
        field_contract = FUNDAMENTAL_FIELD_CONTRACTS.get(canonical)
        if field_contract is None:
            errors.append(
                f"fundamental field {canonical!r} ({table}) has no registered "
                f"period-selection contract (audit §2.10); register it via "
                f"register_fundamental_field()"
            )
            continue
        selector = str(field_contract.period_selector)
        if selector not in {"latest_visible_period", "annual_only", "quarterly_only"}:
            errors.append(
                f"fundamental field {canonical!r} declares invalid period_selector "
                f"{selector!r}"
            )
    return errors


def validate_max_domains(analysis: AnalysisResult, *, max_domains: int = 2) -> list[str]:
    if max_domains < 1:
        raise ValueError("max_domains must be positive")
    auxiliary = {"auxiliary", "calendar", "reference", "status", "classification"}
    domains = {
        str(spec.domain) for spec in analysis.referenced_fields.values()
        if str(spec.domain or "auxiliary") not in auxiliary
    }
    if len(domains) <= max_domains:
        return []
    return [
        f"formula uses {len(domains)} primary domains {sorted(domains)} "
        f"exceeding max_domains={max_domains}"
    ]


class ProductionMarketContextRequiredError(ValueError):
    """R40 #174: a production Analyzer must be constructed with an explicit
    market context.  ``market=None`` keeps the legacy A-share fallback only in
    research/compat mode — a production formula that cannot prove its market
    must fail closed instead of silently resolving leaves A-share-bound."""


class Analyzer:
    """Lower Expr trees to IR and derive deterministic causal history."""

    def __init__(self, *, production: bool = False, market: str | None = None) -> None:
        # Round-7 WS-C (#273): production mode rejects raw ColumnRefs with no
        # registered FieldSpec (no generic float-panel fallback).  Research mode
        # (default) keeps the explicit opt-in.
        self._production = bool(production)
        # R17-037: the analyzer resolves leaf fields through the per-market
        # registry when a market is declared.  ``None`` keeps legacy behavior
        # (A-share-bound) for callers that cannot prove a market.
        # R40 #174: production mode requires an explicit market context — the
        # legacy A-share fallback (market=None) is only a research/compat path.
        if self._production and market is None:
            raise ProductionMarketContextRequiredError(
                "production Analyzer requires an explicit market context "
                "(e.g. Analyzer(production=True, market='ashare')); market=None "
                "keeps the legacy A-share fallback only in research/compat mode "
                "(R40 #174)"
            )
        self._market = market

    def _resolve_field_ir(self, node: Any):
        """Resolve a ColumnRef/FieldRef through the per-market registry (R17-037).

        When a market is declared the legacy A-share registry is never consulted:
        a US formula's leaf fields resolve against ``MULTI_MARKET_FIELD_REGISTRY``
        so A-share aliases / units / price-basis cannot leak into the typed IR.
        ``market=None`` keeps the legacy resolver for callers that cannot prove
        a market.
        """
        if self._market is not None:
            from factor_engine.fields.resolver import resolve_market_field

            resolved = resolve_market_field(node, self._market, strict=False)
            return resolved.spec if resolved is not None else None
        from factor_engine.fields import resolve_field

        return resolve_field(node)

    def lower(self, expr: Expr, *, production: bool | None = None) -> AnalysisResult:
        columns: set[str] = set()
        referenced_fields: dict[str, Any] = {}
        column_schemas: dict[str, Schema] = {}
        has_ts = False
        has_cs = False
        requires_full_history = False
        # R11 P1-01: one mode authority for the whole lower() pass.  The visit
        # closure recomputes it for the raw-ColumnRef gate; the typed-input gate
        # below uses the same value so production never accidentally runs with
        # research permissiveness.
        effective_production = self._production if production is None else bool(production)

        def visit(node: Expr) -> tuple[IRNode, int]:
            nonlocal has_ts, has_cs, requires_full_history

            if isinstance(node, ColumnRef):
                columns.add(node.name)
                spec = self._resolve_field_ir(node)
                # Round-7 WS-C (#273): production mode fails closed on an unknown
                # raw ColumnRef — no generic float-panel fallback.  Research mode
                # keeps the explicit opt-in.  ``effective_production`` is computed
                # once at the top of ``lower()`` (R11 P1-01 mode authority); this
                # closure reads it — it must NOT reassign it here or Python makes
                # it a local of ``visit`` and the operator-path lookback gate
                # (which never hits a ColumnRef) sees an unbound variable.
                if spec is None and effective_production:
                    raise UnknownRawColumnError(
                        f"unknown raw column {node.name!r} has no registered FieldSpec; "
                        "production formulas must reference catalog fields via "
                        "field(...) (typed-IR #273)"
                    )
                schema = Schema.from_field(spec) if spec is not None else DEFAULT_COLUMN_SCHEMA
                column_schemas[node.name] = schema
                if spec is not None:
                    referenced_fields[node.name] = spec
                attrs = {"name": node.name}
                if isinstance(node, FieldRef):
                    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

                    # R17-037: the field-registry hash comes from the SAME market
                    # registry the field was resolved in (A/US catalogs differ).
                    if self._market is not None:
                        current_hash = (
                            MULTI_MARKET_FIELD_REGISTRY.registry_for(self._market).catalog_hash()
                        )
                    else:
                        from factor_engine.fields import FIELD_REGISTRY

                        current_hash = FIELD_REGISTRY.catalog_hash()
                    if not node.catalog_hash or node.catalog_hash != current_hash:
                        raise FieldCatalogMismatchError(
                            f"field {node.field_id or node.canonical_name!r} catalog hash "
                            f"{node.catalog_hash or '<missing>'} does not match active catalog "
                            f"{current_hash}"
                        )
                    if spec is None or str(spec.field_id) != node.field_id:
                        raise FieldCatalogMismatchError(
                            f"field identity {node.field_id!r} no longer resolves to the active catalog"
                        )
                    attrs.update(
                        {
                            "field_id": node.field_id,
                            "field": node.canonical_name,
                            "source_table": node.table,
                            "source_field": node.source_name,
                            "field_registry_hash": node.catalog_hash,
                        }
                    )
                if spec is not None:
                    price_basis = getattr(spec, "price_basis", None) or _price_basis_of_field(
                        node.name, market=self._market
                    )
                    if self._market is not None:
                        from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

                        reg_hash = (
                            MULTI_MARKET_FIELD_REGISTRY.registry_for(self._market).catalog_hash()
                        )
                    else:
                        from factor_engine.fields import FIELD_REGISTRY

                        reg_hash = FIELD_REGISTRY.catalog_hash()
                    attrs.update({
                        "field_id": str(spec.field_id),
                        "field_registry_hash": reg_hash,
                        "field": spec.name,
                        "dtype": schema.dtype,
                        "unit": spec.unit,
                        "source_table": spec.table,
                        "source_field": spec.source_name,
                        "domain": spec.domain,
                        "frequency": spec.frequency,
                        "cardinality": spec.cardinality,
                        "temporal_model": spec.temporal_model,
                        "pit_safe": spec.strict_pit_allowed,
                    })
                    if price_basis:
                        attrs["price_basis"] = price_basis
                # Propagate typed price-basis / flow-semantics / semantic-kind
                # into semantic_attrs (never attrs, to keep IR hash stable).
                semantic: dict[str, Any] = {}
                if spec is not None:
                    semantic["grain"] = tuple(spec.grain)
                    flow_semantics = getattr(spec, "flow_semantics", None)
                    if flow_semantics:
                        semantic["flow_semantics"] = flow_semantics
                    price_basis = attrs.get("price_basis")
                    if price_basis:
                        semantic["price_basis"] = price_basis
                # Round-6 P0-50: propagate the field's PIT/availability
                # coordinate so a factor's root schema can answer
                # "available_at = max(inputs)" (audit §51).  Round-7 WS-C
                # (#277): the descriptor comes from FieldSpec.available_at /
                # knowledge_time_column / role, then an EXACT-name EOD default
                # (no "pe"/"pb"/"close" substring guessing).
                if schema.available_at is None:
                    from factor_engine.ir.schema import _available_at_of

                    name_default = _available_at_of(type("_Spec", (), {"name": node.name})())
                    semantic["available_at"] = name_default
                else:
                    semantic["available_at"] = schema.available_at
                semantic["availability_expr"] = schema.availability_expr
                semantic["knowledge_model"] = schema.knowledge_model
                semantic["fiscal_grain"] = schema.fiscal_grain
                semantic["source_vintage"] = schema.source_vintage
                semantic["universe_id"] = schema.universe_id
                # Round-7 WS-C (#269-#273): semantic-kind resolution order is
                # (1) FieldSpec.semantic_kind (declared, authoritative),
                # (2) the mapping helper (price-basis / flow-semantics / exact
                # name for raw columns), (3) semantic_type_of as a final pass.
                from factor_engine.fields.spec import semantic_kind_of_field

                kind = semantic_kind_of_field(spec) if spec is not None else None
                if kind is None:
                    kind = semantic_kind_of_field(node.name)
                if kind is None:
                    typed = semantic_type_of(
                        price_basis=semantic.get("price_basis"),
                        flow_semantics=semantic.get("flow_semantics"),
                        frequency=schema.frequency,
                        domain=schema.domain,
                        # 2026-08-29 LQTP compat: a logical DataTable field whose
                        # FieldSpec declares role='group_key' (e.g.
                        # IndustryDaily.IndustryCode -> industry_code) must resolve
                        # to GroupKey BEFORE the coarse frequency fallback
                        # (frequency=daily would otherwise mis-tag the group key
                        # as a DailySeries and the typed-input gate rejects it as
                        # group_mean(group=...) input).
                        role=getattr(spec, "role", None) if spec is not None else None,
                    )
                    kind = typed.value if typed is not None else None
                if kind is not None:
                    semantic["semantic_kind"] = kind
                if schema.semantic_lattice is not None:
                    semantic["lattice"] = schema.semantic_lattice
                return IRNode(op="column", attrs=attrs, semantic_attrs=semantic), 0
            if isinstance(node, Literal):
                return IRNode(op="literal", attrs={"value": node.value}), 0
            if not isinstance(node, CleanedCall):
                raise NotImplementedError(f"Unsupported expr: {type(node).__name__}")

            from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
            from factor_engine.cleaned_operators.registry import OperatorRegistry

            ensure_cleaned_loaded()
            if node.op in {"bfill", "causal_bfill"}:
                from factor_engine.backend.polars_long_policy import UnsupportedCausalOperatorError

                raise UnsupportedCausalOperatorError(
                    f"{node.op} is disabled: backward-looking fill is not point-in-time safe"
                )
            canonical = OperatorRegistry.resolve_canonical_strict(node.op)
            implementation = OperatorRegistry.get(canonical)

            # R40 #176: the declared AxisEffectContract is the single authority
            # for has_ts/has_cs.  Production REJECTS an undeclared operator
            # (name/category heuristics are research-only compat fallback).
            axis_contract = axis_effect_contract_for(canonical)
            if axis_contract is not None:
                if axis_contract.has_time_series_effect:
                    has_ts = True
                if axis_contract.has_cross_section_effect:
                    has_cs = True
            elif effective_production:
                check_axis_effect_declared(canonical, production=True)
            else:
                if implementation is None:
                    category = ""
                else:
                    category = getattr(implementation.metadata, "category", "") or ""
                if category in {
                    "time_series",
                    "technical_signal",
                    "price_volume",
                    "price_volume_extension",
                    "ohlc_volatility",
                    "candle_pattern",
                    "intraday_microstructure",
                    "signal",
                    "price_structure",
                    "chart_pattern",
                    "fundamental_period",
                }:
                    has_ts = True
                if category in {"cross_sectional", "group_neutralization"}:
                    has_cs = True
                if canonical.startswith("ts_") or canonical in {
                    "SMA",
                    "EMA",
                    "WMA",
                    "delay",
                    "decay_linear",
                }:
                    has_ts = True
                if canonical in {
                    "rank",
                    "zscore",
                    "scale",
                    "normalize",
                    "winsorize",
                    "quantile",
                    "neutralize",
                } or canonical.startswith("group_"):
                    has_cs = True

            try:
                from factor_engine.cleaned_operators.production_hardening import (
                    FULL_HISTORY_REPLAY_CANONICALS,
                )

                if canonical in FULL_HISTORY_REPLAY_CANONICALS:
                    requires_full_history = True
            except ImportError:
                # R40 #177: a production plan must NEVER silently underestimate
                # lookback because the full-history contract could not be
                # imported — that would under-allocate causal history and
                # silently corrupt the factor.  Production hard-fails; research
                # degrades (no full-history flag) and keeps the heuristic sets
                # below as a compat fallback.
                if effective_production:
                    raise
                pass

            visited_inputs = [visit(argument) for argument in node.args]
            inputs = tuple(item[0] for item in visited_inputs)
            deepest_child = max((item[1] for item in visited_inputs), default=0)
            attrs: dict[str, Any] = {}
            for key, value in node.kwargs_dict().items():
                if isinstance(value, Expr):
                    lowered, keyword_lookback = visit(value)
                    if lowered.op != "literal":
                        raise NotImplementedError(
                            f"cleaned op {node.op!r} kwargs must be literals, got {key!r}"
                        )
                    attrs[key] = lowered.attrs["value"]
                    deepest_child = max(deepest_child, keyword_lookback)
                else:
                    attrs[key] = value

            from factor_engine.backend.parameter_aliases import normalize_parameter_aliases

            attrs = normalize_parameter_aliases(canonical, attrs)
            # P0-03: ONE history authority — ``runtime.execution_contract``.
            # The analyzer no longer guesses lookback from parameter names; the
            # per-node requirement (its own contribution, excluding children)
            # comes from ``own_history_requirement``, and the composed factor
            # requirement is re-derived at the end via ``factor_history_requirement``.
            from factor_engine.runtime.execution_contract import own_history_requirement

            params = _operator_param_values(node, implementation)
            if effective_production and canonical in _FIN_REPORT_PERIOD_CANONICALS:
                # Report-period operators fail closed in production when the
                # data-driven fiscal calendar is unavailable (R11 P1-12).
                _financial_lookback(canonical, params, production=True)
            own_req = own_history_requirement(canonical, params)
            if own_req.is_full_history:
                requires_full_history = True
            own_increment = max(0, int(own_req.rows))
            from factor_engine.backend.operator_types import OPERATOR_SIGNATURES

            signature = OPERATOR_SIGNATURES.get(canonical)
            if signature is not None:
                if signature.output.value == "Series[Bool]":
                    attrs["dtype"] = "bool"
                elif signature.output.value == "Series[String]":
                    attrs["dtype"] = "string"
                elif signature.output.value == "Series[Datetime]":
                    attrs["dtype"] = "datetime64[ns]"

            # P0-60: return-decomposition operators carry the authoritative
            # shared ``price_basis`` on attrs so the runtime kernel forwards it
            # (kw = node.attrs).  Without this stamp the basis is re-derived at
            # runtime from panel column names — instrument codes in a
            # cross-sectional panel — which the fail-closed gate rejects.
            if "price_basis" not in attrs:
                _op_price_basis = _resolve_operator_price_basis(
                    canonical, implementation, visited_inputs
                )
                if _op_price_basis:
                    attrs["price_basis"] = _op_price_basis

            # Round-7 WS-C (#265-#268): propagate field-catalog metadata into
            # semantic_attrs via the SEMANTIC LATTICE join across ALL inputs
            # (not first-input inheritance).  A dimension with one distinct
            # value stays scalar; conflicting values are recorded as mixed_.
            semantic: dict[str, Any] = {}
            if inputs:
                child_views: list[dict[str, Any]] = []
                for child in inputs:
                    view = dict(child.semantic_attrs or {})
                    # Leaf columns keep catalog scalars (domain / frequency /
                    # cardinality / temporal_model / unit / price_basis /
                    # flow_semantics / pit_safe) on ``attrs``; fold those in so
                    # the lattice join sees every child's full type slot.
                    for key in (
                        "domain",
                        "frequency",
                        "cardinality",
                        "temporal_model",
                        "unit",
                        "price_basis",
                        "flow_semantics",
                        "pit_safe",
                    ):
                        if view.get(key) is None:
                            attr_val = (child.attrs or {}).get(key)
                            if attr_val is not None:
                                view[key] = attr_val
                    if "pit_safe" not in view:
                        # R10 #6: a node with NO PIT declaration is UNKNOWN
                        # (None), not safe.  The lattice join propagates UNKNOWN
                        # upward; the production four-layer PIT gate rejects it.
                        # A literal constant is not a data field — it carries no
                        # PIT hazard and must not poison the root's pit_safe to
                        # UNKNOWN (a scalar 3 has no look-ahead).
                        if child.op == "literal":
                            view["pit_safe"] = True
                        else:
                            view["pit_safe"] = None
                    child_views.append(view)
                semantic = lattice_join_semantic_attrs(child_views)
                # knowledge_model / fiscal_grain / universe_id are PIT
                # descriptors: unambiguous across children -> propagate; else
                # the MIXED marker + mixed_ tuple (review R9-P0-003 — never
                # ``distinct[0]``, which made the root semantic operand-order
                # dependent: add(fundamental, price) != add(price, fundamental)).
                for key in ("knowledge_model", "fiscal_grain", "universe_id"):
                    values = [
                        (child.semantic_attrs or {}).get(key)
                        for child in inputs
                        if child.op != "literal"
                    ]
                    distinct = list(dict.fromkeys(v for v in values if v is not None))
                    if len(distinct) == 1:
                        semantic[key] = distinct[0]
                    elif len(distinct) > 1:
                        semantic[key] = MIXED
                        semantic[f"mixed_{key}"] = tuple(distinct)
                # Round-6 P0-50 / WS-C #274-#275: ``available_at`` propagates
                # bottom-up as the LATEST input's knowledge-time descriptor — the
                # factor cannot be used before its last-arriving input is knowable
                # (audit §51).  Literal children carry no availability and are
                # excluded.  An UNKNOWN input fails closed (result "unknown").
                from factor_engine.ir.schema import propagate_available_at

                child_availabilities = tuple(
                    (child.semantic_attrs or {}).get("available_at")
                    for child in inputs
                    if child.op != "literal"
                )
                available_at = propagate_available_at(child_availabilities)
                if available_at is not None:
                    semantic["available_at"] = available_at
                # source_vintage: unambiguous -> the single structured spec;
                # conflicting vintages -> the MIXED marker (the review's
                # "dependency set") + mixed_ tuple, NEVER ``distinct[0]``
                # (review R9-P0-003 — operand-order independence).
                vintage_values = [
                    (child.semantic_attrs or {}).get("source_vintage")
                    for child in inputs
                    if child.op != "literal"
                ]
                vintage_distinct = list(dict.fromkeys(v for v in vintage_values if v is not None))
                if len(vintage_distinct) == 1:
                    semantic["source_vintage"] = vintage_distinct[0]
                elif len(vintage_distinct) > 1:
                    semantic["source_vintage"] = MIXED
                    semantic["mixed_source_vintage"] = tuple(vintage_distinct)
                # Join the children's semantic lattices into the root.
                joined_lattice: SemanticLattice | None = None
                for child in inputs:
                    child_lattice = (child.semantic_attrs or {}).get("lattice")
                    if isinstance(child_lattice, SemanticLattice):
                        joined_lattice = (
                            child_lattice
                            if joined_lattice is None
                            else joined_lattice.join(child_lattice)
                        )
                if joined_lattice is not None:
                    semantic["lattice"] = joined_lattice
            if signature is not None and signature.output_unit and signature.output_unit != "inherit":
                semantic["unit"] = signature.output_unit
            # P0-30 / WS-C: derive the typed-IR semantic kind from the joined
            # attrs — only when the lattice left the kind unresolved (a
            # single-kind join already pinned it; a mixed join stays unresolved).
            if "semantic_kind" not in semantic and "mixed_semantic_kind" not in semantic:
                kind = semantic_type_of(
                    price_basis=semantic.get("price_basis"),
                    flow_semantics=semantic.get("flow_semantics"),
                    frequency=semantic.get("frequency"),
                    domain=semantic.get("domain"),
                )
                if kind is not None:
                    semantic["semantic_kind"] = kind.value

            return (
                IRNode(op=canonical, inputs=inputs, attrs=attrs, semantic_attrs=semantic),
                deepest_child + own_increment,
            )

        ir, lookback = visit(expr)
        # P0-03: the composed factor history comes from the SINGLE authority
        # (``factor_history_requirement``).  The per-node ``visit`` composition
        # stays for structural reasons but the authoritative rows / full-history
        # decision is re-derived here.
        from factor_engine.runtime.execution_contract import factor_history_requirement

        history_req = factor_history_requirement(ir)
        if history_req.is_full_history:
            requires_full_history = True
        if not history_req.is_full_history:
            lookback = int(history_req.rows)
        period_errors = validate_fundamental_period_contracts(referenced_fields)
        if period_errors:
            raise PeriodSelectionContractError("; ".join(period_errors))
        zero_impute_errors = validate_fundamental_zero_imputation(ir)
        if zero_impute_errors:
            raise FundamentalZeroImputationError("; ".join(zero_impute_errors))
        flow_errors = validate_flow_semantics(ir)
        if flow_errors:
            raise FundamentalZeroImputationError("; ".join(flow_errors))
        grain_errors = validate_field_grain_contracts(ir)
        if grain_errors:
            raise FieldGrainContractError("; ".join(grain_errors))
        typed_errors = validate_typed_input_contracts(ir, market=self._market)
        if typed_errors:
            raise TypedInputContractError("; ".join(typed_errors))
        semantic_kind_errors = validate_semantic_kind_contracts(ir)
        if semantic_kind_errors:
            raise TypedInputContractError("; ".join(semantic_kind_errors))
        # Round-7 WS-C (#269): per-parameter typed-input gate (input_types).
        # R11 P1-01: production fails closed on an UNKNOWN kind in a constrained
        # slot; research keeps the permissive "cannot prove a mismatch" path.
        input_type_errors = validate_input_type_contracts(
            ir, strict_unknown=effective_production
        )
        if input_type_errors:
            raise TypedInputContractError("; ".join(input_type_errors))
        return AnalysisResult(
            ir=ir,
            lookback=lookback,
            has_ts_op=has_ts,
            has_cs_op=has_cs,
            referenced_columns=columns,
            requires_full_history=requires_full_history,
            referenced_fields=referenced_fields,
            referenced_field_ids={str(spec.field_id) for spec in referenced_fields.values()},
            column_schemas=column_schemas,
            history_requirement=history_req,
        )
