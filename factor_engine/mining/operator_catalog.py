# -*- coding: utf-8 -*-
"""Single discovery authority for automated factor mining.

The mining layer (AlphaProbe / AlphaMiner / cold-start) must never guess which
operators are usable by poking at the raw ``DAILY_CANONICALS`` /
``EXTENDED_ONLY_CANONICALS`` / ``RESEARCH_ONLY_CANONICALS`` surface partitions.
Every registered canonical is classified once into a :class:`MiningRole` plus a
machine ``mining_eligible`` flag, and the one public entry point is
:func:`get_mining_operators`.

Roles are NOT "how big the search space is" — they are the legal grammar slot:

* ``ALPHA`` — expression intermediate and terminal.
* ``ALPHA_HIGH_COST`` — same, but only inside a cost-constrained deep lane.
* ``STATE`` / ``CONDITION`` / ``EVENT`` — non-terminal slots: ``where``/``if``
  conditions, ``state * alpha`` gates, event masks.
* ``GROUP_STATE`` / ``GLOBAL_STATE`` — interaction-only (never a standalone
  terminal; the per-row value is identical within a group / the whole market).
* ``INTRADAY_EOD`` — minute → daily ``GrainTransform`` (legal once the source
  session is certified).
* ``FUNDAMENTAL_PIT`` — financial / as-of fundamental panels.
* ``RECIPE_INTERNAL`` / ``SOURCE_TRANSFORM`` / ``INTERNAL`` / ``DIAGNOSTIC`` /
  ``LEGACY`` / ``RESEARCH`` — supporting layers, never mined as factors.
* ``DENIED`` — future functions / random kernels / non-causal fills / unsafe
  primitives; permanently excluded from every mining path.

``mining_eligible`` is a *machine* field, not an opinion: it requires
``production_certified`` (all six gates) AND a role that admits mining AND
available sources AND a matching target frequency.  The invariant
``MINING_ELIGIBLE_WITHOUT_CERTIFICATION == ∅`` is guaranteed by construction:
``get_mining_operators`` never returns an uncertified operator unless the caller
explicitly requests ``admission="pending"`` for planning purposes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class MiningRole(str, Enum):
    ALPHA = "alpha"
    ALPHA_HIGH_COST = "alpha_high_cost"
    STATE = "state"
    CONDITION = "condition"
    EVENT = "event"
    GROUP_STATE = "group_state"
    GLOBAL_STATE = "global_state"
    INTRADAY_EOD = "intraday_eod"
    FUNDAMENTAL_PIT = "fundamental_pit"
    RECIPE_INTERNAL = "recipe_internal"
    SOURCE_TRANSFORM = "source_transform"
    INTERNAL = "internal"
    DIAGNOSTIC = "diagnostic"
    RESEARCH = "research"
    LEGACY = "legacy"
    DENIED = "denied"


# Roles that may appear inside an automatic mining search (as terminal or as a
# legal non-terminal grammar slot).  Everything outside is supporting-only.
_MINEABLE_ROLES = frozenset(
    {
        MiningRole.ALPHA,
        MiningRole.ALPHA_HIGH_COST,
        MiningRole.STATE,
        MiningRole.CONDITION,
        MiningRole.EVENT,
        MiningRole.GROUP_STATE,
        MiningRole.GLOBAL_STATE,
        MiningRole.INTRADAY_EOD,
        MiningRole.FUNDAMENTAL_PIT,
    }
)

# Roles that may never be a standalone factor terminal (interaction only).
_NON_TERMINAL_ROLES = frozenset(
    {
        MiningRole.STATE,
        MiningRole.CONDITION,
        MiningRole.EVENT,
        MiningRole.GROUP_STATE,
        MiningRole.GLOBAL_STATE,
    }
)

# AST grammar positions allowed per role.  The mining grammar reads these.
_ROLE_AST_POSITIONS: dict[MiningRole, tuple[str, ...]] = {
    MiningRole.ALPHA: ("intermediate", "terminal"),
    MiningRole.ALPHA_HIGH_COST: ("intermediate", "terminal"),
    MiningRole.STATE: ("gate", "interaction"),
    MiningRole.CONDITION: ("condition",),
    MiningRole.EVENT: ("condition", "mask"),
    MiningRole.GROUP_STATE: ("gate", "interaction"),
    MiningRole.GLOBAL_STATE: ("gate", "interaction"),
    MiningRole.INTRADAY_EOD: ("intermediate", "terminal"),
    MiningRole.FUNDAMENTAL_PIT: ("intermediate", "terminal"),
    MiningRole.RECIPE_INTERNAL: (),
    MiningRole.SOURCE_TRANSFORM: (),
    MiningRole.INTERNAL: (),
    MiningRole.DIAGNOSTIC: (),
    MiningRole.RESEARCH: (),
    MiningRole.LEGACY: (),
    MiningRole.DENIED: (),
}


@dataclass(frozen=True)
class MiningOperator:
    canonical: str
    role: MiningRole
    mining_eligible: bool
    production_certified: bool
    authoring_tier: str
    lifecycle_status: str
    cost_tier: int
    required_sources: tuple[str, ...]
    available_sources: tuple[str, ...]
    input_grain: str | None
    output_grain: str | None
    input_semantic_types: tuple[str, ...]
    output_unit: str | None
    scope: str
    searchable_params: tuple[str, ...]
    allowed_ast_positions: tuple[str, ...]
    terminal_allowed: bool
    stateful: bool
    checkpoint_supported: bool
    incremental_supported: bool
    full_history_replay_required: bool
    missing_policy: str | None
    blockers: tuple[str, ...] = field(default=())
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Source / frequency knowledge
# ---------------------------------------------------------------------------

_SOURCE_OF_GRAIN = {
    "minute": "minute_bar",
    "daily": "daily_bar",
    "fundamental_period": "fundamental_pit",
    "shareholder": "shareholder_pit",
    "relation": "relation_pit",
    "event": "event_pit",
}

_KNOWN_SOURCES = frozenset(
    {
        "daily_bar",
        "minute_bar",
        "fundamental_pit",
        "shareholder_pit",
        "relation_pit",
        "event_pit",
        "index_pit",
    }
)

# Fundamental-period / financial operators need the financial as-of adapter.
_FUNDAMENTAL_PREFIXES = ("fin_", "valuation_", "pe_", "pb_", "ps_", "pcf_")

# Operators that semantically return a boolean/event/state mask rather than a
# continuous alpha (name conventions that are too weak to prove role, so they
# are only hints — the authoritative signal is the catalog ``scope``/``role``).
_STATE_SUFFIX_HINTS = (
    "_state",
    "_flag",
    "_touch",
    "_threshold",
    "_crossed",
    "_break",
)

_STATE_LITERAL_OPS = frozenset(
    {
        "state_latch", "state_hold", "state_slew_limit", "state_deadband",
        "state_ewm_if", "state_since_reduce", "event_refractory",
        "cross_event", "directional_change_state", "state_since_trend_tstat",
        "ts_cumulative_deviation_score", "ts_threshold_cycle_period",
        "ts_threshold_cycle_asymmetry", "state_episode_mfe",
        "state_episode_mae", "state_episode_efficiency",
        "state_episode_retrace_ratio", "state_episode_excursion_balance",
        "ts_interval_nesting_depth", "candle_gap_atr",
        "trade_when", "digital_count", "event_count", "limit_up_close",
        "limit_down_open", "ashare_limit_touch", "ashare_limit_failed",
    }
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _catalog_record(canonical: str) -> dict[str, Any]:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._catalog.get(canonical) or {}


def _surface_of(canonical: str, catalog: dict[str, Any]) -> str:
    surface = str(catalog.get("surface") or "").strip().lower()
    if surface:
        return surface
    try:
        from cleaned_operators.operator_surface import classify_canonical

        return classify_canonical(canonical)
    except Exception:
        return "unclassified"


def _role_field(catalog: dict[str, Any]) -> str | None:
    value = catalog.get("role")
    if value in (None, "", "None"):
        return None
    return str(value).strip().lower()


def cost_tier(canonical: str, catalog: dict[str, Any]) -> int:
    """Derive a 0..7 cost tier from the operator's metadata ``cost:`` tag.

    Tiers follow the plan's ladder: 0 elementwise, 1 rolling simple,
    2 technical / basic stats, 3 regression / group / relation,
    4 nonlinear / spectral, 5+ topology / kernel / optimization.
    """
    tags = tuple(str(tag) for tag in (catalog.get("tags") or ()))
    for tag in tags:
        match = re.fullmatch(r"cost:(\d+)", tag.strip())
        if match:
            return int(match.group(1))
    # Fallback by category prefix when no explicit cost tag exists.
    from cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry.get(canonical, "pandas_numpy")
    metadata = getattr(operator, "metadata", None)
    declared_cost = getattr(metadata, "cost_model", None)
    if declared_cost is not None:
        return 4  # a declared cost model implies non-trivial computation
    category = str(catalog.get("category") or "").lower()
    if "spectral" in category or "topology" in category or "kernel" in category:
        return 5
    if "matrix" in category or "wavelet" in category or "ssa" in category:
        return 5
    return 1


def _required_sources(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    grain = str(catalog.get("input_grain") or "").lower()
    if grain:
        return (_SOURCE_OF_GRAIN.get(grain, grain + "_bar"),)
    operator_name = canonical.lower()
    if operator_name.startswith(("intra_",)) or operator_name in {
        "intraday_activity_duration_curvature", "intraday_impact_decay_rate",
    }:
        return ("minute_bar",)
    if operator_name.startswith(_FUNDAMENTAL_PREFIXES) or str(
        catalog.get("scope") or ""
    ) == "fundamental_period":
        return ("fundamental_pit",)
    return ("daily_bar",)


def assign_mining_role(canonical: str, catalog: dict[str, Any] | None = None) -> MiningRole:
    """Deterministic, fail-closed role assignment for one canonical.

    Order matters and every rule is an exclusion before it is a promotion: the
    permanently-forbidden / non-factor / internal / diagnostic classes always
    win over "looks like an alpha".
    """
    catalog = dict(catalog) if catalog is not None else _catalog_record(canonical)
    from cleaned_operators.registry import OperatorRegistry

    registered = canonical in OperatorRegistry._catalog
    if not registered:
        try:
            from research_tools.registry import ResearchToolRegistry

            if canonical in ResearchToolRegistry.list_canonical():
                return MiningRole.DIAGNOSTIC
        except Exception:
            pass
        return MiningRole.INTERNAL

    surface = _surface_of(canonical, catalog)

    # 1. Permanently forbidden / non-factor / unsafe / internal / legacy.
    try:
        from cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS

        if canonical in PERMANENTLY_FORBIDDEN_CANONICALS:
            return MiningRole.DENIED
    except Exception:
        pass
    if surface in ("unsafe", "internal"):
        return MiningRole.DENIED if surface == "unsafe" else MiningRole.INTERNAL

    try:
        from cleaned_operators.production_hardening import NON_FACTOR_PRODUCTION_CANONICALS

        if canonical in NON_FACTOR_PRODUCTION_CANONICALS:
            if canonical.startswith(("holder_", "relation_")):
                return MiningRole.SOURCE_TRANSFORM
            if canonical == "micro_bvc_vpin":
                return MiningRole.RESEARCH  # spec §16: research-only, never mined
            return MiningRole.DENIED
    except Exception:
        pass

    if surface == "legacy":
        return MiningRole.LEGACY
    if surface == "research":
        return MiningRole.RESEARCH

    # 2. Source-blocked (data absent) — never a mining candidate until the
    #    physical field / vintage exists; role keeps its factor identity so the
    #    admission matrix can say exactly which blocker applies.
    try:
        from cleaned_operators.production_hardening import SOURCE_BLOCKED_CANONICALS

        if canonical in SOURCE_BLOCKED_CANONICALS:
            # Still classify by shape below but mark the mining role as the
            # fundamental pit lane when it is a fundamental operator.
            if canonical.startswith("fin_") or canonical.startswith("intraday_"):
                return MiningRole.INTRADAY_EOD if canonical.startswith("intraday_") else MiningRole.FUNDAMENTAL_PIT
    except Exception:
        pass

    # 3. Explicit review flags.
    if catalog.get("compatibility_only"):
        return MiningRole.INTERNAL  # one mathematical definition == one canonical
    if catalog.get("diagnostic_only"):
        return MiningRole.DIAGNOSTIC
    if catalog.get("benchmark_only"):
        return MiningRole.DIAGNOSTIC
    if catalog.get("hidden_from_default_mining"):
        return MiningRole.RESEARCH

    # 4. In-sample diagnostics / benchmark-only forms — never mined; the
    #    *_prior / *_forecast_error / *_ex_self counterparts are the ALPHA forms.
    if canonical in _DIAGNOSTIC_IN_SAMPLE_CANONICALS:
        return MiningRole.DIAGNOSTIC
    if canonical in _BENCHMARK_ONLY_CANONICALS:
        return MiningRole.DIAGNOSTIC

    # 5. Cross-section-constant / group-constant states (interaction only).
    role_field = _role_field(catalog)
    if role_field in ("group_state", "global_state"):
        return MiningRole.GROUP_STATE if role_field == "group_state" else MiningRole.GLOBAL_STATE

    # 5. Grain: minute → daily is a legal GrainTransform, not a shape violation.
    if str(catalog.get("input_grain") or "").lower() == "minute":
        return MiningRole.INTRADAY_EOD
    operator_name = canonical.lower()
    if operator_name.startswith("intra_") or canonical in {
        "intraday_activity_duration_curvature", "intraday_impact_decay_rate",
    }:
        return MiningRole.INTRADAY_EOD

    # 6. Fundamental PIT lane.
    scope = str(catalog.get("scope") or "")
    if scope == "fundamental_period" or operator_name.startswith(_FUNDAMENTAL_PREFIXES):
        return MiningRole.FUNDAMENTAL_PIT

    # 7. State / condition / event family (recursive rule kernels and literal
    #    -1/0/1 state outputs).  These are legal non-terminal grammar slots.
    if canonical in _STATE_LITERAL_OPS or operator_name in (
        "ts_transition_count", "ts_time_since_change",
    ):
        return MiningRole.STATE if canonical not in _EVENT_OP_HINTS else MiningRole.EVENT

    # 8. Boolean/event-producing names.
    if canonical in _EVENT_OP_HINTS:
        return MiningRole.EVENT
    if operator_name.startswith(("event_", "index_entry_", "index_member")):
        return MiningRole.EVENT

    # 9. Recursive full-history replay operators are still usable factors —
    #    they just cost more (full replay), so they land in the high-cost lane
    #    until checkpoint-backed incremental execution is certified.
    try:
        from cleaned_operators.production_hardening import FULL_HISTORY_REPLAY_CANONICALS

        if canonical in FULL_HISTORY_REPLAY_CANONICALS:
            return MiningRole.ALPHA_HIGH_COST
    except Exception:
        pass

    # 10. Default: stock-level alpha (terminal-eligible).
    return MiningRole.ALPHA


_EVENT_OP_HINTS = frozenset(
    {
        "event_refractory", "cross_event", "event_count", "event_spacing",
        "event_cumulative_return_past", "event_abnormal_return_past",
        "index_entry_exit_event", "limit_up_close", "limit_down_open",
        "ashare_limit_touch", "ashare_limit_one_price", "ashare_limit_failed",
    }
)

# In-sample regression / model diagnostics.  The model is fitted over the same
# window it scores, so ``*_resid`` / ``*_coeff`` / ``*_r2`` are diagnostics, not
# causal alpha.  The causal forms (``*_prior`` / ``*_forecast_error`` /
# ``*_forecast_error_z``) are the mineable equivalents and stay ALPHA.
_DIAGNOSTIC_IN_SAMPLE_CANONICALS = frozenset(
    {
        "ts_multi_regression_coeff",
        "ts_multi_regression_resid",
        "ts_multi_regression_resid_z",
        "ts_multi_regression_r2",
        "ts_huber_regression_coeff",
        "ts_huber_regression_resid",
        "ts_huber_regression_resid_z",
        "ts_ridge_regression_coeff",
        "ts_ridge_regression_resid",
        "ts_ridge_regression_resid_z",
        "ts_ar_forecast",
        "ts_ar_innovation",
        "ts_ar_innovation_z",
        "ts_expectile_regression_coeff",
        "ts_expectile_regression_resid",
        "ts_quantile_regression_coeff",
        "ts_quantile_regression_resid",
    }
)

# Benchmark forms whose ordinary version includes the market (self-inclusion
# bias).  Only the ``*_ex_self`` counterparts are mined.
_BENCHMARK_ONLY_CANONICALS = frozenset(
    {
        "cs_beta_to_market", "cs_alpha_to_market", "rolling_beta_to_market",
    }
)


def _checkpoint_flags(canonical: str) -> tuple[bool, bool, bool]:
    """(stateful, checkpoint_supported, full_history_replay_required)."""
    stateful = False
    checkpoint_supported = False
    full_replay = False
    try:
        from runtime.execution_contract import execution_contract

        contract = execution_contract(canonical)
        state_model = str(getattr(contract, "state_model", "stateless"))
        chunking = str(getattr(contract, "chunking", "independent"))
        stateful = state_model not in ("stateless", "unknown")
        checkpoint_supported = chunking == "checkpoint"
        full_replay = chunking in ("required_full_history", "unknown")
    except Exception:
        pass
    return stateful, checkpoint_supported, full_replay


def _missing_policy(canonical: str) -> str | None:
    try:
        from cleaned_operators.operator_policy import infer_operator_policy
        from cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(operator, canonical=canonical)
        return str(getattr(policy, "nan_policy", "") or "") or None
    except Exception:
        return None


def _input_semantic_types(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    fields = catalog.get("input_fields")
    if isinstance(fields, (list, tuple, frozenset, set)):
        return tuple(sorted(str(f) for f in fields))
    units = catalog.get("input_units")
    if isinstance(units, (list, tuple, frozenset, set)):
        return tuple(sorted(str(u) for u in units))
    return ()


def _searchable_params(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    from cleaned_operators.base import effective_param_role

    out: list[str] = []
    param_names = tuple(catalog.get("param_names") or ())
    specs = catalog.get("param_specs")
    for name in param_names:
        spec = None
        if isinstance(specs, dict):
            spec = specs.get(name)
        role = None
        try:
            role = effective_param_role(spec)
        except Exception:
            role = None
        if role is None or str(role).lower() in (
            "estimator_resolution", "numerical", "missing_policy", "market_policy",
            "policy",
        ):
            continue  # dead / non-economic knobs are not searched
        out.append(name)
    return tuple(out)


def _sources_available(canonical: str, catalog: dict[str, Any], available_sources: Iterable[str]) -> tuple[str, ...]:
    """Intersect required sources with the caller-declared available set."""
    available = frozenset(str(s) for s in (available_sources or ()))
    if not available:
        available = _KNOWN_SOURCES  # unknown → assume everything (admission planning)
    required = _required_sources(canonical, catalog)
    return tuple(sorted(s for s in required if s in available))


def mining_eligible(
    canonical: str,
    *,
    catalog: dict[str, Any] | None = None,
    role: MiningRole | None = None,
    available_sources: Iterable[str] | None = None,
    target_frequency: str | None = None,
) -> bool:
    """Machine eligibility: certified AND role-admissible AND source/freq OK.

    This is the single authority for ``mining_eligible`` — nothing else may
    claim an operator is mineable.
    """
    catalog = dict(catalog) if catalog is not None else _catalog_record(canonical)
    role = role or assign_mining_role(canonical, catalog)
    if role not in _MINEABLE_ROLES:
        return False
    if not bool(catalog.get("production_certified")):
        return False
    # Source availability (when the caller restricts it).
    required = _required_sources(canonical, catalog)
    available = frozenset(str(s) for s in (available_sources or ()))
    if available and not set(required).issubset(available):
        return False
    # Frequency contract: intraday-eod operators need minute source; a caller
    # targeting daily with no minute source must not receive them.
    if target_frequency is not None:
        target = str(target_frequency).lower()
        if role is MiningRole.INTRADAY_EOD and target != "daily":
            return False
        if target == "minute" and role is not MiningRole.INTRADAY_EOD:
            return False
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmissionQuery:
    roles: tuple[MiningRole, ...] = tuple(sorted(_MINEABLE_ROLES, key=lambda r: r.value))
    max_cost: int | None = None
    available_sources: tuple[str, ...] = ()
    target_frequency: str | None = None
    market: str | None = None
    admission: str = "eligible"  # eligible | pending | all


def get_mining_operators(
    *,
    available_sources: Iterable[str] | None = None,
    target_frequency: str | None = None,
    market: str | None = None,
    max_cost: int | None = None,
    roles: Iterable[str | MiningRole] | None = None,
    admission: str = "eligible",
) -> list[MiningOperator]:
    """Return every operator the automated mining layer may use.

    ``admission``:
      * ``eligible`` (default) — ``production_certified`` only.  This is the
        exact AlphaProbe / AlphaMiner / cold-start search space.
      * ``pending`` — also include reviewed targets that are role-admissible
        and PIT-safe but not yet evidence-certified (blockers are the missing
        gates); used by the promote_or_delete planning audit.
      * ``all`` — every registered canonical (admission matrix view).

    Returns a stable list sorted by (role, canonical).
    """
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    mode = str(admission or "eligible").strip().lower()
    if mode not in {"eligible", "pending", "all"}:
        raise ValueError("admission must be eligible, pending, or all")

    role_filter: set[MiningRole] | None = None
    if roles is not None:
        role_filter = set()
        for value in roles:
            role_filter.add(value if isinstance(value, MiningRole) else MiningRole(str(value).lower()))

    sources = tuple(str(s) for s in (available_sources or ()))
    max_cost = int(max_cost) if max_cost is not None else None
    freq = str(target_frequency).lower() if target_frequency else None
    market = str(market).lower() if market else None

    from research_tools.registry import ResearchToolRegistry

    research_tools = set(ResearchToolRegistry.list_canonical())
    out: list[MiningOperator] = []
    for canonical in sorted(OperatorRegistry._catalog):
        if canonical in research_tools:
            continue  # diagnostics / research tools are never mined
        catalog = OperatorRegistry._catalog[canonical]
        role = assign_mining_role(canonical, catalog)
        if role_filter is not None and role not in role_filter:
            continue
        certified = bool(catalog.get("production_certified"))
        stateful, checkpoint, full_replay = _checkpoint_flags(canonical)
        cost = cost_tier(canonical, catalog)
        if max_cost is not None and cost > max_cost:
            continue
        required = _required_sources(canonical, catalog)
        available = _sources_available(canonical, catalog, sources)
        eligible = mining_eligible(
            canonical,
            catalog=catalog,
            role=role,
            available_sources=sources,
            target_frequency=freq,
        )
        if mode == "eligible" and not eligible:
            continue
        if mode == "pending" and role not in _MINEABLE_ROLES:
            continue
        if mode == "pending" and not (certified or catalog.get("pit_safe")):
            continue
        ast_positions = _ROLE_AST_POSITIONS.get(role, ())
        from cleaned_operators.operator_surface import classify_canonical

        out.append(
            MiningOperator(
                canonical=canonical,
                role=role,
                mining_eligible=eligible,
                production_certified=certified,
                authoring_tier=classify_canonical(canonical),
                lifecycle_status=str(catalog.get("lifecycle_status") or catalog.get("status") or "unknown"),
                cost_tier=cost,
                required_sources=required,
                available_sources=available,
                input_grain=catalog.get("input_grain"),
                output_grain=catalog.get("output_grain"),
                input_semantic_types=_input_semantic_types(canonical, catalog),
                output_unit=catalog.get("output_unit"),
                scope=str(catalog.get("scope") or "unknown"),
                searchable_params=_searchable_params(canonical, catalog),
                allowed_ast_positions=ast_positions,
                terminal_allowed=role not in _NON_TERMINAL_ROLES,
                stateful=stateful,
                checkpoint_supported=checkpoint,
                incremental_supported=checkpoint,
                full_history_replay_required=full_replay,
                missing_policy=_missing_policy(canonical),
            )
        )
    out.sort(key=lambda m: (m.role.value, m.canonical))
    return out


def mining_role_manifest() -> dict[str, MiningRole]:
    """canonical → role for every registered canonical (admission matrix input)."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    return {
        canonical: assign_mining_role(canonical, OperatorRegistry._catalog[canonical])
        for canonical in sorted(OperatorRegistry._catalog)
    }


def summary_counts(
    operators: Sequence[MiningOperator],
) -> dict[str, int]:
    from collections import Counter

    counts: dict[str, int] = Counter()
    for op in operators:
        counts["total"] += 1
        counts[op.role.value] += 1
        if op.mining_eligible:
            counts["mining_eligible"] += 1
    return dict(counts)
