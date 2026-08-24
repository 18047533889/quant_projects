# -*- coding: utf-8 -*-
"""Single discovery authority for automated factor mining (R15 fail-closed).

The mining layer (AlphaProbe / AlphaMiner / cold-start) must never guess which
operators are usable by poking at the raw ``DAILY_CANONICALS`` /
``EXTENDED_ONLY_CANONICALS`` / ``RESEARCH_ONLY_CANONICALS`` surface partitions.
Every registered canonical is classified once into a :class:`MiningRole` (with a
:class:`RoleSource` proving where the role came from) plus a machine
``mining_eligible`` flag, and the one public entry point is
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
* ``UNRESOLVED`` — R15 fail-closed: a registered canonical that matches no
  verified rule.  It is NEVER silently promoted to ``ALPHA`` (the pre-R15
  default); the final freeze requires ``UNRESOLVED == 0``.

``mining_eligible`` is a *machine* field, not an opinion: it requires
``production_certified`` (all six gates) AND a role that admits mining AND
available sources AND a matching target frequency AND a declared cost contract.
The invariant ``MINING_ELIGIBLE_WITHOUT_CERTIFICATION == ∅`` is guaranteed by
construction: ``get_mining_operators`` never returns an uncertified operator
unless the caller explicitly requests ``admission="pending"`` for planning
purposes.
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
    # R15-INC-001: the pre-R15 fallback silently promoted any unclassified
    # registered canonical to ALPHA.  Nothing falls through to ALPHA now; an
    # operator that matches no verified rule is explicitly UNRESOLVED and is
    # never a mining candidate until a human/rule assigns it a real role.
    UNRESOLVED = "unresolved"


class RoleSource(str, Enum):
    """Where a role came from.  The final freeze requires no FALLBACK roles."""

    EXPLICIT = "explicit"            # declared metadata role / scope field
    VERIFIED_RULE = "verified_rule"  # reviewed surface / grain / family mapping
    HEURISTIC = "heuristic"          # name/prefix lint — research diagnostics only
    FALLBACK = "fallback"            # catch-all — banned from production roles


class AdmissionState(str, Enum):
    """Role (grammar slot) and admission (can we mine it NOW?) are orthogonal.

    R15-INC-024: a source-blocked operator keeps its factor role (the admission
    matrix can say exactly which blocker applies) but never enters the eligible
    pool — the role and the state are separate fields.
    """

    ELIGIBLE = "eligible"
    PENDING = "pending"
    SOURCE_BLOCKED = "source_blocked"
    DENIED = "denied"
    UNRESOLVED = "unresolved"


class ExecutionModel(str, Enum):
    """R15-INC-017/250: incremental support is an execution property, not a
    synonym for checkpointing.  A stateless bounded-window operator replays a
    warmup chunk; a stateful one needs a checkpoint to resume."""

    INDEPENDENT_WITH_WARMUP = "independent_with_warmup"
    CHECKPOINT = "checkpoint"
    FULL_HISTORY = "full_history"
    UNKNOWN = "unknown"


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

# R15-INC-016/221: the ONLY roles that may ever be a standalone factor terminal.
# Everything else — including STATE/EVENT/GROUP/GLOBAL interaction slots and all
# supporting roles — must be False for ``terminal_allowed``.  The pre-R15 code
# used an exclusion list (``role not in _NON_TERMINAL_ROLES``) which let
# INTERNAL/DIAGNOSTIC/RESEARCH/LEGACY/DENIED fall through as terminal.
_TERMINAL_ROLES = frozenset(
    {
        MiningRole.ALPHA,
        MiningRole.ALPHA_HIGH_COST,
        MiningRole.INTRADAY_EOD,
        MiningRole.FUNDAMENTAL_PIT,
    }
)

# AST grammar positions allowed per role.  The mining grammar reads these.
# Supporting roles have EMPTY positions (R15-INC-225).
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
    MiningRole.UNRESOLVED: (),
}


@dataclass(frozen=True)
class SourceRequirement:
    """One required source with its PIT/concept semantics.

    R15-INC-005/006: grain is NOT a source proxy — a ``daily``-shaped operator
    may still depend on a dedicated PIT source (shareholder / relation / event /
    index).  An operator may carry several SourceRequirements in an AND group
    (price + financial, minute + daily rule, ...); all must be satisfied.
    """

    source_id: str
    required_concepts: tuple[str, ...] = ()
    pit_mode: str | None = None  # asof | point_in_time | snapshot
    note: str = ""


class TargetFrequency(str, Enum):
    """R16-018: the ONLY legal target frequencies.  A free string
    (``"weekly/foo"``) silently failing open is a contract bug — an unknown
    frequency must raise, not be matched loosely."""

    DAILY = "daily"
    MINUTE = "minute"
    FUNDAMENTAL_PERIOD = "fundamental_period"


def bind_target_frequency(value: str | TargetFrequency | None) -> TargetFrequency | None:
    """Strict binder (R16-018): unknown strings raise ValueError."""
    if value is None or isinstance(value, TargetFrequency):
        return value
    s = str(value).strip().lower()
    try:
        return TargetFrequency(s)
    except ValueError:
        raise ValueError(
            f"target_frequency must be one of {[e.value for e in TargetFrequency]}, "
            f"got {value!r}"
        ) from None


@dataclass(frozen=True)
class MiningContext:
    """R16-017: ONE eligibility context object.

    market / frequency / source / source-capability / session / cost all enter
    the SAME ``mining_eligible`` decision — the old design filtered market only
    in the outer ``get_mining_operators``, so a direct ``mining_eligible`` call
    could admit an A-share-only operator into a US search space.
    """

    market: str | None = None
    target_frequency: TargetFrequency | None = None
    available_sources: tuple[str, ...] | None = None
    source_capabilities: dict[str, SourceCapability] | None = None
    max_cost: int | None = None


@dataclass(frozen=True)
class SourceCapability:
    """R16-014: what a source actually provides in a given environment.

    ``source_id`` existing is NOT enough — ``fundamental_pit`` may exist while
    the required depreciation / revision-vintage / analyst-target concepts do
    not.  ``source_status`` performs a required-concepts subset check against
    this capability when a caller supplies capabilities; an id-only legacy
    context cannot prove concepts and is treated as UNKNOWN for that source.
    """

    source_id: str
    concepts: tuple[str, ...] = ()
    grain: str | None = None
    pit_mode: str | None = None          # asof | point_in_time | snapshot
    vintage_support: bool = False
    market: str | None = None


@dataclass(frozen=True)
class SourceStatus:
    """R15-INC-022/004: an environment has *unknown* capability until the caller
    declares one.  ``unknown=True`` means the caller supplied no context (never
    interpreted as "everything available").  ``missing`` lists what a real
    context would have to provide."""

    required: tuple[str, ...]
    satisfied: tuple[str, ...]
    missing: tuple[str, ...]
    unknown: bool = False
    # R16-014: concept gaps, e.g. ``fundamental_pit`` present but no
    # ``depreciation`` concept -> SOURCE_CONCEPT_MISSING.
    concept_gap: tuple[str, ...] = ()


@dataclass(frozen=True)
class MiningOperator:
    canonical: str
    role: MiningRole
    role_source: RoleSource
    admission_state: AdmissionState
    mining_eligible: bool
    production_certified: bool
    authoring_tier: str
    lifecycle_status: str
    cost_tier: int
    cost_contract_declared: bool
    required_sources: tuple[str, ...]
    available_sources: tuple[str, ...]
    source_unknown: bool
    input_grain: str | None
    output_grain: str | None
    input_semantic_types: tuple[str, ...]
    output_unit: str | None
    scope: str
    searchable_params: tuple[str, ...]
    allowed_ast_positions: tuple[str, ...]
    terminal_allowed: bool
    stateful: bool
    execution_model: str
    checkpoint_supported: bool
    incremental_supported: bool
    full_history_replay_required: bool
    missing_policy: str | None
    blockers: tuple[str, ...] = field(default=())
    required_actions: tuple[str, ...] = field(default=())
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Source / frequency / market knowledge
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

# R15-INC-002: the ``market`` argument is a real filter now, not a parsed-but-
# unused API.  Known markets and which operators are market-specific.
_KNOWN_MARKETS = frozenset({"ashare", "us"})

# A-share official-session / limit / status operators.  ``ashare_``-prefixed
# names are an explicit prefix contract (the A-share state machine), not a
# heuristic on arbitrary factor names — they cannot exist in a US context.
_ASHARE_ONLY_CANONICALS = frozenset(
    {
        "limit_up_close", "limit_down_open", "limit_up_open",
        "limit_down_close", "limit_up_distance", "limit_down_distance",
        "limit_up_volume_ratio", "limit_down_volume_ratio",
        "ashare_limit_up_streak", "ashare_limit_down_streak",
        "ashare_days_since_limit_up", "ashare_days_since_limit_down",
        "ashare_limit_touch_count", "ashare_failed_limit_count",
        "ashare_one_price_limit_streak", "ashare_limit_event_density",
        "ashare_limit_asymmetry", "ashare_suspension_episode_length",
        "ashare_limit_open_up_streak", "ashare_limit_open_down_streak",
        "ashare_tradable_state", "ashare_limit_distance",
        "ashare_limit_up_touch", "ashare_limit_down_touch",
        "ashare_limit_one_price", "ashare_limit_failed",
        "ashare_open_at_upper_limit", "ashare_limit_open_failed",
        "ashare_limit_touch", "ashare_status",
        # R22-118..119: minute-limit and suspension families are A-share-only
        # (A-share has a certified minute source; US has none).  Explicitly
        # declaring the market stops ``market_support`` from failing closed to
        # ``()`` for these specialized names — they are ashare-context direct
        # alphas, never a global DELETE (contextual availability, R22-070).
        "intra_limit_first_hit_time", "intra_limit_duration",
        "intra_limit_reopen_count", "intra_limit_pre_hit_pressure_profile",
        "suspension_frequency", "suspension_status_coverage",
    }
)


def market_support(canonical: str, catalog: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Markets a canonical is valid in.

    R16-019: metadata is authoritative first (``supported_markets`` /
    ``market_semantics="agnostic"``).  A-share state/limit/suspension operators
    are ashare-only.  A SPECIALIZED operator (limit/status/suspension-family)
    with NO market declaration fails CLOSED (empty support) — it must never be
    defaulted into both markets.  Generic math operators are market-agnostic by
    nature and keep the two-market default.
    """
    catalog = catalog or {}
    declared = catalog.get("supported_markets")
    if isinstance(declared, (list, tuple, frozenset, set)) and declared:
        return tuple(sorted(str(m) for m in declared))
    if catalog.get("market_semantics") == "agnostic":
        return ("ashare", "us")
    name = canonical.lower()
    if name.startswith("ashare_") or canonical in _ASHARE_ONLY_CANONICALS:
        return ("ashare",)
    if _looks_market_specialized(name):
        # R16-019: specialized-but-undeclared -> fail closed (no market).
        return ()
    return ("ashare", "us")


def _looks_market_specialized(name: str) -> bool:
    """A-share-specific state/limit/suspension family names.  These cannot exist
    in a US context, so without an explicit market declaration they fail closed
    instead of defaulting to both markets (R16-019)."""
    return name.startswith(("limit_", "ashare_", "suspension_")) or any(
        token in name for token in ("_limit_", "_suspension", "one_price_limit")
    )


def validate_market(market: str | None) -> str | None:
    """R15-INC-002: an unknown market fails closed instead of being ignored."""
    if market is None:
        return None
    value = str(market).strip().lower()
    if value not in _KNOWN_MARKETS:
        raise ValueError(
            f"unknown market {market!r}; known markets: {sorted(_KNOWN_MARKETS)}"
        )
    return value


# R15-INC-014: continuous measures *of* state are numeric alphas, NOT Bool/
# discrete state slots.  The pre-R15 ``_STATE_LITERAL_OPS`` lumped
# ``state_episode_mfe`` / ``threshold_cycle_period`` / ``candle_gap_atr`` in with
# true discrete state machines and permanently banned them from the terminal
# position.  These are per-instrument numeric statistics and stay ALPHA.
_CONTINUOUS_STATE_STATISTICS = frozenset(
    {
        "state_episode_mfe", "state_episode_mae", "state_episode_efficiency",
        "state_episode_retrace_ratio", "state_episode_excursion_balance",
        "ts_threshold_cycle_period", "ts_threshold_cycle_asymmetry",
        "ts_cumulative_deviation_score", "ts_interval_nesting_depth",
        "candle_gap_atr", "event_count", "event_spacing",
        "ts_state_integral", "ts_state_entry_strength", "ts_transition_intensity",
        "ts_hysteresis_state", "ts_hysteresis_age", "ts_state_age_percentile",
        "ts_state_exit_hazard", "ts_state_residual_life", "ts_cusum_pressure",
        "ts_recovery_fraction",
    }
)

# R15-INC-015: continuous event statistics (``event_interval_memory`` /
# ``event_fano_factor`` / ``event_local_variation`` ...) are numeric series and
# must NOT be classified as EVENT masks.  The pre-R15 ``event_``-prefix rule
# banned every ``event_*`` name from the terminal position.
_CONTINUOUS_EVENT_STATISTICS = frozenset(
    {
        "event_interval_memory", "event_local_variation",
        "event_fano_factor", "event_fano_excess", "event_fano_ratio",
        "event_interval_mad", "event_interval_cv", "event_age",
        "event_level_survival_share", "event_cumulative_return_past",
        "event_abnormal_return_past", "event_interval_series",
        "event_spacing_mean", "event_spacing_cv",
        "marked_event_*",  # wildcard placeholder removed below if unregistered
    }
)
_CONTINUOUS_EVENT_STATISTICS = frozenset(
    n for n in _CONTINUOUS_EVENT_STATISTICS if "*" not in n
)

# True 0/1/NaN event masks (strict EventBool) — the ONLY names classified EVENT.
_EVENT_BOOL_OPS = frozenset(
    {
        "event_refractory", "cross_event", "index_entry_exit_event",
        "index_member", "index_member_status",
        "limit_up_close", "limit_down_open",
        "ashare_limit_touch", "ashare_limit_one_price", "ashare_limit_failed",
        "event_touch", "event_cross", "event_flag",
    }
)

# True discrete state machines (Bool / discrete state), legal non-terminal gates.
_STATE_LITERAL_OPS = frozenset(
    {
        "state_latch", "state_hold", "state_slew_limit", "state_deadband",
        "state_ewm_if", "state_since_reduce", "event_refractory",
        "cross_event", "directional_change_state", "state_since_trend_tstat",
        "trade_when", "digital_count", "limit_up_close", "limit_down_open",
        "ashare_limit_touch", "ashare_limit_failed",
    }
)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _catalog_record(canonical: str) -> dict[str, Any]:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry._catalog.get(canonical) or {}


def _surface_of(canonical: str, catalog: dict[str, Any]) -> str:
    surface = str(catalog.get("surface") or "").strip().lower()
    if surface:
        return surface
    try:
        from factor_engine.cleaned_operators.operator_surface import classify_canonical

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
    # Fallback by declared cost_model (a callable contract), then category.
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        metadata = getattr(operator, "metadata", None)
        if getattr(metadata, "cost_model", None) is not None:
            return 4  # a declared cost model implies non-trivial computation
    except Exception:
        pass
    category = str(catalog.get("category") or "").lower()
    if "spectral" in category or "topology" in category or "kernel" in category:
        return 5
    if "matrix" in category or "wavelet" in category or "ssa" in category:
        return 5
    # R15-INC-011: an operator with NO declared cost contract must not silently
    # land in the cheap lane.  The tier is still estimated for reporting, but
    # ``cost_contract_declared`` distinguishes estimate from contract.
    return 1


def cost_contract_declared(canonical: str, catalog: dict[str, Any]) -> bool:
    """R15-INC-011: does the canonical carry an explicit cost contract (a
    ``cost:`` tag or a ``cost_model`` callable)?  A missing contract blocks
    eligibility rather than defaulting to the cheap lane."""
    tags = tuple(str(tag) for tag in (catalog.get("tags") or ()))
    if any(re.fullmatch(r"cost:\d+", tag.strip()) for tag in tags):
        return True
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        metadata = getattr(operator, "metadata", None)
        return getattr(metadata, "cost_model", None) is not None
    except Exception:
        return False


def _required_sources(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    """R15-INC-005/006: explicit source requirements per canonical.

    Priority (R16-015): catalog-declared ``source_requirements`` (typed
    ``SourceRequirement`` list on the operator metadata) is authoritative; then
    the legacy migration map (``_MULTI_SOURCE_REQUIREMENTS``); then the
    grain-derived / family default.  R16-013: ``index_`` operators resolve to
    ``index_pit`` — the vocabulary already declares it — never a broad
    ``relation_pit`` name guess.  holder/shareholder/relation/event/index each
    have their OWN typed source.
    """
    declared = catalog.get("source_requirements")
    if isinstance(declared, (list, tuple)) and declared:
        return tuple(sorted({s.source_id for s in declared if hasattr(s, "source_id")}))
    explicit = _MULTI_SOURCE_REQUIREMENTS.get(canonical)
    if explicit:
        return tuple(sorted({s.source_id for s in explicit}))
    grain = str(catalog.get("input_grain") or "").lower()
    if grain:
        return (_SOURCE_OF_GRAIN.get(grain, grain + "_bar"),)
    operator_name = canonical.lower()
    if operator_name.startswith(("intra_",)) or canonical in {
        "intraday_activity_duration_curvature", "intraday_impact_decay_rate",
        "session_event_recovery_score", "session_recovery_time",
    }:
        return ("minute_bar",)
    if operator_name.startswith(_FUNDAMENTAL_PREFIXES) or str(
        catalog.get("scope") or ""
    ) == "fundamental_period":
        return ("fundamental_pit",)
    # R16-013: typed per-family source — an ``index_*`` operator needs the index
    # PIT source, never a relation/name-guess fallback.
    if operator_name.startswith(("holder_", "shareholder_")):
        return ("shareholder_pit",)
    if operator_name.startswith("index_"):
        return ("index_pit",)
    if operator_name.startswith("relation_"):
        return ("relation_pit",)
    if operator_name.startswith("event_"):
        return ("event_pit",)
    return ("daily_bar",)


# R15-INC-006: operators needing more than one source simultaneously.
_MULTI_SOURCE_REQUIREMENTS: dict[str, tuple[SourceRequirement, ...]] = {
    # price level + fundamentals for value/quality composite shapes
    "pe_ttm": (
        SourceRequirement("daily_bar", ("price",)),
        SourceRequirement("fundamental_pit", ("eps", "pe")),
    ),
    "pb_ratio": (
        SourceRequirement("daily_bar", ("price",)),
        SourceRequirement("fundamental_pit", ("book_value",)),
    ),
}


def source_status(
    canonical: str,
    catalog: dict[str, Any],
    available_sources: Iterable[str] | None,
    *,
    source_capabilities: dict[str, SourceCapability] | None = None,
) -> SourceStatus:
    """R15-INC-004: distinguish UNKNOWN / explicit-empty / explicit-set.

    ``None`` (no caller context) is UNKNOWN — the planning layer may only mark
    it unknown, never assume every source is available.  An explicit empty set
    means nothing is available.  A non-empty set is intersected with the AND
    requirements.

    R16-014: when ``source_capabilities`` is supplied, source_id membership is
    NOT enough — each required source's ``required_concepts`` must be a subset
    of the capability's concepts, or the source lands in ``concept_gap``
    (SOURCE_CONCEPT_MISSING) even though the source exists.
    """
    required = _required_sources(canonical, catalog)
    # concept requirements per required source
    required_concepts: dict[str, tuple[str, ...]] = {}
    declared = catalog.get("source_requirements")
    if isinstance(declared, (list, tuple)):
        for s in declared:
            if hasattr(s, "source_id"):
                required_concepts[s.source_id] = tuple(
                    getattr(s, "required_concepts", None) or ()
                )
    else:
        for s in _MULTI_SOURCE_REQUIREMENTS.get(canonical, ()):
            required_concepts[s.source_id] = tuple(s.required_concepts)
    if available_sources is None:
        return SourceStatus(required, (), (), unknown=True)
    avail = frozenset(str(s) for s in available_sources)
    if not avail:
        return SourceStatus(required, (), required, unknown=False)
    satisfied = []
    missing = []
    concept_gap = []
    for s in required:
        if s not in avail:
            missing.append(s)
            continue
        # source present — verify concepts when the caller declared capabilities
        concepts_needed = required_concepts.get(s, ())
        if source_capabilities is not None and concepts_needed:
            cap = source_capabilities.get(s)
            if cap is None:
                # source id listed but no capability declared -> cannot prove
                concept_gap.append(s)
                continue
            cap_concepts = frozenset(cap.concepts)
            if not set(concepts_needed) <= cap_concepts:
                concept_gap.append(s)
                continue
        satisfied.append(s)
    return SourceStatus(tuple(required), tuple(sorted(satisfied)),
                        tuple(sorted(missing)), unknown=False,
                        concept_gap=tuple(sorted(concept_gap)))


def _sources_available(canonical: str, catalog: dict[str, Any], available_sources: Iterable[str]) -> tuple[str, ...]:
    """Backward-compat: the satisfied subset (empty for unknown/empty context)."""
    return source_status(canonical, catalog, available_sources).satisfied


def assign_mining_role(canonical: str, catalog: dict[str, Any] | None = None) -> MiningRole:
    """Deterministic, fail-closed role assignment (backward-compatible wrapper)."""
    return assign_mining_role_ex(canonical, catalog)[0]


def _role_from_direct_status(status: Any) -> MiningRole | None:
    """R22-009..013: map a DIRECT_* DirectUse verdict to the corresponding
    MiningRole so the authoring tier (surface) and the semantic role stay
    orthogonal.  DIRECT_INTERMEDIATE / DIRECT_RECIPE / DIRECT_CONTROL_FLOW map to
    ALPHA (their ast positions and terminal authority come from DirectUse)."""
    from factor_engine.mining.direct_use import DirectUseStatus

    mapping = {
        DirectUseStatus.DIRECT_ALPHA: MiningRole.ALPHA,
        DirectUseStatus.DIRECT_ALPHA_HIGH_COST: MiningRole.ALPHA_HIGH_COST,
        DirectUseStatus.DIRECT_STATE: MiningRole.STATE,
        DirectUseStatus.DIRECT_CONDITION: MiningRole.CONDITION,
        DirectUseStatus.DIRECT_EVENT: MiningRole.EVENT,
        DirectUseStatus.DIRECT_GROUP_STATE: MiningRole.GROUP_STATE,
        DirectUseStatus.DIRECT_GLOBAL_STATE: MiningRole.GLOBAL_STATE,
        DirectUseStatus.DIRECT_INTERMEDIATE: MiningRole.ALPHA,
        DirectUseStatus.DIRECT_SOURCE_TRANSFORM: MiningRole.SOURCE_TRANSFORM,
        DirectUseStatus.DIRECT_RECIPE: MiningRole.ALPHA,
        DirectUseStatus.DIRECT_CONTROL_FLOW: MiningRole.ALPHA,
    }
    return mapping.get(status)


def assign_mining_role_ex(
    canonical: str, catalog: dict[str, Any] | None = None
) -> tuple[MiningRole, RoleSource]:
    """R15-INC-001/013/014/015/016/027: role + proof of origin.

    Order matters and every rule is an exclusion before it is a promotion: the
    permanently-forbidden / non-factor / internal / diagnostic classes always
    win over "looks like an alpha".  The fallback is ``UNRESOLVED`` (never the
    pre-R15 silent ``ALPHA``).  Names/prefixes are lint-only — the authoritative
    signals are the reviewed authoring surface, declared grain and explicit
    role/scope fields.
    """
    catalog = dict(catalog) if catalog is not None else _catalog_record(canonical)
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    registered = canonical in OperatorRegistry._catalog
    if not registered:
        try:
            from research_tools.registry import ResearchToolRegistry

            if canonical in ResearchToolRegistry.list_canonical():
                return MiningRole.DIAGNOSTIC, RoleSource.EXPLICIT
        except Exception:
            pass
        return MiningRole.INTERNAL, RoleSource.VERIFIED_RULE

    surface = _surface_of(canonical, catalog)

    # 1. Permanently forbidden / non-factor / unsafe / internal / legacy.
    try:
        from factor_engine.cleaned_operators.operator_spec import PERMANENTLY_FORBIDDEN_CANONICALS

        if canonical in PERMANENTLY_FORBIDDEN_CANONICALS:
            return MiningRole.DENIED, RoleSource.EXPLICIT
    except Exception:
        pass
    if surface == "unsafe":
        return MiningRole.DENIED, RoleSource.VERIFIED_RULE
    if surface == "internal":
        return MiningRole.INTERNAL, RoleSource.VERIFIED_RULE

    try:
        from factor_engine.cleaned_operators.production_hardening import NON_FACTOR_PRODUCTION_CANONICALS

        if canonical in NON_FACTOR_PRODUCTION_CANONICALS:
            if canonical.startswith(("holder_", "relation_")):
                return MiningRole.SOURCE_TRANSFORM, RoleSource.VERIFIED_RULE
            if canonical == "micro_bvc_vpin":
                return MiningRole.RESEARCH, RoleSource.EXPLICIT  # spec §16: research-only, never mined
            return MiningRole.DENIED, RoleSource.VERIFIED_RULE
    except Exception:
        pass

    if surface == "legacy":
        return MiningRole.LEGACY, RoleSource.VERIFIED_RULE

    # 2. Explicit review flags (role is orthogonal to authoring tier, R15-INC-013).
    if catalog.get("compatibility_only"):
        return MiningRole.INTERNAL, RoleSource.EXPLICIT  # one math == one canonical
    if catalog.get("diagnostic_only") or catalog.get("benchmark_only"):
        return MiningRole.DIAGNOSTIC, RoleSource.EXPLICIT
    if catalog.get("hidden_from_default_mining"):
        return MiningRole.RESEARCH, RoleSource.EXPLICIT

    # 3. In-sample diagnostics / benchmark-only forms — never mined; the
    #    *_prior / *_forecast_error / *_ex_self counterparts are the ALPHA forms.
    if canonical in _DIAGNOSTIC_IN_SAMPLE_CANONICALS:
        return MiningRole.DIAGNOSTIC, RoleSource.VERIFIED_RULE
    if canonical in _BENCHMARK_ONLY_CANONICALS:
        return MiningRole.DIAGNOSTIC, RoleSource.VERIFIED_RULE

    # 4. Explicit role field (R9-OP-024) — machine metadata, not a tag.
    role_field = _role_field(catalog)
    if role_field in ("group_state", "global_state"):
        return (
            MiningRole.GROUP_STATE if role_field == "group_state" else MiningRole.GLOBAL_STATE,
            RoleSource.EXPLICIT,
        )

    # 5. Grain: minute → daily is a legal GrainTransform, not a shape violation.
    if str(catalog.get("input_grain") or "").lower() == "minute":
        return MiningRole.INTRADAY_EOD, RoleSource.VERIFIED_RULE
    operator_name = canonical.lower()
    if operator_name.startswith("intra_") or canonical in {
        "intraday_activity_duration_curvature", "intraday_impact_decay_rate",
        "intraday_wasserstein_pair_distance", "baseline_scaled_wasserstein_distance",
        "intraday_vwap_deviation", "session_event_recovery_score",
        "session_recovery_time",
    }:
        return MiningRole.INTRADAY_EOD, RoleSource.VERIFIED_RULE

    # 6. Fundamental PIT lane.
    scope = str(catalog.get("scope") or "")
    if scope == "fundamental_period" or operator_name.startswith(_FUNDAMENTAL_PREFIXES):
        return MiningRole.FUNDAMENTAL_PIT, RoleSource.VERIFIED_RULE

    # 7. Discrete state / event machines (legal non-terminal slots).  Continuous
    #    state/event STATISTICS are numeric and stay ALPHA via the surface rule.
    if canonical in _STATE_LITERAL_OPS:
        if canonical in _EVENT_BOOL_OPS:
            return MiningRole.EVENT, RoleSource.VERIFIED_RULE
        return MiningRole.STATE, RoleSource.VERIFIED_RULE
    if canonical in _EVENT_BOOL_OPS or canonical in _EVENT_OP_HINTS:
        return MiningRole.EVENT, RoleSource.VERIFIED_RULE

    # 8. Recursive full-history replay operators are still usable factors —
    #    they just cost more (full replay), so they land in the high-cost lane
    #    until checkpoint-backed incremental execution is certified.
    try:
        from factor_engine.cleaned_operators.production_hardening import FULL_HISTORY_REPLAY_CANONICALS

        if canonical in FULL_HISTORY_REPLAY_CANONICALS:
            return MiningRole.ALPHA_HIGH_COST, RoleSource.VERIFIED_RULE
    except Exception:
        pass

    # 9. Reviewed factor-authoring surfaces (daily / extended) → stock-level
    #    alpha.  The surface manifest is version-bound human review
    #    (REVIEWED_MIGRATION_MANIFEST / DAILY_CANONICALS / EXTENDED set), so this
    #    is a VERIFIED mapping, not a name heuristic.
    #    R22-009..012: ``surface == research`` is an AUTHORING/LIFECYCLE tier, not
    #    a semantic role.  A research-surface canonical with an explicit DIRECT_*
    #    DirectUse verdict resolves to that role; only true research-tool / delete
    #    verdicts keep the RESEARCH role.  ``_resolve_explicit`` is a pure name
    #    table lookup, so there is no recursion back into role assignment.
    if surface == "research":
        try:
            from factor_engine.mining.direct_use import _resolve_explicit

            contract = _resolve_explicit(canonical, catalog)
            if contract is not None and contract.status.value.startswith("direct_"):
                role = _role_from_direct_status(contract.status)
                if role is not None:
                    return role, RoleSource.VERIFIED_RULE
        except Exception:
            pass
        return MiningRole.RESEARCH, RoleSource.VERIFIED_RULE
    if surface in ("daily", "extended"):
        return MiningRole.ALPHA, RoleSource.VERIFIED_RULE

    # 10. R15-INC-001: never silently default to ALPHA.
    return MiningRole.UNRESOLVED, RoleSource.FALLBACK


_EVENT_OP_HINTS = frozenset(
    {
        "event_refractory", "cross_event",
        "index_entry_exit_event", "index_member",
        "limit_up_close", "limit_down_open",
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


def _execution_flags(canonical: str) -> tuple[bool, ExecutionModel, bool, bool]:
    """(stateful, execution_model, checkpoint_supported, full_history_replay)."""
    try:
        from factor_engine.runtime.execution_contract import execution_contract

        contract = execution_contract(canonical)
        state_model = str(getattr(contract, "state_model", "stateless"))
        chunking = str(getattr(contract, "chunking", "independent"))
        stateful = state_model not in ("stateless", "unknown")
        if chunking == "checkpoint":
            model = ExecutionModel.CHECKPOINT
        elif chunking in ("required_full_history", "unknown"):
            model = ExecutionModel.FULL_HISTORY
        else:
            model = ExecutionModel.INDEPENDENT_WITH_WARMUP
        checkpoint_supported = chunking == "checkpoint"
        return stateful, model, checkpoint_supported, model is ExecutionModel.FULL_HISTORY
    except Exception:
        return False, ExecutionModel.UNKNOWN, False, False


def _checkpoint_flags(canonical: str) -> tuple[bool, bool, bool]:
    """Backward-compat: (stateful, checkpoint_supported, full_history_replay)."""
    stateful, _model, checkpoint, full_replay = _execution_flags(canonical)
    return stateful, checkpoint, full_replay


def _missing_policy(canonical: str) -> str | None:
    try:
        from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(operator, canonical=canonical)
        return str(getattr(policy, "nan_policy", "") or "") or None
    except Exception:
        return None


def _input_field_concepts(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    """R16-016: declared INPUT FIELD CONCEPTS (field names / concepts), separate
    from semantic types.  A field concept (``close``, ``eps``, ``turnover``) is
    never a SemanticType (``ReturnLike``, ``Volume``).  No fallback between the
    two."""
    fields = catalog.get("input_field_concepts")
    if isinstance(fields, (list, tuple, frozenset, set)):
        return tuple(sorted(str(f) for f in fields))
    legacy = catalog.get("input_fields")
    if isinstance(legacy, (list, tuple, frozenset, set)):
        return tuple(sorted(str(f) for f in legacy))
    return ()


def _input_semantic_types(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    """R16-016: declared INPUT SEMANTIC TYPES ONLY — never field names.

    R15-INC-018 was only half-fixed: returning ``input_fields`` when present
    mixed field names with SemanticTypes in one field.  The two are now separate
    (``input_field_concepts`` vs ``input_semantic_types``) and cannot fall back
    to each other.  A canonical with no type declaration reports an empty tuple,
    NOT a guessed type."""
    declared = catalog.get("input_semantic_types")
    if isinstance(declared, (list, tuple, frozenset, set)):
        return tuple(sorted(str(t) for t in declared))
    return ()


def _searchable_params(canonical: str, catalog: dict[str, Any]) -> tuple[str, ...]:
    """R15-INC-008/009: searchable scalars come from the ResolvedSignature's
    scalar params via ``searchable_param_names`` — the single authority — and
    never from a raw scan of ``param_names`` (which would let panel/context
    objects leak into the search surface).

    R16-020: default mining searches ONLY ECONOMIC / HORIZON / STATE_THRESHOLD /
    MODEL_ORDER.  ``ESTIMATOR_RESOLUTION`` (bins / grid / projections /
    surrogates / ridge) expands the AST without adding economic signal and is
    restricted to an audited preset / robustness lane — it is excluded from the
    default searchable set here."""
    try:
        from factor_engine.cleaned_operators.base import ParamRole, searchable_param_names
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        grades = searchable_param_names(getattr(operator, "metadata", None))
        excluded = {ParamRole.ESTIMATOR_RESOLUTION}
        keep = set(grades.get("full", ())) | set(grades.get("coarse", ()))
        specs = getattr(getattr(operator, "metadata", None), "param_specs", None) or {}
        for name in list(keep):
            spec = specs.get(name)
            if spec is not None and getattr(spec, "param_role", None) in excluded:
                keep.discard(name)
        return tuple(sorted(keep))
    except Exception:
        return ()


def _output_grain(canonical: str, catalog: dict[str, Any], role: MiningRole) -> str:
    """R15-INC-003: separate input_grain from output_grain.  INTRADAY_EOD is
    minute INPUT / daily OUTPUT; ``target_frequency`` must match the OUTPUT
    grain.  Undeclared output grain on a daily-rolling / EOD operator is daily."""
    declared = catalog.get("output_grain")
    if declared:
        return str(declared).lower()
    if role is MiningRole.INTRADAY_EOD or str(
        catalog.get("input_grain") or ""
    ).lower() == "minute":
        return "daily"
    if role is MiningRole.FUNDAMENTAL_PIT:
        return "fundamental_period"
    return "daily"


def mining_eligible(
    canonical: str,
    *,
    catalog: dict[str, Any] | None = None,
    role: MiningRole | None = None,
    available_sources: Iterable[str] | None = None,
    target_frequency: str | None = None,
    context: MiningContext | None = None,
    market: str | None = None,
) -> bool:
    """Machine eligibility: certified AND role-admissible AND source OK AND
    frequency OK AND market OK AND declared cost contract.

    This is the single authority for ``mining_eligible`` — nothing else may
    claim an operator is mineable.  R15-INC-004: ``available_sources=None``
    means UNKNOWN capability → fail closed (never assumed available).
    R16-017: ``market`` is checked HERE (not only in the outer
    ``get_mining_operators``), so a direct call with a market context cannot
    admit an A-share-only operator into a US search space.  ``MiningContext``
    carries market/frequency/source/capability/cost as one object.
    R16-014: a required source whose concepts the environment cannot prove is a
    concept gap — fail closed.
    """
    catalog = dict(catalog) if catalog is not None else _catalog_record(canonical)
    role, _src = (
        assign_mining_role_ex(canonical, catalog)
        if role is None
        else (role, RoleSource.VERIFIED_RULE)
    )
    if role not in _MINEABLE_ROLES:
        return False
    if not bool(catalog.get("production_certified")):
        return False
    # R15-INC-011: no explicit cost contract → not eligible (no cheap default).
    if not cost_contract_declared(canonical, catalog):
        return False

    # --- R16-017: a unified context may override the individual kwargs. ---
    if context is not None:
        ctx_market = context.market
        ctx_freq = context.target_frequency
        ctx_sources = context.available_sources
        ctx_caps = context.source_capabilities
    else:
        ctx_market = market
        ctx_freq = bind_target_frequency(target_frequency)
        ctx_sources = available_sources
        ctx_caps = None

    # Market (R16-017): a specific market mismatch fails closed.
    if ctx_market is not None:
        supported = market_support(canonical, catalog)
        if ctx_market not in supported:
            return False

    # Source availability (R15-INC-004): unknown or incomplete → fail closed.
    status = source_status(canonical, catalog, ctx_sources,
                           source_capabilities=ctx_caps)
    if status.unknown:
        return False
    if status.missing or status.concept_gap:
        return False

    # Frequency contract (R15-INC-003): target matches OUTPUT grain.
    if ctx_freq is not None:
        target = ctx_freq.value if isinstance(ctx_freq, TargetFrequency) else str(ctx_freq).lower()
        output_grain = _output_grain(canonical, catalog, role)
        if output_grain == "minute" and target != "minute":
            return False
        if output_grain in ("daily", "fundamental_period") and target == "minute":
            return False
    return True


# ---------------------------------------------------------------------------
# Blockers / actions (R15-INC-019): every non-eligible entry carries a machine
# reason.  The catalog and the admission matrix share this single decision.
# ---------------------------------------------------------------------------

_BLOCKER_CODES = {
    "B01": "PERMANENT_PIT_UNSAFE",
    "B02": "NON_FACTOR_OPERATOR",
    "B03": "COMPAT_ALIAS",
    "B04": "DIAGNOSTIC_IN_SAMPLE",
    "B05": "BENCHMARK_ONLY",
    "B06": "HIDDEN_FROM_MINING",
    "B08": "IMPLEMENTATION_EVIDENCE_MISSING",
    "B09": "SEMANTIC_GOLDEN_MISSING",
    "B10": "TEMPORAL_PREFIX_MISSING",
    "B11": "SOURCE_PIT_MISSING",
    "B12": "EDGE_CONTRACT_UNDECLARED",
    "B13": "EDGE_EVIDENCE_MISSING",
    "B14": "BACKEND_EVIDENCE_MISSING",
    "B15": "SOURCE_FIELD_MISSING",
    "B18": "STATE_CHECKPOINT_MISSING",
    "B19": "FULL_HISTORY_ONLY",
    "B25": "INSUFFICIENT_EFFECTIVE_SAMPLE",
    "B26": "PARAM_SPACE_UNSAFE",
    "B27": "DEAD_PARAMETER",
    "B28": "SEMANTIC_DUPLICATE",
    "B29": "MATH_DEFINITION_DEFECT",
    "B33": "COST_CONTRACT_MISSING",
    "B34": "ROLE_UNRESOLVED",
    "B35": "MARKET_UNSUPPORTED",
}


def _admission_decision(
    canonical: str,
    catalog: dict[str, Any],
    role: MiningRole,
    role_source: RoleSource,
    source_status_: SourceStatus,
) -> tuple[AdmissionState, tuple[str, ...], tuple[str, ...]]:
    """Compute (state, blocker codes, required actions) — the single shared
    decision the manifest and admission matrix both serialize."""
    blockers: list[str] = []
    actions: list[str] = []
    state: AdmissionState = AdmissionState.PENDING

    # R16-022: NO early exit.  A canonical can carry MULTIPLE independent
    # blockers, and the matrix must keep the FULL precise set (a denied
    # canonical may also be evidence-missing / unresolved; an unresolved role
    # may also have a cost-contract gap).  The terminal state is recorded but
    # evaluation continues so no reason is flattened away.
    if role is MiningRole.DENIED:
        state = AdmissionState.DENIED
        blockers.append("B01")
        actions.append("delete from public registry: permanently-forbidden primitive")
    if role is MiningRole.UNRESOLVED:
        state = AdmissionState.UNRESOLVED
        blockers.append("B34")
        actions.append("assign an explicit MiningRole / output semantic type; UNRESOLVED is never mined")
    if role_source is RoleSource.FALLBACK:
        blockers.append("B34")
        actions.append("replace the fallback role with an explicit/verified classification")
        if state is AdmissionState.PENDING:
            state = AdmissionState.UNRESOLVED
    if role not in _MINEABLE_ROLES and not blockers:
        blockers.append("B02")
        actions.append("supporting layer: never mined as a factor")

    if not bool(catalog.get("production_certified")):
        if state is AdmissionState.PENDING:
            state = AdmissionState.PENDING
        if not catalog.get("implementation_certified"):
            blockers.append("B08"); actions.append("certify implementation evidence")
        if not (catalog.get("semantic_certified") or catalog.get("semantic_golden_verified")):
            blockers.append("B09"); actions.append("certify semantic golden")
        if not (catalog.get("temporal_certified") or catalog.get("temporal_prefix_verified")):
            blockers.append("B10"); actions.append("certify temporal prefix causality")
        if not (catalog.get("source_contract_certified") or catalog.get("source_contract_verified")):
            blockers.append("B11"); actions.append("certify source PIT contract")
        if not catalog.get("edge_case_passed"):
            blockers.append("B13"); actions.append("certify edge-case behaviour")
        if not catalog.get("backend_passed"):
            blockers.append("B14"); actions.append("certify backend evidence")

    # R16-022: a DENIED / UNRESOLVED terminal state is never downgraded by
    # later quality findings — the blockers still accumulate (full precise set),
    # but the terminal admission state stands.
    _TERMINAL = (AdmissionState.DENIED, AdmissionState.UNRESOLVED)
    if source_status_.unknown:
        if state not in _TERMINAL:
            state = AdmissionState.PENDING
        blockers.append("B15"); actions.append("declare available source context (unknown capability is not availability)")
    elif source_status_.missing:
        if state not in _TERMINAL:
            state = AdmissionState.SOURCE_BLOCKED
        blockers.append("B15")
        actions.append(f"provide sources {sorted(source_status_.missing)}")
    elif source_status_.concept_gap:
        # R16-014: source exists but a required concept cannot be proven.
        if state not in _TERMINAL:
            state = AdmissionState.SOURCE_BLOCKED
        blockers.append("B15")
        actions.append(f"provide required concepts for sources {sorted(source_status_.concept_gap)}")

    if not cost_contract_declared(canonical, catalog):
        blockers.append("B33"); actions.append("declare an explicit cost contract (cost: tag or cost_model)")

    # R15-INC-030: a non-terminal role (STATE/EVENT/GROUP/GLOBAL) is a ROUTING
    # decision, not a quality blocker — the consumer routes it to a
    # gate/interaction position.  The admission matrix exposes it as a routing
    # constraint, never as a hard blocker.

    # keep deterministic order, dedupe
    seen: set[str] = set()
    blockers = [b for b in blockers if not (b in seen or seen.add(b))]
    if state is AdmissionState.PENDING and not blockers:
        state = AdmissionState.ELIGIBLE
    return state, tuple(blockers), tuple(actions)


def audit_multi_source_contract(catalog: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """R16-015: every panel_arity>1 operator must carry complete typed source
    requirements.

    A benchmark / daily+fundamental / minute+daily-limit / group-membership /
    relation-exposure operator that is still covered by a single-source
    fallback would silently under-require its data.  This audits each
    multi-panel operator and reports which of its panel inputs have no typed
    SourceConcept binding.  A canonical that relies on the legacy migration map
    is reported as ``legacy_migration`` (accepted for now, not silent).
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    catalog = dict(catalog) if catalog is not None else OperatorRegistry._catalog
    findings: list[dict[str, Any]] = []
    for canonical, entry in sorted(catalog.items()):
        pp = tuple(entry.get("panel_params") or ())
        if len(pp) <= 1:
            continue
        reqs = entry.get("source_requirements")
        src = _required_sources(canonical, entry)
        record = {
            "canonical": canonical,
            "panel_arity": len(pp),
            "panel_params": pp,
            "required_sources": list(src),
            "typed_source_requirements": bool(reqs),
            "legacy_migration_only": canonical in _MULTI_SOURCE_REQUIREMENTS,
        }
        # single-source fallback on a multi-panel operator is a completeness gap
        if not reqs and len(src) <= 1 and canonical not in _MULTI_SOURCE_REQUIREMENTS:
            record["gap"] = "multi-panel operator has no typed multi-source contract"
            findings.append(record)
        else:
            record["gap"] = ""
            findings.append(record)
    return findings


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


def _coerce_max_cost(value: int | None) -> int | None:
    """R15-INC-010: ``max_cost`` must be an exact integer.  ``int(3.9)`` and
    ``int(True)`` silently truncate and make the search config irreproducible."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"max_cost must be an exact integer, got {value!r} "
            "(bool/float config is rejected — no silent truncation)"
        )
    return value


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
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    mode = str(admission or "eligible").strip().lower()
    if mode not in {"eligible", "pending", "all"}:
        raise ValueError("admission must be eligible, pending, or all")

    role_filter: set[MiningRole] | None = None
    if roles is not None:
        role_filter = set()
        for value in roles:
            role_filter.add(value if isinstance(value, MiningRole) else MiningRole(str(value).lower()))

    max_cost = _coerce_max_cost(max_cost)
    # R16-018: strict frequency enum — an unknown string raises.
    freq = bind_target_frequency(target_frequency)
    market = validate_market(market)
    env_sources = tuple(str(s) for s in (available_sources or ())) if available_sources is not None else None

    from research_tools.registry import ResearchToolRegistry

    research_tools = set(ResearchToolRegistry.list_canonical())
    out: list[MiningOperator] = []
    for canonical in sorted(OperatorRegistry._catalog):
        if canonical in research_tools:
            continue  # diagnostics / research tools are never mined
        catalog = OperatorRegistry._catalog[canonical]
        role, role_source = assign_mining_role_ex(canonical, catalog)
        if role_filter is not None and role not in role_filter:
            continue
        certified = bool(catalog.get("production_certified"))
        stateful, execution_model, checkpoint, full_replay = _execution_flags(canonical)
        cost = cost_tier(canonical, catalog)
        cost_declared = cost_contract_declared(canonical, catalog)
        if max_cost is not None and cost > max_cost:
            continue
        # Market filter (R15-INC-002): a market-specific operator is not offered
        # to the other market.
        if market is not None and market not in market_support(canonical):
            continue
        status = source_status(canonical, catalog, env_sources)
        eligible = mining_eligible(
            canonical,
            catalog=catalog,
            role=role,
            available_sources=env_sources,
            target_frequency=freq,
            # R16-017: market is part of the SAME eligibility decision, so a
            # direct admission check and the manifest query cannot diverge.
            market=market,
        )
        if mode == "eligible" and not eligible:
            continue
        if mode == "pending" and role not in _MINEABLE_ROLES:
            continue
        if mode == "pending" and not (certified or catalog.get("pit_safe")):
            continue
        ast_positions = _ROLE_AST_POSITIONS.get(role, ())
        from factor_engine.cleaned_operators.operator_surface import classify_canonical

        state, blockers, actions = _admission_decision(
            canonical, catalog, role, role_source, status
        )
        terminal_allowed = role in _TERMINAL_ROLES
        incremental = execution_model in (
            ExecutionModel.INDEPENDENT_WITH_WARMUP,
            ExecutionModel.CHECKPOINT,
        )

        out.append(
            MiningOperator(
                canonical=canonical,
                role=role,
                role_source=role_source,
                admission_state=state,
                mining_eligible=eligible,
                production_certified=certified,
                authoring_tier=classify_canonical(canonical),
                lifecycle_status=str(catalog.get("lifecycle_status") or catalog.get("status") or "unknown"),
                cost_tier=cost,
                cost_contract_declared=cost_declared,
                required_sources=status.required,
                available_sources=status.satisfied,
                source_unknown=status.unknown,
                input_grain=catalog.get("input_grain"),
                output_grain=_output_grain(canonical, catalog, role),
                input_semantic_types=_input_semantic_types(canonical, catalog),
                output_unit=catalog.get("output_unit"),
                scope=str(catalog.get("scope") or "unknown"),
                searchable_params=_searchable_params(canonical, catalog),
                allowed_ast_positions=ast_positions,
                terminal_allowed=terminal_allowed,
                stateful=stateful,
                execution_model=execution_model.value,
                checkpoint_supported=checkpoint,
                incremental_supported=incremental,
                full_history_replay_required=full_replay,
                missing_policy=_missing_policy(canonical),
                blockers=blockers,
                required_actions=actions,
                recommended_action="; ".join(actions),
            )
        )
    out.sort(key=lambda m: (m.role.value, m.canonical))
    return out


def mining_role_manifest() -> dict[str, MiningRole]:
    """canonical → role for every registered canonical (admission matrix input)."""
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

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
