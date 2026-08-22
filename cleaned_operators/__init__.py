# -*- coding: utf-8 -*-
"""可独立交付的统一算子库（唯一 production runtime 层）。"""
import importlib.util
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cleaned_operators.registry import OperatorRegistry

from cleaned_operators import common  # noqa: F401
from cleaned_operators import price_volume  # noqa: F401
from cleaned_operators import technical  # noqa: F401
from cleaned_operators import fundamental  # noqa: F401
from cleaned_operators import microstructure  # noqa: F401
from cleaned_operators import return_decomp  # noqa: F401
from cleaned_operators import _aliases  # noqa: F401

__all__ = [
    "OperatorRegistry",
    "common",
    "price_volume",
    "technical",
    "fundamental",
    "microstructure",
]

_LOAD_MODULES = (
    "cleaned_operators.common.elementwise",
    "cleaned_operators.common.time_series",
    "cleaned_operators.common.shift_cum",
    "cleaned_operators.common.cross_sectional",
    "cleaned_operators.common.group",
    "cleaned_operators.common.data_cleaning",
    "cleaned_operators.common.statistics",
    "cleaned_operators.common.daily_panel",
    "cleaned_operators.common.gtja_compat",
    "cleaned_operators.common.scalar_compare",
    "cleaned_operators.common.scalar_where",
    "cleaned_operators.lqtp_compat",
    "cleaned_operators.common.polars_ops",
    "cleaned_operators.common.group_polars",
    "cleaned_operators.common.shift_polars",
    "cleaned_operators.common.polars_extended",
    "cleaned_operators.common.polars_auto",
    "cleaned_operators.common.polars_data_cleaning",
    "cleaned_operators.common.polars_statistics",
    "cleaned_operators.common.polars_np_parity",
    "cleaned_operators.common.polars_math_extended",
    "cleaned_operators.common.polars_batch_mirror",
    "cleaned_operators.common.polars_misc_utils",
    "cleaned_operators.common.polars_robust_stats",
    "cleaned_operators.common.polars_state_event",
    "cleaned_operators.common.polars_cs_misc",
    "cleaned_operators.common.polars_limit_misc",
    "cleaned_operators.common.polars_group",
    "cleaned_operators.common.polars_daily_native",
    "cleaned_operators.common.polars_ts_rolling",
    "cleaned_operators.research_polars",
    "cleaned_operators.price_volume.ops",
    "cleaned_operators.price_volume.polars_price_volume",
    "cleaned_operators.technical.signal",
    "cleaned_operators.technical.polars_signal",
    "cleaned_operators.fundamental.ops",
    "cleaned_operators.fundamental.component_score",
    "cleaned_operators.fundamental.polars_component_score",
    "cleaned_operators.fundamental.polars_fiscal_v2",
    "cleaned_operators.fundamental.fiscal_batch1",
    "cleaned_operators.ashare.ops",
    "cleaned_operators.shareholder.ops",
    "cleaned_operators.microstructure.ops",
    "cleaned_operators.microstructure.intraday_agg",
    "cleaned_operators.microstructure.polars_microstructure",
    "cleaned_operators.semantic_hardening",
    "cleaned_operators.operator_overhaul",
    "cleaned_operators.composite_fastpath",
    "cleaned_operators.composite_fastpath_fixes",
    "cleaned_operators.layer_primitives",
    "cleaned_operators.layer_composite_fixes",
    "cleaned_operators.fiscal_event_ops",
    "cleaned_operators.safe_ops",
    # Operator expansion (2026-08): robust statistics, conditional, state/event,
    # downside risk, group ex-self, return decomposition, limit behavior.
    "cleaned_operators.robust_stats",
    "cleaned_operators.direction_concentration",
    "cleaned_operators.conditional_ext",
    "cleaned_operators.state_event",
    "cleaned_operators.downside_risk",
    "cleaned_operators.group_ext",
    "cleaned_operators.cs_batch1",
    "cleaned_operators.return_decomp",
    "cleaned_operators.ashare.limit_ops",
    "cleaned_operators.relation.ops",
    "cleaned_operators.regression_models",
    # Next-stage expansion (2026-08): intraday higher moments / realized beta /
    # time structure / VWAP path / overnight; ts model kernels; cross-section;
    # shareholder churn-network; fundamental quality; valuation; index-listing.
    "cleaned_operators.intraday.higher_moments",
    "cleaned_operators.intraday.realized_beta",
    "cleaned_operators.intraday.time_structure",
    "cleaned_operators.intraday.vwap_path",
    "cleaned_operators.intraday.overnight",
    "cleaned_operators.ts_model.dynamic_regression",
    "cleaned_operators.ts_model.ar_meanrev",
    "cleaned_operators.ts_model.state_space",
    "cleaned_operators.ts_model.volatility",
    "cleaned_operators.ts_model.complexity",
    "cleaned_operators.ts_model.wavelet_spectral",
    "cleaned_operators.ts_model.sequence_anomaly",
    "cleaned_operators.ts_model.path_signature",
    "cleaned_operators.cross_section.robust_cs",
    "cleaned_operators.cross_section.peer_ops",
    "cleaned_operators.cross_section.panel_model",
    "cleaned_operators.fundamental.quality_v2",
    "cleaned_operators.fundamental.accruals_scores",
    "cleaned_operators.shareholder.churn_network",
    "cleaned_operators.valuation.ops_v2",
    "cleaned_operators.index_listing.ops_v2",
    # Next-stage Polars backends (2026-08): genuine expression paths only.
    "cleaned_operators.intraday.polars_next_stage",
    "cleaned_operators.intraday.polars_intraday_full",
    "cleaned_operators.valuation.polars_ops_v2",
    "cleaned_operators.fundamental.polars_quality_v2",
    "cleaned_operators.shareholder.polars_churn_network",
    "cleaned_operators.index_listing.polars_ops_v2",
    "cleaned_operators.ts_model.polars_regression",
    "cleaned_operators.cross_section.polars_peer",
    # 2026-08 final pack (61 atomics): robust tail / nonlinear dependence /
    # sequence complexity / A-share state machine / relation-group distribution /
    # intraday time-structure v2.  Shared rolling kernels live in
    # cleaned_operators.rolling_pack (imported transitively).
    "cleaned_operators.robust_tail",
    "cleaned_operators.nonlinear_dependence",
    "cleaned_operators.sequence_complexity",
    "cleaned_operators.ashare.state_machine",
    "cleaned_operators.relation.distribution",
    "cleaned_operators.intraday.time_structure_v2",
    # Alpha-language expansion (2026-08): run/hysteresis state, path geometry,
    # distribution shift, volatility structure, cs locality, events + report.
    "cleaned_operators.alpha_language_state",
    "cleaned_operators.alpha_language_shape",
    "cleaned_operators.alpha_language_distribution",
    "cleaned_operators.alpha_language_volatility",
    "cleaned_operators.alpha_language_cross",
    "cleaned_operators.alpha_language_events",
    # Stateful rule / episode / rotation pack (2026-08 CTA): latch/hold/slew/
    # deadband rule language, refractory + crossing events, recursive CUSUM,
    # episode reduce + directional-change intrinsic time, state survival,
    # cross-sectional rotation, drawdown-path recovery.
    "cleaned_operators.stateful.rule_language",
    "cleaned_operators.stateful.events",
    "cleaned_operators.stateful.sequential",
    "cleaned_operators.stateful.episode",
    "cleaned_operators.stateful.survival",
    "cleaned_operators.stateful.rotation",
    "cleaned_operators.stateful.drawdown_path",
    # Turnover-survival / chip-cost family (2026-08): shared survival kernel and
    # six cost-distribution primitives; weighted/stratified tail-risk family;
    # cumulative prospect-theory value; order-flow → impact microstructure
    # primitives (BV-C flow, impact regression, Wasserstein return shift, VPIN).
    "cleaned_operators.turnover_survival",
    "cleaned_operators.weighted_tail",
    "cleaned_operators.prospect_theory",
    "cleaned_operators.microstructure.flow_impact",
    "cleaned_operators.polars_chip_tail",
    "cleaned_operators.microstructure.polars_flow_impact",
    # Advanced information-theoretic / structure / intraday / topology operators
    # (2026-08 Gemini round): transfer entropy, Bures/Kramers-Moyal/SW-copula/
    # SPD structure shift, barrier approach, Wasserstein-quantile PCA, Rips H1.
    "cleaned_operators.advanced_information",
    "cleaned_operators.advanced_structure",
    "cleaned_operators.advanced_intraday",
    "cleaned_operators.advanced_topology",
    # Market-state description language (2026-08-08, §18): quantile-hit /
    # extreme dependence, expectile, directional-change, feature covariance
    # geometry, conditional dependence / MODWT, spread estimators, local
    # non-linear cross-section, systemic tail, marked event, update clock.
    "cleaned_operators.advanced_quantile_dynamics",
    "cleaned_operators.advanced_expectile",
    "cleaned_operators.directional_change",
    "cleaned_operators.feature_geometry",
    "cleaned_operators.conditional_dependence",
    "cleaned_operators.spread_estimators",
    "cleaned_operators.cross_section_local",
    "cleaned_operators.tail_systemic",
    "cleaned_operators.marked_event",
    "cleaned_operators.update_clock",
    # State-dynamics / geometry / event-response / spectral-crowding expansion
    # (2026-08 V2/V3): ordinal irreversibility, state density, local Markov
    # persistence/entropy/surprisal, KM local stability, first-passage bias,
    # historical event-response learning, joint energy-distance break, group
    # correlation spectrum, volume-clock path geometry, session shock recovery,
    # dynamic KNN peers, Hill tail index, exact quantile-regression beta,
    # local Lyapunov divergence, report timing / revision magnitude.
    "cleaned_operators.markov_dynamics",
    "cleaned_operators.state_geometry",
    "cleaned_operators.first_passage",
    "cleaned_operators.event_response",
    "cleaned_operators.distribution_break",
    "cleaned_operators.group_spectrum",
    "cleaned_operators.volume_clock",
    "cleaned_operators.session_recovery",
    "cleaned_operators.dynamic_knn",
    "cleaned_operators.extreme_tail",
    "cleaned_operators.local_lyapunov",
    "cleaned_operators.report_timing",
    "cleaned_operators.recurrence_analysis",
    # Genuine Polars UDF backends for the per-column dynamics families.
    "cleaned_operators.polars_dynamics",
    # 2026-08 geometry/math expansion: K-line interval geometry, directional-
    # change structural levels, multiscale trend term structure, envelope/
    # crossing quality, confirmed-extrema divergence, threshold cycles, dynamic
    # state-episode excursions, jump-robust intraday variation, intraday
    # volatility shape, volatility roughness, binned response curves, nonlinear
    # dependence, 2D joint trajectory geometry, point-process interval stats,
    # string/ordinal complexity, spectral shape, Hankel/SSA structure,
    # multifractal spectrum, serial-dependence memory, L-moments / Hartigan dip,
    # intrinsic dimension, persistence entropy, cs/group locality, session shape.
    "cleaned_operators.interval_geometry",
    "cleaned_operators.structural_levels",
    "cleaned_operators.multiscale_trend",
    "cleaned_operators.envelope",
    "cleaned_operators.crossing",
    "cleaned_operators.extrema_divergence",
    "cleaned_operators.threshold_cycle",
    "cleaned_operators.state_episode_excursion",
    "cleaned_operators.jump_robust",
    "cleaned_operators.intraday_vol_ext",
    "cleaned_operators.rough_vol",
    "cleaned_operators.binned_response",
    "cleaned_operators.dependence_ext",
    "cleaned_operators.vector_path",
    "cleaned_operators.event_interval",
    "cleaned_operators.complexity_ext",
    "cleaned_operators.spectral",
    "cleaned_operators.hankel",
    "cleaned_operators.multifractal",
    "cleaned_operators.memory_ext",
    "cleaned_operators.moments_ext",
    "cleaned_operators.intrinsic_dimension",
    "cleaned_operators.topology_ext",
    "cleaned_operators.cross_section_ext",
    "cleaned_operators.intraday_session",
    "cleaned_operators.candle_state_space",
    # Surviving polars backends for the geometry/math expansion (polars I/O
    # around the same numpy kernels; source != bridge so production keeps them).
    "cleaned_operators.polars_geometry_math",
    # 2026-08-08 Gemini-recommended primitives: gathering/distribution, weighted
    # moment / conditional covariance / robust multi-resid, activity clock,
    # spectral-shape, relation PageRank, intraday activity-duration curvature,
    # research-surface transforms (wavelet low-pass / signature Mahalanobis /
    # CROCKER bifurcation).
    "cleaned_operators.gather_ext",
    "cleaned_operators.weighted_moment_ext",
    "cleaned_operators.activity_clock",
    "cleaned_operators.spectral_ext",
    "cleaned_operators.relation.ops_ext",
    "cleaned_operators.intraday_activity_duration",
    "cleaned_operators.research_transform",
    # 2026-08-08 Gemini V2 round: HVG, RQA line structure, GLR/Pettitt change
    # points, EDGE / Abdi-Ranaldo / Pastor-Stambaugh spreads, Qn / Hodges-
    # Lehmann robust scale, Pickands / EVT stability / Allan factor,
    # composition (CoDa), cross-spectral coherence/phase, global-state dip /
    # Wasserstein barycenter, intraday impact decay, multifractal asymmetry,
    # and the research-surface DMD / bicoherence / kernel-Granger / HSIC / BDS /
    # Gaussian-SR primitives.
    "cleaned_operators.hvg_ext",
    "cleaned_operators.rqa_ext",
    "cleaned_operators.glr_change",
    "cleaned_operators.ohlc_spread",
    "cleaned_operators.robust_scale",
    "cleaned_operators.evt_allan",
    "cleaned_operators.composition",
    "cleaned_operators.cross_spectrum",
    "cleaned_operators.cs_state_ops",
    "cleaned_operators.intraday_impact",
    "cleaned_operators.multifractal_asym",
    "cleaned_operators.dmd",
    "cleaned_operators.research_spectral",
    # R47 新增算子开发总规范 (2026-08-11): state / event / slice / profile /
    # limit-EOD intraday primitives, daily technical indicators (HMA/QQE/RSX/
    # ALMA/CoppockCurve/ElderRay/FisherTransform), turnover chip surfaces, and
    # cross-sectional predictability / async-beta panel ops.  The explicit
    # policy pack is merged into ``_EXPLICIT_POLICIES`` in place (additive; it
    # does not touch the concurrent-session edits in ``operator_policy.py``).
    "cleaned_operators.intraday.state_ops",
    "cleaned_operators.intraday.event_response",
    "cleaned_operators.intraday.slice_profile",
    "cleaned_operators.intraday.limit_eod",
    "cleaned_operators.technical.new_indicators",
    "cleaned_operators.technical.chip_ops",
    "cleaned_operators.cross_section.panel_gap",
    "cleaned_operators.r47_policy_pack",
    # Filter Layer optimization (2026-08-12): despike / adaptive smooth / hysteresis.
    "cleaned_operators.filter_contracts",
    "cleaned_operators.filter_despike",
    "cleaned_operators.filter_smooth",
    "cleaned_operators.filter_hysteresis",
)

# R30 §7: explicit production / research / internal loader split.  The full
# default registry ``_LOAD_MODULES`` keeps loading everything (production +
# research model families + internal kernels) because several of those modules
# register R28-certified daily canonicals (``ts_ar_*`` / ``ts_*_regression_*``)
# that must remain in the default runtime.  ``load_all(include_research=False)``
# loads only the strictly production surface (``PRODUCTION_LOAD_MODULES`` +
# reviewed extensions) so a plain production bootstrap never registers research
# tools.  Research-only tools are additionally isolated to
# ``ResearchToolRegistry`` by layer governance regardless of module loading.
#
# ``RESEARCH_LOAD_MODULES`` are the model/research families R30 §7 names; their
# canonicals are never production-admitted (no six-evidence certification) but
# stay available for explicit research opt-in.
RESEARCH_LOAD_MODULES: tuple[str, ...] = (
    "cleaned_operators.research_polars",
    "cleaned_operators.ts_model.dynamic_regression",
    "cleaned_operators.ts_model.ar_meanrev",
    "cleaned_operators.ts_model.state_space",
    "cleaned_operators.ts_model.volatility",
    "cleaned_operators.ts_model.complexity",
    "cleaned_operators.ts_model.wavelet_spectral",
    "cleaned_operators.ts_model.sequence_anomaly",
    "cleaned_operators.ts_model.path_signature",
    "cleaned_operators.ts_model.polars_regression",
    "cleaned_operators.cross_section.panel_model",
    "cleaned_operators.research_transform",
    "cleaned_operators.dmd",
    "cleaned_operators.research_spectral",
)

# Internal kernel utilities: no user-facing factor canonicals; consumed only by
# other operators.  Kept out of the strict production surface but required for
# the full registry (many production operators import them transitively).
INTERNAL_KERNEL_MODULES: tuple[str, ...] = (
    "cleaned_operators.common.polars_auto",
    "cleaned_operators.common.polars_np_parity",
    "cleaned_operators.common.polars_batch_mirror",
    "cleaned_operators.common.polars_robust_stats",
    "cleaned_operators.common.polars_cs_misc",
    "cleaned_operators.common.polars_limit_misc",
    "cleaned_operators.common.polars_math_extended",
    "cleaned_operators.common.polars_state_event",
    "cleaned_operators.common.polars_daily_native",
    "cleaned_operators.common.polars_group",
    "cleaned_operators.common.polars_extended",
    "cleaned_operators.common.polars_statistics",
    "cleaned_operators.common.polars_misc_utils",
)

# The strictly-production module list: everything the default list loads minus
# the research model families and the internal kernel utilities.  The full
# ``_LOAD_MODULES`` remains the default (several research-module canonicals are
# R28-certified daily); ``load_all(include_research=False)`` loads exactly
# ``PRODUCTION_LOAD_MODULES`` + reviewed extensions.
PRODUCTION_LOAD_MODULES: tuple[str, ...] = tuple(
    m for m in _LOAD_MODULES
    if m not in RESEARCH_LOAD_MODULES and m not in INTERNAL_KERNEL_MODULES
)

_REVIEWED_EXTENSIONS = (
    "cleaned_operators.production_repairs",
    "cleaned_operators.price_volume.technical_extensions",
    "cleaned_operators.price_volume.technical_structure_repairs",
    "cleaned_operators.price_volume.candle_patterns_extended",
    "cleaned_operators.price_volume.candle_geometry_v2",
    "cleaned_operators.price_volume.candle_pattern_engine_v2",
    "cleaned_operators.price_volume.candle_pattern_engine_repairs_v2",
    "cleaned_operators.price_volume.structure_patterns_v2",
    "cleaned_operators.price_volume.structure_patterns_extra_v2",
    "cleaned_operators.price_volume.structure_patterns_extra_repairs_v2",
    "cleaned_operators.price_volume.liquidity_v2",
    "cleaned_operators.price_volume.liquidity_naming_v2",
    "cleaned_operators.technical.indicators_v2",
    "cleaned_operators.technical.event_state_v2",
    "cleaned_operators.technical.event_state_derivations_v1",
    "cleaned_operators.technical.exself_cs_v1",
    "cleaned_operators.technical.group_state_v1",
    "cleaned_operators.technical.polars_indicators_v2",
    "cleaned_operators.technical.polars_tech_misc",
    "cleaned_operators.technical.polars_misc_v2",
    "cleaned_operators.price_volume.polars_liquidity_v2",
    "cleaned_operators.price_volume.polars_candle",
    "cleaned_operators.price_volume.polars_structure",
    "cleaned_operators.common.stable_high_moments_v2",
    "cleaned_operators.fundamental.transforms_v2",
    "cleaned_operators.fundamental.transforms_repairs_v2",
    "cleaned_operators.fundamental.flow_semantics_v2",
    "cleaned_operators.fundamental.expectation_v2",
    "cleaned_operators.fundamental.polars_fundamental",
    "cleaned_operators.fundamental.parameter_contract_v2",
    "cleaned_operators.production_policy_extensions_v2",
    # Alpha-language aliases must load AFTER canonical targets exist
    # (fin_* live in fundamental/transforms_v2 above).
    "cleaned_operators.alpha_language_aliases",
)


# ---------------------------------------------------------------------------
# BootstrapModuleSpec (R40 #154).  ``PRODUCTION_LOAD_MODULES`` /
# ``RESEARCH_LOAD_MODULES`` / ``INTERNAL_KERNEL_MODULES`` are plain tuples with
# no typed role.  Each bootstrap module is now a typed spec
# (IMPLEMENTATION | INTERNAL_KERNEL | RESEARCH_EXTENSION | GOVERNANCE) with a
# ``required`` flag; the load loop is driven by the spec and a post-init
# validation guarantees the role invariants.
# ---------------------------------------------------------------------------
class BootstrapModuleRole(Enum):
    IMPLEMENTATION = "implementation"
    INTERNAL_KERNEL = "internal_kernel"
    RESEARCH_EXTENSION = "research_extension"
    GOVERNANCE = "governance"


@dataclass(frozen=True)
class BootstrapModuleSpec:
    """Typed bootstrap module spec (R40 #154)."""

    module: str
    role: BootstrapModuleRole
    required: bool = True

    @property
    def in_production_surface(self) -> bool:
        """True when the module belongs to the strictly-production surface."""
        return (
            self.role in (BootstrapModuleRole.IMPLEMENTATION, BootstrapModuleRole.GOVERNANCE)
            and self.module not in RESEARCH_LOAD_MODULES
            and self.module not in INTERNAL_KERNEL_MODULES
        )


def _build_bootstrap_module_specs() -> tuple[BootstrapModuleSpec, ...]:
    """Classify every default/reviewed module into a typed BootstrapModuleSpec."""
    specs: list[BootstrapModuleSpec] = []
    for mod in _LOAD_MODULES:
        if mod in RESEARCH_LOAD_MODULES:
            role = BootstrapModuleRole.RESEARCH_EXTENSION
        elif mod in INTERNAL_KERNEL_MODULES:
            role = BootstrapModuleRole.INTERNAL_KERNEL
        else:
            role = BootstrapModuleRole.IMPLEMENTATION
        specs.append(BootstrapModuleSpec(module=mod, role=role, required=True))
    for mod in _REVIEWED_EXTENSIONS:
        specs.append(
            BootstrapModuleSpec(module=mod, role=BootstrapModuleRole.IMPLEMENTATION, required=True)
        )
    return tuple(specs)


BOOTSTRAP_MODULE_SPECS: tuple[BootstrapModuleSpec, ...] = _build_bootstrap_module_specs()


def validate_bootstrap_module_specs(specs: tuple[BootstrapModuleSpec, ...]) -> list[str]:
    """R40 #154 post-init validation of the typed module-spec roles.

    Guarantees:
      * an INTERNAL_KERNEL module never appears in the PRODUCTION surface;
      * a RESEARCH_EXTENSION module never auto-enters the PRODUCTION surface;
      * a ``required`` module is never silently skippable (it must be present in
        the spec table).
    Returns a list of violations (empty == valid).
    """
    errors: list[str] = []
    for spec in specs:
        if spec.role is BootstrapModuleRole.INTERNAL_KERNEL and spec.in_production_surface:
            errors.append(
                f"internal-kernel module {spec.module!r} must not be in the "
                "production surface (R40 #154)"
            )
        if spec.role is BootstrapModuleRole.RESEARCH_EXTENSION and spec.in_production_surface:
            errors.append(
                f"research-extension module {spec.module!r} must not auto-enter "
                "the production surface (R40 #154)"
            )
    required_modules = {spec.module for spec in specs if spec.required}
    declared = {spec.module for spec in specs}
    if required_modules - declared:
        errors.append(f"required modules missing from spec table: {sorted(required_modules - declared)}")
    return errors


def check_bootstrap_module_specs() -> None:
    """Raise on bootstrap module-spec role violations (R40 #154)."""
    violations = validate_bootstrap_module_specs(BOOTSTRAP_MODULE_SPECS)
    if violations:
        raise RuntimeError("bootstrap module spec violation: " + "; ".join(violations))


def _optional_backend_available(module: str) -> bool:
    """Return whether an optional backend module can be imported."""
    return importlib.util.find_spec(module) is not None


def _load_module_if_available(mod: str) -> None:
    """Load a module, tolerating a missing optional backend at import time.

    Polars modules guard their own ``import polars`` and register numpy
    backends that are essential (e.g. ``maximum``/``minimum``), so they must
    load even when Polars is absent.  A module that genuinely requires an
    optional backend and fails without it is skipped rather than aborting the
    whole registry.
    """
    try:
        __import__(mod, fromlist=["*"])
    except ImportError as exc:
        # WS-B #247: only swallow an ImportError when the module (or its parent
        # package) does not exist (planned / next-stage surface) OR the only
        # missing dependency is the optional ``polars`` package.  Any other
        # ImportError — a code error in an existing module, a missing non-optional
        # dependency — is fatal and re-raised so registry breakage is never
        # masked.  ``should_skip_optional_import_error`` centralises the policy
        # (single source of truth in cleaned_operators/common/_polars_bridge.py).
        from cleaned_operators.common._polars_bridge import (
            should_skip_optional_import_error,
        )

        if not should_skip_optional_import_error(exc, mod):
            raise
        _SKIPPED_OPTIONAL_MODULES.add(mod)
        logger = None
        try:
            from logging_utils import get_logger
            logger = get_logger("cleaned_operators")
        except Exception:
            pass
        if logger is not None:
            logger.warning("skipped optional backend module %s: %s", mod, exc)


_LOADED = False
_INITIALIZING = False
_SKIPPED_OPTIONAL_MODULES: set[str] = set()

# ---------------------------------------------------------------------------
# RegistryBootstrap (R40 #151-#155).
#
# The legacy ``_LOADED``/``_INITIALIZING`` module bools were unlocked globals:
# a second thread that saw ``_INITIALIZING=True`` returned a HALF-LOADED
# registry.  ``RegistryBootstrap`` is the single thread-safe coordinator:
#   * NEW        -> first caller becomes the owner and initializes;
#   * INITIALIZING -> non-owner threads WAIT (never return half-loaded);
#   * READY      -> every caller returns the same fully-loaded registry;
#   * FAILED     -> every caller receives the SAME recorded initialization error;
#   * ``reset()`` clears a FAILED state for an explicit clean retry.
# ``ensure_ready(include_research=...)`` freezes the surface on FIRST call; a
# later call with a different ``include_research`` value returns the SAME
# already-loaded registry (R40 #152 — the flag never changes global behavior).
# ---------------------------------------------------------------------------
class _RegistryBootstrapState(Enum):
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"


class RegistryBootstrap:
    """Thread-safe, once-only registry initialization coordinator (R40 #151)."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.RLock())
        self._state = _RegistryBootstrapState.NEW
        self._owner_thread_id: int | None = None
        self._error: BaseException | None = None
        self._include_research: bool = True

    # -- state ------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._cond:
            return self._state.value

    def _run_initialization(self, include_research: bool) -> None:
        from cleaned_operators.registry import RegistryInitializationError

        if OperatorRegistry.lifecycle() != "building":
            raise RegistryInitializationError(
                f"unloaded registry cannot initialize from {OperatorRegistry.lifecycle()!r}"
            )
        # R40 #153: staging runs and publishes atomically from the caller's
        # perspective — a failure leaves the registry in the pre-init state
        # (lifecycle stays "building") so a clean retry reproduces the same
        # digest.
        _load_all_impl(include_research=include_research)

    def ensure_ready(self, *, include_research: bool = True) -> None:
        """Initialize once and return the fully-loaded registry.

        First caller becomes the owner (INITIALIZING).  Non-owner callers WAIT
        on the condition until READY/FAILED — they never observe a half-loaded
        registry.  A FAILED bootstrap re-raises the SAME recorded error for
        every caller.
        """
        with self._cond:
            if self._state is _RegistryBootstrapState.READY:
                return
            if self._state is _RegistryBootstrapState.FAILED:
                assert self._error is not None
                raise self._error
            if self._state is _RegistryBootstrapState.INITIALIZING:
                # Owner re-entry: the current thread is already inside
                # ``_run_initialization`` and called back (e.g. SQL emitter
                # capability certification → typed signature → ensure_cleaned_loaded).
                # Waiting here would deadlock — the owner is the only thread that
                # can ever set READY.  Return immediately; the registry already
                # carries the operator definitions the re-entrant caller needs.
                if self._owner_thread_id == threading.get_ident():
                    return
                # Non-owner: wait for the owner to finish (never return early).
                while self._state is _RegistryBootstrapState.INITIALIZING:
                    self._cond.wait()
                if self._state is _RegistryBootstrapState.READY:
                    return
                assert self._state is _RegistryBootstrapState.FAILED
                assert self._error is not None
                raise self._error
            # NEW -> owner.
            self._state = _RegistryBootstrapState.INITIALIZING
            self._owner_thread_id = threading.get_ident()
            self._include_research = bool(include_research)
            try:
                self._run_initialization(self._include_research)
            except BaseException as exc:  # noqa: BLE001 - re-raised identically to all waiters
                self._state = _RegistryBootstrapState.FAILED
                self._error = exc
                self._owner_thread_id = None
                self._cond.notify_all()
                raise
            else:
                self._state = _RegistryBootstrapState.READY
                self._owner_thread_id = None
                self._cond.notify_all()

    def reset(self) -> None:
        """Clear a FAILED state so a later call retries cleanly (R40 #153)."""
        with self._cond:
            if self._state is not _RegistryBootstrapState.FAILED:
                return
            self._state = _RegistryBootstrapState.NEW
            self._error = None


REGISTRY_BOOTSTRAP = RegistryBootstrap()

# R40 #210: typed backend-replacement audit trail.  A direct ``_backends.pop``
# bypassed the registry's replacement audit; ``replace_backend`` records the
# old implementation hash / source / reason / migration id so every replacement
# is traceable.
_BACKEND_REPLACEMENT_AUDIT: list[dict[str, Any]] = []
_REPLACEMENT_MIGRATION_IDS: dict[str, str] = {}


def _implementation_identity(operator: Any) -> str:
    import hashlib

    if operator is None:
        return ""
    return hashlib.sha256(
        f"{type(operator).__module__}.{type(operator).__qualname__}".encode("utf-8")
    ).hexdigest()[:12]


def replace_backend(
    canonical: str,
    backend: str,
    *,
    reason: str = "",
    source: str = "",
) -> str:
    """Replace a canonical's backend slot with a typed audit trail (R40 #210).

    Records the OLD implementation identity/source (before removal), pops the
    slot, and returns a migration id.  The caller then registers the new
    implementation; ``record_backend_replacement_after`` appends the new
    identity so the audit has old -> new.
    """
    import uuid

    from cleaned_operators.registry import OperatorRegistry

    migration_id = uuid.uuid4().hex[:12]
    backends = OperatorRegistry._operators.get(canonical)
    old = backends.get(backend) if backends is not None else None
    old_meta = (
        (OperatorRegistry._catalog.get(canonical, {}).get("backend_meta") or {})
        .get(backend) or {}
    )
    _BACKEND_REPLACEMENT_AUDIT.append({
        "migration_id": migration_id,
        "canonical": canonical,
        "backend": backend,
        "old_source": str(old_meta.get("source", "")),
        "old_type": type(old).__name__ if old is not None else "",
        "old_implementation_hash": _implementation_identity(old),
        "reason": str(reason),
    })
    if backends is not None:
        backends.pop(backend, None)
    return migration_id


def record_backend_replacement_after(
    migration_id: str,
    canonical: str,
    backend: str,
    *,
    source: str = "",
) -> None:
    """Append the NEW implementation identity to a replacement audit entry."""
    from cleaned_operators.registry import OperatorRegistry

    backends = OperatorRegistry._operators.get(canonical)
    new = backends.get(backend) if backends is not None else None
    for row in _BACKEND_REPLACEMENT_AUDIT:
        if row.get("migration_id") == migration_id:
            row["new_source"] = str(source)
            row["new_type"] = type(new).__name__ if new is not None else ""
            row["new_implementation_hash"] = _implementation_identity(new)
            break


def backend_replacement_audit() -> tuple[dict[str, Any], ...]:
    """Read-only view of the backend-replacement audit trail (R40 #210)."""
    return tuple(dict(row) for row in _BACKEND_REPLACEMENT_AUDIT)


# R40 #215: production signature authority availability.  Set to True only after
# ``apply_production_signature_v2`` has actually applied the signature overlay.
_SIGNATURE_AUTHORITY_AVAILABLE = False


def signature_authority_available() -> bool:
    """Whether the production signature authority has been applied."""
    return bool(_SIGNATURE_AUTHORITY_AVAILABLE)


def check_signature_authority(*, production: bool = True) -> None:
    """R40 #215: production capability check — a missing signature authority is
    a hard fail (never silently degraded to an un-signed capability surface).
    Research callers pass ``production=False`` to degrade.
    """
    global _SIGNATURE_AUTHORITY_AVAILABLE
    if production and not _SIGNATURE_AUTHORITY_AVAILABLE:
        raise RuntimeError(
            "production signature authority is not available: apply_production_"
            "signature_v2() did not apply (R40 #215 — an unsigned capability "
            "surface must not enter production)"
        )


def load_all(*, include_research: bool = True) -> None:
    """Initialize the registry exactly once (thread-safe, R40 #151-#155).

    The first caller becomes the initialization owner; concurrent callers WAIT
    and receive the same fully-loaded registry (never a half-loaded one).  A
    failed initialization re-raises the same error to every caller.  The
    ``include_research`` flag is frozen on the FIRST call and never changes the
    global surface afterwards (R40 #152).
    """
    REGISTRY_BOOTSTRAP.ensure_ready(include_research=include_research)


def _load_all_impl(*, include_research: bool = True) -> None:
    """Run the registry initialisation sequence (wrapped by ``load_all``)."""
    global _LOADED
    # Capability modules build immutable module-level sets on first import.
    # Install the base-blob-bound evidence loader before any registry module can
    # snapshot primitive backend certification.
    from backend.evidence_delta import install_evidence_delta
    install_evidence_delta()

    from backend.production_signature_v2 import apply_production_signature_v2
    apply_production_signature_v2()
    global _SIGNATURE_AUTHORITY_AVAILABLE
    _SIGNATURE_AUTHORITY_AVAILABLE = True

    from cleaned_operators.registration_audit import install_registration_audit
    install_registration_audit()

    # R40 #154: the load loop is driven by the typed BootstrapModuleSpec table
    # (IMPLEMENTATION / INTERNAL_KERNEL / RESEARCH_EXTENSION / GOVERNANCE).
    check_bootstrap_module_specs()
    for spec in BOOTSTRAP_MODULE_SPECS:
        if spec.role is BootstrapModuleRole.RESEARCH_EXTENSION and not include_research:
            continue
        _load_module_if_available(spec.module)

    # Snapshot the raw registered lifecycle status now, before any promotion
    # layer (dedupe / overhaul / layer_governance* / production_hardening)
    # rewrites ``catalog["status"]``.  ``apply_production_hardening`` uses this
    # snapshot to refuse to blanket-promote operators that were registered as
    # experimental/research (model families, next-stage kernels).
    from cleaned_operators.semantic_certification import (
        snapshot_registered_statuses,
        stamp_compatibility_metadata,
    )

    snapshot_registered_statuses()
    stamp_compatibility_metadata()

    from cleaned_operators._dedupe import apply_operator_deduplication
    apply_operator_deduplication()

    from cleaned_operators.operator_overhaul import finalize_operator_overhaul
    finalize_operator_overhaul()

    from cleaned_operators.lqtp_policy_patch import apply_lqtp_policy_patch
    apply_lqtp_policy_patch()

    from cleaned_operators.research_factor_enable import enable_research_factor_runtime
    enable_research_factor_runtime()

    from cleaned_operators.layer_governance import finalize_layer_governance
    finalize_layer_governance()

    from cleaned_operators.layer_governance_post import apply_post_governance
    apply_post_governance()

    from cleaned_operators.production_hardening import apply_production_hardening
    apply_production_hardening()

    # Later compatibility layers can overwrite the strict fiscal primitives.
    # Remove only their in-memory backend slots, then re-register the audited
    # ordinal/revision implementation before the final signature audit.
    # R40 #210: replacement goes through the typed ``replace_backend`` audit
    # trail (old/new implementation hash + reason + migration id) instead of a
    # bare ``_backends.pop`` that bypassed the registry audit.
    from cleaned_operators import fiscal_strict, fiscal_event_ops
    for _canonical in (
        "period_lag", "period_change", "period_average", "period_cagr",
        "quarter_from_cumulative", "ttm_from_quarterly", "ttm_from_cumulative",
        "yoy_by_period",
    ):
        for _backend in ("pandas_numpy", "polars"):
            _migration = replace_backend(
                _canonical, _backend,
                reason="fiscal strict ordinal/revision implementation replaces legacy layer",
                source="cleaned_operators.fiscal_strict",
            )
            _REPLACEMENT_MIGRATION_IDS[_canonical] = _migration
    fiscal_strict.register()
    fiscal_event_ops.register()
    for _canonical, _migration in _REPLACEMENT_MIGRATION_IDS.items():
        record_backend_replacement_after(
            _migration, _canonical, "pandas_numpy",
            source="cleaned_operators.fiscal_strict",
        )

    # Strict fiscal implementations are registered after the general cleanup
    # pass, so attach the same explicit native capability contract here.
    from cleaned_operators.overhaul.cleanup import _attach_explicit_polars_contracts
    _attach_explicit_polars_contracts()

    from cleaned_operators.registration_audit import finalize_registration_audit
    finalize_registration_audit()

    from backend.sql_pushdown.sql_registry import register_sql_backends
    register_sql_backends()

    # Close the backend gap: every canonical that still only has a pandas
    # reference gets a polars fallback (exact-parity delegation to the certified
    # pandas_numpy implementation), so the auto router and the SQL backends'
    # non-pushdown fallback run polars instead of pandas.  Runs after all
    # operator modules and governance layers, so names/references are final.
    from cleaned_operators.polars_gap_coverage import register_polars_gap_coverage
    register_polars_gap_coverage()

    # Replace the legacy first-seen fiscal SQL lowering with exact ordinal,
    # revision-aware semantics before evidence and final contracts are consumed.
    from backend.sql_pushdown.fiscal_v2 import apply_fiscal_sql_v2
    apply_fiscal_sql_v2()

    # Evidence overlay now observes both the final implementation identity and
    # all physical SQL registrations. The early loader install above prevents
    # stale module-level capability snapshots.
    from cleaned_operators.production_certification_overlay import apply_evidence_certification_overlay
    apply_evidence_certification_overlay()

    # R23 P1 certification: time_series / price_structure / price_volume_extension
    # categories.  Runs after the evidence overlay (which sets
    # production_certified=False for stale-artifact cases) and before
    # contract_hardening (which freezes the certification fields).  Sets
    # production_certified=True for the 18 operators whose six-way primitive
    # evidence is present in the committed artifact.
    from cleaned_operators.r23_cert_ts_price import apply_r23_certification as apply_r23_cert_ts_price
    apply_r23_cert_ts_price()

    # R23 P1 certification: math + elementwise_math families (tier 0 / tier 1).
    # Runs after the evidence overlay and the ts/price certification, before
    # contract_hardening (which freezes the certification fields).  Sets
    # production_certified=True for the operators whose six-way primitive
    # evidence is present in the committed artifact.
    from cleaned_operators.r23_cert_math import apply_r23_certification
    apply_r23_certification()

    # R23 P1 certification: cross_sectional + ashare categories.  Runs after the
    # ts_price certification and before contract_hardening (which freezes the
    # certification fields).  Sets production_certified=True for the 21
    # cross_sectional operators whose six-way primitive evidence is present in
    # the committed artifact.  No ashare operator has six-way evidence, so none
    # are certified.
    from cleaned_operators.r23_cert_cs_ashare import apply_r23_certification
    apply_r23_certification()

    # R23 P1 certification: time_series_regression / time_series_risk /
    # time_series_volatility categories.  Same data-driven guard as the other R23
    # P1 passes.  Honest finding: no in-scope operator has six-way primitive
    # evidence in the artifact, so the certified set is empty and this is a safe
    # no-op.  It stays wired so genuine future six-way evidence promotes
    # automatically (cost tiers: regression=3, risk/volatility=2).
    from cleaned_operators.r23_cert_ts_regression import apply_r23_certification as _apply_r23_cert_ts_regression
    _apply_r23_cert_ts_regression()

    # R23 P1 certification: candle_pattern / candle_geometry / chart_pattern /
    # candle_state_space categories (tier 2, cost:2).  Same data-driven guard as
    # the other R23 P1 passes.  Honest finding: no in-scope candle/chart operator
    # has six-way primitive evidence in the artifact, so the certified set is
    # empty and this is a safe no-op.  It stays wired so genuine future six-way
    # evidence promotes automatically.
    from cleaned_operators.r23_cert_candle import apply_r23_certification as _apply_r23_cert_candle
    _apply_r23_cert_candle()

    # R23 P1 certification: intraday_microstructure / market_microstructure /
    # intraday_session families.  The intraday family consumes a minute panel
    # and cannot be certified through the daily-primitive six-way fixture (a
    # daily fixture degenerates minute kernels to a single-bar day = fake
    # parity), so the honest gate is the committed minute-shape parity artifact
    # ``evidence/intraday_minute_parity.json``.  Runs after the evidence overlay
    # and the other R23 passes, before contract_hardening (which freezes the
    # certification fields).  Sets production_certified=True for the 14 intra_*
    # operators whose minute-shape parity (polars AND duckdb_sql on genuine
    # minute panels) is present in the artifact.
    from cleaned_operators.r23_cert_intraday import apply_r23_certification as apply_r23_cert_intraday
    apply_r23_cert_intraday()

    # R23 P1 certification: fundamental_period / shareholder / valuation families.
    # These are point-in-time (PIT-sensitive) financial-data operators.  Same
    # data-driven guard as the other R23 P1 passes, with an explicit PIT-denial
    # filter so the R23-P0-PIT11/PIT18 expectation/revision/restatement canonicals
    # (already fail-closed via production_hardening.NON_FACTOR_PRODUCTION_CANONICALS)
    # are NEVER re-certified here.  Honest finding: no in-scope operator has
    # six-way primitive evidence in the artifact, so the certified set is empty
    # and this is a safe no-op.  It stays wired so genuine future six-way evidence
    # promotes automatically (cost tier: financial family = tier 2 => cost:2).
    from cleaned_operators.r23_cert_fundamental import apply_r23_certification as _apply_r23_cert_fundamental
    _apply_r23_cert_fundamental()

    from cleaned_operators.contract_hardening import apply_final_contract_hardening
    apply_final_contract_hardening()

    # R30 §24 (P1-025): backfill an explicit ParamRole for every scalar that
    # still resolves to the silent ECONOMIC fallback.  Rule-based roles are
    # marked ``role_source="rule"`` (audit-separated from authored roles).
    from cleaned_operators.param_role_contract import backfill_scalar_roles

    backfill_scalar_roles()

    # R30 §25 (P1-024): migrate legacy recursive kernels off the
    # ``_STATEFUL_CANONICALS`` name-seed fallback onto explicit self-declarations.
    from cleaned_operators.stateful_contract_migration import apply_stateful_contract_migration

    apply_stateful_contract_migration()

    # R40 #176 收尾：所有注册算子批量声明 AxisEffectContract——production
    # planning 只读契约、绝无 name/category fallback（静态门
    # PRODUCTION_AXIS_EFFECT_UNDECLARED == 0 由本声明覆盖全部算子保证）。
    from ir.types import register_axis_effects_for_surface

    register_axis_effects_for_surface()

    OperatorRegistry.finalize()
    OperatorRegistry.freeze()
    _LOADED = True
