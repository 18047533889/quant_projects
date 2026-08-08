# -*- coding: utf-8 -*-
"""可独立交付的统一算子库（唯一 production runtime 层）。"""
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
    "cleaned_operators.research_polars",
    "cleaned_operators.price_volume.ops",
    "cleaned_operators.price_volume.polars_price_volume",
    "cleaned_operators.technical.signal",
    "cleaned_operators.technical.polars_signal",
    "cleaned_operators.fundamental.ops",
    "cleaned_operators.fundamental.component_score",
    "cleaned_operators.fundamental.polars_component_score",
    "cleaned_operators.fundamental.polars_fiscal_v2",
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


import importlib.util


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
        # A module file that does not exist yet (planned / next-stage surface)
        # carries no operators, so skipping it cannot lose functionality.  A
        # module that *exists* but fails to import (missing dependency, code
        # error) stays fatal so registry breakage is never masked.  For a nested
        # name the error reports the deepest missing parent (e.g.
        # ``cleaned_operators.ts_model`` for ``.ts_model.dynamic_regression``).
        missing_module = isinstance(exc, ModuleNotFoundError) and (
            exc.name == mod or mod.startswith((exc.name or "") + ".")
        )
        if not missing_module and ".polars" not in mod and not mod.endswith("research_polars"):
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


def load_all() -> None:
    global _LOADED, _INITIALIZING
    if _LOADED:
        if OperatorRegistry.lifecycle() == "frozen":
            return
        from cleaned_operators.registry import RegistryInitializationError
        raise RegistryInitializationError(
            f"loaded registry is unexpectedly {OperatorRegistry.lifecycle()!r}"
        )
    if _INITIALIZING:
        return
    if OperatorRegistry.lifecycle() != "building":
        from cleaned_operators.registry import RegistryInitializationError
        raise RegistryInitializationError(
            f"unloaded registry cannot initialize from {OperatorRegistry.lifecycle()!r}"
        )

    _INITIALIZING = True
    try:
        _load_all_impl()
    finally:
        # ``_INITIALIZING`` must always be reset even when a module raises
        # mid-load, so a later load_all() can retry instead of deadlocking.
        _INITIALIZING = False


def _load_all_impl() -> None:
    """Run the registry initialisation sequence (wrapped by ``load_all``)."""
    global _LOADED
    # Capability modules build immutable module-level sets on first import.
    # Install the base-blob-bound evidence loader before any registry module can
    # snapshot primitive backend certification.
    from backend.evidence_delta import install_evidence_delta
    install_evidence_delta()

    from backend.production_signature_v2 import apply_production_signature_v2
    apply_production_signature_v2()

    from cleaned_operators.registration_audit import install_registration_audit
    install_registration_audit()

    for mod in _LOAD_MODULES:
        _load_module_if_available(mod)

    for mod in _REVIEWED_EXTENSIONS:
        _load_module_if_available(mod)

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
    from cleaned_operators import fiscal_strict, fiscal_event_ops
    for _canonical in (
        "period_lag", "period_change", "period_average", "period_cagr",
        "quarter_from_cumulative", "ttm_from_quarterly", "ttm_from_cumulative",
        "yoy_by_period",
    ):
        _backends = OperatorRegistry._operators.get(_canonical)
        if _backends is not None:
            _backends.pop("pandas_numpy", None)
            _backends.pop("polars", None)
    fiscal_strict.register()
    fiscal_event_ops.register()

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

    from cleaned_operators.contract_hardening import apply_final_contract_hardening
    apply_final_contract_hardening()

    OperatorRegistry.finalize()
    OperatorRegistry.freeze()
    _LOADED = True
