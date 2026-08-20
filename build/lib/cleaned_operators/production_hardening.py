# -*- coding: utf-8 -*-
"""Production hardening overlay for factor-shaped operators.

A reviewed operator may be a production *target* before any physical backend is
admitted.  This module attaches semantic/PIT contracts and candidate capability
metadata.  Only ``production_certification_overlay`` may set
``production_certified=True`` after validating immutable execution evidence.
"""
from __future__ import annotations

from typing import Any

from cleaned_operators.semantic_certification import (
    attach_four_certificates,
    should_fail_closed,
)

# Operators that require a specific source dataset (minute bars, relation/
# shareholder tables, index constituent weights).  The A-share sources are
# confirmed in COS (StockMinuteBar / StockTopTenShareholder / IndexConstituent)
# and the dataaccess remote read path is functional, so these operators are
# eligible production targets.  Keep this set for any future source-dependent
# surface that is NOT yet mirrored/readable; runtime execution still enforces
# the dataset/column contract (see storage.sources.data_access_source).
SOURCE_BLOCKED_CANONICALS: frozenset[str] = frozenset({
    # R22-070..071: source eligibility is CONTEXTUAL, never a global ban.  A
    # session-aware operator (close relative to intraday cumulative VWAP) is a
    # DIRECT_ALPHA in any market with a certified minute source (A-share) and is
    # context-blocked where no minute source exists (US) — the DirectUse
    # market-context gate handles that per-market, so it is removed from this
    # global set.  ``intraday_volatility`` was audited 2026-08 and is actually
    # the daily close-to-open rolling volatility (open/close inputs, no minute
    # bars), so it is promoted.
    # ``fin_total_operating_accruals`` needs a standalone, reliable depreciation
    # input.  The data dictionary only exposes net fixed-asset carrying value
    # (after accumulated depreciation), so the operator cannot be PIT-certified
    # without inventing depreciation (review §5.4).  Its DirectUse verdict is
    # DELETE_NO_DATA (R22-073..076) until a confirmed depreciation source exists.
    # The set is intentionally empty of factor-shaped operators.
})

NON_FACTOR_PRODUCTION_CANONICALS: frozenset[str] = frozenset({
    # R30 §2: Lead/next/bfill/causal_bfill/fillna_interpolate/shuffle/dropna/
    # sample/rand_* were physically removed (``tombstones`` is the single
    # authority); they have no runtime and are excluded by the tombstone gate.
    "constant",
    # Source-side relation transforms.  Their legacy DSL names are retained only
    # to fail with a precise migration error; they are not factor-panel targets.
    "holder_concentration_change",
    "holder_count_change_rate",
    # micro_bvc_vpin is P2 / research-only: BV-C carries estimation error, minute
    # bars are not ticks and the VPIN literature is contested.  It must never be a
    # default production-admission subject (spec §16).
    "micro_bvc_vpin",
})

FULL_HISTORY_REPLAY_CANONICALS: frozenset[str] = frozenset({
    "expanding_rank",
    "hump_decay",
    # Recursive technical operators without a checkpoint-restore implementation
    # stay fail-closed: they can never run an incremental segment.  The
    # segmented-execution family (ts_ema / ts_ewm_* / RSI_WILDER / ATR_WILDER /
    # ADX / MACD_*) was removed from this set once
    # ``runtime.stateful_incremental`` wired checkpoint-backed segmented
    # execution into ``run_incremental`` (audit §11).
    "KAMA",
    "Supertrend",
    "SupertrendDirection",
    "PSAR",
    # 2026-08 review P0-05: EMA/Wilder-recursive indicators (``ewm(adjust=False)``
    # via _ema/_wilder) have no checkpoint-restore, so a segmented run would not
    # be bit-exact with a full-history run.  They stay fail-closed to full replay.
    "DMI_plus",
    "DMI_minus",
    "DX",
    "NATR",
    "PPO",
    "PPO_signal",
    "PPO_hist",
    "PVO",
    "PVO_signal",
    "PVO_hist",
    "KeltnerMid",
    "KeltnerUpper",
    "KeltnerLower",
    "KeltnerPosition",
    "TSI",
    "TSI_signal",
    "DEMA",
    "TEMA",
    "ChaikinOscillator",
    "ForceIndex",
    "ts_sma_cn",
    # 2026-08 stateful rule / episode pack (audit P0-B03): these recursive
    # operators have internal state but no serialize_state/restore_state
    # contract, so an incremental segment must NEVER treat the first row of a
    # segment as a true history start — they require full-history replay.
    "state_latch",
    "state_hold",
    "state_slew_limit",
    "state_deadband",
    "state_ewm_if",
    # NEW-008/115: ``state_since_reduce`` was retired (split into
    # state_since_sum/mean/count/last); a retired name must not appear in the
    # full-history-replay seed.  The split canonicals declare their own
    # execution contract.
    "event_refractory",
    "cross_event",
    "directional_change_state",
    "directional_change_extent",
    "state_since_trend_tstat",
    "ts_cumulative_deviation_score",
    # review #4 R4-05/06/07: recursive-state operators that were still running
    # as if they were bounded-window.  threshold_cycle inherits U/L across bars
    # (infinite-history state); state_episode_* track entry/direction/mfe/mae/
    # path; ts_interval_nesting_depth is depth_t = depth_{t-1}+1.  None have a
    # serialize/restore checkpoint, so segmented execution must NEVER treat a
    # segment start as a fresh history — they require full-history replay.
    "ts_threshold_cycle_period",
    "ts_threshold_cycle_asymmetry",
    "state_episode_mfe",
    "state_episode_mae",
    "state_episode_efficiency",
    "state_episode_retrace_ratio",
    "state_episode_excursion_balance",
    "ts_interval_nesting_depth",
    # review #4 R4-04: candle_gap_atr normalizes the open-gap by a *Wilder* ATR
    # (``ewm(alpha=1/w, adjust=False)`` in candle_geometry_v2) — infinite-history
    # recursion with no checkpoint-restore, so it must stay full-replay (it was
    # previously mislabeled ``bounded_history``).
    "candle_gap_atr",
})

# These schemas document the minimum state needed by a future segmented runtime.
# Presence here does not mean segmented execution is implemented or certified.
STATEFUL_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "ts_sma_cn": ("last_value", "last_timestamp"),
    "hump_decay": ("last_output", "last_timestamp"),
    # review #4 R4-05/06/07: document the minimum state a future segmented
    # runtime would need to resume these operators (presence here is the
    # state schema, NOT a promise that segmented execution is implemented).
    "ts_threshold_cycle_period": ("last_defined_state", "last_transition_time"),
    "ts_threshold_cycle_asymmetry": ("last_defined_state", "last_transition_time"),
    "state_episode_mfe": ("direction", "entry_price", "entry_scale", "mfe", "last_valid_price"),
    "state_episode_mae": ("direction", "entry_price", "entry_scale", "mae", "last_valid_price"),
    "state_episode_efficiency": ("direction", "entry_price", "entry_scale", "path_length", "last_valid_price"),
    "state_episode_retrace_ratio": ("direction", "entry_price", "entry_scale", "mfe", "mae", "last_valid_price"),
    "state_episode_excursion_balance": ("direction", "entry_price", "entry_scale", "mfe", "mae", "last_valid_price"),
    "ts_interval_nesting_depth": ("last_depth", "last_valid"),
}

# Only operators that have real restore/resume implementations may enter this
# set.  ``stateful_runtime.execute_stateful_segment`` implements checkpoint
# restore for exactly these canonicals, with numeric parity verified against the
# full-history Pandas recurrence in tests/operators/test_stateful_segment_runtime.py
# and tests/operators/test_stateful_checkpoint_hardening.py.  Keeping the set
# in sync with that runtime is a release gate: any operator registered here must
# have a branch in ``stateful_runtime.py``.
SEGMENTED_EXECUTION_CANONICALS: frozenset[str] = frozenset({
    "ts_ema",
    "ts_ewm_std",
    "ts_ewm_var",
    "ts_ewm_cov",
    "ts_ewm_corr",
    "RSI_WILDER",
    "ATR_WILDER",
    "ADX",
    "MACD_line",
    "MACD_signal",
    "MACD_hist",
})

_SCOPE_OVERRIDES: dict[str, str] = {
    "ADX": "ts",
    "ATR_WILDER": "ts",
    "MACD_hist": "ts",
    "MACD_line": "ts",
    "MACD_signal": "ts",
    "RSI_WILDER": "ts",
    "coskewness_to_market": "ts",
    "digital_count": "ts",
    "expanding_rank": "ts",
    "hump_decay": "ts",
    "idio_skew": "ts",
    "idio_vol": "ts",
    "intraday_vwap_deviation": "session_intraday",
    "lqtp_historical_cvar": "ts",
    "rank_corr": "ts",
    "residual_momentum_capm": "ts",
    "rolling_beta_to_market": "ts",
    "tail_beta": "ts",
    "trade_when": "ts",
    "ts_poly2_coeff": "ts",
    "ts_poly2_resid": "ts",
    "ts_sma_cn": "ts",
    "ts_sum_decay": "ts",
    "candle_gap": "ts",
    "candle_gap_pct": "ts",
    "candle_range_atr": "ts",
    "cdl_engulfing": "ts",
    "cdl_inside_bar": "ts",
    "cdl_outside_bar": "ts",
    # 2026-08 final pack: A-share limit/suspension state machine is a time-series
    # transform; intraday minute-in -> daily-out structure operators are
    # session-aware (their policy rows already carry session_aware=True).
    "ashare_limit_up_streak": "ts",
    "ashare_limit_down_streak": "ts",
    "ashare_days_since_limit_up": "ts",
    "ashare_days_since_limit_down": "ts",
    "ashare_limit_touch_count": "ts",
    "ashare_failed_limit_count": "ts",
    "ashare_one_price_limit_streak": "ts",
    "ashare_limit_event_density": "ts",
    "ashare_limit_asymmetry": "ts",
    "ashare_suspension_episode_length": "ts",
    "ashare_limit_open_up_streak": "ts",
    "ashare_limit_open_down_streak": "ts",
    "ashare_limit_up_volume_ratio": "ts",
    "ashare_limit_down_volume_ratio": "ts",
    "intra_bar_range_persistence": "session_intraday",
    "intra_bar_range_deviation": "session_intraday",
    "intra_tail_volume_share": "session_intraday",
    "intra_volume_price_alignment": "session_intraday",
    "intra_ute_high": "session_intraday",
    "intra_ute_low": "session_intraday",
    "intra_slot_volume_surprise": "session_intraday",
    "intra_slot_amount_surprise": "session_intraday",
    "intra_slot_volatility_surprise": "session_intraday",
    "intra_market_lead_lag_ex_self": "session_intraday",
    "intra_industry_lead_lag_ex_self": "session_intraday",
    "intra_session_return_asymmetry": "session_intraday",
    "intra_close_participation": "session_intraday",
    "intra_high_low_affinity": "session_intraday",
    # 2026-08 stateful rule / episode pack (non ts_/cs_ prefixed names).
    "state_latch": "ts",
    "state_hold": "ts",
    "state_slew_limit": "ts",
    "state_deadband": "ts",
    "event_refractory": "ts",
    "cross_event": "ts",
    "state_ewm_if": "ts",
    # NEW-008/115: state_since_reduce retired -> split canonicals self-declare.
    "directional_change_state": "ts",
    "directional_change_extent": "ts",
    "state_since_trend_tstat": "ts",
    # 2026-08 order-flow family: minute-frequency sources that aggregate one
    # scalar per (date, symbol), so they are session-aware like intra_*.
    "intraday_bvc_imbalance": "session_intraday",
    "intraday_impact_beta": "session_intraday",
    "intraday_impact_asymmetry": "session_intraday",
    "intraday_return_wasserstein_shift": "session_intraday",
    "micro_bvc_vpin": "session_intraday",
}

_FUNDAMENTAL_PERIOD_CANONICALS: frozenset[str] = frozenset({
    "fundamental_staleness",
    "revision_delta",
    "quarter_from_cumulative",
    "ttm_from_cumulative",
    "ttm_from_quarterly",
    "yoy_by_period",
})

_MIN_PERIODS_OVERRIDES: dict[str, int] = {
    "ts_regression_intercept": 3,
    "ts_regression_r2": 3,
    "ts_regression_resid": 3,
    "ts_regression_slope": 3,
    "ts_regression_tstat": 3,
    "ts_trend_tstat": 3,
    "ts_partial_corr": 3,
    "ts_poly2_coeff": 3,
    "ts_poly2_resid": 3,
    "rolling_beta_to_market": 2,
    "tail_beta": 3,
    "idio_vol": 5,
    "idio_skew": 5,
    "residual_momentum_capm": 5,
    "coskewness_to_market": 5,
    "ts_sma_cn": 1,
}


def _infer_scope(canonical: str, catalog: dict[str, Any]) -> str:
    if canonical in _SCOPE_OVERRIDES:
        return _SCOPE_OVERRIDES[canonical]
    if canonical.startswith("ts_"):
        return "ts"
    if canonical.startswith("cs_"):
        return "cs"
    if canonical.startswith("group_"):
        return "group"
    if canonical.startswith("period_") or canonical in _FUNDAMENTAL_PERIOD_CANONICALS:
        return "fundamental_period"
    category = str(catalog.get("category") or "").lower()
    if category in {
        "time_series",
        "technical_signal",
        "price_volume",
        "price_volume_extension",
        "ohlc_volatility",
        "candle_pattern",
        "price_structure",
        "chart_pattern",
        "signal",
    }:
        return "ts"
    if category in {"fundamental_period", "fundamental_expectation"}:
        return "fundamental_period"
    if category in {"cross_sectional", "group_neutralization"}:
        return "cs"
    if category == "intraday_microstructure":
        return "session_intraday"
    return "elementwise"


def _bars_from_tags(catalog: dict[str, Any]) -> int | None:
    for tag in catalog.get("tags") or ():
        text = str(tag).strip().lower()
        if not text.startswith("bars_"):
            continue
        try:
            bars = int(text.split("_", 1)[1])
        except (TypeError, ValueError):
            continue
        if bars > 0:
            return bars
    return None


def _min_periods_from_contract(canonical: str, catalog: dict[str, Any]) -> int | None:
    """R7-248: derive the operator's warmup floor from its DECLARED contract,
    never blanket-default it to 1.

    Resolution order:
    1. ``min_periods:`` tag label;
    2. the ``_MIN_PERIODS_FLOORS`` statistical floor map (skew>=3, kurt>=4, …)
       — the semantic minimum a kernel NEEDS to be well-defined;
    3. ``bars_`` tag (a warmup length the operator itself declared);
    4. ``None`` (fall back to 1 only when nothing is declared anywhere).

    ``cost:`` tags are deliberately NOT consumed here — ``cost`` is a
    computational-cost hint for BackendRouter / ExecutionCostModel, never a
    statistical min-periods floor.  Treating ``cost:8`` as ``min_periods=8``
    desynchronises kernel warmup from policy warmup.
    """
    for tag in catalog.get("tags") or ():
        text = str(tag).strip().lower()
        if text.startswith("min_periods:"):
            try:
                value = int(text.split(":", 1)[1])
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    try:
        from cleaned_operators.contract_hardening import _MIN_PERIODS_FLOORS

        if canonical in _MIN_PERIODS_FLOORS:
            return _MIN_PERIODS_FLOORS[canonical]
    except (ImportError, AttributeError):
        pass
    return _bars_from_tags(catalog)


def _policy_patch(canonical: str, catalog: dict[str, Any]) -> dict[str, Any]:
    scope = _infer_scope(canonical, catalog)
    # Fail-closed intermediate (round-7 P0): a patch must never write
    # ``pit_safe=True`` *before* certification.  The final reconcile below is
    # the only writer of ``pit_safe`` from ``catalog``; any consumer that reads
    # this patch (an intermediate manifest, a bootstrap allowlist, a cache)
    # must see UNKNOWN/False, never an over-certified True.
    patch: dict[str, Any] = {"scope": scope, "pit_safe": False}
    if scope in {"ts", "fundamental_period", "session_intraday"}:
        # R7-248: never blanket-overwrite an operator's declared warmup floor
        # with 1.  Derive from (in priority order): the explicit override map,
        # the operator's declared ParamSpec history contract, then the tag
        # ``cost:``/``min_periods:`` label; only fall back to 1 when NO value
        # is declared anywhere.  The old ``.get(canonical, 1)`` wrote 1 over
        # every high-min-periods operator (skew/kurt/regression needing 3-30),
        # desynchronising kernel warmup from policy warmup from analyzer history.
        declared = _MIN_PERIODS_OVERRIDES.get(canonical)
        if declared is None:
            declared = _min_periods_from_contract(canonical, catalog)
        patch["min_periods"] = declared if declared is not None else 1
    bars = _bars_from_tags(catalog)
    if bars is not None and bars > 1:
        patch["lag"] = max(int(patch.get("lag", 0) or 0), bars - 1)
    if canonical in FULL_HISTORY_REPLAY_CANONICALS:
        patch["lookback_window"] = None
    if scope == "session_intraday":
        patch["session_aware"] = True
        patch["reset_at_session_boundary"] = True
    return patch


def _scope_is_authoritative(canonical: str) -> bool:
    return (
        canonical in _SCOPE_OVERRIDES
        or canonical.startswith(("ts_", "cs_", "group_", "period_", "fin_"))
        or canonical in _FUNDAMENTAL_PERIOD_CANONICALS
    )


def factor_production_targets() -> frozenset[str]:
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
        UNSAFE_CANONICALS,
    )
    from cleaned_operators.registry import OperatorRegistry

    requested = (DAILY_CANONICALS | EXTENDED_ONLY_CANONICALS | RESEARCH_ONLY_CANONICALS).difference(UNSAFE_CANONICALS)
    active = {canonical for canonical in requested if OperatorRegistry.backends_for(canonical)}
    source_blocked = SOURCE_BLOCKED_CANONICALS
    # Operators explicitly flagged compatibility/diagnostic/benchmark-only are
    # reviewed non-default targets: they exist in the DSL but are not default
    # production-admission subjects, so the audit must not iterate them.
    non_default = {
        canonical
        for canonical in active
        if any(
            (OperatorRegistry._catalog.get(canonical) or {}).get(flag)
            for flag in (
                "compatibility_only", "diagnostic_only", "benchmark_only",
                "hidden_from_default_mining",
            )
        )
    }
    return frozenset(
        active.difference(NON_FACTOR_PRODUCTION_CANONICALS)
        .difference(source_blocked)
        .difference(non_default)
    )


def _remove_promoted_legacy_denials(targets: frozenset[str]) -> None:
    import cleaned_operators.operator_spec as spec_mod

    spec_mod.PRODUCTION_DENIED_CANONICALS = frozenset(
        canonical
        for canonical in spec_mod.PRODUCTION_DENIED_CANONICALS
        if canonical not in targets
    )
    overlap = targets & spec_mod.PERMANENTLY_FORBIDDEN_CANONICALS
    if overlap:
        raise RuntimeError(
            "production target intersects permanently forbidden canonicals: "
            + ", ".join(sorted(overlap))
        )


def _sync_final_runtime_contract(canonical: str, catalog: dict[str, Any]) -> None:
    from cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
    if operator is None:
        return
    metadata = getattr(operator, "metadata", None)
    params = list(getattr(metadata, "param_names", ()) or ())
    if params:
        catalog["param_names"] = params
    description = str(getattr(metadata, "description", "") or "").strip()
    if description:
        catalog["description"] = description
    category = str(getattr(metadata, "category", "") or "").strip()
    if category:
        catalog["category"] = category
    tags = [str(tag) for tag in (getattr(metadata, "tags", None) or ())]
    if tags:
        catalog["tags"] = tags


def _mark_experimental(canonical: str, catalog: dict[str, Any]) -> None:
    """Fail-closed lifecycle: an operator that was registered as
    experimental/research (or is in the isolation manifest) must not be
    blanket-promoted to production or marked PIT-safe.  Its four certificates
    are already attached as all-negative by ``attach_four_certificates``.
    """
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES

    catalog["status"] = "experimental"
    catalog["lifecycle_status"] = "experimental"
    catalog["pit_safe"] = False
    catalog["production_certified"] = False
    existing_policy = dict(_EXPLICIT_POLICIES.get(canonical) or {})
    existing_policy["pit_safe"] = False
    existing_policy["scope"] = existing_policy.get("scope", _infer_scope(canonical, catalog))
    _EXPLICIT_POLICIES[canonical] = existing_policy


def apply_production_hardening() -> None:
    from cleaned_operators.operator_policy import (
        PIT_UNSAFE_CANONICALS,
        _EXPLICIT_POLICIES,
    )
    from cleaned_operators.registry import OperatorRegistry

    targets = factor_production_targets()
    _remove_promoted_legacy_denials(targets)

    for canonical in sorted(targets):
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        _sync_final_runtime_contract(canonical, catalog)
        if should_fail_closed(canonical):
            # Registered experimental/research or isolated: do not promote.
            attach_four_certificates(canonical, catalog)
            _mark_experimental(canonical, catalog)
            continue
        attach_four_certificates(canonical, catalog)
        patch = _policy_patch(canonical, catalog)
        existing_policy = dict(_EXPLICIT_POLICIES.get(canonical) or {})
        merged_policy = {**patch, **existing_policy}
        # Structural-causality declaration (review §2.4 / R11 P0-A03): an
        # operator that ships its own ``pit_safe``/``causal`` metadata tags — or
        # whose scope is elementwise / cross-sectional / group (causal by
        # construction, no cross-time lookahead) — declares itself structurally
        # causal.  The lqtp patch may have pre-seeded a placeholder False for
        # un-reviewed canonicals; that placeholder must not mask the operator's
        # real causal contract, or the certifier bootstrap deadlocks (the
        # reconcile below never downgrades an explicit structural True).  The
        # genuinely non-causal set (``PIT_UNSAFE_CANONICALS``) and the
        # fail-closed lifecycle keep pit_safe=False.  Production ADMISSION is
        # still evidence-gated through ``catalog["pit_safe"]``; the runtime
        # audit's prefix-causality check verifies every structural claim before
        # evidence is written.
        if canonical not in PIT_UNSAFE_CANONICALS:
            declared = [str(t).lower() for t in (catalog.get("tags") or [])]
            if (
                "pit_safe" in declared
                or "causal" in declared
                or patch["scope"] in {"elementwise", "cs", "group"}
            ):
                merged_policy["pit_safe"] = True
        if _scope_is_authoritative(canonical):
            merged_policy["scope"] = patch["scope"]
            if "min_periods" in patch:
                merged_policy["min_periods"] = patch["min_periods"]
        if str(catalog.get("category") or "").lower() in {
            "price_volume_extension",
            "ohlc_volatility",
            "candle_pattern",
            "price_structure",
            "chart_pattern",
            "fundamental_period",
            "fundamental_expectation",
        }:
            merged_policy["scope"] = patch["scope"]
            # R7-248: the category default must never overwrite an operator's
            # declared warmup floor.  ``patch.get("min_periods", 1)`` wrote 1
            # over skew/kurt/regression floors; keep the derived value when the
            # patch carried one, and only default to 1 when genuinely absent.
            if "min_periods" in patch:
                merged_policy["min_periods"] = patch["min_periods"]
            elif "min_periods" not in merged_policy:
                merged_policy["min_periods"] = 1
            if "lag" in patch:
                merged_policy["lag"] = patch["lag"]
        # Never downgrade a declared structural True (elementwise / cs / group /
        # trailing-window causality): reconcile gates production ADMISSION via
        # ``catalog["pit_safe"]``, so overwriting structural causality here
        # would deadlock the certifier bootstrap (review §2.4).
        if not merged_policy.get("pit_safe"):
            merged_policy["pit_safe"] = bool(catalog.get("pit_safe", False))
        _EXPLICIT_POLICIES[canonical] = merged_policy

        certified = bool(catalog.get("pit_safe", False))
        catalog["status"] = "production" if certified else "experimental"
        catalog["lifecycle_status"] = "production" if certified else "experimental"
        catalog["pit_safe"] = certified
        catalog["production_backend_policy"] = "at_least_one_certified_backend"
        catalog["production_portability_required"] = False
        catalog["production_hardening"] = "factor_operator_v2"

        backends = set(OperatorRegistry.backends_for(canonical))
        if "pandas_numpy" in backends:
            backend_meta = catalog.setdefault("backend_meta", {}).setdefault(
                "pandas_numpy", {}
            )
            # Candidate only.  Evidence overlay is the sole promotion authority.
            backend_meta["production_certified"] = False
            backend_meta["certification_tier"] = "candidate"
            backend_meta["certification_source"] = None
            backend_meta["reference_backend"] = True
            backend_meta.setdefault("execution_kind", "pandas_numpy_reference")
            backend_meta.setdefault("supports_nulls", True)
            backend_meta.setdefault("supports_nan", True)
            backend_meta.setdefault("supports_inf", True)
            backend_meta.setdefault("supports_scalar_broadcast", True)
            backend_meta.setdefault("supports_min_periods", True)
            backend_meta.setdefault(
                "supports_group", merged_policy.get("scope") in {"cs", "group"}
            )
            backend_meta.setdefault("supports_window", merged_policy.get("scope") == "ts")
            backend_meta.setdefault("supports_lazy", False)
            backend_meta["supports_streaming"] = (
                canonical in SEGMENTED_EXECUTION_CANONICALS
            )
            backend_meta.setdefault("materializes_full_panel", True)

        if canonical in FULL_HISTORY_REPLAY_CANONICALS:
            catalog["full_history_replay_required"] = True
            # Fresh computation from dataset origin still needs full history, but
            # checkpoint restore allows a continuation segment to start after an
            # existing checkpoint without re-reading the whole history.
            catalog["incremental_strategy"] = (
                "full_replay_with_segmented_restore"
                if canonical in SEGMENTED_EXECUTION_CANONICALS
                else "full_replay"
            )

        fields = STATEFUL_CHECKPOINTS.get(canonical)
        if fields:
            segmented = canonical in SEGMENTED_EXECUTION_CANONICALS
            catalog["stateful"] = True
            # A complete checkpoint contract (state schema / semantic version /
            # dependency wiring) may already be attached from
            # StatefulCheckpointRegistry via layer_governance_post; preserve it
            # and only reconcile the execution-support flags with the runtime.
            contract = dict(catalog.get("checkpoint_contract") or {})
            contract.setdefault("canonical", canonical)
            contract.setdefault("checkpoint_fields", list(fields))
            contract.setdefault("checkpoint_required_for_segmented", True)
            contract["segmented_execution_supported"] = segmented
            contract.setdefault("state_schema_version", f"{canonical}.state.v1")
            contract.setdefault(
                "semantic_version", str(catalog.get("semantic_version") or "1.0")
            )
            catalog["checkpoint_contract"] = contract
            catalog["segmented_execution_supported"] = segmented
            if not segmented:
                catalog["incremental_strategy"] = "full_replay"
                catalog["full_history_replay_required"] = True
            else:
                catalog["incremental_strategy"] = "segmented_checkpoint"
                if catalog.get("full_history_replay_required") is None:
                    catalog["full_history_replay_required"] = True


def check_factor_production_hardening() -> list[str]:
    from backend.operator_capability import production_eligible_backends
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canonical in sorted(factor_production_targets()):
        operator = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(
            canonical
        )
        if operator is None:
            errors.append(f"{canonical}: no runtime")
            continue
        catalog = OperatorRegistry._catalog.get(canonical, {})
        final_params = list(
            getattr(getattr(operator, "metadata", None), "param_names", ()) or ()
        )
        if final_params and list(catalog.get("param_names") or []) != final_params:
            errors.append(
                f"{canonical}: catalog parameter contract does not match final runtime"
            )
        if should_fail_closed(canonical):
            # Registered experimental/research or isolated: hardening must keep
            # them experimental and non-PIT-safe (the whole point of the gate).
            if str(catalog.get("status")) != "experimental":
                errors.append(f"{canonical}: experimental-registered status is not experimental")
            policy = infer_operator_policy(operator, canonical=canonical)
            if policy.pit_safe:
                errors.append(f"{canonical}: experimental-registered pit_safe must stay False")
            continue
        if str(catalog.get("status")) != "production":
            errors.append(f"{canonical}: status is not production")
        policy = infer_operator_policy(operator, canonical=canonical)
        if not policy.pit_safe:
            errors.append(f"{canonical}: pit_safe=False after hardening")
        if not policy.shape_preserving:
            # R34 P0-037: shape-changing 合法当且仅当显式 OutputShapeContract
            #（input_grain/output_grain 均声明）。统一 grain 变换门——不再按
            # 算子来源/频率各自例外。
            from cleaned_operators.operator_spec import build_operator_spec

            _spec = build_operator_spec(canonical)
            _shape = getattr(_spec, "output_shape", None)
            if _shape is None or not _shape.is_valid():
                errors.append(
                    f"{canonical}: shape_preserving=False 且无显式 OutputShapeContract"
                )
        if "pandas_numpy" not in OperatorRegistry.backends_for(canonical):
            errors.append(f"{canonical}: no Pandas/Numpy semantic-reference backend")
            continue
        if not production_eligible_backends(canonical):
            errors.append(f"{canonical}: no evidence-backed production backend")
        checkpoint = catalog.get("checkpoint_contract") or {}
        pandas_meta = ((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
        if checkpoint.get("segmented_execution_supported") is False and pandas_meta.get(
            "supports_streaming"
        ):
            errors.append(
                f"{canonical}: streaming advertised without segmented execution support"
            )
    # R5 P0-04 release gate: ANY daily-surface canonical that is not PIT-safe
    # fails the release — not merely the current production targets.  A daily
    # factor with a stale/expired certification must stop the build instead of
    # printing a warning.  Genuinely research/experimental ops (``fail_closed``)
    # are exempt by design.
    from cleaned_operators.operator_surface import classify_canonical

    for canonical in sorted(OperatorRegistry.list_canonical()):
        if classify_canonical(canonical) != "daily":
            continue
        if should_fail_closed(canonical):
            continue
        operator = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(
            canonical
        )
        if operator is None:
            continue
        policy = infer_operator_policy(operator, canonical=canonical)
        if not policy.pit_safe:
            errors.append(
                f"{canonical}: daily-surface pit_safe=False (release gate R5-P0-04)"
            )
    return errors
