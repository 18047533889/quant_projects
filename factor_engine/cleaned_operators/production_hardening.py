# -*- coding: utf-8 -*-
"""Production hardening overlay for factor-shaped operators.

A reviewed operator may be a production *target* before any physical backend is
admitted.  This module attaches semantic/PIT contracts and candidate capability
metadata.  Only ``production_certification_overlay`` may set
``production_certified=True`` after validating immutable execution evidence.
"""
from __future__ import annotations

from typing import Any

# Operators that require a specific source dataset (minute bars, relation/
# shareholder tables, index constituent weights).  The A-share sources are
# confirmed in COS (StockMinuteBar / StockTopTenShareholder / IndexConstituent)
# and the dataaccess remote read path is functional, so these operators are
# eligible production targets.  Keep this set for any future source-dependent
# surface that is NOT yet mirrored/readable; runtime execution still enforces
# the dataset/column contract (see storage.sources.data_access_source).
SOURCE_BLOCKED_CANONICALS: frozenset[str] = frozenset({
    # ``intraday_*`` session-aware operators are lifecycle-experimental and are
    # not part of the factor production target set.  All ``intra_*`` minute→daily
    # operators were admitted to production once ``audit_all_factor_production.py``
    # gained a minute-panel fixture (they aggregate one scalar per (TradeDate,
    # Symbol) from the day's minute bars).
    "intraday_volatility", "intraday_vwap_deviation",
})

NON_FACTOR_PRODUCTION_CANONICALS: frozenset[str] = frozenset({
    "Lead",
    "next",
    "bfill",
    "causal_bfill",
    "fillna_interpolate",
    "shuffle",
    "dropna",
    "constant",
    "sample",
    "rand_exp",
    "rand_lognormal",
    "rand_normal",
    "rand_poisson",
    "rand_uniform",
    # Source-side relation transforms.  Their legacy DSL names are retained only
    # to fail with a precise migration error; they are not factor-panel targets.
    "holder_concentration_change",
    "holder_count_change_rate",
})

FULL_HISTORY_REPLAY_CANONICALS: frozenset[str] = frozenset({
    "expanding_rank",
    "trade_when",
    "hump_decay",
    # Recursive technical operators are fail-closed until their checkpoint
    # restore path is wired into the production DAG executor.
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
    "KAMA",
    "Supertrend",
    "SupertrendDirection",
    "PSAR",
})

# These schemas document the minimum state needed by a future segmented runtime.
# Presence here does not mean segmented execution is implemented or certified.
STATEFUL_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "ts_sma_cn": ("last_value", "last_timestamp"),
    "trade_when": ("last_output", "last_timestamp"),
    "hump_decay": ("last_output", "last_timestamp"),
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


def _policy_patch(canonical: str, catalog: dict[str, Any]) -> dict[str, Any]:
    scope = _infer_scope(canonical, catalog)
    patch: dict[str, Any] = {"scope": scope, "pit_safe": True}
    if scope in {"ts", "fundamental_period", "session_intraday"}:
        patch["min_periods"] = _MIN_PERIODS_OVERRIDES.get(canonical, 1)
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
    return frozenset(
        active.difference(NON_FACTOR_PRODUCTION_CANONICALS).difference(source_blocked)
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


def apply_production_hardening() -> None:
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES
    from cleaned_operators.registry import OperatorRegistry

    targets = factor_production_targets()
    _remove_promoted_legacy_denials(targets)

    for canonical in sorted(targets):
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        _sync_final_runtime_contract(canonical, catalog)
        patch = _policy_patch(canonical, catalog)
        existing_policy = dict(_EXPLICIT_POLICIES.get(canonical) or {})
        merged_policy = {**patch, **existing_policy}
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
            merged_policy["min_periods"] = patch.get("min_periods", 1)
            if "lag" in patch:
                merged_policy["lag"] = patch["lag"]
        merged_policy["pit_safe"] = True
        _EXPLICIT_POLICIES[canonical] = merged_policy

        catalog["status"] = "production"
        catalog["lifecycle_status"] = "production"
        catalog["pit_safe"] = True
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
        if str(catalog.get("status")) != "production":
            errors.append(f"{canonical}: status is not production")
        policy = infer_operator_policy(operator, canonical=canonical)
        if not policy.pit_safe:
            errors.append(f"{canonical}: pit_safe=False after hardening")
        if not policy.shape_preserving:
            errors.append(f"{canonical}: shape_preserving=False")
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
    return errors
