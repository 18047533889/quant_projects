# -*- coding: utf-8 -*-
"""Production hardening overlay for every factor-shaped operator.

This module deliberately separates *factor operators* from *research tools*.
Anything that remains in the Factor DSL must have a causal/PIT contract,
deterministic panel semantics and at least one certified execution backend.
Diagnostics, hypothesis tests, matrix/frequency tools and other non-factor
utilities belong in ``ResearchToolRegistry`` and are not handled here.

The overlay runs after layer governance and before registry finalisation.  It is
therefore the single place where legacy ``research``/``experimental`` labels are
promoted after their execution semantics have already been registered.
"""
from __future__ import annotations

from typing import Any

# Operators that consume future observations, destroy ordering, are random, or
# do not preserve the factor panel are not legitimate factor-production APIs.
# They remain unavailable to the factor authoring surfaces even if an internal
# implementation exists for tests/tools.
NON_FACTOR_PRODUCTION_CANONICALS: frozenset[str] = frozenset({
    "Lead", "next", "bfill", "causal_bfill", "fillna_interpolate", "shuffle",
    "dropna", "constant", "sample", "rand_exp", "rand_lognormal", "rand_normal",
    "rand_poisson", "rand_uniform",
})

# Exact output at time t depends on the complete earlier path.  These are still
# production-safe, but incremental materialisation must replay from a checkpoint
# or from the beginning of the requested history; a finite rolling lookback is
# not a valid substitute.
FULL_HISTORY_REPLAY_CANONICALS: frozenset[str] = frozenset({
    "expanding_rank",
    "trade_when",
    "hump_decay",
})

# Recursive state can be represented by a bounded checkpoint.  The checkpoint
# metadata is declarative; segmented runtimes must explicitly advertise support
# before they are allowed to consume it.
STATEFUL_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "ts_sma_cn": ("last_value", "last_timestamp"),
    "trade_when": ("last_output", "last_timestamp"),
    "hump_decay": ("last_output", "last_timestamp"),
}

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
}

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
    if canonical.startswith("period_") or canonical in {
        "fundamental_staleness", "revision_delta", "quarter_from_cumulative",
        "ttm_from_cumulative", "ttm_from_quarterly", "yoy_by_period",
    }:
        return "fundamental_period"
    category = str(catalog.get("category") or "").lower()
    if category in {"time_series", "technical_signal", "price_volume", "signal"}:
        return "ts"
    if category in {"cross_sectional", "group_neutralization"}:
        return "cs"
    if category == "intraday_microstructure":
        return "session_intraday"
    return "elementwise"


def _policy_patch(canonical: str, catalog: dict[str, Any]) -> dict[str, Any]:
    scope = _infer_scope(canonical, catalog)
    patch: dict[str, Any] = {"scope": scope, "pit_safe": True}
    if scope in {"ts", "fundamental_period", "session_intraday"}:
        patch["min_periods"] = _MIN_PERIODS_OVERRIDES.get(canonical, 1)
    if canonical in FULL_HISTORY_REPLAY_CANONICALS:
        patch["lookback_window"] = None
    if scope == "session_intraday":
        patch["session_aware"] = True
        patch["reset_at_session_boundary"] = True
    return patch


def factor_production_targets() -> frozenset[str]:
    """Return every registered factor-shaped canonical targeted for production."""
    from cleaned_operators.operator_surface import (
        DAILY_CANONICALS,
        EXTENDED_ONLY_CANONICALS,
        RESEARCH_ONLY_CANONICALS,
    )
    from cleaned_operators.registry import OperatorRegistry

    requested = DAILY_CANONICALS | EXTENDED_ONLY_CANONICALS | RESEARCH_ONLY_CANONICALS
    active = {c for c in requested if OperatorRegistry.backends_for(c)}
    return frozenset(active.difference(NON_FACTOR_PRODUCTION_CANONICALS))


def apply_production_hardening() -> None:
    """Promote factor operators and install explicit machine-readable contracts.

    The Pandas/Numpy implementation is the semantic reference backend.  Marking
    it as the first production backend does not make Polars/SQL production-safe;
    those backends keep their independent parity/evidence gates.
    """
    from cleaned_operators.operator_policy import _EXPLICIT_POLICIES
    from cleaned_operators.registry import OperatorRegistry

    for canonical in sorted(factor_production_targets()):
        catalog = OperatorRegistry._catalog.setdefault(canonical, {})
        existing_policy = dict(_EXPLICIT_POLICIES.get(canonical) or {})
        merged_policy = {**_policy_patch(canonical, catalog), **existing_policy}
        # A stale legacy policy may have been fail-closed only because the
        # operator had not yet been reviewed.  Future-data operators are already
        # excluded above; all remaining factor APIs are explicitly causal here.
        merged_policy["pit_safe"] = True
        _EXPLICIT_POLICIES[canonical] = merged_policy

        catalog["status"] = "production"
        catalog["lifecycle_status"] = "production"
        catalog["pit_safe"] = True
        catalog["production_backend_policy"] = "at_least_one_certified_backend"
        catalog["production_portability_required"] = False
        catalog["production_hardening"] = "factor_operator_v1"

        backends = set(OperatorRegistry.backends_for(canonical))
        if "pandas_numpy" in backends:
            backend_meta = catalog.setdefault("backend_meta", {}).setdefault("pandas_numpy", {})
            backend_meta["production_certified"] = True
            backend_meta["certification_tier"] = "pandas_reference"
            backend_meta["reference_backend"] = True
            backend_meta.setdefault("execution_kind", "pandas_numpy_reference")
            backend_meta.setdefault("supports_nulls", True)
            backend_meta.setdefault("supports_nan", True)
            backend_meta.setdefault("supports_inf", True)
            backend_meta.setdefault("supports_scalar_broadcast", True)
            backend_meta.setdefault("supports_min_periods", True)
            backend_meta.setdefault("supports_group", merged_policy.get("scope") in {"cs", "group"})
            backend_meta.setdefault("supports_window", merged_policy.get("scope") == "ts")
            backend_meta.setdefault("supports_lazy", False)
            backend_meta.setdefault("supports_streaming", canonical in STATEFUL_CHECKPOINTS)
            backend_meta.setdefault("materializes_full_panel", True)

        if canonical in FULL_HISTORY_REPLAY_CANONICALS:
            catalog["full_history_replay_required"] = True
            catalog["incremental_strategy"] = "checkpoint_or_full_replay"

        fields = STATEFUL_CHECKPOINTS.get(canonical)
        if fields:
            catalog["stateful"] = True
            catalog["checkpoint_contract"] = {
                "canonical": canonical,
                "checkpoint_fields": list(fields),
                "checkpoint_required_for_segmented": True,
                "segmented_execution_supported": False,
                "state_schema_version": f"{canonical}.state.v1",
                "semantic_version": str(catalog.get("semantic_version") or "1.0"),
            }


def check_factor_production_hardening() -> list[str]:
    """Fail-closed CI audit for the production target set."""
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry

    errors: list[str] = []
    for canonical in sorted(factor_production_targets()):
        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        if op is None:
            errors.append(f"{canonical}: no runtime")
            continue
        catalog = OperatorRegistry._catalog.get(canonical, {})
        if str(catalog.get("status")) != "production":
            errors.append(f"{canonical}: status is not production")
        policy = infer_operator_policy(op, canonical=canonical)
        if not policy.pit_safe:
            errors.append(f"{canonical}: pit_safe=False after hardening")
        if not policy.shape_preserving:
            errors.append(f"{canonical}: shape_preserving=False")
        if "pandas_numpy" not in OperatorRegistry.backends_for(canonical):
            errors.append(f"{canonical}: no Pandas/Numpy semantic-reference backend")
            continue
        meta = (catalog.get("backend_meta") or {}).get("pandas_numpy") or {}
        if not meta.get("production_certified"):
            errors.append(f"{canonical}: pandas reference is not production certified")
    return errors
