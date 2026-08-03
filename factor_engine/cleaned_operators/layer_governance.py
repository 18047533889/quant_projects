# -*- coding: utf-8 -*-
"""Final separation of fields, primitive operators, recipes and research tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import factor_recipes  # noqa: F401 - populate FactorRecipeRegistry
from factor_recipes.registry import FactorRecipeRegistry
from research_tools.registry import ResearchToolRegistry
from cleaned_operators.registry import OperatorRegistry

# Fields or obsolete ambiguous transforms that must not remain executable names.
DELETE_CANONICALS = frozenset({
    "vwap",
    "ttm",
    "quarter",
    "yoy",
    "avg2",
})

# Straightforward DAGs belong in FactorRecipeRegistry, not the primitive registry.
RECIPE_CANONICALS = frozenset({
    "ADXR",
    "AROON",
    "AROON_down",
    "AROON_up",
    "ATR",
    "BollingerBands",
    "BollingerLower",
    "BollingerUpper",
    "CCI",
    "DPO",
    "MOM",
    "OBV",
    "ROC",
    "RSI",
    "StochasticD",
    "StochasticK",
    "TRIX",
    "WMA",
    "WilliamsR",
    "close_gap",
    "current_ratio",
    "debt_to_equity",
    "open_gap",
    "operating_margin",
    "quick_ratio",
    "sharpe_ratio",
    "volatility",
    "vp_weighted_price",
    "vpmacd",
    "vpmacd_signal",
})

# Tools retained for analysis but removed from Factor DSL, catalog and LLM authoring.
RESEARCH_CANONICALS = frozenset({
    # Hypothesis tests and expanding diagnostics.
    "bartlett_test", "chi_square_test", "corr_test", "durbin_watson_test",
    "granger_causality", "jarque_bera_test", "kendall_corr_test", "kpss_test",
    "ks_test", "levene_test", "lilliefors_test", "spearman_corr_test",
    "stationarity_test", "ttest_one_sample", "ttest_paired", "ttest_two_samples",
    # Probability transforms; production keeps only cs_rank_gaussian.
    "cdf_chi2", "cdf_f", "cdf_normal", "cdf_t", "pdf_chi2", "pdf_f",
    "pdf_normal", "pdf_t", "quantile_normal", "quantile_t",
    "blom_transform", "rank_transform", "rankavg_transform", "tukey_transform",
    "van_der_waerden_transform",
    # Matrix, frequency and signal processing.
    "complex", "conj", "convolve", "correlate", "decimate", "eig", "fft",
    "filter_bandpass", "filter_highpass", "filter_lowpass", "filter_notch", "ifft",
    "imag", "lu_decompose", "mat_add", "mat_determinant", "mat_inverse",
    "mat_multiply", "mat_rank", "mat_subtract", "mat_transpose", "phase", "polar",
    "qr_decompose", "real", "svd", "unwrap", "wavelet", "wavelet_denoise",
    # Generic models belong in model/research layers.
    "lasso", "ridge", "regress", "pca",
    # Unanchored expanding/cumulative state.
    "expanding_mean", "expanding_sum", "expanding_std", "expanding_min",
    "expanding_max", "expanding_rank", "expanding_zscore", "cum_count",
    "cum_delta", "cum_first", "cum_prod", "cum_positive_streak", "cum_top_n_avg",
    "cum_top_n_sum", "cumulative_returns", "cum_max", "cum_min", "cum_sum",
    # Ambiguous whole-panel aggregates.
    "aggr_top_n", "at_imax", "at_imin", "count", "first_not_null",
    "geometric_mean", "harmonic_mean", "product", "std_agg", "stdp", "sum_agg",
    "varp", "wavg", "weighted_mean", "wsum",
    # Alternative row dialect.
    "row_avg", "row_beta", "row_corr", "row_count", "row_kurt", "row_max",
    "row_median", "row_min", "row_prod", "row_skew", "row_std", "row_sum", "row_var",
    # Broad data masking and unlimited repair.
    "causal_linear_extrapolate", "dropna", "ffill", "fillna", "interpolate",
    "ifnan", "log_fill_invalid", "nan_to_num", "protected_log",
    "protected_sqrt", "div_or_default",
    # Summary outputs and stateful/experimental indicators.
    "ACF", "Mode", "autocorr", "KAMA", "hump_decay", "max_drawdown", "pacf",
    "r_squared", "residual", "sem", "downside_beta", "idio_vol", "rank_corr",
    "micro_amihud_hf", "micro_bipower_var", "micro_jump_indicator",
    "micro_kyle_lambda", "micro_mid_return", "micro_realized_vol", "micro_spread",
    "micro_trade_imbalance", "micro_vpin",
})

# Historical duplicate that can remain as a compatibility alias without a canonical.
ALIAS_MIGRATIONS = {"log_returns": "ts_log_return"}

_FINALIZED = False


def _drop_aliases_for(canonical: str) -> None:
    for alias, target in list(OperatorRegistry._aliases.items()):
        if alias == canonical or target == canonical:
            OperatorRegistry._aliases.pop(alias, None)
    for catalog in OperatorRegistry._catalog.values():
        catalog["aliases"] = [
            alias for alias in catalog.get("aliases", [])
            if alias != canonical and OperatorRegistry._aliases.get(alias) is not None
        ]


def _unregister(canonical: str) -> None:
    _drop_aliases_for(canonical)
    OperatorRegistry.unregister(canonical)


def _move_to_research(canonical: str, reason: str) -> None:
    implementations = dict(OperatorRegistry._operators.get(canonical, {}))
    catalog = dict(OperatorRegistry._catalog.get(canonical, {}))
    if implementations or catalog:
        ResearchToolRegistry.register_moved(canonical, implementations, catalog, reason=reason)
    _unregister(canonical)


def _recipe_replacement_index() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, recipe in FactorRecipeRegistry.catalog().items():
        for canonical in recipe.get("replacement_for", ()):
            result[canonical] = name
    return result


def _rename_cross_sectional_dialect() -> None:
    for old, new in {
        "c_count": "cs_count",
        "c_mean": "cs_mean",
        "c_std": "cs_std",
        "c_sum": "cs_sum",
    }.items():
        if old in OperatorRegistry._operators:
            OperatorRegistry.rename_canonical(old, new)


def _normalize_semantic_names() -> None:
    """Make ambiguous historical names compatibility aliases only."""
    for old, new in {
        "ewm_std": "ts_ewm_std",
        "ewm_var": "ts_ewm_var",
        "ewm_cov": "ts_ewm_cov",
        "ewm_corr": "ts_ewm_corr",
    }.items():
        if old in OperatorRegistry._operators and new not in OperatorRegistry._operators:
            OperatorRegistry.rename_canonical(old, new)
    for old, target in {"ewm": "ts_ema", "intercept": "ts_regression_intercept"}.items():
        _unregister(old)
        if target in OperatorRegistry._operators:
            OperatorRegistry.register_alias(old, target)


def _collect_formula_fields(payload: Any, out: set[str]) -> None:
    if isinstance(payload, dict):
        canonical = payload.get("canonical")
        usage = payload.get("formula_usage")
        if isinstance(canonical, str) and usage != "forbidden":
            out.add(canonical)
        for value in payload.values():
            _collect_formula_fields(value, out)
    elif isinstance(payload, list):
        for value in payload:
            _collect_formula_fields(value, out)


def formula_field_names() -> set[str]:
    path = Path(__file__).resolve().parents[1] / "docs" / "canonical_data_fields.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fields: set[str] = set()
    _collect_formula_fields(payload, fields)
    # VWAP is an authoritative field even when a market-specific generated scan
    # has not yet refreshed it into every domain scope.
    fields.add("vwap")
    # Logical semantic inputs are bound to physical source columns by the data
    # adapter.  They remain valid, explicit authoring names for audited
    # fundamental and analyst-event formulas.
    fields.update({
        "fundamental_x",
        "fundamental_y",
        "fundamental_scale",
        "period_id",
        "actual",
        "expected",
        "expected_mean",
        "expected_std",
        "scale_base",
        "target_period_id",
    })
    return fields


def _enrich_catalog(daily: set[str]) -> None:
    from cleaned_operators.operator_surface import classify_canonical
    from cleaned_operators.operator_policy import infer_operator_policy

    close_cutoff = {"MACD_line", "MACD_signal", "MACD_hist", "RSI_WILDER", "ATR_WILDER", "ADX", "true_range"}
    for canonical, catalog in OperatorRegistry._catalog.items():
        surface = classify_canonical(canonical)
        operator = OperatorRegistry.get(canonical)
        policy = infer_operator_policy(operator, canonical=canonical) if operator is not None else None
        policy_scope = getattr(policy, "scope", "unknown") if policy is not None else "unknown"
        if policy_scope == "unknown":
            raise RuntimeError(f"active canonical {canonical!r} has no explicit scope policy")
        scope = {
            "ts": "time_series",
            "cs": "cross_sectional",
            "group": "group",
        }.get(policy_scope, policy_scope)
        params = list(catalog.get("param_names") or [])
        if scope == "fundamental_period":
            cutoff, trade_time = "available_at", "next_decision_time"
        elif canonical in close_cutoff:
            cutoff, trade_time = "close", "next_session_open"
        else:
            cutoff, trade_time = "input_dependent", "input_dependent"
        lifecycle_status = (
            "production" if surface == "daily" else
            "research" if surface == "research" else
            "deprecated" if surface == "legacy" else
            "experimental"
        )
        catalog.update({
            "surface": surface,
            "status": lifecycle_status,
            "lifecycle_status": lifecycle_status,
            "backend_status": "implemented",
            "scope": scope,
            "semantic_version": "2.0" if surface == "daily" else "1.0",
            "pit_safe": bool(policy.pit_safe) if policy is not None else False,
            "required_cutoff": cutoff,
            "earliest_trade_time": trade_time,
            # StatefulCheckpointRegistry is attached after governance and is
            # the sole source of truth.  Never infer this from a name list.
            "stateful": False,
            "lookback_fn": "window" if "window" in params else ("periods" if "periods" in params else None),
            "replacement_policy": "sealed_after_bootstrap",
        })


def _seal_registry() -> None:
    if getattr(OperatorRegistry, "_layer_governance_sealed", False):
        return
    original = OperatorRegistry.register.__func__
    OperatorRegistry._replacement_history = []

    def strict_register(
        cls,
        operator: Any,
        *,
        canonical: str,
        backend: str = "pandas_numpy",
        aliases: Optional[list[str]] = None,
        source: str = "",
        status: str = "implemented",
        backend_explicit: bool = True,
        replace: bool = False,
        replacement_reason: str = "",
        semantic_version: str = "1.0",
    ) -> None:
        canonical_name = canonical or operator.metadata.name
        exists = backend in cls._operators.get(canonical_name, {})
        if exists and not replace:
            raise ValueError(
                f"duplicate operator registration rejected: {canonical_name}/{backend}; "
                "pass replace=True with replacement_reason"
            )
        if exists and not replacement_reason:
            raise ValueError("replacement_reason is required when replace=True")
        if exists:
            previous = dict((cls._catalog.get(canonical_name, {}).get("backend_meta") or {}).get(backend) or {})
            cls._replacement_history.append({
                "canonical": canonical_name,
                "backend": backend,
                "previous_source": previous.get("source", ""),
                "new_source": source,
                "replacement_reason": replacement_reason,
                "semantic_version": semantic_version,
            })
        original(
            cls,
            operator,
            canonical=canonical_name,
            backend=backend,
            aliases=aliases,
            source=source,
            status=status,
            backend_explicit=backend_explicit,
        )
        cls._catalog[canonical_name]["semantic_version"] = semantic_version

    OperatorRegistry.register = classmethod(strict_register)
    OperatorRegistry._layer_governance_sealed = True


def finalize_layer_governance() -> None:
    global _FINALIZED
    if _FINALIZED:
        return

    _rename_cross_sectional_dialect()
    _normalize_semantic_names()
    replacements = _recipe_replacement_index()
    missing_recipe_migrations = sorted(RECIPE_CANONICALS - set(replacements))
    if missing_recipe_migrations:
        raise RuntimeError(f"recipe migration missing definitions: {missing_recipe_migrations}")

    for canonical in DELETE_CANONICALS:
        _unregister(canonical)
    for canonical in RECIPE_CANONICALS:
        _unregister(canonical)
    for canonical in RESEARCH_CANONICALS:
        _move_to_research(canonical, "not a production daily factor primitive")

    for old, target in ALIAS_MIGRATIONS.items():
        _unregister(old)
        if target in OperatorRegistry._operators:
            OperatorRegistry.register_alias(old, target)

    from cleaned_operators import operator_policy, operator_surface

    actual = set(OperatorRegistry._operators)
    partitions = {
        "daily": set(operator_surface.DAILY_CANONICALS),
        "extended": set(operator_surface.EXTENDED_ONLY_CANONICALS),
        "research": set(operator_surface.RESEARCH_ONLY_CANONICALS),
        "unsafe": set(operator_surface.UNSAFE_CANONICALS),
        "legacy": set(operator_surface.LEGACY_ONLY_CANONICALS),
        "internal": set(operator_surface.INTERNAL_ONLY_CANONICALS),
    }
    missing = {label: sorted(names - actual) for label, names in partitions.items() if names - actual}
    if missing:
        raise RuntimeError(f"static operator surface contains inactive canonicals: {missing}")
    labels = tuple(partitions)
    overlaps = {
        f"{left}/{right}": sorted(partitions[left] & partitions[right])
        for i, left in enumerate(labels)
        for right in labels[i + 1:]
        if partitions[left] & partitions[right]
    }
    if overlaps:
        raise RuntimeError(f"static operator surfaces overlap: {overlaps}")
    classified = set().union(*partitions.values())
    if classified != actual:
        raise RuntimeError(
            "static operator surface must classify the final registry exactly; "
            f"unclassified={sorted(actual - classified)}, inactive={sorted(classified - actual)}"
        )
    daily = partitions["daily"]

    # Rebuild SAFE sets *after* c_* → cs_* renames.  cleanup.finalize intersects
    # against pre-rename registry names, so evidence keyed as cs_mean would
    # otherwise be dropped even though the renamed op has a native Polars backend.
    from backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )

    daily_polars = {
        name for name in daily if "polars" in OperatorRegistry.backends_for(name)
    }
    operator_policy.POLARS_PARITY_VERIFIED.clear()
    operator_policy.POLARS_PARITY_VERIFIED.update(
        daily_polars & POLARS_REFERENCE_PARITY_VERIFIED
    )
    operator_policy.POLARS_PRODUCTION_SAFE.clear()
    operator_policy.POLARS_PRODUCTION_SAFE.update(
        daily_polars
        & POLARS_REFERENCE_PARITY_VERIFIED
        & POLARS_EDGE_VERIFIED
        & POLARS_NO_FALLBACK_VERIFIED
    )

    collisions = formula_field_names() & set(OperatorRegistry._operators)
    if collisions:
        raise RuntimeError(f"field/operator name collisions are forbidden: {sorted(collisions)}")

    _enrich_catalog(daily)
    _seal_registry()
    _FINALIZED = True
