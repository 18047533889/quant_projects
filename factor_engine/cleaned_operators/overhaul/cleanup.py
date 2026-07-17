# -*- coding: utf-8 -*-
"""Final registry cleanup and truthful backend/surface declarations."""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

FAKE_POLARS_SOURCES = frozenset({
    "factor_dsl_polars_bridge",
    "daily_panel_polars",
})

DEDUPE = {
    "MACD": "MACD_line",
    "Slope": "ts_time_slope",
    "slope": "ts_time_slope",
    "beta": "ts_beta",
    "rolling_beta": "ts_beta",
    "cs_rank_01": "rank",
    "rank_pct": "cs_pct_rank",
    "c_percentile": "cs_quantile",
    "is_inf": "is_infinite",
    # Historical top/bottom-N implementations used window=N and K=N, so they
    # were mathematically identical to ordinary rolling mean/std/sum.
    "ts_top_n_avg": "ts_mean",
    "ts_top_n_std": "ts_std",
    "ts_bottom_n_avg": "ts_mean",
    "ts_bottom_n_sum": "ts_sum",
    "tm_top_n_avg": "ts_topk_mean",
    "tm_top_n_sum": "ts_topk_sum",
    "prev": "ts_delay",
    "avg": "expanding_mean",
    "cum_max": "expanding_max",
    "cum_min": "expanding_min",
    "cum_sum": "expanding_sum",
    "cum_std": "expanding_std",
    "cum_rank": "expanding_rank",
    "cum_last": "ffill",
}

DAILY_ADDITIONS = frozenset({
    "cs_neutralize",
    "ts_regression_intercept",
    "ts_regression_resid",
    "ts_regression_r2",
    "ts_topk_mean",
    "ts_topk_sum",
    "ts_topk_std",
    "ts_bottomk_mean",
    "ts_bottomk_sum",
    "ts_bottomk_std",
    "ts_tail_mean",
})

RECIPE_CANONICALS = frozenset({
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "operating_margin",
})

_FINALIZED = False


def remove_backend(canonical: str, backend: str) -> None:
    implementations = OperatorRegistry._operators.get(canonical)
    if not implementations or backend not in implementations:
        return
    implementations.pop(backend, None)
    catalog = OperatorRegistry._catalog.get(canonical)
    if catalog is not None:
        metadata = dict(catalog.get("backend_meta") or {})
        metadata.pop(backend, None)
        catalog["backend_meta"] = metadata
        catalog["backends"] = sorted(implementations)


def alias_and_remove(old: str, new: str) -> None:
    if old == new or new not in OperatorRegistry._operators:
        return
    if old in OperatorRegistry._operators:
        OperatorRegistry.unregister(old)
    OperatorRegistry.register_alias(old, new)


def finalize() -> None:
    global _FINALIZED
    if _FINALIZED:
        return

    # A backend named Polars must not convert the panel to pandas.  Remove every
    # known bridge registration; audited native replacements have another source.
    for canonical, catalog in list(OperatorRegistry._catalog.items()):
        backend_meta = dict(catalog.get("backend_meta") or {})
        source = str((backend_meta.get("polars") or {}).get("source", ""))
        if source in FAKE_POLARS_SOURCES or "bridge" in source.lower():
            remove_backend(canonical, "polars")

    for old, new in DEDUPE.items():
        alias_and_remove(old, new)

    for alias, target in {
        "MACD": "MACD_line",
        "rolling_beta": "ts_beta",
        "beta": "ts_beta",
        "Slope": "ts_time_slope",
        "slope": "ts_time_slope",
        "ts_regression": "ts_regression_slope",
        "rolling_residual": "ts_regression_resid",
    }.items():
        if target in OperatorRegistry._operators:
            OperatorRegistry.register_alias(alias, target)

    # Financial ratios remain callable for compatibility but are explicitly
    # classified as recipes rather than primitive daily operators.
    for canonical in RECIPE_CANONICALS:
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is not None:
            catalog["status"] = "deprecated_recipe"
            catalog["selected_source"] = "factor_recipes.fundamental_ratios"

    from cleaned_operators import operator_policy, operator_surface

    operator_surface.DAILY_CANONICALS = frozenset(
        (set(operator_surface.DAILY_CANONICALS) - set(DEDUPE) - set(RECIPE_CANONICALS))
        | set(DAILY_ADDITIONS)
    )

    # Production routing must be the intersection of policy and actual native
    # registry support after bridge removal.
    operator_policy.POLARS_PARITY_VERIFIED = frozenset(
        canonical
        for canonical in operator_policy.POLARS_PARITY_VERIFIED
        if "polars" in OperatorRegistry.backends_for(canonical)
    )
    operator_policy.POLARS_PRODUCTION_SAFE = frozenset(
        canonical
        for canonical in operator_policy.POLARS_PRODUCTION_SAFE
        if "polars" in OperatorRegistry.backends_for(canonical)
    )

    _FINALIZED = True
