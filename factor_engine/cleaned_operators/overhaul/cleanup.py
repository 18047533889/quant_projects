# -*- coding: utf-8 -*-
"""Final registry cleanup and truthful backend/surface declarations."""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

PANDAS_BRIDGE_POLARS_CANONICALS = frozenset({
    "cs_quantile", "ewm_corr", "ewm_cov", "arg", "atan2", "ts_regression_slope",
})

FAKE_POLARS_SOURCES = frozenset({"factor_dsl_polars_bridge", "daily_panel_polars"})

DEDUPE = {
    "reverse": "neg",
    "inv": "inverse",
    "reciprocal": "inverse",
    "fmax": "maximum",
    "fmin": "minimum",
    "sqr": "square",
    "MACD": "MACD_line",
    "Slope": "ts_time_slope",
    "slope": "ts_time_slope",
    "beta": "ts_beta",
    "rolling_beta": "ts_beta",
    "cs_rank_01": "rank",
    "rank_pct": "cs_pct_rank",
    "c_percentile": "cs_quantile",
    "quantile": "cs_quantile",
    "is_inf": "is_infinite",
    "div_or_null": "safe_div_null",
    # Omitted K is handled as K=window by the final top/bottom-K layer, so old
    # two-argument formulas preserve their historical numerical result.
    "ts_top_n_avg": "ts_topk_mean",
    "ts_top_n_std": "ts_topk_std",
    "ts_bottom_n_avg": "ts_bottomk_mean",
    "ts_bottom_n_sum": "ts_bottomk_sum",
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
    "cs_neutralize", "ts_regression_intercept", "ts_regression_resid",
    "ts_regression_r2", "ts_topk_mean", "ts_topk_sum", "ts_topk_std",
    "ts_bottomk_mean", "ts_bottomk_sum", "ts_bottomk_std", "ts_tail_mean",
    "fundamental_staleness", "revision_delta", "period_stability",
    "cs_weighted_mean", "cs_weighted_demean", "cs_weighted_zscore",
    "group_weighted_mean", "group_weighted_zscore",
})

RECIPE_CANONICALS = frozenset({"current_ratio", "quick_ratio", "debt_to_equity", "operating_margin"})
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
    for alias, target in list(OperatorRegistry._aliases.items()):
        if target == old:
            OperatorRegistry._aliases[alias] = new
    if old in OperatorRegistry._operators:
        OperatorRegistry.unregister(old)
    OperatorRegistry.register_alias(old, new)


def finalize() -> None:
    global _FINALIZED
    if _FINALIZED:
        return
    for canonical, catalog in list(OperatorRegistry._catalog.items()):
        source = str(((catalog.get("backend_meta") or {}).get("polars") or {}).get("source", ""))
        if (
            source in FAKE_POLARS_SOURCES
            or "bridge" in source.lower()
            or canonical in PANDAS_BRIDGE_POLARS_CANONICALS
        ):
            remove_backend(canonical, "polars")
    for old, new in DEDUPE.items():
        alias_and_remove(old, new)
    # These historical names claimed industry+size regression but the target
    # only performs a group demean.  Silently resolving them is numerically
    # wrong, so fail closed instead.
    for misleading in ("industry_size_neutralize", "size_industry_neutralize"):
        OperatorRegistry._aliases.pop(misleading, None)
        for catalog in OperatorRegistry._catalog.values():
            if misleading in (catalog.get("aliases") or []):
                catalog["aliases"] = [a for a in catalog["aliases"] if a != misleading]
    for alias, target in {
        "Beta": "ts_beta", "rolling_beta": "ts_beta", "beta": "ts_beta",
        "Slope": "ts_time_slope", "slope": "ts_time_slope",
        "ts_regression": "ts_regression_slope",
        "neutralize": "group_neutralize",
        "rolling_residual": "ts_regression_resid", "safe_div": "safe_div_null",
    }.items():
        if target in OperatorRegistry._operators:
            OperatorRegistry.register_alias(alias, target)
    for canonical in RECIPE_CANONICALS:
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is not None:
            catalog["status"] = "deprecated_recipe"
            catalog["selected_source"] = "factor_recipes.fundamental_ratios"
    from cleaned_operators import operator_policy, operator_surface
    # A registered Polars backend is a promise that the implementation stays
    # inside the Polars engine.  NumPy materialisation and Python callbacks are
    # useful research fallbacks, but advertising them as Polars is misleading
    # and prevents lazy/streaming execution.  Fail closed by removing them.
    import inspect

    def implementation_source(operator) -> str:
        """Include wrapped functions and their local helper call graph."""
        seen: set[int] = set()
        chunks: list[str] = []

        def visit(obj, depth: int = 0) -> None:
            if obj is None or id(obj) in seen or depth > 3:
                return
            seen.add(id(obj))
            try:
                chunks.append(inspect.getsource(obj))
            except (OSError, TypeError):
                pass
            if not callable(obj):
                return
            try:
                closure = inspect.getclosurevars(obj)
            except (TypeError, ValueError):
                return
            module = getattr(obj, "__module__", "")
            for value in (*closure.nonlocals.values(), *closure.globals.values()):
                if inspect.isfunction(value) and getattr(value, "__module__", "") == module:
                    visit(value, depth + 1)

        visit(operator.__class__)
        visit(getattr(operator, "_fn", None))
        return "\n".join(chunks)

    for canonical, implementations in OperatorRegistry._operators.items():
        operator = implementations.get("polars")
        if operator is None:
            continue
        implementation = implementation_source(operator)
        forbidden_bridge = any(token in implementation for token in (
            "to_numpy(", "np.", "rolling_map(", "map_elements(", "to_pandas(",
            "panel_pandas_bridge", "bridge_pandas", "bridge_registry",
        ))
        if forbidden_bridge:
            remove_backend(canonical, "polars")
            continue
        materialized = any(token in implementation for token in (
            ".unpivot(", ".pivot(", ".collect(",
            "_cs_long_transform(", "_group_long_transform(",
        ))
        execution_kind = "polars_eager_native" if materialized else "expression_native"
        entry = OperatorRegistry._catalog.get(canonical, {})
        backend_meta = dict(entry.get("backend_meta") or {})
        polars_meta = dict(backend_meta.get("polars") or {})
        polars_meta.update({
            "execution_kind": execution_kind,
            "supports_lazy": execution_kind == "expression_native",
            "materializes_full_panel": materialized,
            "supports_streaming": execution_kind == "expression_native",
        })
        backend_meta["polars"] = polars_meta
        entry["backend_meta"] = backend_meta
    from backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    native = frozenset(
        c for c in OperatorRegistry._operators
        if "polars" in OperatorRegistry.backends_for(c)
    )
    operator_policy.POLARS_PARITY_VERIFIED.intersection_update(
        native & POLARS_REFERENCE_PARITY_VERIFIED
    )
    operator_policy.POLARS_PRODUCTION_SAFE.intersection_update(
        native & POLARS_REFERENCE_PARITY_VERIFIED
        & POLARS_EDGE_VERIFIED & POLARS_NO_FALLBACK_VERIFIED
    )
    _FINALIZED = True
