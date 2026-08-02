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


def _remove_declared_bridges() -> None:
    """Remove only implementations explicitly declared as compatibility bridges.

    Previous versions inspected Python source text for tokens such as ``np.`` or
    ``to_numpy``.  That produced both false positives (a native Polars function
    sharing a module with NumPy validation helpers) and false negatives (an
    indirect helper call).  Runtime evidence, not source spelling, is now the
    production admission authority.
    """
    for canonical, catalog in list(OperatorRegistry._catalog.items()):
        meta = ((catalog.get("backend_meta") or {}).get("polars") or {})
        source = str(meta.get("source", ""))
        declared_bridge = bool(meta.get("pandas_bridge")) or bool(meta.get("uses_pandas_fallback"))
        if (
            source in FAKE_POLARS_SOURCES
            or "bridge" in source.lower()
            or canonical in PANDAS_BRIDGE_POLARS_CANONICALS
            or declared_bridge
        ):
            remove_backend(canonical, "polars")


def _attach_explicit_polars_contracts() -> None:
    """Attach conservative execution metadata without inferring from source text."""
    from cleaned_operators.operator_policy import infer_operator_policy

    for canonical, implementations in OperatorRegistry._operators.items():
        operator = implementations.get("polars")
        if operator is None:
            continue
        entry = OperatorRegistry._catalog.get(canonical, {})
        policy = infer_operator_policy(operator, canonical=canonical)
        scope = getattr(policy, "scope", "unknown")
        params = set(entry.get("param_names") or ())
        backend_meta = dict(entry.get("backend_meta") or {})
        polars_meta = dict(backend_meta.get("polars") or {})

        # Default to the least permissive truthful contract.  Individual native
        # expression implementations may explicitly opt into lazy/streaming by
        # setting these fields at registration time.  No capability is inferred
        # merely from function source code.
        execution_kind = str(polars_meta.get("execution_kind") or "polars_eager_native")
        supports_lazy = bool(polars_meta.get("supports_lazy", False))
        supports_streaming = bool(polars_meta.get("supports_streaming", False))
        materializes = bool(
            polars_meta.get("materializes_full_panel", execution_kind != "expression_native")
        )
        if execution_kind == "expression_native":
            supports_lazy = bool(polars_meta.get("supports_lazy", True))
            supports_streaming = bool(polars_meta.get("supports_streaming", True))
            materializes = bool(polars_meta.get("materializes_full_panel", False))

        polars_meta.update({
            "execution_kind": execution_kind,
            "supports_lazy": supports_lazy,
            "materializes_full_panel": materializes,
            "supports_streaming": supports_streaming,
            "supports_nulls": True,
            "supports_nan": True,
            "supports_inf": True,
            "supports_scalar_broadcast": True,
            "supports_group": scope in {"group", "cs"},
            "supports_window": scope == "ts",
            "supports_min_periods": "min_periods" in params,
            "native_contract_source": "explicit_registration_and_runtime_evidence",
        })
        backend_meta["polars"] = polars_meta
        entry["backend_meta"] = backend_meta


def finalize() -> None:
    global _FINALIZED
    if _FINALIZED:
        return

    _remove_declared_bridges()

    for old, new in DEDUPE.items():
        alias_and_remove(old, new)

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

    _attach_explicit_polars_contracts()

    from cleaned_operators import operator_policy
    from backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )

    native = frozenset(
        canonical
        for canonical in OperatorRegistry._operators
        if "polars" in OperatorRegistry.backends_for(canonical)
    )
    operator_policy.POLARS_PARITY_VERIFIED.intersection_update(
        native & POLARS_REFERENCE_PARITY_VERIFIED
    )
    operator_policy.POLARS_PRODUCTION_SAFE.intersection_update(
        native
        & POLARS_REFERENCE_PARITY_VERIFIED
        & POLARS_EDGE_VERIFIED
        & POLARS_NO_FALLBACK_VERIFIED
    )
    _FINALIZED = True
