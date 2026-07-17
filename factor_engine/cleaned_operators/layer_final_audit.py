# -*- coding: utf-8 -*-
"""Final FactorEngine registry audit.

This module is intentionally loaded after layer governance has sealed the
registry.  It performs compatibility-preserving canonical cleanup and narrows
the default authoring surface without deleting research/extended runtimes.
"""
from __future__ import annotations

from collections.abc import Iterable

from cleaned_operators.registry import OperatorRegistry

_APPLIED = False

# Runtime operators that remain available only through the explicit
# ``extended`` surface.  They are deterministic mathematical/research helpers,
# but are too ambiguous, too specialised, or too easy to misuse for default
# daily-factor authoring.
EXTENDED_ONLY_CANONICALS = frozenset({
    # Extended mathematics.
    "acos", "asin", "atan", "atan2", "cos", "sin", "tan",
    "csc", "sec", "cot", "cosh", "sinh", "tanh", "arg", "cbrt",
    # Ambiguous/low-frequency transforms.
    "fix", "reverse", "unitize", "truncate", "saturate", "lerp",
    "price_spread_deviation", "ts_ratio", "ts_moment",
    "ts_max_buildup", "winsorize_mean",
})

# Compatibility names remain resolvable but are hidden from new formulas.
HIDDEN_COMPATIBILITY_NAMES = frozenset({
    "intercept", "Intercept", "residual", "Residual", "r_squared", "R2",
    "ewm", "ewm_mean", "ewm_std", "ewm_var", "ewm_cov", "ewm_corr",
    "quantile", "div_or_null",
    "ts_top_n_avg", "ts_top_n_std", "ts_bottom_n_avg", "ts_bottom_n_sum",
    "tm_top_n_avg", "tm_top_n_sum",
    "market_cap_neutralize", "cap_neutralize",
})

_EWM_RENAMES = {
    "ewm_std": "ts_ewm_std",
    "ewm_var": "ts_ewm_var",
    "ewm_cov": "ts_ewm_cov",
    "ewm_corr": "ts_ewm_corr",
}

_DIRECT_ALIASES = {
    "intercept": "ts_regression_intercept",
    "residual": "ts_regression_resid",
    "r_squared": "ts_regression_r2",
    "ewm": "ts_ema",
}


def _remove_backend(canonical: str, backend: str) -> None:
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


def _alias_and_remove(old: str, new: str) -> None:
    if new not in OperatorRegistry._operators:
        return
    for alias, target in list(OperatorRegistry._aliases.items()):
        if target == old:
            OperatorRegistry._aliases[alias] = new
    if old in OperatorRegistry._operators:
        OperatorRegistry.unregister(old)
    OperatorRegistry.register_alias(old, new)


def _rename_ewm_canonicals() -> set[str]:
    renamed: set[str] = set()
    for old, new in _EWM_RENAMES.items():
        if old in OperatorRegistry._operators:
            OperatorRegistry.rename_canonical(old, new)
            OperatorRegistry.register_alias(old, new)
            # SQL emitter branches still use the historical spelling.  Do not
            # retain a misleading SQL marker until the canonical emitter and
            # real DuckDB parity suite are upgraded together.
            _remove_backend(new, "sql")
            renamed.add(new)
        elif new in OperatorRegistry._operators:
            OperatorRegistry.register_alias(old, new)
            renamed.add(new)
    return renamed


def _remove_size_neutralize_primitive() -> None:
    """The fixed log-market-cap formula belongs to FactorRecipeRegistry."""
    if "size_neutralize" in OperatorRegistry._operators:
        OperatorRegistry.unregister("size_neutralize")
    for alias, target in list(OperatorRegistry._aliases.items()):
        if target == "size_neutralize" or alias == "size_neutralize":
            OperatorRegistry._aliases.pop(alias, None)


def _install_extended_surface(ewm_names: set[str]) -> None:
    from cleaned_operators import operator_surface

    actual = set(OperatorRegistry._operators)
    extended = frozenset(EXTENDED_ONLY_CANONICALS & actual)
    operator_surface.EXTENDED_ONLY_CANONICALS = extended
    operator_surface.DAILY_CANONICALS = frozenset(
        (set(operator_surface.DAILY_CANONICALS) - set(extended) - {"size_neutralize"})
        | (ewm_names & actual)
    )
    operator_surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(operator_surface.RESEARCH_ONLY_CANONICALS) - set(extended)
    )
    operator_surface.HIDDEN_DAILY_NAMES = frozenset(
        set(operator_surface.HIDDEN_DAILY_NAMES) | set(HIDDEN_COMPATIBILITY_NAMES)
    )

    def classify_canonical(canonical: str) -> str:
        if canonical in operator_surface.DAILY_CANONICALS:
            return "daily"
        if canonical in operator_surface.EXTENDED_ONLY_CANONICALS:
            return "extended"
        if canonical in operator_surface.RESEARCH_ONLY_CANONICALS:
            return "research"
        if canonical in operator_surface.UNSAFE_CANONICALS:
            return "unsafe"
        if canonical in operator_surface.LEGACY_ONLY_CANONICALS:
            return "legacy"
        if canonical in operator_surface.INTERNAL_ONLY_CANONICALS:
            return "internal"
        return "unclassified"

    def is_dsl_name_allowed(name: str, canonical: str, *, surface: str = "daily") -> bool:
        if surface == "all":
            return True
        category = classify_canonical(canonical)
        if surface == "daily":
            return category == "daily" and name not in operator_surface.HIDDEN_DAILY_NAMES
        if surface == "extended":
            return category == "extended"
        if surface == "research":
            return category == "research"
        if surface == "unsafe":
            return category == "unsafe"
        if surface == "legacy":
            return category == "legacy" or name in operator_surface.HIDDEN_DAILY_NAMES
        if surface == "internal":
            return category == "internal"
        if surface == "unclassified":
            return category == "unclassified"
        raise ValueError(f"unknown operator surface: {surface!r}")

    def surface_summary(canonicals: Iterable[str]) -> dict[str, int]:
        out = {
            "daily": 0,
            "extended": 0,
            "research": 0,
            "unsafe": 0,
            "legacy": 0,
            "internal": 0,
            "unclassified": 0,
        }
        for canonical in canonicals:
            out[classify_canonical(canonical)] += 1
        return out

    operator_surface.classify_canonical = classify_canonical
    operator_surface.is_dsl_name_allowed = is_dsl_name_allowed
    operator_surface.surface_summary = surface_summary

    for canonical, catalog in OperatorRegistry._catalog.items():
        catalog["surface"] = classify_canonical(canonical)
        if canonical in extended:
            catalog["authoring_default"] = False
            catalog["extended_reason"] = "specialised or ambiguous; explicit extended surface required"
        elif canonical in operator_surface.DAILY_CANONICALS:
            catalog["authoring_default"] = True


def _validate_alias_graph() -> None:
    for alias, target in OperatorRegistry._aliases.items():
        if alias == target:
            raise RuntimeError(f"self-referential operator alias: {alias}")
        if target in OperatorRegistry._aliases:
            raise RuntimeError(f"operator alias chain is forbidden: {alias} -> {target} -> {OperatorRegistry._aliases[target]}")
        if target not in OperatorRegistry._operators:
            raise RuntimeError(f"operator alias target unavailable: {alias} -> {target}")


def apply_final_operator_audit() -> None:
    global _APPLIED
    if _APPLIED:
        return

    for old, new in _DIRECT_ALIASES.items():
        _alias_and_remove(old, new)
    ewm_names = _rename_ewm_canonicals()
    _remove_size_neutralize_primitive()
    _install_extended_surface(ewm_names)

    # Policy sets must describe final canonical names only.
    from cleaned_operators import operator_policy
    operator_policy.POLARS_PARITY_VERIFIED = frozenset(
        name for name in operator_policy.POLARS_PARITY_VERIFIED
        if name in OperatorRegistry._operators and "polars" in OperatorRegistry.backends_for(name)
    )
    operator_policy.POLARS_PRODUCTION_SAFE = frozenset(
        name for name in operator_policy.POLARS_PRODUCTION_SAFE
        if name in OperatorRegistry._operators and "polars" in OperatorRegistry.backends_for(name)
    )

    _validate_alias_graph()
    _APPLIED = True


__all__ = [
    "EXTENDED_ONLY_CANONICALS",
    "HIDDEN_COMPATIBILITY_NAMES",
    "apply_final_operator_audit",
]
