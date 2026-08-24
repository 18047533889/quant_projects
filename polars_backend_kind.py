# -*- coding: utf-8 -*-
"""True classification of "polars" backend implementations (review #5 R5-22/23).

Several registered ``backend='polars'`` implementations are not native Polars
expressions at all — they are ``pl -> pandas -> pandas operator -> pl`` bridges
that were named ``register_polars_udf`` / ``source='polars_geometry_math'`` and
would therefore survive a "bridge cleanup".  A planner must not treat them as
``native / lazy / streaming / fast``: materializing to pandas on every call has
none of those properties.

``polars_backend_kind`` classifies a registered implementation so routing can
exclude the ``pandas_materialization_fallback`` tier from auto fast-routing
without deleting the operators (they are still correct, just not fast).
"""
from __future__ import annotations

from typing import Any

# Source markers whose "polars" backend is a pandas materialization round-trip.
PANDAS_FALLBACK_SOURCES = frozenset({
    "polars_geometry_math",   # polars I/O around the same numpy kernels
    "factor_dsl_polars_bridge",
    "polars_misc_utils",
})

# Module-level bridges that round-trip through ``to_pandas()``.
PANDAS_FALLBACK_MODULES = frozenset({
    "factor_engine.cleaned_operators.rolling_pack",   # register_polars_udf
    "factor_engine.cleaned_operators.common.polars_ops",  # panel_pandas_bridge paths
})


def polars_backend_kind(operator: Any, *, source: str = "", module: str = "") -> str:
    """Classify a registered ``polars`` implementation.

    Returns one of:
      * ``polars_expression_native`` — real Polars expressions (lazy-safe);
      * ``polars_column_numpy_udf``  — per-column numpy UDF over polars columns;
      * ``pandas_materialization_fallback`` — pl -> pandas -> pandas op -> pl;
      * ``unknown`` — no evidence either way (treated conservatively).
    """
    source_l = (source or "").lower()
    if source_l in PANDAS_FALLBACK_SOURCES:
        return "pandas_materialization_fallback"
    module_l = module or (operator.__class__.__module__ if operator is not None else "")
    if any(module_l.startswith(m) for m in PANDAS_FALLBACK_MODULES):
        return "pandas_materialization_fallback"
    # A real polars implementation exposes ``_calculate_series`` that builds
    # ``pl`` expressions/columns; a bridge ends by calling ``to_pandas()``.
    kernel = getattr(operator, "_calculate_series", None)
    if kernel is not None:
        import inspect

        try:
            body = inspect.getsource(kernel)
        except (OSError, TypeError):
            body = ""
        if "to_pandas()" in body or ".to_pandas(" in body:
            return "pandas_materialization_fallback"
        if "pl." in body or "import polars" in body:
            return "polars_expression_native"
        return "polars_column_numpy_udf"
    return "unknown"


__all__ = ["polars_backend_kind", "PANDAS_FALLBACK_SOURCES", "PANDAS_FALLBACK_MODULES"]
