# -*- coding: utf-8 -*-
"""Fail-closed runtime routing guards discovered by the full operator audit."""
from __future__ import annotations

import os
from typing import Any


_APPLIED = False


def _safe_get_best_backend(
    name: str,
    *,
    mode: str = "production",
    data_source_kind: str = "memory",
    row_count_estimate: int | None = None,
    prefer: str = "auto",
    allow_unverified_backend: bool = False,
) -> tuple[object | None, str]:
    from backend import operator_capability as capability
    from backend.operator_cost import estimate_backend_cost
    from cleaned_operators.registry import OperatorRegistry

    if prefer not in {"auto", "pandas_numpy", "polars", "sql"}:
        raise ValueError(f"unknown backend preference: {prefer!r}")

    canonical = capability.resolve_canonical(name)
    backends = OperatorRegistry.backends_for(canonical)

    if prefer == "pandas_numpy":
        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        if operator is None:
            raise capability.UnsupportedOperatorBackendError(
                f"{canonical!r} has no pandas_numpy backend"
            )
        return operator, "pandas_numpy"

    if prefer == "polars":
        operator = OperatorRegistry.get(canonical, "polars")
        status = capability._polars_status(canonical)
        permitted = status == "production_safe" or (
            mode != "production"
            and allow_unverified_backend
            and status != "unsupported"
        )
        if operator is not None and permitted:
            return operator, "polars"
        if mode == "production":
            raise capability.UnsupportedOperatorBackendError(
                f"{canonical!r} polars backend is not production-safe"
            )
        if not allow_unverified_backend:
            raise capability.UnsupportedOperatorBackendError(
                f"{canonical!r} polars backend is not verified"
            )
        raise capability.UnsupportedOperatorBackendError(
            f"{canonical!r} has no usable polars backend"
        )

    if prefer == "sql":
        operator = OperatorRegistry.get(canonical, "sql")
        dialect = (
            "clickhouse_sql"
            if data_source_kind.lower() in {"clickhouse", "ch"}
            else "duckdb_sql"
        )
        status = capability._sql_status(canonical, dialect=dialect)
        permitted = status == "production_safe" or (
            mode != "production"
            and allow_unverified_backend
            and status != "unsupported"
        )
        if operator is not None and permitted:
            return operator, "sql"
        raise capability.UnsupportedOperatorBackendError(
            f"{canonical!r} SQL backend is not usable for {dialect}"
        )

    use_cost = os.environ.get("FACTOR_ENGINE_COST_ROUTING", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    aggressive_requested = (
        os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "").strip().lower()
        in {"auto_aggressive", "aggressive"}
    )
    # Unverified routing is never allowed in production.  In non-production
    # modes it additionally requires the explicit API opt-in; an environment
    # variable alone is insufficient.
    aggressive_allowed = (
        mode != "production"
        and allow_unverified_backend
        and aggressive_requested
    )

    candidates: list[tuple[str, float]] = []
    if "polars" in backends:
        polars_status = capability._polars_status(canonical)
        polars_ok = polars_status == "production_safe" or (
            aggressive_allowed and polars_status != "unsupported"
        )
        if polars_ok:
            candidates.append(
                (
                    "polars",
                    estimate_backend_cost(
                        canonical,
                        "polars",
                        row_count_estimate=row_count_estimate,
                        requires_conversion=True,
                    ),
                )
            )

    if "pandas_numpy" in backends:
        candidates.append(
            (
                "pandas_numpy",
                estimate_backend_cost(
                    canonical,
                    "pandas_numpy",
                    row_count_estimate=row_count_estimate,
                    requires_conversion=False,
                ),
            )
        )

    if not candidates:
        raise capability.UnsupportedOperatorBackendError(
            f"no usable backend for {canonical!r}"
        )

    if use_cost and len(candidates) > 1:
        chosen = min(candidates, key=lambda item: item[1])[0]
    elif any(backend == "polars" for backend, _ in candidates):
        chosen = "polars"
    else:
        chosen = "pandas_numpy"

    operator = OperatorRegistry.get(canonical, chosen)
    if operator is None:
        raise capability.UnsupportedOperatorBackendError(
            f"selected backend {chosen!r} is not registered for {canonical!r}"
        )
    return operator, chosen


def _safe_registry_get_preferred(
    cls,
    name: str,
    *,
    prefer: str = "auto",
    mode: str = "production",
    allow_unverified_backend: bool = False,
) -> tuple[Any | None, str]:
    from backend.operator_capability import (
        UnsupportedOperatorBackendError,
        get_best_backend,
    )

    try:
        return get_best_backend(
            name,
            prefer=prefer,
            mode=mode,
            allow_unverified_backend=allow_unverified_backend,
        )
    except UnsupportedOperatorBackendError:
        # Preserve the historical production fallback contract, but only for a
        # deliberate capability rejection.  Runtime implementation errors and
        # invalid preferences must propagate instead of being hidden.
        if mode == "production" and prefer in {"polars", "sql"}:
            fallback = cls.get(name, backend="pandas_numpy")
            if fallback is not None:
                return fallback, "pandas_numpy"
        raise


def _safe_duckdb_downgrades(*, refresh: bool = False) -> frozenset[str]:
    from backend import sql_tiers

    if sql_tiers._DUCKDB_DOWNGRADE_CACHE is None or refresh:
        try:
            from backend.sql_pushdown.duckdb_capabilities import (
                downgrade_sql_canonicals,
                get_duckdb_capability_report,
            )

            report = get_duckdb_capability_report(refresh=refresh)
            sql_tiers._DUCKDB_DOWNGRADE_CACHE = downgrade_sql_canonicals(report)
        except Exception:
            # A failed deployment probe certifies nothing.  Downgrade every
            # static production candidate rather than silently treating the
            # failure as "no downgrade required".
            sql_tiers._DUCKDB_DOWNGRADE_CACHE = frozenset(
                sql_tiers.SQL_PRODUCTION_SAFE_CANONICALS
            )
    return sql_tiers._DUCKDB_DOWNGRADE_CACHE


def apply_runtime_hardening() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from backend import operator_capability, sql_tiers
    from cleaned_operators.registry import OperatorRegistry

    operator_capability.get_best_backend = _safe_get_best_backend
    OperatorRegistry.get_preferred = classmethod(_safe_registry_get_preferred)
    sql_tiers.duckdb_downgraded_canonicals = _safe_duckdb_downgrades
    _APPLIED = True
