# -*- coding: utf-8 -*-
"""Unified backend capability registry and evidence-constrained cost router.

Production admission is two-dimensional: the canonical operator must be a
reviewed production target, and the selected physical backend must carry valid
execution evidence.  Registry lifecycle labels, implementation presence and
Pandas-first tier membership are never sufficient by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

BackendName = Literal[
    "pandas_numpy",
    "polars",
    "duckdb_sql",
    "clickhouse_sql",
]
CapabilityStatus = Literal[
    "unsupported",
    "implemented",
    "parity_verified",
    "production_safe",
]


class UnsupportedOperatorBackendError(RuntimeError):
    """Raised when the requested/automatic backend has no eligible implementation."""


_REGISTRY_TO_CAPABILITY: dict[str, BackendName | None] = {
    "pandas_numpy": "pandas_numpy",
    "polars": "polars",
    "sql": "duckdb_sql",
}
_SQL_BACKENDS: tuple[BackendName, ...] = ("duckdb_sql", "clickhouse_sql")


@dataclass(frozen=True)
class BackendCapability:
    canonical: str
    backend: BackendName
    status: CapabilityStatus
    execution_kind: str = "unsupported"
    estimated_speedup: float = 1.0
    supports_nulls: bool = False
    supports_nan: bool = False
    supports_inf: bool = False
    supports_scalar_broadcast: bool = False
    supports_min_periods: bool = False
    supports_group: bool = False
    supports_window: bool = False
    supports_lazy: bool = False
    supports_streaming: bool = False
    materializes_full_panel: bool = False
    notes: str = ""

    def to_csv_row(self) -> dict[str, str | float | bool]:
        return {
            "canonical": self.canonical,
            "backend": self.backend,
            "status": self.status,
            "execution_kind": self.execution_kind,
            "estimated_speedup": self.estimated_speedup,
            "supports_nulls": self.supports_nulls,
            "supports_nan": self.supports_nan,
            "supports_inf": self.supports_inf,
            "supports_scalar_broadcast": self.supports_scalar_broadcast,
            "supports_min_periods": self.supports_min_periods,
            "supports_group": self.supports_group,
            "supports_window": self.supports_window,
            "supports_lazy": self.supports_lazy,
            "supports_streaming": self.supports_streaming,
            "materializes_full_panel": self.materializes_full_panel,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class OperatorCapabilitySummary:
    canonical: str
    pandas_numpy: CapabilityStatus
    polars: CapabilityStatus
    duckdb_sql: CapabilityStatus
    clickhouse_sql: CapabilityStatus
    allow_in_production: bool
    parity_verified: bool
    polars_long_tier: str = "unsupported"
    notes: str = ""


def resolve_canonical(name: str) -> str:
    from cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry.resolve_canonical(name)


def polars_long_native(canonical: str) -> bool:
    from backend.polars_long_policy import POLARS_LONG_NATIVE

    return resolve_canonical(canonical) in POLARS_LONG_NATIVE


def polars_long_tier(canonical: str) -> str:
    from backend.polars_long_policy import infer_polars_long_tier

    return infer_polars_long_tier(canonical)


def polars_long_tier_status(canon: str) -> CapabilityStatus:
    from backend.polars_long_production import polars_long_production_tier

    tier = polars_long_production_tier(resolve_canonical(canon))
    if tier == "production_safe":
        return "production_safe"
    if tier == "parity_verified":
        return "parity_verified"
    if tier in {
        "implemented",
        "stateful",
        "python_rolling",
        "map_groups",
        "registry",
        "passthrough",
        "nonstandard_alg",
    }:
        return "implemented"
    return "unsupported"


def polars_expr_capable(canonical: str) -> bool:
    from backend.polars_long_policy import POLARS_LONG_COMPATIBLE

    return resolve_canonical(canonical) in POLARS_LONG_COMPATIBLE


def polars_long_capable(canonical: str) -> bool:
    from backend.polars_long_policy import get_polars_long_capable

    return resolve_canonical(canonical) in get_polars_long_capable()


def _sql_capable_canonicals() -> frozenset[str]:
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    return SQL_IMPLEMENTED_CANONICALS


_EMITTER_OK_CACHE: dict[tuple[str, str, int], bool] = {}


def _sql_emitter_ok(canon: str, *, dialect: str = "duckdb_sql") -> bool:
    from cleaned_operators.registry import OperatorRegistry

    cache_key = (canon, dialect, OperatorRegistry.version())
    if cache_key in _EMITTER_OK_CACHE:
        return _EMITTER_OK_CACHE[cache_key]
    if canon in {"column", "literal"}:
        _EMITTER_OK_CACHE[cache_key] = True
        return True
    if canon not in _sql_capable_canonicals():
        _EMITTER_OK_CACHE[cache_key] = False
        return False

    ok = False
    try:
        from backend.sql_pushdown.emitter import (
            SqlDialect,
            compile_plan_to_sql,
            plan_is_sql_capable,
        )
        from backend.sql_pushdown.plan_fixtures import minimal_plan

        plan = minimal_plan(canon)
        if plan_is_sql_capable(plan):
            compiled = compile_plan_to_sql(
                plan,
                dataset="_cap_check",
                table="_cap_check",
                time_column="ts",
                instrument_column="inst",
                dialect=(
                    SqlDialect.CLICKHOUSE
                    if dialect == "clickhouse_sql"
                    else SqlDialect.DUCKDB
                ),
            )
            ok = compiled is not None and bool(compiled.query.strip())
    except Exception:
        ok = False
    _EMITTER_OK_CACHE[cache_key] = ok
    return ok


def _polars_status(canon: str) -> CapabilityStatus:
    from backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    from cleaned_operators.registry import OperatorRegistry

    if "polars" not in OperatorRegistry.backends_for(canon):
        return "unsupported"
    if (
        canon in POLARS_REFERENCE_PARITY_VERIFIED
        and canon in POLARS_EDGE_VERIFIED
        and canon in POLARS_NO_FALLBACK_VERIFIED
    ):
        return "production_safe"
    if canon in POLARS_REFERENCE_PARITY_VERIFIED:
        return "parity_verified"
    return "implemented"


def _pandas_status(canon: str) -> CapabilityStatus:
    """Return Pandas status strictly from valid immutable evidence.

    ``catalog.status == 'production'`` means reviewed semantic target only.  It
    must never grant physical execution admission.  Daily primitives are bound
    to primitive evidence; non-Daily factor operators are bound to the
    factor-operator evidence artifact through the certification overlay.
    """
    from cleaned_operators.registry import OperatorRegistry

    if "pandas_numpy" not in OperatorRegistry.backends_for(canon):
        return "unsupported"

    catalog = OperatorRegistry._catalog.get(canon, {})
    meta = ((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
    if bool(meta.get("production_certified")):
        source = str(meta.get("certification_source") or "")
        if source in {"primitive_verified.json", "factor_operator_verified.json"}:
            return "production_safe"

    try:
        from backend.evidence_provenance import evidence_artifact_valid
        from backend.primitive_evidence import PRIMITIVE_BACKEND_EXECUTION_CERTIFIED

        if evidence_artifact_valid() and canon in PRIMITIVE_BACKEND_EXECUTION_CERTIFIED:
            return "production_safe"
    except Exception:
        pass

    # Deliberately no tier/status fallback here.  An implementation without
    # current evidence remains implemented and is unavailable in production.
    return "implemented"


def _sql_status(canon: str, *, dialect: BackendName) -> CapabilityStatus:
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    if canon not in SQL_IMPLEMENTED_CANONICALS:
        return "unsupported"
    if not _sql_emitter_ok(canon, dialect=dialect):
        return "implemented"
    if dialect == "clickhouse_sql":
        from backend.sql_pushdown.clickhouse_capabilities import (
            effective_clickhouse_production_safe,
        )
        from backend.sql_tiers import CLICKHOUSE_SQL_PARITY_VERIFIED

        if effective_clickhouse_production_safe(canon):
            return "production_safe"
        if canon in CLICKHOUSE_SQL_PARITY_VERIFIED:
            return "parity_verified"
        return "implemented"

    from backend.sql_tiers import (
        DUCKDB_SQL_PARITY_VERIFIED,
        effective_sql_production_safe,
    )

    if effective_sql_production_safe(canon):
        return "production_safe"
    if canon in DUCKDB_SQL_PARITY_VERIFIED:
        return "parity_verified"
    return "implemented"


def backend_status(
    canonical: str,
    backend: BackendName,
    *,
    data_source_kind: str = "duckdb",
) -> CapabilityStatus:
    canon = resolve_canonical(canonical)
    if backend == "pandas_numpy":
        return _pandas_status(canon)
    if backend == "polars":
        return _polars_status(canon)
    if backend in _SQL_BACKENDS:
        return _sql_status(canon, dialect=backend)
    return "unsupported"


def production_eligible_backends(
    canonical: str,
    *,
    data_source_kind: str = "duckdb",
) -> tuple[str, ...]:
    """Return only independently production-certified physical backends."""
    canon = resolve_canonical(canonical)
    eligible: list[str] = []
    if _pandas_status(canon) == "production_safe":
        eligible.append("pandas_numpy")
    if _polars_status(canon) == "production_safe":
        eligible.append("polars")
    dialect: BackendName = (
        "clickhouse_sql"
        if data_source_kind.lower() in {"clickhouse", "ch"}
        else "duckdb_sql"
    )
    if _sql_status(canon, dialect=dialect) == "production_safe":
        eligible.append("sql")
    return tuple(eligible)


def capability_for(canonical: str, backend: BackendName) -> BackendCapability:
    from backend.operator_cost import default_backend_speedup
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry

    canon = resolve_canonical(canonical)
    if canon == "if_else":
        canon = "where"
    policy = infer_operator_policy(canon)
    scope = getattr(policy, "scope", "") or ""
    backend_key = "sql" if backend in _SQL_BACKENDS else backend
    backend_meta = dict(
        (
            (OperatorRegistry._catalog.get(canon, {}).get("backend_meta") or {}).get(
                backend_key
            )
            or {}
        )
    )
    if backend == "pandas_numpy":
        status = _pandas_status(canon)
    elif backend == "polars":
        status = _polars_status(canon)
    else:
        status = _sql_status(canon, dialect=backend)

    required_metadata = {
        "execution_kind",
        "supports_lazy",
        "supports_streaming",
        "materializes_full_panel",
        "supports_nulls",
        "supports_nan",
        "supports_inf",
        "supports_scalar_broadcast",
        "supports_group",
        "supports_window",
        "supports_min_periods",
    }
    if status == "production_safe" and backend != "pandas_numpy":
        missing = sorted(required_metadata.difference(backend_meta))
        if missing:
            raise RuntimeError(
                f"{canon}/{backend}: production-safe capability metadata missing {missing}"
            )

    notes = ""
    if backend == "clickhouse_sql" and status != "unsupported":
        notes = "dialect=clickhouse; verify per deployment"
    return BackendCapability(
        canonical=canon,
        backend=backend,
        status=status,
        execution_kind=str(
            backend_meta.get(
                "execution_kind",
                "pandas_numpy_reference" if backend == "pandas_numpy" else "unsupported",
            )
        ),
        estimated_speedup=default_backend_speedup(canon, backend, status),
        supports_nulls=bool(backend_meta.get("supports_nulls", backend == "pandas_numpy")),
        supports_nan=bool(backend_meta.get("supports_nan", backend == "pandas_numpy")),
        supports_inf=bool(backend_meta.get("supports_inf", backend == "pandas_numpy")),
        supports_scalar_broadcast=bool(
            backend_meta.get("supports_scalar_broadcast", backend == "pandas_numpy")
        ),
        supports_min_periods=bool(
            backend_meta.get("supports_min_periods", backend == "pandas_numpy")
        ),
        supports_group=bool(
            backend_meta.get("supports_group", scope in {"cs", "group"})
        ),
        supports_window=bool(backend_meta.get("supports_window", scope == "ts")),
        supports_lazy=bool(backend_meta.get("supports_lazy", False)),
        supports_streaming=bool(backend_meta.get("supports_streaming", False)),
        materializes_full_panel=bool(
            backend_meta.get("materializes_full_panel", backend == "pandas_numpy")
        ),
        notes=notes,
    )


def summarize_operator(canonical: str) -> OperatorCapabilitySummary:
    from cleaned_operators.operator_policy import POLARS_PARITY_VERIFIED
    from cleaned_operators.operator_spec import build_operator_spec

    canon = resolve_canonical(canonical)
    if canon == "if_else":
        canon = "where"
    spec = build_operator_spec(canon)
    return OperatorCapabilitySummary(
        canonical=canon,
        pandas_numpy=_pandas_status(canon),
        polars=_polars_status(canon),
        duckdb_sql=_sql_status(canon, dialect="duckdb_sql"),
        clickhouse_sql=_sql_status(canon, dialect="clickhouse_sql"),
        allow_in_production=bool(spec.allow_in_production) if spec is not None else False,
        parity_verified=canon in POLARS_PARITY_VERIFIED,
        polars_long_tier=spec.polars_long_tier if spec is not None else "unsupported",
    )


def build_capability_matrix(
    canonicals: Sequence[str] | None = None,
) -> list[OperatorCapabilitySummary]:
    from cleaned_operators.registry import OperatorRegistry

    if canonicals is None:
        names = sorted(
            {
                resolve_canonical(c)
                for c in OperatorRegistry.list_canonical()
                if OperatorRegistry.backends_for(c)
            }
        )
    else:
        names = sorted({resolve_canonical(c) for c in canonicals})
    return [summarize_operator(c) for c in names]


def export_flat_capabilities(
    canonicals: Sequence[str] | None = None,
) -> list[BackendCapability]:
    rows: list[BackendCapability] = []
    for summary in build_capability_matrix(canonicals):
        for backend in (
            "pandas_numpy",
            "polars",
            "duckdb_sql",
            "clickhouse_sql",
        ):
            rows.append(capability_for(summary.canonical, backend))
    return rows


def supports_polars(canonical: str, *, mode: str = "production") -> bool:
    status = _polars_status(resolve_canonical(canonical))
    return status == "production_safe" if mode == "production" else status != "unsupported"


def supports_sql(
    canonical: str,
    data_source_kind: str = "duckdb",
    *,
    mode: str = "production",
) -> bool:
    canon = resolve_canonical(canonical)
    dialect: BackendName = (
        "clickhouse_sql"
        if data_source_kind.lower() in {"clickhouse", "ch"}
        else "duckdb_sql"
    )
    status = _sql_status(canon, dialect=dialect)
    return status == "production_safe" if mode == "production" else status != "unsupported"


def supports_pandas(canonical: str, *, mode: str = "production") -> bool:
    status = _pandas_status(resolve_canonical(canonical))
    return status == "production_safe" if mode == "production" else status != "unsupported"


def _backend_cost(
    canonical: str,
    backend: str,
    *,
    row_count_estimate: int | None,
    requires_conversion: bool,
) -> float:
    from backend.operator_cost import estimate_backend_cost

    return estimate_backend_cost(
        canonical,
        backend,
        row_count_estimate=row_count_estimate,
        requires_conversion=requires_conversion,
    )


def get_best_backend(
    name: str,
    *,
    mode: str = "production",
    data_source_kind: str = "memory",
    row_count_estimate: int | None = None,
    prefer: str = "auto",
    allow_unverified_backend: bool = False,
) -> tuple[object | None, str]:
    """Select the cheapest eligible operator backend.

    Production never silently falls back to an implementation that lacks
    current evidence.  SQL is normally selected at the plan/subtree layer, but
    explicit ``prefer='sql'`` remains supported.
    """
    import os

    from cleaned_operators.registry import OperatorRegistry

    canonical = resolve_canonical(name)
    mode = str(mode or "research").lower()
    backends = OperatorRegistry.backends_for(canonical)
    prod = mode == "production"

    def permitted(registry_backend: str) -> bool:
        if registry_backend == "pandas_numpy":
            status = _pandas_status(canonical)
        elif registry_backend == "polars":
            status = _polars_status(canonical)
        elif registry_backend == "sql":
            dialect: BackendName = (
                "clickhouse_sql"
                if data_source_kind.lower() in {"clickhouse", "ch"}
                else "duckdb_sql"
            )
            status = _sql_status(canonical, dialect=dialect)
        else:
            return False
        if prod:
            return status == "production_safe"
        return status != "unsupported" and (
            allow_unverified_backend
            or status in {"parity_verified", "production_safe"}
        )

    requested = str(prefer or "auto").lower()
    if requested in {"pandas_numpy", "polars", "sql"}:
        op = OperatorRegistry.get(canonical, requested)
        if op is None or not permitted(requested):
            raise UnsupportedOperatorBackendError(
                f"{canonical!r} backend={requested!r} is not eligible in mode={mode!r}"
            )
        return op, requested
    if requested != "auto":
        raise UnsupportedOperatorBackendError(
            f"unknown backend preference {prefer!r}"
        )

    aggressive_requested = (
        os.environ.get("FACTOR_ENGINE_OPERATOR_BACKEND", "").strip().lower()
        in {"auto_aggressive", "aggressive"}
    )
    aggressive = aggressive_requested and not prod and allow_unverified_backend
    candidates: list[tuple[str, float]] = []

    if "pandas_numpy" in backends and (permitted("pandas_numpy") or aggressive):
        candidates.append(
            (
                "pandas_numpy",
                _backend_cost(
                    canonical,
                    "pandas_numpy",
                    row_count_estimate=row_count_estimate,
                    requires_conversion=False,
                ),
            )
        )
    if "polars" in backends and (permitted("polars") or aggressive):
        candidates.append(
            (
                "polars",
                _backend_cost(
                    canonical,
                    "polars",
                    row_count_estimate=row_count_estimate,
                    requires_conversion=True,
                ),
            )
        )

    if not candidates:
        raise UnsupportedOperatorBackendError(
            f"no {'production-certified ' if prod else ''}operator backend for {canonical!r}"
        )

    env_cost = os.environ.get("FACTOR_ENGINE_COST_ROUTING", "").strip().lower()
    use_cost = prod or env_cost in {"1", "true", "yes", "on"}
    if use_cost and len(candidates) > 1:
        chosen = min(candidates, key=lambda item: (item[1], item[0]))[0]
    else:
        chosen = (
            "polars"
            if any(candidate == "polars" for candidate, _ in candidates)
            else candidates[0][0]
        )

    op = OperatorRegistry.get(canonical, chosen)
    if op is None:
        raise UnsupportedOperatorBackendError(
            f"selected backend {chosen!r} disappeared for {canonical!r}"
        )
    return op, chosen
