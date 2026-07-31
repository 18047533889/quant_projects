# -*- coding: utf-8 -*-
"""Single authority for evidence-constrained operator backend selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RequestedBackend = Literal[
    "auto", "pandas_numpy", "polars", "sql", "duckdb_sql", "clickhouse_sql"
]
FallbackPolicy = Literal["error", "allow"]


@dataclass(frozen=True)
class BackendSelection:
    operator: object
    backend: str
    canonical: str
    requested_backend: str
    fell_back: bool = False


def _assert_semantic_production_evidence(canonical: str, run_mode: str) -> None:
    """Extended factor operators require the all-factor semantic audit artifact."""
    if str(run_mode or "research").lower() != "production":
        return
    from cleaned_operators.operator_surface import DAILY_CANONICALS

    if canonical in DAILY_CANONICALS:
        # Daily primitives are governed by primitive backend evidence.
        return
    from backend.factor_operator_evidence import (
        factor_operator_evidence_valid,
        pandas_reference_production_safe,
        validation_errors,
    )
    if not factor_operator_evidence_valid() or not pandas_reference_production_safe(canonical):
        detail = "; ".join(validation_errors()[:3])
        raise RuntimeError(
            f"{canonical!r} is not semantically certified for production by "
            f"factor_operator_verified.json{': ' + detail if detail else ''}"
        )


class BackendRouter:
    """Select only eligible backends; Registry remains a storage/index service."""

    @staticmethod
    def select(
        canonical: str,
        *,
        requested_backend: RequestedBackend = "auto",
        run_mode: str = "production",
        fallback_policy: FallbackPolicy = "error",
        data_source_kind: str = "memory",
        row_count_estimate: int | None = None,
        allow_unverified_backend: bool = False,
    ) -> BackendSelection:
        from backend.operator_capability import (
            UnsupportedOperatorBackendError,
            get_best_backend,
            resolve_canonical,
        )

        name = resolve_canonical(canonical)
        try:
            _assert_semantic_production_evidence(name, run_mode)
        except RuntimeError as exc:
            raise UnsupportedOperatorBackendError(str(exc)) from exc

        requested = str(requested_backend or "auto").lower()
        if requested == "duckdb_sql":
            requested = "sql"
            data_source_kind = "duckdb"
        elif requested == "clickhouse_sql":
            requested = "sql"
            data_source_kind = "clickhouse"
        if requested not in {"auto", "pandas_numpy", "polars", "sql"}:
            raise UnsupportedOperatorBackendError(
                f"unknown requested backend {requested_backend!r} for {name!r}"
            )

        try:
            operator, backend = get_best_backend(
                name,
                mode=run_mode,
                data_source_kind=data_source_kind,
                row_count_estimate=row_count_estimate,
                prefer=requested,
                allow_unverified_backend=allow_unverified_backend,
            )
        except UnsupportedOperatorBackendError as exc:
            if requested == "auto" or fallback_policy != "allow":
                if str(run_mode).lower() == "production" and requested != "auto":
                    raise UnsupportedOperatorBackendError(
                        f"{name!r} backend={requested!r} is not production-safe: {exc}"
                    ) from exc
                raise
            operator, backend = get_best_backend(
                name,
                mode=run_mode,
                data_source_kind=data_source_kind,
                row_count_estimate=row_count_estimate,
                prefer="auto",
                allow_unverified_backend=allow_unverified_backend,
            )
        if operator is None:
            raise UnsupportedOperatorBackendError(
                f"backend selection returned no implementation for {name!r}"
            )
        return BackendSelection(
            operator=operator,
            backend=backend,
            canonical=name,
            requested_backend=requested,
            fell_back=requested != "auto" and backend != requested,
        )
