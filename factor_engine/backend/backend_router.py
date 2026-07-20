# -*- coding: utf-8 -*-
"""Single authority for operator backend selection."""
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


class BackendRouter:
    """Select a backend; Registry remains a storage/index service only."""

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
        except UnsupportedOperatorBackendError:
            if requested == "auto" or fallback_policy != "allow":
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
