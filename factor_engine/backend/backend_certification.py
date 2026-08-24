# -*- coding: utf-8 -*-
"""Backend-specific production certification.

An operator's semantic lifecycle is independent from the set of engines that
can execute it efficiently.  This module exposes that orthogonal view and is
intended for routing, manifests and monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendStatus = Literal["production", "candidate", "research", "unsupported"]


@dataclass(frozen=True)
class BackendCertification:
    canonical: str
    pandas_numpy: BackendStatus
    polars: BackendStatus
    duckdb_sql: BackendStatus
    portable: bool
    has_any_production_backend: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical": self.canonical,
            "pandas_numpy": self.pandas_numpy,
            "polars": self.polars,
            "duckdb_sql": self.duckdb_sql,
            "portable": self.portable,
            "has_any_production_backend": self.has_any_production_backend,
        }


def _pandas_status(canonical: str) -> BackendStatus:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if "pandas_numpy" not in OperatorRegistry.backends_for(canonical):
        return "unsupported"
    catalog = OperatorRegistry._catalog.get(canonical, {})
    pandas_meta = dict((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
    if bool(pandas_meta.get("production_certified")):
        return "production"

    # Existing daily primitives have a Pandas reference by construction; their
    # production admission remains governed by operator_spec/evidence.
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    if canonical in DAILY_CANONICALS:
        return "production"
    return "research"


def _polars_status(canonical: str) -> BackendStatus:
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if "polars" not in OperatorRegistry.backends_for(canonical):
        return "unsupported"
    from factor_engine.backend.primitive_evidence import (
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    if (
        canonical in POLARS_REFERENCE_PARITY_VERIFIED
        and canonical in POLARS_EDGE_VERIFIED
        and canonical in POLARS_NO_FALLBACK_VERIFIED
    ):
        return "production"
    if canonical in POLARS_REFERENCE_PARITY_VERIFIED:
        return "candidate"
    return "research"


def _duckdb_status(canonical: str) -> BackendStatus:
    from factor_engine.backend.primitive_evidence import (
        DUCKDB_EDGE_VERIFIED,
        DUCKDB_NAN_EDGE_VERIFIED,
        DUCKDB_REAL_SQL_VERIFIED,
        DUCKDB_REFERENCE_PARITY_VERIFIED,
    )
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    if canonical not in SQL_IMPLEMENTED_CANONICALS:
        return "unsupported"
    if (
        canonical in DUCKDB_REFERENCE_PARITY_VERIFIED
        and canonical in DUCKDB_REAL_SQL_VERIFIED
        and (
            canonical in DUCKDB_EDGE_VERIFIED
            or canonical in DUCKDB_NAN_EDGE_VERIFIED
        )
    ):
        return "production"
    if canonical in DUCKDB_REFERENCE_PARITY_VERIFIED:
        return "candidate"
    return "research"


def backend_certification(canonical: str) -> BackendCertification:
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    canon = OperatorRegistry.resolve_canonical_strict(canonical)
    pd_status = _pandas_status(canon)
    pl_status = _polars_status(canon)
    db_status = _duckdb_status(canon)
    statuses = (pd_status, pl_status, db_status)
    return BackendCertification(
        canonical=canon,
        pandas_numpy=pd_status,
        polars=pl_status,
        duckdb_sql=db_status,
        portable=all(status == "production" for status in statuses),
        has_any_production_backend=any(status == "production" for status in statuses),
    )


def production_backends(canonical: str) -> tuple[str, ...]:
    cert = backend_certification(canonical)
    out: list[str] = []
    if cert.pandas_numpy == "production":
        out.append("pandas_numpy")
    if cert.polars == "production":
        out.append("polars")
    if cert.duckdb_sql == "production":
        out.append("duckdb_sql")
    return tuple(out)


def build_backend_certification_manifest() -> dict[str, dict[str, object]]:
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    return {
        canonical: backend_certification(canonical).to_dict()
        for canonical in sorted(OperatorRegistry.list_canonical())
    }
