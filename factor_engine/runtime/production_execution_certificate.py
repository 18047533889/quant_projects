# -*- coding: utf-8 -*-
"""R39-P1-PERF-082: compile-time ``ProductionExecutionCertificate``.

Problem (spec §24): the production hard gate re-imports and re-walks the plan
tree for every root.  With 10k factors that is 10k tree walks of the compile
stage just to re-check what the router already computed.

Fix: at compile time (``backend.plan_cost_router.choose_plan_route``) we build
a small immutable certificate that binds

    structural_hash      -> sha256 of ``planner.plan_hash.structural_key(plan)``
    bound_ops            -> frozenset of canonical ops (already computed)
    backend_eligibility  -> frozenset of backend names the router accepted
    output_shape_hash    -> sha256 of (row_count_estimate, occurrence count)
    certificate_hash     -> sha256 over the four fields above

At runtime the executor records a *runtime backend event*
``{"backend": ..., "execution_kind": ..., "no_fallback": bool}`` and validates in
O(1) — compare ``certificate_hash`` (recomputed over frozen fields, bounded, no
tree walk) + the event's backend against ``backend_eligibility`` + the
``no_fallback`` flag.  No plan import, no tree walk, no capability lookup.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

#: Backend names used by the runtime executors (HybridExecutor classification).
#: The certificate's eligibility set is normalized to these names so a
#: ``polars_panel`` route and a ``polars`` executor event match.
_EXECUTOR_BACKEND_ALIASES: dict[str, str] = {
    "pandas_numpy": "pandas_numpy",
    "polars": "polars",
    "polars_panel": "polars",
    "polars_long": "polars",
    "duckdb_sql": "duckdb_sql",
    "clickhouse_sql": "duckdb_sql",
    "sql": "duckdb_sql",
    "hybrid": "hybrid",
}


def normalize_backend(name: str | None) -> str:
    """Map a router/planner backend name to the executor-normalized name."""
    n = str(name or "")
    return _EXECUTOR_BACKEND_ALIASES.get(n, n)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProductionExecutionCertificate:
    """Compile-time certificate for O(1) runtime backend validation.

    All fields are immutable.  ``certificate_hash`` is the sha256 of the four
    semantic fields and is what the runtime compares for integrity.
    """

    structural_hash: str
    bound_ops: frozenset[str]
    backend_eligibility: frozenset[str]
    output_shape_hash: str
    certificate_hash: str
    # R40 #141: backend 选择链 —— requested_backend / resolved_dialect /
    # datasource_identity 显式记录在证书里（如 clickhouse_sql -> dialect
    # ``clickhouse``），runtime backend 事件校验不再把 clickhouse 路由静默当
    # 无记录的 DuckDB 路径。
    requested_backend: str = ""
    resolved_dialect: str = ""
    datasource_identity: str = ""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        structural_hash: str,
        bound_ops: Iterable[str],
        backend_eligibility: Iterable[str],
        output_shape_hash: str,
        requested_backend: str = "",
        resolved_dialect: str = "",
        datasource_identity: str = "",
    ) -> "ProductionExecutionCertificate":
        """Build a certificate with a self-consistent ``certificate_hash``."""
        ops = frozenset(str(op) for op in bound_ops)
        elig = frozenset(normalize_backend(b) for b in backend_eligibility)
        payload = {
            "structural_hash": str(structural_hash or ""),
            "bound_ops": sorted(ops),
            "backend_eligibility": sorted(elig),
            "output_shape_hash": str(output_shape_hash or ""),
            "requested_backend": str(requested_backend or ""),
            "resolved_dialect": str(resolved_dialect or ""),
            "datasource_identity": str(datasource_identity or ""),
        }
        cert_hash = _digest(payload)
        return cls(
            structural_hash=str(structural_hash or ""),
            bound_ops=ops,
            backend_eligibility=elig,
            output_shape_hash=str(output_shape_hash or ""),
            certificate_hash=cert_hash,
            requested_backend=str(requested_backend or ""),
            resolved_dialect=str(resolved_dialect or ""),
            datasource_identity=str(datasource_identity or ""),
        )

    # ------------------------------------------------------------------
    # O(1) validation
    # ------------------------------------------------------------------
    def recompute_hash(self) -> str:
        """Recompute the certificate hash over the frozen fields (O(1), bounded)."""
        payload = {
            "structural_hash": self.structural_hash,
            "bound_ops": sorted(self.bound_ops),
            "backend_eligibility": sorted(self.backend_eligibility),
            "output_shape_hash": self.output_shape_hash,
            "requested_backend": self.requested_backend,
            "resolved_dialect": self.resolved_dialect,
            "datasource_identity": self.datasource_identity,
        }
        return _digest(payload)

    def validate(
        self,
        runtime_backend_event: dict[str, Any] | None,
        allowed_fallbacks: Iterable[str] = (),
    ) -> bool:
        """O(1) validation — no tree walk, no capability lookup.

        Checks (in order):

        1. integrity: ``certificate_hash`` must equal a recompute of the frozen
           fields (a stale / tampered certificate fails closed);
        2. ``no_fallback``: if the runtime event reports a fallback, the
           executed backend must be explicitly listed in ``allowed_fallbacks``;
        3. backend match: the executed backend (normalized) must be in
           ``backend_eligibility``;
        4. R21-P026 dialect identity: if both the certificate
           (``resolved_dialect``) and the runtime event (``dialect``) declare a
           datasource dialect, they must match — a DuckDB-issued certificate
           must NOT validate a ClickHouse execution (and vice versa) even
           though both normalize to the ``duckdb_sql`` executor family.
        """
        if runtime_backend_event is None:
            return False
        if self.certificate_hash != self.recompute_hash():
            return False
        backend = normalize_backend(runtime_backend_event.get("backend"))
        if not backend:
            return False
        no_fallback = bool(runtime_backend_event.get("no_fallback", True))
        if not no_fallback:
            allowed = {normalize_backend(f) for f in (allowed_fallbacks or ())}
            if backend not in allowed:
                return False
        if backend not in self.backend_eligibility:
            return False
        event_dialect = str(runtime_backend_event.get("dialect") or "")
        cert_dialect = str(self.resolved_dialect or "")
        if event_dialect and cert_dialect and event_dialect != cert_dialect:
            return False
        return True

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_hash": self.structural_hash,
            "bound_ops": sorted(self.bound_ops),
            "backend_eligibility": sorted(self.backend_eligibility),
            "output_shape_hash": self.output_shape_hash,
            "certificate_hash": self.certificate_hash,
            "requested_backend": self.requested_backend,
            "resolved_dialect": self.resolved_dialect,
            "datasource_identity": self.datasource_identity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProductionExecutionCertificate":
        return cls.build(
            structural_hash=str(payload.get("structural_hash") or ""),
            bound_ops=list(payload.get("bound_ops") or ()),
            backend_eligibility=list(payload.get("backend_eligibility") or ()),
            output_shape_hash=str(payload.get("output_shape_hash") or ""),
            requested_backend=str(payload.get("requested_backend") or ""),
            resolved_dialect=str(payload.get("resolved_dialect") or ""),
            datasource_identity=str(payload.get("datasource_identity") or ""),
        )


# ---------------------------------------------------------------------------
# module-level convenience (matches the ``validate(certificate, event, ...)``
# call shape the spec and HybridExecutor wire into)
# ---------------------------------------------------------------------------
def validate(
    certificate: ProductionExecutionCertificate | None,
    runtime_backend_event: dict[str, Any] | None,
    allowed_fallbacks: Iterable[str] = (),
) -> bool:
    """O(1) validate a certificate against a runtime backend event.

    ``None`` certificate fails closed (no certificate → no compile-time proof).
    """
    if certificate is None:
        return False
    return certificate.validate(runtime_backend_event, allowed_fallbacks)


__all__ = [
    "ProductionExecutionCertificate",
    "normalize_backend",
    "validate",
]
