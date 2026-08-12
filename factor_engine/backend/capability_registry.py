# -*- coding: utf-8 -*-
"""Unified BackendCapability authority (MB-P2-001).

This module provides the single source of truth for backend capability queries.
All capability lookups must go through this registry to ensure:
- Consistent versioning across the codebase
- No fragmentation between registry/cost/polars-long/sql-lowerer
- Explicit capability version tracking
- Centralized capability evolution

Prior to this unification, capability knowledge was scattered across:
- backend.operator_capability (capability_for, supports_*)
- backend.polars_long_policy (POLARS_LONG_NATIVE, tier classification)
- backend.sql_tiers (SQL_IMPLEMENTED_CANONICALS)
- backend.plan_cost_router (inline capability checks)

This registry delegates to existing authorities but provides a unified interface
and version binding.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

# Version: incremented when capability semantics change
CAPABILITY_REGISTRY_VERSION = "v2.0.0"


class BackendKind(str, Enum):
    """Physical backend execution engines."""
    PANDAS_NUMPY = "pandas_numpy"
    POLARS = "polars"
    POLARS_LONG = "polars_long"
    DUCKDB_SQL = "duckdb_sql"
    CLICKHOUSE_SQL = "clickhouse_sql"
    Q_KDB = "q_kdb"  # Future


class CapabilityLevel(str, Enum):
    """Backend capability levels (unified from CapabilityStatus)."""
    UNSUPPORTED = "unsupported"
    IMPLEMENTED = "implemented"
    PARITY_VERIFIED = "parity_verified"
    PRODUCTION_SAFE = "production_safe"


class ExecutionKind(str, Enum):
    """How a backend executes an operator."""
    UNSUPPORTED = "unsupported"
    NATIVE_EXPR = "native_expr"  # Polars expr, SQL native
    NATIVE_GROUP = "native_group"  # Native group-by aggregation
    NATIVE_STREAMING = "native_streaming"  # Streaming-capable
    DELEGATE_PYTHON = "delegate_python"  # Python UDF/map_groups
    DELEGATE_PANDAS = "delegate_pandas"  # Falls back to Pandas
    REFERENCE = "reference"  # Pandas reference implementation


@dataclass(frozen=True)
class BackendCapabilityRecord:
    """Complete capability record for a canonical×backend pair."""
    canonical: str
    backend: BackendKind
    level: CapabilityLevel
    execution_kind: ExecutionKind

    # Semantic support flags
    supports_nulls: bool = False
    supports_nan: bool = False
    supports_inf: bool = False
    supports_scalar_broadcast: bool = False
    supports_min_periods: bool = False
    supports_group: bool = False
    supports_window: bool = False

    # Execution mode flags
    supports_lazy: bool = False
    supports_streaming: bool = False
    materializes_full_panel: bool = False

    # Cost estimation
    estimated_speedup: float = 1.0

    # Version tracking
    registry_version: str = CAPABILITY_REGISTRY_VERSION

    # Notes
    notes: str = ""

    def is_production_eligible(self) -> bool:
        """Check if this capability is production-safe."""
        return self.level == CapabilityLevel.PRODUCTION_SAFE

    def is_native_execution(self) -> bool:
        """Check if execution is truly native (not delegate)."""
        return self.execution_kind in {
            ExecutionKind.NATIVE_EXPR,
            ExecutionKind.NATIVE_GROUP,
            ExecutionKind.NATIVE_STREAMING,
        }


@dataclass(frozen=True)
class CapabilityQueryResult:
    """Result of a capability query with reasoning."""
    supported: bool
    production_safe: bool
    record: BackendCapabilityRecord | None
    reason: str
    registry_version: str = CAPABILITY_REGISTRY_VERSION


class BackendCapabilityRegistry:
    """Unified capability authority.

    This registry provides the single source of truth for all backend capability
    queries. It delegates to existing subsystems but ensures consistent versioning
    and prevents fragmentation.

    Design principles (MB-P2-001):
    - Single version number for all capability state
    - No inline capability checks in router/cost/emitter
    - Explicit version tracking for physical plan cache invalidation
    - Clear separation: registry owns capability, cost model owns estimates
    """

    _version_hash: str | None = None

    @classmethod
    def version(cls) -> str:
        """Return current capability registry version."""
        return CAPABILITY_REGISTRY_VERSION

    @classmethod
    def version_hash(cls) -> str:
        """Return hash of capability state for cache keys."""
        if cls._version_hash is not None:
            return cls._version_hash

        # Hash includes:
        # - Registry version
        # - Polars NATIVE set hash
        # - SQL IMPLEMENTED set hash
        # - Backend evidence hashes

        from backend.polars_long_policy import POLARS_LONG_NATIVE
        from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

        components = [
            CAPABILITY_REGISTRY_VERSION,
            str(sorted(POLARS_LONG_NATIVE)),
            str(sorted(SQL_IMPLEMENTED_CANONICALS)),
        ]

        try:
            from backend.primitive_evidence import evidence_generation
            components.append(str(evidence_generation()))
        except Exception:
            pass

        combined = "|".join(components)
        cls._version_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return cls._version_hash

    @classmethod
    def query(
        cls,
        canonical: str,
        backend: BackendKind | str,
        *,
        mode: Literal["production", "research"] = "production",
        data_source_kind: str = "duckdb",
        bound_params: dict[str, Any] | None = None,
    ) -> CapabilityQueryResult:
        """Query capability for canonical×backend×params.

        Args:
            canonical: Operator canonical name
            backend: Physical backend kind
            mode: Execution mode (production requires production_safe)
            data_source_kind: Data source dialect for SQL backends
            bound_params: Bound call parameters (for SQL validation)

        Returns:
            Complete capability query result with reasoning
        """
        from backend.operator_capability import (
            backend_status,
            resolve_canonical,
            BackendName,
        )

        canon = resolve_canonical(canonical)

        # Normalize backend kind
        if isinstance(backend, str):
            backend_str = backend.lower()
            if backend_str in {"polars_long", "polars"}:
                backend_kind = BackendKind.POLARS
            elif backend_str == "pandas_numpy":
                backend_kind = BackendKind.PANDAS_NUMPY
            elif backend_str in {"duckdb_sql", "sql"}:
                backend_kind = BackendKind.DUCKDB_SQL
            elif backend_str == "clickhouse_sql":
                backend_kind = BackendKind.CLICKHOUSE_SQL
            elif backend_str == "q_kdb":
                backend_kind = BackendKind.Q_KDB
            else:
                return CapabilityQueryResult(
                    supported=False,
                    production_safe=False,
                    record=None,
                    reason=f"Unknown backend: {backend}",
                )
        else:
            backend_kind = backend

        # Handle SQL bound params validation
        if backend_kind in {BackendKind.DUCKDB_SQL, BackendKind.CLICKHOUSE_SQL} and bound_params:
            from backend.operator_capability import check_call_capability

            dialect = "duckdb_sql" if backend_kind == BackendKind.DUCKDB_SQL else "clickhouse_sql"
            decision = check_call_capability(canon, bound_params, dialect=dialect)

            if not decision.supported:
                return CapabilityQueryResult(
                    supported=False,
                    production_safe=False,
                    record=None,
                    reason=decision.reason,
                )

            # Build record from decision
            record = cls._build_record(canon, backend_kind, data_source_kind)
            return CapabilityQueryResult(
                supported=True,
                production_safe=decision.production_safe,
                record=record,
                reason=decision.reason,
            )

        # Standard capability lookup
        backend_name_map = {
            BackendKind.PANDAS_NUMPY: "pandas_numpy",
            BackendKind.POLARS: "polars",
            BackendKind.DUCKDB_SQL: "duckdb_sql",
            BackendKind.CLICKHOUSE_SQL: "clickhouse_sql",
        }

        if backend_kind not in backend_name_map:
            return CapabilityQueryResult(
                supported=False,
                production_safe=False,
                record=None,
                reason=f"Backend {backend_kind} not yet implemented",
            )

        backend_name: BackendName = backend_name_map[backend_kind]  # type: ignore
        status = backend_status(canon, backend_name, data_source_kind=data_source_kind)

        supported = status != "unsupported"
        production_safe = status == "production_safe"

        if mode == "production" and not production_safe:
            record = cls._build_record(canon, backend_kind, data_source_kind) if supported else None
            return CapabilityQueryResult(
                supported=False,
                production_safe=False,
                record=record,
                reason=f"Status {status} insufficient for production",
            )

        record = cls._build_record(canon, backend_kind, data_source_kind)
        return CapabilityQueryResult(
            supported=supported,
            production_safe=production_safe,
            record=record,
            reason=f"Status: {status}",
        )

    @classmethod
    def _build_record(
        cls,
        canonical: str,
        backend: BackendKind,
        data_source_kind: str,
    ) -> BackendCapabilityRecord:
        """Build complete capability record."""
        from backend.operator_capability import capability_for, BackendName

        backend_name_map = {
            BackendKind.PANDAS_NUMPY: "pandas_numpy",
            BackendKind.POLARS: "polars",
            BackendKind.DUCKDB_SQL: "duckdb_sql",
            BackendKind.CLICKHOUSE_SQL: "clickhouse_sql",
        }

        backend_name: BackendName = backend_name_map[backend]  # type: ignore
        cap = capability_for(canonical, backend_name)

        # Map old CapabilityStatus to CapabilityLevel
        level_map = {
            "unsupported": CapabilityLevel.UNSUPPORTED,
            "implemented": CapabilityLevel.IMPLEMENTED,
            "parity_verified": CapabilityLevel.PARITY_VERIFIED,
            "production_safe": CapabilityLevel.PRODUCTION_SAFE,
        }
        level = level_map.get(cap.status, CapabilityLevel.UNSUPPORTED)

        # Map execution_kind string to ExecutionKind enum
        execution_kind_map = {
            "unsupported": ExecutionKind.UNSUPPORTED,
            "native_expr": ExecutionKind.NATIVE_EXPR,
            "native_streaming": ExecutionKind.NATIVE_STREAMING,
            "python_udf": ExecutionKind.DELEGATE_PYTHON,
            "pandas_fallback": ExecutionKind.DELEGATE_PANDAS,
            "pandas_materialization_fallback": ExecutionKind.DELEGATE_PANDAS,
            "pandas_numpy_reference": ExecutionKind.REFERENCE,
        }
        exec_kind = execution_kind_map.get(
            cap.execution_kind,
            ExecutionKind.UNSUPPORTED,
        )

        return BackendCapabilityRecord(
            canonical=canonical,
            backend=backend,
            level=level,
            execution_kind=exec_kind,
            supports_nulls=cap.supports_nulls,
            supports_nan=cap.supports_nan,
            supports_inf=cap.supports_inf,
            supports_scalar_broadcast=cap.supports_scalar_broadcast,
            supports_min_periods=cap.supports_min_periods,
            supports_group=cap.supports_group,
            supports_window=cap.supports_window,
            supports_lazy=cap.supports_lazy,
            supports_streaming=cap.supports_streaming,
            materializes_full_panel=cap.materializes_full_panel,
            estimated_speedup=cap.estimated_speedup,
            notes=cap.notes,
        )

    @classmethod
    def supports_backend(
        cls,
        canonical: str,
        backend: BackendKind | str,
        *,
        mode: Literal["production", "research"] = "production",
        data_source_kind: str = "duckdb",
    ) -> bool:
        """Check if canonical supports backend at given mode."""
        result = cls.query(
            canonical,
            backend,
            mode=mode,
            data_source_kind=data_source_kind,
        )
        return result.supported and (
            result.production_safe if mode == "production" else True
        )

    @classmethod
    def list_backends(
        cls,
        canonical: str,
        *,
        mode: Literal["production", "research"] = "production",
    ) -> list[BackendKind]:
        """List all backends supporting canonical at given mode."""
        backends = []
        for backend in [
            BackendKind.PANDAS_NUMPY,
            BackendKind.POLARS,
            BackendKind.DUCKDB_SQL,
        ]:
            if cls.supports_backend(canonical, backend, mode=mode):
                backends.append(backend)
        return backends


# Convenience functions for migration compatibility

def supports_polars(
    canonical: str,
    *,
    mode: Literal["production", "research"] = "production",
) -> bool:
    """Unified polars capability check (delegates to registry)."""
    return BackendCapabilityRegistry.supports_backend(
        canonical,
        BackendKind.POLARS,
        mode=mode,
    )


def supports_sql(
    canonical: str,
    *,
    data_source_kind: str = "duckdb",
    mode: Literal["production", "research"] = "production",
) -> bool:
    """Unified SQL capability check (delegates to registry)."""
    backend = (
        BackendKind.CLICKHOUSE_SQL
        if data_source_kind.lower() in {"clickhouse", "ch"}
        else BackendKind.DUCKDB_SQL
    )
    return BackendCapabilityRegistry.supports_backend(
        canonical,
        backend,
        mode=mode,
        data_source_kind=data_source_kind,
    )


def supports_pandas(
    canonical: str,
    *,
    mode: Literal["production", "research"] = "production",
) -> bool:
    """Unified pandas capability check (delegates to registry)."""
    return BackendCapabilityRegistry.supports_backend(
        canonical,
        BackendKind.PANDAS_NUMPY,
        mode=mode,
    )
