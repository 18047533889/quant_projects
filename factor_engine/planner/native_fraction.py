# -*- coding: utf-8 -*-
"""Backend-specific native implementation fraction calculation (MB-P0-003).

The native fraction must be calculated per backend, not generically across all backends.
A Polars plan should report Polars-native fraction, not SQL-native fraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NativeFractionReport:
    """Backend-specific native implementation coverage report.

    MB-P0-003: Native fraction must be backend-specific. Delegate operations
    do not count as native.
    """

    backend: str
    node_count_total: int
    node_count_native: int
    node_count_delegate: int
    node_count_unsupported: int
    estimated_compute_fraction: float  # Weighted by compute cost
    native_nodes: tuple[str, ...]
    delegate_nodes: tuple[str, ...]
    unsupported_nodes: tuple[str, ...]

    @property
    def node_count_fraction(self) -> float:
        """Simple node count fraction (native / total)."""
        if self.node_count_total == 0:
            return 0.0
        return self.node_count_native / self.node_count_total


def plan_native_subgraph_fraction(
    plan: Any,
    backend: str,
    mode: str = "research",
    data_source_kind: str = "memory",
) -> NativeFractionReport:
    """Calculate backend-specific native implementation fraction.

    MB-P0-003: Must use backend-specific capability checks. A polars backend
    should check polars capability, not SQL capability.

    Args:
        plan: Logical plan node
        backend: Specific backend (pandas_numpy, polars_panel, duckdb_sql, etc.)
        mode: Execution mode (research, production)
        data_source_kind: Source kind for SQL backends (duckdb, clickhouse, memory)

    Returns:
        Backend-specific native fraction report with separate counts for
        native, delegate, and unsupported nodes.
    """
    from factor_engine.backend.operator_capability import (
        supports_pandas,
        supports_polars,
        supports_sql,
    )

    backend_normalized = backend.lower().strip()
    is_sql = backend_normalized in {"duckdb_sql", "clickhouse_sql", "sql"}
    is_polars = backend_normalized in {"polars", "polars_panel", "polars_long"}
    is_pandas = backend_normalized in {"pandas", "pandas_numpy"}

    # MB-P0-003: Choose the correct capability check function
    if is_sql:
        capability_check = lambda op: supports_sql(
            op, data_source_kind=data_source_kind, mode=mode
        )
    elif is_polars:
        capability_check = lambda op: supports_polars(op, mode=mode)
    elif is_pandas:
        capability_check = lambda op: supports_pandas(op, mode=mode)
    else:
        # Unknown backend - assume nothing is native
        capability_check = lambda op: False

    native_ops: list[str] = []
    delegate_ops: list[str] = []
    unsupported_ops: list[str] = []
    all_ops: list[str] = []

    # Delegate detection for polars
    delegate_check = None
    if is_polars:
        try:
            from factor_engine.backend.polars_backend_kind import canonical_polars_is_delegate

            delegate_check = canonical_polars_is_delegate
        except ImportError:
            delegate_check = None

    def walk(node: Any) -> None:
        """Walk the plan tree and classify each operator."""
        if not hasattr(node, "op"):
            return

        op = str(node.op or "")
        if not op or op in {"column", "literal", "plan_ref", "materialized_series"}:
            # Meta operations - O(1) reads, don't count
            for child in getattr(node, "inputs", ()) or ():
                walk(child)
            return

        # Get canonical name
        canonical = op
        try:
            from factor_engine.cleaned_operators.registry import OperatorRegistry

            canonical = OperatorRegistry._aliases.get(op, op)
        except Exception:
            pass

        all_ops.append(canonical)

        # Classify as native, delegate, or unsupported
        if capability_check(canonical):
            # Check if it's a delegate implementation
            if delegate_check and delegate_check(canonical):
                delegate_ops.append(canonical)
            else:
                native_ops.append(canonical)
        else:
            unsupported_ops.append(canonical)

        # Recurse to children
        for child in getattr(node, "inputs", ()) or ():
            walk(child)

    walk(plan)

    total = len(all_ops)
    native_count = len(native_ops)
    delegate_count = len(delegate_ops)
    unsupported_count = len(unsupported_ops)

    # Estimate compute fraction (simple heuristic - could be refined)
    # Assume each node has equal compute weight for now
    compute_fraction = native_count / total if total > 0 else 0.0

    return NativeFractionReport(
        backend=backend_normalized,
        node_count_total=total,
        node_count_native=native_count,
        node_count_delegate=delegate_count,
        node_count_unsupported=unsupported_count,
        estimated_compute_fraction=compute_fraction,
        native_nodes=tuple(native_ops),
        delegate_nodes=tuple(delegate_ops),
        unsupported_nodes=tuple(unsupported_ops),
    )
