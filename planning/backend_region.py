# -*- coding: utf-8 -*-
"""Backend region contracts and representation types (re-export shim).

R21-PLANNER-TYPE-UNIFICATION: This module is a compatibility re-export of the
single physical planner ABI owned by ``planner.backend_region``.  The dual
PhysicalBackend/Representation/StateContract/BackendRegion type systems are
eliminated — ``planner.backend_region`` is the sole physical ABI authority
(PANDAS_NUMPY/POLARS_PANEL/POLARS_LONG/DUCKDB_SQL/CLICKHOUSE_SQL/Q_KDB).

Numba is an Accelerator (``backend.contracts.Accelerator.NUMBA_CPU`` /
``ExecutionKind.NUMBA_CPU_KERNEL``), NOT a backend.  Arrow is an interchange
Representation (``Representation.ARROW_TABLE``), NOT a backend.  Neither may
appear on ``PhysicalBackend``.

No code in this file defines physical types; everything is imported from the
single authority so ``planning.*`` and ``planner.*`` consumers observe the same
enum values, dataclasses, and validation semantics.
"""
from __future__ import annotations

from planner.backend_region import (
    BackendRegion,
    ExecutionAxis,
    PhysicalBackend,
    PhysicalProperties,
    PhysicalRegionPlan,
    Representation,
    StateContract,
    TransferEdge,
    TransferTransform,
    backend_supports_direct_sink,
    estimate_transfer_cost_ms,
    infer_representation,
    infer_transfer_transform,
    normalize_backend_name,
    supports_streaming,
)

# MB-P2-002 name kept for planning/transfer_edge.py and planning consumers.
# ``planner.backend_region`` calls the same concept ``PhysicalProperties``.
PhysicalProperty = PhysicalProperties

__all__ = [
    "BackendRegion",
    "ExecutionAxis",
    "PhysicalBackend",
    "PhysicalProperties",
    "PhysicalProperty",
    "PhysicalRegionPlan",
    "Representation",
    "StateContract",
    "TransferEdge",
    "TransferTransform",
    "backend_supports_direct_sink",
    "estimate_transfer_cost_ms",
    "infer_representation",
    "infer_transfer_transform",
    "normalize_backend_name",
    "supports_streaming",
]
