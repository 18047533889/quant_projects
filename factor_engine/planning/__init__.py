# -*- coding: utf-8 -*-
"""Multi-backend physical region planning.

This package implements the core architecture for cost-optimal backend region
partitioning across Pandas, Polars, DuckDB, and q/K execution engines.

Key modules:
- physical_region_plan: Core data structures for backend regions and physical plans
- backend_region: Backend region contracts and representation types
- transfer_edge: Transfer edge cost modeling and conversion tracking
- region_optimizer: Cost-optimal region partitioning algorithm
- region_scheduler: Region DAG scheduler with resource tokens
- execution_axis: Execution axis classification and sharding rules
- data_shape: Metadata-only data shape estimation
"""
from __future__ import annotations

__all__ = [
    "BackendRegion",
    "TransferEdge",
    "PhysicalRegionPlan",
    "Representation",
    "PhysicalBackend",
]
