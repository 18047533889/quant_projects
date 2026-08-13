# -*- coding: utf-8 -*-
"""Multi-backend performance and memory optimization framework.

This package implements 27 P1-level optimizations for FactorEngine's multi-backend
execution system, covering memory management (MB-P1-001 to MB-P1-010) and performance
improvements (MB-P1-011 to MB-P1-027).

Memory Optimizations:
    - MB-P1-001: DataShapeEstimate strict metadata-only
    - MB-P1-002: Fine-grained memory model
    - MB-P1-003: Batch-global optimizer
    - MB-P1-004: CSE native representation
    - MB-P1-005: Liveness analysis
    - MB-P1-006: Streaming-first execution
    - MB-P1-007: Memory budget token management
    - MB-P1-008: Concurrent region isolation
    - MB-P1-009: Spill strategy
    - MB-P1-010: Arrow zero-copy boundary

Performance Optimizations:
    - MB-P1-011: Polars Lazy DAG fusion
    - MB-P1-012: DuckDB Arrow boundary
    - MB-P1-013: Source scan cost estimation
    - MB-P1-014: Concurrent token management
    - MB-P1-015: Region operator fusion
    - MB-P1-016: Batch transfer
    - MB-P1-017: Cost model calibration
    - MB-P1-018: Physical properties propagation
    - MB-P1-019: Smart representation selection
    - MB-P1-020: Batch Polars expression compilation
    - MB-P1-021: DuckDB prepared statement reuse
    - MB-P1-022: CSE cache hit rate optimization
    - MB-P1-023: Parallel region scheduling
    - MB-P1-024: Dynamic work-stealing
    - MB-P1-025: q/K backend baseline
    - MB-P1-026: Backend capability declaration
    - MB-P1-027: Region boundary cost minimization
"""

# Import memory optimization modules
from .data_shape_estimator import DataShapeEstimator, DataShapeEstimate, estimate_task_shape
from .fine_grained_memory_model import (
    FineGrainedMemoryModel,
    RepresentationCoefficients,
    OperatorMemoryProfile,
    global_memory_model,
)
from .batch_global_optimizer import (
    BatchGlobalOptimizer,
    GlobalOptimizationResult,
    OptimizationOpportunity,
)
from .cse_native_representation import (
    NativeCSECache,
    NativeCSEManager,
    NativeCSEKey,
    CSEEntry,
    global_cse_manager,
)
from .liveness_analyzer import (
    LivenessAnalyzer,
    LivenessTracker,
    LiveRange,
    LivenessState,
)
from .streaming_executor import (
    StreamingExecutionPlanner,
    StreamingExecutor,
    StreamingRegion,
    StreamingExecutionPlan,
)
from .memory_budget_manager import (
    MemoryBudgetManager,
    TokenBasedAdmissionController,
    MemoryToken,
    BudgetAllocation,
    global_budget_manager,
)
from .concurrent_region_isolation import (
    ConcurrentRegionIsolationManager,
    ConcurrentRegionMemoryPool,
    RegionMemoryStats,
    RegionAllocation,
    global_region_manager,
)
from .spill_strategy import (
    SpillStrategyManager,
    SpillStore,
    SpillMetadata,
    SpillStats,
    global_spill_manager,
)
from .arrow_zerocopy_boundary import (
    ArrowZeroCopyBoundary,
    ZeroCopyBoundaryOptimizer,
    TransferMetadata,
    TransferStats,
    global_zerocopy_boundary,
)

__all__ = [
    # Memory optimizations - core classes
    "DataShapeEstimator",
    "DataShapeEstimate",
    "estimate_task_shape",
    "FineGrainedMemoryModel",
    "RepresentationCoefficients",
    "OperatorMemoryProfile",
    "global_memory_model",
    "BatchGlobalOptimizer",
    "GlobalOptimizationResult",
    "OptimizationOpportunity",
    "NativeCSECache",
    "NativeCSEManager",
    "NativeCSEKey",
    "CSEEntry",
    "global_cse_manager",
    "LivenessAnalyzer",
    "LivenessTracker",
    "LiveRange",
    "LivenessState",
    "StreamingExecutionPlanner",
    "StreamingExecutor",
    "StreamingRegion",
    "StreamingExecutionPlan",
    "MemoryBudgetManager",
    "TokenBasedAdmissionController",
    "MemoryToken",
    "BudgetAllocation",
    "global_budget_manager",
    "ConcurrentRegionIsolationManager",
    "ConcurrentRegionMemoryPool",
    "RegionMemoryStats",
    "RegionAllocation",
    "global_region_manager",
    "SpillStrategyManager",
    "SpillStore",
    "SpillMetadata",
    "SpillStats",
    "global_spill_manager",
    "ArrowZeroCopyBoundary",
    "ZeroCopyBoundaryOptimizer",
    "TransferMetadata",
    "TransferStats",
    "global_zerocopy_boundary",
]
