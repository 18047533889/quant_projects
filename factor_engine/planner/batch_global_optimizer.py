# -*- coding: utf-8 -*-
"""Compatibility aliases for the runtime-owned physical batch optimizer."""

from factor_engine.runtime.multibackend.batch_global_optimizer import (
    BatchGlobalOptimizer,
    BatchOptimizationResult,
    NodeBackendChoice,
    SharedNodeBenefit,
    estimate_shared_benefits,
    optimize_batch_global,
)

__all__ = [
    "BatchGlobalOptimizer",
    "BatchOptimizationResult",
    "NodeBackendChoice",
    "SharedNodeBenefit",
    "estimate_shared_benefits",
    "optimize_batch_global",
]
