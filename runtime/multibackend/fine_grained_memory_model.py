# -*- coding: utf-8 -*-
"""MB-P1-002: Fine-grained memory model (per-representation coefficients).

Provides accurate per-backend memory footprint models accounting for:
- Representation-specific overhead (pandas/polars/arrow/duckdb)
- Operator-specific memory amplification
- Intermediate buffer requirements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepresentationCoefficients:
    """Per-representation memory coefficients.

    Attributes:
        base_overhead: Fixed overhead per DataFrame (bytes)
        per_row_overhead: Overhead per row (bytes)
        per_column_overhead: Overhead per column (bytes)
        index_overhead_factor: Index memory as fraction of data
        copy_multiplier: Multiplier for copy operations
        intermediate_multiplier: Peak intermediate buffer multiplier
        compression_ratio: Effective compression (< 1.0 means compressed)
    """
    base_overhead: int
    per_row_overhead: float
    per_column_overhead: float
    index_overhead_factor: float
    copy_multiplier: float
    intermediate_multiplier: float
    compression_ratio: float


# Empirically calibrated coefficients per backend
_REPRESENTATION_COEFFICIENTS = {
    "pandas": RepresentationCoefficients(
        base_overhead=1024 * 1024,  # 1 MB BlockManager
        per_row_overhead=8.0,  # Index pointer
        per_column_overhead=512.0,  # Block metadata
        index_overhead_factor=0.15,  # 15% for DatetimeIndex
        copy_multiplier=2.0,  # Full copy on mutation
        intermediate_multiplier=1.8,  # Intermediate buffers
        compression_ratio=1.0,  # No compression
    ),
    "polars": RepresentationCoefficients(
        base_overhead=256 * 1024,  # 256 KB
        per_row_overhead=0.5,  # Minimal row overhead
        per_column_overhead=128.0,  # Arrow column metadata
        index_overhead_factor=0.0,  # No index
        copy_multiplier=1.0,  # COW (copy-on-write)
        intermediate_multiplier=1.3,  # Efficient intermediates
        compression_ratio=0.85,  # Arrow compression
    ),
    "arrow": RepresentationCoefficients(
        base_overhead=128 * 1024,  # 128 KB
        per_row_overhead=0.25,
        per_column_overhead=64.0,
        index_overhead_factor=0.0,
        copy_multiplier=1.0,  # Zero-copy where possible
        intermediate_multiplier=1.2,
        compression_ratio=0.80,  # Efficient compression
    ),
    "duckdb": RepresentationCoefficients(
        base_overhead=512 * 1024,  # 512 KB catalog
        per_row_overhead=0.1,  # Column-store efficiency
        per_column_overhead=256.0,  # Chunk metadata
        index_overhead_factor=0.0,
        copy_multiplier=1.0,  # In-place updates
        intermediate_multiplier=1.5,  # Pipeline buffers
        compression_ratio=0.70,  # Best compression
    ),
}


@dataclass(frozen=True)
class OperatorMemoryProfile:
    """Memory profile for specific operator.

    Attributes:
        input_multiplier: Peak = input_size * multiplier
        output_size_factor: Output as fraction of input
        temp_buffer_factor: Temporary buffer as fraction of input
        supports_streaming: Can process in chunks
        max_materialization_rows: Max rows before forced spill
    """
    input_multiplier: float
    output_size_factor: float
    temp_buffer_factor: float
    supports_streaming: bool
    max_materialization_rows: int | None


# Operator-specific memory profiles
_OPERATOR_PROFILES = {
    # Low memory operators
    "add": OperatorMemoryProfile(1.0, 1.0, 0.0, True, None),
    "multiply": OperatorMemoryProfile(1.0, 1.0, 0.0, True, None),
    "col": OperatorMemoryProfile(1.0, 1.0, 0.0, True, None),

    # Window operators (need buffer)
    "ts_mean": OperatorMemoryProfile(1.2, 1.0, 0.2, True, None),
    "ts_std": OperatorMemoryProfile(1.3, 1.0, 0.3, True, None),
    "ts_corr": OperatorMemoryProfile(1.5, 1.0, 0.5, True, 10_000_000),

    # Sorting operators (high memory)
    "rank": OperatorMemoryProfile(2.0, 1.0, 1.0, False, 5_000_000),
    "quantile": OperatorMemoryProfile(2.2, 1.0, 1.2, False, 3_000_000),

    # Cross-sectional operators
    "cs_rank": OperatorMemoryProfile(1.8, 1.0, 0.8, False, None),
    "neutralize": OperatorMemoryProfile(3.0, 1.0, 2.0, False, 1_000_000),

    # Group operators
    "group_mean": OperatorMemoryProfile(1.4, 1.0, 0.4, True, None),
    "group_neutralize": OperatorMemoryProfile(3.5, 1.0, 2.5, False, 500_000),
}


class FineGrainedMemoryModel:
    """Fine-grained per-representation memory model.

    Provides accurate memory footprint prediction accounting for:
    - Backend-specific representation overhead
    - Operator-specific memory amplification
    - Intermediate buffer requirements
    """

    def __init__(self):
        self._coefficients = _REPRESENTATION_COEFFICIENTS.copy()
        self._operator_profiles = _OPERATOR_PROFILES.copy()

    def estimate_operator_memory(
        self,
        operator: str,
        input_bytes: int,
        *,
        representation: str = "pandas",
        rows: int = 500_000,
        cols: int = 1,
    ) -> dict[str, int]:
        """Estimate memory footprint for operator execution.

        Args:
            operator: Operator name
            input_bytes: Input data size (bytes)
            representation: Backend representation
            rows: Number of rows
            cols: Number of columns

        Returns:
            Dict with peak_bytes, output_bytes, temp_bytes
        """
        coeff = self._coefficients.get(representation, self._coefficients["pandas"])
        profile = self._operator_profiles.get(
            operator,
            OperatorMemoryProfile(1.5, 1.0, 0.5, True, None)  # Default profile
        )

        # Apply compression ratio
        effective_input = int(input_bytes * coeff.compression_ratio)

        # Representation overhead
        base = coeff.base_overhead
        row_overhead = int(coeff.per_row_overhead * rows)
        col_overhead = int(coeff.per_column_overhead * cols)
        index_overhead = int(effective_input * coeff.index_overhead_factor)
        representation_overhead = base + row_overhead + col_overhead + index_overhead

        # Operator-specific amplification
        temp_bytes = int(effective_input * profile.temp_buffer_factor)
        output_bytes = int(effective_input * profile.output_size_factor)

        # Peak = input + temp buffers + output (before input release)
        peak_bytes = int(
            effective_input * profile.input_multiplier + temp_bytes + output_bytes
        )
        peak_bytes += representation_overhead

        # Intermediate multiplier for this representation
        peak_bytes = int(peak_bytes * coeff.intermediate_multiplier)

        return {
            "peak_bytes": peak_bytes,
            "output_bytes": output_bytes + representation_overhead // 2,
            "temp_bytes": temp_bytes,
            "representation_overhead": representation_overhead,
            "supports_streaming": profile.supports_streaming,
            "max_materialization_rows": profile.max_materialization_rows,
        }

    def estimate_plan_memory(
        self,
        plan: Any,
        *,
        input_shape: Any | None = None,
        representation: str = "pandas",
    ) -> dict[str, int]:
        """Estimate memory for entire plan execution.

        Args:
            plan: Execution plan
            input_shape: Optional input DataShapeEstimate
            representation: Target representation

        Returns:
            Dict with peak_bytes, output_bytes estimates
        """
        try:
            # Extract plan structure
            op = getattr(plan, "op", "unknown")

            # Get input size
            if input_shape is not None:
                input_bytes = input_shape.total_bytes
                rows = input_shape.rows
                cols = input_shape.cols
            else:
                # Conservative estimate
                rows = 500_000
                cols = 10
                input_bytes = rows * cols * 8  # float64

            # Recurse through plan tree
            inputs = getattr(plan, "inputs", [])
            if inputs:
                # Aggregate child memory requirements
                child_peak = 0
                child_output = 0
                for child_plan in inputs:
                    child_est = self.estimate_plan_memory(
                        child_plan,
                        input_shape=input_shape,
                        representation=representation,
                    )
                    child_peak = max(child_peak, child_est["peak_bytes"])
                    child_output += child_est["output_bytes"]

                # This operator consumes child outputs
                input_bytes = max(input_bytes, child_output)

            # Estimate this operator
            op_est = self.estimate_operator_memory(
                op,
                input_bytes,
                representation=representation,
                rows=rows,
                cols=cols,
            )

            # Peak = max(child_peak, this_operator_peak)
            return {
                "peak_bytes": max(
                    op_est["peak_bytes"],
                    child_peak if inputs else 0
                ),
                "output_bytes": op_est["output_bytes"],
                "temp_bytes": op_est["temp_bytes"],
                "supports_streaming": op_est["supports_streaming"],
            }

        except Exception as exc:
            _logger.warning(f"Plan memory estimation failed: {exc}")
            # Conservative fallback
            return {
                "peak_bytes": 500 * 1024 * 1024,  # 500 MB
                "output_bytes": 100 * 1024 * 1024,  # 100 MB
                "temp_bytes": 0,
                "supports_streaming": False,
            }

    def register_operator_profile(
        self,
        operator: str,
        profile: OperatorMemoryProfile,
    ) -> None:
        """Register custom operator memory profile."""
        self._operator_profiles[operator] = profile

    def register_representation_coefficients(
        self,
        representation: str,
        coefficients: RepresentationCoefficients,
    ) -> None:
        """Register custom representation coefficients."""
        self._coefficients[representation] = coefficients

    def get_operator_profile(self, operator: str) -> OperatorMemoryProfile:
        """Get operator memory profile."""
        return self._operator_profiles.get(
            operator,
            OperatorMemoryProfile(1.5, 1.0, 0.5, True, None)
        )

    def get_representation_coefficients(
        self, representation: str
    ) -> RepresentationCoefficients:
        """Get representation coefficients."""
        return self._coefficients.get(
            representation,
            self._coefficients["pandas"]
        )


# Global singleton
_GLOBAL_MEMORY_MODEL: FineGrainedMemoryModel | None = None


def global_memory_model() -> FineGrainedMemoryModel:
    """Get global fine-grained memory model singleton."""
    global _GLOBAL_MEMORY_MODEL
    if _GLOBAL_MEMORY_MODEL is None:
        _GLOBAL_MEMORY_MODEL = FineGrainedMemoryModel()
    return _GLOBAL_MEMORY_MODEL
