# -*- coding: utf-8 -*-
"""MB-P1-001: DataShapeEstimate strict metadata-only (no data loading).

Provides accurate memory footprint estimation without materializing data,
using schema metadata, row counts, and column statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataShapeEstimate:
    """Memory footprint estimate from metadata only.

    Attributes:
        rows: Number of rows
        cols: Number of columns
        bytes_per_row: Average bytes per row
        total_bytes: Total estimated memory footprint
        nullable_fraction: Fraction of nullable columns
        sparse_fraction: Fraction of sparse/missing values
        representation: Data representation (pandas/polars/arrow)
        overhead_bytes: Framework overhead (index, metadata)
        confidence: Estimate confidence [0.0, 1.0]
    """
    rows: int
    cols: int
    bytes_per_row: float
    total_bytes: int
    nullable_fraction: float
    sparse_fraction: float
    representation: str
    overhead_bytes: int
    confidence: float

    def scale_to_rows(self, target_rows: int) -> DataShapeEstimate:
        """Scale estimate to different row count."""
        if self.rows <= 0:
            return self
        scale = target_rows / self.rows
        return DataShapeEstimate(
            rows=target_rows,
            cols=self.cols,
            bytes_per_row=self.bytes_per_row,
            total_bytes=int(self.total_bytes * scale),
            nullable_fraction=self.nullable_fraction,
            sparse_fraction=self.sparse_fraction,
            representation=self.representation,
            overhead_bytes=self.overhead_bytes,
            confidence=max(0.5, self.confidence - 0.1 * abs(scale - 1.0)),
        )


class DataShapeEstimator:
    """Strict metadata-only shape estimator.

    Estimates memory footprint from schema without loading data.
    Integrates with adaptive_batch_scheduler for accurate admission.
    """

    # Type size mapping (bytes)
    _TYPE_SIZES = {
        "int8": 1, "int16": 2, "int32": 4, "int64": 8,
        "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
        "float32": 4, "float64": 8,
        "bool": 1,
        "datetime64[ns]": 8,
        "timedelta64[ns]": 8,
    }

    # Framework overhead multipliers
    _OVERHEAD_MULTIPLIERS = {
        "pandas": 1.35,  # Index + BlockManager overhead
        "polars": 1.10,  # Arrow + minimal overhead
        "arrow": 1.05,   # Pure Arrow minimal overhead
        "duckdb": 1.08,  # Column-store compression
    }

    def estimate_from_schema(
        self,
        schema: dict[str, str],
        row_count: int,
        *,
        representation: str = "pandas",
        column_stats: dict[str, Any] | None = None,
    ) -> DataShapeEstimate:
        """Estimate footprint from schema metadata only.

        Args:
            schema: Column name -> dtype mapping
            row_count: Number of rows
            representation: Target representation framework
            column_stats: Optional per-column statistics (nullable, cardinality)

        Returns:
            DataShapeEstimate with confidence score
        """
        if not schema or row_count <= 0:
            return DataShapeEstimate(
                rows=0, cols=0, bytes_per_row=0.0, total_bytes=0,
                nullable_fraction=0.0, sparse_fraction=0.0,
                representation=representation, overhead_bytes=0,
                confidence=0.0,
            )

        col_stats = column_stats or {}
        cols = len(schema)
        base_bytes = 0
        nullable_count = 0
        sparse_sum = 0.0

        for col_name, dtype in schema.items():
            # Base type size
            type_size = self._type_size_for(dtype)
            base_bytes += type_size

            # Nullable overhead (1 bit per row for validity buffer)
            stats = col_stats.get(col_name, {})
            if stats.get("nullable", True):
                nullable_count += 1
                base_bytes += 0.125  # 1 bit per row

            # Sparse/missing value fraction
            null_frac = stats.get("null_fraction", 0.0)
            sparse_sum += null_frac

        bytes_per_row = base_bytes
        nullable_fraction = nullable_count / max(1, cols)
        sparse_fraction = sparse_sum / max(1, cols)

        # Framework overhead
        overhead_mult = self._OVERHEAD_MULTIPLIERS.get(representation, 1.2)
        data_bytes = int(bytes_per_row * row_count)
        overhead_bytes = int(data_bytes * (overhead_mult - 1.0))
        total_bytes = data_bytes + overhead_bytes

        # Confidence: high when we have column stats, medium otherwise
        confidence = 0.85 if column_stats else 0.70

        return DataShapeEstimate(
            rows=row_count,
            cols=cols,
            bytes_per_row=bytes_per_row,
            total_bytes=total_bytes,
            nullable_fraction=nullable_fraction,
            sparse_fraction=sparse_fraction,
            representation=representation,
            overhead_bytes=overhead_bytes,
            confidence=confidence,
        )

    def estimate_from_plan(
        self,
        plan: Any,
        *,
        ctx: Any | None = None,
    ) -> DataShapeEstimate:
        """Estimate from execution plan (uses plan metadata, not data).

        Args:
            plan: PlanNode with schema/row_count metadata
            ctx: Optional execution context for source schema

        Returns:
            DataShapeEstimate
        """
        try:
            schema = getattr(plan, "output_schema", None)
            if schema is None and ctx is not None:
                # Try to infer from source
                schema = self._infer_schema_from_context(plan, ctx)

            if schema is None:
                # Fallback: conservative estimate
                return self._conservative_estimate(plan)

            row_count = self._estimate_row_count(plan, ctx)
            representation = getattr(plan, "backend", "pandas")

            return self.estimate_from_schema(
                schema=schema,
                row_count=row_count,
                representation=representation,
            )
        except Exception as exc:
            _logger.warning(f"Shape estimation failed: {exc}, using conservative")
            return self._conservative_estimate(plan)

    def _type_size_for(self, dtype: str) -> int:
        """Map dtype string to byte size."""
        dtype_lower = dtype.lower()
        for key, size in self._TYPE_SIZES.items():
            if key in dtype_lower:
                return size
        # String/object: conservative 64 bytes average
        if "str" in dtype_lower or "object" in dtype_lower:
            return 64
        # Unknown: assume float64
        return 8

    def _estimate_row_count(self, plan: Any, ctx: Any | None) -> int:
        """Estimate output row count from plan."""
        # Try plan annotation
        if hasattr(plan, "estimated_rows"):
            return max(0, int(plan.estimated_rows))

        # Try context
        if ctx is not None:
            try:
                from backend.plan_cost_router import estimate_plan_rows
                rows = estimate_plan_rows(ctx)
                if rows is not None and rows > 0:
                    return rows
            except Exception:
                pass

        # Conservative fallback
        return 500_000

    def _infer_schema_from_context(
        self, plan: Any, ctx: Any
    ) -> dict[str, str] | None:
        """Infer schema from context data source."""
        try:
            ds = getattr(ctx, "data_source", None)
            if ds is None:
                return None
            schema_info = getattr(ds, "schema", None)
            if schema_info is None:
                return None
            # Convert to dict[str, str]
            if isinstance(schema_info, dict):
                return {k: str(v) for k, v in schema_info.items()}
            return None
        except Exception:
            return None

    def _conservative_estimate(self, plan: Any) -> DataShapeEstimate:
        """Conservative fallback when metadata unavailable."""
        # Assume 500k rows × 10 columns × 8 bytes (float64)
        rows = 500_000
        cols = 10
        bytes_per_row = 80.0
        representation = getattr(plan, "backend", "pandas")
        overhead_mult = self._OVERHEAD_MULTIPLIERS.get(representation, 1.2)

        data_bytes = int(rows * bytes_per_row)
        overhead_bytes = int(data_bytes * (overhead_mult - 1.0))
        total_bytes = data_bytes + overhead_bytes

        return DataShapeEstimate(
            rows=rows,
            cols=cols,
            bytes_per_row=bytes_per_row,
            total_bytes=total_bytes,
            nullable_fraction=0.5,
            sparse_fraction=0.1,
            representation=representation,
            overhead_bytes=overhead_bytes,
            confidence=0.40,  # Low confidence fallback
        )


def estimate_task_shape(task: Any, ctx: Any | None = None) -> DataShapeEstimate:
    """Convenience function to estimate task memory footprint.

    Integrates with adaptive_batch_scheduler task admission.
    """
    estimator = DataShapeEstimator()
    plan = getattr(task, "node_ref", None)
    if plan is None:
        return estimator._conservative_estimate(task)
    return estimator.estimate_from_plan(plan, ctx=ctx)
