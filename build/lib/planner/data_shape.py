# -*- coding: utf-8 -*-
"""MB-P1-002: Metadata-only DataShapeEstimate for cost planning.

Provides shape estimation without loading actual data, using:
  - Parquet metadata
  - DataAccess manifest
  - Schema information
  - Calendar session counts
  - Universe metadata
  - Row-group statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataShapeEstimate:
    """MB-P1-002: Metadata-only data shape estimate for cost planning.

    All fields derived from metadata without loading actual data:
      - estimated_rows: total row count from parquet metadata or manifest
      - estimated_dates: trading session count from calendar
      - estimated_instruments: universe size or instrument filter count
      - estimated_columns: schema column count
      - estimated_bytes: total data size in bytes
      - average_row_width_bytes: bytes per row
      - density: data sparsity (0.0 = all null, 1.0 = dense)
      - frequency: 'daily' | 'minute' | 'tick' | 'quarterly' | ...
      - bars_per_session: intraday bars per trading session
      - group_count: number of groups for group-by operations
      - remote: whether data is remote (COS/S3) or local
      - storage_kind: 'parquet' | 'clickhouse' | 'duckdb' | 'memory'
      - sorted_by: physical sort order
      - partition_by: partition columns
      - projected_columns: subset of columns needed for this plan
    """

    estimated_rows: int
    estimated_dates: int
    estimated_instruments: int
    estimated_columns: int
    estimated_bytes: int
    average_row_width_bytes: float
    density: float
    frequency: str
    bars_per_session: int | None = None
    group_count: int | None = None
    remote: bool = False
    storage_kind: str = "parquet"
    sorted_by: tuple[str, ...] = ()
    partition_by: tuple[str, ...] = ()
    projected_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_rows": self.estimated_rows,
            "estimated_dates": self.estimated_dates,
            "estimated_instruments": self.estimated_instruments,
            "estimated_columns": self.estimated_columns,
            "estimated_bytes": self.estimated_bytes,
            "average_row_width_bytes": round(self.average_row_width_bytes, 2),
            "density": round(self.density, 3),
            "frequency": self.frequency,
            "bars_per_session": self.bars_per_session,
            "group_count": self.group_count,
            "remote": self.remote,
            "storage_kind": self.storage_kind,
            "sorted_by": list(self.sorted_by),
            "partition_by": list(self.partition_by),
            "projected_columns": list(self.projected_columns),
        }


def estimate_shape_from_context(ctx: Any) -> DataShapeEstimate:
    """MB-P1-002: Build DataShapeEstimate from context metadata (no data load).

    Extracts shape information from:
      - ctx.data_source (instrument_filter, start_date, end_date, schema)
      - ctx.scan_shape or ctx.scan_cost (parquet metadata)
      - Calendar service (session count)
      - Universe metadata

    Falls back to conservative priors when metadata unavailable.
    """
    # Try to get shape from scan_shape/scan_cost first
    scan_shape = getattr(ctx, "scan_shape", None) or getattr(ctx, "scan_cost", None)
    if scan_shape is not None:
        return _estimate_from_scan_shape(scan_shape, ctx)

    # Fall back to data_source inspection
    ds = getattr(ctx, "data_source", None)
    if ds is not None:
        return _estimate_from_data_source(ds, ctx)

    # Last resort: conservative defaults
    return _conservative_default_shape()


def _estimate_from_scan_shape(scan_shape: Any, ctx: Any) -> DataShapeEstimate:
    """Build shape estimate from scan_shape/scan_cost metadata."""
    estimated_rows = int(getattr(scan_shape, "estimated_rows", 0) or 0)
    estimated_bytes = int(
        getattr(scan_shape, "selected_bytes", 0)
        or getattr(scan_shape, "total_bytes", 0)
        or 0
    )
    projected_cols = int(getattr(scan_shape, "projected_columns", 0) or 0)
    total_cols = int(getattr(scan_shape, "total_columns", 0) or 0)

    # MB-P1-003: Try to get real trading calendar dates
    estimated_dates = _estimate_dates_from_context(ctx)

    # MB-P1-001: Get actual instrument count from universe/filter
    estimated_instruments = _estimate_instruments_from_context(ctx)

    if estimated_rows == 0 and estimated_dates > 0 and estimated_instruments > 0:
        estimated_rows = estimated_dates * estimated_instruments

    avg_row_width = (
        (estimated_bytes / estimated_rows) if (estimated_rows > 0 and estimated_bytes > 0) else 64.0
    )

    density = float(getattr(scan_shape, "density", 0.95))
    frequency = str(getattr(scan_shape, "frequency", "") or "daily")
    remote = bool(getattr(scan_shape, "remote", False))
    storage = str(getattr(scan_shape, "storage_kind", "") or "parquet")

    bars_per_session = None
    if frequency in {"minute", "tick", "second"}:
        bars_per_session = _estimate_bars_per_session(frequency)

    projected_columns = tuple(getattr(scan_shape, "projected_column_names", ()) or ())

    return DataShapeEstimate(
        estimated_rows=estimated_rows or 1,
        estimated_dates=estimated_dates,
        estimated_instruments=estimated_instruments,
        estimated_columns=total_cols or projected_cols or 1,
        estimated_bytes=estimated_bytes or (estimated_rows * int(avg_row_width)),
        average_row_width_bytes=avg_row_width,
        density=density,
        frequency=frequency,
        bars_per_session=bars_per_session,
        group_count=None,
        remote=remote,
        storage_kind=storage,
        sorted_by=(),
        partition_by=(),
        projected_columns=projected_columns,
    )


def _estimate_from_data_source(ds: Any, ctx: Any) -> DataShapeEstimate:
    """Build shape estimate from data_source inspection."""
    # MB-P1-001: Get actual instrument filter
    estimated_instruments = _estimate_instruments_from_data_source(ds)

    # MB-P1-003: Get real trading calendar dates
    estimated_dates = _estimate_dates_from_data_source(ds, ctx)

    estimated_rows = max(1, estimated_dates * estimated_instruments)

    # Estimate columns from schema if available
    schema = getattr(ds, "schema", None) or getattr(ds, "_schema", None)
    estimated_columns = len(schema) if schema else 8

    # Conservative byte estimate: 8 bytes per cell
    avg_row_width = estimated_columns * 8.0
    estimated_bytes = int(estimated_rows * avg_row_width)

    # Try to determine frequency
    frequency = str(getattr(ds, "frequency", "") or getattr(ctx, "frequency", "") or "daily")

    bars_per_session = None
    if frequency in {"minute", "tick", "second"}:
        bars_per_session = _estimate_bars_per_session(frequency)

    # Check if remote
    ds_inner = ds
    remote = False
    for _ in range(5):  # Unwrap up to 5 layers
        name = type(ds_inner).__name__.lower()
        if "cos" in name or "s3" in name or "remote" in name:
            remote = True
            break
        ds_inner = getattr(ds_inner, "inner", None) or getattr(ds_inner, "_inner", None)
        if ds_inner is None:
            break

    return DataShapeEstimate(
        estimated_rows=estimated_rows,
        estimated_dates=estimated_dates,
        estimated_instruments=estimated_instruments,
        estimated_columns=estimated_columns,
        estimated_bytes=estimated_bytes,
        average_row_width_bytes=avg_row_width,
        density=0.95,
        frequency=frequency,
        bars_per_session=bars_per_session,
        group_count=None,
        remote=remote,
        storage_kind="parquet",
        sorted_by=(),
        partition_by=(),
        projected_columns=(),
    )


def _conservative_default_shape() -> DataShapeEstimate:
    """Conservative default when no metadata available."""
    # Use conservative ALL_A stock market estimate
    estimated_instruments = 5500
    estimated_dates = 250  # ~1 year of trading days
    estimated_rows = estimated_instruments * estimated_dates

    return DataShapeEstimate(
        estimated_rows=estimated_rows,
        estimated_dates=estimated_dates,
        estimated_instruments=estimated_instruments,
        estimated_columns=8,
        estimated_bytes=estimated_rows * 64,
        average_row_width_bytes=64.0,
        density=0.95,
        frequency="daily",
        bars_per_session=None,
        group_count=None,
        remote=False,
        storage_kind="parquet",
        sorted_by=(),
        partition_by=(),
        projected_columns=(),
    )


def _estimate_instruments_from_context(ctx: Any) -> int:
    """MB-P1-001: Extract actual instrument count from context/universe."""
    ds = getattr(ctx, "data_source", None)
    if ds is not None:
        return _estimate_instruments_from_data_source(ds)

    # Try universe metadata
    universe = getattr(ctx, "universe", None)
    if universe is not None:
        if hasattr(universe, "__len__"):
            try:
                return max(1, len(universe))
            except Exception:
                pass

    # Conservative ALL_A default
    return 5500


def _estimate_instruments_from_data_source(ds: Any) -> int:
    """MB-P1-001: Extract instrument count from data_source filter."""
    filt = getattr(ds, "instrument_filter", None)

    if filt is None:
        # No filter = all available instruments (conservative estimate)
        return 5500

    if isinstance(filt, (list, tuple, set, frozenset)):
        count = len(filt)
        if count == 0:
            # Empty filter = no instruments
            return 1
        return count

    # Unknown filter type, use conservative estimate
    return 5500


def _estimate_dates_from_context(ctx: Any) -> int:
    """MB-P1-003: Get trading calendar session count from context."""
    # Try calendar service first
    try:
        calendar = getattr(ctx, "calendar", None)
        if calendar is not None:
            start = getattr(ctx, "start_date", None)
            end = getattr(ctx, "end_date", None)
            if start and end:
                sessions = calendar.sessions_between(start, end)
                if hasattr(sessions, "__len__"):
                    return max(1, len(sessions))
    except Exception:
        pass

    # Fall back to data_source
    ds = getattr(ctx, "data_source", None)
    if ds is not None:
        return _estimate_dates_from_data_source(ds, ctx)

    # Conservative default: 1 year of trading
    return 250


def _estimate_dates_from_data_source(ds: Any, ctx: Any) -> int:
    """MB-P1-003: Estimate date count from data_source date range."""
    start = getattr(ds, "start_date", None)
    end = getattr(ds, "end_date", None)

    if not (start and end):
        return 250  # Conservative default

    # Try calendar service
    try:
        calendar = getattr(ctx, "calendar", None) or getattr(ds, "calendar", None)
        if calendar is not None:
            sessions = calendar.sessions_between(start, end)
            if hasattr(sessions, "__len__"):
                return max(1, len(sessions))
    except Exception:
        pass

    # Fall back to business day approximation (better than pd.bdate_range)
    try:
        import datetime
        if isinstance(start, str):
            start = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))

        delta = (end - start).days
        # Approximate: 5 trading days per 7 calendar days
        return max(1, int(delta * 5.0 / 7.0))
    except Exception:
        pass

    return 250


def _estimate_bars_per_session(frequency: str) -> int:
    """Estimate intraday bars per trading session."""
    freq_lower = frequency.lower()

    if "tick" in freq_lower:
        return 10000  # ~10k ticks per session
    if "second" in freq_lower or "1s" in freq_lower:
        return 14400  # 4 hours * 3600
    if "minute" in freq_lower or "1m" in freq_lower:
        return 240  # 4 hours * 60
    if "5m" in freq_lower:
        return 48
    if "15m" in freq_lower:
        return 16

    return 240  # Default to 1-minute
