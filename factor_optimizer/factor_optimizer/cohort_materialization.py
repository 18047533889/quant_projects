"""Bounded, ordered staging of decoded COS panels into a FactorBatch array.

The accounting covers declared materialization buffers, not total process RSS,
Arrow decoding before this admission, or arbitrary third-party allocations.
"""
from __future__ import annotations

import numpy as np
from factor_optimizer.research_price_reader import QUERY_RESULT_BYTES, ESTIMATED_LONG_ROW_BYTES

DEFAULT_MATERIALIZATION_BUDGET_BYTES = 8 * 1024**3
MAX_MATERIALIZATION_BUDGET_BYTES = 16 * 1024**3


def _nonnegative_bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative built-in integer")
    return value


def validate_cohort_dimensions(rows, assets, factors):
    for value, name, maximum in (
        (rows, "rows", 1260), (assets, "assets", 5000),
        (factors, "factors", 16),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return rows, assets, factors


def estimate_factor_batch_materialization(
    rows, assets, factors, *, source_columns, calendar_rows,
    resident_pandas_bytes, resident_arrow_bytes,
):
    """Conservative sum of identifiable simultaneous buffers; no RSS promise."""
    validate_cohort_dimensions(rows, assets, factors)
    if type(calendar_rows) is not int or calendar_rows < rows:
        raise ValueError("calendar_rows must be an integer at least rows")
    widths = tuple(source_columns)
    if len(widths) != factors or any(type(n) is not int or n < assets for n in widths):
        raise ValueError("source_columns must cover each factor's selected assets")
    cells, one_panel = rows * assets * factors, rows * assets
    query_days = min(calendar_rows, 64, QUERY_RESULT_BYTES //
                     (ESTIMATED_LONG_ROW_BYTES * assets))
    query_rows = query_days * assets
    components = {
        "resident_pandas_bytes": _nonnegative_bytes(resident_pandas_bytes, "resident_pandas_bytes"),
        "resident_arrow_bytes": _nonnegative_bytes(resident_arrow_bytes, "resident_arrow_bytes"),
        "factor_staging_bytes": cells * 8,
        "factor_freeze_copy_bytes": cells * 8,
        "validity_and_freeze_bytes": cells * 2,
        "full_width_reindex_bytes": rows * max(widths) * 8 * 2,
        "selected_panel_conversion_bytes": one_panel * 8 * 2,
        "price_panel_bytes": calendar_rows * assets * 8,
        "price_reindex_and_conversion_bytes": calendar_rows * assets * 8 * 2,
        "price_query_arrow_bytes": min(QUERY_RESULT_BYTES, query_rows * ESTIMATED_LONG_ROW_BYTES),
        "price_query_pandas_estimate_bytes": query_rows * ESTIMATED_LONG_ROW_BYTES,
        "price_query_pivot_bytes": query_rows * 8 * 2,
        "label_arithmetic_and_freeze_bytes": one_panel * (5 * 8 + 2),
        "axis_and_index_headroom_bytes": rows * 32 + assets * 160,
    }
    return {**components, "total_bytes": sum(components.values())}


def admit_factor_batch_materialization(estimate, memory_budget_bytes):
    if (type(memory_budget_bytes) is not int
            or not 0 < memory_budget_bytes <= MAX_MATERIALIZATION_BUDGET_BYTES):
        raise ValueError("materialization budget must be an integer in 1..16 GiB")
    if estimate["total_bytes"] > memory_budget_bytes:
        raise MemoryError(
            f"materialization budget exceeded: estimated {estimate['total_bytes']} "
            f"bytes, allowed {memory_budget_bytes} bytes"
        )



def arrow_table_buffer_bytes(table):
    """Require a measurable Arrow table; missing evidence never means zero."""
    measure = getattr(table, "get_total_buffer_size", None)
    if not callable(measure):
        raise TypeError("Arrow table must expose get_total_buffer_size() for admission")
    size = measure()
    if type(size) is not int or size < 0:
        raise ValueError("Arrow table buffer size must be a nonnegative integer")
    return size


def estimate_pandas_conversion_bytes(arrow_bytes):
    """Numeric-frame conversion allowance; excludes arbitrary object expansion."""
    return _nonnegative_bytes(arrow_bytes, "arrow_bytes") * 2


def admit_decoded_source(retained_pandas_bytes, current_arrow_bytes, memory_budget_bytes):
    """Admit identifiable source buffers while one Arrow table is decoded."""
    retained = _nonnegative_bytes(retained_pandas_bytes, "retained_pandas_bytes")
    current = _nonnegative_bytes(current_arrow_bytes, "current_arrow_bytes")
    if (type(memory_budget_bytes) is not int
            or not 0 < memory_budget_bytes <= MAX_MATERIALIZATION_BUDGET_BYTES):
        raise ValueError("materialization budget must be an integer in 1..16 GiB")
    estimated = retained + current
    if estimated > memory_budget_bytes:
        raise MemoryError(
            f"materialization budget exceeded during source conversion: "
            f"estimated {estimated} bytes, allowed {memory_budget_bytes} bytes"
        )
    return estimated


def assemble_factor_values(
    panels, dates, assets, *, calendar_rows, resident_pandas_bytes,
    resident_arrow_bytes, memory_budget_bytes=DEFAULT_MATERIALIZATION_BUDGET_BYTES,
):
    """Match the former reindex/column-select/stack semantics one factor at a time."""
    rows, columns, factors = validate_cohort_dimensions(len(dates), len(assets), len(panels))
    for panel in panels:
        missing = [asset for asset in assets if asset not in panel.columns]
        if missing:
            raise KeyError(f"selected assets are absent from a factor panel: {missing!r}")
    estimate = estimate_factor_batch_materialization(
        rows, columns, factors, source_columns=tuple(len(panel.columns) for panel in panels),
        calendar_rows=calendar_rows, resident_pandas_bytes=resident_pandas_bytes,
        resident_arrow_bytes=resident_arrow_bytes,
    )
    admit_factor_batch_materialization(estimate, memory_budget_bytes)
    values = np.empty((rows, columns, factors), dtype=np.float64)
    for index, panel in enumerate(panels):
        selected = panel.reindex(dates)[assets].to_numpy(dtype=np.float64)
        values[:, :, index] = selected
        del selected
    return values
