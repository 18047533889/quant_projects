"""Synthetic contracts for bounded FactorBatch staging; no COS access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_optimizer.cohort_materialization import (
    assemble_factor_values,
    estimate_factor_batch_materialization,
    validate_cohort_dimensions,
)


def _panels():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    return dates, [
        pd.DataFrame({"B.SZ": [2.0, np.nan, 6.0], "A.SH": [1.0, 3.0, 5.0]}, index=dates),
        pd.DataFrame({"A.SH": [10.0, 30.0, 50.0], "B.SZ": [20.0, 40.0, 60.0]}, index=dates),
    ]


@pytest.mark.parametrize(
    "dimensions",
    [(1, 1, 1), (500, 5000, 16), (1260, 5000, 16)],
)
def test_validate_cohort_dimensions_accepts_bounded_exact_integer_shapes(dimensions):
    assert validate_cohort_dimensions(*dimensions) == dimensions


@pytest.mark.parametrize(
    "dimensions",
    [(True, 1, 1), (0, 1, 1), (1261, 1, 1), (1, 5001, 1), (1, 1, 17)],
)
def test_validate_cohort_dimensions_rejects_bool_and_out_of_range(dimensions):
    with pytest.raises(ValueError):
        validate_cohort_dimensions(*dimensions)


def test_estimate_counts_staging_freeze_masks_panels_and_calendar():
    estimate = estimate_factor_batch_materialization(
        3, 2, 2, source_columns=(2, 2), calendar_rows=5,
        resident_pandas_bytes=100, resident_arrow_bytes=200,
    )

    assert estimate["factor_staging_bytes"] == 3 * 2 * 2 * 8
    assert estimate["factor_freeze_copy_bytes"] == 3 * 2 * 2 * 8
    assert estimate["validity_and_freeze_bytes"] == 3 * 2 * 2 * 2
    assert estimate["price_panel_bytes"] == 5 * 2 * 8
    assert estimate["resident_pandas_bytes"] == 100
    assert estimate["resident_arrow_bytes"] == 200
    assert estimate["total_bytes"] == sum(
        value for key, value in estimate.items() if key != "total_bytes"
    )


def test_assembler_matches_stack_reference_for_asset_and_factor_order():
    dates, panels = _panels()
    assets = ["A.SH", "B.SZ"]
    actual = assemble_factor_values(
        panels, dates, assets, calendar_rows=5, resident_pandas_bytes=0,
        resident_arrow_bytes=0, memory_budget_bytes=10**7,
    )
    expected = np.stack(
        [panel.reindex(dates)[assets].to_numpy(dtype=np.float64)
         for panel in panels], axis=-1,
    )

    assert actual.shape == (3, 2, 2)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    assert actual[1, 1, 0] != actual[1, 1, 0]
    assert actual[2, 0, 1] == 50.0


def test_assembler_rejects_over_budget_before_output_allocation(monkeypatch):
    dates, panels = _panels()
    estimate = estimate_factor_batch_materialization(
        3, 2, 2, source_columns=(2, 2), calendar_rows=5,
        resident_pandas_bytes=100, resident_arrow_bytes=200,
    )
    allocations = []

    def forbidden_allocation(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("output allocated before admission")

    monkeypatch.setattr(np, "empty", forbidden_allocation)
    with pytest.raises(MemoryError, match="materialization budget"):
        assemble_factor_values(
            panels, dates, ["A.SH", "B.SZ"], calendar_rows=5,
            resident_pandas_bytes=100, resident_arrow_bytes=200,
            memory_budget_bytes=estimate["total_bytes"] - 1,
        )

    assert allocations == []


@pytest.mark.parametrize("resident", [True, -1, 1.5])
def test_assembler_rejects_invalid_resident_memory_evidence(resident):
    dates, panels = _panels()
    with pytest.raises(ValueError):
        assemble_factor_values(
            panels, dates, ["A.SH", "B.SZ"], calendar_rows=5,
            resident_pandas_bytes=resident, resident_arrow_bytes=0,
            memory_budget_bytes=10**7,
        )
