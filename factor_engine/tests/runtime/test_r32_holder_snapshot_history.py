"""Snapshot report windows cannot be treated as a fixed daily-bar overlap."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import execution_contract, history_requirement


@pytest.mark.parametrize("canonical", [
    "holder_concentration_slope", "holder_concentration_acceleration",
])
def test_snapshot_windows_require_report_history_not_daily_overlap(canonical):
    load_all()
    # Four reports across 40 observations. A window of three means reports,
    # not three daily rows; the final visible report is far outside overlap.
    idx = pd.date_range("2025-01-01", periods=40)
    reports = pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30"])
    values = pd.DataFrame({"A": np.repeat([1.0, 2.0, 5.0, 11.0], 10)}, index=idx)
    dates = pd.DataFrame({"A": np.repeat(reports.to_numpy(), 10)}, index=idx)
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    full = op.calculate(values, 3, dates)
    short = op.calculate(values.iloc[-4:], 3, dates.iloc[-4:])
    assert np.isfinite(full.iloc[-1, 0])
    assert np.isnan(short.iloc[-1, 0])
    # Until a real report-state checkpoint exists, scheduler must replay history.
    contract = execution_contract(canonical)
    history = history_requirement(canonical, {"window": 3})
    assert contract.requires_full_history
    assert history.is_full_history
    assert history.kind == "report_count"
    assert history.count == (4 if canonical.endswith("acceleration") else 3)


@pytest.mark.parametrize("canonical", [
    "holder_concentration_slope", "holder_concentration_acceleration",
])
def test_holder_history_reaches_analyzer_and_incremental_planner(canonical):
    from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
    from factor_engine.api.columns import col
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.runtime.incremental import build_incremental_plan

    load_all()
    # Execution contracts are canonical-wide until mode-specific checkpoint
    # restore exists; omitting snapshot_date must not advertise unsafe support.
    analysis = Analyzer().lower(F(canonical)(col("close"), window=3))
    assert analysis.requires_full_history
    plan = build_incremental_plan(
        factor_id=canonical,
        analysis_lookback=analysis.lookback,
        watermark={"end_date": "2025-01-20"},
        end_date="2025-02-01",
        factor_freq="1d",
        source_bar_freq="1d",
    )
    assert plan.full_history_required
    assert plan.is_full_run
    assert plan.load_start is None


@pytest.mark.parametrize("canonical", [
    "holder_concentration_slope", "holder_concentration_acceleration",
])
@pytest.mark.parametrize("backend", ["pandas_numpy", "polars"])
def test_optional_snapshot_default_is_visible_through_generated_wrapper(canonical, backend):
    import polars as pl

    load_all()
    frame = pd.DataFrame({"A": np.arange(12, dtype=float) ** 2})
    if backend == "polars":
        frame = pl.from_pandas(frame)
    op = OperatorRegistry.get(canonical, backend)
    omitted = op.calculate(frame, window=4)
    explicit = op.calculate(frame, window=4, snapshot_date=None)
    np.testing.assert_allclose(omitted.to_numpy(), explicit.to_numpy(), equal_nan=True)
    assert np.isfinite(omitted.to_numpy()).any()
