import numpy as np
import pandas as pd
from datetime import date, timedelta

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model import _rolling_core as rc
from factor_engine.cleaned_operators.ts_model import dynamic_regression as dr


def _owner(canonical):
    return rc.FitScope(
        canonical=canonical, backend="pandas_numpy", profile="research",
        execution_id="exec-b", run_id="run-b", task_id="task-b",
        factor_id="factor-b")


def _panels():
    index = pd.date_range("2026-01-01", periods=24)
    x = pd.DataFrame({"GOOD": np.arange(24.0), "BAD": np.ones(24)}, index=index)
    y = pd.DataFrame({"GOOD": 2.0 + 3.0 * x["GOOD"], "BAD": np.arange(24.0)}, index=index)
    return y, x


def test_public_operator_emits_owned_instrument_and_window_failure_receipt():
    y, x = _panels()
    sink = rc.BoundedFitFailureSink(detail_capacity=64)
    canonical = "ts_multi_regression_coeff"
    op = OperatorRegistry.get(canonical, "pandas_numpy", mode="research")
    with rc.fit_receipt_scope(_owner(canonical)), rc.fit_failure_receipts(sink):
        result = op.calculate(y, x, window=10, coefficient_index=1,
                              min_periods=2, add_intercept=True)

    np.testing.assert_allclose(result["GOOD"].dropna(), 3.0)
    receipts = [r for r in sink.page() if r.status.reason == "singular"]
    assert receipts
    receipt = receipts[-1]
    assert receipt.scope_kind == "factor_window"
    assert receipt.scope.instrument == "BAD"
    assert receipt.scope.window_start in y.index
    assert receipt.scope.window_end in y.index
    assert receipt.scope.fit_cutoff == receipt.scope.window_end
    assert receipt.scope.maturity_cutoff == receipt.scope.window_end


def test_public_operator_without_owner_is_explicitly_kernel_only():
    y, x = _panels()
    sink = rc.BoundedFitFailureSink(detail_capacity=64)
    op = OperatorRegistry.get(
        "ts_multi_regression_coeff", "pandas_numpy", mode="research")
    with rc.fit_failure_receipts(sink):
        op.calculate(y, x, window=10, coefficient_index=1,
                     min_periods=2, add_intercept=True)
    assert sink.page()
    assert all(r.scope_kind == "kernel_only" for r in sink.page())
    assert all(r.scope.factor_id is None for r in sink.page())


def test_public_receipts_preserve_datetimeindex_and_python_date_index():
    indexes = (
        pd.date_range("2026-02-01", periods=16),
        pd.Index([date(2026, 2, 1) + timedelta(days=i) for i in range(16)]),
    )
    canonical = "ts_multi_regression_coeff"
    op = OperatorRegistry.get(canonical, "pandas_numpy", mode="research")
    for index in indexes:
        x = pd.DataFrame({"BAD": np.ones(16)}, index=index)
        y = pd.DataFrame({"BAD": np.arange(16.0)}, index=index)
        sink = rc.BoundedFitFailureSink(detail_capacity=32)
        with rc.fit_receipt_scope(_owner(canonical)), rc.fit_failure_receipts(sink):
            op.calculate(y, x, window=10, coefficient_index=1,
                         min_periods=2, add_intercept=True)
        receipt = next(r for r in reversed(sink.page()) if r.status.reason == "singular")
        assert receipt.scope_kind == "factor_window"
        assert receipt.scope.window_start in index
        assert receipt.scope.window_end in index
        assert receipt.scope.output_row in index


def test_public_stability_operator_records_main_and_subfit_coordinates(monkeypatch):
    index = pd.date_range("2026-01-01", periods=30)
    x = pd.DataFrame({"AAA": np.arange(30.0)}, index=index)
    y = pd.DataFrame({"AAA": 1.0 + 2.0 * x["AAA"] + 0.01 * np.sin(np.arange(30))},
                     index=index)
    captured = []

    def capture(result, **coordinates):
        captured.append((result, coordinates))

    monkeypatch.setattr(dr, "record_current_fit", capture)
    op = OperatorRegistry.get(
        "ts_multi_regression_coeff_stability", "pandas_numpy", mode="research")
    with rc.fit_receipt_scope(_owner("ts_multi_regression_coeff_stability")):
        result = op.calculate(y, x, window=10, coefficient_index=1,
                              min_periods=2, add_intercept=True)

    assert np.isfinite(result.iloc[-1, 0])
    last = [coords for fit, coords in captured
            if fit.value is not None and coords["output_row"] == index[-1]]
    assert len(last) >= 6  # main fit plus five stability subfits
    assert len({coords["window_end"] for coords in last}) >= 5
    assert all(coords["instrument"] == "AAA" for coords in last)
