import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.fiscal_event_ops import (
    _FunctionOperator,
    _RowSum,
    OperatorRegistry,
    pd_fiscal_sign_consistency,
)
from factor_engine.cleaned_operators.time_semantic_ops import (
    _EventWindowReturnAsof,
    _SameClockLag,
)
from factor_engine.cleaned_operators.fiscal_strict import (
    pd_period_cagr,
    pd_period_change,
    pd_period_lag,
    pd_quarter_from_cumulative,
    pd_ttm_from_cumulative,
    pd_ttm_from_quarterly,
    pd_yoy_by_period,
)
from factor_engine.cleaned_operators.report_timing import (
    ReportFilingDelaySurprise,
    ReportRevisionMagnitude,
)
from factor_engine.cleaned_operators.update_clock import _update_kernel
from factor_engine.cleaned_operators.fundamental.transforms_v2 import _SPECS
from factor_engine.cleaned_operators.fundamental.expectation_v2 import _SPECS as _EXPECTATION_SPECS
from factor_engine.cleaned_operators.fundamental import transforms_repairs_v2 as _repairs  # noqa: F401
from factor_engine.cleaned_operators.fundamental import quality_v2 as _quality
from factor_engine.cleaned_operators.fundamental import accruals_scores as _accruals  # noqa: F401
from factor_engine.cleaned_operators import wave1_earnings as _wave1_earnings  # noqa: F401
from factor_engine.cleaned_operators import wave1_valuation as _wave1_valuation  # noqa: F401


def _fiscal_panels():
    index = pd.date_range("2024-01-01", periods=6)
    values = pd.DataFrame({"A": [1.0, 2.0, -3.0, 4.0, 5.0, 6.0]}, index=index)
    periods = pd.DataFrame(
        {"A": ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2"]},
        index=index,
    )
    return values, periods


def test_fiscal_event_aliases_load_with_their_own_contracts():
    for name in ("fiscal_direction_consistency", "fiscal_pair_direction_agreement"):
        assert name in OperatorRegistry._operators
        metadata = OperatorRegistry._operators[name]["pandas_numpy"].metadata
        assert metadata.panel_params
        assert metadata.param_specs["periods"].default == 8
        assert metadata.param_specs["revision_policy"].choices == (
            "latest_available",
            "first_available",
        )


def test_fiscal_sign_consistency_real_call_and_invalid_periods():
    values, periods = _fiscal_panels()
    operator = _FunctionOperator(
        "fiscal_sign_consistency",
        ["signal", "period_id", "periods", "min_periods", "require_consecutive", "revision_policy"],
        pd_fiscal_sign_consistency,
        "test instance",
    )
    result = operator.calculate(values, periods, periods=3, min_periods=2)
    assert result.iloc[-1, 0] == pytest.approx(1.0)
    for invalid in (True, 1.5, 0):
        with pytest.raises(OperatorParameterError):
            operator.calculate(values, periods, periods=invalid)


def test_fiscal_sign_consistency_rejects_invalid_policy_and_bound_relation():
    values, periods = _fiscal_panels()
    operator = OperatorRegistry._operators["fiscal_sign_consistency"]["pandas_numpy"]
    with pytest.raises(OperatorParameterError):
        operator.calculate(values, periods, revision_policy="future_visible")
    with pytest.raises(ValueError, match="min_periods must not exceed periods"):
        operator.calculate(values, periods, periods=2, min_periods=3)


def test_row_sum_variadic_execution_is_preserved():
    values, _ = _fiscal_panels()
    result = _RowSum().calculate(values, values * 2.0, min_count=2)
    assert result.iloc[-1, 0] == pytest.approx(18.0)
    missing = values.copy()
    missing.iloc[-1, 0] = np.nan
    assert np.isnan(_RowSum().calculate(values, missing, min_count=2).iloc[-1, 0])


def test_time_semantic_scalar_contracts_are_not_panels():
    clock = _SameClockLag.metadata
    assert clock.panel_params == ("x", "clock_time")
    assert clock.param_specs["lag_minutes"].default == 5
    assert clock.param_specs["lag_minutes"].min == 1
    event = _EventWindowReturnAsof.metadata
    assert event.panel_params == ("ret", "event_date")
    assert event.param_specs["window_before"].min == 0
    assert event.param_specs["window_after"].default == 5


def test_fiscal_strict_revision_policy_is_prefix_visible():
    index = pd.date_range("2024-01-01", periods=5)
    values = pd.DataFrame({"A": [10.0, 20.0, 25.0, 40.0, 80.0]}, index=index)
    periods = pd.DataFrame(
        {"A": ["2023Q1", "2023Q2", "2023Q2", "2023Q3", "2023Q4"]},
        index=index,
    )
    latest = pd_period_lag(values, periods, periods=1, revision_policy="latest_available")
    first = pd_period_lag(values, periods, 1, "first_available")
    assert latest.iloc[3, 0] == pytest.approx(25.0)
    assert first.iloc[3, 0] == pytest.approx(20.0)
    assert pd_period_change(values, periods, 1).iloc[3, 0] == pytest.approx(15.0)
    assert not np.isnan(latest.iloc[1, 0])
    assert np.isnan(latest.iloc[0, 0])


def test_fiscal_strict_quarter_ttm_yoy_and_cagr_boundaries():
    index = pd.date_range("2024-01-01", periods=5)
    periods = pd.DataFrame(
        {"A": ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"]},
        index=index,
    )
    quarter = pd.DataFrame({"A": [1, 2, 3, 4, 1]}, index=index)
    cumulative = pd.DataFrame({"A": [10.0, 30.0, 60.0, 100.0, 12.0]}, index=index)
    single = pd_quarter_from_cumulative(cumulative, periods, quarter)
    assert single.iloc[:4, 0].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert pd_ttm_from_quarterly(single, periods).iloc[3, 0] == pytest.approx(100.0)
    assert pd_ttm_from_cumulative(cumulative, periods, quarter).iloc[3, 0] == pytest.approx(100.0)

    level = pd.DataFrame({"A": [10.0, 12.0, 14.0, 16.0, 20.0]}, index=index)
    assert pd_yoy_by_period(level, periods).iloc[-1, 0] == pytest.approx(1.0)
    assert pd_period_cagr(level, periods, periods=4).iloc[-1, 0] == pytest.approx(1.0)
    assert pd_quarter_from_cumulative(cumulative, periods).isna().all().all()


def test_fiscal_strict_final_registry_contract_and_invalid_scalars():
    expected = {
        "period_lag", "period_change", "period_average", "period_cagr",
        "quarter_from_cumulative", "ttm_from_quarterly",
        "ttm_from_cumulative", "yoy_by_period",
    }
    for canonical in expected:
        metadata = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert metadata.panel_params
        assert all(
            name in metadata.panel_params or name in metadata.param_specs
            for name in metadata.param_names
        )

    values, periods = _fiscal_panels()
    operator = OperatorRegistry._operators["period_change"]["pandas_numpy"]
    for kwargs in ({"periods": True}, {"periods": 1.5}, {"periods": 0}, {"mode": "pct"}):
        with pytest.raises((OperatorParameterError, ValueError)):
            operator.calculate(values, periods, **kwargs)


def test_report_timing_real_events_and_fail_closed_revision_pair():
    index = pd.date_range("2024-01-01", periods=7)
    delay = pd.DataFrame({"A": [10.0, 10.0, 12.0, 12.0, 14.0, 14.0, 30.0]}, index=index)
    events = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}, index=index)
    result = ReportFilingDelaySurprise().calculate(
        delay, events, window=8, min_periods=3
    )
    assert np.isfinite(result.iloc[-1, 0])
    assert result.iloc[1:6:2, 0].isna().all()
    bad_events = events.copy(); bad_events.iloc[0, 0] = 2.0
    with pytest.raises(ValueError, match="strict boolean indicator"):
        ReportFilingDelaySurprise().calculate(delay, bad_events)

    current = pd.DataFrame({"A": [11.0] * 7}, index=index)
    previous = pd.DataFrame({"A": [10.0] * 7}, index=index)
    cp = pd.DataFrame({"A": ["2023Q4"] * 7}, index=index)
    pp = cp.copy(); pp.iloc[-1, 0] = "2023Q3"
    revision = ReportRevisionMagnitude().calculate(
        current, previous, cp, pp, events, window=8, min_periods=3
    )
    assert np.isnan(revision.iloc[-1, 0])


def test_update_clock_contract_real_call_invalid_and_missing_boundary():
    operator = OperatorRegistry._operators["update_path_efficiency"]["pandas_numpy"]
    index = pd.date_range("2024-01-01", periods=4)
    values = pd.DataFrame({"A": [1.0, 2.0, 4.0, 7.0]}, index=index)
    events = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=index)
    result = operator.calculate(values, events, n_updates=3)
    assert result.iloc[-1, 0] == pytest.approx(1.0)
    for invalid in (True, 2.5, 2):
        with pytest.raises((OperatorParameterError, ValueError)):
            operator.calculate(values, events, n_updates=invalid)
    broken = events.copy(); broken.iloc[1, 0] = np.nan
    assert np.isnan(operator.calculate(values, broken, n_updates=3).iloc[-1, 0])


def test_report_and_update_final_metadata_complete():
    report = ReportFilingDelaySurprise.metadata
    assert report.panel_params == ("delay", "filing_event")
    assert report.param_specs["min_periods"].default == 3
    for canonical, minimum in {
        "update_path_efficiency": 3,
        "update_acceleration": 4,
        "update_surprise": 5,
        "update_direction_persistence": 3,
    }.items():
        metadata = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert metadata.panel_params == ("x", "update_event")
        assert metadata.param_specs["n_updates"].min == minimum
        assert metadata.param_specs["n_updates"].default == 5


def test_fundamental_transform_factory_declares_every_panel_and_scalar():
    for canonical, _params, _fn, _description in _SPECS:
        metadata = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert set(metadata.panel_params) | set(metadata.param_specs) == set(metadata.param_names)
        assert set(metadata.panel_params).isdisjoint(metadata.param_specs)
    assert OperatorRegistry._operators["fin_lag"]["pandas_numpy"].metadata.param_specs["periods"].default == 1
    assert OperatorRegistry._operators["fin_cagr"]["pandas_numpy"].metadata.param_specs["periods"].default == 4
    assert OperatorRegistry._operators["fin_ttm"]["pandas_numpy"].metadata.param_specs["periods_per_year"].default == 4


def test_fundamental_transform_actual_values_kwargs_and_invalid_controls():
    index = pd.date_range("2024-01-01", periods=5)
    values = pd.DataFrame({"A": [10.0, 12.0, 18.0, 20.0, 30.0]}, index=index)
    periods = pd.DataFrame(
        {"A": ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"]},
        index=index,
    )
    lag = OperatorRegistry._operators["fin_lag"]["pandas_numpy"]
    pct = OperatorRegistry._operators["fin_pct_change"]["pandas_numpy"]
    cagr = OperatorRegistry._operators["fin_cagr"]["pandas_numpy"]
    assert lag.calculate(values, periods, 1).iloc[-1, 0] == pytest.approx(20.0)
    assert pct.calculate(values, periods, periods=1).iloc[-1, 0] == pytest.approx(0.5)
    assert cagr.calculate(values, periods, periods=4, periods_per_year=4).iloc[-1, 0] == pytest.approx(2.0)
    for bad in (True, 1.5, 0):
        with pytest.raises(OperatorParameterError):
            lag.calculate(values, periods, periods=bad)
    with pytest.raises((OperatorParameterError, ValueError)):
        pct.calculate(values, periods, flow_type="UnknownFlow")
    with pytest.raises(ValueError, match="CumulativeYTDFlow"):
        pct.calculate(values, periods, flow_type="CumulativeYTDFlow")


def test_fundamental_transform_revision_and_missing_period_are_pit_safe():
    index = pd.date_range("2024-01-01", periods=5)
    values = pd.DataFrame({"A": [10.0, 20.0, 25.0, 40.0, 80.0]}, index=index)
    periods = pd.DataFrame(
        {"A": ["2023Q1", "2023Q2", "2023Q2", "2023Q3", "2023Q4"]},
        index=index,
    )
    lag = OperatorRegistry._operators["fin_lag"]["pandas_numpy"].calculate(values, periods)
    assert lag.iloc[1, 0] == pytest.approx(10.0)
    assert lag.iloc[2, 0] == pytest.approx(10.0)
    assert lag.iloc[3, 0] == pytest.approx(25.0)
    assert np.isnan(lag.iloc[0, 0])

    gap_periods = periods.copy(); gap_periods.iloc[3:, 0] = ["2023Q4", "2024Q1"]
    gap_lag = OperatorRegistry._operators["fin_lag"]["pandas_numpy"].calculate(values, gap_periods)
    assert np.isnan(gap_lag.iloc[3, 0])


@pytest.mark.parametrize("canonical", [spec[0] for spec in _SPECS])
def test_each_fundamental_transform_executes_and_is_prefix_causal(canonical):
    rows = 20
    index = pd.date_range("2020-01-01", periods=rows)
    numeric = pd.DataFrame(
        {"A": np.linspace(1.0, 20.0, rows) + 0.2 * np.sin(np.arange(rows))},
        index=index,
    )
    period_values = [f"{2020 + i // 4}Q{i % 4 + 1}" for i in range(rows)]
    period = pd.DataFrame({"A": period_values}, index=index)
    operator = OperatorRegistry._operators[canonical]["pandas_numpy"]
    metadata = operator.metadata

    def panel(name, length=rows):
        if "period_id" in name:
            return period.iloc[:length].copy()
        base = numeric.iloc[:length].copy()
        if name in {"denominator", "base", "scale", "scale_base", "assets", "earnings"}:
            return base + 5.0
        if name in {"cashflow", "flow", "y"}:
            return base * 0.7
        return base

    full_args = [panel(name) for name in metadata.panel_params]
    prefix_args = [panel(name, 13) for name in metadata.panel_params]
    full = operator.calculate(*full_args)
    prefix = operator.calculate(*prefix_args)
    assert isinstance(full, pd.DataFrame)
    assert full.shape == numeric.shape
    assert np.isfinite(full.to_numpy(dtype=float)[~np.isnan(full.to_numpy(dtype=float))]).all()
    assert np.isfinite(full.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(
        full.iloc[:13].to_numpy(dtype=float),
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_expectation_factory_contracts_and_actual_surprise():
    for canonical, params, _fn, _description in _EXPECTATION_SPECS:
        metadata = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert set(metadata.panel_params) | set(metadata.param_specs) == set(params)
        assert set(metadata.panel_params).isdisjoint(metadata.param_specs)
    index = pd.date_range("2024-01-01", periods=3)
    actual = pd.DataFrame({"A": [1.0, 3.0, 5.0]}, index=index)
    expected = pd.DataFrame({"A": [1.0, 2.0, 4.0]}, index=index)
    scale = pd.DataFrame({"A": [2.0, 2.0, 2.0]}, index=index)
    operator = OperatorRegistry._operators["fin_surprise"]["pandas_numpy"]
    assert operator.calculate(actual, expected, scale).iloc[-1, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("canonical", [spec[0] for spec in _EXPECTATION_SPECS])
def test_each_expectation_operator_executes_finite_and_prefix_causal(canonical):
    rows = 260
    prefix_rows = 200
    index = pd.date_range("2023-01-01", periods=rows)
    expected_values = 10.0 + np.arange(rows) * 0.03 + np.sin(np.arange(rows) / 5.0)
    expected = pd.DataFrame({"A": expected_values}, index=index)
    actual = pd.DataFrame(
        {"A": expected_values + np.cos(np.arange(rows) / 3.0)}, index=index
    )
    positive = pd.DataFrame({"A": np.linspace(5.0, 8.0, rows)}, index=index)
    fiscal_ids = [f"{2020 + i // 4}Q{i % 4 + 1}" for i in range(rows)]
    period_id = pd.DataFrame({"A": fiscal_ids}, index=index)
    target_id = pd.DataFrame({"A": ["2026Q4"] * rows}, index=index)
    operator = OperatorRegistry._operators[canonical]["pandas_numpy"]

    def panel(name, length):
        if name == "actual":
            return actual.iloc[:length].copy()
        if name in {"expected", "expected_mean"}:
            return expected.iloc[:length].copy()
        if name in {"period_id"}:
            return period_id.iloc[:length].copy()
        if name == "target_period_id":
            return target_id.iloc[:length].copy()
        return positive.iloc[:length].copy()

    full = operator.calculate(*[panel(name, rows) for name in operator.metadata.panel_params])
    prefix = operator.calculate(
        *[panel(name, prefix_rows) for name in operator.metadata.panel_params]
    )
    assert full.shape == (rows, 1)
    assert np.isfinite(full.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(
        full.iloc[:prefix_rows].to_numpy(dtype=float),
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "canonical",
    [
        "fin_positive_streak", "fin_negative_streak", "fin_cash_earnings_gap",
        "fin_revision_delta", "fin_revision_pct", "fin_revision_direction",
        "fin_revision_count", "fin_revision_magnitude", "fin_restated_flag",
        "fin_days_since_update", "fin_staleness",
    ],
)
def test_each_fundamental_repair_executes_finite_and_prefix_causal(canonical):
    rows = 270
    prefix_rows = 180
    index = pd.date_range("2023-01-01", periods=rows)
    # Same-period value changes are genuine revision events; periodic target
    # changes are new reports.  This supplies both event types and full coverage.
    values = pd.DataFrame(
        {"A": 10.0 + np.arange(rows) * 0.02 + (np.arange(rows) % 3 == 0) * 0.5},
        index=index,
    )
    period_id = pd.DataFrame(
        {"A": [f"{2023 + i // 80}Q{(i // 20) % 4 + 1}" for i in range(rows)]},
        index=index,
    )
    operator = OperatorRegistry._operators[canonical]["pandas_numpy"]

    def args(length):
        result = []
        for name in operator.metadata.panel_params:
            if name == "period_id":
                result.append(period_id.iloc[:length].copy())
            elif name in {"scale", "scale_base"}:
                result.append(pd.DataFrame({"A": [5.0] * length}, index=index[:length]))
            elif name == "cashflow":
                result.append(values.iloc[:length].copy() * 0.7)
            else:
                result.append(values.iloc[:length].copy())
        return result

    full = operator.calculate(*args(rows))
    prefix = operator.calculate(*args(prefix_rows))
    assert full.shape == (rows, 1)
    assert np.isfinite(full.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(
        full.iloc[:prefix_rows].to_numpy(dtype=float),
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )


def test_fin_cash_earnings_gap_polars_twin_kwargs_and_flow_grain_gate():
    pl = pytest.importorskip("polars")
    # Import the authoritative twin directly; the full-load test separately
    # proves that no later two-input backend overwrites it.
    from factor_engine.cleaned_operators.fundamental import polars_fundamental  # noqa: F401

    operator = OperatorRegistry._operators["fin_cash_earnings_gap"]["polars"]
    earnings = pl.DataFrame({"A": [10.0, 12.0]})
    cashflow = pl.DataFrame({"A": [7.0, 8.0]})
    scale = pl.DataFrame({"A": [5.0, 5.0]})
    result = operator.calculate(
        earnings=earnings,
        cashflow=cashflow,
        scale=scale,
        flow_type="SinglePeriodFlow",
    )
    assert result["A"].to_list() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError, match="mixing incompatible flow grains"):
        operator.calculate(
            earnings=earnings,
            cashflow=cashflow,
            scale=scale,
            flow_type=("SinglePeriodFlow", "TTMFlow"),
        )


def test_quality_factory_registers_all_inventory_contracts():
    assert len(set(_quality._CANONICALS)) == 55
    for canonical in set(_quality._CANONICALS):
        metadata = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert set(metadata.panel_params) | set(metadata.param_specs) == set(metadata.param_names)
        assert set(metadata.panel_params).isdisjoint(metadata.param_specs)


@pytest.mark.parametrize("canonical", sorted(set(_quality._CANONICALS)))
def test_each_quality_operator_executes_finite_and_prefix_causal(canonical):
    rows = 24
    prefix_rows = 16
    index = pd.date_range("2020-01-01", periods=rows)
    base_values = 10.0 + np.arange(rows) * 0.4 + np.sin(np.arange(rows) / 2.0)
    base = pd.DataFrame({"A": base_values}, index=index)
    period = pd.DataFrame(
        {"A": [f"{2020 + i // 4}Q{i % 4 + 1}" for i in range(rows)]},
        index=index,
    )
    operator = OperatorRegistry._operators[canonical]["pandas_numpy"]

    def panel(name, length):
        if name == "period_id":
            return period.iloc[:length].copy()
        if name == "mask":
            return pd.DataFrame({"A": [1.0] * length}, index=index[:length])
        values = base.iloc[:length].copy()
        if name in {"annualized_ocf", "capex", "debt_repayment", "dividend_interest_payment"}:
            values = -values.abs()
        elif name in {"cash", "cash_equivalents"}:
            values = values * 0.4
        elif name in {"total_assets", "avg_assets", "avg_equity", "revenue", "operating_revenue"}:
            values = values + 50.0
        elif name in {"total_liabilities", "current_liabilities", "short_term_debt", "long_term_debt"}:
            values = values * 0.3
        return values

    full = operator.calculate(*[panel(name, rows) for name in operator.metadata.panel_params])
    prefix = operator.calculate(
        *[panel(name, prefix_rows) for name in operator.metadata.panel_params]
    )
    assert full.shape == (rows, 1)
    assert np.isfinite(full.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(
        full.iloc[:prefix_rows].to_numpy(dtype=float),
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )


_WAVE1_CANONICALS = sorted(
    name for name in OperatorRegistry._operators
    if name.startswith(("es1_", "aq1_", "ep1_", "val1_"))
)


def test_wave1_earnings_and_valuation_contract_inventory_is_complete():
    assert len(_WAVE1_CANONICALS) == 27
    for canonical in _WAVE1_CANONICALS:
        metadata = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert set(metadata.panel_params) | set(metadata.param_specs) == set(metadata.param_names)
        assert set(metadata.panel_params).isdisjoint(metadata.param_specs)
        assert all(spec.param_role is not None for spec in metadata.param_specs.values())


@pytest.mark.parametrize("canonical", _WAVE1_CANONICALS)
def test_each_wave1_earnings_and_valuation_operator_is_finite_and_prefix_causal(canonical):
    rows, prefix_rows, cols = 80, 55, 12
    index = pd.date_range("2023-01-01", periods=rows)
    columns = [f"S{i:02d}" for i in range(cols)]
    rr = np.arange(rows, dtype=float)[:, None]
    cc = np.arange(cols, dtype=float)[None, :]
    base = pd.DataFrame(10.0 + 0.15 * rr + 0.3 * cc + np.sin(rr / 4.0 + cc), index=index, columns=columns)
    operator = OperatorRegistry._operators[canonical]["pandas_numpy"]

    def panels(length):
        result = []
        for i, name in enumerate(operator.metadata.panel_params):
            panel = base.iloc[:length].copy() * (1.0 + i * 0.07)
            if name in {"operating_cash_flow", "working_capital"}:
                panel = panel * (0.7 if name == "operating_cash_flow" else 0.2)
            result.append(panel)
        return result

    full = operator.calculate(*panels(rows))
    prefix = operator.calculate(*panels(prefix_rows))
    assert full.shape == (rows, cols)
    assert np.isfinite(full.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(
        full.iloc[:prefix_rows].to_numpy(dtype=float),
        prefix.to_numpy(dtype=float),
        equal_nan=True,
    )
