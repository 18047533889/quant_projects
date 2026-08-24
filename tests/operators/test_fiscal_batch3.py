# -*- coding: utf-8 -*-
"""Tests for fiscal_batch3 operators: capital stock, lifecycle, efficiency.

Covers 5 TRUE_GAP fiscal operators:
- fiscal_capital_stock
- fiscal_perpetual_inventory
- cash_flow_lifecycle_stage
- laborforce_efficiency
- years_since_date
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def setup_module():
    """Reset registry lifecycle to allow operator registration during test imports."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._lifecycle != OperatorRegistry.Lifecycle.BUILDING:
        OperatorRegistry._lifecycle = OperatorRegistry.Lifecycle.BUILDING

    # Bypass layer governance check
    try:
        import factor_engine.cleaned_operators.layer_governance as gov
        gov._FINALIZED = False
    except (ImportError, AttributeError):
        pass

    # Bypass mutation token check for test isolation
    try:
        from factor_engine.cleaned_operators import registry
        registry.OperatorRegistry._mutation_token = registry._BOOTSTRAP_TOKEN
        # Reset _operators and _catalog to mutable dicts if they're frozen
        if not isinstance(registry.OperatorRegistry._operators, type({})):
            registry.OperatorRegistry._operators = {}
        if not isinstance(registry.OperatorRegistry._catalog, type({})):
            registry.OperatorRegistry._catalog = {}
    except (ImportError, AttributeError):
        pass


from factor_engine.cleaned_operators.registry import OperatorRegistry


@pytest.fixture(scope="module")
def ensure_registered():
    """Ensure fiscal_batch3 operators are registered."""
    import factor_engine.cleaned_operators.fundamental.fiscal_batch3
    return True


def _panel(values, periods=5, instruments=3):
    """Create test panel."""
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    cols = [f"S{i}" for i in range(instruments)]
    if isinstance(values, (int, float)):
        arr = np.full((periods, instruments), values, dtype=float)
    elif isinstance(values, list):
        arr = np.array(values, dtype=float)
    else:
        arr = values
    return pd.DataFrame(arr, index=idx, columns=cols)


def _period_panel(periods_list, n_periods=5, n_instruments=3):
    """Create period_id panel."""
    idx = pd.date_range("2024-01-01", periods=n_periods, freq="D")
    cols = [f"S{i}" for i in range(n_instruments)]
    if isinstance(periods_list[0], str):
        arr = np.array([periods_list] * n_periods, dtype=object)
    else:
        arr = np.array(periods_list, dtype=object)
    return pd.DataFrame(arr, index=idx, columns=cols)


# =============================================================================
# fiscal_capital_stock
# =============================================================================

def test_fiscal_capital_stock_basic(ensure_registered):
    """Test basic capital stock accumulation."""
    capex = _panel([
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ], periods=10)

    period_id = _period_panel(
        ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"],
        n_periods=10, n_instruments=3
    )

    op = OperatorRegistry.get("fiscal_capital_stock", backend="pandas_numpy")
    result = op(capex, period_id, depreciation=0.2, periods_per_year=4, warmup_periods=4)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == capex.shape
    # After warmup (row 4+), should have positive stock
    assert result.iloc[5, 0] > 100  # Stock accumulates
    assert np.all(np.isfinite(result.iloc[5:]))


def test_fiscal_capital_stock_warmup_requirement(ensure_registered):
    """Test that warmup_periods gate is enforced."""
    capex = _panel([
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ], periods=3)

    period_id = _period_panel(["2024Q1", "2024Q2", "2024Q3"], n_periods=3, n_instruments=3)

    op = OperatorRegistry.get("fiscal_capital_stock", backend="pandas_numpy")
    result = op(capex, period_id, warmup_periods=8)

    # Insufficient warmup: all NaN
    assert np.all(np.isnan(result))


def test_fiscal_capital_stock_depreciation_effect(ensure_registered):
    """Test that higher depreciation reduces stock."""
    capex = _panel([
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
        [100] * 3,
    ], periods=10)

    period_id = _period_panel(
        ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"],
        n_periods=10, n_instruments=3
    )

    op = OperatorRegistry.get("fiscal_capital_stock", backend="pandas_numpy")
    result_low_dep = op(capex, period_id, depreciation=0.05, warmup_periods=4)
    result_high_dep = op(capex, period_id, depreciation=0.30, warmup_periods=4)

    # Lower depreciation => higher accumulated stock
    assert result_low_dep.iloc[-1, 0] > result_high_dep.iloc[-1, 0]


def test_fiscal_capital_stock_nan_handling(ensure_registered):
    """Test handling of missing CAPEX values."""
    capex = _panel([
        [100, np.nan, 100],
        [100, 100, 100],
        [100, 100, np.nan],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ], periods=9)

    period_id = _period_panel(
        ["2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1"],
        n_periods=9, n_instruments=3
    )

    op = OperatorRegistry.get("fiscal_capital_stock", backend="pandas_numpy")
    result = op(capex, period_id, warmup_periods=4)

    # S0 should compute despite S1 having NaN
    assert np.isfinite(result.iloc[-1, 0])


# =============================================================================
# fiscal_perpetual_inventory
# =============================================================================

def test_fiscal_perpetual_inventory_basic(ensure_registered):
    """Test generic perpetual inventory."""
    flow = _panel([
        [50] * 2,
        [50] * 2,
        [50] * 2,
        [50] * 2,
        [50] * 2,
        [50] * 2,
        [50] * 2,
        [50] * 2,
        [50] * 2,
    ], periods=9, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=9, n_instruments=2)

    op = OperatorRegistry.get("fiscal_perpetual_inventory", backend="pandas_numpy")
    result = op(flow, period_id, depreciation=0.10, warmup_periods=4)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == flow.shape
    # Stock accumulates over time
    assert result.iloc[-1, 0] > flow.iloc[-1, 0]


def test_fiscal_perpetual_inventory_matches_capital_stock(ensure_registered):
    """Test that fiscal_perpetual_inventory is equivalent to fiscal_capital_stock."""
    flow = _panel([
        [80] * 2,
        [80] * 2,
        [80] * 2,
        [80] * 2,
        [80] * 2,
        [80] * 2,
        [80] * 2,
        [80] * 2,
        [80] * 2,
    ], periods=9, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=9, n_instruments=2)

    op_inv = OperatorRegistry.get("fiscal_perpetual_inventory", backend="pandas_numpy")
    op_cap = OperatorRegistry.get("fiscal_capital_stock", backend="pandas_numpy")

    result_inv = op_inv(flow, period_id, depreciation=0.15, warmup_periods=3)
    result_cap = op_cap(flow, period_id, depreciation=0.15, warmup_periods=3)

    # Should be identical
    pd.testing.assert_frame_equal(result_inv, result_cap)


# =============================================================================
# cash_flow_lifecycle_stage
# =============================================================================

def test_cash_flow_lifecycle_stage_introduction(ensure_registered):
    """Test Introduction stage: OCF-, ICF-, FCF+."""
    ocf = _panel([
        [-10, -10, -10],
        [-10, -10, -10],
        [-10, -10, -10],
        [-10, -10, -10],
    ], periods=4)

    icf = _panel([
        [-50, -50, -50],
        [-50, -50, -50],
        [-50, -50, -50],
        [-50, -50, -50],
    ], periods=4)

    fcf = _panel([
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
        [100, 100, 100],
    ], periods=4)

    period_id = _period_panel(["2023Q1", "2023Q2", "2023Q3"], n_periods=4, n_instruments=3)

    op = OperatorRegistry.get("cash_flow_lifecycle_stage", backend="pandas_numpy")
    result = op(ocf, icf, fcf, period_id, periods=4, min_periods=3)

    assert result.iloc[-1, 0] == 1.0  # Introduction stage


def test_cash_flow_lifecycle_stage_growth(ensure_registered):
    """Test Growth stage: OCF+, ICF-, FCF+."""
    ocf = _panel([
        [50, 50, 50],
        [50, 50, 50],
        [50, 50, 50],
        [50, 50, 50],
    ], periods=4)

    icf = _panel([
        [-80, -80, -80],
        [-80, -80, -80],
        [-80, -80, -80],
        [-80, -80, -80],
    ], periods=4)

    fcf = _panel([
        [30, 30, 30],
        [30, 30, 30],
        [30, 30, 30],
        [30, 30, 30],
    ], periods=4)

    period_id = _period_panel(["2023Q1", "2023Q2", "2023Q3"], n_periods=4, n_instruments=3)

    op = OperatorRegistry.get("cash_flow_lifecycle_stage", backend="pandas_numpy")
    result = op(ocf, icf, fcf, period_id, periods=4, min_periods=3)

    assert result.iloc[-1, 0] == 2.0  # Growth stage


def test_cash_flow_lifecycle_stage_mature(ensure_registered):
    """Test Mature stage: OCF+, ICF-, FCF-."""
    ocf = _panel([[80] * 2] * 4, periods=4, instruments=2)
    icf = _panel([[-20] * 2] * 4, periods=4, instruments=2)
    fcf = _panel([[-40] * 2] * 4, periods=4, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=4, n_instruments=2)

    op = OperatorRegistry.get("cash_flow_lifecycle_stage", backend="pandas_numpy")
    result = op(ocf, icf, fcf, period_id, periods=4, min_periods=3)

    assert result.iloc[-1, 0] == 3.0  # Mature stage


def test_cash_flow_lifecycle_stage_decline(ensure_registered):
    """Test Decline stage: OCF+, ICF+, FCF-."""
    ocf = _panel([[60] * 2] * 4, periods=4, instruments=2)
    icf = _panel([[10] * 2] * 4, periods=4, instruments=2)
    fcf = _panel([[-30] * 2] * 4, periods=4, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=4, n_instruments=2)

    op = OperatorRegistry.get("cash_flow_lifecycle_stage", backend="pandas_numpy")
    result = op(ocf, icf, fcf, period_id, periods=4, min_periods=3)

    assert result.iloc[-1, 0] == 8.0  # Decline stage


def test_cash_flow_lifecycle_stage_insufficient_data(ensure_registered):
    """Test that insufficient periods returns NaN."""
    ocf = _panel([[50] * 2] * 2, periods=2, instruments=2)
    icf = _panel([[-30] * 2] * 2, periods=2, instruments=2)
    fcf = _panel([[20] * 2] * 2, periods=2, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=2, n_instruments=2)

    op = OperatorRegistry.get("cash_flow_lifecycle_stage", backend="pandas_numpy")
    result = op(ocf, icf, fcf, period_id, periods=4, min_periods=3)

    # Insufficient history
    assert np.all(np.isnan(result))


# =============================================================================
# laborforce_efficiency
# =============================================================================

def test_laborforce_efficiency_positive_growth(ensure_registered):
    """Test positive labor efficiency growth."""
    revenue = _panel([
        [1000, 1000],
        [1100, 1100],
        [1200, 1200],
        [1300, 1300],
        [1400, 1400],
    ], periods=5, instruments=2)

    employees = _panel([
        [100, 100],
        [100, 100],
        [100, 100],
        [100, 100],
        [100, 100],
    ], periods=5, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=5, n_instruments=2)

    op = OperatorRegistry.get("laborforce_efficiency", backend="pandas_numpy")
    result = op(revenue, employees, period_id, periods=4, min_periods=3)

    assert result.iloc[-1, 0] > 0  # Positive CAGR


def test_laborforce_efficiency_negative_growth(ensure_registered):
    """Test negative labor efficiency growth."""
    revenue = _panel([
        [1400, 1400],
        [1300, 1300],
        [1200, 1200],
        [1100, 1100],
        [1000, 1000],
    ], periods=5, instruments=2)

    employees = _panel([
        [100, 100],
        [100, 100],
        [100, 100],
        [100, 100],
        [100, 100],
    ], periods=5, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=5, n_instruments=2)

    op = OperatorRegistry.get("laborforce_efficiency", backend="pandas_numpy")
    result = op(revenue, employees, period_id, periods=4, min_periods=3)

    assert result.iloc[-1, 0] < 0  # Negative CAGR


def test_laborforce_efficiency_zero_employees(ensure_registered):
    """Test handling of zero/negative employees."""
    revenue = _panel([[1000] * 2] * 5, periods=5, instruments=2)
    employees = _panel([
        [100, 0],
        [100, -10],
        [100, 100],
        [100, 100],
        [100, 100],
    ], periods=5, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=5, n_instruments=2)

    op = OperatorRegistry.get("laborforce_efficiency", backend="pandas_numpy")
    result = op(revenue, employees, period_id, periods=4, min_periods=3)

    # S0 should compute, S1 should be NaN (invalid employees)
    assert np.isfinite(result.iloc[-1, 0])
    assert np.isnan(result.iloc[-1, 1])


def test_laborforce_efficiency_insufficient_history(ensure_registered):
    """Test insufficient history returns NaN."""
    revenue = _panel([[1000] * 2] * 2, periods=2, instruments=2)
    employees = _panel([[100] * 2] * 2, periods=2, instruments=2)

    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=2, n_instruments=2)

    op = OperatorRegistry.get("laborforce_efficiency", backend="pandas_numpy")
    result = op(revenue, employees, period_id, periods=4, min_periods=3)

    assert np.all(np.isnan(result))


# =============================================================================
# years_since_date
# =============================================================================

def test_years_since_date_basic(ensure_registered):
    """Test basic years since event calculation."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    cols = ["S0", "S1"]

    event_date = pd.DataFrame([
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-15")],
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-15")],
        [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-15")],
    ], index=idx, columns=cols)

    period_id = pd.DataFrame([
        ["2024Q1", "2024Q1"],
        ["2024Q1", "2024Q1"],
        ["2024Q1", "2024Q1"],
    ], index=idx, columns=cols, dtype=object)

    op = OperatorRegistry.get("years_since_date", backend="pandas_numpy")
    result = op(event_date, period_id)

    assert result.iloc[0, 0] > 3.9  # ~4 years
    assert result.iloc[0, 0] < 4.5
    assert result.iloc[0, 1] > 2.5  # ~2.75 years
    assert result.iloc[0, 1] < 3.0


def test_years_since_date_nan_handling(ensure_registered):
    """Test handling of missing event dates."""
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = ["S0", "S1"]

    event_date = pd.DataFrame([
        [pd.Timestamp("2020-01-01"), pd.NaT],
        [pd.Timestamp("2020-01-01"), pd.NaT],
    ], index=idx, columns=cols)

    period_id = pd.DataFrame([
        ["2024Q1", "2024Q1"],
        ["2024Q1", "2024Q1"],
    ], index=idx, columns=cols, dtype=object)

    op = OperatorRegistry.get("years_since_date", backend="pandas_numpy")
    result = op(event_date, period_id)

    assert np.isfinite(result.iloc[0, 0])
    assert np.isnan(result.iloc[0, 1])


def test_years_since_date_recent_event(ensure_registered):
    """Test recent event (less than 1 year)."""
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    cols = ["S0"]

    event_date = pd.DataFrame([
        [pd.Timestamp("2023-10-01")],
        [pd.Timestamp("2023-10-01")],
    ], index=idx, columns=cols)

    period_id = pd.DataFrame([
        ["2024Q1"],
        ["2024Q1"],
    ], index=idx, columns=cols, dtype=object)

    op = OperatorRegistry.get("years_since_date", backend="pandas_numpy")
    result = op(event_date, period_id)

    assert result.iloc[0, 0] > 0.2  # ~0.25 years
    assert result.iloc[0, 0] < 0.5


# =============================================================================
# Polars backend tests
# =============================================================================

@pytest.mark.parametrize("op_name", [
    "fiscal_capital_stock",
    "fiscal_perpetual_inventory",
    "cash_flow_lifecycle_stage",
    "laborforce_efficiency",
    "years_since_date",
])
def test_polars_backend_registered(ensure_registered, op_name):
    """Test that Polars backend is registered for all operators."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not available")

    op = OperatorRegistry.get_operator(op_name, backend="polars")
    assert op is not None


def test_fiscal_capital_stock_polars_smoke(ensure_registered):
    """Smoke test for Polars backend."""
    try:
        import polars as pl
    except ImportError:
        pytest.skip("polars not available")

    capex = _panel([[100] * 2] * 9, periods=9, instruments=2)
    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=9, n_instruments=2)

    capex_pl = pl.from_pandas(capex)
    period_id_pl = pl.from_pandas(period_id)

    op = OperatorRegistry.get_operator("fiscal_capital_stock", backend="polars")
    result = op(capex_pl, period_id_pl, warmup_periods=4)

    assert isinstance(result, pl.DataFrame)
    assert result.shape == capex_pl.shape


# =============================================================================
# Parameter validation
# =============================================================================

def test_fiscal_capital_stock_invalid_depreciation(ensure_registered):
    """Test invalid depreciation parameter."""
    capex = _panel([[100] * 2] * 5, periods=5, instruments=2)
    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=5, n_instruments=2)

    op = OperatorRegistry.get("fiscal_capital_stock", backend="pandas_numpy")

    with pytest.raises(ValueError, match="depreciation must satisfy"):
        op(capex, period_id, depreciation=1.5)

    with pytest.raises(ValueError, match="depreciation must satisfy"):
        op(capex, period_id, depreciation=-0.1)


def test_laborforce_efficiency_invalid_min_periods(ensure_registered):
    """Test invalid min_periods parameter."""
    revenue = _panel([[1000] * 2] * 5, periods=5, instruments=2)
    employees = _panel([[100] * 2] * 5, periods=5, instruments=2)
    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=5, n_instruments=2)

    op = OperatorRegistry.get("laborforce_efficiency", backend="pandas_numpy")

    with pytest.raises(ValueError, match="min_periods must not exceed periods"):
        op(revenue, employees, period_id, periods=3, min_periods=5)


def test_cash_flow_lifecycle_stage_invalid_revision_policy(ensure_registered):
    """Test invalid revision_policy parameter."""
    ocf = _panel([[50] * 2] * 4, periods=4, instruments=2)
    icf = _panel([[-30] * 2] * 4, periods=4, instruments=2)
    fcf = _panel([[20] * 2] * 4, periods=4, instruments=2)
    period_id = _period_panel(["2023Q1", "2023Q2"], n_periods=4, n_instruments=2)

    op = OperatorRegistry.get("cash_flow_lifecycle_stage", backend="pandas_numpy")

    with pytest.raises(ValueError, match="revision_policy must be"):
        op(ocf, icf, fcf, period_id, revision_policy="invalid_policy")
