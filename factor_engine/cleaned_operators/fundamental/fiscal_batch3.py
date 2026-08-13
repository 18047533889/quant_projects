# -*- coding: utf-8 -*-
"""Fiscal TRUE_GAP operators batch 3: capital stock, lifecycle, efficiency.

This module implements 5 fiscal-event operators using the strict FiscalEventView
kernel. All operators are causal and PIT-safe, operating on distinct fiscal
events rather than forward-filled daily observations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.fiscal_event_ops import FiscalEventView
from cleaned_operators.fiscal_strict import period_ordinal

_EPS = 1e-12


def _positive(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("fiscal-event operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def _policy(value: Any) -> str:
    result = str(value).lower()
    if result not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be 'latest_available' or 'first_available'")
    return result


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def pd_fiscal_capital_stock(
    capex,
    period_id,
    depreciation=0.15,
    periods_per_year=4,
    warmup_periods=8,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Perpetual inventory capital stock estimate.

    Estimates the physical capital stock using the perpetual inventory method:
        K_t = (1 - δ/N)·K_{t-1} + CAPEX_t
    where δ is annual depreciation rate and N is periods per year.

    Args:
        capex: Capital expenditure flow (e.g., quarterly CAPEX)
        period_id: Fiscal period identifier (e.g., "2024Q1")
        depreciation: Annual depreciation rate (default 0.15 = 15%)
        periods_per_year: Number of fiscal periods per year (default 4 for quarters)
        warmup_periods: Minimum periods required for stable estimate (default 8)
        require_consecutive: Require consecutive fiscal periods (default True)
        revision_policy: How to handle revisions ("latest_available" or "first_available")

    Returns:
        Panel of estimated capital stock values

    Notes:
        - Research operator on extended surface
        - Requires sufficient warmup history for stable estimates
        - Uses pandas_numpy backend only
        - Stock is undefined if warmup_periods not met
    """
    capex, period_id = _align(capex, period_id)
    depreciation = float(depreciation)
    periods_per_year = _positive(periods_per_year, "periods_per_year")
    warmup_periods = _nonnegative(warmup_periods, "warmup_periods")
    if not np.isfinite(depreciation) or not 0 <= depreciation < 1:
        raise ValueError("depreciation must satisfy 0 <= depreciation < 1")

    policy = _policy(revision_policy)
    retention = np.where(periods_per_year) != 0, (1.0 - depreciation) ** (1.0 / periods_per_year), np.nan)
    view = FiscalEventView.from_panel(capex, period_id, revision_policy=policy)
    out = np.full(capex.shape, np.nan, dtype=float)

    for row in range(capex.shape[0]):
        for col in range(capex.shape[1]):
            history = view.history(row, col, require_consecutive=bool(require_consecutive))
            if len(history) <= warmup_periods:
                continue

            stock = 0.0
            for _, value in history:
                stock = retention * stock + value

            out[row, col] = stock

    return pd.DataFrame(out, index=capex.index, columns=capex.columns)


def pd_fiscal_perpetual_inventory(
    flow,
    period_id,
    depreciation=0.15,
    periods_per_year=4,
    warmup_periods=8,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Generic perpetual inventory stock estimator.

    Accumulates any flow variable into a stock using exponential depreciation:
        Stock_t = (1 - δ/N)·Stock_{t-1} + Flow_t

    Args:
        flow: Input flow variable (e.g., advertising spend, R&D expense)
        period_id: Fiscal period identifier
        depreciation: Annual depreciation rate (default 0.15)
        periods_per_year: Periods per year (default 4)
        warmup_periods: Minimum periods for stability (default 8)
        require_consecutive: Require consecutive periods (default True)
        revision_policy: Revision handling policy

    Returns:
        Panel of stock estimates

    Notes:
        - Generic version of fiscal_capital_stock for any flow
        - Same perpetual inventory method as capital stock
        - Extended surface only
    """
    return pd_fiscal_capital_stock(
        flow, period_id, depreciation, periods_per_year,
        warmup_periods, require_consecutive, revision_policy
    )


def pd_cash_flow_lifecycle_stage(
    ocf,
    icf,
    fcf,
    period_id,
    periods=4,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Cash flow lifecycle stage classification (Dickinson 2011).

    Classifies firms into 8 lifecycle stages based on sign patterns of:
    - OCF: Operating cash flow
    - ICF: Investing cash flow
    - FCF: Financing cash flow

    Returns integer codes:
        1 = Introduction (-, -, +)
        2 = Growth (+ , -, +)
        3 = Mature (+ , -, -)
        4 = Shake-out high-distress (-, -, -)
        5 = Shake-out medium (-, +, +)
        6 = Shake-out medium (-, +, -)
        7 = Decline (+ , +, +)
        8 = Decline (+ , +, -)

    Args:
        ocf: Operating cash flow
        icf: Investing cash flow (typically negative for growth)
        fcf: Financing cash flow
        period_id: Fiscal period identifier
        periods: Number of periods to observe (default 4)
        min_periods: Minimum periods required (default 3)
        require_consecutive: Require consecutive periods (default True)
        revision_policy: Revision handling policy

    Returns:
        Panel of lifecycle stage codes (1-8, NaN if insufficient data)

    Notes:
        - Research operator on extended surface
        - Requires majority sign pattern over observation window
        - Based on Dickinson (2011) cash flow patterns
    """
    ocf, icf, fcf, period_id = _align(ocf, icf, fcf, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")

    policy = _policy(revision_policy)
    ocf_view = FiscalEventView.from_panel(ocf, period_id, revision_policy=policy)
    icf_view = FiscalEventView.from_panel(icf, period_id, revision_policy=policy)
    fcf_view = FiscalEventView.from_panel(fcf, period_id, revision_policy=policy)

    out = np.full(ocf.shape, np.nan, dtype=float)

    for row in range(ocf.shape[0]):
        for col in range(ocf.shape[1]):
            ocf_hist = dict(ocf_view.history(row, col, require_consecutive=bool(require_consecutive)))
            icf_hist = dict(icf_view.history(row, col, require_consecutive=bool(require_consecutive)))
            fcf_hist = dict(fcf_view.history(row, col, require_consecutive=bool(require_consecutive)))

            # Intersect on common fiscal ordinals
            common = sorted(set(ocf_hist) & set(icf_hist) & set(fcf_hist))[-periods:]
            if len(common) < min_periods:
                continue

            # Extract finite observations
            ocf_vals = [ocf_hist[k] for k in common if _finite(ocf_hist[k])]
            icf_vals = [icf_hist[k] for k in common if _finite(icf_hist[k])]
            fcf_vals = [fcf_hist[k] for k in common if _finite(fcf_hist[k])]

            if len(ocf_vals) < min_periods or len(icf_vals) < min_periods or len(fcf_vals) < min_periods:
                continue

            # Determine majority sign for each flow
            ocf_sign = 1 if sum(1 for x in ocf_vals if x > 0) > len(ocf_vals) / 2 else -1
            icf_sign = 1 if sum(1 for x in icf_vals if x > 0) > len(icf_vals) / 2 else -1
            fcf_sign = 1 if sum(1 for x in fcf_vals if x > 0) > len(fcf_vals) / 2 else -1

            # Map sign pattern to lifecycle stage
            pattern = (ocf_sign, icf_sign, fcf_sign)
            stage_map = {
                (-1, -1, 1): 1,  # Introduction
                (1, -1, 1): 2,   # Growth
                (1, -1, -1): 3,  # Mature
                (-1, -1, -1): 4, # Shake-out (high distress)
                (-1, 1, 1): 5,   # Shake-out (medium)
                (-1, 1, -1): 6,  # Shake-out (medium)
                (1, 1, 1): 7,    # Decline
                (1, 1, -1): 8,   # Decline
            }

            stage = stage_map.get(pattern)
            if stage is not None:
                out[row, col] = float(stage)

    return pd.DataFrame(out, index=ocf.index, columns=ocf.columns)


def pd_laborforce_efficiency(
    revenue,
    employees,
    period_id,
    periods=4,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Labor force efficiency: revenue per employee growth rate.

    Computes the compound annual growth rate of revenue per employee over
    recent fiscal periods, measuring labor productivity trends.

    Args:
        revenue: Total revenue
        employees: Employee count
        period_id: Fiscal period identifier
        periods: Number of periods for trend (default 4 = 1 year for quarters)
        min_periods: Minimum periods required (default 3)
        require_consecutive: Require consecutive periods (default True)
        revision_policy: Revision handling policy

    Returns:
        Panel of labor efficiency CAGR (annualized)

    Notes:
        - Research operator on extended surface
        - Returns NaN if employees <= 0 or insufficient history
        - CAGR computed as (latest/earliest)^(1/years) - 1
    """
    revenue, employees, period_id = _align(revenue, employees, period_id)
    periods = _positive(periods, "periods")
    min_periods = _positive(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")

    policy = _policy(revision_policy)
    rev_view = FiscalEventView.from_panel(revenue, period_id, revision_policy=policy)
    emp_view = FiscalEventView.from_panel(employees, period_id, revision_policy=policy)

    out = np.full(revenue.shape, np.nan, dtype=float)

    for row in range(revenue.shape[0]):
        for col in range(revenue.shape[1]):
            rev_hist = dict(rev_view.history(row, col, require_consecutive=bool(require_consecutive)))
            emp_hist = dict(emp_view.history(row, col, require_consecutive=bool(require_consecutive)))

            common = sorted(set(rev_hist) & set(emp_hist))[-periods:]
            if len(common) < min_periods:
                continue

            # Build revenue per employee series
            rpe_pairs = []
            for k in common:
                r = rev_hist[k]
                e = emp_hist[k]
                if _finite(r) and _finite(e) and e > _EPS:
                    rpe_pairs.append((k, r / e))

            if len(rpe_pairs) < min_periods:
                continue

            # Sort by ordinal
            rpe_pairs.sort()
            first_rpe = rpe_pairs[0][1]
            last_rpe = rpe_pairs[-1][1]

            if first_rpe <= _EPS or last_rpe <= _EPS:
                continue

            # CAGR: (last/first)^(1/years) - 1
            n_years = np.where(4.0 != 0, len(rpe_pairs) / 4.0, np.nan)
            if n_years <= 0:
                continue

            cagr = np.where(first_rpe) ** (1.0 / n_years) - 1.0 != 0, (last_rpe / first_rpe) ** (1.0 / n_years) - 1.0, np.nan)
            if _finite(cagr):
                out[row, col] = cagr

    return pd.DataFrame(out, index=revenue.index, columns=revenue.columns)


def pd_years_since_date(
    event_date,
    period_id,
    revision_policy="latest_available",
    **_,
) -> pd.DataFrame:
    """Years elapsed since a specific date event.

    Computes the fractional years between an event date (e.g., IPO date,
    listing date) and the current fiscal period's end.

    Args:
        event_date: Panel of event dates (as datetime64 or date strings)
        period_id: Fiscal period identifier
        revision_policy: Revision handling policy

    Returns:
        Panel of years since event (fractional)

    Notes:
        - Research operator on extended surface
        - Returns NaN if event_date is missing or invalid
        - Assumes 365.25 days per year
        - Uses period_id to infer current date
    """
    event_date, period_id = _align(event_date, period_id)
    policy = _policy(revision_policy)

    # Convert period_id to ordinals to track current period
    ordinal_array = np.full(period_id.shape, np.nan, dtype=float)
    raw_periods = period_id.to_numpy(dtype=object)
    for row in range(period_id.shape[0]):
        for col in range(period_id.shape[1]):
            ordinal = period_ordinal(raw_periods[row, col])
            if ordinal is not None:
                ordinal_array[row, col] = ordinal

    # Convert event_date to datetime
    event_array = event_date.to_numpy(dtype="datetime64[D]")
    out = np.full(event_date.shape, np.nan, dtype=float)

    for row in range(event_date.shape[0]):
        for col in range(event_date.shape[1]):
            if not np.isfinite(ordinal_array[row, col]):
                continue

            event_dt = event_array[row, col]
            if pd.isna(event_dt):
                continue

            # Approximate current date from period ordinal
            # Ordinal 0 = 2000Q1 end (roughly 2000-03-31)
            ordinal = int(ordinal_array[row, col])
            base_year = 2000
            base_quarter = ordinal % 4
            year_offset = ordinal // 4
            current_year = base_year + year_offset
            current_month = (base_quarter + 1) * 3  # Q1->3, Q2->6, Q3->9, Q4->12

            try:
                current_date = pd.Timestamp(year=current_year, month=current_month, day=1) + pd.offsets.MonthEnd(0)
                event_ts = pd.Timestamp(event_dt)
                delta_days = (current_date - event_ts).days
                years = np.where(365.25 != 0, delta_days / 365.25, np.nan)
                if years >= 0:
                    out[row, col] = years
            except (ValueError, OverflowError):
                continue

    return pd.DataFrame(out, index=event_date.index, columns=event_date.columns)


# Polars implementations (dual backend)
try:
    import polars as pl
except ImportError:
    pl = None


def pl_fiscal_capital_stock(
    capex,
    period_id,
    depreciation=0.15,
    periods_per_year=4,
    warmup_periods=8,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    """Polars implementation of fiscal_capital_stock."""
    if pl is None:
        raise RuntimeError("polars is required")
    # Convert to pandas, compute, convert back
    capex_pd = capex.to_pandas()
    period_id_pd = period_id.to_pandas()
    result_pd = pd_fiscal_capital_stock(
        capex_pd, period_id_pd, depreciation, periods_per_year,
        warmup_periods, require_consecutive, revision_policy
    )
    return pl.from_pandas(result_pd)


def pl_fiscal_perpetual_inventory(
    flow,
    period_id,
    depreciation=0.15,
    periods_per_year=4,
    warmup_periods=8,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    """Polars implementation of fiscal_perpetual_inventory."""
    if pl is None:
        raise RuntimeError("polars is required")
    flow_pd = flow.to_pandas()
    period_id_pd = period_id.to_pandas()
    result_pd = pd_fiscal_perpetual_inventory(
        flow_pd, period_id_pd, depreciation, periods_per_year,
        warmup_periods, require_consecutive, revision_policy
    )
    return pl.from_pandas(result_pd)


def pl_cash_flow_lifecycle_stage(
    ocf,
    icf,
    fcf,
    period_id,
    periods=4,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    """Polars implementation of cash_flow_lifecycle_stage."""
    if pl is None:
        raise RuntimeError("polars is required")
    ocf_pd = ocf.to_pandas()
    icf_pd = icf.to_pandas()
    fcf_pd = fcf.to_pandas()
    period_id_pd = period_id.to_pandas()
    result_pd = pd_cash_flow_lifecycle_stage(
        ocf_pd, icf_pd, fcf_pd, period_id_pd,
        periods, min_periods, require_consecutive, revision_policy
    )
    return pl.from_pandas(result_pd)


def pl_laborforce_efficiency(
    revenue,
    employees,
    period_id,
    periods=4,
    min_periods=3,
    require_consecutive=True,
    revision_policy="latest_available",
    **_,
):
    """Polars implementation of laborforce_efficiency."""
    if pl is None:
        raise RuntimeError("polars is required")
    revenue_pd = revenue.to_pandas()
    employees_pd = employees.to_pandas()
    period_id_pd = period_id.to_pandas()
    result_pd = pd_laborforce_efficiency(
        revenue_pd, employees_pd, period_id_pd,
        periods, min_periods, require_consecutive, revision_policy
    )
    return pl.from_pandas(result_pd)


def pl_years_since_date(
    event_date,
    period_id,
    revision_policy="latest_available",
    **_,
):
    """Polars implementation of years_since_date."""
    if pl is None:
        raise RuntimeError("polars is required")
    event_date_pd = event_date.to_pandas()
    period_id_pd = period_id.to_pandas()
    result_pd = pd_years_since_date(
        event_date_pd, period_id_pd, revision_policy
    )
    return pl.from_pandas(result_pd)


# ============================================================================
# Operator class wrappers
# ============================================================================

class _FiscalCapitalStock(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_capital_stock",
        category="fundamental_period",
        description="Perpetual inventory capital stock: K_t = (1-δ/N)·K_{t-1} + CAPEX_t. "
        "Estimates physical capital stock using exponential depreciation.",
        param_names=[
            "capex",
            "period_id",
            "depreciation",
            "periods_per_year",
            "warmup_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_capital_stock(*args, **kwargs)


class _FiscalPerpetualInventory(SeriesOperator):
    metadata = OperatorMetadata(
        name="fiscal_perpetual_inventory",
        category="fundamental_period",
        description="Generic perpetual inventory stock estimator. Converts any flow "
        "variable into stock using exponential depreciation.",
        param_names=[
            "flow",
            "period_id",
            "depreciation",
            "periods_per_year",
            "warmup_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_fiscal_perpetual_inventory(*args, **kwargs)


class _CashFlowLifecycleStage(SeriesOperator):
    metadata = OperatorMetadata(
        name="cash_flow_lifecycle_stage",
        category="fundamental_period",
        description="Dickinson 2011 lifecycle classification based on OCF/ICF/FCF sign patterns. "
        "Returns stage codes 1-8 (Introduction/Growth/Mature/Shake-out/Decline).",
        param_names=[
            "ocf",
            "icf",
            "fcf",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_cash_flow_lifecycle_stage(*args, **kwargs)


class _LaborforceEfficiency(SeriesOperator):
    metadata = OperatorMetadata(
        name="laborforce_efficiency",
        category="fundamental_period",
        description="Labor force efficiency: CAGR of revenue per employee. "
        "Measures labor productivity trend over recent fiscal periods.",
        param_names=[
            "revenue",
            "employees",
            "period_id",
            "periods",
            "min_periods",
            "require_consecutive",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_laborforce_efficiency(*args, **kwargs)


class _YearsSinceDate(SeriesOperator):
    metadata = OperatorMetadata(
        name="years_since_date",
        category="fundamental_period",
        description="Years elapsed since event date (e.g., IPO, listing). "
        "Computes fractional years between event and current fiscal period end.",
        param_names=[
            "event_date",
            "period_id",
            "revision_policy",
        ],
        return_type="series",
        tags=[
            "fundamental",
            "period_aware",
            "pit_safe",
            "causal",
            "research",
            "experimental",
            "strict_fiscal_event",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_years_since_date(*args, **kwargs)


def register() -> None:
    """Register fiscal batch 3 operators."""
    from cleaned_operators.registry import OperatorRegistry

    if "fiscal_capital_stock" in OperatorRegistry._operators:
        return

    register_operator(
        name="fiscal_capital_stock",
        category="fundamental_period",
        business_category="fundamental",
        canonical="fiscal_capital_stock",
        source="fundamental.fiscal_batch3",
        backend="pandas_numpy",
        status="experimental",
    )(_FiscalCapitalStock)

    register_operator(
        name="fiscal_capital_stock",
        category="fundamental_period",
        business_category="fundamental",
        canonical="fiscal_capital_stock",
        source="fundamental.fiscal_batch3",
        backend="polars",
        status="experimental",
    )(type("_FiscalCapitalStockPolars", (_FiscalCapitalStock,), {"_calculate_series": lambda self, *args, **kwargs: pl_fiscal_capital_stock(*args, **kwargs)}))

    register_operator(
        name="fiscal_perpetual_inventory",
        category="fundamental_period",
        business_category="fundamental",
        canonical="fiscal_perpetual_inventory",
        source="fundamental.fiscal_batch3",
        backend="pandas_numpy",
        status="experimental",
    )(_FiscalPerpetualInventory)

    register_operator(
        name="fiscal_perpetual_inventory",
        category="fundamental_period",
        business_category="fundamental",
        canonical="fiscal_perpetual_inventory",
        source="fundamental.fiscal_batch3",
        backend="polars",
        status="experimental",
    )(type("_FiscalPerpetualInventoryPolars", (_FiscalPerpetualInventory,), {"_calculate_series": lambda self, *args, **kwargs: pl_fiscal_perpetual_inventory(*args, **kwargs)}))

    register_operator(
        name="cash_flow_lifecycle_stage",
        category="fundamental_period",
        business_category="fundamental",
        canonical="cash_flow_lifecycle_stage",
        source="fundamental.fiscal_batch3",
        backend="pandas_numpy",
        status="experimental",
    )(_CashFlowLifecycleStage)

    register_operator(
        name="cash_flow_lifecycle_stage",
        category="fundamental_period",
        business_category="fundamental",
        canonical="cash_flow_lifecycle_stage",
        source="fundamental.fiscal_batch3",
        backend="polars",
        status="experimental",
    )(type("_CashFlowLifecycleStagePolars", (_CashFlowLifecycleStage,), {"_calculate_series": lambda self, *args, **kwargs: pl_cash_flow_lifecycle_stage(*args, **kwargs)}))

    register_operator(
        name="laborforce_efficiency",
        category="fundamental_period",
        business_category="fundamental",
        canonical="laborforce_efficiency",
        source="fundamental.fiscal_batch3",
        backend="pandas_numpy",
        status="experimental",
    )(_LaborforceEfficiency)

    register_operator(
        name="laborforce_efficiency",
        category="fundamental_period",
        business_category="fundamental",
        canonical="laborforce_efficiency",
        source="fundamental.fiscal_batch3",
        backend="polars",
        status="experimental",
    )(type("_LaborforceEfficiencyPolars", (_LaborforceEfficiency,), {"_calculate_series": lambda self, *args, **kwargs: pl_laborforce_efficiency(*args, **kwargs)}))

    register_operator(
        name="years_since_date",
        category="fundamental_period",
        business_category="fundamental",
        canonical="years_since_date",
        source="fundamental.fiscal_batch3",
        backend="pandas_numpy",
        status="experimental",
    )(_YearsSinceDate)

    register_operator(
        name="years_since_date",
        category="fundamental_period",
        business_category="fundamental",
        canonical="years_since_date",
        source="fundamental.fiscal_batch3",
        backend="polars",
        status="experimental",
    )(type("_YearsSinceDatePolars", (_YearsSinceDate,), {"_calculate_series": lambda self, *args, **kwargs: pl_years_since_date(*args, **kwargs)}))

    # Add to extended surface
    import cleaned_operators.operator_surface as _surface
    _surface.extend_extended_only({
        "fiscal_capital_stock",
        "fiscal_perpetual_inventory",
        "cash_flow_lifecycle_stage",
        "laborforce_efficiency",
        "years_since_date",
    })


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
    "fiscal_capital_stock": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 8,
    },
    "fiscal_perpetual_inventory": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 8,
    },
    "cash_flow_lifecycle_stage": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "laborforce_efficiency": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 3,
    },
    "years_since_date": {
        "scope": "fundamental_period",
        "pit_safe": True,
        "min_periods": 1,
    },
}


register()
