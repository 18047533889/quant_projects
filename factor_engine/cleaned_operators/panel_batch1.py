# -*- coding: utf-8 -*-
"""Panel batch1: day/night beta gap, liquidity-adjusted beta, price delay score.

This module implements 3 panel operators for cross-sectional pricing and
microstructure analysis.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12


def _positive_int(value: Any, name: str) -> int:
    """Validate positive integer parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    """Validate non-negative integer parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a non-negative integer") from exc
    if result < 0 or float(value) != result:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _validate_hour(value: Any, name: str) -> int:
    """Validate hour parameter (0-23)."""
    hour = _nonnegative_int(value, name)
    if hour > 23:
        raise ValueError(f"{name} must be in [0, 23]")
    return hour


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Validate and align input DataFrames."""
    if not frames:
        return ()
    base = frames[0]
    if not isinstance(base, pd.DataFrame):
        raise TypeError("panel operators require pandas DataFrame inputs")
    for i, frame in enumerate(frames[1:], 1):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"input {i} must be a pandas DataFrame")
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            raise ValueError(f"input {i} is not aligned with the primary panel")
    return frames


def pd_panel_day_night_beta_gap(
    ret,
    benchmark_ret,
    day_start=9,
    day_end=15,
    night_start=16,
    night_end=23,
    window=60,
    **_,
) -> pd.DataFrame:
    """Day minus night beta gap (intraday pricing anomaly).

    Computes rolling beta during day hours vs night hours, then returns
    beta_day - beta_night. Positive values indicate stronger day comovement.

    Args:
        ret: Asset returns panel (must have datetime index with hour information)
        benchmark_ret: Benchmark returns panel (aligned with ret)
        day_start: Day session start hour (default 9, inclusive)
        day_end: Day session end hour (default 15, inclusive)
        night_start: Night session start hour (default 16, inclusive)
        night_end: Night session end hour (default 23, inclusive)
        window: Trailing window for beta estimation (default 60 observations)

    Returns:
        Panel of beta_day - beta_night values

    Notes:
        - PIT-safe: uses only trailing observations
        - Requires datetime index with hour component
        - Returns NaN when insufficient observations in either regime
        - Backend: pandas_numpy only (requires datetime operations)
    """
    ret, benchmark_ret = _align(ret, benchmark_ret)
    day_start = _validate_hour(day_start, "day_start")
    day_end = _validate_hour(day_end, "day_end")
    night_start = _validate_hour(night_start, "night_start")
    night_end = _validate_hour(night_end, "night_end")
    window = _positive_int(window, "window")

    if day_start > day_end:
        raise ValueError("day_start must be <= day_end")
    if night_start > night_end:
        raise ValueError("night_start must be <= night_end")

    # Extract hour from index
    if not isinstance(ret.index, pd.DatetimeIndex):
        raise TypeError("panel_day_night_beta_gap requires DatetimeIndex")

    hours = ret.index.hour
    day_mask = (hours >= day_start) & (hours <= day_end)
    night_mask = (hours >= night_start) & (hours <= night_end)

    out = np.full(ret.shape, np.nan, dtype=float)

    for col_idx in range(ret.shape[1]):
        ret_series = ret.iloc[:, col_idx].values
        bench_series = benchmark_ret.iloc[:, col_idx].values

        for row_idx in range(ret.shape[0]):
            if row_idx < window - 1:
                continue

            start = row_idx - window + 1
            ret_window = ret_series[start : row_idx + 1]
            bench_window = bench_series[start : row_idx + 1]
            day_window = day_mask[start : row_idx + 1]
            night_window = night_mask[start : row_idx + 1]

            # Day beta
            day_ret = ret_window[day_window]
            day_bench = bench_window[day_window]
            valid_day = np.isfinite(day_ret) & np.isfinite(day_bench)
            if valid_day.sum() >= 3:
                day_r = day_ret[valid_day]
                day_b = day_bench[valid_day]
                cov = np.cov(day_r, day_b, ddof=1)[0, 1]
                var_b = np.var(day_b, ddof=1)
                beta_day = cov / var_b if var_b > _EPS else np.nan
            else:
                beta_day = np.nan

            # Night beta
            night_ret = ret_window[night_window]
            night_bench = bench_window[night_window]
            valid_night = np.isfinite(night_ret) & np.isfinite(night_bench)
            if valid_night.sum() >= 3:
                night_r = night_ret[valid_night]
                night_b = night_bench[valid_night]
                cov = np.cov(night_r, night_b, ddof=1)[0, 1]
                var_b = np.var(night_b, ddof=1)
                beta_night = cov / var_b if var_b > _EPS else np.nan
            else:
                beta_night = np.nan

            if np.isfinite(beta_day) and np.isfinite(beta_night):
                out[row_idx, col_idx] = beta_day - beta_night

    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def pd_pastor_stambaugh_beta(
    ret,
    benchmark_ret,
    liquidity_proxy,
    window=60,
    **_,
) -> pd.DataFrame:
    """Liquidity-adjusted beta (Pastor-Stambaugh).

    Computes beta after orthogonalizing returns with respect to liquidity proxy.
    Returns beta from regression: ret ~ benchmark_ret after both are residualized
    against liquidity_proxy.

    Args:
        ret: Asset returns panel
        benchmark_ret: Benchmark returns panel (aligned with ret)
        liquidity_proxy: Liquidity factor panel (e.g., volume, spread)
        window: Trailing window for beta estimation (default 60)

    Returns:
        Panel of liquidity-adjusted beta values

    Notes:
        - PIT-safe: uses only trailing observations
        - Three-step: (1) ret ~ liq, (2) bench ~ liq, (3) resid_ret ~ resid_bench
        - Returns NaN when insufficient observations or rank deficiency
        - Backend: pandas_numpy only
    """
    ret, benchmark_ret, liquidity_proxy = _align(ret, benchmark_ret, liquidity_proxy)
    window = _positive_int(window, "window")

    out = np.full(ret.shape, np.nan, dtype=float)

    for col_idx in range(ret.shape[1]):
        ret_series = ret.iloc[:, col_idx].values
        bench_series = benchmark_ret.iloc[:, col_idx].values
        liq_series = liquidity_proxy.iloc[:, col_idx].values

        for row_idx in range(window - 1, ret.shape[0]):
            start = row_idx - window + 1
            r = ret_series[start : row_idx + 1]
            b = bench_series[start : row_idx + 1]
            liq = liq_series[start : row_idx + 1]

            valid = np.isfinite(r) & np.isfinite(b) & np.isfinite(liq)
            if valid.sum() < 5:
                continue

            r_v = r[valid]
            b_v = b[valid]
            liq_v = liq[valid]

            # Orthogonalize ret wrt liquidity
            try:
                X_liq = np.column_stack([np.ones(len(liq_v)), liq_v])
                if np.linalg.matrix_rank(X_liq) != 2:
                    continue
                beta_r_liq = np.linalg.lstsq(X_liq, r_v, rcond=None)[0]
                resid_ret = r_v - X_liq @ beta_r_liq

                # Orthogonalize benchmark wrt liquidity
                beta_b_liq = np.linalg.lstsq(X_liq, b_v, rcond=None)[0]
                resid_bench = b_v - X_liq @ beta_b_liq

                # Beta of residuals
                cov = np.cov(resid_ret, resid_bench, ddof=1)[0, 1]
                var_b = np.var(resid_bench, ddof=1)
                if var_b > _EPS:
                    out[row_idx, col_idx] = (cov) / var_b if var_b != 0 else np.nan
            except (np.linalg.LinAlgError, ValueError):
                continue

    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def pd_price_delay_score(
    ret,
    benchmark_ret,
    window=60,
    max_lags=5,
    **_,
) -> pd.DataFrame:
    """Price discovery delay score (Hou-Moskowitz 2005).

    Measures the fraction of variance explained by lagged benchmark returns
    relative to total variance (current + lagged). Higher values indicate
    slower price discovery.

    Formula:
        R²_restricted = R²(ret ~ benchmark_ret)
        R²_unrestricted = R²(ret ~ benchmark_ret + lag1_benchmark + ... + lagK_benchmark)
        delay = 1 - (R²_restricted / R²_unrestricted)

    Args:
        ret: Asset returns panel
        benchmark_ret: Benchmark returns panel (aligned with ret)
        window: Trailing window for regression (default 60)
        max_lags: Maximum lags of benchmark to include (default 5)

    Returns:
        Panel of delay scores in [0, 1], where higher values indicate slower
        price discovery (more variance explained by lagged benchmark)

    Notes:
        - PIT-safe: uses only trailing observations
        - Returns NaN when insufficient observations or rank deficiency
        - Backend: pandas_numpy only
    """
    ret, benchmark_ret = _align(ret, benchmark_ret)
    window = _positive_int(window, "window")
    max_lags = _positive_int(max_lags, "max_lags")

    if window <= max_lags + 2:
        raise ValueError(f"window must be > max_lags + 2 ({max_lags + 2})")

    out = np.full(ret.shape, np.nan, dtype=float)

    for col_idx in range(ret.shape[1]):
        ret_series = ret.iloc[:, col_idx].values
        bench_series = benchmark_ret.iloc[:, col_idx].values

        for row_idx in range(window - 1, ret.shape[0]):
            start = row_idx - window + 1
            r = ret_series[start : row_idx + 1]
            b = bench_series[start : row_idx + 1]

            valid = np.isfinite(r) & np.isfinite(b)
            if valid.sum() < max_lags + 5:
                continue

            r_v = r[valid]
            b_v = b[valid]

            try:
                # Restricted: ret ~ benchmark_ret (current only)
                X_restricted = np.column_stack([np.ones(len(b_v)), b_v])
                if np.linalg.matrix_rank(X_restricted) != 2:
                    continue
                beta_restricted = np.linalg.lstsq(X_restricted, r_v, rcond=None)[0]
                pred_restricted = X_restricted @ beta_restricted
                ss_res_restricted = np.sum((r_v - pred_restricted) ** 2)
                ss_tot = np.sum((r_v - np.mean(r_v)) ** 2)
                if ss_tot < _EPS:
                    continue
                r2_restricted = (1.0 - ss_res_restricted) / ss_tot if ss_tot != 0 else np.nan

                # Unrestricted: ret ~ benchmark_ret + lag1 + ... + lagK
                # Build lagged features
                lagged_features = [b_v]
                for lag in range(1, max_lags + 1):
                    lagged = np.full(len(b_v), np.nan)
                    lagged[lag:] = b_v[:-lag]
                    lagged_features.append(lagged)

                # Stack features
                X_unrestricted = np.column_stack([np.ones(len(b_v))] + lagged_features)
                # Remove rows with NaN from lagging
                valid_rows = np.all(np.isfinite(X_unrestricted), axis=1)
                if valid_rows.sum() < max_lags + 3:
                    continue

                X_u = X_unrestricted[valid_rows]
                r_u = r_v[valid_rows]

                if np.linalg.matrix_rank(X_u) != X_u.shape[1]:
                    continue

                beta_unrestricted = np.linalg.lstsq(X_u, r_u, rcond=None)[0]
                pred_unrestricted = X_u @ beta_unrestricted
                ss_res_unrestricted = np.sum((r_u - pred_unrestricted) ** 2)
                ss_tot_u = np.sum((r_u - np.mean(r_u)) ** 2)
                if ss_tot_u < _EPS:
                    continue
                r2_unrestricted = (1.0 - ss_res_unrestricted) / ss_tot_u if ss_tot_u != 0 else np.nan

                # Delay score
                if r2_unrestricted > _EPS:
                    delay = (1.0 - r2_restricted / r2_unrestricted) if r2_unrestricted != 0 else np.nan
                    # Clamp to [0, 1]
                    out[row_idx, col_idx] = np.clip(delay, 0.0, 1.0)

            except (np.linalg.LinAlgError, ValueError):
                continue

    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


# Operator wrapper classes
class _PanelDayNightBetaGap(SeriesOperator):
    metadata = OperatorMetadata(
        name="panel_day_night_beta_gap",
        category="panel_microstructure",
        description="Day minus night beta gap. Computes rolling beta during day hours "
        "vs night hours, returns beta_day - beta_night. Positive values indicate "
        "stronger day comovement with benchmark.",
        param_names=[
            "ret",
            "benchmark_ret",
            "day_start",
            "day_end",
            "night_start",
            "night_end",
            "window",
        ],
        return_type="series",
        tags=[
            "panel",
            "microstructure",
            "pit_safe",
            "causal",
            "daily",
            "typed_v2",
            "deterministic",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_panel_day_night_beta_gap(*args, **kwargs)


class _PastorStambaughBeta(SeriesOperator):
    metadata = OperatorMetadata(
        name="pastor_stambaugh_beta",
        category="panel_microstructure",
        description="Liquidity-adjusted beta (Pastor-Stambaugh). Computes beta after "
        "orthogonalizing returns with respect to liquidity proxy. Returns beta from "
        "residualized regression.",
        param_names=["ret", "benchmark_ret", "liquidity_proxy", "window"],
        return_type="series",
        tags=[
            "panel",
            "microstructure",
            "pit_safe",
            "causal",
            "daily",
            "typed_v2",
            "deterministic",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_pastor_stambaugh_beta(*args, **kwargs)


class _PriceDelayScore(SeriesOperator):
    metadata = OperatorMetadata(
        name="price_delay_score",
        category="panel_microstructure",
        description="Price discovery delay score (Hou-Moskowitz 2005). Measures fraction "
        "of variance explained by lagged benchmark returns. Returns delay score in [0, 1] "
        "where higher values indicate slower price discovery.",
        param_names=["ret", "benchmark_ret", "window", "max_lags"],
        return_type="series",
        tags=[
            "panel",
            "microstructure",
            "pit_safe",
            "causal",
            "daily",
            "typed_v2",
            "deterministic",
        ],
    )

    def _calculate_series(self, *args, **kwargs):
        return pd_price_delay_score(*args, **kwargs)


def register() -> None:
    """Register panel batch1 operators."""
    from cleaned_operators.registry import OperatorRegistry

    # Check if already registered
    if "price_delay_score" in OperatorRegistry._operators:
        return

    # Register pandas_numpy backend
    for name, cls in [
        ("panel_day_night_beta_gap", _PanelDayNightBetaGap),
        ("pastor_stambaugh_beta", _PastorStambaughBeta),
        ("price_delay_score", _PriceDelayScore),
    ]:
        register_operator(
            name=name,
            category="panel_microstructure",
            business_category="panel",
            canonical=name,
            source="panel_batch1",
            backend="pandas_numpy",
            status="experimental",
        )(cls)

    # Add to extended surface
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(
        {
            "panel_day_night_beta_gap",
            "pastor_stambaugh_beta",
            "price_delay_score",
        }
    )


# Explicit policy declaration (R47 convention)
_EXPLICIT_POLICIES = {
    "panel_day_night_beta_gap": {
        "scope": "panel_microstructure",
        "pit_safe": True,
        "min_periods": 3,
    },
    "pastor_stambaugh_beta": {
        "scope": "panel_microstructure",
        "pit_safe": True,
        "min_periods": 5,
    },
    "price_delay_score": {
        "scope": "panel_microstructure",
        "pit_safe": True,
        "min_periods": 5,
    },
}


register()
