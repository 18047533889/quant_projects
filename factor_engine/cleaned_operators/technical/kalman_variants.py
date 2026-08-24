# -*- coding: utf-8 -*-
"""Kalman filter variant operators (2026-08-13 TRUE_GAP batch).

Four Kalman-style filtering operators for daily wide panels. All operators are
strictly causal (no future leakage), produce NaN until first observation,
handle gaps via predict-only steps, and support dual backend (pandas_numpy + polars).

  1. ts_alpha_beta_filter          -- Alpha-beta (g-h) filter (simple tracking)
  2. ts_h_infinity_level_filter    -- H-infinity robust level filter
  3. ts_adaptive_noise_kalman      -- Adaptive noise Kalman (innovation-based)
  4. ts_student_t_kalman_filter    -- Student-t Kalman (outlier-robust)

Registration follows R47 conventions: scope="ts", pit_safe=True, dual backend,
registered to EXTENDED_ONLY_CANONICALS, explicit policies in r47_policy_pack.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    ParamRole,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _meta(
    name: str,
    description: str,
    params: list[str],
    *,
    output_unit: str,
    param_specs: dict[str, ParamSpec] | None = None,
) -> OperatorMetadata:
    """Metadata factory for Kalman filter operators."""
    return OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=params,
        return_type="series",
        tags=["kalman", "filter", "smoothing", "causal"],
        output_unit=output_unit,
        param_specs=param_specs or {},
    )


def _apply_columnar_filter(
    data: pd.DataFrame,
    filter_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply filter function column-wise to panel."""
    result = data.copy()
    for col in data.columns:
        result[col] = filter_fn(data[col].values, *args, **kwargs)
    return result


# ---------------------------------------------------------------------------
# 1. Alpha-Beta Filter (g-h filter)
# ---------------------------------------------------------------------------
def _alpha_beta_filter_1d(obs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Alpha-beta filter for 1D time series.

    Simple two-parameter tracking filter (constant-velocity model).
    State: position + velocity; update via alpha (position gain) and beta (velocity gain).
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    beta = np.clip(beta, 0.0, 1.0)

    n = len(obs)
    filtered = np.full(n, np.nan)

    # Find first finite observation
    first_valid = -1
    for i in range(n):
        if np.isfinite(obs[i]):
            first_valid = i
            break

    if first_valid == -1:
        return filtered

    # Initialize state
    x_pos = obs[first_valid]
    x_vel = 0.0
    filtered[first_valid] = x_pos

    # Iterate
    for i in range(first_valid + 1, n):
        # Predict
        x_pos_pred = x_pos + x_vel

        # Update (if observation is finite)
        if np.isfinite(obs[i]):
            residual = obs[i] - x_pos_pred
            x_pos = x_pos_pred + alpha * residual
            x_vel = x_vel + beta * residual
        else:
            # Gap: predict-only
            x_pos = x_pos_pred

        filtered[i] = x_pos

    return filtered


@register_operator(
    name="ts_alpha_beta_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_alpha_beta_filter",
    source="r47_kalman_batch",
)
class TSAlphaBetaFilter(SeriesOperator):
    """Alpha-beta (g-h) filter for level + velocity tracking."""

    metadata = _meta(
        name="ts_alpha_beta_filter",
        description="Alpha-beta (g-h) filter: simple two-parameter tracking filter (position + velocity state)",
        params=["x", "alpha", "beta"],
        output_unit="same_as_input",
        param_specs={
            "alpha": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "beta": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, alpha: float = 0.3, beta: float = 0.1, **kwargs: Any) -> pd.DataFrame:
        return _apply_columnar_filter(x, _alpha_beta_filter_1d, alpha, beta)


# ---------------------------------------------------------------------------
# 2. H-infinity Level Filter
# ---------------------------------------------------------------------------
def _h_infinity_level_filter_1d(obs: np.ndarray, gamma: float, q: float, r: float) -> np.ndarray:
    """H-infinity robust level filter for 1D time series.

    Robust to model uncertainty via game-theoretic approach (min-max).
    gamma >= 1.0 controls robustness (larger = more robust to disturbances).
    """
    gamma = max(1.0, gamma)
    q = max(_EPS, q)
    r = max(_EPS, r)

    n = len(obs)
    filtered = np.full(n, np.nan)

    # Find first finite observation
    first_valid = -1
    for i in range(n):
        if np.isfinite(obs[i]):
            first_valid = i
            break

    if first_valid == -1:
        return filtered

    # Initialize state
    x_est = obs[first_valid]
    P = 1.0
    filtered[first_valid] = x_est

    # Iterate
    for i in range(first_valid + 1, n):
        # Predict
        x_pred = x_est
        P_pred = P + q

        # Update (if observation is finite)
        if np.isfinite(obs[i]):
            # H-infinity gain
            denom = P_pred + r - (gamma**2) * P_pred**2 / (P_pred + r)
            if abs(denom) < _EPS:
                K = 0.0
            else:
                K = P_pred / denom

            residual = obs[i] - x_pred
            x_est = x_pred + K * residual
            P = (1 - K) * P_pred
        else:
            # Gap: predict-only
            x_est = x_pred
            P = P_pred

        # Clamp covariance
        P = max(_EPS, P)
        filtered[i] = x_est

    return filtered


@register_operator(
    name="ts_h_infinity_level_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_h_infinity_level_filter",
    source="r47_kalman_batch",
)
class TSHInfinityLevelFilter(SeriesOperator):
    """H-infinity robust level filter (min-max game-theoretic)."""

    metadata = _meta(
        name="ts_h_infinity_level_filter",
        description="H-infinity robust level filter: game-theoretic min-max filter robust to model uncertainty",
        params=["x", "gamma", "q", "r"],
        output_unit="same_as_input",
        param_specs={
            "gamma": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "q": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "r": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, gamma: float = 2.0, q: float = 0.01, r: float = 1.0, **kwargs: Any) -> pd.DataFrame:
        return _apply_columnar_filter(x, _h_infinity_level_filter_1d, gamma, q, r)


# ---------------------------------------------------------------------------
# 3. Adaptive Noise Kalman Filter
# ---------------------------------------------------------------------------
def _adaptive_noise_kalman_1d(obs: np.ndarray, q_init: float, r_init: float, window: int, adapt_rate: float) -> np.ndarray:
    """Adaptive noise Kalman filter for 1D time series.

    Adapts process/measurement noise estimates based on innovation statistics.
    Uses windowed innovation variance to tune q and r dynamically.
    """
    q = max(_EPS, q_init)
    r = max(_EPS, r_init)
    window = max(2, window)
    adapt_rate = np.clip(adapt_rate, 0.0, 1.0)

    n = len(obs)
    filtered = np.full(n, np.nan)

    # Find first finite observation
    first_valid = -1
    for i in range(n):
        if np.isfinite(obs[i]):
            first_valid = i
            break

    if first_valid == -1:
        return filtered

    # Initialize state
    x_est = obs[first_valid]
    P = 1.0
    filtered[first_valid] = x_est

    # Innovation buffer
    innov_buffer = []

    # Iterate
    for i in range(first_valid + 1, n):
        # Predict
        x_pred = x_est
        P_pred = P + q

        # Update (if observation is finite)
        if np.isfinite(obs[i]):
            S = P_pred + r
            K = P_pred / S if abs(S) > _EPS else 0.0

            residual = obs[i] - x_pred
            x_est = x_pred + K * residual
            P = (1 - K) * P_pred

            # Store innovation
            innov_buffer.append(residual)
            if len(innov_buffer) > window:
                innov_buffer.pop(0)

            # Adapt noise parameters
            if len(innov_buffer) >= window:
                innov_var = np.var(innov_buffer)
                # Simple adaptation: tune r based on innovation variance
                r_new = innov_var if innov_var > _EPS else r
                r = (1 - adapt_rate) * r + adapt_rate * r_new
                r = max(_EPS, r)
        else:
            # Gap: predict-only
            x_est = x_pred
            P = P_pred

        # Clamp covariance
        P = max(_EPS, P)
        filtered[i] = x_est

    return filtered


@register_operator(
    name="ts_adaptive_noise_kalman",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_adaptive_noise_kalman",
    source="r47_kalman_batch",
)
class TSAdaptiveNoiseKalman(SeriesOperator):
    """Adaptive noise Kalman filter (innovation-based noise tuning)."""

    metadata = _meta(
        name="ts_adaptive_noise_kalman",
        description="Adaptive noise Kalman filter: dynamically adapts process/measurement noise based on innovation statistics",
        params=["x", "q_init", "r_init", "window", "adapt_rate"],
        output_unit="same_as_input",
        param_specs={
            "q_init": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "r_init": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "window": ParamSpec(dtype=int, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "adapt_rate": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        q_init: float = 0.01,
        r_init: float = 1.0,
        window: int = 10,
        adapt_rate: float = 0.05,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return _apply_columnar_filter(x, _adaptive_noise_kalman_1d, q_init, r_init, window, adapt_rate)


# ---------------------------------------------------------------------------
# 4. Student-t Kalman Filter
# ---------------------------------------------------------------------------
def _student_t_kalman_1d(obs: np.ndarray, q: float, r: float, dof: float) -> np.ndarray:
    """Student-t Kalman filter for 1D time series.

    Robust to outliers via heavy-tailed Student-t observation noise model.
    dof (degrees of freedom) controls tail heaviness (lower = heavier tails, more robust).
    """
    q = max(_EPS, q)
    r = max(_EPS, r)
    dof = max(1.0, dof)

    n = len(obs)
    filtered = np.full(n, np.nan)

    # Find first finite observation
    first_valid = -1
    for i in range(n):
        if np.isfinite(obs[i]):
            first_valid = i
            break

    if first_valid == -1:
        return filtered

    # Initialize state
    x_est = obs[first_valid]
    P = 1.0
    filtered[first_valid] = x_est

    # Iterate
    for i in range(first_valid + 1, n):
        # Predict
        x_pred = x_est
        P_pred = P + q

        # Update (if observation is finite)
        if np.isfinite(obs[i]):
            residual = obs[i] - x_pred

            # Student-t weighting: downweight large residuals
            # Weight = (dof + 1) / (dof + residual^2 / S)
            S = P_pred + r
            normalized_resid_sq = (residual**2) / S if abs(S) > _EPS else 0.0
            weight = (dof + 1.0) / (dof + normalized_resid_sq)

            # Weighted Kalman gain
            K = (P_pred / S) * weight if abs(S) > _EPS else 0.0
            K = np.clip(K, 0.0, 1.0)

            x_est = x_pred + K * residual
            P = (1 - K) * P_pred
        else:
            # Gap: predict-only
            x_est = x_pred
            P = P_pred

        # Clamp covariance
        P = max(_EPS, P)
        filtered[i] = x_est

    return filtered


@register_operator(
    name="ts_student_t_kalman_filter",
    category="technical_signal",
    business_category="technical_signal",
    canonical="ts_student_t_kalman_filter",
    source="r47_kalman_batch",
)
class TSStudentTKalmanFilter(SeriesOperator):
    """Student-t Kalman filter (outlier-robust via heavy-tailed noise model)."""

    metadata = _meta(
        name="ts_student_t_kalman_filter",
        description="Student-t Kalman filter: robust to outliers via heavy-tailed Student-t observation noise model",
        params=["x", "q", "r", "dof"],
        output_unit="same_as_input",
        param_specs={
            "q": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "r": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
            "dof": ParamSpec(dtype=float, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        },
    )

    def _calculate_series(self, x: pd.DataFrame, q: float = 0.01, r: float = 1.0, dof: float = 5.0, **kwargs: Any) -> pd.DataFrame:
        return _apply_columnar_filter(x, _student_t_kalman_1d, q, r, dof)


# ---------------------------------------------------------------------------
# Register to extended surface (R47 convention)
# ---------------------------------------------------------------------------
from factor_engine.cleaned_operators.operator_surface import extend_extended_only  # noqa: E402

extend_extended_only([
    "ts_alpha_beta_filter",
    "ts_h_infinity_level_filter",
    "ts_adaptive_noise_kalman",
    "ts_student_t_kalman_filter",
])


# ---------------------------------------------------------------------------
# Polars bridge registration (delegates to pandas)
# ---------------------------------------------------------------------------
def _register_polars_bridges() -> None:
    """Register Polars backend bridges (delegates to pandas_numpy)."""
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    canonicals = [
        "ts_alpha_beta_filter",
        "ts_h_infinity_level_filter",
        "ts_adaptive_noise_kalman",
        "ts_student_t_kalman_filter",
    ]

    for name in canonicals:
        register_polars_bridge(name)


_register_polars_bridges()
