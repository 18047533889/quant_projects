"""
Exposure decomposition and soft neutralization for modeling.

Provides interfaces for exposure-based transformations that can be
implemented directly or delegated to factor_preprocess.
"""
from typing import Optional, List
import logging
import numpy as np

from modeling.errors import InsufficientDataError, ContractViolation

logger = logging.getLogger(__name__)


def compute_exposure_residual(
    factor_values: np.ndarray,
    exposures: np.ndarray,
    method: str = "ols",
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute residuals after regressing factors on exposures.

    Args:
        factor_values: Factor values (T, N) where T=time, N=assets
        exposures: Exposure matrix (T, N, K) where K=number of exposures
        method: Regression method ('ols', 'ridge', 'weighted_ols')
        weights: Optional weights (T, N) for weighted regression

    Returns:
        Residuals with shape (T, N)

    Note:
        This is a minimal reference. For production, use factor_preprocess
        neutralization via adapter.
    """
    if factor_values.shape[0] != exposures.shape[0]:
        raise ContractViolation("factor_values and exposures must have same time dimension")

    if factor_values.shape[1] != exposures.shape[1]:
        raise ContractViolation("factor_values and exposures must have same asset dimension")

    if exposures.ndim != 3:
        raise ContractViolation("exposures must be 3D: (T, N, K)")

    T, N, K = exposures.shape
    residuals = np.full_like(factor_values, np.nan, dtype=np.float64)

    # Process each time slice independently (cross-sectional regression)
    for t in range(T):
        factor_t = factor_values[t, :]
        exposure_t = exposures[t, :, :]  # (N, K)
        weights_t = weights[t, :] if weights is not None else None

        residuals[t, :] = _cross_sectional_residual(
            factor_t, exposure_t, method=method, weights=weights_t
        )

    return residuals


def _cross_sectional_residual(
    y: np.ndarray,
    X: np.ndarray,
    method: str,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute cross-sectional residual for one time slice.

    Args:
        y: Factor values (N,)
        X: Exposure matrix (N, K)
        method: Regression method
        weights: Optional weights (N,)

    Returns:
        Residuals (N,)
    """
    # Handle missing values
    valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
    if weights is not None:
        valid_mask &= ~np.isnan(weights)

    if np.sum(valid_mask) < X.shape[1] + 1:
        # Insufficient data for regression
        return y

    y_valid = y[valid_mask]
    X_valid = X[valid_mask, :]

    if method == "ols":
        # Simple OLS: beta = (X'X)^-1 X'y
        try:
            result = np.linalg.lstsq(X_valid, y_valid, rcond=None)
            beta = result[0]
            rank = result[2]
            # Check if matrix is rank-deficient
            if rank < X_valid.shape[1]:
                logger.warning(
                    f"Exposure neutralization failed: matrix is rank-deficient "
                    f"(rank={rank}, expected={X_valid.shape[1]}). "
                    f"Returning NaN to signal computation failure."
                )
                return np.full_like(y, np.nan)
        except np.linalg.LinAlgError as e:
            logger.warning(
                f"Exposure neutralization failed due to singular matrix: {e}. "
                f"Returning NaN to signal computation failure."
            )
            return np.full_like(y, np.nan)
    elif method == "weighted_ols" and weights is not None:
        # Weighted OLS: beta = (X'WX)^-1 X'Wy
        w_valid = weights[valid_mask]
        W = np.diag(w_valid)
        try:
            XtWX = X_valid.T @ W @ X_valid
            XtWy = X_valid.T @ W @ y_valid
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError as e:
            logger.warning(
                f"Weighted OLS neutralization failed due to singular matrix: {e}. "
                f"Returning NaN to signal computation failure."
            )
            return np.full_like(y, np.nan)
    elif method == "ridge":
        # Ridge regression with small lambda
        lambda_ridge = 0.01
        try:
            XtX = X_valid.T @ X_valid
            Xty = X_valid.T @ y_valid
            beta = np.linalg.solve(XtX + lambda_ridge * np.eye(X_valid.shape[1]), Xty)
        except np.linalg.LinAlgError as e:
            logger.warning(
                f"Ridge neutralization failed due to singular matrix: {e}. "
                f"Returning NaN to signal computation failure."
            )
            return np.full_like(y, np.nan)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute residuals
    residual = np.full_like(y, np.nan, dtype=np.float64)
    residual[valid_mask] = y_valid - X_valid @ beta
    residual[~valid_mask] = y[~valid_mask]  # Keep original NaN

    return residual


def soft_neutralization(
    factor_values: np.ndarray,
    exposures: np.ndarray,
    alpha: float = 0.5,
    method: str = "ols",
) -> np.ndarray:
    """
    Soft neutralization: blend original and residual.

    Args:
        factor_values: Factor values (T, N)
        exposures: Exposure matrix (T, N, K)
        alpha: Blending weight, 0=original, 1=fully neutralized
        method: Regression method

    Returns:
        Softly neutralized factors (T, N)
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    residuals = compute_exposure_residual(factor_values, exposures, method=method)

    # Blend: (1-alpha)*original + alpha*residual
    return (1 - alpha) * factor_values + alpha * residuals
