"""Shared convex solvers for first- and second-difference L1 filters."""
from __future__ import annotations

import numpy as np


class TrendFilterConvergenceError(RuntimeError):
    """The bounded convex solver did not establish its stopping certificate."""


def _difference_transpose(values: np.ndarray, order: int) -> np.ndarray:
    kernel = np.array([-1.0, 1.0]) if order == 1 else np.array([1.0, -2.0, 1.0])
    return np.convolve(values, kernel, mode="full")


def _system_bands(size: int, order: int, rho: float) -> np.ndarray:
    if order == 1:
        diagonal = np.full(size, 1.0 + 2.0 * rho)
        diagonal[[0, -1]] = 1.0 + rho
        bands = np.zeros((2, size))
        bands[0, 1:] = -rho
        bands[1] = diagonal
        return bands
    if size == 3:
        diagonal = np.array([1.0 + rho, 1.0 + 4.0 * rho, 1.0 + rho])
        bands = np.zeros((3, size))
        bands[0, 2] = rho
        bands[1, 1:] = -2.0 * rho
        bands[2] = diagonal
        return bands
    diagonal = np.full(size, 1.0 + 6.0 * rho)
    diagonal[[0, -1]] = 1.0 + rho
    diagonal[[1, -2]] = 1.0 + 5.0 * rho
    bands = np.zeros((3, size))
    bands[0, 2:] = rho
    bands[1, 1:] = -4.0 * rho
    bands[1, [1, -1]] = -2.0 * rho
    bands[2] = diagonal
    return bands


def _residual_within(
    residual: np.ndarray,
    references: tuple[np.ndarray, ...],
    absolute_term: float,
    relative_tolerance: float,
) -> bool:
    """Compare norms after common scaling, without overflow certification."""
    arrays = (residual,) + references
    if not all(np.isfinite(array).all() for array in arrays):
        return False
    scale = max(1.0, *(float(np.max(np.abs(array), initial=0.0)) for array in arrays))
    residual_norm = np.linalg.norm(residual / scale)
    reference_norm = max(np.linalg.norm(array / scale) for array in references)
    return residual_norm <= absolute_term / scale + relative_tolerance * reference_norm


def difference_l1_filter(
    x: np.ndarray,
    penalty: float,
    *,
    order: int,
    max_iter: int = 4000,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-7,
) -> np.ndarray:
    """Minimize ``.5*||y-x||² + penalty*||D_order y||₁`` by ADMM.

    The banded positive-definite system keeps each iteration linear in the
    series length.  A result is returned only after primal and dual residuals
    satisfy the standard absolute/relative stopping certificate.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("trend-filter input must be a finite one-dimensional array")
    if order not in (1, 2):
        raise ValueError("difference order must be 1 or 2")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("trend-filter penalty must be finite and nonnegative")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    size = values.size
    difference_size = size - order
    if penalty == 0 or difference_size <= 0:
        return values.copy()

    from scipy.linalg import cholesky_banded, cho_solve_banded

    # Scaling rho with the penalty avoids very slow dual progress for strongly
    # regularized signals while keeping the same objective and solution.
    rho = max(1.0, min(float(penalty) * np.sqrt(size), 1e8))
    bands = _system_bands(size, order, rho)
    factor = cholesky_banded(bands, lower=False, check_finite=False)
    y = values.copy()
    z = np.diff(y, n=order)
    dual = np.zeros(difference_size)
    threshold = penalty / rho
    for _ in range(max_iter):
        rhs = values + rho * _difference_transpose(z - dual, order)
        if not np.isfinite(rhs).all():
            raise TrendFilterConvergenceError("trend-filter iteration became nonfinite")
        y = cho_solve_banded((factor, False), rhs, check_finite=False)
        dy = np.diff(y, n=order)
        z_previous = z
        shifted = dy + dual
        z = np.sign(shifted) * np.maximum(np.abs(shifted) - threshold, 0.0)
        primal = dy - z
        dual += primal
        dual_residual = rho * _difference_transpose(z - z_previous, order)
        dual_reference = rho * _difference_transpose(dual, order)
        if (_residual_within(
                primal, (dy, z), np.sqrt(difference_size) * absolute_tolerance,
                relative_tolerance,
            ) and _residual_within(
                dual_residual, (dual_reference,), np.sqrt(size) * absolute_tolerance,
                relative_tolerance,
            )):
            if not np.isfinite(y).all():
                raise TrendFilterConvergenceError("trend-filter solution is nonfinite")
            return y
    raise TrendFilterConvergenceError(
        f"trend-filter solver did not converge within {max_iter} iterations"
    )


def total_variation_filter(x: np.ndarray, penalty: float, *, max_iter: int = 4000) -> np.ndarray:
    return difference_l1_filter(x, penalty, order=1, max_iter=max_iter)


def l1_trend_filter(x: np.ndarray, penalty: float, *, max_iter: int = 4000) -> np.ndarray:
    return difference_l1_filter(x, penalty, order=2, max_iter=max_iter)
