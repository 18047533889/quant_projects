"""
Johansen cointegration test for multivariate time series.

Cointegration tests whether non-stationary time series share common stochastic
trends such that a linear combination is stationary. This is fundamental for
pairs trading and mean-reversion strategies.

References:
    Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration
    Vectors in Gaussian Vector Autoregressive Models". Econometrica. 59 (6):
    1551-1580. doi:10.2307/2938278

    Johansen, S. (1995). Likelihood-Based Inference in Cointegrated Vector
    Autoregressive Models. Oxford University Press.

Critical values from:
    Osterwald-Lenum, M. (1992). "A Note with Quantiles of the Asymptotic
    Distribution of the Maximum Likelihood Cointegration Rank Test Statistics".
    Oxford Bulletin of Economics and Statistics. 54 (3): 461-472.
"""

from typing import Tuple, Optional, Literal
import numpy as np
from scipy import stats


# Critical values for trace statistic (no trend, constant in cointegration space)
# Source: Osterwald-Lenum (1992), Table 1
# Rows: r (number of cointegrating vectors), Columns: significance level
_TRACE_CRITICAL_VALUES = {
    # r=0
    0: {0.10: 7.52, 0.05: 9.24, 0.01: 12.97},
    # r≤1
    1: {0.10: 17.85, 0.05: 19.96, 0.01: 24.60},
    # r≤2
    2: {0.10: 32.00, 0.05: 34.91, 0.01: 41.07},
    # r≤3
    3: {0.10: 49.65, 0.05: 53.12, 0.01: 60.16},
    # r≤4
    4: {0.10: 71.86, 0.05: 76.07, 0.01: 84.45},
}

# Critical values for maximum eigenvalue statistic
_MAXEIG_CRITICAL_VALUES = {
    # r=0
    0: {0.10: 7.52, 0.05: 9.24, 0.01: 12.97},
    # r=1
    1: {0.10: 13.75, 0.05: 15.67, 0.01: 20.20},
    # r=2
    2: {0.10: 19.77, 0.05: 22.00, 0.01: 26.81},
    # r=3
    3: {0.10: 25.56, 0.05: 28.14, 0.01: 33.24},
    # r=4
    4: {0.10: 31.66, 0.05: 34.40, 0.01: 39.79},
}


def _lag_matrix(data: np.ndarray, lags: int) -> np.ndarray:
    """
    Create lagged matrix for VAR estimation.

    Args:
        data: Time series (T, K)
        lags: Number of lags

    Returns:
        Lagged matrix (T-lags, K*lags) with column ordering:
        [lag1_s0, lag2_s0, ..., lagP_s0, lag1_s1, lag2_s1, ..., lagP_s1, ...]
    """
    T, K = data.shape
    lagged = []
    # For each series
    for k in range(K):
        # For each lag
        for lag in range(1, lags + 1):
            lagged.append(data[lags - lag : T - lag, k : k + 1])
    return np.hstack(lagged)


def _residuals_from_deterministics(
    data: np.ndarray, det_order: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regress out deterministic components.

    Args:
        data: Time series (T, K)
        det_order: -1 (no deterministics), 0 (constant), 1 (constant + trend)

    Returns:
        (residuals, fitted_deterministics) both (T, K)
    """
    T, K = data.shape

    if det_order == -1:
        return data.copy(), np.zeros_like(data)

    # Build deterministic regressor matrix
    X_det = []
    if det_order >= 0:
        X_det.append(np.ones(T))  # Constant
    if det_order >= 1:
        X_det.append(np.arange(T, dtype=np.float64))  # Linear trend

    X_det = np.column_stack(X_det) if X_det else np.zeros((T, 0))

    # Regress each series on deterministics
    residuals = np.zeros_like(data)
    fitted = np.zeros_like(data)

    for k in range(K):
        if X_det.shape[1] > 0:
            coef = np.linalg.lstsq(X_det, data[:, k], rcond=None)[0]
            fitted[:, k] = X_det @ coef
            residuals[:, k] = data[:, k] - fitted[:, k]
        else:
            residuals[:, k] = data[:, k]

    return residuals, fitted


def johansen_test(
    data: np.ndarray,
    det_order: int = 0,
    lags: int = 1,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Johansen cointegration test via maximum likelihood.

    The test estimates eigenvalues from the canonical correlation between
    first differences and lagged levels. Eigenvalues correspond to squared
    canonical correlations.

    Args:
        data: Time series (T, K) where K is number of series
        det_order: Deterministic component specification:
            -1: no deterministic components
             0: constant in cointegration space (default)
             1: constant and linear trend in cointegration space
        lags: Number of lags in VAR specification (default 1)
        alpha: Significance level (0.01, 0.05, or 0.10)

    Returns:
        (trace_stats, max_eig_stats, rank, eigenvalues)
        trace_stats: Trace test statistics for r=0,1,...,K-1
        max_eig_stats: Maximum eigenvalue statistics for r=0,1,...,K-1
        rank: Estimated cointegration rank (number of cointegrating vectors)
        eigenvalues: Sorted eigenvalues (K,) in descending order

    Raises:
        ValueError: If invalid inputs or insufficient data
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError("Data must be 2D (T, K)")

    T, K = data.shape

    if T < 2 * lags + K + 10:
        raise ValueError(f"Insufficient observations: T={T} too small for K={K}, lags={lags}")

    if det_order not in (-1, 0, 1):
        raise ValueError(f"det_order must be -1, 0, or 1, got {det_order}")

    if alpha not in (0.01, 0.05, 0.10):
        raise ValueError(f"alpha must be 0.01, 0.05, or 0.10, got {alpha}")

    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")

    if isinstance(lags, (bool, np.bool_)) or not isinstance(lags, (int, np.integer)):
        raise ValueError("lags must be an integer VAR-level lag")
    if isinstance(det_order, (bool, np.bool_)):
        raise ValueError("det_order must not be boolean")
    if not 1 <= K <= 12 or not np.isfinite(data).all():
        raise ValueError("unsupported dimension or incomplete joint calendar")
    if K == 1 and det_order >= 0:
        raise ValueError("unsupported Johansen K=1 deterministic domain in installed reference implementation")
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    try:
        result = coint_johansen(data, det_order, lags - 1)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular Johansen design") from exc
    if (not np.isfinite(result.eig).all() or np.any(result.eig < -1e-10)
            or np.any(result.eig >= 1) or not np.isfinite(result.lr1).all()):
        raise ValueError("numerically invalid Johansen result")
    column = {0.10: 0, 0.05: 1, 0.01: 2}[alpha]
    rank = 0
    for statistic, critical in zip(result.lr1, result.cvt[:, column]):
        if not np.isfinite(critical) or critical <= 0:
            raise ValueError("unsupported Johansen critical distribution")
        if statistic <= critical:
            break
        rank += 1
    return result.lr1.copy(), result.lr2.copy(), rank, result.eig.copy()

def engle_granger_test(
    y: np.ndarray,
    x: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, bool, np.ndarray]:
    """
    Engle-Granger two-step cointegration test for bivariate series.

    This is a simpler alternative to Johansen for two series. Tests whether
    the residuals from y ~ x are stationary using augmented Dickey-Fuller test.

    Reference:
        Engle, R.F.; Granger, C.W.J. (1987). "Co-integration and Error Correction:
        Representation, Estimation, and Testing". Econometrica. 55 (2): 251-276.

    Args:
        y: Dependent series (T,)
        x: Independent series (T,)
        alpha: Significance level

    Returns:
        (adf_stat, p_value, is_cointegrated, residuals)
        adf_stat: Augmented Dickey-Fuller statistic on residuals
        p_value: Approximate p-value
        is_cointegrated: True if reject null of no cointegration
        residuals: Residuals from cointegration regression (T,)

    Note:
        This is a simplified implementation using critical values for ADF test.
        For production, consider using statsmodels.tsa.stattools.coint.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    if len(y) != len(x):
        raise ValueError(f"Series must have equal length: {len(y)} vs {len(x)}")

    T = len(y)
    if T < 30:
        raise ValueError(f"Insufficient data: T={T} < 30")

    # Step 1: Cointegrating regression (OLS)
    X = np.column_stack([np.ones(T), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coef

    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite in (0, 1)")
    if not np.isfinite(y).all() or not np.isfinite(x).all() or np.linalg.matrix_rank(X) < 2:
        raise ValueError("Engle-Granger requires finite nondegenerate joint observations")
    from statsmodels.tsa.stattools import coint
    adf_stat, p_value, _ = coint(y, x, trend="c", autolag="aic")
    if not np.isfinite(adf_stat) or not np.isfinite(p_value):
        raise ValueError("degenerate Engle-Granger inference")
    return float(adf_stat), float(p_value), bool(p_value < alpha), residuals
