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

    # Remove deterministic components if needed
    if det_order >= 0:
        data_adj, _ = _residuals_from_deterministics(data, det_order)
    else:
        data_adj = data.copy()

    # Construct differenced and lagged level matrices
    # Δy_t = y_t - y_{t-1}
    dy = np.diff(data_adj, axis=0)  # (T-1, K)

    # Lagged levels: y_{t-1}
    y_lag = data_adj[:-1, :]  # (T-1, K)

    # If lags > 1, include lagged differences as regressors
    if lags > 1:
        dy_lags = _lag_matrix(dy, lags - 1)  # (T-lags, K*(lags-1))
        # Align all matrices
        dy = dy[lags - 1 :, :]  # (T-lags, K)
        y_lag = y_lag[lags - 1 :, :]  # (T-lags, K)
        X_short = dy_lags  # (T-lags, K*(lags-1))
    else:
        X_short = None

    T_eff = dy.shape[0]

    # Regress dy and y_lag on short-run dynamics (if any) to get residuals
    if X_short is not None:
        # Residuals from dy ~ X_short
        R0 = dy - X_short @ np.linalg.lstsq(X_short, dy, rcond=None)[0]
        # Residuals from y_lag ~ X_short
        R1 = y_lag - X_short @ np.linalg.lstsq(X_short, y_lag, rcond=None)[0]
    else:
        R0 = dy
        R1 = y_lag

    # Compute moment matrices
    S00 = (R0.T @ R0) / T_eff
    S11 = (R1.T @ R1) / T_eff
    S01 = (R0.T @ R1) / T_eff
    S10 = S01.T

    # Solve generalized eigenvalue problem: S10 @ S00^{-1} @ S01 @ v = λ S11 @ v
    try:
        S00_inv = np.linalg.inv(S00)
        S11_inv = np.linalg.inv(S11)
    except np.linalg.LinAlgError:
        raise ValueError("Singular moment matrix; check for perfect collinearity")

    # Equivalent: solve eigenvalue problem for S11^{-1} @ S10 @ S00^{-1} @ S01
    M = S11_inv @ S10 @ S00_inv @ S01

    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvalues = np.real(eigenvalues)  # Should be real

    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]

    # Clip to [0, 1) range (theoretical bound)
    eigenvalues = np.clip(eigenvalues, 0, 1 - 1e-10)

    # Compute test statistics
    # Trace statistic: -T * sum_{i=r}^{K-1} ln(1 - λ_i)
    # Max eigenvalue statistic: -T * ln(1 - λ_r)

    trace_stats = np.zeros(K, dtype=np.float64)
    max_eig_stats = np.zeros(K, dtype=np.float64)

    for r in range(K):
        # Trace: sum from r to K-1
        trace_stats[r] = -T_eff * np.sum(np.log(1 - eigenvalues[r:]))
        # Max eigenvalue: single eigenvalue at r
        max_eig_stats[r] = -T_eff * np.log(1 - eigenvalues[r])

    # Determine rank by comparing trace statistic to critical values
    rank = 0
    for r in range(K):
        if r >= len(_TRACE_CRITICAL_VALUES):
            break  # No critical values for r > 4

        crit_val = _TRACE_CRITICAL_VALUES[r].get(alpha, np.inf)
        if trace_stats[r] > crit_val:
            rank = r + 1
        else:
            break  # Stop at first non-rejection

    return trace_stats, max_eig_stats, rank, eigenvalues


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

    # Step 2: ADF test on residuals (no constant, no trend)
    # Δe_t = ρ e_{t-1} + ε_t
    # Under null: ρ = 0 (unit root, no cointegration)
    de = np.diff(residuals)
    e_lag = residuals[:-1]

    # OLS: de ~ e_lag (no intercept for residuals)
    rho = np.sum(de * e_lag) / np.sum(e_lag**2)
    residual_adf = de - rho * e_lag
    sigma = np.std(residual_adf, ddof=1)
    se_rho = sigma / np.sqrt(np.sum(e_lag**2))

    adf_stat = rho / se_rho

    # Critical values for Engle-Granger test (more negative than standard ADF)
    # Approximate values for T > 100, 2 variables
    critical_values = {
        0.01: -3.90,
        0.05: -3.34,
        0.10: -3.04,
    }

    crit_val = critical_values.get(alpha, -3.34)
    is_cointegrated = adf_stat < crit_val

    # Approximate p-value (rough interpolation)
    if adf_stat < -3.90:
        p_value = 0.01
    elif adf_stat < -3.34:
        p_value = 0.05
    elif adf_stat < -3.04:
        p_value = 0.10
    else:
        p_value = 0.20

    return adf_stat, p_value, is_cointegrated, residuals
