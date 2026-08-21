"""
Granger causality tests for time series.

Granger causality assesses whether one time series helps predict another.
X Granger-causes Y if past values of X contain information that helps predict Y
beyond the information contained in past values of Y alone.

Reference:
    Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
    and Cross-spectral Methods". Econometrica. 37 (3): 424-438.
    doi:10.2307/1912791

Implementation follows:
    Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
    Chapter 11: Vector Autoregressions.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import stats
from scipy.linalg import lstsq


def _fit_ar_model(y: np.ndarray, lags: int) -> Tuple[np.ndarray, float, int]:
    """
    Fit univariate AR(p) model via OLS.

    Args:
        y: Time series (T,)
        lags: Number of lags

    Returns:
        (coefficients, rss, nobs)
        coefficients: shape (lags,)
        rss: residual sum of squares
        nobs: number of observations used
    """
    T = len(y)
    if T <= lags:
        raise ValueError(f"Time series length {T} must exceed lags {lags}")

    # Build design matrix: [y_{t-1}, y_{t-2}, ..., y_{t-p}]
    X = np.column_stack([y[lags - i - 1 : T - i - 1] for i in range(lags)])
    y_target = y[lags:]

    # OLS regression
    coef, residuals, rank, s = lstsq(X, y_target)

    # residuals is a scalar (sum of squared residuals) if rank is full, empty array otherwise
    if isinstance(residuals, np.ndarray) and len(residuals) > 0:
        rss = float(residuals[0])
    elif np.isscalar(residuals):
        rss = float(residuals)
    else:
        rss = np.sum((y_target - X @ coef) ** 2)

    nobs = len(y_target)

    return coef, rss, nobs


def _fit_var_model(y: np.ndarray, x: np.ndarray, lags: int) -> Tuple[np.ndarray, float, int]:
    """
    Fit bivariate VAR model: y_t = c + sum(a_i * y_{t-i}) + sum(b_i * x_{t-i}) + e_t

    Args:
        y: Dependent time series (T,)
        x: Independent time series (T,)
        lags: Number of lags

    Returns:
        (coefficients, rss, nobs)
        coefficients: shape (2*lags,) - [y_lags..., x_lags...]
        rss: residual sum of squares
        nobs: number of observations
    """
    T = len(y)
    if T <= lags:
        raise ValueError(f"Time series length {T} must exceed lags {lags}")
    if len(x) != T:
        raise ValueError(f"Series must have same length: {T} vs {len(x)}")

    # Build design matrix: [y_{t-1}, ..., y_{t-p}, x_{t-1}, ..., x_{t-p}]
    X_y = np.column_stack([y[lags - i - 1 : T - i - 1] for i in range(lags)])
    X_x = np.column_stack([x[lags - i - 1 : T - i - 1] for i in range(lags)])
    X = np.hstack([X_y, X_x])

    y_target = y[lags:]

    # OLS regression
    coef, residuals, rank, s = lstsq(X, y_target)

    # residuals is a scalar (sum of squared residuals) if rank is full, empty array otherwise
    if isinstance(residuals, np.ndarray) and len(residuals) > 0:
        rss = float(residuals[0])
    elif np.isscalar(residuals):
        rss = float(residuals)
    else:
        rss = np.sum((y_target - X @ coef) ** 2)

    nobs = len(y_target)

    return coef, rss, nobs


def granger_causality_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lags: int = 5,
    min_obs: int = 30,
) -> Tuple[float, float, float, int]:
    """
    Test whether x Granger-causes y using F-test.

    Null hypothesis: x does NOT Granger-cause y (coefficients on x lags are zero).

    The test compares:
    - Restricted model: y_t ~ y_{t-1}, ..., y_{t-p}
    - Unrestricted model: y_t ~ y_{t-1}, ..., y_{t-p}, x_{t-1}, ..., x_{t-p}

    Args:
        y: Dependent time series (T,)
        x: Independent time series (T,)
        max_lags: Maximum number of lags to include
        min_obs: Minimum observations required

    Returns:
        (f_stat, p_value, df_constraint, best_lag)
        f_stat: F-statistic for joint significance of x lags
        p_value: p-value under F distribution
        df_constraint: degrees of freedom for constraint (number of x lags)
        best_lag: lag order used (best by BIC if not specified)

    Raises:
        ValueError: If insufficient data or invalid inputs
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("Input series must be 1-dimensional")

    T = len(y)
    if T != len(x):
        raise ValueError(f"Series must have equal length: {T} vs {len(x)}")

    # Remove NaN/Inf
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    T = len(y)

    if T < min_obs:
        raise ValueError(f"Insufficient observations: {T} < {min_obs}")

    if max_lags >= T // 2:
        raise ValueError(f"max_lags {max_lags} too large for T={T}")

    # Select lag order by BIC
    best_lag = 1
    best_bic = np.inf

    for lag in range(1, max_lags + 1):
        if T - lag < min_obs:
            break
        try:
            _, rss_unr, nobs = _fit_var_model(y, x, lag)
            k = 2 * lag  # number of parameters in unrestricted model
            bic = nobs * np.log(rss_unr / nobs) + k * np.log(nobs)

            if bic < best_bic:
                best_bic = bic
                best_lag = lag
        except (ValueError, np.linalg.LinAlgError):
            continue

    # Fit models at best lag
    _, rss_restricted, nobs = _fit_ar_model(y, best_lag)
    _, rss_unrestricted, nobs = _fit_var_model(y, x, best_lag)

    # F-statistic: ((RSS_r - RSS_ur) / q) / (RSS_ur / (n - k))
    # q = number of restrictions (x lags)
    # k = total parameters in unrestricted model
    q = best_lag
    k = 2 * best_lag

    numerator = (rss_restricted - rss_unrestricted) / q
    denominator = rss_unrestricted / (nobs - k)

    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan, np.nan, q, best_lag

    f_stat = numerator / denominator

    # p-value from F(q, n-k) distribution
    df1 = q
    df2 = nobs - k
    p_value = 1.0 - stats.f.cdf(f_stat, df1, df2)

    return f_stat, p_value, q, best_lag


def pairwise_granger_causality(
    data: np.ndarray,
    max_lags: int = 5,
    min_obs: int = 30,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise Granger causality matrix for multiple time series.

    Args:
        data: Time series data (T, N) where N is number of series
        max_lags: Maximum lags for Granger test
        min_obs: Minimum observations required
        alpha: Significance level for rejection

    Returns:
        (f_matrix, p_matrix, sig_matrix)
        f_matrix: F-statistics (N, N) where [i,j] tests j -> i
        p_matrix: p-values (N, N)
        sig_matrix: Significance indicators (N, N) - 1 if significant

    Note:
        Diagonal elements are NaN (series cannot cause itself).
        Element [i,j] tests whether series j Granger-causes series i.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Data must be 2D (T, N)")

    T, N = data.shape

    f_matrix = np.full((N, N), np.nan, dtype=np.float64)
    p_matrix = np.full((N, N), np.nan, dtype=np.float64)
    sig_matrix = np.zeros((N, N), dtype=np.int32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # Skip diagonal

            try:
                f_stat, p_value, _, _ = granger_causality_test(
                    data[:, i], data[:, j], max_lags=max_lags, min_obs=min_obs
                )
                f_matrix[i, j] = f_stat
                p_matrix[i, j] = p_value

                if np.isfinite(p_value) and p_value < alpha:
                    sig_matrix[i, j] = 1

            except (ValueError, np.linalg.LinAlgError):
                # Leave as NaN if test fails
                pass

    return f_matrix, p_matrix, sig_matrix
