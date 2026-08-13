# -*- coding: utf-8 -*-
"""Intraday topology and manifold operators (2026-08-13).

Four TRUE_GAP topological/dynamical feature extractors for minute-frequency data:
1. Matrix Profile: time-series motif/discord discovery via all-pairs distance
2. DMD Koopman: dynamic mode decomposition eigen-features
3. Covariance Manifold Shift: Riemannian distance on SPD manifold
4. Critical Transition Score: early-warning signal composite

Contract
--------
* Causal: each day's scalar uses only that day's own minute data (never future).
* Missing-value policy: NaN is never treated as 0; degenerate windows return NaN.
* All kernels are fail-closed: invalid params raise ValueError at call time.
* TRUE_GAP scope (intraday): these are experimental deep-feature operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import register_operator
from cleaned_operators.intraday._core import (
    DataDegeneracy,
    SessionAggregationOperator,
    _EPS,
    daily_agg,
    log_returns,
    metadata,
    np_errstate,
    register_surface,
)

_CANONICALS: list[str] = []

# Minimum bars for meaningful topology/manifold computation
_MIN_BARS_TOPOLOGY = 30


# ---------------------------------------------------------------------------
# § Matrix Profile Session Features
# ---------------------------------------------------------------------------

def _sliding_dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute sliding dot product of y against x (FFT-based)."""
    n = len(x)
    m = len(y)
    if m > n:
        return np.array([])
    # Zero-pad to next power of 2 for FFT efficiency
    pad_len = int(2 ** np.ceil(np.log2(n + m - 1)))
    x_pad = np.zeros(pad_len)
    y_pad = np.zeros(pad_len)
    x_pad[:n] = x
    y_pad[:m] = y[::-1]  # reverse y for convolution

    with np_errstate():
        x_fft = np.fft.fft(x_pad)
        y_fft = np.fft.fft(y_pad)
        prod = x_fft * y_fft
        result = np.fft.ifft(prod).real

    return result[m-1:n]


def _matrix_profile_features(close_v: np.ndarray, window: int = 20) -> tuple[float, float, float]:
    """Extract matrix profile summary features: min_dist, mean_dist, discord_score.

    Matrix profile computes the z-normalized Euclidean distance from each
    subsequence to its nearest neighbor (excluding trivial matches).  Features:
    - min_dist: minimum profile value (strongest motif)
    - mean_dist: average profile value (global regularity)
    - discord_score: maximum profile value (strongest anomaly)
    """
    r = log_returns(close_v)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < window + 1 or window < 4:
        return np.nan, np.nan, np.nan

    # Z-normalize the full series
    with np_errstate():
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        if not np.isfinite(sigma) or sigma <= _EPS:
            return np.nan, np.nan, np.nan
        r_norm = (r - mu) / sigma if sigma > 1e-10 else np.nan

    # Compute matrix profile (simplified: only distance to nearest neighbor)
    profile = np.full(n - window + 1, np.inf)

    for i in range(n - window + 1):
        subseq = r_norm[i:i+window]
        subseq_mean = float(np.mean(subseq))
        subseq_std = float(np.std(subseq, ddof=1))

        if not np.isfinite(subseq_std) or subseq_std <= _EPS:
            continue

        subseq_z = (subseq - subseq_mean) / subseq_std if subseq_std > 1e-10 else np.nan

        # Compare to all other subsequences (excluding trivial zone)
        for j in range(n - window + 1):
            if abs(i - j) < window // 2:  # exclude trivial matches
                continue

            other = r_norm[j:j+window]
            other_mean = float(np.mean(other))
            other_std = float(np.std(other, ddof=1))

            if not np.isfinite(other_std) or other_std <= _EPS:
                continue

            other_z = (other - other_mean) / other_std if other_std > 1e-10 else np.nan

            with np_errstate():
                dist = float(np.sqrt(np.sum((subseq_z - other_z) ** 2)))

            if np.isfinite(dist):
                profile[i] = min(profile[i], dist)

    # Filter out inf values
    valid = profile[np.isfinite(profile)]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan

    min_dist = float(np.min(valid))
    mean_dist = float(np.mean(valid))
    discord_score = float(np.max(valid))

    return min_dist, mean_dist, discord_score


@register_operator(
    name="intra_matrix_profile_session_features",
    category="intraday_microstructure",
    business_category="intraday_topology",
    canonical="intra_matrix_profile_session_features",
    source="intraday.topology_manifold",
    backend="pandas_numpy",
    status="experimental",
)
class IntraMatrixProfileSessionFeatures(SessionAggregationOperator):
    """日内矩阵剖面特征：最近邻距离的最小值/均值/最大值（motif/discord发现）。

    参数
    ----
    close : minute-frequency panel
        分钟收盘价面板。
    window : int, default=20
        子序列窗口长度（分钟数）。
    feature : {"min_dist", "mean_dist", "discord_score"}, default="discord_score"
        返回的特征：min_dist=最强重复模式，mean_dist=全局规律性，
        discord_score=最强异常（最大距离）。

    返回
    ----
    daily panel
        每日一个标量：所选矩阵剖面特征值。
    """

    metadata = metadata(
        "intra_matrix_profile_session_features",
        "日内矩阵剖面特征（motif/discord/regularity）。",
        ["close", "window", "feature"],
        unit="level",
        cost=10,
        extra_tags=["topology", "time_series_mining", "TRUE_GAP"],
    )

    def _calculate_series(self, close, window=20, feature="discord_score", **_):
        w = int(window)
        if w < 4:
            raise ValueError("window must be >= 4")

        feat = str(feature)
        if feat not in ("min_dist", "mean_dist", "discord_score"):
            raise ValueError("feature must be 'min_dist', 'mean_dist', or 'discord_score'")

        idx = {"min_dist": 0, "mean_dist": 1, "discord_score": 2}[feat]

        def _fn(v, t):
            features = _matrix_profile_features(v, w)
            return features[idx]

        return daily_agg(close, _fn, min_finite=_MIN_BARS_TOPOLOGY)


# ---------------------------------------------------------------------------
# § DMD Koopman Features
# ---------------------------------------------------------------------------

def _dmd_koopman_features(close_v: np.ndarray, rank: int = 5) -> tuple[float, float, float]:
    """Extract DMD (Dynamic Mode Decomposition) Koopman features.

    DMD approximates the Koopman operator for nonlinear dynamics via SVD of
    Hankel-like data matrices.  Features:
    - dominant_freq: frequency of the dominant mode (normalized)
    - growth_rate: real part of dominant eigenvalue (growth/decay)
    - mode_energy: fraction of energy in dominant mode
    """
    r = log_returns(close_v)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < 2 * rank + 2:
        return np.nan, np.nan, np.nan

    # Build data matrices X (r[0:n-1]) and Y (r[1:n])
    # For time-delay embedding: stack delays
    delay = min(rank, n // 4)
    if delay < 2:
        return np.nan, np.nan, np.nan

    # Hankel-style embedding
    X_list = []
    Y_list = []
    for i in range(n - delay):
        X_list.append(r[i:i+delay])
        if i + delay < n:
            Y_list.append(r[i+1:i+delay+1])

    if len(X_list) < delay or len(Y_list) < delay:
        return np.nan, np.nan, np.nan

    X = np.array(X_list).T  # shape (delay, n-delay)
    Y = np.array(Y_list).T

    # SVD of X
    try:
        with np_errstate():
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # Truncate to rank
    r_actual = min(rank, len(S))
    if r_actual < 1:
        return np.nan, np.nan, np.nan

    U_r = U[:, :r_actual]
    S_r = S[:r_actual]
    V_r = Vt[:r_actual, :].T

    # Check for zero singular values
    if S_r[0] <= _EPS:
        return np.nan, np.nan, np.nan

    # Compute DMD operator A_tilde = U_r^T @ Y @ V_r @ S_r^{-1}
    with np_errstate():
        S_inv = np.where(S_r) != 0, np.diag(1.0 / S_r), np.nan)
        A_tilde = U_r.T @ Y @ V_r @ S_inv

    # Eigenvalues of A_tilde
    try:
        eigvals, eigvecs = np.linalg.eig(A_tilde)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    # Find dominant mode (largest magnitude)
    mags = np.abs(eigvals)
    if not np.any(np.isfinite(mags)) or np.max(mags) <= _EPS:
        return np.nan, np.nan, np.nan

    dom_idx = int(np.argmax(mags))
    dom_eigval = eigvals[dom_idx]

    # Extract features
    with np_errstate():
        # Frequency: angle / (2*pi), normalized to [0, 0.5]
        freq = np.where((2.0 * np.pi)) != 0, float(np.abs(np.angle(dom_eigval)) / (2.0 * np.pi)), np.nan)
        # Growth rate: log magnitude (dt=1)
        growth = float(np.log(np.abs(dom_eigval)))
        # Mode energy: fraction of energy in dominant mode
        total_energy = float(np.sum(mags ** 2))
        mode_energy = float(mags[dom_idx] ** 2 / (total_energy + _EPS))

    if not np.isfinite(freq) or not np.isfinite(growth) or not np.isfinite(mode_energy):
        return np.nan, np.nan, np.nan

    return freq, growth, mode_energy


@register_operator(
    name="intra_dmd_koopman_features",
    category="intraday_microstructure",
    business_category="intraday_topology",
    canonical="intra_dmd_koopman_features",
    source="intraday.topology_manifold",
    backend="pandas_numpy",
    status="experimental",
)
class IntraDmdKoopmanFeatures(SessionAggregationOperator):
    """日内动态模态分解（DMD）Koopman 特征：主导频率/增长率/模态能量。

    参数
    ----
    close : minute-frequency panel
        分钟收盘价面板。
    rank : int, default=5
        DMD 截断秩（保留的主导模态数）。
    feature : {"dominant_freq", "growth_rate", "mode_energy"}, default="growth_rate"
        返回的特征：dominant_freq=主导频率，growth_rate=增长率（对数），
        mode_energy=主模态能量占比。

    返回
    ----
    daily panel
        每日一个标量：所选 DMD/Koopman 特征值。
    """

    metadata = metadata(
        "intra_dmd_koopman_features",
        "日内DMD Koopman特征（主导模态动力学）。",
        ["close", "rank", "feature"],
        unit="level",
        cost=12,
        extra_tags=["topology", "koopman", "dynamical_systems", "TRUE_GAP"],
    )

    def _calculate_series(self, close, rank=5, feature="growth_rate", **_):
        r = int(rank)
        if r < 2:
            raise ValueError("rank must be >= 2")
        if r > 20:
            raise ValueError("rank must be <= 20 (computational cost)")

        feat = str(feature)
        if feat not in ("dominant_freq", "growth_rate", "mode_energy"):
            raise ValueError("feature must be 'dominant_freq', 'growth_rate', or 'mode_energy'")

        idx = {"dominant_freq": 0, "growth_rate": 1, "mode_energy": 2}[feat]

        def _fn(v, t):
            features = _dmd_koopman_features(v, r)
            return features[idx]

        return daily_agg(close, _fn, min_finite=_MIN_BARS_TOPOLOGY)


# ---------------------------------------------------------------------------
# § Covariance Manifold Shift
# ---------------------------------------------------------------------------

def _cov_manifold_shift(close_v: np.ndarray, window: int = 30) -> float:
    """Riemannian distance between early-session and late-session covariance.

    Models returns as a multivariate process (via time-delay embedding) and
    computes the geodesic distance on the SPD (symmetric positive-definite)
    manifold between two epoch covariance matrices.  Large shift signals
    regime change within the session.
    """
    r = log_returns(close_v)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < 2 * window:
        return np.nan

    # Split into two halves
    mid = n // 2
    r_early = r[:mid]
    r_late = r[mid:]

    # Time-delay embedding dimension (small to avoid singularity)
    embed_dim = min(5, window // 6)
    if embed_dim < 2:
        return np.nan

    def _embed_cov(x: np.ndarray) -> np.ndarray:
        """Build delay-embedded covariance matrix."""
        m = len(x)
        if m < embed_dim + 1:
            return np.array([[]])

        # Hankel embedding
        rows = []
        for i in range(m - embed_dim + 1):
            rows.append(x[i:i+embed_dim])

        if len(rows) < embed_dim:
            return np.array([[]])

        X = np.array(rows).T  # shape (embed_dim, m-embed_dim+1)

        # Covariance with regularization
        with np_errstate():
            C = np.cov(X, ddof=1)
            if C.shape[0] != embed_dim:
                return np.array([[]])
            # Regularize: C + eps * I
            C += _EPS * np.eye(embed_dim)

        return C

    C_early = _embed_cov(r_early)
    C_late = _embed_cov(r_late)

    if C_early.size == 0 or C_late.size == 0:
        return np.nan

    # Riemannian distance on SPD manifold: d(C1, C2) = ||log(C1^{-1/2} C2 C1^{-1/2})||_F
    try:
        with np_errstate():
            # C_early^{-1/2}
            eigvals_e, eigvecs_e = np.linalg.eigh(C_early)
            if np.any(eigvals_e <= _EPS):
                return np.nan
            C_early_inv_sqrt = np.where(np.sqrt(eigvals_e)) @ eigvecs_e.T != 0, eigvecs_e @ np.diag(1.0 / np.sqrt(eigvals_e)) @ eigvecs_e.T, np.nan)

            # M = C_early^{-1/2} @ C_late @ C_early^{-1/2}
            M = C_early_inv_sqrt @ C_late @ C_early_inv_sqrt

            # log(M) via eigendecomposition
            eigvals_m, eigvecs_m = np.linalg.eigh(M)
            if np.any(eigvals_m <= _EPS):
                return np.nan
            log_M = eigvecs_m @ np.diag(np.log(eigvals_m)) @ eigvecs_m.T

            # Frobenius norm
            dist = float(np.sqrt(np.trace(log_M @ log_M.T)))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    if not np.isfinite(dist):
        return np.nan

    return dist


@register_operator(
    name="intra_covariance_manifold_shift",
    category="intraday_microstructure",
    business_category="intraday_topology",
    canonical="intra_covariance_manifold_shift",
    source="intraday.topology_manifold",
    backend="pandas_numpy",
    status="experimental",
)
class IntraCovarianceManifoldShift(SessionAggregationOperator):
    """日内协方差流形偏移：早盘与晚盘协方差矩阵的黎曼距离（SPD流形）。

    参数
    ----
    close : minute-frequency panel
        分钟收盘价面板。
    window : int, default=30
        最小所需分钟数（每个半场）。

    返回
    ----
    daily panel
        每日一个标量：早晚盘协方差在对称正定流形上的测地距离。
        大偏移=盘中regime剧烈变化。
    """

    metadata = metadata(
        "intra_covariance_manifold_shift",
        "日内协方差流形偏移（Riemannian距离）。",
        ["close", "window"],
        unit="level",
        cost=15,
        extra_tags=["topology", "manifold", "regime_change", "TRUE_GAP"],
    )

    def _calculate_series(self, close, window=30, **_):
        w = int(window)
        if w < 10:
            raise ValueError("window must be >= 10")

        def _fn(v, t):
            return _cov_manifold_shift(v, w)

        return daily_agg(close, _fn, min_finite=_MIN_BARS_TOPOLOGY)


# ---------------------------------------------------------------------------
# § Critical Transition Score
# ---------------------------------------------------------------------------

def _critical_transition_score(close_v: np.ndarray, window: int = 40) -> float:
    """Early-warning signal composite for critical transitions (tipping points).

    Combines three generic indicators of approaching regime shifts:
    1. Increasing autocorrelation (critical slowing down)
    2. Increasing variance (loss of stability)
    3. Increasing skewness (asymmetric flickering)

    Returns a z-scored composite; positive = warning signal.
    """
    r = log_returns(close_v)
    r = r[np.isfinite(r)]
    n = len(r)

    if n < 2 * window:
        return np.nan

    # Compute indicators in rolling windows
    half_n = n // 2
    n_windows = max(2, half_n // 5)  # at least 2 windows

    if n_windows < 2:
        return np.nan

    step = max(1, half_n // n_windows)

    autocorrs = []
    variances = []
    skews = []

    for i in range(0, half_n, step):
        if i + window > n:
            break

        seg = r[i:i+window]
        if len(seg) < window // 2:
            continue

        # Autocorrelation at lag 1
        with np_errstate():
            if len(seg) > 1:
                ac = float(np.corrcoef(seg[:-1], seg[1:])[0, 1])
                if np.isfinite(ac):
                    autocorrs.append(ac)

        # Variance
        var = float(np.var(seg, ddof=1))
        if np.isfinite(var):
            variances.append(var)

        # Skewness
        if len(seg) >= 3:
            m3 = float(np.mean((seg - np.mean(seg)) ** 3))
            s3 = float(np.std(seg, ddof=1) ** 3)
            if s3 > _EPS:
                skew = m3 / s3 if s3 != 0 else np.nan
                if np.isfinite(skew):
                    skews.append(skew)

    if len(autocorrs) < 2 or len(variances) < 2 or len(skews) < 2:
        return np.nan

    # Compute trend slopes (increasing = warning)
    def _slope(y: list[float]) -> float:
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)
        y_arr = np.array(y)
        with np_errstate():
            x_mean = np.mean(x)
            y_mean = np.mean(y_arr)
            num = np.sum((x - x_mean) * (y_arr - y_mean))
            den = np.sum((x - x_mean) ** 2)
            if den <= _EPS:
                return np.nan
            return np.where(den) != 0, float(num / den), np.nan)

    slope_ac = _slope(autocorrs)
    slope_var = _slope(variances)
    slope_skew = _slope(np.abs(skews))  # absolute value: either tail

    if not np.isfinite(slope_ac) or not np.isfinite(slope_var) or not np.isfinite(slope_skew):
        return np.nan

    # Combine into composite (simple average of z-scores)
    # Use robust normalization (median/MAD)
    slopes = np.array([slope_ac, slope_var, slope_skew])
    with np_errstate():
        med = float(np.median(slopes))
        mad = float(np.median(np.abs(slopes - med)))
        if mad <= _EPS:
            return 0.0
        composite = np.where((mad * 1.4826))) != 0, float(np.mean((slopes - med) / (mad * 1.4826))), np.nan)

    return composite


@register_operator(
    name="intra_critical_transition_score",
    category="intraday_microstructure",
    business_category="intraday_topology",
    canonical="intra_critical_transition_score",
    source="intraday.topology_manifold",
    backend="pandas_numpy",
    status="experimental",
)
class IntraCriticalTransitionScore(SessionAggregationOperator):
    """日内临界转换得分：早期预警信号复合指标（自相关/方差/偏度趋势）。

    参数
    ----
    close : minute-frequency panel
        分钟收盘价面板。
    window : int, default=40
        滚动窗口长度（计算各指标）。

    返回
    ----
    daily panel
        每日一个标量：临界转换得分（z-scored）。正值=预警信号（接近tipping point）。
    """

    metadata = metadata(
        "intra_critical_transition_score",
        "日内临界转换得分（早期预警复合指标）。",
        ["close", "window"],
        unit="level",
        cost=10,
        extra_tags=["topology", "regime_shift", "early_warning", "TRUE_GAP"],
    )

    def _calculate_series(self, close, window=40, **_):
        w = int(window)
        if w < 10:
            raise ValueError("window must be >= 10")

        def _fn(v, t):
            return _critical_transition_score(v, w)

        return daily_agg(close, _fn, min_finite=_MIN_BARS_TOPOLOGY)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_CANONICALS.extend([
    "intra_matrix_profile_session_features",
    "intra_dmd_koopman_features",
    "intra_covariance_manifold_shift",
    "intra_critical_transition_score",
])

register_surface(_CANONICALS)
