import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Literal, Optional
import logging
from numba import njit, prange

def robust_winsorization(factor_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    稳健去极值 (Robust Winsorization)
    必须使用 MAD (Median Absolute Deviation) 进行去极值。
    注意：MAD 乘数已写死为 3.148 (对应正态分布的 3-Sigma)。

    参数:
        factor_matrix: 截面特征宽表 (index=datetime, columns=asset)

    返回:
        截断后的因子暴露矩阵。
    """
    n_mad = 3.148
    
    # numpy vectorized execution for efficiency
    mat = factor_matrix.values
    # compute median ignoring nan
    medians = np.nanmedian(mat, axis=1, keepdims=True)
    abs_dev = np.abs(mat - medians)
    mads = np.nanmedian(abs_dev, axis=1, keepdims=True)
    
    upper_bounds = medians + n_mad * mads
    lower_bounds = medians - n_mad * mads
    
    # clip using numpy
    winsorized_mat = np.clip(mat, lower_bounds, upper_bounds)
    
    winsorized = pd.DataFrame(winsorized_mat, index=factor_matrix.index, columns=factor_matrix.columns)
    return winsorized


def risk_orthogonalization(
    factor_matrix: pd.DataFrame,
    risk_exposures: pd.DataFrame,
    weights: pd.Series
) -> pd.DataFrame:
    """
    高阶风险正交化 (WLS Orthogonalization)
    剥离已知风格因素 (如行业、市值、Beta)，获取纯净残差。

    参数:
        factor_matrix: 暴露于风险的因子宽表 (index=datetime, columns=asset)。
        risk_exposures: 风险因子矩阵 (index=asset, columns=K个风险因子)。
        weights: WLS 权重 (index=asset)。

    返回:
        正交后的纯净 Alpha 残差宽表。
    """
    residuals = pd.DataFrame(np.nan, index=factor_matrix.index, columns=factor_matrix.columns)
    
    # 确保风险和权重的 asset 对齐因子矩阵的列
    common_assets = factor_matrix.columns.intersection(risk_exposures.index).intersection(weights.index)
    if len(common_assets) == 0:
        logging.warning("No common assets found between factor_matrix, risk_exposures and weights.")
        return residuals
        
    Y = factor_matrix[common_assets].values # T x N
    X = sm.add_constant(risk_exposures.loc[common_assets]).values # N x K+1
    W = weights.loc[common_assets].values # N
    
    T, N = Y.shape
    K = X.shape[1]
    
    resid_mat = np.full((T, N), np.nan)
    
    # Numba cannot easily handle arbitrary statsmodels logic if we want exactly statsmodels WLS.
    # But WLS is just OLS on scaled data: Y_w = Y * sqrt(W), X_w = X * sqrt(W).
    # resid = Y - X * beta
    
    # Extract valid masks for X and W to avoid redundant checks
    valid_xw_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(W)
    X_valid = X[valid_xw_mask]
    W_valid = W[valid_xw_mask]
    sqrt_W = np.sqrt(W_valid)[:, np.newaxis]
    X_w = X_valid * sqrt_W
    
    # Pre-calculate pseudo-inverse if possible, but valid assets might change per row if Y has nans.
    # Usually, Y has different nan patterns per row. We will use a numpy vectorized approach where possible.
    
    for t in range(T):
        y_t = Y[t]
        valid_mask = valid_xw_mask & ~np.isnan(y_t)
        if not np.any(valid_mask):
            continue
            
        # extract valid subset
        y_v = y_t[valid_mask]
        X_v = X[valid_mask]
        W_v = W[valid_mask]
        
        # WLS transformation
        sqrt_w_v = np.sqrt(W_v)
        X_w_v = X_v * sqrt_w_v[:, np.newaxis]
        y_w_v = y_v * sqrt_w_v
        
        try:
            # solve OLS on weighted data
            beta, _, _, _ = np.linalg.lstsq(X_w_v, y_w_v, rcond=None)
            # residuals in original space
            resid_mat[t, valid_mask] = y_v - X_v @ beta
        except Exception as e:
            # fallback to nan
            pass

    residuals.loc[:, common_assets] = resid_mat
    return residuals


def dynamic_imputation(
    factor_matrix: pd.DataFrame,
    method: Literal['industry_weighted', 'forward_fill_decay'] = 'industry_weighted',
    industry_labels: Optional[pd.Series] = None,
    weights: Optional[pd.Series] = None,
    max_delay: int = 5
) -> pd.DataFrame:
    """
    动态缺失值填补 (Dynamic Imputation)
    针对停牌或数据缺失执行多策略填补。

    参数:
        factor_matrix: 因子宽表 (index=datetime, columns=asset)。
        method: 填补策略，'industry_weighted' 或 'forward_fill_decay'。
        industry_labels: 行业分类标签序列 (index=asset)。
        weights: 权重序列 (index=asset)。
        max_delay: 时序填补的最大容忍延迟期数。
    """
    imputed_matrix = factor_matrix.copy()

    if method == 'industry_weighted':
        if industry_labels is None or weights is None:
            raise ValueError("Method 'industry_weighted' requires both 'industry_labels' and 'weights'.")
            
        common_assets = factor_matrix.columns.intersection(industry_labels.index).intersection(weights.index)
        if len(common_assets) == 0:
            return imputed_matrix
            
        mat = factor_matrix[common_assets].values # T x N
        ind = industry_labels.loc[common_assets].values
        w = weights.loc[common_assets].values
        
        # factorize industry labels for numpy grouping
        unique_inds, ind_idx = np.unique(ind, return_inverse=True)
        num_inds = len(unique_inds)
        
        # we can compute industry weighted means for all T rows efficiently
        # mat is T x N. 
        # w is N. We broadcast w to T x N.
        W_mat = np.tile(w, (mat.shape[0], 1))
        
        # mask out where mat is nan or W_mat is nan
        valid_mask = ~np.isnan(mat) & ~np.isnan(W_mat)
        
        mat_zeroed = np.where(valid_mask, mat, 0)
        w_zeroed = np.where(valid_mask, W_mat, 0)
        
        # aggregate by industry
        # using matrix multiplication with indicator matrix
        # indicator: N x num_inds
        ind_indicator = np.zeros((len(common_assets), num_inds))
        ind_indicator[np.arange(len(common_assets)), ind_idx] = 1
        
        # weighted sum: T x num_inds
        sum_weighted_vals = (mat_zeroed * w_zeroed) @ ind_indicator
        # sum of weights: T x num_inds
        sum_weights = w_zeroed @ ind_indicator
        
        # compute means
        with np.errstate(divide='ignore', invalid='ignore'):
            ind_means = np.where(sum_weights > 0, sum_weighted_vals / sum_weights, np.nan)
            
        # For industries with sum_weights == 0, we can fallback to simple mean
        # (if weights were all 0 but vals existed, though rare)
        sum_vals = mat_zeroed @ ind_indicator
        count_vals = valid_mask @ ind_indicator
        with np.errstate(divide='ignore', invalid='ignore'):
            simple_means = np.where(count_vals > 0, sum_vals / count_vals, np.nan)
            
        ind_means = np.where(np.isnan(ind_means), simple_means, ind_means)
        
        # broadcast means back to T x N
        means_broadcast = ind_means[:, ind_idx]
        
        # fill where originally nan
        mat_filled = np.where(np.isnan(mat) & ~np.isnan(means_broadcast), means_broadcast, mat)
        imputed_matrix.loc[:, common_assets] = mat_filled

    elif method == 'forward_fill_decay':
        # Numpy/pandas fast forward fill decay
        ffilled = factor_matrix.ffill(limit=max_delay)
        
        decay_rate = 0.9
        mask_nan = factor_matrix.isna()
        
        dist = pd.DataFrame(0, index=factor_matrix.index, columns=factor_matrix.columns)
        for column in factor_matrix.columns:
            column_missing = mask_nan[column]
            missing_group = (~column_missing).cumsum()
            dist[column] = column_missing.groupby(missing_group).cumsum().astype(int)
        
        # vectorized decay
        decay_factor = np.power(decay_rate, dist.values)
        imputed_matrix = factor_matrix.where(~mask_nan, ffilled * decay_factor)
        imputed_matrix = imputed_matrix.where(dist <= max_delay, np.nan)

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return imputed_matrix
