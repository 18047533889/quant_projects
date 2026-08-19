"""
Diagnostics for neutralization quality and numerical stability.

Provides exposure diagnostics, condition number checks,
and post-neutralization validation.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple


def compute_condition_number(
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> pd.DataFrame:
    """
    Compute condition number of exposure matrix per date.

    Parameters
    ----------
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column

    Returns
    -------
    pd.DataFrame
        Columns: [date, condition_number, rank, n_exposures, n_observations]

    Notes
    -----
    Condition number κ(X) = σ_max / σ_min measures numerical stability.
    κ > 100: potential numerical issues
    κ > 1000: serious ill-conditioning
    """
    exposure_cols = [c for c in exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    results = []

    for date, group in exposures.groupby(date_col):
        X = group[exposure_cols].values

        # Remove rows with NaN
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_valid = X[valid_mask]

        if X_valid.shape[0] > 0:
            try:
                # Compute singular values
                _, s, _ = np.linalg.svd(X_valid, full_matrices=False)

                # Condition number
                if s[-1] > 0:
                    cond = s[0] / s[-1]
                else:
                    cond = np.inf

                # Matrix rank
                rank = np.sum(s > 1e-10 * s[0])

                results.append({
                    date_col: date,
                    "condition_number": cond,
                    "rank": rank,
                    "n_exposures": X_valid.shape[1],
                    "n_observations": X_valid.shape[0],
                })

            except np.linalg.LinAlgError:
                results.append({
                    date_col: date,
                    "condition_number": np.nan,
                    "rank": np.nan,
                    "n_exposures": X_valid.shape[1],
                    "n_observations": X_valid.shape[0],
                })
        else:
            results.append({
                date_col: date,
                "condition_number": np.nan,
                "rank": np.nan,
                "n_exposures": len(exposure_cols),
                "n_observations": 0,
            })

    return pd.DataFrame(results)


def compute_exposure_correlation(
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> pd.DataFrame:
    """
    Compute correlation matrix of exposures per date.

    Parameters
    ----------
    exposures : pd.DataFrame
        Exposure matrix
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column

    Returns
    -------
    pd.DataFrame
        Pairwise correlations with columns: [date, exposure1, exposure2, correlation]

    Notes
    -----
    High correlation (|ρ| > 0.9) indicates potential multicollinearity.
    """
    exposure_cols = [c for c in exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    results = []

    for date, group in exposures.groupby(date_col):
        X = group[exposure_cols].values

        valid_mask = np.all(np.isfinite(X), axis=1)
        X_valid = X[valid_mask]

        if X_valid.shape[0] > 1:
            # Compute correlation matrix
            corr_matrix = np.corrcoef(X_valid, rowvar=False)

            # Extract upper triangle (exclude diagonal)
            n_features = len(exposure_cols)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    results.append({
                        date_col: date,
                        "exposure1": exposure_cols[i],
                        "exposure2": exposure_cols[j],
                        "correlation": corr_matrix[i, j],
                    })

    return pd.DataFrame(results)


def check_residual_exposures(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "residual",
) -> pd.DataFrame:
    """
    Check residual exposures after neutralization.

    Parameters
    ----------
    residuals : pd.DataFrame
        Neutralized residuals. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Original exposure matrix
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Residual column name

    Returns
    -------
    pd.DataFrame
        Columns: [date, exposure, correlation, t_stat, p_value]

    Notes
    -----
    Successful neutralization should produce near-zero correlations.
    t-statistic tests H0: ρ = 0
    """
    exposure_cols = [c for c in exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    merged = residuals.merge(
        exposures,
        on=[date_col, asset_col],
        how="inner",
    )

    results = []

    for date, group in merged.groupby(date_col):
        r = group[value_col].values
        X = group[exposure_cols].values

        valid_mask = np.isfinite(r) & np.all(np.isfinite(X), axis=1)
        r_valid = r[valid_mask]
        X_valid = X[valid_mask]

        n = len(r_valid)

        if n > 2:
            for i, col in enumerate(exposure_cols):
                exp = X_valid[:, i]

                # Compute correlation
                if np.std(r_valid) > 0 and np.std(exp) > 0:
                    corr = np.corrcoef(r_valid, exp)[0, 1]

                    # t-statistic for testing ρ = 0
                    t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2 + 1e-10)

                    # Two-tailed p-value (approximate using normal)
                    from scipy import stats
                    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

                    results.append({
                        date_col: date,
                        "exposure": col,
                        "correlation": corr,
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "n_observations": n,
                    })

    return pd.DataFrame(results)


def diagnose_neutralization(
    values: pd.DataFrame,
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    residual_col: str = "residual",
) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive neutralization diagnostics.

    Parameters
    ----------
    values : pd.DataFrame
        Original factor values
    residuals : pd.DataFrame
        Neutralized residuals
    exposures : pd.DataFrame
        Exposure matrix
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Original value column name
    residual_col : str
        Residual column name

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys:
        - 'condition_numbers': Condition numbers per date
        - 'exposure_correlations': Pairwise exposure correlations
        - 'residual_exposures': Residual-exposure correlations
        - 'variance_reduction': Variance reduction per date

    Notes
    -----
    Provides a complete picture of neutralization quality:
    - Numerical stability (condition numbers)
    - Multicollinearity (exposure correlations)
    - Neutralization effectiveness (residual exposures)
    - Information preservation (variance reduction)
    """
    # Condition numbers
    cond_df = compute_condition_number(exposures, date_col, asset_col)

    # Exposure correlations
    exp_corr_df = compute_exposure_correlation(exposures, date_col, asset_col)

    # Residual exposures
    residual_exp_df = check_residual_exposures(
        residuals, exposures, date_col, asset_col, residual_col
    )

    # Variance reduction
    variance_reduction = compute_variance_reduction(
        values, residuals, date_col, asset_col, value_col, residual_col
    )

    return {
        "condition_numbers": cond_df,
        "exposure_correlations": exp_corr_df,
        "residual_exposures": residual_exp_df,
        "variance_reduction": variance_reduction,
    }


def compute_variance_reduction(
    values: pd.DataFrame,
    residuals: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    residual_col: str = "residual",
) -> pd.DataFrame:
    """
    Compute variance reduction from neutralization.

    Parameters
    ----------
    values : pd.DataFrame
        Original factor values
    residuals : pd.DataFrame
        Neutralized residuals
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Original value column
    residual_col : str
        Residual column

    Returns
    -------
    pd.DataFrame
        Columns: [date, original_var, residual_var, variance_explained, r_squared]

    Notes
    -----
    R² = 1 - Var(residuals) / Var(original)
    High R² indicates strong exposure effects.
    """
    merged = values[[date_col, asset_col, value_col]].merge(
        residuals[[date_col, asset_col, residual_col]],
        on=[date_col, asset_col],
        how="inner",
    )

    results = []

    for date, group in merged.groupby(date_col):
        orig = group[value_col].values
        resid = group[residual_col].values

        valid_mask = np.isfinite(orig) & np.isfinite(resid)
        orig_valid = orig[valid_mask]
        resid_valid = resid[valid_mask]

        if len(orig_valid) > 1:
            orig_var = np.var(orig_valid, ddof=1)
            resid_var = np.var(resid_valid, ddof=1)

            if orig_var > 0:
                r_squared = 1 - resid_var / orig_var
                variance_explained = orig_var - resid_var
            else:
                r_squared = np.nan
                variance_explained = 0.0

            results.append({
                date_col: date,
                "original_var": orig_var,
                "residual_var": resid_var,
                "variance_explained": variance_explained,
                "r_squared": r_squared,
                "n_observations": len(orig_valid),
            })

    return pd.DataFrame(results)


def flag_ill_conditioned_dates(
    exposures: pd.DataFrame,
    threshold: float = 100.0,
    date_col: str = "date",
    asset_col: str = "asset_id",
) -> pd.DataFrame:
    """
    Flag dates with ill-conditioned exposure matrices.

    Parameters
    ----------
    exposures : pd.DataFrame
        Exposure matrix
    threshold : float
        Condition number threshold (default 100)
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column

    Returns
    -------
    pd.DataFrame
        Dates exceeding threshold with columns: [date, condition_number, severity]
        severity: 'moderate' (100-1000), 'severe' (>1000)

    Notes
    -----
    Regularization (Ridge/Elastic Net) recommended for flagged dates.
    """
    cond_df = compute_condition_number(exposures, date_col, asset_col)

    # NaN condition numbers (SVD failure / no valid rows) are the most
    # ill-conditioned dates of all — flag them as 'severe' rather than
    # letting the > threshold comparison silently drop them.
    nan_mask = cond_df["condition_number"].isna()
    flagged = cond_df[
        nan_mask | (cond_df["condition_number"] > threshold)
    ].copy()

    if len(flagged) > 0:
        flagged["severity"] = pd.cut(
            flagged["condition_number"],
            bins=[threshold, 1000, np.inf],
            labels=["moderate", "severe"],
        )
        flagged.loc[flagged["condition_number"].isna(), "severity"] = "severe"

    return flagged


def summarize_diagnostics(diagnostics: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Summarize diagnostics into key metrics.

    Parameters
    ----------
    diagnostics : Dict[str, pd.DataFrame]
        Output from diagnose_neutralization

    Returns
    -------
    Dict[str, any]
        Summary statistics including:
        - median_condition_number
        - max_condition_number
        - pct_ill_conditioned (condition > 100)
        - max_exposure_correlation
        - mean_residual_exposure_correlation
        - median_r_squared

    Notes
    -----
    Provides quick health check of neutralization quality.
    """
    cond = diagnostics["condition_numbers"]
    exp_corr = diagnostics["exposure_correlations"]
    resid_exp = diagnostics["residual_exposures"]
    var_red = diagnostics["variance_reduction"]

    summary = {}

    # Condition numbers
    if len(cond) > 0:
        summary["median_condition_number"] = cond["condition_number"].median()
        summary["max_condition_number"] = cond["condition_number"].max()
        summary["pct_ill_conditioned"] = (
            (cond["condition_number"] > 100).sum() / len(cond) * 100
        )
    else:
        summary["median_condition_number"] = np.nan
        summary["max_condition_number"] = np.nan
        summary["pct_ill_conditioned"] = np.nan

    # Exposure correlations
    if len(exp_corr) > 0:
        summary["max_exposure_correlation"] = exp_corr["correlation"].abs().max()
        summary["mean_exposure_correlation"] = exp_corr["correlation"].abs().mean()
    else:
        summary["max_exposure_correlation"] = np.nan
        summary["mean_exposure_correlation"] = np.nan

    # Residual exposures
    if len(resid_exp) > 0:
        summary["max_residual_exposure_correlation"] = (
            resid_exp["correlation"].abs().max()
        )
        summary["mean_residual_exposure_correlation"] = (
            resid_exp["correlation"].abs().mean()
        )
    else:
        summary["max_residual_exposure_correlation"] = np.nan
        summary["mean_residual_exposure_correlation"] = np.nan

    # Variance reduction
    if len(var_red) > 0:
        summary["median_r_squared"] = var_red["r_squared"].median()
        summary["mean_r_squared"] = var_red["r_squared"].mean()
    else:
        summary["median_r_squared"] = np.nan
        summary["mean_r_squared"] = np.nan

    return summary
