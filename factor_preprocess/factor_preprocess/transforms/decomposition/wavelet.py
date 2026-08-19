"""
Wavelet decomposition for offline multi-scale time series analysis.

These public operations reconstruct each output from the available full
series. A one-observation shift does not make that reconstruction prefix-stable,
so these transforms are restricted to offline/research use.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import pywt

# Narrowed catch set: the documented NaN fail-closed contract covers only
# the *numerical* failures these libraries raise on degenerate-but-valid
# input (too few observations, all-NaN windows, SVD non-convergence).
# Programming errors (TypeError, AttributeError, KeyError from bad column
# names) must propagate so they surface instead of being silently NaN'd.
_NUMERICAL_FAILURES = (ValueError, IndexError, np.linalg.LinAlgError, ArithmeticError)


def wavelet_decompose(
    values: pd.DataFrame,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    mode: str = 'symmetric',
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> Dict[str, pd.Series]:
    """
    Offline-only wavelet decomposition into approximation and detail coefficients.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    wavelet : str
        Wavelet family to use. Common choices:
        - 'db4': Daubechies 4 (good for financial data)
        - 'sym4': Symlet 4 (symmetric)
        - 'coif3': Coiflet 3
    level : int, optional
        Decomposition level. If None, uses maximum useful level.
    mode : str
        Signal extension mode for padding.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    Dict[str, pd.Series]
        Dictionary with keys:
        - 'a{level}': approximation coefficients at level
        - 'd{i}': detail coefficients at level i (i=1..level)
        All aligned with input index.
        First observation per asset is NaN (causal lag).

    Notes
    -----
    Uses shift(1) to exclude current observation, ensuring causality.
    Approximation = low-frequency (smooth trend)
    Details = high-frequency (noise/cycles)
    """
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Validate wavelet
    if wavelet not in pywt.wavelist():
        raise ValueError(f"Unknown wavelet: {wavelet}. Use pywt.wavelist() for valid options.")

    def _wavelet_decompose_single(series: pd.Series) -> Dict[str, pd.Series]:
        """Apply wavelet decomposition to a single asset's time series."""
        # Shift to exclude current observation
        lagged = series.shift(1)

        if lagged.isna().all():
            na_series = pd.Series(np.nan, index=series.index)
            return {'a1': na_series, 'd1': na_series.copy()}

        # Extract non-NaN values
        valid_mask = lagged.notna()
        y = lagged[valid_mask].values

        # Determine decomposition level
        max_level = pywt.dwt_max_level(len(y), wavelet)
        decomp_level = level if level is not None else min(max_level, 5)

        if decomp_level < 1:
            na_series = pd.Series(np.nan, index=series.index)
            return {'a1': na_series, 'd1': na_series.copy()}

        if len(y) < 2 * decomp_level:
            # Insufficient data for this level
            na_series = pd.Series(np.nan, index=series.index)
            return {f'a{decomp_level}': na_series, f'd{decomp_level}': na_series.copy()}

        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(y, wavelet, mode=mode, level=decomp_level)

            # Reconstruct full signal from approximation only
            approx_coeffs_recon = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            approx_signal = pywt.waverec(approx_coeffs_recon, wavelet, mode=mode)

            # Handle length mismatch
            if len(approx_signal) > len(y):
                approx_signal = approx_signal[:len(y)]
            elif len(approx_signal) < len(y):
                approx_signal = np.pad(approx_signal, (0, len(y) - len(approx_signal)), mode='edge')

            result_dict = {}

            # Store approximation
            approx_series = pd.Series(np.nan, index=series.index)
            approx_series.loc[valid_mask] = approx_signal
            result_dict[f'a{decomp_level}'] = approx_series

            # Reconstruct each detail level
            for i in range(1, len(coeffs)):
                detail_coeffs_recon = [np.zeros_like(coeffs[0])] + [
                    coeffs[j] if j == i else np.zeros_like(coeffs[j])
                    for j in range(1, len(coeffs))
                ]
                detail_signal = pywt.waverec(detail_coeffs_recon, wavelet, mode=mode)

                # Handle length mismatch
                if len(detail_signal) > len(y):
                    detail_signal = detail_signal[:len(y)]
                elif len(detail_signal) < len(y):
                    detail_signal = np.pad(detail_signal, (0, len(y) - len(detail_signal)), mode='edge')

                detail_series = pd.Series(np.nan, index=series.index)
                detail_series.loc[valid_mask] = detail_signal
                result_dict[f'd{decomp_level - i + 1}'] = detail_series

            return result_dict

        except _NUMERICAL_FAILURES:
            # Wavelet decomposition may fail on degenerate-but-valid input
            # (too few points, non-invertible coefficient shape).  A bare
            # `except Exception` would also swallow programming bugs; keep
            # the NaN fail-closed contract for numerical failures only.
            na_series = pd.Series(np.nan, index=series.index)
            return {f'a{decomp_level}': na_series, f'd{decomp_level}': na_series.copy()}

    # Apply per asset and aggregate results
    all_results = []
    for asset_id, group in values.groupby(asset_col, sort=False):
        asset_results = _wavelet_decompose_single(group[value_col])
        for key, series in asset_results.items():
            all_results.append(pd.DataFrame({
                'key': key,
                'value': series,
            }, index=group.index))

    if not all_results:
        na_series = pd.Series(np.nan, index=values.index)
        return {'a1': na_series, 'd1': na_series.copy()}

    combined = pd.concat(all_results)

    # Pivot to get one series per component
    result_dict = {}
    for key in combined['key'].unique():
        key_data = combined[combined['key'] == key]['value']
        key_data = key_data.loc[values.index]  # Ensure order
        result_dict[key] = key_data

    return result_dict


def wavelet_smooth(
    values: pd.DataFrame,
    wavelet: str = 'db4',
    level: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Offline-only wavelet smoothing by reconstructing approximation coefficients.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    wavelet : str
        Wavelet family to use
    level : int
        Decomposition level. Higher level = smoother result.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to smooth

    Returns
    -------
    pd.Series
        Smoothed series (low-frequency approximation).
        First observation per asset is NaN.

    Notes
    -----
    Extracts approximation coefficients and reconstructs.
    Removes high-frequency noise while preserving trend.
    """
    decomp = wavelet_decompose(
        values,
        wavelet=wavelet,
        level=level,
        asset_col=asset_col,
        time_col=time_col,
        value_col=value_col,
    )

    # Return approximation component
    approx_key = f'a{level}'
    if approx_key in decomp:
        return decomp[approx_key]
    else:
        # Fallback to any approximation key
        for key in decomp:
            if key.startswith('a'):
                return decomp[key]

    return pd.Series(np.nan, index=values.index)


def wavelet_denoise(
    values: pd.DataFrame,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_mode: str = 'soft',
    threshold_scale: float = 1.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Offline-only wavelet denoising using soft/hard thresholding.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    wavelet : str
        Wavelet family to use
    level : int, optional
        Decomposition level. If None, uses maximum useful level.
    threshold_mode : str
        'soft' or 'hard' thresholding
    threshold_scale : float
        Scale factor for threshold. Higher = more aggressive denoising.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to denoise

    Returns
    -------
    pd.Series
        Denoised series.
        First observation per asset is NaN.

    Notes
    -----
    Uses universal threshold: sigma * sqrt(2 * log(n))
    where sigma is estimated from finest scale detail coefficients.
    """
    if threshold_mode not in ['soft', 'hard']:
        raise ValueError(f"threshold_mode must be 'soft' or 'hard', got {threshold_mode}")

    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def _wavelet_denoise_single(series: pd.Series) -> pd.Series:
        """Apply wavelet denoising to a single asset's time series."""
        # Shift to exclude current observation
        lagged = series.shift(1)

        if lagged.isna().all():
            return pd.Series(np.nan, index=series.index)

        # Extract non-NaN values
        valid_mask = lagged.notna()
        y = lagged[valid_mask].values

        if len(y) < 4:
            result = pd.Series(np.nan, index=series.index)
            return result

        try:
            # Perform wavelet decomposition
            max_level = pywt.dwt_max_level(len(y), wavelet)
            decomp_level = level if level is not None else min(max_level, 5)

            coeffs = pywt.wavedec(y, wavelet, level=decomp_level)

            # Estimate noise level from finest detail coefficients
            sigma = _estimate_sigma(coeffs[-1])

            # Universal threshold
            threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(y)))

            # Threshold detail coefficients
            coeffs_thresh = [coeffs[0]]  # Keep approximation
            for detail in coeffs[1:]:
                if threshold_mode == 'soft':
                    detail_thresh = pywt.threshold(detail, threshold, mode='soft')
                else:
                    detail_thresh = pywt.threshold(detail, threshold, mode='hard')
                coeffs_thresh.append(detail_thresh)

            # Reconstruct
            denoised = pywt.waverec(coeffs_thresh, wavelet)

            # Handle length mismatch
            if len(denoised) > len(y):
                denoised = denoised[:len(y)]
            elif len(denoised) < len(y):
                denoised = np.pad(denoised, (0, len(y) - len(denoised)), mode='edge')

            # Map back to original index
            result = pd.Series(np.nan, index=series.index)
            result.loc[valid_mask] = denoised

            return result

        except _NUMERICAL_FAILURES:
            # Soft-fail to NaN on degenerate-but-valid input only; do not
            # swallow programming errors (a bare `except Exception` would
            # hide TypeError/AttributeError from bad column wiring).
            result = pd.Series(np.nan, index=series.index)
            return result

    result = values.groupby(asset_col, sort=False)[value_col].apply(_wavelet_denoise_single)

    # Realign by label, not position — see decomposition.cycle.bandpass_filter
    # for why a positional index reassignment mislabels interleaved layouts.
    if isinstance(result.index, pd.MultiIndex):
        result = result.droplevel(0)
    result = result.loc[values.index].copy()
    result.index = values.index

    return result


def _estimate_sigma(detail_coeffs: np.ndarray) -> float:
    """
    Estimate noise standard deviation from detail coefficients.

    Uses median absolute deviation (MAD) estimator:
    sigma = MAD / 0.6745
    """
    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    sigma = mad / 0.6745
    return sigma


def _reconstruct_to_length(
    approx_coeffs: np.ndarray,
    target_length: int,
    wavelet: str,
    level: int,
    mode: str,
) -> np.ndarray:
    """Reconstruct approximation coefficients to original length."""
    # Create zero details and reconstruct
    coeffs = [approx_coeffs]
    current_len = len(approx_coeffs)

    for i in range(level):
        # Each level roughly doubles the length
        current_len = current_len * 2
        coeffs.append(np.zeros(min(current_len, target_length)))

    reconstructed = pywt.waverec(coeffs, wavelet, mode=mode)

    # Trim or pad to target length
    if len(reconstructed) > target_length:
        return reconstructed[:target_length]
    elif len(reconstructed) < target_length:
        return np.pad(reconstructed, (0, target_length - len(reconstructed)), mode='edge')
    return reconstructed


def _reconstruct_detail_to_length(
    coeffs: List[np.ndarray],
    detail_index: int,
    target_length: int,
    wavelet: str,
    mode: str,
) -> np.ndarray:
    """Reconstruct a single detail level to original length."""
    # Create coeffs list with zeros except for the target detail
    level = len(coeffs) - 1
    recon_coeffs = [np.zeros_like(coeffs[0])]  # Zero approximation

    for i in range(1, len(coeffs)):
        if i == detail_index:
            recon_coeffs.append(coeffs[i])
        else:
            recon_coeffs.append(np.zeros_like(coeffs[i]))

    reconstructed = pywt.waverec(recon_coeffs, wavelet, mode=mode)

    # Trim or pad to target length
    if len(reconstructed) > target_length:
        return reconstructed[:target_length]
    elif len(reconstructed) < target_length:
        return np.pad(reconstructed, (0, target_length - len(reconstructed)), mode='edge')
    return reconstructed
