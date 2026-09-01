"""
Time Series Decomposition Transforms

This package provides causal time series decomposition methods for factor preprocessing:

- trend: HP filter and Hodrick-Prescott decomposition
- seasonal: STL seasonal decomposition
- cycle: Bandpass filters and cycle extraction (including Christiano-Fitzgerald)
- wavelet: Wavelet decomposition, smoothing, and denoising

All methods are causal (use only past data via shift(1)) and handle multiple assets independently.
"""

from factor_preprocess.transforms.decomposition.trend import (
    hp_filter,
    hp_decompose,
)

from factor_preprocess.transforms.decomposition.seasonal import (
    stl_decompose,
    seasonal_component,
    trend_component,
    residual_component,
)

from factor_preprocess.transforms.decomposition.cycle import (
    bandpass_filter,
    extract_cycle,
    christiano_fitzgerald_filter,
)

try:
    from factor_preprocess.transforms.decomposition.wavelet import (
        wavelet_decompose,
        wavelet_smooth,
        wavelet_denoise,
    )
except ImportError:  # pragma: no cover - PyWavelets is an optional heavy dep
    wavelet_decompose = None  # type: ignore[assignment]
    wavelet_smooth = None  # type: ignore[assignment]
    wavelet_denoise = None  # type: ignore[assignment]

__all__ = [
    # Trend
    "hp_filter",
    "hp_decompose",
    # Seasonal
    "stl_decompose",
    "seasonal_component",
    "trend_component",
    "residual_component",
    # Cycle
    "bandpass_filter",
    "extract_cycle",
    "christiano_fitzgerald_filter",
    # Wavelet
    "wavelet_decompose",
    "wavelet_smooth",
    "wavelet_denoise",
]
