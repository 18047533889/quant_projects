"""Fused GPU rolling mean and information-ratio statistics for daily IC."""
from __future__ import annotations

from numbers import Integral
import numpy as np

_KERNEL = r"""
extern "C" __global__
void rolling_ic_stats(const double* x, double* means, double* irs,
                      const long long T, const long long F,
                      const long long W, const long long minp) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long total = T * F;
    if (idx >= total) return;
    const long long t = idx / F;
    const long long f = idx - t * F;
    const long long first = t + 1 > W ? t + 1 - W : 0;
    long long n = 0;
    double sum = 0.0;
    for (long long j = first; j <= t; ++j) {
        const double v = x[j * F + f];
        if (isfinite(v)) { ++n; sum += v; }
    }
    if (n < minp) { means[idx] = nan(""); irs[idx] = nan(""); return; }
    const double mean = sum / (double)n;
    means[idx] = mean;
    if (n < 2) { irs[idx] = nan(""); return; }
    double ss = 0.0;
    for (long long j = first; j <= t; ++j) {
        const double v = x[j * F + f];
        if (isfinite(v)) { const double d = v - mean; ss += d * d; }
    }
    const double sd = sqrt(ss / (double)(n - 1));
    irs[idx] = (sd > 1.0e-12) ? mean / sd : nan("");
}
"""

_compiled_kernel = None


def rolling_ic_mean_ir(ic, window=60, min_periods=20):
    """Return rolling means and IRs with O(TF) outputs and no window cube."""
    def positive_integer(name, value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    window = positive_integer("window", window)
    min_periods = positive_integer("min_periods", min_periods)
    import cupy as cp

    global _compiled_kernel
    if _compiled_kernel is None:
        _compiled_kernel = cp.RawKernel(_KERNEL, "rolling_ic_stats")
    data = cp.ascontiguousarray(ic, dtype=cp.float64)
    if data.ndim != 2:
        raise ValueError("ic must be a two-dimensional (T, F) array")
    T, F = data.shape
    means = cp.full((T, F), cp.nan, dtype=cp.float64)
    irs = cp.full((T, F), cp.nan, dtype=cp.float64)
    total = T * F
    if total:
        threads = 256
        kernel_window = min(window, T)
        kernel_min_periods = min(min_periods, T + 1)
        _compiled_kernel(((total + threads - 1) // threads,), (threads,),
                         (data, means, irs, np.int64(T), np.int64(F),
                          np.int64(kernel_window), np.int64(kernel_min_periods)))
    return means, irs
