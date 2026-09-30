"""Device-only average ranks for sorted finite equal-value runs."""
_RUN_KERNEL = None


def sorted_average_ranks(sorted_values, cp):
    """Find tied run bounds on device; singleton ranks need no binary search.

    The caller replaces nonfinite inputs by +Inf before sorting. The final
    scatter masks invalid positions. No device scalar is read on the host.
    """
    global _RUN_KERNEL
    if _RUN_KERNEL is None:
        _RUN_KERNEL = cp.ElementwiseKernel(
            "raw T values, int64 width", "float64 ranks",
            r'''
            const long long column = i % width;
            const long long base = i - column;
            const T value = values[i];
            if (!isfinite((double)value)) {
                ranks = nan("");
            } else if ((column == 0 || values[i - 1] != value) &&
                       (column + 1 == width || values[i + 1] != value)) {
                ranks = (double)column + 1.0;
            } else {
                long long lo = 0, hi = column;
                while (lo < hi) {
                    const long long mid = lo + (hi - lo) / 2;
                    if (values[base + mid] < value) lo = mid + 1;
                    else hi = mid;
                }
                const long long first = lo;
                lo = column + 1;
                hi = width;
                while (lo < hi) {
                    const long long mid = lo + (hi - lo) / 2;
                    if (values[base + mid] <= value) lo = mid + 1;
                    else hi = mid;
                }
                ranks = ((double)first + (double)lo + 1.0) * 0.5;
            }
            ''',
            "qe_sorted_average_run_rank_v1",
        )
    rows, width = sorted_values.shape
    if rows == 0 or width == 0:
        return cp.empty(sorted_values.shape, dtype=cp.float64)
    return _RUN_KERNEL(sorted_values, width, size=rows * width).reshape(rows, width)
