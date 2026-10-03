"""Fused centered reductions for ordinary GPU Pearson rows."""
from __future__ import annotations


_KERNEL = r'''
__device__ __forceinline__ double load_value(const char* base, long long offset,
                                              int dtype_bytes) {
    if (dtype_bytes == 4) return (double)(*((const float*)(base + offset)));
    return *((const double*)(base + offset));
}

extern "C" __global__
void pearson_centered_fused(const char* x, const char* y,
                            const unsigned char* finite,
                            long long n, int nf, int y_broadcast,
                            long long xs0, long long xs1, long long xs2,
                            long long ys0, long long ys1, long long ys2,
                            int xbytes, int ybytes,
                            long long* count_out, double* sx_out, double* sy_out,
                            double* mx_out, double* my_out,
                            double* vx_out, double* vy_out, double* cov_out) {
    const int row = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int t = row / nf, f = row - t * nf;
    const int yf = y_broadcast ? 0 : f;
    const long long xbase = (long long)t * xs0 + (long long)f * xs1;
    const long long ybase = (long long)t * ys0 + (long long)yf * ys1;
    const long long maskbase = (long long)row * n;
    __shared__ double a[128], b[128], meanx, meany, sumx, sumy;
    __shared__ long long q[128];

    double sx = 0.0, sy = 0.0;
    long long cnt = 0;
    for (long long i = lane; i < n; i += 128) {
        if (finite[maskbase + i]) {
            sx += load_value(x, xbase + (long long)i * xs2, xbytes);
            sy += load_value(y, ybase + (long long)i * ys2, ybytes);
            ++cnt;
        }
    }
    a[lane] = sx; b[lane] = sy; q[lane] = cnt;
    __syncthreads();
    for (int d = 64; d > 0; d >>= 1) {
        if (lane < d) { a[lane] += a[lane+d]; b[lane] += b[lane+d]; q[lane] += q[lane+d]; }
        __syncthreads();
    }
    if (lane == 0) {
        const long long safe_count = q[0] > 0 ? q[0] : 1;
        sumx = a[0]; sumy = b[0];
        meanx = a[0] / (double)safe_count;
        meany = b[0] / (double)safe_count;
    }
    __syncthreads();

    double vx = 0.0, vy = 0.0, cov = 0.0;
    for (long long i = lane; i < n; i += 128) {
        if (finite[maskbase + i]) {
            const double dx = load_value(x, xbase + (long long)i * xs2, xbytes) - meanx;
            const double dy = load_value(y, ybase + (long long)i * ys2, ybytes) - meany;
            vx += dx * dx; vy += dy * dy; cov += dx * dy;
        }
    }
    a[lane] = vx; b[lane] = vy;
    __shared__ double c[128];
    c[lane] = cov;
    __syncthreads();
    for (int d = 64; d > 0; d >>= 1) {
        if (lane < d) { a[lane] += a[lane+d]; b[lane] += b[lane+d]; c[lane] += c[lane+d]; }
        __syncthreads();
    }
    if (lane == 0) {
        count_out[row] = q[0]; sx_out[row] = sumx; sy_out[row] = sumy;
        mx_out[row] = meanx; my_out[row] = meany;
        vx_out[row] = a[0]; vy_out[row] = b[0]; cov_out[row] = c[0];
    }
}
'''


def fused_centered_sums(x, y, finite):
    """Return count, sums, means, variances and covariance for each (T,F) row.

    Inputs must be float32/float64 CuPy arrays and y must already have shape
    (T,1,N) or (T,F,N). Strides are passed to the kernel, so views are supported.
    """
    import cupy as cp

    if x.ndim != 3 or y.ndim != 3 or finite.shape != x.shape:
        raise ValueError("fused Pearson reduction expects x/y (T,F,N) and a matching mask")
    if finite.dtype != cp.dtype(cp.bool_):
        raise TypeError("fused Pearson reduction requires a boolean finite mask")
    if y.shape[0] != x.shape[0] or y.shape[2] != x.shape[2] or y.shape[1] not in (1, x.shape[1]):
        raise ValueError("fused Pearson reduction received incompatible y shape")
    if x.dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)) or y.dtype not in (cp.dtype(cp.float32), cp.dtype(cp.float64)):
        raise TypeError("fused Pearson reduction supports only float32/float64 inputs")
    shape = x.shape[:2]
    rows = int(shape[0] * shape[1])
    if rows > 0x7fffffff:
        raise ValueError("fused Pearson row count exceeds the CUDA grid/index limit")
    count = cp.empty(shape, dtype=cp.int64)
    sx = cp.empty(shape, dtype=cp.float64)
    sy = cp.empty(shape, dtype=cp.float64)
    mx = cp.empty(shape, dtype=cp.float64)
    my = cp.empty(shape, dtype=cp.float64)
    vx = cp.empty(shape, dtype=cp.float64)
    vy = cp.empty(shape, dtype=cp.float64)
    cov = cp.empty(shape, dtype=cp.float64)
    if rows:
        mask = finite if finite.flags.c_contiguous else cp.ascontiguousarray(finite)
        _kernel()((rows,), (128,), (
            x, y, mask, cp.int64(x.shape[2]), cp.int32(x.shape[1]),
            cp.int32(y.shape[1] == 1),
            cp.int64(x.strides[0]), cp.int64(x.strides[1]), cp.int64(x.strides[2]),
            cp.int64(y.strides[0]), cp.int64(y.strides[1]), cp.int64(y.strides[2]),
            cp.int32(x.dtype.itemsize), cp.int32(y.dtype.itemsize),
            count, sx, sy, mx, my, vx, vy, cov,
        ))
    return count, sx, sy, mx, my, vx, vy, cov


def _kernel():
    import cupy as cp
    global _COMPILED_KERNEL
    try:
        return _COMPILED_KERNEL
    except NameError:
        _COMPILED_KERNEL = cp.RawKernel(_KERNEL, "pearson_centered_fused", options=("--std=c++11",))
        return _COMPILED_KERNEL
