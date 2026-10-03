"""Independent CUDA soundness audit for the bounded-Q finance guard."""
from fractions import Fraction

import numpy as np
import pytest

from quant_evaluator.kernels.gpu import quantile_finance_guard as finance_guard


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device unavailable")
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(f"CUDA runtime unavailable: {exc}")
    return cp


def _fraction_mean(values):
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return np.nan
    exact_total = sum((Fraction.from_float(v) for v in finite), Fraction())
    return float(exact_total / len(finite))


def _finite_counts(bucket, values, n_quantiles):
    return np.stack([
        np.sum((bucket == q) & np.isfinite(values), axis=1, dtype=np.int64)
        for q in range(n_quantiles)
    ], axis=1)


def _fast_cupy_means(cp, bucket, values, n_quantiles):
    bucket_gpu = cp.asarray(bucket, dtype=cp.int32)
    values_gpu = cp.asarray(values, dtype=cp.float64)
    q_ids = cp.arange(n_quantiles, dtype=cp.int32)[None, None, :]
    finite = cp.isfinite(values_gpu)[:, :, None]
    selected = (bucket_gpu[:, :, None] == q_ids) & finite
    counts = cp.sum(selected, axis=1, dtype=cp.int64)
    terms = cp.where(selected, values_gpu[:, :, None], 0.0)
    sums = cp.sum(terms, axis=1, dtype=cp.float64)
    means = cp.where(counts > 0, sums / cp.maximum(counts, 1), cp.nan)
    return bucket_gpu, values_gpu, counts, means


def _run_guard(cp, bucket, values, counts, means, n_quantiles, *, min_assets=1):
    rows, ncols = bucket.shape
    risk = cp.empty((rows * n_quantiles,), dtype=cp.uint8)
    error = cp.zeros((), dtype=cp.int32)
    kernel = finance_guard.compile_finance_risk_guard(cp)
    kernel((rows,), (1,), (
        cp.asarray(bucket, dtype=cp.int32),
        cp.asarray(values, dtype=cp.float64),
        cp.asarray(counts, dtype=cp.int64),
        cp.asarray(means, dtype=cp.float64),
        risk, error, rows, ncols, n_quantiles, min_assets,
    ))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(risk).reshape(rows, n_quantiles), int(error.item())


def test_q1_q2_q5_q20_q32_guard_never_marks_inaccurate_fast_means_safe():
    cp = _cuda()
    epsilon = np.finfo(np.float64).eps
    for n_quantiles in (1, 2, 5, 20, 32):
        ncols = 256
        rng = np.random.default_rng(20261004 + n_quantiles)
        bucket = np.tile(
            np.arange(n_quantiles, dtype=np.int32),
            (ncols + n_quantiles - 1) // n_quantiles,
        )[:ncols]
        rng.shuffle(bucket)
        # Seeded adversarial cancellation values and shuffled bucket membership.
        basis = np.array([-1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0])
        base = rng.choice(basis, size=ncols)
        mean_abs = float(np.mean(np.abs(base)))
        threshold_scale = 1e-12 / (
            2.0 * ncols * epsilon * n_quantiles * mean_abs
        )
        scales = np.array([
            0.25, 0.99, 1.01,
            threshold_scale * (1.0 - 1e-6),
            threshold_scale * (1.0 + 1e-6),
        ])
        values = base[None, :] * scales[:, None]
        bucket_rows = np.broadcast_to(bucket, values.shape).copy()

        bucket_gpu, values_gpu, counts_gpu, means_gpu = _fast_cupy_means(
            cp, bucket_rows, values, n_quantiles,
        )
        counts = cp.asnumpy(counts_gpu)
        means = cp.asnumpy(means_gpu)
        expected_counts = _finite_counts(bucket_rows, values, n_quantiles)
        np.testing.assert_array_equal(counts, expected_counts)
        risks, error = _run_guard(
            cp, cp.asnumpy(bucket_gpu), cp.asnumpy(values_gpu),
            counts, means, n_quantiles,
        )
        assert error == 0

        for row in range(values.shape[0]):
            for q in range(n_quantiles):
                selected = values[row, (bucket_rows[row] == q)]
                exact = _fraction_mean(selected)
                actual = means[row, q]
                if counts[row, q] == 0:
                    assert np.isnan(actual)
                    assert risks[row, q] == 0
                elif risks[row, q] == 0:
                    assert np.isfinite(actual)
                    assert abs(actual - exact) <= 1e-12, (
                        f"false-safe Q={n_quantiles}, row={row}, q={q}: "
                        f"actual={actual!r}, exact={exact!r}"
                    )


def test_q5_guard_forces_overflow_subnormal_mixedscale_nan_and_count_mismatch():
    cp = _cuda()
    rows, ncols, n_quantiles = 5, 64, 5
    bucket = np.full((rows, ncols), -1, dtype=np.int32)
    values = np.full((rows, ncols), np.nan, dtype=np.float64)

    bucket[0] = 0
    values[0] = np.finfo(np.float64).max  # raw fast sum overflows

    unit_subnormal = float.fromhex("0x0.0000000000001p-1022")
    bucket[1] = 0
    values[1] = unit_subnormal

    mixed = np.array([1.0, 2.0**-53, 0.0, 1e4, -1e4, 0.0])
    bucket[2, :len(mixed)] = 0
    values[2, :len(mixed)] = mixed

    bucket[3] = 0
    values[3] = 0.02

    bucket[4] = 0
    values[4] = 0.02

    bucket_gpu, values_gpu, counts_gpu, means_gpu = _fast_cupy_means(
        cp, bucket, values, n_quantiles,
    )
    counts = cp.asnumpy(counts_gpu)
    means = cp.asnumpy(means_gpu)
    means[3, 0] = np.nan  # selected mean metadata must fail closed
    counts[4, 0] -= 1  # deliberately inconsistent with finite bucket members

    risks, error = _run_guard(
        cp, cp.asnumpy(bucket_gpu), cp.asnumpy(values_gpu),
        counts, means, n_quantiles,
    )
    assert risks[0, 0] == 1  # overflow
    assert risks[1, 0] == 1  # subnormal
    assert risks[2, 0] == 1  # mixed-scale cancellation
    assert risks[3, 0] == 1  # selected mean is nonfinite
    assert error == 1  # the finite-member count disagrees with metadata

    # Compare actual CuPy means with a sum of exact input Float64 rationals.
    assert np.isinf(means[0, 0])
    assert abs(means[1, 0] - _fraction_mean(values[1])) <= 1e-12
    assert abs(means[2, 0] - _fraction_mean(mixed)) <= 1e-12
