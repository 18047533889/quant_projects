import numpy as np
import pytest
from types import SimpleNamespace

cp = pytest.importorskip("cupy")

import quant_evaluator.kernels.gpu.correlation as correlation
from quant_evaluator.kernels.gpu.pearson_centered_fused import fused_centered_sums


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("factor_specific_y", [False, True])
def test_fused_centered_pearson_noncontiguous_pairwise_rows(dtype, factor_specific_y):
    rng = np.random.default_rng(61003)
    t, f, n = 4, 3, 193
    x_source = rng.normal(size=(t, f, 2 * n)).astype(dtype)
    x_host = x_source[:, :, ::2]
    y_source = rng.normal(size=(t, 2 * n)).astype(dtype)
    y_host = y_source[:, ::2]
    x_host[rng.random(x_host.shape) < 0.12] = np.nan
    y_host[rng.random(y_host.shape) < 0.09] = np.inf
    x = cp.asarray(x_source)[:, :, ::2]
    y = cp.asarray(y_source)[:, ::2]
    if factor_specific_y:
        y = cp.broadcast_to(y[:, None, :], (t, f, n)).copy()
        y_host = np.broadcast_to(y_host[:, None, :], (t, f, n))

    yb = y if y.ndim == 3 else y[:, None, :]
    finite = cp.isfinite(x) & cp.isfinite(yb)
    count, sx, sy, mx, my, vx, vy, cov = fused_centered_sums(x, yb, finite)
    actual, actual_counts = correlation.batched_pearson_ic(x, y, min_obs=2)
    host_count = cp.asnumpy(count)
    np.testing.assert_array_equal(cp.asnumpy(actual_counts), host_count)
    outputs = [cp.asnumpy(value) for value in (sx, sy, mx, my, vx, vy, cov)]
    host_x = np.asarray(x_host, dtype=np.float64)
    host_y = np.asarray(y_host, dtype=np.float64)
    if host_y.ndim == 2:
        host_y = np.broadcast_to(host_y[:, None, :], (t, f, n))
    expected_ic = np.full((t, f), np.nan)
    for ti in range(t):
        for fi in range(f):
            valid = np.isfinite(host_x[ti, fi]) & np.isfinite(host_y[ti, fi])
            a, b = host_x[ti, fi, valid], host_y[ti, fi, valid]
            expected = (a.sum(), b.sum(), a.mean(), b.mean(),
                        np.sum((a-a.mean())**2), np.sum((b-b.mean())**2),
                        np.sum((a-a.mean())*(b-b.mean())))
            for actual_stat, wanted in zip(outputs, expected):
                assert actual_stat[ti, fi] == pytest.approx(wanted, rel=2e-12, abs=2e-12)
            if a.size >= 2 and expected[4] > 0 and expected[5] > 0:
                expected_ic[ti, fi] = expected[6] / np.sqrt(expected[4] * expected[5])
    actual_host = cp.asnumpy(actual)
    max_absolute_error = float(np.nanmax(np.abs(actual_host - expected_ic)))
    assert max_absolute_error <= 1e-10 + 1e-8 * float(np.nanmax(np.abs(expected_ic)))
    np.testing.assert_allclose(actual_host, expected_ic, rtol=1e-8, atol=1e-10, equal_nan=True)


def test_fused_centered_mask_contract_and_noncontiguous_bool_mask():
    x_source = cp.arange(96, dtype=cp.float64).reshape(1, 2, 48)
    x = x_source[:, :, ::2]
    y = cp.arange(48, dtype=cp.float64)[None, None, ::2]
    mask_storage = cp.ones((1, 2, 48), dtype=cp.bool_)
    finite = mask_storage[:, :, ::2]
    stats = fused_centered_sums(x, y, finite)
    assert cp.asnumpy(stats[0]).tolist() == [[24, 24]]
    with pytest.raises(TypeError, match="boolean finite mask"):
        fused_centered_sums(x, y, finite.astype(cp.uint8))


def test_fused_centered_rejects_row_count_beyond_cuda_grid_limit():
    shape = (1, 0x80000000, 1)
    x = SimpleNamespace(ndim=3, shape=shape, dtype=cp.dtype(cp.float64))
    y = SimpleNamespace(ndim=3, shape=(1, 1, 1), dtype=cp.dtype(cp.float64))
    finite = SimpleNamespace(shape=shape, dtype=cp.dtype(cp.bool_))
    with pytest.raises(ValueError, match="grid/index limit"):
        fused_centered_sums(x, y, finite)


def test_fused_centered_negative_stride_and_mixed_float_dtypes():
    rng = np.random.default_rng(61004)
    x_host = rng.normal(size=(2, 3, 67)).astype(np.float32)
    y_host = rng.normal(size=(2, 67)).astype(np.float64)
    x = cp.asarray(x_host)[:, :, ::-1]
    y = cp.asarray(y_host)[:, ::-1]
    assert x.strides[2] < 0 and y.strides[1] < 0
    actual, counts = correlation.batched_pearson_ic(x, y, min_obs=2)
    expected = np.empty((2, 3), dtype=np.float64)
    for ti in range(2):
        for fi in range(3):
            expected[ti, fi] = np.corrcoef(x_host[ti, fi], y_host[ti])[0, 1]
    np.testing.assert_array_equal(cp.asnumpy(counts), np.full((2, 3), 67, dtype=np.int32))
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("shape", [(0, 2, 64), (3, 0, 64), (2, 3, 0)])
def test_fused_centered_empty_rows_and_empty_asset_axis(shape):
    x = cp.empty(shape, dtype=cp.float64)
    y = cp.empty((shape[0], shape[2]), dtype=cp.float64)
    actual, counts = correlation.batched_pearson_ic(x, y, min_obs=2)
    assert actual.shape == shape[:2]
    assert counts.shape == shape[:2]
    if actual.size:
        assert bool(cp.all(cp.isnan(actual)))
        assert bool(cp.all(counts == 0))


@pytest.mark.parametrize("n_factors", [32, 48])
def test_public_pearson_chain_cpu_cuda_parity_with_missing_and_min_obs(n_factors):
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate

    rng = np.random.default_rng(61005 + n_factors)
    t, n = 8, 64
    values = rng.normal(size=(t, n, n_factors))
    labels = rng.normal(size=(t, n))
    validity = rng.random((t, n, n_factors)) > 0.13
    label_validity = rng.random((t, n)) > 0.08
    # Pin two threshold rows: one below min_assets and one exactly at it.
    validity[0] = False
    label_validity[0] = True
    validity[0, :19, 0] = True
    validity[0, :20, 1] = True
    values[0, 19:, 0] = np.nan
    values[0, 20:, 1] = np.nan
    values[1, ::11, 2] = np.nan
    labels[2, ::13] = np.inf
    times = np.arange(t, dtype=np.int64)
    assets = np.arange(n, dtype=np.int64)
    asset_axis = AxisRef("asset", "int64", n, assets)
    batch = FactorBatch(
        tuple(f"f{i}" for i in range(n_factors)),
        AxisRef("time", "int64", t, times), asset_axis, values,
        validity=validity,
    )
    label = LabelBundle(
        "ret", labels, 1, decision_time=times, label_start_time=times,
        label_end_time=times + 1, asset_axis=asset_axis, validity=label_validity,
    )
    metrics = ("pearson_ic_series", "pearson_ic", "pearson_ic_std", "pearson_ic_ir")
    parameters = {name: {"min_assets": 20} for name in metrics}
    parameters["pearson_ic_ir"]["min_periods"] = 3
    cpu = evaluate(batch, label, backend="cpu", metrics=metrics, metric_parameters=parameters)
    gpu = evaluate(batch, label, backend="cuda_strict", metrics=metrics, metric_parameters=parameters)

    max_absolute_error = 0.0
    for metric in metrics:
        left, right = cpu.artifacts[metric], gpu.artifacts[metric]
        np.testing.assert_array_equal(np.isfinite(right.values), np.isfinite(left.values))
        difference = np.abs(np.asarray(right.values) - np.asarray(left.values))
        finite_difference = difference[np.isfinite(difference)]
        if finite_difference.size:
            max_absolute_error = max(max_absolute_error, float(finite_difference.max()))
        np.testing.assert_allclose(right.values, left.values, rtol=1e-8, atol=1e-10, equal_nan=True)
        left_counts = left.provenance.get("observation_counts")
        right_counts = right.provenance.get("observation_counts")
        if left_counts is not None or right_counts is not None:
            assert left_counts is not None and right_counts is not None
            np.testing.assert_array_equal(right_counts, left_counts)
    threshold_counts = np.sum(
        validity[0] & np.isfinite(values[0])
        & (label_validity[0] & np.isfinite(labels[0]))[:, None], axis=0,
    )
    assert int(threshold_counts[0]) == 19
    assert int(threshold_counts[1]) == 20
    assert not np.isfinite(cpu.artifacts["pearson_ic_series"].values[0, 0])
    assert np.isfinite(cpu.artifacts["pearson_ic_series"].values[0, 1])
    assert max_absolute_error <= 1e-10 + 1e-8 * max(
        1.0, float(np.nanmax(np.abs(cpu.artifacts["pearson_ic_series"].values)))
    )


def test_bounded_spearman_and_float16_pearson_keep_legacy_reductions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported/bounded paths must retain legacy reductions")

    monkeypatch.setattr(correlation, "fused_centered_sums", forbidden)
    x16 = cp.arange(40, dtype=cp.float16)[None, None, :]
    y16 = cp.arange(40, dtype=cp.float16)[None, :]
    ic, counts = correlation.batched_pearson_ic(x16, y16, min_obs=2)
    assert float(ic[0, 0]) == pytest.approx(1.0)
    assert int(counts[0, 0]) == 40

    values = cp.arange(40, dtype=cp.float64)[None, None, :]
    rank_ic, rank_counts = correlation.batched_spearman_ic(values, values[:, 0, :], min_obs=2)
    assert float(rank_ic[0, 0]) == pytest.approx(1.0)
    assert int(rank_counts[0, 0]) == 40
