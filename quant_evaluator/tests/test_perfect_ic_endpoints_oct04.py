"""Exact endpoint coefficients must not manufacture gigantic ICIR from roundoff."""
import numpy as np
import pytest
from quant_evaluator.metrics.correlation_endpoint import certify_correlation_endpoint
from quant_evaluator.api.factor_source import evaluate_factor_source_batch
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_all24_oracle import REFERENCE_METRICS, reference_source_all24
from test_source_all24_native_parity_oct04 import _fixture, _ArraySource


@pytest.mark.parametrize("sign", [1, -1])
def test_endpoint_is_certified_by_exact_affinity_not_tolerance(sign):
    x = np.arange(1, 101, dtype=np.float64)
    y = sign * (3 * x + 7)
    rounded = sign * np.nextafter(1.0, 0.0)
    assert certify_correlation_endpoint(rounded, x, y) == float(sign)


def test_near_affinity_is_not_coerced_to_an_exact_endpoint():
    x = np.arange(1, 101, dtype=np.float64)
    y = 3 * x + 7
    y[-1] = np.nextafter(y[-1], np.inf)
    rounded = np.nextafter(1.0, 0.0)
    assert certify_correlation_endpoint(rounded, x, y) == rounded


def _perfect_inputs(dtype, affine=False):
    axes, _ = _fixture(dtype)
    rng = np.random.default_rng(81004)
    if affine:
        y = np.stack([rng.permutation(100) for _ in range(65)]).astype(dtype)
    else:
        y = rng.normal(size=(65, 100)).astype(dtype)
    valid = rng.random(y.shape) > .2
    values = np.stack((3 * y + 7, 7 - 3 * y) if affine else (y, -y), axis=2)
    batch = FactorBatch(("perfect_positive", "perfect_negative"), axes.time_axis,
                        axes.asset_axis, values,
                        validity=np.repeat(valid[:, :, None], 2, axis=2))
    times = tuple(axes.time_axis.values)
    labels = LabelBundle("perfect_target", y, 1, decision_time=times,
                         label_start_time=times,
                         label_end_time=tuple(axes.time_axis.values + 1),
                         validity=valid, asset_axis=axes.asset_axis)
    return _ArraySource(batch), labels


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["cpu", "cuda_strict", "independent"])
@pytest.mark.parametrize("affine", [False, True], ids=["signed_identity", "exact_affine"])
def test_perfect_signed_factor_has_constant_ic_and_undefined_ir(dtype, backend, affine):
    # Varying pairwise missingness changes rounding of normalization. That must
    # not turn a mathematically constant +/-1 series into a 10^16 ICIR.
    if backend == "cuda_strict":
        pytest.importorskip("cupy")
    source, labels = _perfect_inputs(dtype, affine=affine)
    if backend == "independent":
        out = reference_source_all24(source, labels)
    else:
        out = evaluate_factor_source_batch(source, labels, metrics=REFERENCE_METRICS,
                                          backend=backend, max_tile_size=2)
    expected = np.broadcast_to(np.array([1., -1.]), (65, 2))
    for metric in ("rank_ic_series", "pearson_ic_series"):
        np.testing.assert_array_equal(out.series_metrics[metric], expected, err_msg=metric)
        np.testing.assert_array_equal(out.observation_counts[metric], [65, 65])
    for metric in ("ic_std", "pearson_ic_std"):
        np.testing.assert_array_equal(out.scalar_metrics[metric], [0., 0.], err_msg=metric)
    for metric in ("ic_ir", "pearson_ic_ir"):
        assert np.isnan(out.scalar_metrics[metric]).all(), metric


@pytest.mark.parametrize("backend", ["exact", "numba", "polars", "gpu"])
@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_explicit_daily_ic_backends_share_exact_endpoint_contract(backend, method, dtype):
    from quant_evaluator.metrics.ic import compute_daily_ic
    source, labels = _perfect_inputs(dtype, affine=True)
    batch = source.read_tile(0, 2).batch
    values, counts = compute_daily_ic(batch, labels, method=method, backend=backend)
    np.testing.assert_array_equal(values, np.broadcast_to([1., -1.], (65, 2)))
    expected_counts = np.sum(batch.validity[:, :, 0] & labels.validity, axis=1)
    np.testing.assert_array_equal(counts, np.repeat(expected_counts[:, None], 2, axis=1))
    np.testing.assert_array_equal(np.std(values, axis=0, ddof=1), [0., 0.])
