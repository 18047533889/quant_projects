"""Independent high-precision oracles for quantile-bucket label means.

These cases catch float64 bucket-sum overflow and loss of low-order terms even
when two optimized backends happen to agree with each other.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from itertools import permutations

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quantile import (
    _aggregate_quantile_assignments,
    assign_quantiles_batch,
    compute_quantile_returns_fast,
    compute_quantile_returns_optimized,
)


def _decimal_mean(values):
    finite = [float(value) for value in values if np.isfinite(value)]
    assert finite
    with localcontext() as context:
        context.prec = 2000
        total = sum((Decimal.from_float(value) for value in finite), Decimal(0))
        return float(total / Decimal(len(finite)))


def _mean_cases():
    min_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    max_subnormal = np.nextafter(np.finfo(np.float64).tiny, np.float64(0.0))
    return [
        pytest.param(np.array([1e308, 1e308]), id="repeated-huge-finite"),
        pytest.param(np.array([1e308, 1e308, -1e308, -1e308]),
                     id="positive-negative-overflow-cancellation"),
        pytest.param(np.array([1e308, 1.0, -1e308, 3.0]),
                     id="preserve-low-order-cancellation-terms"),
        pytest.param(np.array([max_subnormal, 0.0]), id="largest-subnormal-mean"),
        pytest.param(np.array([min_subnormal, 0.0]), id="smallest-subnormal-mean"),
    ]


@pytest.mark.parametrize("labels", _mean_cases())
def test_assignment_aggregator_matches_decimal_mean_for_finite_float64_labels(labels):
    count = labels.size
    assignments = np.zeros((1, count, 1), dtype=np.int32)
    returns, counts = _aggregate_quantile_assignments(
        assignments, labels.reshape(1, count), n_quantiles=1,
        min_assets=count,
    )

    assert counts[0, 0, 0] == count
    expected = _decimal_mean(labels)
    assert np.isfinite(returns[0, 0, 0])
    assert returns[0, 0, 0] == expected


def _batch_and_label(labels, *, factor_validity=None, label_validity=None):
    labels = np.asarray(labels, dtype=np.float64)
    n = labels.size
    times = AxisRef("time", "int", 1, np.array([0]))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    values = np.arange(n, dtype=np.float64).reshape(1, n, 1)
    batch = FactorBatch(
        ("f0",), times, assets, values,
        validity=None if factor_validity is None else np.asarray(factor_validity, dtype=bool).reshape(1, n, 1),
    )
    bundle = LabelBundle(
        "decimal-quantile-oracle", labels.reshape(1, n), 1,
        validity=None if label_validity is None else np.asarray(label_validity, dtype=bool).reshape(1, n),
        decision_time=(0,), label_start_time=(1,), label_end_time=(2,),
        asset_axis=assets,
    )
    return batch, bundle


@pytest.mark.parametrize("labels", _mean_cases())
def test_public_numpy_quantile_path_matches_decimal_mean_for_finite_float64_labels(labels):
    batch, bundle = _batch_and_label(labels)
    returns, counts = compute_quantile_returns_fast(
        batch, bundle, n_quantiles=1, min_assets=labels.size, use_numba=False,
    )

    assert counts[0, 0, 0] == labels.size
    expected = _decimal_mean(labels)
    assert np.isfinite(returns[0, 0, 0])
    assert returns[0, 0, 0] == expected


def test_public_numpy_quantile_path_counts_masks_and_min_assets_before_mean():
    labels = np.array([1e308, 1e308, np.nan, 1e308, -1e308, 1e308])
    factor_validity = np.array([True, True, True, True, True, False])
    label_validity = np.array([True, True, False, True, False, True])
    batch, bundle = _batch_and_label(
        labels, factor_validity=factor_validity, label_validity=label_validity,
    )

    returns, counts = compute_quantile_returns_fast(
        batch, bundle, n_quantiles=2, min_assets=2, use_numba=False,
    )

    np.testing.assert_array_equal(counts[0, :, 0], np.array([2, 1], dtype=np.int32))
    assert returns[0, 0, 0] == _decimal_mean([1e308, 1e308])
    assert np.isnan(returns[0, 1, 0])


def _force_numba_python_implementations(monkeypatch):
    import quant_evaluator.metrics.quantile_numba as quantile_numba

    assign = quantile_numba._assign_quantiles_jit
    aggregate = quantile_numba._compute_quantile_returns_jit
    monkeypatch.setattr(quantile_numba, "_assign_quantiles_jit",
                        getattr(assign, "py_func", assign))
    monkeypatch.setattr(quantile_numba, "_compute_quantile_returns_jit",
                        getattr(aggregate, "py_func", aggregate))
    repair_calls = []
    repair = quantile_numba.repair_quantile_panel

    def observed_repair(*args, **kwargs):
        repair_calls.append(1)
        return repair(*args, **kwargs)

    monkeypatch.setattr(quantile_numba, "repair_quantile_panel", observed_repair)
    return repair_calls


def _run_path(path, labels, *, monkeypatch=None, factor_validity=None,
              label_validity=None, n_quantiles=1, min_assets=None):
    labels = np.asarray(labels, dtype=np.float64)
    min_assets = labels.size if min_assets is None else min_assets
    batch, bundle = _batch_and_label(
        labels, factor_validity=factor_validity, label_validity=label_validity,
    )
    if path == "aggregate":
        count = labels.size
        assignments = np.zeros((1, count, 1), dtype=np.int32)
        assignments[:, 0, :] = -1 if factor_validity is not None and not factor_validity[0] else 0
        return _aggregate_quantile_assignments(
            assignments, labels.reshape(1, count), n_quantiles=n_quantiles,
            min_assets=min_assets,
        )
    if path == "optimized_numpy":
        return compute_quantile_returns_optimized(
            batch, bundle, n_quantiles=n_quantiles, min_assets=min_assets,
        )
    if path == "numba_python":
        from quant_evaluator.metrics.quantile_numba import compute_quantile_returns_numba
        return compute_quantile_returns_numba(
            batch, bundle, n_quantiles=n_quantiles, min_assets=min_assets,
        )
    if path == "public_numpy":
        return compute_quantile_returns_fast(
            batch, bundle, n_quantiles=n_quantiles, min_assets=min_assets,
            use_numba=False,
        )
    raise AssertionError(f"unknown test path {path}")


@pytest.mark.parametrize("labels", _mean_cases())
def test_optimized_numpy_path_matches_decimal_mean_for_float64_labels(labels):
    batch, bundle = _batch_and_label(labels)
    returns, counts = compute_quantile_returns_optimized(
        batch, bundle, n_quantiles=1, min_assets=labels.size,
    )

    assert counts[0, 0, 0] == labels.size
    expected = _decimal_mean(labels)
    assert np.isfinite(returns[0, 0, 0])
    assert returns[0, 0, 0] == expected


@pytest.mark.parametrize("labels", _mean_cases())
def test_numba_wrapper_python_kernel_matches_decimal_mean_without_jit(labels, monkeypatch):
    repair_calls = _force_numba_python_implementations(monkeypatch)
    returns, counts = _run_path(
        "numba_python", labels, monkeypatch=monkeypatch,
        n_quantiles=1, min_assets=labels.size,
    )

    assert repair_calls == [1]
    assert counts[0, 0, 0] == labels.size
    expected = _decimal_mean(labels)
    assert np.isfinite(returns[0, 0, 0])
    assert returns[0, 0, 0] == expected


@pytest.mark.parametrize("path", ["aggregate", "optimized_numpy", "numba_python", "public_numpy"])
@pytest.mark.parametrize("labels", [
    np.array([1e308, 1.0, -1e308, 3.0]),
])
def test_all_cancellation_label_permutations_match_decimal_mean(path, labels, monkeypatch):
    """Bucket membership is fixed; input order must not change the Decimal mean."""
    repair_calls = (_force_numba_python_implementations(monkeypatch)
                    if path == "numba_python" else None)
    expected = _decimal_mean(labels)
    assert expected == 1.0
    for ordering in permutations(labels.tolist()):
        returns, counts = _run_path(
            path, np.asarray(ordering), monkeypatch=monkeypatch,
            n_quantiles=1, min_assets=4,
        )
        assert counts[0, 0, 0] == 4
        assert returns[0, 0, 0] == expected
    if repair_calls is not None:
        assert len(repair_calls) == 24


@pytest.mark.parametrize("path", ["optimized_numpy", "numba_python", "public_numpy"])
def test_optimized_and_numba_paths_preserve_masked_counts_and_min_assets(path, monkeypatch):
    labels = np.array([1e308, 1e308, np.nan, 1e308, -1e308, 1e308])
    factor_validity = np.array([True, True, True, True, True, False])
    label_validity = np.array([True, True, False, True, False, True])
    repair_calls = (_force_numba_python_implementations(monkeypatch)
                    if path == "numba_python" else None)
    result = _run_path(
        path, labels, monkeypatch=monkeypatch,
        factor_validity=factor_validity, label_validity=label_validity,
        n_quantiles=2, min_assets=2,
    )

    returns, counts = result
    np.testing.assert_array_equal(counts[0, :, 0], np.array([2, 1], dtype=np.int32))
    assert returns[0, 0, 0] == _decimal_mean([1e308, 1e308])
    assert np.isnan(returns[0, 1, 0])
    if repair_calls is not None:
        # The wrapper postprocessed the actual pure-Python kernel output.
        assert repair_calls == [1]


def test_mixed_scale_mean_exposes_fsum_then_divide_double_rounding():
    # The first Q bucket's exact sum is 1 + 2**-53, exactly halfway above 1.
    # fsum rounds that tie to 1.0 before division by 3; Decimal proves that
    # rounding the exact mean instead gives the next float above 1/3.
    labels = np.array([1.0, 2.0 ** -53, 0.0, 1e4, -1e4, 0.0])
    batch, bundle = _batch_and_label(labels)
    returns, counts = compute_quantile_returns_fast(
        batch, bundle, n_quantiles=2, min_assets=3, use_numba=False,
    )

    expected = _decimal_mean(labels[:3])
    assert expected == np.nextafter(np.float64(1.0 / 3.0), np.float64(np.inf))
    assert counts[0, 0, 0] == 3
    assert returns[0, 0, 0] == expected


@pytest.mark.parametrize("path", ["optimized_numpy", "numba_python", "public_numpy"])
def test_normal_financial_scale_buckets_stay_on_fast_path(path, monkeypatch):
    n_assets = 5461
    labels = np.linspace(-0.01, 0.01, n_assets, dtype=np.float64)
    batch, bundle = _batch_and_label(labels)

    # Normal financial-scale labels must not invoke the expensive stable
    # per-bucket mean repair. For the Numba wrapper, execute Python kernels only.
    from quant_evaluator.metrics import quantile_numeric

    def fail_if_repaired(_values):
        raise AssertionError("normal-scale bucket entered stable mean repair")

    monkeypatch.setattr(quantile_numeric, "stable_finite_mean", fail_if_repaired)
    if path == "numba_python":
        repair_calls = _force_numba_python_implementations(monkeypatch)
        # The wrapper's panel postprocessor may run, but its fast-path guard
        # should avoid the per-bucket stable mean repair.
    else:
        repair_calls = None

    returns, counts = _run_path(
        path, labels, monkeypatch=monkeypatch,
        n_quantiles=5, min_assets=1,
    )

    q_ids = assign_quantiles_batch(batch.values[:, :, 0], n_quantiles=5)[0]
    expected_counts = np.bincount(q_ids, minlength=5).astype(np.int32)
    expected_sums = np.bincount(q_ids, weights=labels, minlength=5)
    expected_means = expected_sums / expected_counts
    np.testing.assert_array_equal(counts[0, :, 0], expected_counts)
    np.testing.assert_allclose(returns[0, :, 0], expected_means, rtol=0.0, atol=1e-15)
    if repair_calls is not None:
        # Panel-level postprocessing was reached, without invoking stable repair.
        assert repair_calls == [1]
