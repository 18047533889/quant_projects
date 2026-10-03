"""Independent bucket error-bound refinement regressions."""
from decimal import Decimal, localcontext

import numpy as np

from quant_evaluator.metrics import quantile_numeric as numeric


def test_small_finance_bucket_does_not_use_exact_rational_path(monkeypatch):
    labels = np.full(5461, 0.02, dtype=np.float64)
    ids = np.ones(5461, dtype=np.int32)
    ids[:10] = 0
    counts = np.array([10, 5451], dtype=np.int64)
    means = np.array([np.sum(labels[:10]) / 10, np.sum(labels[10:]) / 5451])
    before = means.copy()
    row_bound = numeric.label_sum_error_bounds(labels[None, :])[0]
    assert row_bound / counts[0] > 1e-12
    def forbidden(_values):
        raise AssertionError("safe bucket unnecessarily uses Fraction")
    monkeypatch.setattr(numeric, "stable_finite_mean", forbidden)
    numeric.repair_bucket_means(ids, labels, means, counts, 10, row_bound)
    np.testing.assert_array_equal(means, before)
    assert abs(means[0] - 0.02) <= 1e-12

def test_nonfinite_unassigned_labels_do_not_disable_finance_refinement(monkeypatch):
    labels = np.concatenate((np.full(5461, 0.02), [np.nan, np.inf, -np.inf]))
    ids = np.ones(labels.size, dtype=np.int32)
    ids[:10] = 0
    ids[-3:] = -1
    means = np.array([0.02, 0.02])
    def forbidden(_values):
        raise AssertionError("invalid labels disabled safe refinement")
    monkeypatch.setattr(numeric, "stable_finite_mean", forbidden)
    numeric.repair_bucket_means(ids, labels, means, np.array([10, 5451]), 10,
                               numeric.label_sum_error_bounds(labels[None, :])[0])
    np.testing.assert_array_equal(means, [0.02, 0.02])


def test_extreme_bucket_still_repairs_after_refinement():
    labels = np.array([1e308, 1., -1e308, 3., 0.02, 0.02])
    ids = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    counts = np.array([4, 2], dtype=np.int64)
    means = np.array([0.75, 0.02])
    numeric.repair_bucket_means(ids, labels, means, counts, 2, np.inf)
    with localcontext() as context:
        context.prec = 2000
        expected = float(sum(map(Decimal.from_float, labels[:4]), Decimal(0)) / 4)
    assert means[0] == expected == 1.0
    assert means[1] == 0.02


def test_nonfinite_mean_repairs_even_when_refined_bound_is_small():
    labels = np.array([0.01, 0.03])
    means = np.array([np.inf])
    numeric.repair_bucket_means(np.zeros(2, dtype=np.int32), labels, means,
                               np.array([2]), 2, 0.0)
    with localcontext() as context:
        context.prec = 2000
        expected = float(sum(map(Decimal.from_float, labels), Decimal(0)) / 2)
    assert means[0] == expected
