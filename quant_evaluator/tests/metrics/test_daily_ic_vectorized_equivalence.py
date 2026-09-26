"""Bit-exact equivalence tests: vectorized compute_daily_ic vs the
historical per-day loop reference implementation.

Contract under test (see ``_compute_daily_ic_reference`` in
``quant_evaluator.metrics.ic``): the vectorized public path must produce
**bit-identical** ``(ic_series, valid_counts)`` on every input the
reference accepts, across seeds, dtypes, missingness patterns, ties,
validity masks and boundary sizes.  These tests are part of the default
pytest suite so any future change that perturbs IC semantics fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import _compute_daily_ic_reference, compute_daily_ic

_METHODS = ("pearson", "spearman")


def _make_batch(values, labels, factor_validity=None, label_validity=None):
    T, N, F = values.shape
    fb = FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=AxisRef("t", "int", T),
        asset_axis=AxisRef("a", "str", N),
        values=values,
        validity=factor_validity,
    )
    lb = LabelBundle(
        target_id="r",
        values=labels,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
        validity=label_validity,
    )
    return fb, lb


def _assert_bit_identical(fb, lb, methods=_METHODS, min_assets=(20,)):
    for method in methods:
        for ma in min_assets:
            ref_ic, ref_n = _compute_daily_ic_reference(fb, lb, method=method, min_assets=ma)
            new_ic, new_n = compute_daily_ic(fb, lb, method=method, min_assets=ma)
            assert new_ic.dtype == ref_ic.dtype == np.float64
            assert new_n.dtype == ref_n.dtype
            assert np.array_equal(new_n, ref_n), (
                f"valid_counts differ: method={method} min_assets={ma}"
            )
            # Bit-identical: NaN must sit at exactly the same positions AND
            # the finite payload must match bit-for-bit (no tolerance).
            assert np.array_equal(new_ic, ref_ic, equal_nan=True), (
                f"ic_series differ: method={method} min_assets={ma}\n"
                f"max abs diff={np.nanmax(np.abs(new_ic - ref_ic))}"
            )


def test_equivalence_random_gaussian_no_missing():
    rng = np.random.default_rng(101)
    vals = rng.normal(size=(120, 40, 2))
    labels = rng.normal(size=(120, 40))
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_heavy_missing_and_ties():
    rng = np.random.default_rng(102)
    vals = rng.normal(size=(90, 35, 2))
    labels = rng.normal(size=(90, 35))
    vals[rng.random(vals.shape) < 0.30] = np.nan
    labels[rng.random(labels.shape) < 0.30] = np.nan
    vals = np.round(vals, 1)  # heavy ties
    labels = np.round(labels, 1)
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(5, 10, 20))


def test_equivalence_with_validity_masks():
    rng = np.random.default_rng(103)
    vals = rng.normal(size=(70, 30, 1))
    labels = rng.normal(size=(70, 30))
    f_valid = rng.random(vals.shape) > 0.2
    l_valid = rng.random(labels.shape) > 0.2
    fb, lb = _make_batch(vals, labels, factor_validity=f_valid, label_validity=l_valid)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_1d_label_broadcast():
    rng = np.random.default_rng(104)
    vals = rng.normal(size=(60, 25, 2))
    labels_1d = rng.normal(size=60)
    labels_1d[::7] = np.nan
    fb, lb = _make_batch(vals, labels_1d)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_float32_inputs():
    rng = np.random.default_rng(105)
    vals = (rng.normal(size=(80, 30, 1))).astype(np.float32)
    labels = rng.normal(size=(80, 30)).astype(np.float32)
    vals[rng.random(vals.shape) < 0.15] = np.nan
    labels[rng.random(labels.shape) < 0.15] = np.nan
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_all_constant_columns():
    """Constant factor/label columns must yield NaN identically."""
    rng = np.random.default_rng(106)
    vals = rng.normal(size=(50, 30, 2))
    labels = rng.normal(size=(50, 30))
    vals[:, :, 0] = 3.14  # constant factor
    labels[:, 5] = 2.0
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_boundary_sizes_below_min_assets():
    """Cross-sections below min_assets must be NaN in both paths."""
    rng = np.random.default_rng(107)
    vals = rng.normal(size=(40, 25, 1))
    labels = rng.normal(size=(40, 25))
    vals[rng.random(vals.shape) < 0.4] = np.nan  # some days fall below 20 valid
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(10, 20, 25))


def test_equivalence_all_nan_days_and_columns():
    rng = np.random.default_rng(108)
    vals = rng.normal(size=(45, 30, 2))
    labels = rng.normal(size=(45, 30))
    vals[5:10] = np.nan       # fully missing days
    vals[:, 3, 1] = np.nan    # fully missing factor column slice
    labels[20:25] = np.nan
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_extreme_values_and_inf():
    rng = np.random.default_rng(109)
    vals = rng.normal(size=(60, 30, 1))
    labels = rng.normal(size=(60, 30))
    vals[3, 0, 0] = np.inf    # non-finite values must be excluded pairwise
    vals[4, 1, 0] = -np.inf
    labels[5, 2] = np.inf
    vals[6] = 1e300           # extreme but finite magnitudes
    labels[7] = -1e300
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(5, 20))


def test_equivalence_min_assets_one_guard():
    """n == 0 rows must stay NaN for every min_assets >= 1.

    (min_assets == 0 is out of the equivalence contract: the historical
    reference loop raises ValueError on empty cross-sections via
    np.nanmax; the vectorized path returns NaN instead, which is a
    documented graceful-degradation improvement, not a divergence.)
    """
    rng = np.random.default_rng(110)
    vals = rng.normal(size=(30, 20, 1))
    labels = rng.normal(size=(30, 20))
    vals[10] = np.nan
    labels[11] = np.nan
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(1, 5))


def test_equivalence_single_row_single_asset():
    vals = np.array([[[1.0, 2.0], [3.0, 4.0]]])          # (1, 2, 2)
    labels = np.array([[0.5, -0.5]])                      # (1, 2)
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(1, 2))


def test_equivalence_multi_factor_independent_labels():
    """Each factor shares the label panel; ranks must match per (t, f)."""
    rng = np.random.default_rng(111)
    vals = rng.normal(size=(100, 45, 4))
    labels = rng.normal(size=(100, 45))
    vals[rng.random(vals.shape) < 0.2] = np.nan
    labels[rng.random(labels.shape) < 0.2] = np.nan
    fb, lb = _make_batch(vals, labels)
    _assert_bit_identical(fb, lb, min_assets=(10,))


@pytest.mark.parametrize("method", _METHODS)
def test_known_value_oracle_small_case(method):
    """Hand-checkable tiny case: perfectly monotone cross-section.

    Day 0: factor = [1, 2, 3, 4], label = [10, 20, 30, 40] -> IC = 1.0.
    Day 1: label reversed -> IC = -1.0.
    """
    vals = np.array([[[1.0], [2.0], [3.0], [4.0]],
                     [[1.0], [2.0], [3.0], [4.0]]])       # (2, 4, 1)
    labels = np.array([[10.0, 20.0, 30.0, 40.0],
                       [40.0, 30.0, 20.0, 10.0]])          # (2, 4)
    fb, lb = _make_batch(vals, labels)
    ic, n = compute_daily_ic(fb, lb, method=method, min_assets=2)
    assert n.tolist() == [[4], [4]]
    assert ic[0, 0] == 1.0
    assert ic[1, 0] == -1.0


@pytest.mark.parametrize("method", _METHODS)
def test_asset_permutation_invariance(method):
    """IC must not depend on asset ordering (rank-based, pairwise mask)."""
    rng = np.random.default_rng(112)
    vals = rng.normal(size=(80, 40, 1))
    labels = rng.normal(size=(80, 40))
    vals[rng.random(vals.shape) < 0.2] = np.nan
    labels[rng.random(labels.shape) < 0.2] = np.nan
    fb1, lb1 = _make_batch(vals, labels)

    perm = rng.permutation(40)
    fb2, lb2 = _make_batch(vals[:, perm, :], labels[:, perm])
    ic1, _ = compute_daily_ic(fb1, lb1, method=method, min_assets=5)
    ic2, _ = compute_daily_ic(fb2, lb2, method=method, min_assets=5)
    # Mathematically exact; compressed-entry order changes corrcoef
    # rounding at the last ulp, so compare with a tight 1e-12 bound.
    assert np.allclose(ic1, ic2, rtol=0.0, atol=1e-12, equal_nan=True)


def test_positive_affine_invariance_spearman():
    """Spearman IC is invariant under strictly increasing affine maps and
    flips sign under negation (decreasing map)."""
    rng = np.random.default_rng(113)
    vals = rng.normal(size=(70, 35, 1))
    labels = rng.normal(size=(70, 35))
    vals[rng.random(vals.shape) < 0.2] = np.nan
    fb1, lb1 = _make_batch(vals, labels)
    fb2, lb2 = _make_batch(3.0 * vals + 7.5, 2.0 * labels + 1.0)
    ic1, _ = compute_daily_ic(fb1, lb1, method="spearman", min_assets=5)
    ic2, _ = compute_daily_ic(fb2, lb2, method="spearman", min_assets=5)
    assert np.allclose(ic1, ic2, rtol=0.0, atol=1e-12, equal_nan=True)

    fb3, lb3 = _make_batch(vals, -labels)
    ic3, _ = compute_daily_ic(fb3, lb3, method="spearman", min_assets=5)
    assert np.allclose(ic3, -ic1, rtol=0.0, atol=1e-12, equal_nan=True)


def test_valid_counts_consistency_with_mask():
    """valid_counts must equal the pairwise finite count per day/factor."""
    rng = np.random.default_rng(114)
    vals = rng.normal(size=(50, 28, 2))
    labels = rng.normal(size=(50, 28))
    f_valid = rng.random(vals.shape) > 0.25
    l_valid = rng.random(labels.shape) > 0.25
    fb, lb = _make_batch(vals, labels, factor_validity=f_valid, label_validity=l_valid)
    for method in _METHODS:
        _, n = compute_daily_ic(fb, lb, method=method, min_assets=5)
        expected = (np.isfinite(np.where(f_valid, vals, np.nan))
                    & np.isfinite(np.where(l_valid, labels, np.nan))[:, :, None]).sum(axis=1)
        assert np.array_equal(n, expected.astype(np.int32))
