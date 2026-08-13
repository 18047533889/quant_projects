# -*- coding: utf-8 -*-
"""R26-005..012: target-dependent tie handling (Chatterjee ξ) and the
copula MI/entropy estimator (no double correction).

Goldens (R26-008 / R26-012):
* continuous independent x,y -> ξ ≈ 0;
* discrete tied x + independent y -> NOT systematically positive;
* perfect dependence -> high ξ;
* within-x-tie permutation of y -> no systematic boost;
* copula MI >= 0, independent ≈ 0, identical -> high, entropy in [0,1].
"""
from __future__ import annotations

import numpy as np
import pytest

from cleaned_operators.dependence_ext import _chatterjee_xi
from cleaned_operators.cross_section_local import _copula_cross_series


def _xi(xv, yv):
    return _chatterjee_xi(np.asarray(xv, dtype=float), np.asarray(yv, dtype=float))


def test_chatterjee_continuous_independent_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    assert abs(_xi(x, y) < 0.1


def test_chatterjee_tied_x_independent_y_not_systematically_positive():
    rng = np.random.default_rng(1)
    vals = [_xi(rng.integers(0, 4, size=300).astype(float), rng.normal(size=300)) for _ in range(5)]
    # R26-006: the old lexsort((yv,xv)) biased tied-x to ~0.9; the fix must
    # centre near 0.
    assert abs(float(np.mean(vals)) < 0.2, vals


def test_chatterjee_perfect_dependence_high():
    x = np.linspace(0.0, 1.0, 400)
    assert _xi(x, x) > 0.9


def test_chatterjee_within_x_tie_permutation_y_no_boost():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 4, size=300).astype(float)
    y = rng.normal(size=300)
    # The FIX removes the estimator's INTERNAL y-sort (lexsort((yv,xv))) — the
    # estimator must not actively sort tied-X rows by the response.  Random
    # within-tie permutations of the input rows only reorder the (stable, input-
    # order) tie-break, so the estimate stays centred near 0 across permutations
    # (no systematic boost; finite-sample noise ~O(1/sqrt(n)) only).
    vals = [_xi(x, rng.permutation(y)) for _ in range(10)]
    assert abs(float(np.mean(vals)) < 0.2, vals


def test_copula_mi_independent_near_zero_and_nonnegative():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(2, 500))
    b = rng.normal(size=(2, 500))
    mi = _copula_cross_series(a, b, 8, entropy=False)
    assert not np.any(mi[mi == mi] < 0), "R26-011: MI must be >= 0"
    assert float(np.nanmean(mi) < 0.3


def test_copula_mi_identical_variables_high():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(2, 500))
    mi = _copula_cross_series(a, a.copy(), 8, entropy=False)
    assert float(np.nanmean(mi) > 0.5


def test_copula_entropy_in_declared_domain():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 500))
    b = rng.normal(size=(2, 500))
    e = _copula_cross_series(a, b, 8, entropy=True)
    vals = e[e == e]
    assert float(vals.min() >= 0.0 and float(vals.max() <= 1.0
