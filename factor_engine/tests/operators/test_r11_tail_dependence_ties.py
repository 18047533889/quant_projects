# -*- coding: utf-8 -*-
"""R11 tail-dependence operators — honest naming + boundary-tie symmetry.

ISSUE 1 (honest naming)
-----------------------
``ts_upper_tail_dependence`` / ``ts_lower_tail_dependence`` compute at a FIXED q
the conditional probability

    upper:  P(y >  Q_y(q) | x >  Q_x(q))
    lower:  P(y <= Q_y(q) | x <= Q_x(q))

i.e. a fixed-q *tail coexceedance probability*, NOT the asymptotic tail-
dependence coefficient λ = lim_{q→1} P(y > F_y^{-1}(q) | x > F_x^{-1}(q)).
The canonical names are now ``ts_upper_tail_coexceedance_probability`` /
``ts_lower_tail_coexceedance_probability``; the old names are kept as
deprecated resolving aliases.

ISSUE 2 (upper/lower tie asymmetry)
-----------------------------------
The old kernel used ``x > Q`` (strict) for the upper tail and ``x <= Q``
(inclusive) for the lower tail.  With discrete A-share-style ties (0 returns,
limit up/down) the upper condition had far fewer members while the lower
swallowed an entire boundary-tie block — the same q gave asymmetric effective
tail mass.  Fractional boundary-tie membership now makes the effective tail
mass equal the target tail size on BOTH sides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from cleaned_operators import load_all

load_all()

from cleaned_operators.nonlinear_dependence import (
    TsLowerTailDependence,
    TsUpperTailDependence,
    _fractional_tail_membership,
    _tail_dependence,
)
from cleaned_operators.registry import OperatorRegistry

UPPER_COEX = "ts_upper_tail_coexceedance_probability"
LOWER_COEX = "ts_lower_tail_coexceedance_probability"
UPPER_OLD = "ts_upper_tail_dependence"
LOWER_OLD = "ts_lower_tail_dependence"


def _col(values) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(-1, 1), columns=["c"])


# ---------------------------------------------------------------------------
# ISSUE 1 — honest naming: coexceedance canonical + deprecated alias
# ---------------------------------------------------------------------------
def test_both_tail_dependence_names_resolve_to_coexceedance_canonicals():
    # The old names must still resolve (deprecated aliases) and the new names
    # must be the registered canonicals with a pandas_numpy runtime.
    assert OperatorRegistry.resolve_canonical(UPPER_OLD) == UPPER_COEX
    assert OperatorRegistry.resolve_canonical(LOWER_OLD) == LOWER_COEX
    assert UPPER_COEX in OperatorRegistry._operators
    assert LOWER_COEX in OperatorRegistry._operators
    assert OperatorRegistry.get(UPPER_OLD, "pandas_numpy") is not None
    assert OperatorRegistry.get(UPPER_COEX, "pandas_numpy") is not None
    assert OperatorRegistry.get(LOWER_OLD, "pandas_numpy") is not None
    assert OperatorRegistry.get(LOWER_COEX, "pandas_numpy") is not None


def test_alias_and_canonical_produce_identical_output():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0.0, 1.0, n)
    y = 0.8 * x + 0.6 * rng.normal(0.0, 1.0, n)

    fresh_upper = TsUpperTailDependence()
    canonical_op = OperatorRegistry.get(UPPER_COEX, "pandas_numpy")
    alias_op = OperatorRegistry.get(UPPER_OLD, "pandas_numpy")

    out_fresh = fresh_upper._calculate_series(_col(x), _col(y), window=n, q=0.9, min_tail_count=5)
    out_canon = canonical_op._calculate_series(_col(x), _col(y), window=n, q=0.9, min_tail_count=5)
    out_alias = alias_op._calculate_series(_col(x), _col(y), window=n, q=0.9, min_tail_count=5)
    np.testing.assert_allclose(
        out_canon.to_numpy(dtype=float), out_fresh.to_numpy(dtype=float), equal_nan=True
    )
    np.testing.assert_allclose(
        out_alias.to_numpy(dtype=float), out_fresh.to_numpy(dtype=float), equal_nan=True
    )
    # The fixed-q coexceedance is a real probability on this correlated pair.
    assert np.isfinite(float(out_fresh.iloc[-1, 0]))
    assert 0.0 <= float(out_fresh.iloc[-1, 0]) <= 1.0

    # Same check for the lower tail operator.
    fresh_lower = TsLowerTailDependence()
    lo = fresh_lower._calculate_series(_col(x), _col(y), window=n, q=0.1, min_tail_count=5)
    assert np.isfinite(float(lo.iloc[-1, 0]))
    assert 0.0 <= float(lo.iloc[-1, 0]) <= 1.0


def test_coexceedance_canonical_metadata_is_honest():
    for canonical, old in ((UPPER_COEX, UPPER_OLD), (LOWER_COEX, LOWER_OLD)):
        entry = OperatorRegistry.catalog().get(canonical, {})
        description = (entry.get("description") or "").lower()
        assert "coexceedance" in description, canonical
        # The formula (fixed-q conditional probability) is documented and the
        # legacy name is declared as the alias.
        assert "y >" in description or "y ≤" in description, canonical
        assert old in (entry.get("aliases") or []), canonical


# ---------------------------------------------------------------------------
# ISSUE 2 — fractional boundary-tie symmetry
# ---------------------------------------------------------------------------
def test_fractional_effective_mass_equals_target_on_boundary_ties():
    n = 120
    # A huge block of boundary ties exactly at the q-th quantile: 115 zeros,
    # 5 ones.  Q(0.9) = Q(0.1) = 0.0, so the boundary-tie group is 115 values.
    x = np.array([0.0] * 115 + [1.0] * 5)

    up_w = _fractional_tail_membership(x, 0.9, "upper")
    lo_w = _fractional_tail_membership(x, 0.1, "lower")

    # Upper target is (1-q)*N, lower target is q*N — both equal 0.1*N here.
    assert up_w.sum() == pytest.approx((1.0 - 0.9) * n, abs=1e-6)
    assert lo_w.sum() == pytest.approx(0.1 * n, abs=1e-6)
    # Weights stay in [0, 1], with the boundary-tie group split fractionally
    # (strict members weigh 1, boundary ties weigh f in (0, 1)).
    assert np.all(up_w >= 0.0) and np.all(up_w <= 1.0)
    assert np.all(lo_w >= 0.0) and np.all(lo_w <= 1.0)
    assert np.any((up_w > 0.0) & (up_w < 1.0))
    assert np.any((lo_w > 0.0) & (lo_w < 1.0))
    # The boundary fraction is chosen so the effective mass reaches the target.
    up_boundary = up_w[(up_w > 0.0) & (up_w < 1.0)]
    lo_boundary = lo_w[(lo_w > 0.0) & (lo_w < 1.0)]
    assert np.allclose(up_boundary, up_boundary[0])
    assert np.allclose(lo_boundary, lo_boundary[0])


def test_tail_dependence_denominator_uses_fractional_mass():
    n = 120
    x = np.array([0.0] * 115 + [1.0] * 5)
    y = x.copy()
    for q, upper, target in (
        (0.9, True, (1.0 - 0.9) * n),
        (0.1, False, 0.1 * n),
    ):
        direction = "upper" if upper else "lower"
        denom = float(_fractional_tail_membership(x, q, direction).sum())
        assert denom == pytest.approx(target, abs=1e-6)
        # The implementation must be using the same weighted joint / denom.
        joint = float(np.sum(
            _fractional_tail_membership(x, q, direction)
            * _fractional_tail_membership(y, q, direction)
        ))
        p_manual = joint / denom
        p_impl = _tail_dependence(x, y, q, upper, 2)
        assert p_impl == pytest.approx(p_manual, abs=1e-9)


def test_symmetric_gaussian_upper_lower_coexceedance_close():
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.normal(0.0, 1.0, n)
    y = 0.7 * x + 0.71 * rng.normal(0.0, 1.0, n)
    p_up = _tail_dependence(x, y, 0.9, True, 20)
    p_lo = _tail_dependence(x, y, 0.1, False, 20)
    assert np.isfinite(p_up) and np.isfinite(p_lo)
    # Upper at q=0.9 and lower at q=0.1 are mirror-image tail events; for a
    # symmetric bivariate Gaussian they must be close (loose 0.1 tolerance).
    assert abs(p_up - p_lo) < 0.1


def test_boundary_tie_upper_lower_no_longer_degenerate():
    # The old strict/inclusive asymmetry: x has a huge tie block at Q so the
    # old upper tail had 1 member (P≈0) while the old lower tail swallowed the
    # whole block (P≈1).  With y = 10 - x (anti-correlated) the OLD result was
    # upper≈0 vs lower≈1; the fractional scheme must make BOTH sensible.
    x = np.array([0.0] * 99 + [10.0])
    y = 10.0 - x
    p_up = _tail_dependence(x, y, 0.9, True, 2)
    p_lo = _tail_dependence(x, y, 0.1, False, 2)
    assert np.isfinite(p_up) and np.isfinite(p_lo)
    assert 0.0 < p_up < 1.0
    assert 0.0 < p_lo < 1.0
    # Both sides carry the same effective tail mass (0.1*N), so they agree.
    assert p_up == pytest.approx(p_lo, abs=1e-6)
