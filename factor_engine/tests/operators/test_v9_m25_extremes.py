from fractions import Fraction
from itertools import permutations

import numpy as np


def _load_peer():
    from factor_engine.cleaned_operators.cross_section import peer_ops as module
    return module


def _decimal_oracle(values, weights):
    out = []
    xv = [Fraction.from_float(float(x)) for x in values]
    wv = [Fraction.from_float(float(w)) for w in weights]
    for excluded in range(len(values)):
        peers = [i for i in range(len(values)) if i != excluded]
        numerator = sum((wv[i] * xv[i] for i in peers), Fraction(0))
        denominator = sum((wv[i] for i in peers), Fraction(0))
        out.append(float(numerator / denominator))
    return np.asarray(out)


def _run(values, weights):
    peer = _load_peer()
    return peer._peer_weighted_mean_ex_self_row(
        np.asarray(values, dtype=float),
        np.asarray(["g"] * len(values)),
        np.asarray(weights, dtype=float),
    )


def test_finite_prefix_overflow_matches_decimal_oracle():
    values = np.array([1e308, 1e308, -1e308])
    weights = np.ones(3)
    np.testing.assert_allclose(
        _run(values, weights), _decimal_oracle(values, weights), rtol=0, atol=0
    )


def test_finite_weighted_product_overflow_matches_decimal_oracle():
    values = np.array([1e200, -1e200, 3.0])
    weights = np.array([1e200, 1e200, 1.0])
    np.testing.assert_allclose(
        _run(values, weights), _decimal_oracle(values, weights), rtol=1e-15
    )


def test_extreme_permutations_are_equivariant_against_decimal():
    values = np.array([1e308, 1e308, -1e308, -1e308])
    weights = np.ones(4)
    for order in set(permutations(range(4))):
        order = np.asarray(order)
        actual = _run(values[order], weights[order])
        expected = _decimal_oracle(values[order], weights[order])
        np.testing.assert_allclose(actual, expected, rtol=1e-15)


def test_mixed_extreme_cancellation_preserves_unit_residual():
    values = np.array([1e308, 1e308, -1e308, -1e308, 1.0])
    weights = np.ones(5)
    actual = _run(values, weights)
    expected = _decimal_oracle(values, weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)
    # Excluding the finite unit peer leaves exact cancellation; excluding one
    # extreme retains the tiny +1 contribution in the Decimal numerator.
    assert actual[-1] == 0.0


def test_original_large_self_small_peers_case_stays_exact():
    values = np.array([1e16, 1.0, 1.0])
    weights = np.ones(3)
    actual = _run(values, weights)
    assert actual[0] == 1.0
    np.testing.assert_allclose(actual[1:], [5e15, 5e15], rtol=2e-16)
