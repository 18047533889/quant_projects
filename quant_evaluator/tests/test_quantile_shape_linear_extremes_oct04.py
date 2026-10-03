"""Exact Fraction oracles for linear quantile-shape metrics at float64 edges."""
from fractions import Fraction

import numpy as np

from quant_evaluator.metrics import quantile_shape as shape
from quant_evaluator.metrics.shape_linear_numeric import linear_risk_columns


def _rounded_fraction(value):
    try:
        return float(value)
    except OverflowError:
        return float("-inf") if value < 0 else float("inf")


def _exact_linear_value(terms, divisor=1):
    return _rounded_fraction(sum((Fraction.from_float(float(x)) * int(c)
                                  for x, c in terms), Fraction()) / divisor)


def _exact_mean_abs_differences(values):
    diffs = [abs(Fraction.from_float(float(b)) - Fraction.from_float(float(a)))
             for a, b in zip(values[:-1], values[1:])]
    return _rounded_fraction(sum(diffs, Fraction()) / len(diffs))


def _legacy_linear_metrics(values):
    columns = values.shape[1]
    results = {name: np.full(columns, np.nan) for name in (
        "curvature", "adjacent", "tail", "extreme", "top", "bottom",
    )}
    nq = values.shape[0]
    for f in range(columns):
        col = values[:, f]
        finite = np.isfinite(col)
        if nq >= 3:
            interior = finite[1:-1] & finite[:-2] & finite[2:]
            if np.any(interior):
                d2 = col[2:][interior] - 2.0 * col[1:-1][interior] + col[:-2][interior]
                results["curvature"][f] = np.mean(d2)
            mid = nq // 2
            if finite[0] and finite[-1] and finite[mid]:
                results["tail"][f] = (col[-1] - col[mid]) - (col[mid] - col[0])
        pairs = finite[:-1] & finite[1:]
        if nq >= 2 and np.any(pairs):
            results["adjacent"][f] = np.mean(np.abs(col[1:][pairs] - col[:-1][pairs]))
        if nq >= 2 and finite[0] and finite[1] and finite[-1] and finite[-2]:
            top = col[-1] - col[-2]
            bottom = col[1] - col[0]
            results["extreme"][f] = (top + bottom) / 2.0
        if nq >= 2 and finite[-1] and finite[-2]:
            results["top"][f] = col[-1] - col[-2]
        if nq >= 2 and finite[1] and finite[0]:
            results["bottom"][f] = col[1] - col[0]
    return results


def test_curvature_aggregates_exact_stencils_before_single_rounding():
    largest = np.finfo(np.float64).max
    flat_huge = np.array([largest, largest, largest])
    cancelling_stencils = np.array([0.0, largest, 0.0, -largest, 0.0])
    assert shape.compute_quantile_curvature(flat_huge)[0] == 0.0
    stencils = [
        ((cancelling_stencils[i + 2], 1),
         (cancelling_stencils[i + 1], -2),
         (cancelling_stencils[i], 1))
        for i in range(len(cancelling_stencils) - 2)
    ]
    exact = _rounded_fraction(sum((Fraction.from_float(float(x)) * c
                                   for stencil in stencils for x, c in stencil), Fraction())
                              / len(stencils))
    assert shape.compute_quantile_curvature(cancelling_stencils)[0] == exact


def test_large_curvature_matches_exact_binary64_linear_expression():
    largest = np.finfo(np.float64).max
    previous = np.nextafter(largest, 0.0)
    values = np.array([largest, previous, largest])
    expected = _exact_linear_value([
        (values[2], 1), (values[1], -2), (values[0], 1),
    ])
    assert expected == 2.0 * (largest - previous)
    assert shape.compute_quantile_curvature(values)[0] == expected


def test_finite_large_cancellation_curvature_matches_fraction_oracle():
    values = np.array([
        float.fromhex("-0x1.c410b118ae524p+660"),
        float.fromhex("-0x1.c3132eb82f949p+663"),
        float.fromhex("0x1.5ae902d53fdbbp+664"),
        float.fromhex("0x1.27e04abef903ap+665"),
        float.fromhex("0x1.b53035839fa8fp+663"),
    ])
    stencils = [
        ((values[i + 2], 1), (values[i + 1], -2), (values[i], 1))
        for i in range(len(values) - 2)
    ]
    expected = _exact_linear_value(
        [term for stencil in stencils for term in stencil], divisor=len(stencils),
    )
    actual = shape.compute_quantile_curvature(values)[0]
    assert actual == expected
    assert abs(actual - expected) <= 1.0e-12


def test_adjacent_spread_aggregates_before_division_at_float_limit():
    largest = np.finfo(np.float64).max
    values = np.array([-largest, 0.0, largest, 0.0, -largest])
    expected = _exact_mean_abs_differences(values)
    assert expected == largest
    assert shape.compute_quantile_adjacent_spread(values)[0] == expected


def test_extreme_cliffs_combine_exactly_instead_of_inf_minus_inf():
    largest = np.finfo(np.float64).max
    values = np.array([largest, -largest, -largest, largest])
    expected = _exact_linear_value([
        (values[-1], 1), (values[-2], -1),
        (values[1], 1), (values[0], -1),
    ], divisor=2)
    assert expected == 0.0
    assert shape.compute_quantile_extreme_cliff(values)[0] == expected


def test_extreme_cliff_nonzero_result_uses_denominator_two():
    values = np.array([-1.0e308, 0.0, 1.0e308])
    expected = _exact_linear_value([
        (values[-1], 1), (values[-2], -1),
        (values[1], 1), (values[0], -1),
    ], divisor=2)
    assert expected == 1.0e308
    assert shape.compute_quantile_extreme_cliff(values)[0] == expected


def test_individual_cliffs_and_tail_asymmetry_return_signed_infinity_on_true_overflow():
    largest = np.finfo(np.float64).max
    top = shape.compute_top_quantile_cliff(np.array([-largest, largest]))[0]
    bottom = shape.compute_bottom_quantile_cliff(np.array([largest, -largest]))[0]
    tail = shape.compute_quantile_tail_asymmetry(np.array([largest, -largest, largest]))[0]
    assert top == np.inf
    assert bottom == -np.inf
    assert tail == np.inf


def test_ordinary_finite_profiles_keep_legacy_float64_operation_order():
    values = np.array([-0.031, -0.012, 0.004, 0.019, 0.027])
    curvature = np.mean(values[2:] - 2.0 * values[1:-1] + values[:-2])
    adjacent = np.mean(np.abs(values[1:] - values[:-1]))
    extreme = ((values[-1] - values[-2]) + (values[1] - values[0])) / 2.0
    tail = (values[-1] - values[len(values) // 2]) - (
        values[len(values) // 2] - values[0]
    )
    np.testing.assert_array_equal(shape.compute_quantile_curvature(values), [curvature])
    np.testing.assert_array_equal(shape.compute_quantile_adjacent_spread(values), [adjacent])
    np.testing.assert_array_equal(shape.compute_quantile_extreme_cliff(values), [extreme])
    np.testing.assert_array_equal(shape.compute_quantile_tail_asymmetry(values), [tail])
    np.testing.assert_array_equal(shape.compute_top_quantile_cliff(values), [values[-1] - values[-2]])
    np.testing.assert_array_equal(shape.compute_bottom_quantile_cliff(values), [values[1] - values[0]])


def test_subnormal_mixed_scale_stencils_use_exact_binary64_inputs():
    unit = float.fromhex("0x0.0000000000001p-1022")
    largest = np.finfo(np.float64).max
    curvature_values = np.array([largest, unit, -largest])
    curvature_exact = _rounded_fraction(
        sum((Fraction.from_float(float(value)) * coefficient
             for value, coefficient in (
                 (curvature_values[2], 1), (curvature_values[1], -2),
                 (curvature_values[0], 1))), Fraction())
    )
    assert shape.compute_quantile_curvature(curvature_values)[0] == curvature_exact
    adjacent_values = np.array([0.0, unit, 3 * unit])
    assert shape.compute_quantile_adjacent_spread(adjacent_values)[0] == (
        _exact_mean_abs_differences(adjacent_values)
    )


def test_missing_point_rules_remain_unchanged():
    assert np.isnan(shape.compute_quantile_curvature([1.0, np.nan, 2.0])[0])
    assert np.isnan(shape.compute_quantile_tail_asymmetry([1.0, np.nan, 2.0])[0])
    assert np.isnan(shape.compute_quantile_adjacent_spread([0.0, np.nan, 1.0])[0])
    assert np.isnan(shape.compute_quantile_extreme_cliff([np.nan, 1.0])[0])
    assert np.isnan(shape.compute_top_quantile_cliff([np.nan, 1.0])[0])
    assert np.isnan(shape.compute_bottom_quantile_cliff([1.0, np.nan])[0])


def test_only_finite_stencils_and_adjacent_pairs_enter_their_denominators():
    curvature_values = np.array([0.0, 1.0, 4.0, np.nan, 10.0, 13.0, 18.0])
    valid_stencils = [
        ((curvature_values[i + 2], 1), (curvature_values[i + 1], -2),
         (curvature_values[i], 1))
        for i in range(len(curvature_values) - 2)
        if np.isfinite(curvature_values[i:i + 3]).all()
    ]
    expected_curvature = _exact_linear_value(
        [term for stencil in valid_stencils for term in stencil],
        divisor=len(valid_stencils),
    )
    assert len(valid_stencils) == 2
    assert shape.compute_quantile_curvature(curvature_values)[0] == expected_curvature

    adjacent_values = np.array([0.0, 3.0, np.nan, 10.0, 12.0, 17.0])
    valid_pairs = [
        (left, right) for left, right in zip(adjacent_values[:-1], adjacent_values[1:])
        if np.isfinite(left) and np.isfinite(right)
    ]
    exact_differences = [abs(Fraction.from_float(float(right))
                             - Fraction.from_float(float(left)))
                         for left, right in valid_pairs]
    expected_spread = _rounded_fraction(sum(exact_differences, Fraction())
                                        / len(valid_pairs))
    assert len(valid_pairs) == 3
    assert shape.compute_quantile_adjacent_spread(adjacent_values)[0] == expected_spread


def test_seeded_normal_profiles_match_legacy_operation_order_bit_for_bit():
    rng = np.random.default_rng(20261004)
    values = rng.normal(0.0, 0.025, size=(9, 16))
    nq, columns = values.shape
    expected_curvature = np.empty(columns)
    expected_adjacent = np.empty(columns)
    expected_tail = np.empty(columns)
    expected_extreme = np.empty(columns)
    expected_top = np.empty(columns)
    expected_bottom = np.empty(columns)
    for f in range(columns):
        col = values[:, f]
        expected_curvature[f] = np.mean(col[2:] - 2.0 * col[1:-1] + col[:-2])
        expected_adjacent[f] = np.mean(np.abs(col[1:] - col[:-1]))
        mid = nq // 2
        expected_tail[f] = (col[-1] - col[mid]) - (col[mid] - col[0])
        expected_extreme[f] = ((col[-1] - col[-2]) + (col[1] - col[0])) / 2.0
        expected_top[f] = col[-1] - col[-2]
        expected_bottom[f] = col[1] - col[0]
    np.testing.assert_array_equal(shape.compute_quantile_curvature(values), expected_curvature)
    np.testing.assert_array_equal(shape.compute_quantile_adjacent_spread(values), expected_adjacent)
    np.testing.assert_array_equal(shape.compute_quantile_tail_asymmetry(values), expected_tail)
    np.testing.assert_array_equal(shape.compute_quantile_extreme_cliff(values), expected_extreme)
    np.testing.assert_array_equal(shape.compute_top_quantile_cliff(values), expected_top)
    np.testing.assert_array_equal(shape.compute_bottom_quantile_cliff(values), expected_bottom)


def test_q512_f48_batched_curvature_and_spread_match_legacy_bitwise():
    values = np.random.default_rng(5122048).normal(0.02, 0.004, size=(512, 48))
    expected = _legacy_linear_metrics(values)
    np.testing.assert_array_equal(shape.compute_quantile_curvature(values), expected["curvature"])
    np.testing.assert_array_equal(shape.compute_quantile_adjacent_spread(values), expected["adjacent"])


def test_q20_f48_mixed_finite_missing_groups_match_legacy_bitwise():
    values = np.random.default_rng(2048048).normal(0.02, 0.004, size=(20, 48))
    values[3, 1] = np.nan
    values[11, 2] = np.inf
    values[4, 3] = np.nan
    values[15, 3] = -np.inf
    expected = _legacy_linear_metrics(values)
    np.testing.assert_array_equal(shape.compute_quantile_curvature(values), expected["curvature"])
    np.testing.assert_array_equal(shape.compute_quantile_adjacent_spread(values), expected["adjacent"])


def test_unrelated_nan_and_inf_do_not_mark_normal_columns_risky():
    values = np.array([
        [0.011, 0.013], [0.017, 0.019], [np.inf, np.nan],
        [0.023, 0.029], [0.031, 0.037], [0.041, 0.043],
    ])
    assert not linear_risk_columns(values, 16).any()
    np.testing.assert_array_equal(
        shape.compute_quantile_curvature(values),
        np.array([0.041 - 2.0 * 0.031 + 0.023,
                  0.043 - 2.0 * 0.037 + 0.029]),
    )
    # The direct legacy expressions retain their finite-pair/required-endpoint rules.
    expected_tail = np.array([
        (0.041 - 0.023) - (0.023 - 0.011),
        (0.043 - 0.029) - (0.029 - 0.013),
    ])
    np.testing.assert_array_equal(shape.compute_quantile_tail_asymmetry(values), expected_tail)
    np.testing.assert_array_equal(
        shape.compute_top_quantile_cliff(values), values[-1] - values[-2],
    )
    np.testing.assert_array_equal(
        shape.compute_bottom_quantile_cliff(values), values[1] - values[0],
    )
