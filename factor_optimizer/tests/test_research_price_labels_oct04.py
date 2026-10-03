"""Contract tests for pure price-to-return research labels."""

import numpy as np
import pytest

from factor_optimizer.research_price_labels import price_to_return_labels


def test_price_to_return_labels_computes_end_over_start_minus_one():
    start = np.array([[2.0, 4.0], [8.0, 5.0]])
    end = np.array([[3.0, 2.0], [4.0, 10.0]])

    returns, valid = price_to_return_labels(start, end)

    np.testing.assert_allclose(returns, [[0.5, -0.5], [-0.5, 1.0]], rtol=0, atol=0)
    np.testing.assert_array_equal(valid, [[True, True], [True, True]])


def test_invalid_endpoints_and_nonfinite_result_are_nan_and_invalid():
    start = np.array([[1.0, 0.0, -2.0, np.nan, np.inf, 1e-300],
                      [2.0, 3.0, 4.0, 5.0, 6.0, 1.0]])
    end = np.array([[2.0, 1.0, 3.0, 2.0, 2.0, 1e300],
                    [0.0, np.inf, 4.0, np.nan, 6.0, 1.0]])

    returns, valid = price_to_return_labels(start, end)

    np.testing.assert_array_equal(
        valid, [[True, False, False, False, False, False],
                [False, False, True, False, True, True]])
    np.testing.assert_allclose(returns[valid], [1.0, 0.0, 0.0, 0.0], rtol=0, atol=0)
    assert np.isnan(returns[~valid]).all()


@pytest.mark.parametrize("bad_dtype", [bool, object, np.complex128, np.str_])
def test_non_real_or_non_numeric_dtypes_are_rejected(bad_dtype):
    start = np.ones((2, 2), dtype=bad_dtype)
    end = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(TypeError, match="real numeric"):
        price_to_return_labels(start, end)


@pytest.mark.parametrize(
    "start,end",
    [
        (np.ones(3), np.ones(3)),
        (np.ones((2, 2)), np.ones((2, 1))),
    ],
)
def test_endpoint_shape_must_be_matching_two_dimensional(start, end):
    with pytest.raises(ValueError, match="same two-dimensional shape"):
        price_to_return_labels(start, end)


def test_inputs_are_not_modified_or_reused_as_output_storage():
    start = np.array([[2.0, 0.0], [np.nan, 4.0]])
    end = np.array([[4.0, 2.0], [1.0, 8.0]])
    start_before, end_before = start.copy(), end.copy()

    returns, valid = price_to_return_labels(start, end)

    np.testing.assert_array_equal(start, start_before)
    np.testing.assert_array_equal(end, end_before)
    assert not np.shares_memory(returns, start)
    assert not np.shares_memory(returns, end)
    assert not np.shares_memory(valid, start)
    assert not np.shares_memory(valid, end)


def test_integer_prices_are_accepted_as_real_endpoints():
    returns, valid = price_to_return_labels(
        np.array([[2, 4]], dtype=np.int64), np.array([[3, 2]], dtype=np.int64))

    np.testing.assert_allclose(returns, [[0.5, -0.5]], rtol=0, atol=0)
    np.testing.assert_array_equal(valid, [[True, True]])
