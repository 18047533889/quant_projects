"""Freeze-array ownership is unchanged without the transient ndarray copy."""
import numpy as np
import pytest

from quant_evaluator.contracts.metric_artifacts import _freeze_array


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_ndarray_freezes_exact_values_with_immutable_bytes_owner(layout):
    source = np.arange(24, dtype=np.float64).reshape(4, 6)
    if layout == "F":
        source = np.asfortranarray(source)
    elif layout == "strided":
        source = source[:, ::2]
    expected = np.array(source, copy=True)
    frozen = _freeze_array(source, "factor values")
    assert frozen.shape == expected.shape
    assert frozen.dtype == expected.dtype
    assert frozen.flags.c_contiguous
    np.testing.assert_array_equal(frozen, expected)
    source.flat[0] = -999
    np.testing.assert_array_equal(frozen, expected)
    with pytest.raises(ValueError):
        frozen.flags.writeable = True
    assert isinstance(frozen.base.base, bytes)


def test_object_array_still_fails_closed():
    with pytest.raises(Exception, match="object arrays"):
        _freeze_array(np.array(["x"], dtype=object), "axis")


def test_list_input_retains_owned_immutable_result():
    source = [1.0, 2.0]
    frozen = _freeze_array(source, "values")
    source[0] = 7.0
    np.testing.assert_array_equal(frozen, [1.0, 2.0])
    assert isinstance(frozen.base.base, bytes)
