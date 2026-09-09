"""V9-Q04: NumPy scalar types must not bypass the shared integer domain."""
import numpy as np
import pytest

from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.cleaned_operators.common.strict_params import (
    strict_int, strict_nonnegative_int, strict_positive_int,
)


@pytest.mark.parametrize("dtype", [int, np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("value", [0, 1, 5, 10, 11])
def test_all_integer_types_share_bounds(dtype, value):
    if 1 <= value <= 10:
        result = strict_int(dtype(value), "window", minimum=1, maximum=10)
        assert type(result) is int and result == value
    else:
        with pytest.raises(OperatorParameterError):
            strict_int(dtype(value), "window", minimum=1, maximum=10)


@pytest.mark.parametrize("dtype", [int, np.int8, np.int16, np.int32, np.int64])
def test_negative_and_zero_rejected_by_derived_gates(dtype):
    with pytest.raises(OperatorParameterError):
        strict_nonnegative_int(dtype(-1), "lag")
    with pytest.raises(OperatorParameterError):
        strict_positive_int(dtype(0), "window")


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 5.0, np.float64(5),
                                  "5", None, np.nan, np.inf])
def test_non_integer_types_remain_rejected(value):
    with pytest.raises(OperatorParameterError):
        strict_int(value, "window", minimum=1, maximum=10)


def test_uint64_and_unbounded_python_int_never_round_through_float():
    large = 2**64 - 1
    assert strict_int(np.uint64(large), "id", minimum=large, maximum=large) == large
    with pytest.raises(OperatorParameterError):
        strict_int(np.uint64(large), "id", maximum=large - 1)
    larger = 2**256 + 1
    assert strict_int(larger, "id", minimum=larger, maximum=larger) == larger


def test_real_time_series_kernel_uses_shared_numpy_range_gate():
    import pandas as pd
    from factor_engine.cleaned_operators.common.time_series import TSArgmax
    panel = pd.DataFrame({"a": [1., 2., 3.]})
    for window in (0, np.int64(0)):
        with pytest.raises(OperatorParameterError):
            TSArgmax()._calculate_series(panel, window=window)
