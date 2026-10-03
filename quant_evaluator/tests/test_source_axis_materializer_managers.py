"""Storage-manager compatibility for bounded source materialization."""
import warnings

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.adapters.source_axis_materializer import write_axis_aligned_float64


@pytest.mark.parametrize("bounded", [False, True])
def test_array_manager_frame_matches_float64_alignment(bounded):
    try:
        pd.get_option("mode.data_manager")
    except pd.errors.OptionError:
        pytest.skip("pandas removed the optional ArrayManager")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pd.option_context("mode.data_manager", "array"):
            frame = pd.DataFrame({"A": [1., np.nan, 3.], "B": [4., 5., 6.]},
                                 index=[10, 11, 12])
    output = np.empty((2, 2))
    write_axis_aligned_float64(frame, [11, 12], ["B", "A"], output,
                               memory_bounded=bounded, chunk_bytes=64)
    np.testing.assert_equal(output, [[5., np.nan], [6., 3.]])


def test_array_manager_alias_rejected_before_bounded_write():
    try:
        pd.get_option("mode.data_manager")
    except pd.errors.OptionError:
        pytest.skip("pandas removed the optional ArrayManager")
    backing = np.arange(8, dtype=float).reshape(4, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pd.option_context("mode.data_manager", "array"):
            frame = pd.DataFrame({"A": backing[:, 0], "B": backing[:, 1]},
                                 index=range(4), copy=False)
    before = backing.copy()
    with pytest.raises(ValueError, match="must not share storage"):
        write_axis_aligned_float64(frame, range(4), ["A", "B"],
                                   backing, memory_bounded=True)
    np.testing.assert_equal(backing, before)
