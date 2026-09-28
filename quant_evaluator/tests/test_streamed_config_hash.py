"""Exact-compatibility checks for bounded EvaluationConfig.v2 array hashing."""
import numpy as np
import pytest

from quant_evaluator.contracts._hashutil import (
    stable_content_hex, stable_content_hex_streamed_arrays,
)


@pytest.mark.parametrize("chunk_bytes", [3, 4, 7, 31, 3 * 1024 * 1024])
def test_streamed_arrays_match_v2_for_chunk_boundaries(chunk_bytes):
    bits = np.array([0x7FF8000000000001, 0x8000000000000000,
                     0x3FF0000000000000, 0], dtype=np.uint64)
    values = np.tile(bits.view(np.float64), 19).reshape(2, 19, 2)
    validity = np.isfinite(values)
    fields = {
        "factor_values": values,
        "factor_validity": validity,
        "factor_ids": ("a", "β"),
        "versions": {"rank_ic": "4.0.0"},
        "parameters": {"rank_ic": {"min_assets": 2}},
        "label_hash": "abc",
        "none": None,
    }
    expected = stable_content_hex(tag="EvaluationConfig.v2", fields=fields)
    actual = stable_content_hex_streamed_arrays(
        tag="EvaluationConfig.v2", fields=fields,
        array_keys=("factor_values", "factor_validity"), chunk_bytes=chunk_bytes)
    assert actual == expected


@pytest.mark.parametrize("array", [
    np.array(1.5),
    np.empty((0, 3), dtype=np.float64),
    np.arange(24, dtype=np.int8).reshape(2, 3, 4),
    np.array([True, False, True]),
    np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
    np.arange(48, dtype=np.float64).reshape(4, 4, 3)[:, ::2, :],
])
def test_streamed_arrays_match_v2_for_dtype_shape_and_layout(array):
    fields = {"factor_values": array, "factor_validity": None}
    assert stable_content_hex_streamed_arrays(
        tag="EvaluationConfig.v2", fields=fields,
        array_keys=("factor_values", "factor_validity"), chunk_bytes=5
    ) == stable_content_hex(tag="EvaluationConfig.v2", fields=fields)


def test_streamed_arrays_preserve_reserved_mapping_and_fail_closed():
    fields = {"__bytes__": "literal", "factor_values": np.arange(5)}
    assert stable_content_hex_streamed_arrays(
        tag="t", fields=fields, array_keys=("factor_values",)
    ) == stable_content_hex(tag="t", fields=fields)
    with pytest.raises(TypeError):
        stable_content_hex_streamed_arrays(
            tag="t", fields={"factor_values": np.array([object()], dtype=object)},
            array_keys=("factor_values",))
    with pytest.raises(TypeError):
        stable_content_hex_streamed_arrays(
            tag="t", fields={1: np.arange(3)}, array_keys=("factor_values",))
    with pytest.raises(ValueError):
        stable_content_hex_streamed_arrays(
            tag="t", fields={"factor_values": np.arange(3)},
            array_keys=("factor_values",), chunk_bytes=2)
