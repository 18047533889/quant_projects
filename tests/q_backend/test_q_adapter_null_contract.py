"""Deterministic Q adapter boundary contracts; no PyKX runtime required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.q_backend.q_adapter import QType, QTypeAdapter, QZeroCopyUnavailable


class FakeQ:
    def __init__(self, *, zero_copy_error: Exception | None = None):
        self.zero_copy_error = zero_copy_error
        self.calls: list[tuple[pd.DataFrame, bool]] = []

    def toq(self, frame: pd.DataFrame, *, zero_copy: bool):
        self.calls.append((frame, zero_copy))
        if zero_copy and self.zero_copy_error is not None:
            raise self.zero_copy_error
        return frame


def test_object_null_empty_string_and_values_remain_distinct():
    adapter = QTypeAdapter()
    q = FakeQ()
    source = pd.DataFrame({"label": [None, pd.NA, "", "alpha"]}, dtype=object)

    result = adapter.pandas_to_q(
        source, schema={"label": QType.SYMBOL}, q_module=q
    )

    values = result["label"].tolist()
    assert pd.isna(values[0]) and pd.isna(values[1])
    assert values[2] == ""
    assert values[3] == "alpha"
    assert result["label"].dtype == "string"


def test_object_without_schema_fails_closed_instead_of_guessing_symbol():
    adapter = QTypeAdapter()
    with pytest.raises(TypeError, match="explicit QType schema"):
        adapter._prepare_pandas_for_q(pd.DataFrame({"label": [None, ""]}))


def test_integer_null_sentinels_are_width_specific():
    adapter = QTypeAdapter()
    source = pd.DataFrame(
        {
            "short": np.array([adapter.semantics.null_short, 7], dtype=np.int16),
            "int": np.array([adapter.semantics.null_int, 7], dtype=np.int32),
            "long": np.array([adapter.semantics.null_long, 7], dtype=np.int64),
        }
    )

    result = adapter._handle_q_nulls(
        source,
        schema={"short": QType.SHORT, "int": QType.INT, "long": QType.LONG},
    )

    assert result.dtypes.tolist() == [pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()]
    assert all(pd.isna(result[col].iloc[0]) for col in result)
    assert all(result[col].iloc[1] == 7 for col in result)


def test_nullable_boolean_preserves_true_false_and_null():
    adapter = QTypeAdapter()
    q = FakeQ()
    source = pd.DataFrame({"flag": pd.Series([True, False, pd.NA], dtype="boolean")})

    result = adapter.pandas_to_q(source, schema={"flag": QType.BOOLEAN}, q_module=q)

    assert result["flag"].dtype == "boolean"
    assert result["flag"].tolist() == [True, False, pd.NA]


def test_semantic_conversion_errors_are_not_fallbacked():
    adapter = QTypeAdapter()
    q = FakeQ()
    source = pd.DataFrame({"amount": ["not-a-number"]})

    with pytest.raises(ValueError, match="could not convert string to float"):
        adapter.pandas_to_q(
            source, schema={"amount": QType.FLOAT}, zero_copy=True, q_module=q
        )
    assert q.calls == []


def test_only_explicit_zero_copy_unavailable_falls_back():
    adapter = QTypeAdapter()
    q = FakeQ(zero_copy_error=QZeroCopyUnavailable("layout unavailable"))
    source = pd.DataFrame({"amount": np.array([1.0], dtype=np.float64)})

    result = adapter.pandas_to_q(source, zero_copy=True, q_module=q)

    assert result["amount"].tolist() == [1.0]
    assert [zero_copy for _, zero_copy in q.calls] == [True, False]


def test_type_error_from_q_boundary_is_not_fallbacked():
    adapter = QTypeAdapter()
    q = FakeQ(zero_copy_error=TypeError("semantic dtype error"))
    source = pd.DataFrame({"amount": np.array([1.0], dtype=np.float64)})

    with pytest.raises(TypeError, match="semantic dtype error"):
        adapter.pandas_to_q(source, zero_copy=True, q_module=q)
    assert [zero_copy for _, zero_copy in q.calls] == [True]
