from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pytest

from data_access.adapters import arrow_table_to_multiindex_columns
from data_access.exceptions import ValidationError
from data_access.key_policy import KeyPolicy


def _table(revisions: list[int], values: list[float]) -> pa.Table:
    return pa.table(
        {
            "ts": [datetime(2024, 1, 2)] * len(revisions),
            "inst": ["AAPL"] * len(revisions),
            "revision_id": revisions,
            "value": values,
        }
    )


def test_duplicate_resolution_selects_highest_revision() -> None:
    out = arrow_table_to_multiindex_columns(
        _table([1, 3, 2], [10.0, 30.0, 20.0]),
        timestamp_column="ts",
        instrument_column="inst",
        value_columns=["value"],
        key_policy=KeyPolicy(
            invalid_key="error",
            duplicate_key="error",
            duplicate_resolution="revision_id",
        ),
    )

    series = out["value"]
    assert len(series) == 1
    assert series.iloc[0] == 30.0


def test_duplicate_resolution_rejects_equal_revision_ties() -> None:
    with pytest.raises(ValidationError, match="版本列无法提供唯一顺序"):
        arrow_table_to_multiindex_columns(
            _table([2, 2], [10.0, 20.0]),
            timestamp_column="ts",
            instrument_column="inst",
            value_columns=["value"],
            key_policy=KeyPolicy(duplicate_resolution="revision_id"),
        )


def test_duplicate_resolution_rejects_null_revision() -> None:
    table = pa.table(
        {
            "ts": [datetime(2024, 1, 2)],
            "inst": ["AAPL"],
            "revision_id": pa.array([None], type=pa.int64()),
            "value": [10.0],
        }
    )
    with pytest.raises(ValidationError, match="存在 1 行 NULL"):
        arrow_table_to_multiindex_columns(
            table,
            timestamp_column="ts",
            instrument_column="inst",
            value_columns=["value"],
            key_policy=KeyPolicy(duplicate_resolution="revision_id"),
        )


def test_invalid_key_policy_values_fail_fast() -> None:
    with pytest.raises(ValidationError, match="invalid_key"):
        KeyPolicy(invalid_key="ignore")
    with pytest.raises(ValidationError, match="duplicate_key"):
        KeyPolicy(duplicate_key="first")
    with pytest.raises(ValidationError, match="独立版本列"):
        KeyPolicy(duplicate_resolution="timestamp")
