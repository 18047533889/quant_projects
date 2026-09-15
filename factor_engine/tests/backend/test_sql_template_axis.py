"""Exact SQL result/template axis contract regressions."""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.sql_pushdown.executor import (
    SqlResultSchemaError,
    _align_to_explicit_template,
)
from tests.helpers import InMemorySeriesSource


def _index(unit: str, assets=("A", "B")) -> pd.MultiIndex:
    timestamps = pd.DatetimeIndex(["2024-01-02", "2024-01-02"]).as_unit(unit)
    return pd.MultiIndex.from_arrays(
        [timestamps, list(assets)], names=["timestamp", "instrument"]
    )


def test_explicit_template_preserves_dtype_keys_and_values():
    result = pd.Series([2.0, 1.0], index=_index("s", assets=("B", "A")))
    template = _index("ns")
    ctx = ExecutionContext(data_source=InMemorySeriesSource(data={}))
    ctx.template_index = template

    aligned = _align_to_explicit_template(result, ctx)

    assert aligned.index.equals(template)
    assert aligned.index.levels[0].dtype == template.levels[0].dtype
    assert aligned.tolist() == [1.0, 2.0]


def test_explicit_template_rejects_different_key_set():
    result = pd.Series([1.0, 2.0], index=_index("s"))
    template = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).as_unit("us"), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    ctx = ExecutionContext(data_source=InMemorySeriesSource(data={}))
    ctx.template_index = template

    with pytest.raises(SqlResultSchemaError, match="key set"):
        _align_to_explicit_template(result, ctx)
