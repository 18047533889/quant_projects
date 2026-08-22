# -*- coding: utf-8
"""Rolling warmup parameter parity across production backends."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from cleaned_operators.operator_surface import DAILY_CANONICALS
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.backend_parity.test_production_core_triple_parity import (
    _result_series,
    _run,
    duckdb_source,
    mem_source,
)


_WARMUP_CASES = [
    pytest.param(1, 1, [False, False, True, False, False], id="window-one"),
    pytest.param(3, 1, [False, False, True, False, False], id="partial-window"),
    pytest.param(3, 2, [True, False, True, False, False], id="minimum-support"),
    pytest.param(3, 3, [True, True, True, True, True], id="full-window-support"),
]


def _rank_expr(column: str, window: int, min_periods: int):
    return make_cleaned_call_factory("ts_rank")(
        col(column), window, min_periods=min_periods
    )


@pytest.mark.parametrize("window,min_periods,expected_a_nulls", _WARMUP_CASES)
def test_ts_rank_warmup_and_window_boundaries_match_all_backends(
    mem_source,
    duckdb_source,
    window,
    min_periods,
    expected_a_nulls,
):
    assert "ts_rank" in DAILY_CANONICALS

    memory_expr = _rank_expr("close", window, min_periods)
    pandas_out = _result_series(_run(mem_source, memory_expr, "pandas"))
    polars_run = _run(mem_source, memory_expr, "polars_long")
    assert polars_run.get("used_polars_long_path") is True, polars_run
    assert not polars_run.get("polars_long_fallback_reason"), polars_run
    polars_out = _result_series(polars_run)

    sql_expr = _rank_expr("Close", window, min_periods)
    duckdb_run = _run(duckdb_source, sql_expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(duckdb_run)
    duckdb_out = _result_series(duckdb_run)

    pd.testing.assert_series_equal(
        pandas_out, polars_out, check_names=False, rtol=1e-6, atol=1e-6
    )
    pd.testing.assert_series_equal(
        pandas_out, duckdb_out, check_names=False, rtol=1e-6, atol=1e-6
    )

    a_out = pandas_out.xs("A", level="instrument")
    assert a_out.isna().tolist() == expected_a_nulls
    if window == 1:
        assert a_out.dropna().eq(1.0).all()
    else:
        b_out = pandas_out.xs("B", level="instrument")
        first_valid = min_periods - 1
        assert b_out.iloc[:first_valid].isna().all()
        assert b_out.iloc[first_valid:].notna().all()
