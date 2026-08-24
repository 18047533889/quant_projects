# -*- coding: utf-8
"""SQL IO 与 fully_sql prefetch 跳过逻辑。"""

from __future__ import annotations

from factor_engine.api import rank, ts_mean, ts_sharpe
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.factory import build_backend
from factor_engine.planner.sql_io import plan_is_fully_sql, should_skip_column_prefetch
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _compile_plan(expr):
    ensure_cleaned_loaded()
    idx = __import__("pandas").MultiIndex.from_product(
        [[__import__("pandas").Timestamp("2024-01-01")], ["A"]],
        names=["timestamp", "instrument"],
    )
    close = __import__("pandas").Series([1.0], index=idx)
    source = InMemorySeriesSource(data={"close": close, "ret": close})
    eng = FactorEngine(backend=build_backend("pandas"), data_source=source)
    plan, _ = eng.compile(Factor(name="t", expr=expr))
    return plan


def test_rank_ts_mean_is_fully_sql():
    plan = _compile_plan(rank(ts_mean(col("close"), 3)))
    assert plan_is_fully_sql(plan)


def test_ts_sharpe_is_fully_sql():
    plan = _compile_plan(ts_sharpe(col("ret"), 20))
    assert plan_is_fully_sql(plan)


def test_should_skip_prefetch_without_dq():
    plan = _compile_plan(rank(ts_mean(col("close"), 3)))
    assert should_skip_column_prefetch([plan], input_dq_check=False)


def test_should_not_skip_prefetch_when_dq_enabled():
    plan = _compile_plan(rank(ts_mean(col("close"), 3)))
    assert not should_skip_column_prefetch([plan], input_dq_check=True)
