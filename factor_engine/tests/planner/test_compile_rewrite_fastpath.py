# -*- coding: utf-8
"""compile 管线接入 rewrite_fastpath。"""
from __future__ import annotations

import pandas as pd
import pytest

from api.factor import Factor
from backend.factory import build_backend
from planner.logical_plan import PlanNode
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _walk_ops(plan: PlanNode) -> set[str]:
    out = {plan.op}
    for c in plan.inputs:
        out |= _walk_ops(c)
    return out


@pytest.fixture(scope="module")
def _loaded():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


@pytest.fixture
def engine(_loaded):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01", "2024-01-02"]), ["A"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0], index=idx)
    volume = pd.Series([100.0, 200.0], index=idx)
    src = InMemorySeriesSource(data={"close": close, "volume": volume})
    return FactorEngine(backend=build_backend("pandas"), data_source=src)


def test_compile_rewrites_divide_to_protected_div(engine):
    from api import divide
    from api.columns import col

    factor = Factor(name="rw_div", expr=divide(col("close"), col("volume")))
    plan, _ = engine.compile(factor)
    assert "protected_div" in _walk_ops(plan)
    assert "divide" not in _walk_ops(plan)
