"""Polars auto 路径：有 polars 实现时优先选用。"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("polars")

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.cleaned_bridge import make_cleaned_kernel, _resolve_operator
from backend.context import ExecutionContext
from backend.factory import build_backend
from backend.pandas_backend import PandasBackend
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from runtime.engine import FactorEngine
from runtime.perf_config import PerfConfig
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def panel_source():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5, freq="D"), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    close = pd.Series(range(10, 10 + len(idx)), index=idx, dtype=float)
    return InMemorySeriesSource(data={"close": close})


def test_auto_prefers_polars_for_ts_mean():
    load_all()
    op, backend = OperatorRegistry.get_preferred("ts_mean", prefer="auto")
    assert backend == "polars"
    assert op is not None


def test_pandas_only_ops_fallback():
    load_all()
    op, backend = OperatorRegistry.get_preferred("ts_moment", prefer="auto")
    assert backend == "pandas_numpy"
    assert op is not None


def test_polars_backend_uses_polars_kernel(panel_source):
    load_all()
    ctx = ExecutionContext(data_source=panel_source, perf=PerfConfig(operator_backend="auto"))
    op, backend = _resolve_operator("ts_mean", ctx)
    assert backend == "polars"

    backend_obj = PandasBackend()
    kernel = make_cleaned_kernel(backend_obj._eval, "ts_mean")
    from planner.logical_plan import PlanNode

    col_node = PlanNode(op="column", inputs=[], attrs={"name": "close"})
    node = PlanNode(op="ts_mean", inputs=[col_node], attrs={"window": 2})
    ctx.panel_cache = {}
    ctx.template_series = panel_source.load_column("close")
    result = kernel(node, ctx)
    assert len(result) > 0


def test_polars_engine_matches_pandas(panel_source):
    expr = rank(ts_mean(col("close"), 2))
    eng_pd = FactorEngine(backend=build_backend("pandas"), data_source=panel_source)
    eng_pl = FactorEngine(backend=build_backend("polars"), data_source=panel_source)
    a = eng_pd.run(Factor(name="t", expr=expr))["result"]
    b = eng_pl.run(Factor(name="t", expr=expr))["result"]
    pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-10, atol=1e-10)


def test_removed_bfill_has_no_backend():
    load_all()
    assert OperatorRegistry.get("bfill", backend="pandas_numpy") is None
    assert OperatorRegistry.get("bfill", backend="polars") is None
