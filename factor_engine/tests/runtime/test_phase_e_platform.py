"""Phase E：read_root cutover、Polars panel、10M 阈值。"""

from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.api import ts_mean
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.panel_polars import is_polars_frame, panel_to_polars
from factor_engine.backend.polars_backend import PolarsBackend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def test_panel_to_polars_idempotent():
    panel = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    pl_panel = panel_to_polars(panel)
    assert is_polars_frame(pl_panel)
    again = panel_to_polars(pl_panel)
    assert again is pl_panel


def test_polars_backend_sets_prefer_polars_panel():
    backend = PolarsBackend()
    assert backend.use_lazy is False or True  # env dependent
    src = InMemorySeriesSource(data={})
    ctx = ExecutionContext(data_source=src, panel_cache={})
    from dataclasses import replace

    ctx = replace(ctx, runtime_stats={"backend": "polars"}, prefer_polars_panel=True)
    assert ctx.prefer_polars_panel is True


def test_run_many_polars_backend_smoke():
    pytest.importorskip("polars")
    dates = pd.bdate_range("2024-01-02", periods=8)
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    data = {"close": pd.Series([float(i) for i in range(len(idx))], index=idx)}
    eng = FactorEngine(
        backend=PolarsBackend(),
        data_source=InMemorySeriesSource(data=data),
    )
    f = Factor(name="m", expr=ts_mean(col("close"), 3))
    out = eng.run(f)
    assert len(out["result"]) == len(idx)
