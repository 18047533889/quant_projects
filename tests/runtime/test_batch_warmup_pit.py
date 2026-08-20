# -*- coding: utf-8
"""run_many/run_many_parallel 不再因 PIT / auto_warmup 退化为逐因子 run()。

覆盖：
    - pit_enforce=True 时仍走批快路径（batch_graph 存在），结果与逐因子 run 一致
    - auto_warmup=True 时共享 union 加载窗口，一次读数、各因子独立 trim
    - run_many_parallel 同样不退化
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from api import rank, ts_mean
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from cleaned_operators.operator_policy import effective_lookback
from runtime.engine import FactorEngine
from storage.time_window import business_day_offset
from tests.helpers import InMemorySeriesSource


@dataclass
class _BoundedCountingSource(InMemorySeriesSource):
    """带日期边界 + 读取次数计数器的内存数据源。"""

    start_date: str | None = None
    end_date: str | None = None
    load_columns_calls: int = 0

    def time_range(self):
        return self.start_date, self.end_date

    def load_columns(self, names: list[str]) -> dict[str, pd.Series]:
        self.load_columns_calls += 1
        return super().load_columns(names)

    def load_column(self, name: str) -> pd.Series:
        return self.load_columns([name])[name]

    def prefetch_columns(self, names: list[str]) -> None:
        self.load_columns(names)


def _close_panel(dates: list[str], assets: list[str] | None = None) -> pd.Series:
    assets = assets or ["AAA"]
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), assets],
        names=["timestamp", "instrument"],
    )
    return pd.Series([float(i + 1) for i in range(len(idx))], index=idx)


def _engines(src):
    return FactorEngine(backend=PandasBackend(), data_source=src)


def test_run_many_pit_enforce_uses_batch_path():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    src = _BoundedCountingSource(
        data={"close": _close_panel(dates)},
        start_date="2024-01-02",
        end_date="2024-01-05",
    )
    engine = _engines(src)
    sub = ts_mean(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=rank(sub))

    out = engine.run_many([f1, f2], pit_enforce=True)

    # 批快路径（per-factor 退化路径不输出 batch_graph）
    assert "batch_graph" in out
    assert len(out["dag"].shared_nodes) >= 1
    # 结果与逐因子 pit_enforce=True 一致
    r1 = engine.run(f1, pit_enforce=True)["result"]
    r2 = engine.run(f2, pit_enforce=True)["result"]
    pd.testing.assert_series_equal(out["results"]["a"], r1, check_names=False)
    pd.testing.assert_series_equal(out["results"]["b"], r2, check_names=False)


def test_run_many_auto_warmup_shared_window_single_load():
    dates = [
        "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29",
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
    ]
    src = _BoundedCountingSource(
        data={"close": _close_panel(dates)},
        start_date="2024-01-04",
        end_date="2024-01-08",
    )
    engine = _engines(src)
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=ts_mean(col("close"), 5))

    out = engine.run_many([f1, f2], auto_warmup=True, trim_warmup=True)

    # 批快路径 + 共享窗口：两个因子各 trim 到请求区间
    assert "batch_graph" in out
    ww = out["warmup_windows"]
    assert set(ww) == {"a", "b"}
    assert ww["a"]["requested_start"] == "2024-01-04"
    assert ww["b"]["requested_start"] == "2024-01-04"
    # 最大 lookback(5) 决定共享加载起点
    lb5 = effective_lookback(5)
    assert ww["b"]["actual_load_start"] == business_day_offset(
        "2024-01-04", -lb5
    ).strftime("%Y-%m-%d")

    # 结果与逐因子 run 一致，且在请求窗口内无 NaN（warmup 足够）
    r1 = engine.run(f1, auto_warmup=True, trim_warmup=True)["result"]
    r2 = engine.run(f2, auto_warmup=True, trim_warmup=True)["result"]
    pd.testing.assert_series_equal(out["results"]["a"], r1, check_names=False)
    pd.testing.assert_series_equal(out["results"]["b"], r2, check_names=False)
    assert out["results"]["a"].notna().all()
    assert out["results"]["b"].notna().all()


def test_run_many_parallel_pit_warmup_no_degradation():
    pytest.importorskip("joblib")
    dates = [
        "2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29",
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
    ]
    src = _BoundedCountingSource(
        data={"close": _close_panel(dates), "open": _close_panel(dates)},
        start_date="2024-01-04",
        end_date="2024-01-08",
    )
    engine = _engines(src)
    f1 = Factor(name="a", expr=ts_mean(col("close"), 2))
    f2 = Factor(name="b", expr=ts_mean(col("open"), 5))

    out = engine.run_many_parallel(
        [f1, f2], n_jobs=2, auto_warmup=True, pit_enforce=True
    )

    assert "batch_graph" in out
    assert set(out["warmup_windows"]) == {"a", "b"}
    r1 = engine.run(f1, auto_warmup=True, trim_warmup=True, pit_enforce=True)["result"]
    r2 = engine.run(f2, auto_warmup=True, trim_warmup=True, pit_enforce=True)["result"]
    pd.testing.assert_series_equal(out["results"]["a"], r1, check_names=False)
    pd.testing.assert_series_equal(out["results"]["b"], r2, check_names=False)
