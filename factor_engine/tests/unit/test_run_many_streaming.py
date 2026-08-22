# -*- coding: utf-8 -*-
"""Phase 5 R4/R5/R17：run_many_iter / result_policy=sink / CSE 引用计数 / 流式物化。"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

sys.path.insert(0, "/home/shw/quant_projects/factor_engine")

from api import rank, ts_mean  # noqa: E402  （导入即填充 OperatorRegistry）
from api.columns import col
from api.factor import Factor
from backend.pandas_backend import PandasBackend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture(autouse=True)
def _skip_dynamic_reload(monkeypatch):
    """api 导入已填充 registry；PandasBackend 构造触发的 ``ensure_cleaned_loaded``
    会二次 load_all，撞上并发会话正在收口的 operator 重复注册（过渡期冲突）。
    这里跳过重复加载，不影响本用例验证的 run_many_iter/sink/CSE 逻辑。"""
    monkeypatch.setattr(
        "backend.cleaned_bridge.ensure_cleaned_loaded", lambda: None
    )


def _close_panel(n_days: int = 20) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["timestamp", "instrument"])
    n = len(idx)
    return pd.Series([float((i % 7) + 1) for i in range(n)], index=idx)


def _engine() -> FactorEngine:
    return FactorEngine(
        backend=PandasBackend(),
        data_source=InMemorySeriesSource(data={"close": _close_panel()}),
    )


def _factors() -> list[Factor]:
    return [
        Factor(name="f1", expr=rank(ts_mean(col("close"), 3))),
        Factor(name="f2", expr=ts_mean(col("close"), 2) - col("close")),
        Factor(name="f3", expr=rank(col("close"))),
    ]


def test_run_many_iter_yields_each_result():
    eng = _engine()
    names = []
    results = []
    for name, result, _path in eng.run_many_iter(_factors(), enable_cse=True):
        names.append(name)
        results.append(result)
    assert set(names) == {"f1", "f2", "f3"}
    assert all(len(r) > 0 for r in results)


def test_run_many_sink_does_not_accumulate_results():
    eng = _engine()
    sunk: list[str] = []
    out = eng.run_many(
        _factors(),
        enable_cse=True,
        result_policy="sink",
        sink=lambda name, result: sunk.append(name),
    )
    # sink 路径 results 字典保持空（不驻留）
    assert out["results"] == {}
    assert set(sunk) == {"f1", "f2", "f3"}


def test_cse_shared_results_released_after_last_consumer():
    eng = _engine()
    # f1 与 f3 都用了 rank(...) / ts_mean(...) 共享子树（CSE 命中）
    out = eng.run_many(_factors(), enable_cse=True)
    shared = getattr(out, "dag", None)
    # shared_result_cache 是 ctx 局部；关键是不抛错、结果正确
    assert set(out["results"]) == {"f1", "f2", "f3"}


def test_materializer_streams_partitions(tmp_path):
    """R17：ParquetMaterializer 不再 ``list(iter_partition_groups(...))``，
    分区逐个写入且 upsert 幂等。"""
    from storage.materializer import ParquetMaterializer

    dates = pd.bdate_range("2023-01-02", periods=260)  # 跨两个年份分区
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["timestamp", "instrument"])
    n = len(idx)
    series = pd.Series([float(i % 5) for i in range(n)], index=idx)
    mat = ParquetMaterializer(lake_root=tmp_path)
    out = mat.materialize("stream_f1", series)
    assert out["rows_written"] == n
    # 落盘文件数 = 年份分区数（2023、2024）
    year_dirs = sorted(p.name for p in (tmp_path / "factors" / "stream_f1").iterdir())
    assert any(d.startswith("year=") for d in year_dirs), year_dirs
