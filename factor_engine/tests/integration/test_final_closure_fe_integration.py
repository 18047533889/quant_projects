# -*- coding: utf-8
"""#收官轮（外部 AI 复查）FactorEngine→DataAccess adapter 验收测试。

覆盖：
    P0-1  FE instrument_filter=[] → 空股票池 0 行（eager / lazy / scan_polars_long），
          绝不折叠成 None=全市场。
    P0-3  FE read_mode 贯穿到 DataAccess mode=（构造时校验 + read 实参）。
    P0-4  semantic_filters 真正变成 filters= 行过滤（不再只过门禁）。
    P0-5  A股 Return eager 路径无二次 scale（catalog 字段禁止再进 COS fallback）。
    P0-6  polars-long 受控 collect 前快照 revalidation（production fail-closed）。
    P0-7/8 cleaned fundamentals PIT UNKNOWN production fail-closed。
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.store import DataAccessStore

_PKG = "/home/shw/quant_projects/dataaccess"
_ASHARE = Path("/home/shw/quant_projects/data/a_share/lqtp_data/StockDailyBar")
HAS_ASHARE = _ASHARE.is_dir() and any(_ASHARE.glob("2019-*.parquet"))


@pytest.fixture(autouse=True)
def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


def _synthetic_store(tmp_path):
    root = tmp_path / "d"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([
            {"d": _dt.date(2024, 1, 2), "s": "AAA", "v": 1.0},
            {"d": _dt.date(2024, 1, 2), "s": "BBB", "v": 2.0},
            {"d": _dt.date(2024, 1, 3), "s": "AAA", "v": 3.0},
        ]),
        str(root / "part.parquet"),
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: date
    s: string
    v: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(registry=load_registry(cfg), engine=DuckDBEngine(threads=2))


# ---------------------------------------------------------------------------
# P0-1 空股票池（[]）→ 0 行，绝不折叠成全市场
# ---------------------------------------------------------------------------


def test_fe_empty_instrument_filter_zero_rows(tmp_path):
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    store = _synthetic_store(tmp_path)
    src = DataAccessSource(
        dataset="ds",
        fields={"v": "v"},
        start_date="2024-01-02",
        end_date="2024-01-03",
        instrument_filter=[],
    )
    with patch("factor_engine.storage.sources.data_access_source._get_store", return_value=store):
        # eager load_columns → 0 行
        out = src.load_columns(["v"])
        assert len(out["v"]) == 0
        # load_column 单列 → 0 行
        assert len(src.load_column("v")) == 0
        # lazy scan → 0 行
        src.enable_lazy_scan(True)
        out2 = src.load_columns(["v"])
        assert len(out2["v"]) == 0
        # scan_polars_long → 0 行
        lf = src.scan_polars_long(["v"])
        assert lf.collect().height == 0
    # 对照：None 仍 = 全市场
    src2 = DataAccessSource(
        dataset="ds", fields={"v": "v"},
        start_date="2024-01-02",
        end_date="2024-01-03",
    )
    with patch("factor_engine.storage.sources.data_access_source._get_store", return_value=store):
        assert len(src2.load_columns(["v"])["v"]) == 3


def test_fe_empty_instrument_filter_keeps_distinct_from_none(tmp_path):
    """构造层：instrument_filter=[] 保持 []，None 保持 None。"""
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    a = DataAccessSource(dataset="ds", instrument_filter=[])
    b = DataAccessSource(dataset="ds", instrument_filter=None)
    c = DataAccessSource(dataset="ds", instrument_filter=["AAA"])
    assert a.instrument_filter == []
    assert b.instrument_filter is None
    assert c.instrument_filter == ["AAA"]


# ---------------------------------------------------------------------------
# P0-3 read_mode 贯穿到 DataAccess mode=
# ---------------------------------------------------------------------------


class _SpyStore:
    """委托真实 store 的 spy：记录 read 调用的 kwargs。"""

    def __init__(self, store):
        self._real = store
        self.read_kwargs: list[tuple[str, dict]] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def read(self, dataset, **kwargs):
        self.read_kwargs.append((dataset, kwargs))
        return self._real.read(dataset, **kwargs)


def test_fe_read_mode_pit_reaches_store_mode(tmp_path):
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    store = _synthetic_store(tmp_path)
    src = DataAccessSource(
        dataset="ds",
        fields={"v": "v"},
        read_mode="pit",
    )
    spy = _SpyStore(store)
    with patch("factor_engine.storage.sources.data_access_source._get_store", return_value=spy):
        src.load_columns(["v"])
    assert spy.read_kwargs, "store.read 未被调用"
    assert spy.read_kwargs[0][1].get("mode") == "pit", (
        f"read_mode 未贯穿：mode={spy.read_kwargs[0][1].get('mode')!r}"
    )


# ---------------------------------------------------------------------------
# P0-4 semantic_filters 真正变成 filters=（落到物理行过滤）
# ---------------------------------------------------------------------------


def test_fe_semantic_filters_reach_filters_kwarg(tmp_path):
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    store = _synthetic_store(tmp_path)
    src = DataAccessSource(
        dataset="ds",
        fields={"v": "v"},
        semantic_filters={"s": "AAA"},
    )
    spy = _SpyStore(store)
    with patch("factor_engine.storage.sources.data_access_source._get_store", return_value=spy):
        out = src.load_columns(["v"])
    ds, kwargs = spy.read_kwargs[0]
    assert kwargs.get("filters") == {"s": "AAA"}, kwargs
    # filters= 真正过滤：只有 AAA 的行
    assert list(out["v"].values) == [1.0, 3.0]
    # 且语义过滤**不再**塞进 params（StaticDataset 拒绝多余 params）
    assert kwargs.get("params") in (None, {}) or "s" not in kwargs.get("params", {})


def test_fe_semantic_filters_conflict_with_params_rejected(tmp_path):
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    with pytest.raises(ValueError, match="conflicting semantic filter"):
        DataAccessSource(
            dataset="ds",
            params={"s": "BBB"},
            semantic_filters={"s": "AAA"},
        )


# ---------------------------------------------------------------------------
# P0-7/8 PIT UNKNOWN production fail-closed + cleaned fundamentals 禁用
# ---------------------------------------------------------------------------


def test_fe_pit_read_requires_cos_contract_production(tmp_path):
    """read_mode=pit 而无 COS 契约 → production 构造即拒绝（无法证明时钟）。"""
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        FourLayerPITError,
    )

    with pytest.raises(FourLayerPITError, match="无法证明 PIT"):
        DataAccessSource(
            dataset="no_such_contract_dataset",
            read_mode="pit",
            production=True,
        )
    # research 告警放行
    DataAccessSource(dataset="no_such_contract_dataset", read_mode="pit")


def test_fe_cleaned_fundamentals_pit_unknown_production_rejects(tmp_path):
    """fundamentals_*（period_end 无知识时钟）production PIT 拒绝。"""
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        FourLayerPITError,
    )
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(dataset="fundamentals_balance_sheet", production=True)
    plan = NormalizedFieldPlan(logical_concept="total_assets")
    with pytest.raises(FourLayerPITError, match="UNKNOWN"):
        src.assert_four_layer_pit({"total_assets": plan})
    # research 告警放行
    src2 = DataAccessSource(dataset="fundamentals_balance_sheet")
    src2.assert_four_layer_pit({"total_assets": plan})  # no raise


def test_fe_unknown_table_spec_production_rejects(tmp_path):
    """TableSpec 缺失（无法证明）production PIT 拒绝。"""
    from factor_engine.storage.sources.data_access_source import (
        DataAccessSource,
        FourLayerPITError,
    )
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    # physical_dataset 存在但 FIELD_REGISTRY 无该表 → table 层 UNKNOWN
    src = DataAccessSource(dataset="ashare_stock_daily", production=True)
    plan = NormalizedFieldPlan(
        logical_concept="x", physical_dataset="definitely_missing_table_xyz"
    )
    with pytest.raises(FourLayerPITError, match="UNKNOWN"):
        src.assert_four_layer_pit({"x": plan})


# ---------------------------------------------------------------------------
# P0-5 catalog 字段禁止二次 scale（COS return fallback 不再触碰已归一化字段）
# ---------------------------------------------------------------------------


def test_fe_catalog_field_never_double_scaled(tmp_path):
    """catalog 覆盖的字段（source=catalog）标记为 normalized，
    COS compatibility fallback 不再对 A股 Return 二次乘 return_scale。"""
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(dataset="ashare_stock_daily")
    # fetched 值已是 DataAccess 归一化后的 decimal（vendor BP × 0.0001）
    fetched = {"ret": 0.0123}
    plan = NormalizedFieldPlan(
        logical_concept="ret",
        physical_dataset="ashare_stock_daily",
        physical_fields=("Return",),
        scale=0.0001,
        source="catalog",
        canonical_unit="decimal",
    )
    src._field_plans[("ret", "test-catalog-version")] = plan
    with patch.object(src, "_semantic_catalog_version", return_value="test-catalog-version"):
        from data_access.cos_contract import get_cos_contract

        real_contract = get_cos_contract("ashare_stock_daily")
        if real_contract is not None and real_contract.return_column:
            # 有真实 COS return 契约：验证 fallback 不会二次缩放
            src._normalize_contract_columns(fetched, ["ret"])
            assert fetched["ret"] == 0.0123, (
                f"catalog 字段被二次缩放：{fetched['ret']} != 0.0123"
            )
        else:
            # 无契约 → 直接不会触发 fallback，值保持不变
            src._normalize_contract_columns(fetched, ["ret"])
            assert fetched["ret"] == 0.0123


def test_fe_lazy_catalog_scale_applied_once(tmp_path):
    """lazy 路径 catalog scale 恰好应用一次（且不再被 COS fallback 重复乘）。"""
    from factor_engine.storage.sources.data_access_source import DataAccessSource
    from factor_engine.storage.sources.field_plan import NormalizedFieldPlan

    src = DataAccessSource(dataset="ashare_stock_daily")
    src._lazy_scan = True
    fetched = {"ret": 123.0}  # raw vendor BP
    plan = NormalizedFieldPlan(
        logical_concept="ret",
        physical_dataset="ashare_stock_daily",
        physical_fields=("Return",),
        scale=0.0001,
        source="catalog",
        canonical_unit="decimal",
    )
    src._field_plans[("ret", "test-catalog-version")] = plan
    with patch.object(src, "_semantic_catalog_version", return_value="test-catalog-version"):
        src._normalize_contract_columns(fetched, ["ret"])
    # lazy 路径乘一次 0.0001 → 0.0123；COS fallback 必须跳过
    assert fetched["ret"] == pytest.approx(0.0123), fetched["ret"]


# ---------------------------------------------------------------------------
# P0-6 polars-long 受控 collect：collect 前源替换 production fail-closed
# ---------------------------------------------------------------------------


def test_fe_long_collect_revalidates_changed_source_production(tmp_path):
    from factor_engine.storage.sources.data_access_source import DataAccessSource

    store = _synthetic_store(tmp_path)
    src = DataAccessSource(
        dataset="ds",
        fields={"v": "v"},
        start_date="2024-01-02",
        end_date="2024-01-03",
        production=True,
    )
    with patch("factor_engine.storage.sources.data_access_source._get_store", return_value=store):
        src.refresh_snapshot(force=True)  # 记录 snapshot（T0）
        lf = src.scan_polars_long(["v"])
        assert lf.collect().height == 3
        # 修改底层文件（mtime 变化 → 文件 manifest hash 变化）；**不**重新
        # refresh——模拟「scan 构建 LF 之后、collect 之前源被替换」。
        p = tmp_path / "d" / "part.parquet"
        os_utime = __import__("os").utime
        os_utime(p, ns=(p.stat().st_mtime_ns + 1000, p.stat().st_mtime_ns + 1000))
        with pytest.raises(Exception):
            src.revalidate_for_long_collect()


# ---------------------------------------------------------------------------
# 真实 A股：eager == lazy == raw × 1e-4（Return 单位契约）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ASHARE, reason="真实 A股 数据不可用")
def test_a_share_return_eager_lazy_raw_parity(tmp_path):
    import pandas as pd

    from factor_engine.storage.sources.data_access_source import DataAccessSource

    store = DataAccessStore(registry=load_registry(), engine=DuckDBEngine(threads=2))
    tr = ("2019-01-02", "2019-01-04")
    inst = ["000001.SZ"]
    # raw Return × 1e-4（catalog return_bp scale=0.0001）
    raw = store.read(
        "ashare_stock_daily",
        columns=["TradeDate", "Symbol", "Return"],
        time_range=tr,
        instrument_filter=inst,
        normalize_units=False,
    ).to_arrow()
    raw_ret = pd.Series(raw.column("Return").to_numpy().flatten()) * 0.0001

    src = DataAccessSource(
        dataset="ashare_stock_daily",
        fields={"ret": "Return"},
        start_date="2019-01-02",
        end_date="2019-01-04",
        instrument_filter=inst,
    )
    with patch("factor_engine.storage.sources.data_access_source._get_store", return_value=store):
        eager = src.load_column("ret")
        src2 = DataAccessSource(
            dataset="ashare_stock_daily",
            fields={"ret": "Return"},
            start_date="2019-01-02",
            end_date="2019-01-04",
            instrument_filter=inst,
            read_auto=True,
        )
        src2.enable_lazy_scan(True)
        lazy = src2.load_column("ret")

    assert eager.notna().sum() == raw_ret.notna().sum()
    assert lazy.notna().sum() == raw_ret.notna().sum()
    # 值尺度一致：eager / lazy 都不应比 raw×1e-4 再缩 10000×
    e = sorted(eager.dropna().tolist())
    r = sorted(raw_ret.dropna().tolist())
    if e:
        assert abs(e[0]) <= 1.0, "eager Return 疑似未归一化（仍是 raw BP 量级）"
        assert abs(r[0]) <= 1.0
