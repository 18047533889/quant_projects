"""#33 正确性 gates：两市场最危险的单位 / PIT / 表模型 / 日历 / Capital 语义。

这些比「字段有没有成功读出来」更值得固定：
    - A股 Return(bp)/10000 vs 美股 Ret 小数不除；
    - A/美 Factor 都是乘法后复权；
    - A 财务 PubDate vs 美股 filing_date（+timeframe 强制）；
    - 旧报告期晚修订不得回滚（period_selection=latest_period）；
    - US timeframe 缺失必须报错；X0/EMPTY panel 必须拒绝；
    - A 分钟 QuoteTime(UTC)→Asia/Shanghai、午休 bar index 正确；
    - effective-only 事件未来数据不前视；跨市场字段歧义 fail-closed。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.exceptions import ValidationError
from data_access.core.engine import DuckDBEngine
from data_access.read.aggregation import (
    AggregationItem,
    AggregationSpec,
    aggregate_minute_bundle,
    aggregate_minute_to_daily,
)
from data_access.registry import load_registry
from data_access.store import DataAccessStore


def _write_cfg(tmp_path: Path, datasets: dict[str, dict]) -> Path:
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    lines.append(f"    {sk}: {sv}")
            else:
                # 全加双引号，避免 glob:*.parquet 被 YAML 当 alias 解析
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return cfg


def _store(tmp_path: Path, datasets: dict[str, dict]) -> DataAccessStore:
    cfg = _write_cfg(tmp_path, datasets)
    return DataAccessStore(load_registry(cfg), DuckDBEngine())


@pytest.fixture(autouse=True)
def _skip_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")


# ---------------------------------------------------------------------------
# 单位：A股 Return(bp)/10000，美股 Ret 小数不除
# ---------------------------------------------------------------------------

def _price_store(tmp_path: Path) -> DataAccessStore:
    ashare = tmp_path / "ashare"
    ashare.mkdir()
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02"]).date,
            "Symbol": ["000001.SZ"],
            "Close": [10.0],
            "Return": [100],  # bp
        }
    ).to_parquet(ashare / "2024-01-02.parquet")
    us = tmp_path / "us"
    us.mkdir()
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02"]).date,
            "Ticker": ["AAPL"],
            "Close": [190.0],
            "Ret": [0.0123],  # 小数
        }
    ).to_parquet(us / "2024-01-02.parquet")
    return _store(
        tmp_path,
        {
            "ashare_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "format": "parquet", "root": str(ashare), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string", "Close": "double", "Return": "int"},
            },
            "us_stock_daily": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "format": "parquet", "root": str(us), "glob": "*.parquet",
                "time_column": "TradeDate", "instrument_column": "Ticker",
                "schema": {"TradeDate": "date", "Ticker": "string", "Close": "double", "Ret": "double"},
            },
        },
    )


def test_ashare_return_is_bp_not_divided(tmp_path):
    """A股 Return 是 bp（+100 = +1%），读取归一化后除以 10000。"""
    from data_access.cos_contract import normalize_return_values

    store = _price_store(tmp_path)
    t = store.read_arrow("ashare_stock_daily", columns=["Return"])
    raw = int(t.column("Return").to_pylist()[0])
    assert raw == 100  # 原始 bp
    norm = normalize_return_values(t.column("Return"), "ashare_stock_daily")
    assert float(norm.to_pylist()[0]) == pytest.approx(0.01)  # 100bp = 1%


def test_us_ret_not_divided(tmp_path):
    """美股 Ret 是小数，禁止再除。"""
    from data_access.cos_contract import normalize_return_values

    store = _price_store(tmp_path)
    t = store.read_arrow("us_stock_daily", columns=["Ret"])
    raw = float(t.column("Ret").to_pylist()[0])
    assert raw == pytest.approx(0.0123)
    norm = normalize_return_values(t.column("Ret"), "us_stock_daily")
    assert float(norm.to_pylist()[0]) == pytest.approx(0.0123)  # 乘 1.0 不变


# ---------------------------------------------------------------------------
# 表模型：X0/EMPTY 禁止当普通面板；美股财务必须 filter timeframe
# ---------------------------------------------------------------------------

def test_x0_sparse_panel_rejected(tmp_path):
    """美股 X0 稀疏表（valuation/indicator）不能当完整面板读。"""
    from data_access.cos_contract import validate_panel_request

    with pytest.raises(ValidationError):
        validate_panel_request("us_stock_valuation_daily")
    # 显式 allow_sparse 才放行
    contract = validate_panel_request("us_stock_valuation_daily", allow_sparse=True)
    assert contract.is_sparse
    # 非 X0 数据集正常通过
    validate_panel_request("ashare_stock_daily")


def test_empty_panel_rejected(tmp_path):
    """EMPTY 占位表（us_ticker_alias）禁止读取。"""
    store = _price_store(tmp_path)
    with pytest.raises(ValidationError):
        store.read_arrow("us_ticker_alias")


def test_us_financial_timeframe_required(tmp_path):
    """美股财务 E2 必须 filter timeframe（annual/quarterly/trailing_twelve_months）。"""
    from data_access.read.semantic_catalog import get_semantic_catalog

    f = get_semantic_catalog().resolve_one("total_assets", dataset="us_stock_balance")
    assert f is not None
    assert "timeframe" in f.required_filters


# ---------------------------------------------------------------------------
# 财务 PIT：旧报告期晚修订不回滚（period_selection=latest_period）
# ---------------------------------------------------------------------------

def _fin_store(tmp_path: Path) -> tuple[DataAccessStore, str, str]:
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    pd.DataFrame(
        {
            "TradeDate": pd.to_datetime(["2024-01-02", "2024-07-02", "2024-08-02"]).date,
            "Symbol": ["A"] * 3,
        }
    ).to_parquet(anchor / "a.parquet")
    fin = tmp_path / "fin"
    fin.mkdir()
    pd.DataFrame(
        {
            "Symbol": ["A", "A", "A"],
            "PubDate": pd.to_datetime(["2024-01-01", "2024-07-01", "2024-08-01"]).date,
            "ReportPeriodEndDate": pd.to_datetime(
                ["2023-12-31", "2024-06-30", "2023-12-31"]
            ).date,
            "UpdateTime": pd.to_datetime(
                ["2024-01-01 09:00", "2024-07-01 09:00", "2024-08-01 09:00"]
            ),
            "NetProfit": [100.0, 200.0, 999.0],  # 旧期(2023-12-31)晚修订
        }
    ).to_parquet(fin / "f.parquet")
    store = _store(
        tmp_path,
        {
            "daily_ds": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "format": "parquet", "root": str(anchor), "glob": "a.parquet",
                "time_column": "TradeDate", "instrument_column": "Symbol",
                "schema": {"TradeDate": "date", "Symbol": "string"},
            },
            "fin_ds": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "format": "parquet", "root": str(fin), "glob": "f.parquet",
                "time_column": "PubDate", "instrument_column": "Symbol",
                "schema": {
                    "Symbol": "string", "PubDate": "date",
                    "ReportPeriodEndDate": "date", "UpdateTime": "timestamp",
                    "NetProfit": "double",
                },
            },
        },
    )
    return store, "daily_ds", "fin_ds"


def test_latest_period_no_rollback(tmp_path):
    """最新报告期 PIT：旧报告期晚修订不得回滚当前财务状态。"""
    store, anchor, fin = _fin_store(tmp_path)
    h = store.read_joined(
        anchor,
        {fin: ["NetProfit"]},
        joins={
            fin: {
                "policy": "pit_asof", "knowledge_time": "PubDate",
                "period_time": "ReportPeriodEndDate",
                "revision_order": ("UpdateTime",),
                "period_selection": "latest_period",
            }
        },
        time_range=("2024-01-01", "2024-08-02"),
    )
    df = h.to_pandas().sort_values("TradeDate")
    vals = dict(zip(df["TradeDate"].astype(str), df["NetProfit"]))
    assert vals["2024-01-02"] == 100.0
    assert vals["2024-07-02"] == 200.0
    assert vals["2024-08-02"] == 200.0  # 不回滚到旧期修订 999


def test_plain_asof_rolls_back(tmp_path):
    """对照：普通 ASOF 在旧期晚修订时会回滚（证明 latest_period 的必要性）。"""
    store, anchor, fin = _fin_store(tmp_path)
    h = store.read_joined(
        anchor,
        {fin: ["NetProfit"]},
        joins={fin: {"policy": "pit_asof", "knowledge_time": "PubDate"}},
        time_range=("2024-01-01", "2024-08-02"),
    )
    df = h.to_pandas().sort_values("TradeDate")
    vals = dict(zip(df["TradeDate"].astype(str), df["NetProfit"]))
    assert vals["2024-08-02"] == 999.0  # 旧行为确实回滚


def test_exact_period_selection(tmp_path):
    """exact_period 只保留指定报告期。"""
    store, anchor, fin = _fin_store(tmp_path)
    h = store.read_joined(
        anchor,
        {fin: ["NetProfit"]},
        joins={
            fin: {
                "policy": "pit_asof", "knowledge_time": "PubDate",
                "period_time": "ReportPeriodEndDate",
                "period_selection": "exact_period",
                "period_values": ["2024-06-30"],
            }
        },
        time_range=("2024-01-01", "2024-08-02"),
    )
    df = h.to_pandas().sort_values("TradeDate")
    vals = dict(zip(df["TradeDate"].astype(str), df["NetProfit"]))
    assert vals["2024-01-02"] is None or pd.isna(vals["2024-01-02"])  # 无该期
    assert vals["2024-07-02"] == 200.0
    assert vals["2024-08-02"] == 200.0


def test_annual_period_selection(tmp_path):
    """annual：只保留 12-31 报告期。"""
    store, anchor, fin = _fin_store(tmp_path)
    h = store.read_joined(
        anchor,
        {fin: ["NetProfit"]},
        joins={
            fin: {
                "policy": "pit_asof", "knowledge_time": "PubDate",
                "period_time": "ReportPeriodEndDate",
                "period_selection": "annual",
            }
        },
        time_range=("2024-01-01", "2024-08-02"),
    )
    df = h.to_pandas().sort_values("TradeDate")
    vals = dict(zip(df["TradeDate"].astype(str), df["NetProfit"]))
    assert vals["2024-07-02"] == 100.0  # 只有 2023-12-31 annual 期
    assert vals["2024-08-02"] == 999.0  # annual 期的晚修订可见


# ---------------------------------------------------------------------------
# A股分钟：QuoteTime(UTC)→Asia/Shanghai、午休 bar index
# ---------------------------------------------------------------------------

@pytest.fixture()
def minute_store(tmp_path, monkeypatch):
    m = tmp_path / "minute"
    m.mkdir()
    # UTC 时间；北京 09:31=01:31Z，11:30=03:30Z，13:01=05:01Z，15:00=07:00Z
    idx = pd.to_datetime(
        ["2024-01-02 01:31", "2024-01-02 03:30", "2024-01-02 05:01", "2024-01-02 07:00"]
    )
    pd.DataFrame(
        {
            "QuoteTime": idx,
            "Symbol": ["A"] * 4,
            "Volume": [100, 200, 300, 400],
            "Close": [10.0, 10.5, 11.0, 11.5],
        }
    ).to_parquet(m / "m.parquet")
    return _store(
        tmp_path,
        {
            "minute_ds": {
                "kind": "static", "access_mode": "published", "layout": "plain",
                "format": "parquet", "root": str(m), "glob": "m.parquet",
                "time_column": "QuoteTime", "instrument_column": "Symbol",
                "schema": {"QuoteTime": "timestamp", "Symbol": "string",
                           "Volume": "int", "Close": "double"},
            }
        },
    )


def test_minute_at_beijing_timezone(minute_store):
    """minute_at('09:31') 匹配北京 09:31（=UTC 01:31），不是 UTC 09:31。"""
    t = aggregate_minute_to_daily(
        minute_store, "minute_ds", "Volume",
        AggregationSpec(aggregation="minute_at", hhmm="09:31", market="ashare"),
    ).to_arrow()
    assert int(t.to_pydict()["value"][0]) == 100


def test_minute_range_afternoon_no_lunch_error(minute_store):
    """午休修正：北京 13:01–15:00 区间匹配 05:01Z/07:00Z。"""
    t = aggregate_minute_to_daily(
        minute_store, "minute_ds", "Volume",
        AggregationSpec(aggregation="minute_range", start="13:01", end="15:00", market="ashare"),
    ).to_arrow()
    assert int(t.to_pydict()["value"][0]) == 300 + 400


def test_minute_of_day_session_elapsed_index(minute_store):
    """minute_of_day 用 session elapsed bar index：11:30=119，13:01=120。"""
    # 120 分钟桶：11:30(elapsed 119) 在桶 1（60~120），13:01(elapsed 120) 在桶 2
    # 这里验证桶索引映射不把午后多算 90 分钟（旧 bug）。
    t = aggregate_minute_to_daily(
        minute_store, "minute_ds", "Volume",
        AggregationSpec(aggregation="minute_of_day", period=120, index=0, market="ashare"),
    ).to_arrow()
    vals = t.to_pydict()["value"]
    # 最后一桶 elapsed 120..240 → 13:01(120) 和 15:00(239) 都在 → 700
    assert int(vals[0]) == 300 + 400


def test_minute_bundle_multi_output(minute_store):
    """一次 scan 多聚合：morning_volume + afternoon_vol + range。"""
    h = aggregate_minute_bundle(
        minute_store, "minute_ds",
        [
            AggregationItem("Volume", AggregationSpec(aggregation="minute_range", start="09:31", end="11:30", market="ashare"), "morning_volume"),
            AggregationItem("Volume", AggregationSpec(aggregation="minute_at", hhmm="15:00", market="ashare"), "close_vol"),
        ],
        market="ashare",
    )
    d = h.to_arrow().to_pydict()
    assert int(d["morning_volume"][0]) == 100 + 200
    assert int(d["close_vol"][0]) == 400


# ---------------------------------------------------------------------------
# effective-only 事件：未来数据不前视（future cutoff）
# ---------------------------------------------------------------------------

def test_dividend_future_cutoff():
    """effective_time_only 事件表：时间窗上界晚于今天必须拒绝（production）。

    （us_stock_dividend 已被升级为 strict-PIT + declaration_date availability，
    不再走 effective-only 路径；这里用仍为 effective_time_only 的
    us_stock_capital_split 验证同一条 cutoff 语义。）
    """
    import os

    from data_access.cos_contract import enforce_event_cutoff, require_cos_contract

    contract = require_cos_contract("us_stock_capital_split")
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        with pytest.raises(ValidationError):
            enforce_event_cutoff(contract, time_range=("2020-01-01", "2099-12-31"), production=True)
        # as_of 在过去 → 放行
        enforce_event_cutoff(contract, as_of=dt.date(2020, 1, 1), production=True)
    finally:
        os.environ.pop("QUANT_PRODUCTION_MODE", None)


# ---------------------------------------------------------------------------
# 跨市场字段歧义 fail-closed
# ---------------------------------------------------------------------------

def test_cross_market_ambiguity_fail_closed():
    import os

    from data_access.read.semantic_catalog import get_semantic_catalog
    from data_access.core.exceptions import AmbiguousSemanticFieldError

    catalog = get_semantic_catalog()
    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        with pytest.raises(AmbiguousSemanticFieldError):
            catalog.resolve_one("market_cap")
    finally:
        os.environ.pop("QUANT_PRODUCTION_MODE", None)
    # 显式 market → 正常
    f = catalog.resolve_one("market_cap", market="ashare")
    assert f.dataset == "ashare_stock_valuation_daily"
    f2 = catalog.resolve_one("market_cap", dataset="us_stock_valuation_daily")
    assert f2.dataset == "us_stock_valuation_daily"


# ---------------------------------------------------------------------------
# allowed_filter_values 值校验（production fail-closed）
# ---------------------------------------------------------------------------

def test_allowed_filter_values_validation(tmp_path):
    """IndustrySource 过滤值必须在契约允许集合内。"""
    import os

    from data_access.read.semantic_catalog import get_semantic_catalog
    from data_access.store import DataAccessStore as _S

    os.environ["QUANT_PRODUCTION_MODE"] = "1"
    try:
        # 直接验证契约校验
        from data_access.cos_contract import validate_panel_request

        with pytest.raises(ValidationError):
            validate_panel_request(
                "ashare_stock_industry", semantic_filters={"IndustrySource": "NOT_A_SOURCE"}
            )
        # 合法值通过
        validate_panel_request(
            "ashare_stock_industry", semantic_filters={"IndustrySource": "sw_l1"}
        )
    finally:
        os.environ.pop("QUANT_PRODUCTION_MODE", None)
