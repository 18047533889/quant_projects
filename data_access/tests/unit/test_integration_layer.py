"""DataAccess ↔ FactorEngine 集成层单测。

覆盖（对应 docs/UNIVERSAL_IO_LAYER_PLAN.md 集成段）：
    - SemanticFieldCatalog（resolve_fields / aliases / physical 反查 / 单位归一化）
    - DataRequest / ReadPlan（plan.explain / plan.execute，单数据集 + 多数据集）
    - read_joined（多数据集批量 join：exact + pit_asof）
    - store.manifest_version / is_snapshot_stale（query-scoped snapshot token）
    - store.sql_relation（受控 RelationHandle：collect / 追加表达式 / 预算治理）
"""
from __future__ import annotations

from datetime import date, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import DataRequest, ValidationError, get_semantic_catalog
from data_access.core.engine import DuckDBEngine
from data_access.read.query_budget import QueryBudget
from data_access.read.semantic_catalog import normalize_table_units
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture()
def tree(tmp_path):
    """构建合成数据集：daily / valuation / balance / universe。"""
    daily = tmp_path / "daily"
    valuation = tmp_path / "valuation"
    balance = tmp_path / "balance"
    universe = tmp_path / "universe"
    for d in (daily, valuation, balance, universe):
        d.mkdir(parents=True)

    dates = [date(2024, 1, 1 + i) for i in range(5)]  # 01-01..01-05
    daily_rows = [
        (d, s, float(i + 1), (i + 1) * 100)
        for i, d in enumerate(dates)
        for s in ("A", "B")
    ]
    pq.write_table(
        pa.table(
            {
                "TradeDate": [r[0] for r in daily_rows],
                "Symbol": [r[1] for r in daily_rows],
                "Close": [r[2] for r in daily_rows],
                "Volume": [r[3] for r in daily_rows],
            }
        ),
        str(daily / "daily.parquet"),
    )
    val_rows = [
        (d, s, 100.0 + i, 2.5)
        for i, d in enumerate(dates)
        for s in ("A", "B")
    ]
    pq.write_table(
        pa.table(
            {
                "TradeDate": [r[0] for r in val_rows],
                "Symbol": [r[1] for r in val_rows],
                "MarketCap": [r[2] for r in val_rows],
                "TurnoverRatio": [r[3] for r in val_rows],
            }
        ),
        str(valuation / "val.parquet"),
    )
    # 财报表：每标的两条可见记录（PIT asof 用）
    bal_rows = [
        (date(2024, 1, 1), "A", 100.0),
        (date(2024, 1, 4), "A", 200.0),
        (date(2024, 1, 1), "B", 300.0),
        (date(2024, 1, 4), "B", 400.0),
    ]
    pq.write_table(
        pa.table(
            {
                "TradeDate": [r[0] for r in bal_rows],
                "Symbol": [r[1] for r in bal_rows],
                "TotalAssets": [r[2] for r in bal_rows],
            }
        ),
        str(balance / "bal.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "TradeDate": [d for d in dates for _ in ("A", "B")],
                "Symbol": [s for _ in dates for s in ("A", "B")],
            }
        ),
        str(universe / "univ.parquet"),
    )

    (tmp_path / "datasets.yaml").write_text(
        f"""
ashare_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {daily}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
    Volume: int
ashare_stock_valuation_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {valuation}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    MarketCap: double
    TurnoverRatio: double
ashare_stock_balance:
  kind: static
  access_mode: published
  layout: plain
  root: {balance}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    TotalAssets: double
ashare_universe_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {universe}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
""",
        encoding="utf-8",
    )
    return {
        "tmp": tmp_path,
        "store": DataAccessStore(
            registry=load_registry(tmp_path / "datasets.yaml"),
            engine=DuckDBEngine(threads=2, enable_object_cache=False),
        ),
        "daily": daily,
        "valuation": valuation,
        "balance": balance,
    }


def test_semantic_catalog_resolve(tree):
    store = tree["store"]
    fs = store.resolve_fields(["close", "market_cap", "total_assets"])
    by = {f.logical_name: f for f in fs}
    assert by["close"].dataset == "ashare_stock_daily"
    assert by["close"].physical_name == "Close"
    assert by["market_cap"].physical_name == "MarketCap"
    assert by["total_assets"].dataset == "ashare_stock_balance"
    # 物理列反查 → 拿到 scale（单位归一化）
    fp = store.resolve_fields(["TurnoverRatio"], dataset="ashare_stock_valuation_daily")
    assert fp[0].scale == 0.01
    # alias
    c = get_semantic_catalog()
    assert c.resolve_one("revenue").physical_name == "OperatingRevenue"
    # 未知字段
    with pytest.raises(ValidationError):
        store.resolve_fields(["nope_nope"])


def test_normalize_table_units(tree):
    table = pa.table({"TurnoverRatio": [2.5, 3.0], "Close": [1.0, 2.0]})
    catalog = get_semantic_catalog()
    field = catalog.resolve_by_physical("ashare_stock_valuation_daily", "TurnoverRatio")
    out = normalize_table_units(table, [field])
    assert out.column("TurnoverRatio").to_pylist() == [0.025, 0.03]
    # 无 scale 的字段不改变
    out2 = normalize_table_units(table, [])
    assert out2.column("TurnoverRatio").to_pylist() == [2.5, 3.0]


def test_read_normalize_units(tree):
    store = tree["store"]
    h = store.read("ashare_stock_valuation_daily", columns=["TurnoverRatio"], normalize_units=True)
    vals = h.to_arrow().column("TurnoverRatio").to_pylist()
    assert abs(vals[0] - 0.025) < 1e-12
    # 默认不归一化，保持旧行为
    h2 = store.read("ashare_stock_valuation_daily", columns=["TurnoverRatio"])
    assert abs(h2.to_arrow().column("TurnoverRatio").to_pylist()[0] - 2.5) < 1e-9


def test_read_joined_exact(tree):
    store = tree["store"]
    h = store.read_joined(
        "ashare_stock_daily",
        {
            "ashare_stock_daily": ["Close", "Volume"],
            "ashare_stock_valuation_daily": ["MarketCap", "TurnoverRatio"],
        },
        time_range=("2024-01-01", "2024-01-05"),
    )
    tbl = h.to_arrow()
    assert tbl.column_names[:2] == ["TradeDate", "Symbol"]
    for c in ("Close", "Volume", "MarketCap", "TurnoverRatio"):
        assert c in tbl.column_names
    assert tbl.num_rows == 10  # 2 syms × 5 days
    # 对齐正确：同一 (TradeDate, Symbol) 的 close/mcap 来自同一天
    dates = tbl.column("TradeDate").to_pylist()
    closes = tbl.column("Close").to_pylist()
    assert closes[dates.index(date(2024, 1, 2))] == 2.0


def test_read_joined_asof(tree):
    store = tree["store"]
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["TotalAssets"]},
        joins={"ashare_stock_balance": "pit_asof"},
        time_range=("2024-01-01", "2024-01-05"),
    )
    tbl = h.to_arrow()
    syms = tbl.column("Symbol").to_pylist()
    dates_ = tbl.column("TradeDate").to_pylist()
    assets = tbl.column("TotalAssets").to_pylist()
    lookup = {(s, d): a for s, d, a in zip(syms, dates_, assets)}
    # 财报 01-04 发布前 → 用 01-01 记录；01-04 及之后 → 用 01-04 记录
    assert lookup[("A", date(2024, 1, 1))] == 100.0
    assert lookup[("A", date(2024, 1, 3))] == 100.0
    assert lookup[("A", date(2024, 1, 4))] == 200.0
    assert lookup[("A", date(2024, 1, 5))] == 200.0
    assert lookup[("B", date(2024, 1, 5))] == 400.0


def test_read_joined_normalize_units(tree):
    store = tree["store"]
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_valuation_daily": ["TurnoverRatio"]},
        time_range=("2024-01-01", "2024-01-05"),
        normalize_units=True,
    )
    vals = h.to_arrow().column("TurnoverRatio").to_pylist()
    assert abs(vals[0] - 0.025) < 1e-12


def test_read_joined_qualified_names(tree):
    store = tree["store"]
    h = store.read_joined(
        "ashare_stock_daily",
        ["Close", "ashare_stock_valuation_daily.MarketCap"],
        time_range=("2024-01-01", "2024-01-05"),
    )
    tbl = h.to_arrow()
    assert "Close" in tbl.column_names and "MarketCap" in tbl.column_names
    assert tbl.num_rows == 10


def test_plan_explain_and_execute(tree):
    store = tree["store"]
    plan = store.plan(
        DataRequest(
            fields=["close", "market_cap", "total_assets"],
            start="2024-01-01",
            end="2024-01-05",
            anchor="ashare_stock_daily",
            joins={"ashare_stock_balance": "pit_asof"},
        )
    )
    txt = plan.explain()
    assert "ashare_stock_daily" in txt
    assert "pit_asof" in txt
    assert "close -> ashare_stock_daily.Close" in txt
    h = plan.execute()
    assert h.to_arrow().num_rows == 10


def test_plan_single_dataset(tree):
    store = tree["store"]
    plan = store.plan(
        DataRequest(fields=["close", "volume"], anchor="ashare_stock_daily",
                    start="2024-01-01", end="2024-01-05")
    )
    assert len(plan.datasets) == 1
    h = plan.execute()
    tbl = h.to_arrow()
    assert tbl.num_rows == 10
    assert set(tbl.column_names) >= {"TradeDate", "Symbol", "Close", "Volume"}


def test_plan_requires_anchor_when_multi(tree):
    store = tree["store"]
    with pytest.raises(ValidationError):
        store.plan(DataRequest(fields=["close", "total_assets"]))


def test_plan_universe(tree):
    store = tree["store"]
    plan = store.plan(
        DataRequest(
            fields=["close"],
            anchor="ashare_stock_daily",
            universe="ashare_universe_daily",
            instruments=["A"],
            start="2024-01-01",
            end="2024-01-05",
        )
    )
    h = plan.execute()
    tbl = h.to_arrow()
    assert set(tbl.column("Symbol").to_pylist()) == {"A"}


def test_read_joined_right_table_instrument_filter_pushdown(tree, monkeypatch):
    """#1 右表 instrument_filter 必须下推（exact/asof 一律），不能只筛锚点。"""
    store = tree["store"]
    captured: dict[str, list[str] | None] = {}

    orig = store._prepare_dataset_read

    def spy(dsobj, *, time_range, params, instrument_filter=None, **kwargs):
        captured[dsobj.name] = instrument_filter
        return orig(
            dsobj,
            time_range=time_range,
            params=params,
            instrument_filter=instrument_filter,
            **kwargs,  # #P0-9 time_column 时钟参数
        )

    monkeypatch.setattr(store, "_prepare_dataset_read", spy)
    h = store.read_joined(
        "ashare_stock_daily",
        {
            "ashare_stock_daily": ["Close"],
            "ashare_stock_valuation_daily": ["MarketCap"],
            "ashare_stock_balance": ["TotalAssets"],
        },
        joins={"ashare_stock_balance": "pit_asof"},
        time_range=("2024-01-01", "2024-01-05"),
        instrument_filter=["A"],
    )
    assert h.to_arrow().num_rows == 5  # 只有 A 的 5 天
    assert captured["ashare_stock_daily"] == ["A"]
    assert captured["ashare_stock_valuation_daily"] == ["A"]
    assert captured["ashare_stock_balance"] == ["A"]


@pytest.fixture()
def cross_cols_tree(tmp_path, monkeypatch):
    """锚点与右表用不同时间/标的列名（#5 跨表列名 bug 回归）。"""
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    daily = tmp_path / "daily"
    events = tmp_path / "events"
    daily.mkdir(parents=True)
    events.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2024, 1, 1), date(2024, 1, 2)],
                "Symbol": ["AAPL", "AAPL"],
                "Close": [10.0, 11.0],
            }
        ),
        str(daily / "d.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "filing_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["AAPL", "AAPL"],
                "TotalAssets": [100.0, 110.0],
            }
        ),
        str(events / "e.parquet"),
    )
    (tmp_path / "datasets.yaml").write_text(
        f"""
us_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {daily}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
us_stock_balance:
  kind: static
  access_mode: published
  layout: plain
  root: {events}
  glob: "*.parquet"
  time_column: filing_date
  instrument_column: ticker
  schema:
    filing_date: date
    ticker: string
    TotalAssets: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    )


def test_read_joined_cross_table_column_names(cross_cols_tree):
    """#5 锚点 TradeDate/Symbol vs 右表 filing_date/ticker，join 必须显式列名。"""
    store = cross_cols_tree
    h = store.read_joined(
        "us_stock_daily",
        {"us_stock_daily": ["Close"], "us_stock_balance": ["TotalAssets"]},
        joins={"us_stock_balance": "pit_asof"},
        time_range=("2024-01-01", "2024-01-02"),
    )
    tbl = h.to_arrow()
    assert set(tbl.column("Symbol").to_pylist()) == {"AAPL"}
    dates_ = tbl.column("TradeDate").to_pylist()
    assets = tbl.column("TotalAssets").to_pylist()
    ordered = dict(sorted(zip(dates_, assets)))
    assert list(ordered.values()) == [100.0, 110.0]  # 01-01→100, 01-02→110


@pytest.fixture()
def pit_tree(tmp_path, monkeypatch):
    """带窗口外历史的财报表 + 多版本（#2 seed+window / #6 revision 去重）。"""
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    daily = tmp_path / "daily"
    balance = tmp_path / "balance"
    daily.mkdir(parents=True)
    balance.mkdir(parents=True)
    dates = [date(2024, 1, i + 1) for i in range(5)]  # 01-01..01-05
    pq.write_table(
        pa.table(
            {
                "TradeDate": [d for d in dates for _ in ("A", "B")],
                "Symbol": [s for _ in dates for s in ("A", "B")],
                "Close": [1.0] * 10,
            }
        ),
        str(daily / "d.parquet"),
    )
    # 窗口外历史（12-30）+ 窗口内 + 同日双版本
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2023, 12, 30), date(2024, 1, 3),
                              date(2024, 1, 3), date(2024, 1, 3)],
                "Symbol": ["A", "A", "A", "B"],
                "PubDate": [date(2023, 12, 30), date(2024, 1, 3),
                            date(2024, 1, 3), date(2024, 1, 3)],
                "UpdateTime": [datetime(2023, 12, 30, 18, 0),
                               datetime(2024, 1, 3, 8, 0),
                               datetime(2024, 1, 3, 16, 0),
                               datetime(2024, 1, 3, 12, 0)],
                # #9：financial_event 字段默认 latest_period → period_time 列必须存在
                "ReportPeriodEndDate": [date(2023, 12, 31), date(2024, 1, 1),
                                        date(2024, 1, 1), date(2024, 1, 1)],
                "TotalAssets": [1.0, 2.0, 3.0, 30.0],
            }
        ),
        str(balance / "b.parquet"),
    )
    (tmp_path / "datasets.yaml").write_text(
        f"""
ashare_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {daily}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
ashare_stock_balance:
  kind: static
  access_mode: published
  layout: plain
  root: {balance}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    PubDate: date
    UpdateTime: timestamp
    ReportPeriodEndDate: date
    TotalAssets: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    )


def test_read_joined_seed_window_picks_prestart_record(pit_tree):
    """#2 asof 右表用 seed（窗口外最后一条可见记录）+ window，语义与全历史一致。"""
    store = pit_tree
    spec = {
        "knowledge_time": "PubDate",
        "revision_order": ("UpdateTime",),
        "availability": "same_day",
    }
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["TotalAssets"]},
        joins={"ashare_stock_balance": spec},
        time_range=("2024-01-01", "2024-01-05"),
        instrument_filter=["A", "B"],
    )
    tbl = h.to_arrow()
    syms = tbl.column("Symbol").to_pylist()
    dates_ = tbl.column("TradeDate").to_pylist()
    assets = tbl.column("TotalAssets").to_pylist()
    lookup = {(s, d): a for s, d, a in zip(syms, dates_, assets)}
    # 窗口 start=01-01，A 在 01-03 才有新记录 → 01-01/01-02 用窗口外 12-30 的 seed
    assert lookup[("A", date(2024, 1, 1))] == 1.0
    assert lookup[("A", date(2024, 1, 2))] == 1.0
    assert lookup[("A", date(2024, 1, 3))] == 3.0  # 同日双版本取最新 UpdateTime
    assert lookup[("A", date(2024, 1, 4))] == 3.0
    # B 没有窗口前记录；01-03 前 same_day 无可见 → None，01-05 用 01-03 记录
    assert lookup[("B", date(2024, 1, 1))] is None
    assert lookup[("B", date(2024, 1, 5))] == 30.0


def test_read_joined_next_trading_day_availability(pit_tree):
    """#3 availability=next_trading_day：PubDate 当天 bar 不可用，次日才可见。"""
    store = pit_tree
    spec = {
        "knowledge_time": "PubDate",
        "revision_order": ("UpdateTime",),
        "availability": "next_trading_day",
    }
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["TotalAssets"]},
        joins={"ashare_stock_balance": spec},
        time_range=("2024-01-01", "2024-01-05"),
        instrument_filter=["A"],
    )
    tbl = h.to_arrow()
    syms = tbl.column("Symbol").to_pylist()
    dates_ = tbl.column("TradeDate").to_pylist()
    assets = tbl.column("TotalAssets").to_pylist()
    lookup = {(s, d): a for s, d, a in zip(syms, dates_, assets)}
    # seed（12-30）在 01-01 起就可见（01-01 > 12-30）
    assert lookup[("A", date(2024, 1, 1))] == 1.0
    assert lookup[("A", date(2024, 1, 2))] == 1.0
    # PubDate=01-03 的记录 01-03 当天不可用（严格大于）→ 仍用 seed；01-04 可用
    assert lookup[("A", date(2024, 1, 3))] == 1.0
    assert lookup[("A", date(2024, 1, 4))] == 3.0


def test_read_joined_catalog_derived_pit_join(pit_tree):
    """#3 逻辑字段（total_assets）从 catalog 自动推导 pit_asof 语义：
    knowledge_time=PubDate + next_trading_day + revision_order，无需调用方指定 joins。"""
    store = pit_tree
    # list-mode 字段 + catalog → 自动推导语义 join（默认 next_trading_day）
    h = store.read_joined(
        "ashare_stock_daily",
        ["total_assets"],
        time_range=("2024-01-01", "2024-01-05"),
        instrument_filter=["A"],
    )
    tbl = h.to_arrow()
    dates_ = tbl.column("TradeDate").to_pylist()
    assets = tbl.column("TotalAssets").to_pylist()
    lookup = dict(zip(dates_, assets))
    # next_trading_day：PubDate=01-03 的记录 01-03 当天不可见 → 用 12-30 seed
    assert lookup[date(2024, 1, 1)] == 1.0
    assert lookup[date(2024, 1, 3)] == 1.0
    assert lookup[date(2024, 1, 4)] == 3.0


@pytest.fixture()
def constituent_tree(tmp_path, monkeypatch):
    """带 IndexConstituent 的注册表（required_filters 端到端测试）。"""
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    daily = tmp_path / "daily"
    idx = tmp_path / "idx"
    daily.mkdir(parents=True)
    idx.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2024, 1, 1), date(2024, 1, 2)],
                "Symbol": ["000001.SZ", "000001.SZ"],
                "Close": [10.0, 11.0],
            }
        ),
        str(daily / "d.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2024, 1, 1), date(2024, 1, 2)],
                "Symbol": ["000001.SZ", "000001.SZ"],
                "IndexSymbol": ["000300.SH", "000300.SH"],
                "Weight": [0.6, 0.7],
            }
        ),
        str(idx / "i.parquet"),
    )
    (tmp_path / "datasets.yaml").write_text(
        f"""
ashare_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {daily}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
ashare_index_constituent:
  kind: static
  access_mode: published
  layout: plain
  root: {idx}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    IndexSymbol: string
    Weight: double
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    )


def test_required_filters_research_warns(constituent_tree):
    """#8 required_filters：research 模式缺 IndexSymbol 只告警不失败。"""
    store = constituent_tree
    field = get_semantic_catalog().get("index_weight")
    store._enforce_required_filters([field], {})  # 不应抛错


def test_required_filters_production_fails(constituent_tree, monkeypatch):
    """#8 required_filters：production 模式缺 IndexSymbol 必须 fail-closed。"""
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    store = constituent_tree
    field = get_semantic_catalog().get("index_weight")
    with pytest.raises(ValidationError):
        store._enforce_required_filters([field], {})
    # 提供 IndexSymbol 后放行
    store._enforce_required_filters([field], {"ashare_index_constituent": {"IndexSymbol": "000300.SH"}})


def test_read_joined_exact_dedup_revision(pit_tree):
    """#6 exact join 同日双版本：join 前按 revision_order 去重，行数不膨胀。"""
    store = pit_tree
    spec = {
        "policy": "exact",
        "revision_order": ("UpdateTime",),
    }
    h = store.read_joined(
        "ashare_stock_daily",
        {"ashare_stock_daily": ["Close"], "ashare_stock_balance": ["TotalAssets"]},
        joins={"ashare_stock_balance": spec},
        time_range=("2024-01-03", "2024-01-03"),
        instrument_filter=["A"],
    )
    tbl = h.to_arrow()
    # A 在 01-03 有两条版本（UpdateTime 08:00 / 16:00）→ 去重后取 3.0，且只 1 行
    assert tbl.num_rows == 1
    assert tbl.column("TotalAssets").to_pylist() == [3.0]


@pytest.fixture()
def timevar_universe_tree(tmp_path, monkeypatch):
    """universe 成员随时间变化（#15 时变 panel）：01-01 只有 A，01-02 起 A+B。"""
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    daily = tmp_path / "daily"
    uni = tmp_path / "universe"
    daily.mkdir(parents=True)
    uni.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2024, 1, 1), date(2024, 1, 2),
                              date(2024, 1, 1), date(2024, 1, 2)],
                "Symbol": ["A", "A", "B", "B"],
                "Close": [1.0, 2.0, 10.0, 20.0],
            }
        ),
        str(daily / "d.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)],
                "Symbol": ["A", "A", "B"],
            }
        ),
        str(uni / "u.parquet"),
    )
    (tmp_path / "datasets.yaml").write_text(
        f"""
ashare_stock_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {daily}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
ashare_universe_daily:
  kind: static
  access_mode: published
  layout: plain
  root: {uni}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
""",
        encoding="utf-8",
    )
    return DataAccessStore(
        registry=load_registry(tmp_path / "datasets.yaml"),
        engine=DuckDBEngine(threads=2, enable_object_cache=False),
    )


def test_plan_universe_time_varying_membership(timevar_universe_tree):
    """#15 时变 universe：01-01 只有 A，01-02 A+B，跨日成分变化必须保留。"""
    store = timevar_universe_tree
    plan = store.plan(
        DataRequest(
            fields=["close"],
            anchor="ashare_stock_daily",
            universe="ashare_universe_daily",
            start="2024-01-01",
            end="2024-01-02",
        )
    )
    h = plan.execute()
    tbl = h.to_arrow()
    dates_ = tbl.column("TradeDate").to_pylist()
    syms = tbl.column("Symbol").to_pylist()
    members = {(d, s) for d, s in zip(dates_, syms)}
    # 01-01 B 不在 universe → 被过滤；01-02 A+B 都在
    assert (date(2024, 1, 1), "A") in members
    assert (date(2024, 1, 1), "B") not in members
    assert (date(2024, 1, 2), "A") in members
    assert (date(2024, 1, 2), "B") in members


def test_plan_universe_static_intersection(tree):
    """#15 显式 time_varying_universe=False → 退回窗口内静态集合求交（旧行为）。"""
    store = tree["store"]
    plan = store.plan(
        DataRequest(
            fields=["close"],
            anchor="ashare_stock_daily",
            universe="ashare_universe_daily",
            instruments=["A"],
            start="2024-01-01",
            end="2024-01-05",
            time_varying_universe=False,
        )
    )
    h = plan.execute()
    assert set(h.to_arrow().column("Symbol").to_pylist()) == {"A"}


def test_manifest_version_and_stale(tree):
    store = tree["store"]
    assert store.manifest_version("ashare_stock_daily")["has_manifest"] is False
    store.build_dataset_manifest("ashare_stock_daily")
    mv = store.manifest_version("ashare_stock_daily")
    assert mv["has_manifest"] is True and mv["fresh"] is True
    assert mv["dataset_version"] and mv["partition_version"]
    assert mv["file_count"] == 1
    # 相同 token → 不 stale
    assert (
        store.is_snapshot_stale(
            "ashare_stock_daily", dataset_version=mv["dataset_version"],
            partition_version=mv["partition_version"],
        )
        is False
    )
    # 错误 token → stale
    assert store.is_snapshot_stale("ashare_stock_daily", dataset_version="deadbeef") is True
    # #17/#18：新增文件后写路径 bump manifest epoch → 立即 stale（read path 不再 glob）
    epoch0 = store.manifest_version("ashare_stock_daily")["manifest_epoch"]
    pq.write_table(
        pa.table(
            {
                "TradeDate": [date(2024, 1, 6)],
                "Symbol": ["A"],
                "Close": [9.0],
                "Volume": [9],
            }
        ),
        str(tree["daily"] / "extra.parquet"),
    )
    # 尚未 bump → 仍视为 fresh（read path 信任写路径维护的 epoch）
    assert store.is_snapshot_stale(
        "ashare_stock_daily",
        dataset_version=mv["dataset_version"],
        partition_version=mv["partition_version"],
        manifest_epoch=epoch0,
    ) is False
    # 写路径 bump epoch → stale
    new_epoch = store.touch_manifest_epoch("ashare_stock_daily")
    assert new_epoch is not None and new_epoch != epoch0
    assert store.is_snapshot_stale(
        "ashare_stock_daily", manifest_epoch=epoch0
    ) is True


def test_sql_relation(tree):
    store = tree["store"]
    parquet = str(tree["daily"] / "daily.parquet")
    rh = store.sql_relation(
        "SELECT TradeDate, Symbol, Close FROM read_parquet(?)",
        params=[parquet],
    )
    tbl = rh.collect()
    assert tbl.num_rows == 10
    # 追加表达式（scan + factor expression 融合）
    rh2 = rh.sql("SELECT Symbol, AVG(Close) AS avg_close FROM _sub GROUP BY Symbol")
    agg = rh2.collect()
    assert agg.num_rows == 2
    assert set(agg.column("Symbol").to_pylist()) == {"A", "B"}
    # 逃生口 relation（只读检查）
    assert rh.relation.columns is not None


def test_sql_relation_budget_enforced(tree):
    store = tree["store"]
    parquet = str(tree["daily"] / "daily.parquet")
    small = QueryBudget(max_rows=5)
    with pytest.raises(ValidationError):
        store.sql_relation(
            "SELECT * FROM read_parquet(?)",
            params=[parquet],
            query_budget=small,
        ).collect()


def test_sql_relation_snapshot_datasets(tree):
    store = tree["store"]
    parquet = str(tree["daily"] / "daily.parquet")
    rh = store.sql_relation(
        "SELECT * FROM read_parquet(?)",
        params=[parquet],
        snapshot_datasets=["ashare_stock_daily"],
    )
    assert rh._snapshot is not None
    tbl = rh.collect()
    assert tbl.num_rows == 10
