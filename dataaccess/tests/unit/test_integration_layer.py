"""DataAccess ↔ FactorEngine 集成层单测。

覆盖（对应 docs/UNIVERSAL_IO_LAYER_PLAN.md 集成段）：
    - SemanticFieldCatalog（resolve_fields / aliases / physical 反查 / 单位归一化）
    - DataRequest / ReadPlan（plan.explain / plan.execute，单数据集 + 多数据集）
    - read_joined（多数据集批量 join：exact + pit_asof）
    - store.manifest_version / is_snapshot_stale（query-scoped snapshot token）
    - store.sql_relation（受控 RelationHandle：collect / 追加表达式 / 预算治理）
"""
from __future__ import annotations

from datetime import date

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
    # 新增文件 → partition_version 变化 → stale
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
    assert (
        store.is_snapshot_stale(
            "ashare_stock_daily", partition_version=mv["partition_version"]
        )
        is True
    )


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
