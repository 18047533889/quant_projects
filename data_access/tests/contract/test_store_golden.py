"""
contract 测试：端到端跑通 store API，用 golden parquet 验证读出结果语义正确。

WHY golden：迁移过程中，要能断言 DuckDB 路径的输出和老 pandas 路径一样。
           这里用语义 hash（列值排序后对比），不是 parquet 字节 hash。
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access import reset_store
from data_access.core.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.store import DataAccessStore


# ---- fixture 工具 -----------------------------------------------------------

def _make_day_aggs_fixture(root: Path) -> dict:
    """造一个模仿 day_aggs 结构的小数据集：
        {root}/{YYYY}/{MM}/{YYYY-MM-DD}.parquet
    """
    rows = []
    dates = [
        ("2024-01-02", "AAPL", 190.0, 1000000),
        ("2024-01-02", "MSFT", 370.0, 800000),
        ("2024-01-03", "AAPL", 192.0, 1100000),
        ("2024-01-03", "MSFT", 371.0, 850000),
        ("2024-02-01", "AAPL", 185.0, 1200000),
        ("2024-02-01", "MSFT", 405.0, 900000),
    ]
    for date_str, ticker, close, volume in dates:
        year = date_str[:4]
        month = date_str[5:7]
        out_dir = root / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            "align_time": pd.Timestamp(f"{date_str} 05:00:00", tz="UTC"),
            "ticker": ticker,
            "close": close,
            "volume": volume,
        }])
        file_path = out_dir / f"{date_str}.parquet"
        # 同一天两个 ticker 合并到一个 parquet（模拟真实 day_aggs 每日一文件的结构）
        existing = None
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(file_path)
        rows.append({"date": date_str, "ticker": ticker, "close": close})
    return {"rows": rows}


def _make_factor_fixture(root: Path, factor_id: str) -> dict:
    """造一个模仿 factor_lake 结构的小数据集：
        {root}/factors/{factor_id}/year=YYYY/data.parquet
    """
    factor_dir = root / "factors" / factor_id
    years = {
        2023: [
            ("2023-06-01", "AAPL", 0.5),
            ("2023-06-01", "MSFT", -0.3),
            ("2023-12-01", "AAPL", 0.6),
        ],
        2024: [
            ("2024-01-01", "AAPL", 0.7),
            ("2024-01-01", "MSFT", -0.1),
            ("2024-06-01", "AAPL", 0.8),
        ],
    }
    rows = []
    for year, items in years.items():
        out_dir = factor_dir / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {"datetime": pd.Timestamp(d), "asset": a, "value": v}
            for d, a, v in items
        ])
        df.to_parquet(out_dir / "data.parquet")
        rows.extend(items)
    return {"rows": rows}


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


@pytest.fixture
def golden_store(tmp_path, monkeypatch):
    """搭一个纯 tmp_path 的 store 实例，带 golden 数据 + YAML。"""
    data_root = tmp_path / "day_aggs"
    _make_day_aggs_fixture(data_root)

    lake_root = tmp_path / "lake"
    _make_factor_fixture(lake_root, "mom_3d")

    # 要用 env 展开，否则 YAML 里写 tmp_path 会跟测试隔离不一致
    monkeypatch.setenv("TEST_DAY_ROOT", str(data_root))
    monkeypatch.setenv("TEST_LAKE_ROOT", str(lake_root))

    yaml_path = _write_yaml(tmp_path / "datasets.yaml", """
        demo_day:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_DAY_ROOT}
          time_column: align_time
          instrument_column: ticker
          union_by_name: true

        demo_factor:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: ${TEST_LAKE_ROOT}/factors/{factor_id}
          glob_template: "year=*/data.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
    """)

    reset_store()  # 清进程单例，避免测试间串扰
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)  # 测试里少开点线程
    store = DataAccessStore(registry=reg, engine=engine)
    yield store
    engine.close()
    reset_store()


# ---- 实际测试 ---------------------------------------------------------------

def test_read_arrow_static_full(golden_store):
    """静态数据集：全读 → 应该拿到所有 6 行。"""
    tbl = golden_store.read_arrow("demo_day", columns=["align_time", "ticker", "close"])
    assert tbl.num_rows == 6
    assert set(tbl.column_names) == {"align_time", "ticker", "close"}


def test_read_arrow_time_range_filter(golden_store):
    """time_range 按真实时间戳过滤（注意 align_time 在数据里是 05:00 UTC）。"""
    tbl = golden_store.read_arrow(
        "demo_day",
        columns=["align_time", "ticker"],
        time_range=("2024-01-01", "2024-01-31 23:59:59"),
    )
    # 1 月的 4 条（2 日 2 条 + 3 日 2 条）
    assert tbl.num_rows == 4


def test_read_arrow_instrument_filter(golden_store):
    tbl = golden_store.read_arrow(
        "demo_day",
        columns=["ticker", "close"],
        instrument_filter=["AAPL"],
    )
    assert tbl.num_rows == 3  # AAPL 三天
    assert set(tbl.column(0).to_pylist()) == {"AAPL"}


def test_read_frame_returns_dataframe(golden_store):
    df = golden_store.read_frame("demo_day", columns=["ticker", "close"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert list(df.columns) == ["ticker", "close"]


def test_load_columns_batch(golden_store):
    """批量多列 → dict[name, Series with MultiIndex]。"""
    cols = golden_store.load_columns(
        "demo_day",
        columns=["close", "volume"],
        instrument_filter=["AAPL"],
    )
    assert set(cols.keys()) == {"close", "volume"}
    close = cols["close"]
    assert close.name == "close"
    assert close.index.names == ["timestamp", "instrument"]
    assert len(close) == 3
    # 按索引取值
    sample_key = close.index[0]
    assert sample_key[1] == "AAPL"


def test_parametric_factor_lake(golden_store):
    """参数化数据集：factor_id 必传。"""
    tbl = golden_store.read_arrow(
        "demo_factor",
        factor_id="mom_3d",
        columns=["datetime", "asset", "value"],
    )
    assert tbl.num_rows == 6  # 每年 3 条 * 2 年


def test_parametric_hive_partition_prune(golden_store):
    """WHERE datetime >= '2024-01-01' 应该只扫 year=2024 的 parquet。

    这里只能断言结果正确（行数）；要验证 DuckDB 真的做了分区剪枝，
    需要 EXPLAIN——放到 performance 测试里。
    """
    tbl = golden_store.read_arrow(
        "demo_factor",
        factor_id="mom_3d",
        columns=["datetime", "value"],
        time_range=("2024-01-01", "2024-12-31"),
    )
    assert tbl.num_rows == 3  # 2024 年 3 条


def test_parametric_missing_param_raises(golden_store):
    from data_access.core.exceptions import ValidationError
    with pytest.raises(ValidationError, match="缺参数"):
        golden_store.read_arrow("demo_factor", columns=["datetime"])


def test_unknown_dataset_raises(golden_store):
    from data_access.core.exceptions import ValidationError
    with pytest.raises(ValidationError, match="未注册"):
        golden_store.read_arrow("no_such_dataset")


def test_load_columns_empty_result_raises_data_error(golden_store):
    from data_access.core.exceptions import DataError
    with pytest.raises(DataError, match="0 行"):
        golden_store.load_columns(
            "demo_day",
            columns=["close"],
            instrument_filter=["NONEXISTENT"],
        )


def test_parity_with_direct_pd_read_parquet(golden_store, tmp_path):
    """关键契约测试：DuckDB 读出的结果 == pandas 读出的结果（同一批文件）。

    这是迁移期最重要的保险：确保没有语义漂移。
    """
    store_result = golden_store.read_frame(
        "demo_day",
        columns=["align_time", "ticker", "close"],
    )

    # 直接用 pandas 把所有 parquet 合并起来
    data_root = Path(store_result.attrs.get("source_root",
                     tmp_path / "day_aggs"))  # fallback，实际从 fixture 拿
    # 直接扫 tmp_path 下所有 parquet
    pandas_frames = []
    # golden_store fixture 里数据写在 TEST_DAY_ROOT env 下，通过 store 的 registry 能拿到
    day_ds = golden_store._registry.get("demo_day")
    from pathlib import Path as _Path
    for p in _Path(str(day_ds.root)).rglob("*.parquet"):
        pandas_frames.append(pd.read_parquet(p, columns=["align_time", "ticker", "close"]))
    pandas_result = pd.concat(pandas_frames, ignore_index=True)

    # 语义 hash：按 (ticker, align_time) 排序后比较
    def _canonical(df):
        return (
            df.sort_values(["ticker", "align_time"])
              .reset_index(drop=True)
        )

    store_canonical = _canonical(store_result)
    pandas_canonical = _canonical(pandas_result)

    assert len(store_canonical) == len(pandas_canonical)
    pd.testing.assert_frame_equal(
        store_canonical, pandas_canonical,
        check_dtype=False,  # DuckDB 可能把 tz 去掉，pandas 读保留；列值一致即可
        check_exact=False,
    )
