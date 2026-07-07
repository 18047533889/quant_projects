"""
contract 测试：PR7 流式读 + Polars LazyFrame

覆盖：
- read_arrow_stream 按 batch_size 切分，多 batch 合并后 == read_arrow 结果
- read_arrow_stream 的 columns / time_range / instrument_filter 过滤下推生效
- 中途 break 能释放 reader（不崩、不泄）
- scan_polars 返回的 LazyFrame .collect() 与 read_frame 语义一致
- scan_polars 时间/标的/列过滤下推（lazy）
- scan_polars 在 polars 未安装时抛清晰的 ImportError

命名沿用 test_store_golden 的 fixture 风格；不复用 fixture 是因为 batch_size
需要造大一点的数据（>>100）才能观察到分批，跟 golden 的 6 行数据不兼容。
"""
from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pytest

from data_access import reset_store
from data_access.engine import DuckDBEngine
from data_access.registry import load_registry
from data_access.store import DataAccessStore


# ---- fixture 工具 -----------------------------------------------------------


def _make_large_factor_fixture(root: Path, factor_id: str, rows_per_year: int) -> int:
    """造一个 hive 分区的因子表，行数足够触发 batch 分批。

    结构：{root}/factors/{factor_id}/year=YYYY/data.parquet
    返回总行数。
    """
    factor_dir = root / "factors" / factor_id
    total = 0
    # 3 年 × rows_per_year 行
    years = [2022, 2023, 2024]
    assets = ["AAPL", "MSFT", "GOOG", "META", "NVDA"]
    for year in years:
        out_dir = factor_dir / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        # 按天 + 资产展开，保证行数 == rows_per_year
        base = pd.Timestamp(f"{year}-01-01")
        for i in range(rows_per_year):
            rows.append(
                {
                    "datetime": base + pd.Timedelta(days=i % 300),
                    "asset": assets[i % len(assets)],
                    "value": float(i) / 100.0 + year,
                }
            )
        pd.DataFrame(rows).to_parquet(out_dir / "data.parquet")
        total += rows_per_year
    return total


@pytest.fixture
def streaming_store(tmp_path, monkeypatch):
    """搭一个 tmp_path 下的 store，带一个 1500 行的 hive 分区因子表。"""
    lake_root = tmp_path / "lake"
    total = _make_large_factor_fixture(lake_root, "mom_big", rows_per_year=500)

    monkeypatch.setenv("TEST_LAKE_ROOT", str(lake_root))
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(
        dedent(
            """
            big_factor:
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
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    reset_store()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store, total
    engine.close()
    reset_store()


# ---- read_arrow_stream ------------------------------------------------------


def test_stream_yields_multiple_batches(streaming_store):
    """batch_size 小于总行数 → 至少 2 个 batch，合并后行数一致。"""
    store, total = streaming_store
    batches = list(
        store.read_arrow_stream(
            "big_factor",
            factor_id="mom_big",
            columns=["datetime", "asset", "value"],
            batch_size=200,
        )
    )
    assert len(batches) >= 2, f"期望至少 2 个 batch，实际 {len(batches)}"
    assert all(isinstance(b, pa.RecordBatch) for b in batches)

    merged = pa.Table.from_batches(batches)
    assert merged.num_rows == total


def test_stream_parity_with_read_arrow(streaming_store):
    """流式读合并后与 read_arrow 一次拿全是同一批数据（语义一致）。"""
    store, _ = streaming_store
    full = store.read_arrow(
        "big_factor",
        factor_id="mom_big",
        columns=["datetime", "asset", "value"],
    )
    streamed = pa.Table.from_batches(
        list(
            store.read_arrow_stream(
                "big_factor",
                factor_id="mom_big",
                columns=["datetime", "asset", "value"],
                batch_size=300,
            )
        )
    )
    # 行数一致
    assert streamed.num_rows == full.num_rows
    # 按 (datetime, asset) 规范化后内容一致
    df_full = full.to_pandas().sort_values(["datetime", "asset"]).reset_index(drop=True)
    df_stream = (
        streamed.to_pandas().sort_values(["datetime", "asset"]).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(df_full, df_stream, check_dtype=False)


def test_stream_applies_time_range(streaming_store):
    """time_range 下推到 DuckDB，只扫目标年份。"""
    store, _ = streaming_store
    batches = list(
        store.read_arrow_stream(
            "big_factor",
            factor_id="mom_big",
            columns=["datetime", "asset"],
            time_range=("2024-01-01", "2024-12-31"),
            batch_size=500,
        )
    )
    merged = pa.Table.from_batches(batches)
    # fixture 每年固定 500 行
    assert merged.num_rows == 500


def test_stream_applies_instrument_filter(streaming_store):
    store, total = streaming_store
    batches = list(
        store.read_arrow_stream(
            "big_factor",
            factor_id="mom_big",
            columns=["asset", "value"],
            instrument_filter=["AAPL"],
            batch_size=500,
        )
    )
    merged = pa.Table.from_batches(batches)
    # assets 池共 5 个，均匀分布：AAPL 占 1/5
    assert merged.num_rows == total // 5
    assert set(merged.column("asset").to_pylist()) == {"AAPL"}


def test_stream_break_midway_releases(streaming_store):
    """中途 break 不应抛错；GC 清理 reader 时 cursor 随之释放。"""
    store, _ = streaming_store
    gen = store.read_arrow_stream(
        "big_factor",
        factor_id="mom_big",
        columns=["asset"],
        batch_size=100,
    )
    first = next(gen)
    assert first.num_rows > 0
    # 立即 close 生成器，模拟调用方中途 break
    gen.close()
    # 后续能继续发起新查询，证明 engine 没挂
    tbl = store.read_arrow(
        "big_factor",
        factor_id="mom_big",
        columns=["asset"],
    )
    assert tbl.num_rows > 0


# ---- scan_polars ------------------------------------------------------------


def test_scan_polars_returns_lazyframe(streaming_store):
    import polars as pl

    store, _ = streaming_store
    lf = store.scan_polars(
        "big_factor",
        factor_id="mom_big",
        columns=["datetime", "asset", "value"],
    )
    assert isinstance(lf, pl.LazyFrame)


def test_scan_polars_collect_parity_with_read_frame(streaming_store):
    """scan_polars().collect() 语义 == read_frame。"""
    store, _ = streaming_store
    df_eager = store.read_frame(
        "big_factor",
        factor_id="mom_big",
        columns=["datetime", "asset", "value"],
    )
    lf = store.scan_polars(
        "big_factor",
        factor_id="mom_big",
        columns=["datetime", "asset", "value"],
    )
    df_lazy = lf.collect().to_pandas()

    # 规范化排序后对比
    def _c(df):
        return (
            df.sort_values(["datetime", "asset"])
            .reset_index(drop=True)[["datetime", "asset", "value"]]
        )

    pd.testing.assert_frame_equal(_c(df_eager), _c(df_lazy), check_dtype=False)


def test_scan_polars_time_range_pushdown(streaming_store):
    """time_range 作为 lazy filter 附加：collect() 后只返回目标区间的行。"""
    store, _ = streaming_store
    lf = store.scan_polars(
        "big_factor",
        factor_id="mom_big",
        columns=["datetime", "asset", "value"],
        time_range=("2024-01-01", "2024-12-31"),
    )
    df = lf.collect().to_pandas()
    assert len(df) == 500
    # 全部落在 2024 内
    assert df["datetime"].min() >= pd.Timestamp("2024-01-01")
    assert df["datetime"].max() <= pd.Timestamp("2024-12-31 23:59:59")


def test_scan_polars_instrument_filter(streaming_store):
    store, total = streaming_store
    lf = store.scan_polars(
        "big_factor",
        factor_id="mom_big",
        columns=["asset", "value"],
        instrument_filter=["AAPL", "MSFT"],
    )
    df = lf.collect().to_pandas()
    # AAPL + MSFT 占 2/5
    assert len(df) == total * 2 // 5
    assert set(df["asset"].unique()) == {"AAPL", "MSFT"}


def test_scan_polars_chains_user_filter(streaming_store):
    """调用方可以在返回的 LazyFrame 上继续加 filter，Polars optimizer 一起推。"""
    import polars as pl

    store, _ = streaming_store
    lf = store.scan_polars(
        "big_factor",
        factor_id="mom_big",
        columns=["datetime", "asset", "value"],
        time_range=("2024-01-01", None),
    )
    df = (
        lf.filter(pl.col("value") >= 2024.0)
        .sort(["datetime", "asset"])
        .collect()
        .to_pandas()
    )
    # 2024 年所有 value 都 >= 2024.0（value = i/100 + year）
    assert len(df) == 500
    assert df["value"].min() >= 2024.0


def test_scan_polars_missing_raises_importerror(streaming_store, monkeypatch):
    """polars 没装时报清晰 ImportError。"""
    store, _ = streaming_store

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "polars":
            raise ImportError("No module named 'polars'")
        return real_import(name, globals, locals, fromlist, level)

    # 清掉已加载的 polars 模块，强制走 import 路径
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "polars", raising=False)

    with pytest.raises(ImportError, match="polars"):
        store.scan_polars("big_factor", factor_id="mom_big")
