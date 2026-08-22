"""
PR3 store.sql 有限 SQL 逃生口的 contract 测试。

覆盖：
    - 基本 SELECT：单表、GROUP BY、聚合
    - 多数据集 JOIN
    - 用户 `?` 参数绑定
    - 禁词拦截：INSERT / UPDATE / DELETE / COPY / read_parquet / ...
    - 空 query / 空 read_datasets / 未注册 dataset
    - TEMP VIEW 生命周期：执行后 catalog 里不应残留
    - 审计日志：op=sql 每次都有
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pytest

from data_access import reset_store
from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import EngineError, ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture
def sql_store(tmp_path, monkeypatch):
    """搭两个 published 数据集，用于 sql() 读。写入通过 pyarrow 预先铺好。"""
    lake_root = tmp_path / "lake"
    lake_root.mkdir()

    # factors 参数化（factor_id），hive 分区 year
    factor_dir = lake_root / "factors" / "mom_3d" / "year=2024"
    factor_dir.mkdir(parents=True)
    pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-01"]),
        "asset": ["AAPL", "AAPL", "AAPL", "MSFT"],
        "value": [1.0, 2.0, 3.0, 10.0],
    }).to_parquet(factor_dir / "data.parquet")

    # prices 静态表，没有分区
    price_dir = lake_root / "prices"
    price_dir.mkdir()
    pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-01"]),
        "asset": ["AAPL", "AAPL", "AAPL", "MSFT"],
        "close": [100.0, 110.0, 120.0, 400.0],
    }).to_parquet(price_dir / "prices.parquet")

    monkeypatch.setenv("TEST_LAKE_ROOT", str(lake_root))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "shw")
    monkeypatch.setenv("QUANT_OPERATOR", "shw@test")
    monkeypatch.setenv("QUANT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))

    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        factors:
          kind: parametric
          access_mode: published
          layout: hive
          root_template: ${TEST_LAKE_ROOT}/factors/{factor_id}
          glob_template: "year=*/*.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true

        prices:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_LAKE_ROOT}/prices
          time_column: datetime
          instrument_column: asset
          union_by_name: true
    """).strip() + "\n", encoding="utf-8")

    reset_store()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store, tmp_path, engine
    engine.close()
    reset_store()


# ---- 基本 SELECT ------------------------------------------------------------

def test_sql_basic_select(sql_store):
    store, _, _ = sql_store
    tbl = store.sql(
        "SELECT asset, value FROM {{factors}} WHERE value > 1.5",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    assert tbl.num_rows == 3
    assert set(tbl.column_names) == {"asset", "value"}


def test_sql_group_by(sql_store):
    store, _, _ = sql_store
    tbl = store.sql(
        "SELECT asset, COUNT(*) AS n, AVG(value) AS mean_val "
        "FROM {{factors}} GROUP BY asset ORDER BY asset",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    df = tbl.to_pandas()
    assert df.loc[df["asset"] == "AAPL", "n"].iloc[0] == 3
    assert df.loc[df["asset"] == "MSFT", "n"].iloc[0] == 1


def test_sql_join_two_datasets(sql_store):
    """prices JOIN factors on (datetime, asset)，验证两个 view 能并存。"""
    store, _, _ = sql_store
    tbl = store.sql(
        dedent("""
            SELECT f.asset, f.datetime, f.value, p.close
            FROM {{factors}} f
            JOIN {{prices}} p USING (datetime, asset)
            ORDER BY f.asset, f.datetime
        """).strip(),
        read_datasets=["factors", "prices"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    assert tbl.num_rows == 4
    assert set(tbl.column_names) == {"asset", "datetime", "value", "close"}


def test_sql_user_params_binding(sql_store):
    """用户 SQL 里的 ? 绑定参数。"""
    store, _, _ = sql_store
    tbl = store.sql(
        "SELECT COUNT(*) AS n FROM {{factors}} WHERE value > ? AND asset = ?",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
        params=[1.5, "AAPL"],
    )
    assert tbl.to_pandas()["n"].iloc[0] == 2


# ---- 禁词拦截 ---------------------------------------------------------------

@pytest.mark.parametrize("bad_query", [
    "INSERT INTO factors VALUES (1)",
    "UPDATE factors SET value=1",
    "DELETE FROM factors",
    "CREATE TABLE foo AS SELECT * FROM factors",
    "DROP TABLE factors",
    "ALTER TABLE factors ADD COLUMN x INT",
    "COPY factors TO '/tmp/out.csv'",
    "ATTACH '/tmp/foo.db'",
    "PRAGMA threads=1",
    "SELECT * FROM read_parquet('/tmp/evil.parquet')",
    "SELECT * FROM read_csv('/tmp/evil.csv')",
    "SELECT * FROM glob('/etc/passwd')",
])
def test_sql_rejects_forbidden_keywords(sql_store, bad_query):
    store, _, _ = sql_store
    with pytest.raises(ValidationError, match="禁用关键字"):
        store.sql(
            bad_query,
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
        )


def test_sql_rejects_empty_query(sql_store):
    store, _, _ = sql_store
    with pytest.raises(ValidationError, match="空"):
        store.sql("   ", read_datasets=["factors"])


def test_sql_rejects_empty_read_datasets(sql_store):
    store, _, _ = sql_store
    with pytest.raises(ValidationError, match="read_datasets"):
        store.sql("SELECT 1", read_datasets=[])


def test_sql_rejects_unknown_dataset(sql_store):
    store, _, _ = sql_store
    # registry.get 对未登记名会 raise ValidationError
    with pytest.raises(ValidationError):
        store.sql(
            "SELECT * FROM never_registered",
            read_datasets=["never_registered"],
        )


def test_sql_rejects_missing_params(sql_store):
    """参数化数据集没带必需参数 → raise（来自 registry 的参数校验）。"""
    store, _, _ = sql_store
    with pytest.raises(ValidationError):
        store.sql(
            "SELECT * FROM {{factors}} LIMIT 1",
            read_datasets=["factors"],
            # 缺 factor_id
        )


def test_sql_sql_syntax_error_raises_engine_error(sql_store):
    store, _, _ = sql_store
    with pytest.raises(EngineError, match="失败"):
        store.sql(
            "SELECT FROM FROM FROM {{factors}}",  # 语法错
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
        )


# ---- TEMP VIEW 生命周期 -----------------------------------------------------

def test_sql_cleans_up_views_after_success(sql_store):
    """成功执行后 DuckDB catalog 里不能留下 scoped TEMP VIEW。"""
    store, _, engine = sql_store
    store.sql(
        "SELECT asset FROM {{factors}}",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    rows = engine._conn.execute(
        "SELECT view_name FROM duckdb_views() WHERE view_name LIKE '__da_%'"
    ).fetchall()
    assert rows == [], f"scoped TEMP VIEW 没清: {rows}"


def test_sql_cleans_up_views_after_failure(sql_store):
    """用户 SQL 报错后，view 也要清掉，不能污染后续查询。"""
    store, _, engine = sql_store
    with pytest.raises(EngineError):
        store.sql(
            "SELECT * FROM {{factors}} WHERE no_such_col = 1",  # 列不存在 → duckdb raise
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
        )
    rows = engine._conn.execute(
        "SELECT view_name FROM duckdb_views() WHERE view_name LIKE '__da_%'"
    ).fetchall()
    assert rows == [], f"失败路径 scoped TEMP VIEW 没清: {rows}"

    # 失败后还能再跑一次成功查询（证明 view 清干净了）
    tbl = store.sql(
        "SELECT COUNT(*) AS n FROM {{factors}}",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    assert tbl.to_pandas()["n"].iloc[0] == 4


# ---- 审计 -------------------------------------------------------------------

def test_sql_records_audit_on_success(sql_store, tmp_path):
    store, root, _ = sql_store
    audit_path = root / "audit.jsonl"
    store.sql(
        "SELECT asset FROM {{factors}}",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    sql_lines = [json.loads(l) for l in lines if json.loads(l)["op"] == "sql"]
    assert len(sql_lines) == 1
    rec = sql_lines[0]
    assert rec["ok"] is True
    assert rec["rows"] == 4
    assert rec["dataset"] == "factors"
    assert rec["extra"]["datasets"] == ["factors"]


def test_sql_records_audit_on_failure(sql_store, tmp_path):
    store, root, _ = sql_store
    audit_path = root / "audit.jsonl"
    with pytest.raises(EngineError):
        store.sql(
            "SELECT * FROM {{factors}} WHERE no_such_col = 1",
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
        )
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    sql_lines = [json.loads(l) for l in lines if json.loads(l)["op"] == "sql"]
    assert len(sql_lines) == 1
    rec = sql_lines[0]
    assert rec["ok"] is False
    assert "EngineError" in rec["error"]


# ---- 回归：验证不能用 read_parquet 绕过路径白名单 --------------------------

def test_sql_cannot_bypass_path_authorizer(sql_store, tmp_path):
    """
    哪怕构造一个看起来像白名单路径的 read_parquet，sql() 也会在禁词检测阶段
    拒掉。这条线保证了 sql() 逃生口不会把 PathAuthorizer 的保护给打穿。
    """
    store, _, _ = sql_store
    # 即便路径真实存在，也必须 raise
    with pytest.raises(ValidationError, match="禁用关键字"):
        store.sql(
            "SELECT * FROM read_parquet('/tmp/whatever.parquet')",
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
        )


def test_sql_stream_matches_sql(sql_store):
    store, _, _ = sql_store
    full = store.sql(
        "SELECT asset, value FROM {{factors}} ORDER BY datetime",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
    )
    batches = list(
        store.sql_stream(
            "SELECT asset, value FROM {{factors}} ORDER BY datetime",
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
            batch_size=2,
        )
    )
    import pyarrow as pa

    merged = pa.Table.from_batches(batches)
    assert merged.num_rows == full.num_rows
    assert merged.column_names == full.column_names


def test_sql_requires_view_columns_in_production_mode(sql_store, monkeypatch):
    store, _, _ = sql_store
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    with pytest.raises(ValidationError, match="view_columns"):
        store.sql(
            "SELECT asset FROM {{factors}}",
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
        )


def test_sql_with_view_columns_in_production_mode(sql_store, monkeypatch):
    store, _, _ = sql_store
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    tbl = store.sql(
        "SELECT asset, value FROM {{factors}} WHERE value > 1.5",
        read_datasets=["factors"],
        read_params={"factors": {"factor_id": "mom_3d"}},
        view_columns={"factors": ["datetime", "asset", "value"]},
    )
    assert tbl.num_rows == 3


def test_parallel_sql_queries(sql_store):
    """并发 sql() 使用唯一 TEMP VIEW，不应 catalog 冲突。"""
    store, _, _ = sql_store

    def run_one():
        tbl = store.sql(
            "SELECT COUNT(*) AS n FROM {{factors}} WHERE asset = ?",
            read_datasets=["factors"],
            read_params={"factors": {"factor_id": "mom_3d"}},
            params=["AAPL"],
        )
        return tbl.to_pandas()["n"].iloc[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result(timeout=30) for f in [pool.submit(run_one) for _ in range(16)]]

    assert all(r == 3 for r in results)
