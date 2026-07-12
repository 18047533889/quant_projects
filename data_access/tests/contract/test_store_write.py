"""PR2 write 路径的端到端 contract 测试。

覆盖：
    - write_arrow 写 namespaced 数据集（overwrite / append）
    - 带 partition_by 的 hive 写入
    - 权限拒绝：published 直写 raise
    - write 后能立刻 read 回来（读写闭环）
    - 审计日志每次写都落一行
    - publish_from_staging 占位 API 明确 raise NotImplementedError
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pyarrow as pa
import pytest

from data_access import reset_store
from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import ValidationError
from data_access.registry import load_registry
from data_access.store import DataAccessStore


@pytest.fixture
def write_store(tmp_path, monkeypatch):
    """搭一个带 published / namespaced / staging 三种数据集的 store。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # 给 published 数据集准备一个已存在的目录（读测试用）
    pub_root = tmp_path / "published" / "day_aggs"
    pub_root.mkdir(parents=True)
    pd.DataFrame({
        "align_time": [pd.Timestamp("2024-01-01 05:00:00", tz="UTC")],
        "ticker": ["AAPL"],
        "close": [190.0],
    }).to_parquet(pub_root / "seed.parquet")

    monkeypatch.setenv("TEST_WORKSPACE", str(workspace))
    monkeypatch.setenv("TEST_PUB_ROOT", str(pub_root))
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "shw")
    monkeypatch.setenv("QUANT_OPERATOR", "shw@test")
    # 审计日志重定向到 tmp，免得污染真实 workspace
    monkeypatch.setenv("QUANT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))

    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        pub_ds:
          kind: static
          access_mode: published
          layout: plain
          root: ${TEST_PUB_ROOT}
          time_column: align_time
          instrument_column: ticker
          union_by_name: true

        my_ns_ds:
          kind: parametric
          access_mode: namespaced
          layout: plain
          root_template: ${TEST_WORKSPACE}/users/${RUN_NAMESPACE}/runs/{strategy_id}
          glob_template: "**/*.parquet"
          params_schema:
            strategy_id: str
          time_column: timestamp
          instrument_column: symbol
          union_by_name: true

        my_stg_ds:
          kind: parametric
          access_mode: staging
          layout: hive
          root_template: ${TEST_WORKSPACE}/staging/${RUN_NAMESPACE}/factors/{factor_id}
          glob_template: "year=*/*.parquet"
          params_schema:
            factor_id: str
          time_column: datetime
          instrument_column: asset
          hive_partitioning: true
          union_by_name: true
    """).strip() + "\n", encoding="utf-8")

    reset_store()
    reg = load_registry(yaml_path)
    engine = DuckDBEngine(threads=2)
    store = DataAccessStore(registry=reg, engine=engine)
    yield store, tmp_path
    engine.close()
    reset_store()


def _sample_table(n: int = 5) -> pa.Table:
    return pa.table({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "symbol": [f"SYM{i:02d}" for i in range(n)],
        "pnl": [i * 0.5 for i in range(n)],
    })


# ---- 基本写入 ---------------------------------------------------------------

def test_write_namespaced_overwrite_then_read(write_store):
    store, _ = write_store
    tbl = _sample_table(10)
    result = store.write_arrow("my_ns_ds", tbl, strategy_id="mom_3d", mode="overwrite")

    assert result["rows"] == 10
    assert result["mode"] == "overwrite"
    assert Path(result["path"]).exists()

    # 读回来
    got = store.read_frame("my_ns_ds", strategy_id="mom_3d", columns=["symbol", "pnl"])
    assert len(got) == 10
    assert set(got["symbol"]) == {f"SYM{i:02d}" for i in range(10)}


def test_write_overwrite_clears_previous(write_store):
    store, _ = write_store
    store.write_arrow("my_ns_ds", _sample_table(10), strategy_id="mom_3d", mode="overwrite")
    # 第二次 overwrite，只剩 3 行
    store.write_arrow("my_ns_ds", _sample_table(3), strategy_id="mom_3d", mode="overwrite")
    got = store.read_frame("my_ns_ds", strategy_id="mom_3d", columns=["symbol"])
    assert len(got) == 3


def test_write_append_keeps_previous(write_store):
    store, _ = write_store
    store.write_arrow("my_ns_ds", _sample_table(5), strategy_id="mom_3d", mode="overwrite")
    store.write_arrow("my_ns_ds", _sample_table(3), strategy_id="mom_3d", mode="append")
    got = store.read_frame("my_ns_ds", strategy_id="mom_3d", columns=["symbol"])
    assert len(got) == 8


# ---- 分区写 -----------------------------------------------------------------

def test_write_with_partition_by(write_store):
    store, _ = write_store
    tbl = pa.table({
        "datetime": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01"]),
        "asset": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "value": [1.0, 2.0, 3.0, 4.0],
        "year": [2023, 2023, 2024, 2024],
    })
    result = store.write_arrow(
        "my_stg_ds", tbl, factor_id="mom_3d",
        mode="overwrite", partition_by=["year"],
    )
    assert result["rows"] == 4
    # 确认生成了 hive 分区结构
    base = Path(result["path"])
    assert (base / "year=2023").exists()
    assert (base / "year=2024").exists()

    # 读回来（staging 的 glob 是 year=*/data.parquet；write 用 pa.dataset 生成的是
    # year=*/part-*.parquet，这里直接用 read_arrow 通过 glob 展开 year=*/）
    got = store.read_frame("my_stg_ds", factor_id="mom_3d", columns=["asset", "value"])
    assert len(got) == 4


# ---- 权限拒绝 ---------------------------------------------------------------

def test_write_to_published_dataset_raises(write_store):
    store, _ = write_store
    with pytest.raises(ValidationError, match="published"):
        store.write_arrow("pub_ds", _sample_table(1), mode="overwrite")


def test_invalid_mode_raises(write_store):
    store, _ = write_store
    with pytest.raises(ValidationError, match="mode"):
        store.write_arrow("my_ns_ds", _sample_table(1), strategy_id="x", mode="upsert")


def test_write_rejects_non_arrow_table(write_store):
    store, _ = write_store
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValidationError, match="pyarrow.Table"):
        store.write_arrow("my_ns_ds", df, strategy_id="x")  # type: ignore[arg-type]


def test_publish_from_staging_requires_matching_params(write_store):
    """PR3 起 publish_from_staging 不再是占位；但 staging 和 target 的任何关键元数据
    不一致都会 raise。详细的 publish 成功/归档/回滚测试在 test_publish.py。"""
    store, _ = write_store
    # my_stg_ds 是 parametric（factor_id） + time=datetime；pub_ds 是 static + time=align_time
    # 多点不一致，publish 的校验顺序是：access_mode → params → time_column → instrument → layout
    # 这里 time_column 先失败，这已经足够证明"不匹配就拒"。
    with pytest.raises(ValidationError, match="不一致|params_schema|layout"):
        store.publish_from_staging("my_stg_ds", "pub_ds", factor_id="x")


# ---- 审计日志 ---------------------------------------------------------------

def test_audit_log_recorded_on_successful_write(write_store, tmp_path):
    store, root = write_store
    audit_path = root / "audit.jsonl"
    store.write_arrow("my_ns_ds", _sample_table(7), strategy_id="mom_3d", mode="overwrite")

    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    # 至少有一条 write 成功记录（可能之前 read 的也进来，取决于 QUANT_AUDIT_READS）
    write_records = [json.loads(l) for l in lines if json.loads(l)["op"] == "write"]
    assert len(write_records) == 1
    r = write_records[0]
    assert r["dataset"] == "my_ns_ds"
    assert r["ok"] is True
    assert r["rows"] == 7
    assert r["mode"] == "overwrite"
    assert r["namespace"] == "shw"
    assert r["operator"] == "shw@test"


def test_audit_log_recorded_on_failed_write(write_store, tmp_path, monkeypatch):
    store, root = write_store
    audit_path = root / "audit.jsonl"

    # 把 write_table_to_dir 搞坏，模拟写盘失败
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(DataAccessStore, "_write_table_to_dir", staticmethod(boom))

    with pytest.raises(OSError):
        store.write_arrow("my_ns_ds", _sample_table(3), strategy_id="mom_3d", mode="overwrite")

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    write_records = [json.loads(l) for l in lines if json.loads(l)["op"] == "write"]
    assert len(write_records) == 1
    assert write_records[0]["ok"] is False
    assert "disk full" in write_records[0]["error"]
