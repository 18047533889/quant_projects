"""最终收官轮 round6 DoD：本轮第三批审计 9 项修复的组合验证。

对应审计清单（main @ 3550257f0）：
    1.  Manifest save 每次 mint 全新 generation（不复用旧 gen）
    2.  datasets.yaml 顶层 Dataset strict schema（unknown key 启动即失败）
    3.  partitioning 唯一 schema（loader 编译 typed PartitionSpec；split-brain 消除）
    4.  storage 唯一 schema（StorageSpec.from_yaml；registry 存 typed spec）
    5.  TemporalJoinSpec 自身 invariant（direct 构造也 fail-closed）+ dict unknown-key
    6.  early-close 加载 fail-closed（production/strict 不把「读不到」当「没有」）
    7.  calendar 必须有 trading-day flag（strict 下缺标志 fail-closed）
    8.  QueryPolicy strict typed config（unknown key / bool 进数值预算拒绝）
    9.  factor PIVOT 唯一性门 + FactorCatalog corruption fail-closed + save 确定性

全部在本地临时数据集上验证，不依赖生产数据 / GitHub。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import DataError, ValidationError
from data_access.core.storage import StorageSpec, parse_storage_spec
from data_access.read.factors import (
    FactorCatalog,
    FactorMeta,
    build_factor_duplicate_check_sql,
)
from data_access.read.manifest import DatasetManifest
from data_access.read.partition_planner import PartitionSpec, parse_partitioning
from data_access.read.query_budget import parse_dataset_query_policy
from data_access.read.temporal_join import TemporalJoinSpec, parse_join_spec
from data_access.registry import load_registry
from data_access.store import DataAccessStore


def _load_cfg(tmp_path: Path, yaml_text: str):
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    return load_registry(cfg)


# ---------------------------------------------------------------------------
# 1) Manifest generation：每次 commit 全新 generation
# ---------------------------------------------------------------------------

def test_manifest_save_never_reuses_old_generation(tmp_path):
    root = tmp_path / "rg"
    root.mkdir()
    m = DatasetManifest(dataset="d", time_column="ts", files=(), manifest_generation_id="gen-A")
    m.save(root)
    gen1 = DatasetManifest.load(root).manifest_generation_id
    assert gen1 is not None and gen1 != "gen-A"
    # 同一对象再 commit（模拟增量重建）→ 也必须换新代
    m2 = DatasetManifest.load(root) or DatasetManifest(dataset="d", time_column="ts")
    m2.save(root)
    gen2 = DatasetManifest.load(root).manifest_generation_id
    assert gen2 is not None and gen2 != gen1


# ---------------------------------------------------------------------------
# 2) datasets.yaml 顶层 strict schema
# ---------------------------------------------------------------------------

def test_dataset_top_level_unknown_key_fails(tmp_path):
    with pytest.raises(ValidationError, match="未知配置"):
        _load_cfg(
            tmp_path,
            f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
  query_polcy: {{}}          # typo → 启动即失败
""",
        )
    # parametric 侧同样
    with pytest.raises(ValidationError, match="未知配置"):
        _load_cfg(
            tmp_path,
            f"""
a:
  kind: parametric
  access_mode: published
  layout: plain
  root_template: "{tmp_path}/{{p}}"
  glob_template: "*.parquet"
  params_schema: {{p: str}}
  time_column: t
  formatt: parquet            # typo
""",
        )


def test_dataset_static_parametric_key_mismatch_fails(tmp_path):
    # static 用 root_template → 未知 key
    with pytest.raises(ValidationError, match="未知配置"):
        _load_cfg(
            tmp_path,
            f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root_template: "{tmp_path}/{{p}}"
  glob: "*.parquet"
  time_column: t
""",
        )


# ---------------------------------------------------------------------------
# 3) partitioning 唯一 schema
# ---------------------------------------------------------------------------

def test_loader_compiles_typed_partitioning(tmp_path):
    reg = _load_cfg(
        tmp_path,
        f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
  partitioning:
    time:
      source: filename
      field: date
      frequency: daily
      pattern: "{{date}}.parquet"
    hive: [date]
""",
    )
    ds = reg.get("a")
    assert isinstance(ds.partitioning, PartitionSpec)
    assert ds.partitioning.time.pattern == "{date}.parquet"
    assert ds.partitioning.hive == ("date",)
    # typed passthrough：planner 不再二次猜 dict
    assert parse_partitioning(ds.partitioning) is ds.partitioning


def test_partitioning_splitbrain_keys_rejected(tmp_path):
    # 旧 loader 允许但 planner 不认的 keys → 现在直接报错
    with pytest.raises(ValidationError, match="partitioning.*未知"):
        parse_partitioning({"columns": ["a"], "bucket": 4})
    with pytest.raises(ValidationError, match="partitioning.*未知"):
        parse_partitioning({"time_column": "ts"})
    with pytest.raises(ValidationError, match="time.*未知"):
        parse_partitioning({"time": {"source": "filename", "typo": 1}})


# ---------------------------------------------------------------------------
# 4) storage 唯一 schema
# ---------------------------------------------------------------------------

def test_storage_nested_source_resolves_cos(tmp_path):
    reg = _load_cfg(
        tmp_path,
        f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
  storage:
    source:
      type: cos
      uri: cos://bucket/prefix
      layout: hive_date
""",
    )
    ds = reg.get("a")
    assert isinstance(ds.storage, StorageSpec)
    assert ds.storage.type == "cos"
    assert ds.storage.uri == "cos://bucket/prefix"
    assert ds.storage.layout == "hive_date"
    # typed spec 被 parse_storage_spec 原样返回
    assert parse_storage_spec(ds.storage) is ds.storage


def test_storage_top_level_wins_and_unknown_rejected(tmp_path):
    spec = parse_storage_spec({"type": "s3", "source": {"type": "cos"}, "uri": "s3://b/k"})
    assert spec.type == "s3"
    with pytest.raises(ValidationError, match="未知配置"):
        parse_storage_spec({"typo": 1})
    with pytest.raises(ValidationError, match="source.*未知"):
        parse_storage_spec({"source": {"type": "cos", "nope": 1}})
    # 顶层缺 type 但 source 提供了 → 不再静默当 local
    spec2 = parse_storage_spec({"source": {"type": "cos", "uri": "cos://b/k"}})
    assert spec2.type == "cos"


# ---------------------------------------------------------------------------
# 5) TemporalJoinSpec 自身 invariant + dict unknown-key
# ---------------------------------------------------------------------------

def test_temporal_join_spec_direct_construction_fail_closed():
    with pytest.raises(ValidationError):
        TemporalJoinSpec(policy="xxx")
    with pytest.raises(ValidationError):
        TemporalJoinSpec(availability="whatever")
    with pytest.raises(ValidationError):
        TemporalJoinSpec(availability_latency=-1)
    with pytest.raises(ValidationError):
        TemporalJoinSpec(availability_latency=True)
    with pytest.raises(ValidationError):
        TemporalJoinSpec(duplicate_policy="bogus")
    with pytest.raises(ValidationError):
        TemporalJoinSpec(period_selection="bogus")


def test_temporal_join_dict_unknown_and_latency_guards():
    with pytest.raises(ValidationError, match="未知 key"):
        parse_join_spec({"policy": "pit_asof", "availability_latncy": 2})
    with pytest.raises(ValidationError, match="bool"):
        parse_join_spec({"availability_latency": True})
    with pytest.raises(ValidationError, match="不能为负数"):
        parse_join_spec({"availability_latency": -5})
    with pytest.raises(ValidationError, match="bool 值无法识别"):
        parse_join_spec({"deduplicate": 2})


# ---------------------------------------------------------------------------
# 6) early-close 加载 fail-closed
# ---------------------------------------------------------------------------

def test_early_close_fail_closed_strict(tmp_path):
    # 未注册 us_is_early_close → strict 下抛
    reg = _load_cfg(
        tmp_path,
        f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
""",
    )
    from data_access.read.session_calendar import load_us_early_close_dates

    store = DataAccessStore(reg, DuckDBEngine(threads=2))
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(ValidationError, match="us_is_early_close"):
            load_us_early_close_dates(store)
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]


def test_early_close_tolerant_research(tmp_path):
    reg = _load_cfg(
        tmp_path,
        f"""
a:
  kind: static
  access_mode: published
  layout: plain
  root: {tmp_path}
  glob: "*.parquet"
  time_column: t
  instrument_column: s
""",
    )
    from data_access.read.session_calendar import load_us_early_close_dates

    store = DataAccessStore(reg, DuckDBEngine(threads=2))
    assert load_us_early_close_dates(store) == frozenset()


# ---------------------------------------------------------------------------
# 7) calendar 必须有 trading-day flag（strict）
# ---------------------------------------------------------------------------

def test_calendar_strict_requires_trading_day_flag(tmp_path):
    # 日历数据集 schema 里无 IsTradeDay/is_trading_day，物理列也没有 → strict 抛
    cal_root = tmp_path / "cal"
    cal_root.mkdir()
    pd.DataFrame({"cal_date": ["2024-01-01", "2024-01-02"]}).to_parquet(
        cal_root / "cal.parquet"
    )
    reg = _load_cfg(
        tmp_path,
        f"""
us_calendar:
  kind: static
  access_mode: published
  layout: plain
  root: {cal_root}
  glob: "*.parquet"
  time_column: cal_date
  instrument_column: cal_date
  schema:
    cal_date: string
""",
    )
    from data_access.read.session_calendar import get_market_calendar, reset_calendars

    reset_calendars()
    store = DataAccessStore(reg, DuckDBEngine(threads=2))
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(ValidationError, match="交易日标志列"):
            get_market_calendar("us", store=store)
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]
        reset_calendars()


def test_calendar_strict_accepts_flag_column(tmp_path):
    cal_root = tmp_path / "cal"
    cal_root.mkdir()
    pd.DataFrame(
        {
            "cal_date": ["2024-01-01", "2024-01-02"],
            "is_trading_day": [False, True],
        }
    ).to_parquet(cal_root / "cal.parquet")
    # strict 下加载 US 日历还依赖 early-close 合约（#6 fail-closed）——注册且非空
    ec_root = tmp_path / "ec"
    ec_root.mkdir()
    pd.DataFrame({"d": ["2024-01-03"]}).to_parquet(ec_root / "ec.parquet")
    reg = _load_cfg(
        tmp_path,
        f"""
us_calendar:
  kind: static
  access_mode: published
  layout: plain
  root: {cal_root}
  glob: "*.parquet"
  time_column: cal_date
  instrument_column: cal_date
  schema:
    cal_date: string
    is_trading_day: bool
us_is_early_close:
  kind: static
  access_mode: published
  layout: plain
  root: {ec_root}
  glob: "*.parquet"
  time_column: d
  instrument_column: d
  schema:
    d: string
""",
    )
    from data_access.read.session_calendar import get_market_calendar, reset_calendars

    reset_calendars()
    store = DataAccessStore(reg, DuckDBEngine(threads=2))
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        cal = get_market_calendar("us", store=store)
        # 只有 is_trading_day=True 的 01-02 进交易日
        assert dt.date(2024, 1, 2) in cal.trading_days
        assert dt.date(2024, 1, 1) not in cal.trading_days
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]
        reset_calendars()


# ---------------------------------------------------------------------------
# 8) QueryPolicy strict typed config
# ---------------------------------------------------------------------------

def test_query_policy_unknown_key_and_bool_rejected():
    with pytest.raises(ValidationError, match="未知 key"):
        parse_dataset_query_policy({"max_scan_file": 100}, context="ds")
    with pytest.raises(ValidationError, match="正整数"):
        parse_dataset_query_policy({"max_scan_files": True}, context="ds")
    with pytest.raises(ValidationError, match="有限正数"):
        parse_dataset_query_policy({"max_elapsed_ms": True}, context="ds")


# ---------------------------------------------------------------------------
# 9) factor PIVOT 唯一性门 + FactorCatalog corruption fail-closed + save 确定性
# ---------------------------------------------------------------------------

def test_factor_pivot_duplicate_check_sql():
    union = "SELECT 'f1' AS factor_id, TIMESTAMP '2024-01-01 09:00' AS datetime, 'A' AS asset, 1.0 AS value"
    sql, params = build_factor_duplicate_check_sql(union, [])
    assert "HAVING COUNT(*) > 1" in sql
    assert "LIMIT 1" in sql


def test_factor_pivot_duplicate_gate_fail_closed(tmp_path):
    # 因子目录带重复 (datetime, asset) → wide 读 strict 下抛
    lake = tmp_path / "lake"
    factors_dir = lake / "factors"
    fdir = factors_dir / "f1"
    fdir.mkdir(parents=True)
    pd.DataFrame(
        {
            "datetime": ["2024-01-02", "2024-01-02"],
            "asset": ["A", "A"],
            "value": [1.0, 2.0],
        }
    ).to_parquet(fdir / "part-0.parquet")
    (fdir / "_factor_meta.json").write_text(
        json.dumps({"factor_id": "f1", "dtype": "double"}), encoding="utf-8"
    )
    reg = _load_cfg(
        tmp_path,
        f"""
factor_lake:
  kind: parametric
  access_mode: published
  layout: plain
  root_template: "{lake}/factors/{{factor_id}}"
  glob_template: "*.parquet"
  params_schema: {{factor_id: str}}
  time_column: datetime
  instrument_column: asset
  schema:
    datetime: timestamp
    asset: string
    value: double
""",
    )
    store = DataAccessStore(reg, DuckDBEngine(threads=2))
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(ValidationError, match="重复"):
            store.read_factors(["f1"], layout="wide", time_range=("2024-01-01", "2024-01-31"))
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]


def test_factor_catalog_corruption_strict(tmp_path):
    root = tmp_path / "lake"
    root.mkdir()
    (root / "_factor_catalog.json").write_text("{ not json", encoding="utf-8")
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(DataError, match="损坏"):
            FactorCatalog.load(root)
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]
    # research 宽容返回空
    assert len(FactorCatalog.load(root)) == 0


def test_factor_catalog_meta_int_corruption_strict(tmp_path):
    root = tmp_path / "lake"
    root.mkdir()
    (root / "_factor_catalog.json").write_text(
        json.dumps({"factors": [{"factor_id": "f1", "instruments": "abc"}]}),
        encoding="utf-8",
    )
    os.environ["DATA_ACCESS_STRICT_READ"] = "1"
    try:
        with pytest.raises(ValidationError, match="instruments"):
            FactorCatalog.load(root)
    finally:
        del os.environ["DATA_ACCESS_STRICT_READ"]


def test_factor_catalog_save_is_deterministic(tmp_path):
    root = tmp_path / "lake"
    root.mkdir()
    cat = FactorCatalog(
        root=root,
        records={
            "z": FactorMeta(factor_id="z", dtype="double"),
            "a": FactorMeta(factor_id="a", dtype="double"),
        },
    )
    cat.save(root)
    first = (root / "_factor_catalog.json").read_text(encoding="utf-8")
    cat.save(root)
    second = (root / "_factor_catalog.json").read_text(encoding="utf-8")
    assert first == second
    assert json.loads(first)["factors"][0]["factor_id"] == "a"
