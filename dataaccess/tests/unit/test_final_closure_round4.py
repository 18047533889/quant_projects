# -*- coding: utf-8
"""第四轮收口回归：最后一批 P0 correctness 尾巴。

覆盖（对应 audit 清单）：
    P0-1  Governed Lazy：collect_polars_with_budget 返回 pa.Table，ReadHandle 的
          to_arrow/to_polars 不能再对 Arrow Table 调 .to_arrow() / 直接塞 _polars_df
    P0-2  Publish unique-key gate 必须 fail-closed（验证不了就拒绝），file-list 绑定
    P0-3  Publish candidate 实际数据必须符合 target 声明 schema 契约
    P0-4  Publish manifest 路径必须相对（files[].path 相对、base_dir="."）
    P0-5  Publish 拒绝 symlink（不跟随复制）
    P0-6  Upsert 列序错位必须拒绝（UNION ALL 按位置对齐）
    P0-7  Upsert dtype/nullability 不一致必须拒绝（禁止隐式 cast）
    P0-8  delete_rows 无范围 + 非 delete_all → 拒绝（防全删 footgun）
    P0-9  delete_rows 文件缺 time_column → production abort（不静默跳过）
    P0-10 COS panel columns=None + semantic_filters → 完整 declared schema 投影
    P0-11 COS event columns=None → 完整 declared schema + PIT 必需列投影
    P0-12 read_cos_events_asof 应用统一 availability（next_trading_day 下一交易日起见）
    P0-13 ContractIR event_time 不得取自值字段（Close/Open… 的 physical_name）
    P0-14 Manifest rowgroup sidecar 必须与 manifest 同 generation
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_access.core.engine import DuckDBEngine
from data_access.core.exceptions import DataError, ValidationError
from data_access.read.manifest import DatasetManifest, ManifestRowGroup, _save_row_groups
from data_access.read.query_budget import QueryBudget
from data_access.read.read_handle import ReadHandle
from data_access.registry import load_registry
from data_access.store import DataAccessStore


def _no_cos(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "mirror")
    monkeypatch.delenv("DATA_ACCESS_STRICT_READ", raising=False)
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)


def _store(tmp_path, datasets: dict[str, dict]) -> DataAccessStore:
    cfg = tmp_path / "datasets.yaml"
    lines = []
    for name, body in datasets.items():
        lines.append(f"{name}:")
        for k, v in body.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for sk, sv in v.items():
                    lines.append(f"    {sk}: {json.dumps(str(sv))}")
            else:
                lines.append(f"  {k}: {json.dumps(str(v))}")
    cfg.write_text("\n".join(lines), encoding="utf-8")
    return DataAccessStore(load_registry(cfg), DuckDBEngine(threads=2))


def _writable_ds(root: Path, name: str) -> dict[str, dict]:
    return {
        name: {
            "kind": "static", "access_mode": "staging", "layout": "plain",
            "root": str(root), "glob": "part-*.parquet",
            "time_column": "ts", "instrument_column": "sym",
            "schema": {"ts": "date", "sym": "string", "val": "double"},
        }
    }


# ---------------------------------------------------------------------------
# P0-1 Governed Lazy 类型 bug
# ---------------------------------------------------------------------------

def test_governed_lazy_to_arrow_polars_types():
    import polars as pl

    lf = pl.LazyFrame({"a": [1, 2, 3]})
    h = ReadHandle(lazy=lf, budget=QueryBudget(), govern_lazy=True)
    table = h.to_arrow()
    assert isinstance(table, pa.Table), "governed lazy to_arrow 必须返回 pyarrow.Table"
    df = h.to_polars()
    assert isinstance(df, pl.DataFrame), "governed lazy to_polars 必须是 polars.DataFrame"
    assert df["a"].to_list() == [1, 2, 3]
    # stream 也从统一 Arrow 源切 batch
    batches = list(h.stream(batch_size=2))
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    assert sum(b.num_rows for b in batches) == 3


# ---------------------------------------------------------------------------
# P0-2 / P0-3 Publish gate
# ---------------------------------------------------------------------------

def test_publish_unique_key_duplicate_rejected(tmp_path):
    from data_access.write.publish import _validate_unique_key

    root = tmp_path / "p"
    root.mkdir()
    pq.write_table(pa.table({"k": ["a", "a"], "v": [1, 2]}), str(root / "f.parquet"))
    with pytest.raises(DataError):
        _validate_unique_key(SimpleNamespace(name="t", unique_key=("k",)), root)


def test_publish_unique_key_missing_col_fail_closed(tmp_path):
    """#P0-2 key 列缺失 → DuckDB 验证失败 → 必须拒绝（不能吞掉放行）。"""
    from data_access.write.publish import _validate_unique_key

    root = tmp_path / "q"
    root.mkdir()
    pq.write_table(pa.table({"v": [1, 2]}), str(root / "f.parquet"))
    with pytest.raises(DataError):
        _validate_unique_key(SimpleNamespace(name="t", unique_key=("k",)), root)


def test_publish_unique_key_ok(tmp_path):
    from data_access.write.publish import _validate_unique_key

    root = tmp_path / "r"
    root.mkdir()
    pq.write_table(pa.table({"k": ["a", "b"], "v": [1, 2]}), str(root / "f.parquet"))
    _validate_unique_key(SimpleNamespace(name="t", unique_key=("k",)), root)  # no raise


def test_publish_candidate_missing_declared_column_rejected(tmp_path):
    from data_access.write.publish import _validate_candidate_contract

    root = tmp_path / "c"
    root.mkdir()
    pq.write_table(pa.table({"a": [1.0]}), str(root / "f.parquet"))
    with pytest.raises(DataError):
        _validate_candidate_contract(
            SimpleNamespace(schema={"a": "double", "b": "double"}), root
        )


def test_publish_candidate_type_mismatch_rejected(tmp_path):
    from data_access.write.publish import _validate_candidate_contract

    root = tmp_path / "c2"
    root.mkdir()
    pq.write_table(pa.table({"a": ["x"]}), str(root / "f.parquet"))
    with pytest.raises(DataError):
        _validate_candidate_contract(SimpleNamespace(schema={"a": "double"}), root)


def test_publish_candidate_contract_ok(tmp_path):
    from data_access.write.publish import _validate_candidate_contract

    root = tmp_path / "c3"
    root.mkdir()
    pq.write_table(pa.table({"a": [1.0]}), str(root / "f.parquet"))
    _validate_candidate_contract(SimpleNamespace(schema={"a": "double"}), root)


# ---------------------------------------------------------------------------
# P0-4 Publish manifest 相对路径
# ---------------------------------------------------------------------------

def test_publish_manifest_relative_paths(tmp_path):
    from data_access.write.publish_manifest import write_publish_manifest

    root = tmp_path / "m"
    root.mkdir()
    pq.write_table(pa.table({"a": [1]}), str(root / "f.parquet"))
    mp = write_publish_manifest(
        root, staging_name="s", target_name="t", params=None,
        rows=1, archive_path=None, elapsed_ms=0.1,
    )
    payload = json.loads(mp.read_text(encoding="utf-8"))
    assert payload["base_dir"] == "."
    assert payload["files"][0]["path"] == "f.parquet"
    assert all(not Path(e["path"]).is_absolute() for e in payload["files"])


# ---------------------------------------------------------------------------
# P0-5 Publish symlink 拒绝
# ---------------------------------------------------------------------------

def test_publish_rejects_symlink(tmp_path):
    from data_access.write.publish import _copy_tree

    src = tmp_path / "s"
    src.mkdir()
    real = src / "real.parquet"
    real.write_text("x")
    os.symlink(str(real), str(src / "link.parquet"))
    with pytest.raises(ValidationError):
        _copy_tree(src, tmp_path / "dst")


# ---------------------------------------------------------------------------
# P0-6 / P0-7 Upsert 列序 / dtype 对齐
# ---------------------------------------------------------------------------

def test_upsert_column_order_mismatch_rejected():
    from data_access.write.upsert import _assert_schema_compatible

    ex = pa.table({"a": [1], "b": [2.0]})
    new = pa.table({"b": [2.0], "a": [1]})
    with pytest.raises(ValidationError):
        _assert_schema_compatible(ex, new)


def test_upsert_dtype_mismatch_rejected():
    from data_access.write.upsert import _assert_schema_compatible

    ex = pa.table({"a": [1], "b": [2.0]})
    new = pa.table({"a": ["x"], "b": [2.0]})
    with pytest.raises(ValidationError):
        _assert_schema_compatible(ex, new)


def test_upsert_schema_ok():
    from data_access.write.upsert import _assert_schema_compatible

    ex = pa.table({"a": [1], "b": [2.0]})
    new = pa.table({"a": [3], "b": [4.0]})
    _assert_schema_compatible(ex, new)  # no raise


# ---------------------------------------------------------------------------
# P0-8 / P0-9 delete_rows 防护
# ---------------------------------------------------------------------------

def test_delete_rows_no_range_requires_delete_all(tmp_path):
    from data_access.write.upsert import delete_rows_from_dataset

    with pytest.raises(ValidationError):
        delete_rows_from_dataset(
            ds=SimpleNamespace(name="d"), authorizer=None, target_dir=tmp_path,
            time_column="ts", start=None, end=None, after=None,
        )
    # delete_all=True 但缺 reason → 仍拒绝（break-glass 语义）
    with pytest.raises(ValidationError):
        delete_rows_from_dataset(
            ds=SimpleNamespace(name="d"), authorizer=None, target_dir=tmp_path,
            time_column="ts", start=None, end=None, after=None, delete_all=True,
        )


def test_delete_rows_missing_time_column_aborts(tmp_path):
    from data_access.registry.paths import PathAuthorizer
    from data_access.write.upsert import delete_rows_from_dataset

    d = tmp_path / "d"
    d.mkdir()
    pq.write_table(pa.table({"sym": ["A"], "val": [1.0]}), str(d / "part-1.parquet"))
    auth = PathAuthorizer(allowed_roots=[tmp_path])
    # production：缺谓词列不能静默跳过 → 整个 delete abort
    with pytest.raises(DataError):
        delete_rows_from_dataset(
            ds=SimpleNamespace(name="myds"), authorizer=auth, target_dir=d,
            time_column="ts", start="2024-01-01", end="2024-01-02",
        )
    # 只有显式 best_effort=True 才允许跳过并记录 failed_files
    res = delete_rows_from_dataset(
        ds=SimpleNamespace(name="myds"), authorizer=auth, target_dir=d,
        time_column="ts", start="2024-01-01", end="2024-01-02",
        best_effort=True,
    )
    assert res.get("failed_files")


# ---------------------------------------------------------------------------
# P0-10 / P0-11 COS panel / event 投影
# ---------------------------------------------------------------------------

class _FakeReg:
    def __init__(self, schema, time_column):
        self._schema = schema
        self._tc = time_column

    def get(self, name):
        return SimpleNamespace(schema=self._schema, time_column=self._tc)


class _FakeSqlStore:
    """记录 view_columns 并返回同名空表；只测投影，不测数据。"""

    def __init__(self, schema, time_column):
        self._registry = _FakeReg(schema, time_column)
        self.captured: dict | None = None

    def sql(self, query, *, view_columns=None, **kwargs):
        self.captured = view_columns
        ds = next(iter(view_columns))
        cols = {c: [] for c in view_columns[ds]}
        return pa.table(cols) if cols else pa.table({"__e": []})


def test_cos_panel_columns_none_full_declared_schema():
    from data_access.cos_panel_runtime import read_cos_panel

    schema = {"TradeDate": "date", "Symbol": "string", "Close": "double", "Volume": "int"}
    store = _FakeSqlStore(schema, "TradeDate")
    read_cos_panel(store, "ashare_stock_daily", semantic_filters={"Industry": ["A"]})
    vc = set(store.captured["ashare_stock_daily"])
    assert {"TradeDate", "Symbol", "Close", "Volume", "Industry"} <= vc


def test_cos_panel_normalize_returns_adds_return_col():
    from data_access.cos_panel_runtime import read_cos_panel

    schema = {"TradeDate": "date", "Symbol": "string", "Close": "double"}
    store = _FakeSqlStore(schema, "TradeDate")
    read_cos_panel(
        store, "ashare_stock_daily", semantic_filters={"Industry": ["A"]},
        normalize_returns=True,
    )
    assert "Return" in store.captured["ashare_stock_daily"]


def test_cos_events_columns_none_full_declared_schema():
    from data_access.cos_event_runtime import read_cos_events

    schema = {
        "Symbol": "string", "PubDate": "date", "period_end": "date",
        "total_assets": "double",
    }
    store = _FakeSqlStore(schema, "TradeDate")
    read_cos_events(store, "ashare_stock_balance")
    vc = set(store.captured["ashare_stock_balance"])
    # 完整 declared schema 必须进 view（SELECT * 才看得见全列）
    assert {"Symbol", "PubDate", "period_end", "total_assets"} <= vc


# ---------------------------------------------------------------------------
# P0-12 read_cos_events_asof 统一 availability
# ---------------------------------------------------------------------------

class _FakeCal:
    """把每个 knowledge 映射成「下一自然日可用」的假日历。"""

    has_data = True
    timezone = "Asia/Shanghai"

    def available_from(self, knowledge, availability="next_trading_day"):
        return knowledge + dt.timedelta(days=1)


def _ev_contract() -> object:
    from data_access.cos_contract import COSDatasetContract

    return COSDatasetContract(
        name="ev", market="ashare", temporal_model="E1", join_policy="event",
        instrument_column="Symbol", panel_policy="event_only", pit_policy="strict",
        availability_column="knowledge", period_column=None, event_column=None,
    )


def _decisions():
    return {
        "__pit_position": np.arange(2, dtype=np.int64),
        "instrument": ["A", "A"],
        "decision_timestamp": [
            pd.Timestamp("2024-01-02 15:00", tz="UTC"),
            pd.Timestamp("2024-01-03 15:00", tz="UTC"),
        ],
    }


def test_asof_next_trading_day_visibility_with_calendar():
    from data_access.cos_event_runtime import _select

    events = pd.DataFrame({
        "Symbol": ["A", "A"],
        "knowledge": [
            pd.Timestamp("2024-01-02 09:00", tz="UTC"),
            pd.Timestamp("2024-01-03 09:00", tz="UTC"),
        ],
        "val": [10, 20],
    })
    decisions = pd.DataFrame(_decisions())
    # next_trading_day + 日历 → knowledge 当天不可见，下一交易日才可见。
    picked = _select(
        decisions, events, _ev_contract(), "decision_timestamp", "instrument",
        "knowledge", "latest_available", availability="next_trading_day",
        calendar=_FakeCal(),
    )
    # decision@2024-01-02：knowledge@01-02 的 available_from=01-03 > 决策 → 无
    # decision@2024-01-03：knowledge@01-02 的 available_from=01-03 <= 决策 → 取 10
    assert picked[0] is None
    assert picked[1] is not None and picked[1]["val"] == 10


def test_asof_next_trading_day_strict_without_calendar():
    from data_access.cos_event_runtime import _select

    events = pd.DataFrame({
        "Symbol": ["A"],
        "knowledge": [pd.Timestamp("2024-01-02 09:00", tz="UTC")],
        "val": [10],
    })
    decisions = pd.DataFrame({
        "__pit_position": [0, 1],
        "instrument": ["A", "A"],
        "decision_timestamp": [
            pd.Timestamp("2024-01-02 09:00", tz="UTC"),   # == knowledge → 严格不可见
            pd.Timestamp("2024-01-02 10:00", tz="UTC"),   # > knowledge → 可见
        ],
    })
    # 无日历 → strict 回退：knowledge < decision 才可见（同 comparison_operator）。
    picked = _select(
        decisions, events, _ev_contract(), "decision_timestamp", "instrument",
        "knowledge", "latest_available", availability="next_trading_day",
        calendar=None,
    )
    assert picked[0] is None  # decision == knowledge → 严格不可见
    assert picked[1] is not None and picked[1]["val"] == 10


def test_asof_same_day_keeps_inclusive():
    from data_access.cos_event_runtime import _select

    events = pd.DataFrame({
        "Symbol": ["A"],
        "knowledge": [pd.Timestamp("2024-01-02 09:00", tz="UTC")],
        "val": [10],
    })
    decisions = pd.DataFrame({
        "__pit_position": [0],
        "instrument": ["A"],
        "decision_timestamp": [pd.Timestamp("2024-01-02 15:00", tz="UTC")],
    })
    picked = _select(
        decisions, events, _ev_contract(), "decision_timestamp", "instrument",
        "knowledge", "latest_available", availability="same_day", calendar=None,
    )
    assert picked[0] is not None and picked[0]["val"] == 10


# ---------------------------------------------------------------------------
# P0-13 ContractIR event_time 不得取自值字段
# ---------------------------------------------------------------------------

def test_contract_ir_event_time_not_a_value_field():
    from data_access.read.contract_ir import build_contract_ir
    from data_access.read.semantic_catalog import get_semantic_catalog

    repo_root = Path(__file__).resolve().parents[2]
    ir = build_contract_ir(
        load_registry(str(repo_root / "config" / "datasets.yaml")),
        catalog=get_semantic_catalog(),
    )
    d = ir.get("ashare_stock_daily")
    assert d is not None
    ta = d.temporal_axes
    assert ta is not None and ta.event_time is not None
    # 值字段（Close/Open/Volume/MarketCap/PeRatio）绝不能被编译成 event_time 轴
    assert ta.event_time not in {
        "Close", "Open", "High", "Low", "Volume", "Amount",
        "Vwap", "MarketCap", "CirculatingMarketCap", "PeRatio",
    }
    # 面板数据集的事件时间轴 = registry time_column（bar 时间即事件时间）
    assert ta.event_time == ta.partition_time == "TradeDate"
    # 事件数据集：event_time = 契约时钟列（availability_column），不是值字段
    ev = ir.get("ashare_stock_balance")
    assert ev is not None and ev.temporal_axes is not None
    assert ev.temporal_axes.event_time == "PubDate"


# ---------------------------------------------------------------------------
# P0-14 Manifest rowgroup 绑定 generation
# ---------------------------------------------------------------------------

def test_manifest_rowgroup_generation_binding(tmp_path):
    root = tmp_path / "rg"
    root.mkdir()
    m = DatasetManifest(
        dataset="d", time_column="ts",
        files=(), manifest_generation_id="gen-A",
    )
    m.save(root)
    rg = ManifestRowGroup(
        path="f.parquet", row_group=0, column="ts",
        min="2024-01-01", max="2024-01-02", null_count=0, rows=10,
    )
    # 异代 sidecar（gen-B）配 manifest（gen-A）→ load 必须忽略
    _save_row_groups(root, [rg], generation="gen-B")
    loaded = DatasetManifest.load(root)
    assert loaded is not None
    assert loaded.row_groups is None, "异代 rowgroup sidecar 不得被新 generation 使用"
    # 同代 sidecar → 正常加载
    _save_row_groups(root, [rg], generation="gen-A")
    loaded2 = DatasetManifest.load(root)
    assert loaded2 is not None and loaded2.row_groups is not None
    assert loaded2.row_groups[0].rows == 10
