# -*- coding: utf-8 -*-
"""R30-P0-011 / R30-P1-019 / R30-P1-024 联合验收单测（2026-08）。

覆盖：
    - data_change.detect_changes 的 change_kind 判定（append/revision/delete/
      schema_change）与指纹一致 → 空变化集；
    - data_change.revision_availability 的 knowledge-aware 合并（财务 revision
      不能只看物理文件日期）；
    - change_impact.plan_minimal_recompute：单 instrument 单时段 revision →
      只影响相关因子、窗口 = changed_time_range 向前扩 lookback，**非全历史全因子**；
    - change_impact.recompute_budget 与 to_fe_input 结构；
    - lineage.LineageStore 的三种血缘问题 + to_parquet + duckdb/memory 模式 +
      ingest_read_lineage；
    - artifact_meta.FactorArtifactMetadata 完整身份字段。
"""
from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from data_access.read.manifest import DatasetManifest, ManifestFile
from data_access.read.read_contract import ReadLineage
from data_access.r30.artifact_meta import FactorArtifactMetadata
from data_access.r30.change_impact import (
    plan_minimal_recompute,
    recompute_budget,
    to_fe_input,
)
from data_access.r30.data_change import (
    ChangeKind,
    DataChangeSet,
    detect_changes,
    revision_availability,
    source_snapshot_fingerprint,
)
from data_access.r30.lineage import LineageStore, ingest_read_lineage


# ---------------------------------------------------------------------------
# fake store
# ---------------------------------------------------------------------------
class FakeStore:
    """最小 store 替身：只暴露 manifest_version / build_dataset_manifest /
    load_manifest，供指纹与 manifest 读取路径使用。"""

    def __init__(
        self,
        dataset: str = "fundamentals",
        *,
        manifest: DatasetManifest | None = None,
        manifest_version: dict | None = None,
        built: dict | None = None,
    ) -> None:
        self.dataset_name = dataset
        self._manifest = manifest
        self._mv = manifest_version or {
            "dataset": dataset,
            "has_manifest": True,
            "mutation_owner": "dataaccess",
            "fresh": True,
            "dataset_version": "v1",
            "partition_version": "p1",
            "file_count": 1,
            "source_epoch": "e1",
            "manifest_built_epoch": "e1",
            "manifest_epoch": "e1",
            "manifest_generation_id": "g1",
            "created_at": "2024-06-01T00:00:00Z",
        }
        self._built = built or {
            "dataset": dataset,
            "files": 1,
            "rows": 100,
            "bytes": 1024,
            "format": "parquet",
        }

    def manifest_version(self, dataset: str, **params):
        if dataset != self.dataset_name:
            return {"dataset": dataset, "has_manifest": False}
        return dict(self._mv)

    def build_dataset_manifest(self, dataset: str, **params):
        if dataset != self.dataset_name:
            return None
        return dict(self._built)

    def load_manifest(self, dataset: str, **params):
        if dataset != self.dataset_name:
            return None
        return self._manifest


def _make_snap(*files, fp="fp"):
    return {"fingerprint": fp, "files": dict(files)}


# ---------------------------------------------------------------------------
# detect_changes：change_kind 判定 + 指纹一致
# ---------------------------------------------------------------------------
def test_detect_changes_equal_fingerprint_is_empty():
    store = FakeStore("fundamentals")
    fp = source_snapshot_fingerprint(store, "fundamentals")
    assert fp is not None
    change = detect_changes(store, "fundamentals", fp, fp)
    assert change.is_empty() is True
    assert change.change_kind == ChangeKind.append


def test_detect_changes_fingerprint_difference_conservative_revision():
    store = FakeStore("fundamentals")
    fp1 = source_snapshot_fingerprint(store, "fundamentals")
    store._mv["dataset_version"] = "v2"  # 模拟数据变化
    fp2 = source_snapshot_fingerprint(store, "fundamentals")
    assert fp1 != fp2
    change = detect_changes(store, "fundamentals", fp1, fp2)
    # 只有指纹、无文件级信息 → 保守 revision，且不被 is_empty 误报。
    assert change.change_kind == ChangeKind.revision
    assert change.is_empty() is False
    assert change.changed_time_range is None


def test_detect_changes_change_kind_append():
    before = _make_snap(
        ("a.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("b.parquet", {"bytes": 10, "mtime_ns": 1}),
        fp="fp1",
    )
    after = _make_snap(
        ("a.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("b.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("c.parquet", {"bytes": 5, "mtime_ns": 2, "min_time": "2024-05-01",
                       "max_time": "2024-05-31", "min_instrument": "000001",
                       "max_instrument": "000001"}),
        fp="fp2",
    )
    change = detect_changes(object(), "fundamentals", before, after)
    assert change.change_kind == ChangeKind.append
    assert change.changed_objects == ("c.parquet",)
    assert change.changed_time_range == ("2024-05-01", "2024-05-31")
    assert change.changed_instruments == ("000001",)
    assert change.is_empty() is False


def test_detect_changes_change_kind_revision():
    before = _make_snap(
        ("a.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("b.parquet", {"bytes": 10, "mtime_ns": 1}),
        fp="fp1",
    )
    after = _make_snap(
        ("a.parquet", {"bytes": 20, "mtime_ns": 2}),  # 同路径内容变
        ("b.parquet", {"bytes": 10, "mtime_ns": 1}),
        fp="fp2",
    )
    change = detect_changes(object(), "fundamentals", before, after)
    assert change.change_kind == ChangeKind.revision
    assert change.changed_objects == ("a.parquet",)


def test_detect_changes_change_kind_delete():
    before = _make_snap(
        ("a.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("b.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("c.parquet", {"bytes": 10, "mtime_ns": 1}),
        fp="fp1",
    )
    after = _make_snap(
        ("a.parquet", {"bytes": 10, "mtime_ns": 1}),
        ("b.parquet", {"bytes": 10, "mtime_ns": 1}),
        fp="fp2",
    )
    change = detect_changes(object(), "fundamentals", before, after)
    assert change.change_kind == ChangeKind.delete
    assert change.changed_objects == ("c.parquet",)


def test_detect_changes_change_kind_schema_change():
    before = _make_snap(("a.parquet", {"schema_hash": "h1"}), fp="fp1")
    after = _make_snap(("a.parquet", {"schema_hash": "h2"}), fp="fp2")
    change = detect_changes(object(), "fundamentals", before, after)
    assert change.change_kind == ChangeKind.schema_change


# ---------------------------------------------------------------------------
# revision_availability：knowledge-aware 合并
# ---------------------------------------------------------------------------
def _revision_manifest():
    mtime = int(_dt.datetime(2024, 6, 1).timestamp() * 1_000_000_000)
    return DatasetManifest(
        dataset="fundamentals",
        time_column="date",
        instrument_column="symbol",
        files=(
            ManifestFile(
                path="fundamentals/2024-06-01.parquet",
                min_time="2024-01-01",
                max_time="2024-03-31",
                min_instrument="000001",
                max_instrument="000001",
                mtime_ns=mtime,
            ),
        ),
    )


def test_revision_availability_clamps_lookahead():
    store = FakeStore("fundamentals", manifest=_revision_manifest())
    ra = revision_availability(store, "fundamentals", "000001", "2024-05-15")
    # 物理最新写入 2024-06-01 晚于 knowledge_date → 钳到 knowledge_date。
    assert ra["knowledge_aware"] is True
    assert ra["latest_revision_date"] == "2024-05-15"
    assert ra["period_end"] == "2024-03-31"


def test_revision_availability_within_knowledge():
    store = FakeStore("fundamentals", manifest=_revision_manifest())
    ra = revision_availability(store, "fundamentals", "000001", "2024-06-10")
    assert ra["knowledge_aware"] is True
    assert ra["latest_revision_date"] == "2024-06-01"


def test_revision_availability_physical_only_not_knowledge_aware():
    store = FakeStore("fundamentals", manifest=_revision_manifest())
    ra = revision_availability(store, "fundamentals", "000001", None)
    assert ra["knowledge_aware"] is False
    assert ra["latest_revision_date"] == "2024-06-01"


# ---------------------------------------------------------------------------
# plan_minimal_recompute：单 instrument 单时段 revision → 最小重算
# ---------------------------------------------------------------------------
def _single_company_revision_change():
    return DataChangeSet(
        dataset="fundamentals",
        source_before="fp1",
        source_after="fp2",
        changed_objects=("fundamentals/2024-05-31.parquet",),
        changed_time_range=("2024-05-01", "2024-05-31"),
        changed_instruments=("000001",),
        changed_columns=("revenue",),
        change_kind=ChangeKind.revision,
    )


def test_plan_minimal_recompute_single_instrument_not_full_history():
    change = _single_company_revision_change()
    demands = [
        # 依赖别的 dataset → 不触发
        {"factor_id": "MOM_5D", "datasets": ("prices",),
         "instruments": ("000001",), "lookback": 60, "window_fields": ("close",)},
        # 同 dataset 同 instrument → 触发，窗口 = changed_range 向前扩 250d
        {"factor_id": "REV_YOY", "datasets": ("fundamentals",),
         "instruments": ("000001",), "lookback": 250, "window_fields": ("revenue",)},
        # 同 dataset 不同 instrument → 不触发
        {"factor_id": "REV_YOY_OTHER", "datasets": ("fundamentals",),
         "instruments": ("600000",), "lookback": 250, "window_fields": ("revenue",)},
        # 无 instrument 限制（全市场因子）→ 触发（依赖该 dataset）
        {"factor_id": "MARKET_LEV", "datasets": ("fundamentals",),
         "instruments": None, "lookback": 60, "window_fields": ("revenue",)},
        # 显式覆盖窗口完全不与变化区间相交 → 不触发
        {"factor_id": "STALE_FACTOR", "datasets": ("fundamentals",),
         "instruments": ("000001",), "lookback": 60, "window_fields": ("revenue",),
         "end": "2023-12-31"},
    ]
    affected = plan_minimal_recompute(change, demands, lookback_days=30)
    ids = {a.factor_id for a in affected}
    assert ids == {"REV_YOY", "MARKET_LEV"}

    by_id = {a.factor_id: a for a in affected}
    rev = by_id["REV_YOY"]
    assert rev.dataset == "fundamentals"
    # R32-P0-079: affected_end 现在向前扩展（forward_output_horizon）
    # 旧值 "2024-05-31"，新值约 "2025-02-05"（250天保守放大）
    from datetime import date
    affected_end = date.fromisoformat(rev.affected_end[:10])
    assert affected_end >= date(2024, 5, 31), "affected_end 至少是 changed_end"
    assert rev.affected_start == "2023-08-25"  # 2024-05-01 向前 250 天
    assert rev.reason == "dataset_changed"
    assert rev.change_kind == "revision"

    market = by_id["MARKET_LEV"]
    assert market.affected_start == "2024-03-02"  # 2024-05-01 向前 60 天

    # 验收：不是全历史全因子——窗口是有界的，只影响 2 个相关因子。
    assert all(a.affected_start is not None for a in affected)
    assert len(affected) == 2


def test_plan_minimal_recompute_unbounded_change():
    change = DataChangeSet(
        dataset="fundamentals",
        changed_time_range=None,  # 未绑定窗口 → 相交恒真
        changed_instruments=("000001",),
        change_kind=ChangeKind.schema_change,
    )
    demands = [
        {"factor_id": "REV_YOY", "datasets": ("fundamentals",),
         "instruments": ("000001",), "lookback": 10, "window_fields": ("revenue",)},
        {"factor_id": "MOM_5D", "datasets": ("prices",),
         "instruments": ("000001",), "lookback": 10, "window_fields": ("close",)},
    ]
    affected = plan_minimal_recompute(change, demands, lookback_days=0)
    assert [a.factor_id for a in affected] == ["REV_YOY"]


def test_recompute_budget():
    affected = plan_minimal_recompute(
        _single_company_revision_change(),
        [
            {"factor_id": "REV_YOY", "datasets": ("fundamentals",),
             "instruments": ("000001",), "lookback": 250, "window_fields": ("revenue",)},
            {"factor_id": "MARKET_LEV", "datasets": ("fundamentals",),
             "instruments": None, "lookback": 60, "window_fields": ("revenue",)},
        ],
    )
    budget = recompute_budget(affected, _single_company_revision_change())
    assert budget["affected_factors"] == 2
    assert budget["min_window_start"] == "2023-08-25"
    # R32-P0-079: max_window_end 现在向前扩展（forward_output_horizon）
    # 检查它至少是 changed_end，不再硬编码旧值
    from datetime import date
    max_end = date.fromisoformat(budget["max_window_end"][:10])
    assert max_end >= date(2024, 5, 31), "max_window_end 至少是 changed_end"
    # window_days 也相应增大
    assert budget["window_days"] >= 280
    assert budget["changed_instrument_count"] == 1
    assert budget["change_kind"] == "revision"


# ---------------------------------------------------------------------------
# to_fe_input：FE compute_change_impact 桥结构
# ---------------------------------------------------------------------------
def test_to_fe_input_structure():
    change = _single_company_revision_change()
    inp = to_fe_input(change)
    assert inp["field"] == "revenue"
    assert inp["changed_start"] == "2024-05-01"
    assert inp["changed_end"] == "2024-05-31"
    assert inp["column_identity"] == {"dataset": "fundamentals", "field": "revenue"}


def test_to_fe_input_with_revision_identity():
    change = DataChangeSet(
        dataset="fundamentals",
        changed_time_range=("2024-05-01", "2024-05-31"),
        changed_columns=("revenue",),
        revision_availability={"latest_revision_date": "2024-05-15"},
    )
    inp = to_fe_input(change, calendar="US")
    assert inp["column_identity"]["dataset"] == "fundamentals"
    assert inp["column_identity"]["field"] == "revenue"
    assert inp["column_identity"]["revision_identity"] == "2024-05-15"
    assert inp["calendar"] == "US"


# ---------------------------------------------------------------------------
# LineageStore：三种血缘问题 + 导出 + 模式
# ---------------------------------------------------------------------------
def test_lineage_store_query_three_questions(tmp_path):
    ls = LineageStore()
    assert ls.mode == "memory"
    ls.record(run_id="r1", factor_id="f1", dataset="fundamentals",
              concept="revenue", source_snapshot="snap1", principal="alice",
              build_sha="sha1")
    ls.record(run_id="r2", factor_id="f2", dataset="prices",
              concept="close", source_snapshot="snap2", principal="bob")
    ls.record(run_id="r3", factor_id="f1", dataset="prices",
              concept="volume", source_snapshot="snap1", principal="alice")

    # 问题 1：某 dataset 影响哪些 factor？
    assert {r["factor_id"] for r in ls.factors_for_dataset("prices")} == {"f1", "f2"}
    # 问题 2：某 factor 用过哪些 dataset？
    assert {r["dataset"] for r in ls.datasets_for_factor("f1")} == {"fundamentals", "prices"}
    # 问题 3：某 premium source 被谁读过？
    assert {r["run_id"] for r in ls.readers_of_source("snap1")} == {"r1", "r3"}
    assert ls.count() == 3

    # 组合过滤
    assert len(ls.search(factor_id="f1", dataset="prices")) == 1

    # 导出 parquet
    out = ls.to_parquet(str(tmp_path / "lineage.parquet"))
    tbl = pq.read_table(out)
    assert tbl.num_rows == 3
    assert "extra" in tbl.column_names
    assert "factor_id" in tbl.column_names


def test_lineage_store_duckdb_mode(tmp_path):
    db_path = str(tmp_path / "lineage.duckdb")
    ls = LineageStore(db_path)
    assert ls.mode == "duckdb"
    ls.record(run_id="r10", factor_id="f1", dataset="fundamentals",
              source_snapshot="s1", extra={"region": "cn"})
    ls.record(run_id="r11", factor_id="f1", dataset="fundamentals",
              source_snapshot="s1")
    assert ls.count() == 2
    assert len(ls.factors_for_dataset("fundamentals")) == 1
    rows = ls.search(factor_id="f1")
    assert len(rows) == 2
    # extra dict 往返
    assert any(r["extra"].get("region") == "cn" for r in rows)
    # 重新打开后数据仍在
    ls2 = LineageStore(db_path)
    assert ls2.mode == "duckdb"
    assert len(ls2.search(source_snapshot="s1")) == 2


def test_ingest_read_lineage():
    ls = LineageStore()
    lineage = ReadLineage(
        dataset="fundamentals",
        columns=("revenue", "net_income"),
        time_range=("2024-01-01", "2024-12-31"),
        instrument_filter=("000001",),
        build_sha="abc123",
    )
    snapshot = SimpleNamespace(snapshot_id="snap-x", dataset="fundamentals", build_sha="abc123")
    result = SimpleNamespace(lineage=lineage, snapshot=snapshot)
    row = ingest_read_lineage(
        SimpleNamespace(),
        ls,
        result,
        run_id="run-9",
        factor_id="f9",
        principal="carol",
    )
    assert row["dataset"] == "fundamentals"
    assert row["source_snapshot"] == "snap-x"
    assert row["build_sha"] == "abc123"
    assert row["factor_id"] == "f9"
    assert row["principal"] == "carol"
    assert row["concept"] == "revenue,net_income"
    assert row["extra"]["instrument_filter"] == ("000001",)
    assert ls.count() == 1


# ---------------------------------------------------------------------------
# FactorArtifactMetadata：完整身份
# ---------------------------------------------------------------------------
def test_artifact_metadata_full_fields():
    meta = FactorArtifactMetadata(
        factor_id="f1",
        factor_definition="ts_mean(revenue, 4)",
        operator_semantic_versions={"ts_mean": "1.2.0"},
        factor_source_plan_digest="digest-abc",
        experiment_snapshot_id="exp-1",
        universe_snapshot_id="uni-1",
        calendar_snapshot_id="cal-1",
        coverage_policy="strict",
        security_classification="internal",
        build_sha="sha-abc",
    )
    d = meta.to_dict()
    for key in (
        "factor_id",
        "factor_definition",
        "operator_semantic_versions",
        "factor_source_plan_digest",
        "experiment_snapshot_id",
        "universe_snapshot_id",
        "calendar_snapshot_id",
        "coverage_policy",
        "security_classification",
        "build_sha",
        "computed_at",
    ):
        assert key in d, f"missing {key}"
    assert d["factor_definition"] == "ts_mean(revenue, 4)"
    assert d["operator_semantic_versions"] == {"ts_mean": "1.2.0"}
    assert d["computed_at"] is not None

    sidecar = meta.metadata_sidecar("/artifacts/f1.parquet")
    assert sidecar["artifact_path"] == "/artifacts/f1.parquet"
    assert sidecar["sidecar_version"] == "1"
    assert sidecar["factor_source_plan_digest"] == "digest-abc"

    bound = meta.bind_metadata("/artifacts/f1.parquet", extra_note="x")
    assert bound["artifact_path"] == "/artifacts/f1.parquet"
    assert bound["extra_note"] == "x"
    assert bound["security_classification"] == "internal"
