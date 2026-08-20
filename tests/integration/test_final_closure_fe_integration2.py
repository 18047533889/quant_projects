# -*- coding: utf-8
"""#收官轮（外部 AI 复查）DA+FE 联合链路 adversarial 验收测试。

覆盖（Freeze 前最后一批 integration root issues + incomplete-fix bypass）：

    P0-01  production SourceRef child 继承父级 execution policy（run_mode /
            production / strict / mining / PIT）。
    P0-02  pit_enforce=True 真正挂到 source preflight（_ensure_field_plans 自动
            跑四层 PIT；UNKNOWN 层 production fail-closed）。
    P0-03  CompositeDataSource 对 PIT 敏感源（event/pit / 无知识时钟财务 / E2
            strict-pit）禁止自行 merge_asof——production fail-closed。
    P0-04  instrument_filter None / [] / ["AAPL"] → 三个不同的 cache scope；
            instrument_filter="AAPL"（裸 str）构造期拒绝。
    P0-05  Composite cache 子源 A→B：refresh_snapshot 协议驱动失效，下次缓存读
            返回 B（不再永远旧数据）。
    P0-06  materialize write_target 拼写错误 → 在任何 catalog/file/watermark
            side effect 之前拒绝。
    P0-07  staging 成功 + publish 失败 → 权威水位线 unchanged；publish 成功 →
            水位线推进（exactly once）。
    P0-08  production 任何 public entrypoint 的 direct-local 写 → 拒绝。
    P0-09  native Polars long 路径走 governed ``store.scan()``（不再 scan_polars
            裸 LazyFrame 旁路）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from data_access.core.exceptions import ValidationError as DAValidationError

from storage.data_scope import compute_data_scope
from storage.materializer import ParquetMaterializer
from storage.materialize.lake_publish import publish_factor_lake
from storage.sources.composite_source import CompositeDataSource
from storage.sources.data_access_source import (
    DataAccessSource,
    FourLayerPITError,
)
from storage.sources.field_plan import NormalizedFieldPlan
from storage.sources.lqtp_logical_source import LQTPLogicalDataSource

pytest.importorskip("pandas")


def _series(rows):
    frame = pd.DataFrame(
        [(pd.Timestamp(d), a, v) for d, a, v in rows],
        columns=["timestamp", "instrument", "value"],
    )
    return frame.set_index(["timestamp", "instrument"])["value"].sort_index()


# ---------------------------------------------------------------------------
# P0-01 production SourceRef child 继承父级 execution policy
# ---------------------------------------------------------------------------


class _PolicyInner:
    """模拟 production DataAccessSource 内层（携带 resolved execution policy）。

    LQTPLogicalDataSource._child 只 ``getattr`` 内层属性，无需 DataSource 抽象基类。
    """

    start_date = None
    end_date = None
    instrument_filter = None
    run_mode = "production"
    production = True
    strict_unknown_fields = True
    enforce_mining_gate = True
    snapshot_now_only = False
    pit_enforce = True
    mining_coverage_threshold = None


class _ResearchInner:
    start_date = None
    end_date = None
    instrument_filter = None
    run_mode = "research"
    production = False
    strict_unknown_fields = False
    enforce_mining_gate = False
    snapshot_now_only = False
    pit_enforce = False
    mining_coverage_threshold = None


def test_source_ref_child_inherits_production_policy():
    """二级 SourceRef child 必须继承父级 policy，不能退回 research/fail-open。"""
    parent = LQTPLogicalDataSource(_PolicyInner())
    child = parent._child("ashare_turnover_base_daily")
    assert child.run_mode == "production"
    assert child.production is True
    assert child.strict_unknown_fields is True
    assert child.enforce_mining_gate is True
    assert child.pit_enforce is True


def test_source_ref_child_research_stays_research():
    parent = LQTPLogicalDataSource(_ResearchInner())
    child = parent._child("ashare_turnover_base_daily")
    assert child.production is False
    assert child.strict_unknown_fields is False
    assert child.pit_enforce is False


# ---------------------------------------------------------------------------
# P0-02 pit_enforce 挂到 source preflight：_ensure_field_plans 自动跑四层 PIT
# ---------------------------------------------------------------------------


def test_pit_enforce_auto_runs_four_layer_pit_production(tmp_path):
    """pit_enforce=True + production：UNKNOWN 层在 _ensure_field_plans 自动拒绝。"""
    src = DataAccessSource(
        dataset="fundamentals_balance_sheet",
        production=True,
        pit_enforce=True,
    )
    plan = NormalizedFieldPlan(logical_concept="total_assets")
    version = "pit-test-version"
    src._field_plans[("total_assets", version)] = plan
    with patch.object(src, "_semantic_catalog_version", return_value=version):
        with pytest.raises(FourLayerPITError, match="UNKNOWN"):
            src._ensure_field_plans(["total_assets"])


def test_pit_enforce_research_warns_not_raises(tmp_path):
    """pit_enforce=True + research：UNKNOWN 层告警降级，不抛（research fail-open）。"""
    src = DataAccessSource(
        dataset="fundamentals_balance_sheet",
        pit_enforce=True,
    )
    plan = NormalizedFieldPlan(logical_concept="total_assets")
    version = "pit-test-version"
    src._field_plans[("total_assets", version)] = plan
    with patch.object(src, "_semantic_catalog_version", return_value=version):
        src._ensure_field_plans(["total_assets"])  # no raise


def test_no_pit_enforce_skips_four_layer_gate(tmp_path):
    """pit_enforce=False（默认）：四层 PIT 不自动跑（opt-in，与 mining/coverage 一致）。"""
    src = DataAccessSource(
        dataset="fundamentals_balance_sheet",
        production=True,
        pit_enforce=False,
    )
    plan = NormalizedFieldPlan(logical_concept="total_assets")
    version = "pit-test-version"
    src._field_plans[("total_assets", version)] = plan
    with patch.object(src, "_semantic_catalog_version", return_value=version):
        src._ensure_field_plans(["total_assets"])  # no raise


# ---------------------------------------------------------------------------
# P0-03 Composite 对 PIT 敏感源禁止自行 merge_asof（authority boundary）
# ---------------------------------------------------------------------------


def _anchor_source():
    return {
        "close": _series(
            [("2024-01-02", "AAA", 10.0), ("2024-01-03", "AAA", 12.0)]
        ),
    }


def test_composite_pit_source_asof_forbidden_in_production():
    """production 下 PIT 敏感源（无知识时钟财务）禁止 Composite 非 exact 对齐。"""
    child = DataAccessSource(
        dataset="fundamentals_balance_sheet",
        production=True,
        fields={"total_assets": "total_assets"},
    )
    anchor = DataAccessSource(dataset="ashare_stock_daily", production=True)
    composite = CompositeDataSource(
        anchor_source="anchor",
        anchor_column="close",
        sources={"anchor": anchor, "fundamental": child},
        joins={"fundamental": "asof_backward"},
    )
    with pytest.raises(ValueError, match="PIT 敏感"):
        composite._enforce_join_authority(
            source_name="fundamental",
            join_spec=composite.joins["fundamental"],
        )


def test_composite_pit_source_exact_allowed():
    """exact 对齐是安全操作：PIT 敏感源已由 child 对齐好，Composite 只 reindex。"""
    child = DataAccessSource(
        dataset="fundamentals_balance_sheet",
        production=True,
        fields={"total_assets": "total_assets"},
    )
    anchor = DataAccessSource(dataset="ashare_stock_daily", production=True)
    composite = CompositeDataSource(
        anchor_source="anchor",
        anchor_column="close",
        sources={"anchor": anchor, "fundamental": child},
        joins={"fundamental": "exact"},
    )
    composite._enforce_join_authority(
        source_name="fundamental",
        join_spec=composite.joins["fundamental"],
    )  # no raise


def test_composite_pit_source_asof_research_warns():
    """research 下 PIT 敏感源 asof：告警放行（不抛）。"""
    child = DataAccessSource(
        dataset="fundamentals_balance_sheet",
        production=False,
        fields={"total_assets": "total_assets"},
    )
    anchor = DataAccessSource(dataset="ashare_stock_daily", production=False)
    composite = CompositeDataSource(
        anchor_source="anchor",
        anchor_column="close",
        sources={"anchor": anchor, "fundamental": child},
        joins={"fundamental": "asof_backward"},
    )
    composite._enforce_join_authority(
        source_name="fundamental",
        join_spec=composite.joins["fundamental"],
    )  # warning only


# ---------------------------------------------------------------------------
# P0-04 instrument_filter None / [] / ["AAPL"] scope + 裸 str 拒绝
# ---------------------------------------------------------------------------


def test_instrument_filter_three_distinct_scopes():
    a = DataAccessSource(dataset="ds", instrument_filter=None)
    b = DataAccessSource(dataset="ds", instrument_filter=[])
    c = DataAccessSource(dataset="ds", instrument_filter=["AAPL"])
    scopes = {compute_data_scope(a), compute_data_scope(b), compute_data_scope(c)}
    assert len(scopes) == 3, (
        "None（全市场）/ []（空股票池）/ LIST 必须得到不同的 cache scope，"
        "否则空 universe 会命中全市场缓存"
    )


_BAD_FILTER_EXC = (DAValidationError, ValueError, TypeError)


def test_instrument_filter_bare_str_rejected():
    with pytest.raises(_BAD_FILTER_EXC, match="instrument_filter"):
        DataAccessSource(dataset="ds", instrument_filter="AAPL")


def test_instrument_filter_rejects_dict_and_generator():
    with pytest.raises(_BAD_FILTER_EXC, match="instrument_filter"):
        DataAccessSource(dataset="ds", instrument_filter={"AAPL": 1})
    with pytest.raises(_BAD_FILTER_EXC, match="instrument_filter"):
        DataAccessSource(dataset="ds", instrument_filter=(x for x in ["AAPL"]))


def test_instrument_filter_rejects_non_string_element():
    with pytest.raises(_BAD_FILTER_EXC, match="instrument_filter"):
        DataAccessSource(dataset="ds", instrument_filter=["AAPL", 123])


def test_instrument_filter_accepts_list_tuple_set():
    src = DataAccessSource(dataset="ds", instrument_filter=("AAPL", "MSFT"))
    assert src.instrument_filter == ["AAPL", "MSFT"]
    src2 = DataAccessSource(dataset="ds", instrument_filter={"MSFT", "AAPL"})
    assert set(src2.instrument_filter) == {"AAPL", "MSFT"}
    # [] 保持 []（空股票池），不折叠成 None
    src3 = DataAccessSource(dataset="ds", instrument_filter=[])
    assert src3.instrument_filter == []


# ---------------------------------------------------------------------------
# P0-05 Composite cache 子源 A→B：refresh 协议驱动失效
# ---------------------------------------------------------------------------


class _RefreshSource:
    """子源模拟 DataAccessSource refresh 协议：

    ``_current`` 是底层真实 token；``_reported`` 是 data_snapshot_id/snapshot_token
    报告的**陈旧**值——只有 ``refresh_snapshot()`` 被调用后才更新。这精确复现
    「底层 dataset A→B，但 child.data_snapshot_id 仍报 A」的 cache-of-cache 场景。
    """

    def __init__(self, data: dict, token: str | None = None):
        self.data = data
        self._current = token
        self._reported = token
        self.load_calls: dict[str, int] = {}
        self.refresh_calls = 0

    @property
    def data_snapshot_id(self):
        return self._reported

    @property
    def snapshot_token(self):
        return self._reported

    def refresh_snapshot(self):
        self.refresh_calls += 1
        self._reported = self._current  # refresh 发现真实 token
        return self._reported

    def set_underlying(self, data: dict, token: str):
        self.data = data
        self._current = token  # 底层变化；_reported 保持陈旧直到 refresh

    def load_column(self, name: str):
        self.load_calls[name] = self.load_calls.get(name, 0) + 1
        return self.data[name]


def test_composite_cache_refreshes_and_invalidates_on_child_change():
    price = _RefreshSource(
        {"close": _series([("2024-01-02", "AAA", 10.0), ("2024-01-03", "AAA", 12.0)])},
        token="A",
    )
    fund = _RefreshSource(
        {"pe": _series([("2024-01-01", "AAA", 2.0), ("2024-01-03", "AAA", 3.0)])},
        token="A",
    )
    composite = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fund": fund},
        joins={"fund": "asof_backward"},
    )
    first = composite.load_column("fund.pe")
    assert first.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(3.0)
    assert fund.load_calls["pe"] == 1

    # 底层 A→B；child._reported 仍是 A（未 refresh 前 composite 只会看到 A）
    fund.set_underlying(
        {"pe": _series([("2024-01-01", "AAA", 2.0), ("2024-01-03", "AAA", 9.0)])},
        token="B",
    )
    assert fund.snapshot_token == "A"  # 未 refresh：陈旧
    second = composite.load_column("fund.pe")
    # Composite 先 refresh 子源 → 发现 B → 清自己的缓存 → 重新读子源
    assert fund.load_calls["pe"] == 2, "Composite 缓存未在子源 A→B 后失效"
    assert second.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(9.0)


def test_composite_cache_hit_when_child_unchanged():
    price = _RefreshSource(
        {"close": _series([("2024-01-02", "AAA", 10.0)])},
        token="A",
    )
    fund = _RefreshSource(
        {"pe": _series([("2024-01-01", "AAA", 2.0)])},
        token="A",
    )
    composite = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fund": fund},
        joins={"fund": "asof_backward"},
    )
    composite.load_column("fund.pe")
    assert fund.load_calls["pe"] == 1
    composite.load_column("fund.pe")
    assert fund.load_calls["pe"] == 1  # 无变化 → 命中缓存


# ---------------------------------------------------------------------------
# P0-06 write_target 拼写错误 → 任何 side effect 之前拒绝
# ---------------------------------------------------------------------------


def test_write_target_typo_rejected_before_side_effects(tmp_path):
    mat = ParquetMaterializer(lake_root=tmp_path)
    series = _series([("2024-01-15", "A", 1.0)])
    with pytest.raises(ValueError, match="locla"):
        mat.materialize(
            factor_id="typo_target",
            result=series,
            ast_hash="h1",
            write_target="locla",
        )
    # 未注册因子、无本地文件、无水位线——metadata 绝不能「说已提交」
    assert mat.catalog.get_factor_info("typo_target") is None
    assert mat.catalog.get_watermark("typo_target") is None
    assert not (tmp_path / "factors" / "typo_target").exists()


def test_write_target_staging_custom_dataset_still_valid(tmp_path, monkeypatch):
    mat = ParquetMaterializer(lake_root=tmp_path)
    series = _series([("2024-01-15", "A", 1.0)])

    class _FakeStore:
        def upsert(self, dataset, table, **kwargs):
            return {"rows_upserted": table.num_rows}

    monkeypatch.setattr("data_access.get_store", lambda: _FakeStore())
    out = mat.materialize(
        factor_id="staging_custom",
        result=series,
        ast_hash="h1",
        write_target="staging:my_staging_set",
    )
    assert out["write_target"] == "staging:my_staging_set"
    assert out["staging"]["dataset"] == "my_staging_set"


# ---------------------------------------------------------------------------
# P0-07 staging+publish 水位线状态机
# ---------------------------------------------------------------------------


def _published_lake(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir(parents=True, exist_ok=True)
    pdf = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16")],
            "asset": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )
    pdf.to_parquet(pub / "part.parquet", index=False)
    return pub


class _PublishStore:
    """fake DA store：resolve_dataset_path + 可选失败的 publish_from_staging。"""

    def __init__(self, published_dir, *, fail_publish=False):
        self._dir = published_dir
        self._fail = fail_publish

    def resolve_dataset_path(self, dataset, factor_id=None):
        return self._dir

    def publish_from_staging(self, *args, **kwargs):
        if self._fail:
            raise RuntimeError("publish failed (downstream)")
        return {"rows": 1}


def _register_factor(tmp_path, factor_id):
    mat = ParquetMaterializer(lake_root=tmp_path)
    mat.catalog.register(
        factor_id=factor_id,
        author="t",
        frequency="1d",
        ast_hash="h1",
    )
    return mat


def test_publish_failure_keeps_watermark_unchanged(tmp_path):
    mat = _register_factor(tmp_path, "wm_fail")
    pub = _published_lake(tmp_path)
    fake = _PublishStore(pub, fail_publish=True)
    with patch("data_access.get_store", return_value=fake):
        with pytest.raises(RuntimeError, match="publish failed"):
            publish_factor_lake(
                factor_id="wm_fail",
                lake_root=tmp_path,
                approve=True,
                sync_from_local=False,
                reconcile=False,
            )
    # staging 成功但 publish 失败 → 权威水位线 unchanged（None）
    assert mat.catalog.get_watermark("wm_fail") is None


def test_publish_success_advances_watermark_exactly_once(tmp_path):
    mat = _register_factor(tmp_path, "wm_ok")
    pub = _published_lake(tmp_path)
    fake = _PublishStore(pub, fail_publish=False)
    with patch("data_access.get_store", return_value=fake):
        result = publish_factor_lake(
            factor_id="wm_ok",
            lake_root=tmp_path,
            approve=True,
            sync_from_local=False,
            reconcile=False,
        )
        wm = result["watermark"]
        assert wm is not None
        assert wm["start_date"] <= "2024-01-15" <= wm["end_date"]
        assert wm["row_count"] == 2
        # 再 publish 一次：幂等，不重复推进（start/end/row_count 一致，仅
        # last_updated 更新）
        result2 = publish_factor_lake(
            factor_id="wm_ok",
            lake_root=tmp_path,
            approve=True,
            sync_from_local=False,
            reconcile=False,
        )
        wm2 = result2["watermark"]
        assert wm2 is not None
        for key in ("start_date", "end_date", "row_count"):
            assert wm2[key] == wm[key], f"watermark {key} 被重复推进：{wm} → {wm2}"


# ---------------------------------------------------------------------------
# P0-08 production 任何 public entrypoint 的 direct-local 写拒绝
# ---------------------------------------------------------------------------


def test_production_materialize_local_and_both_rejected(tmp_path):
    mat = ParquetMaterializer(lake_root=tmp_path)
    series = _series([("2024-01-15", "A", 1.0)])
    for target in ("local", "both"):
        with pytest.raises(ValueError, match="production 禁止 direct-local"):
            mat.materialize(
                factor_id=f"prod_{target}",
                result=series,
                ast_hash="h1",
                production=True,
                write_target=target,
            )
        assert mat.catalog.get_factor_info(f"prod_{target}") is None


def test_production_local_write_target_rejects(tmp_path, monkeypatch):
    """LocalParquetWriteTarget 的 production guard 也生效（不因 materialize 主路径
    存在而放松）。"""
    from storage.materialize.write_targets import LocalParquetWriteTarget

    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    target = LocalParquetWriteTarget(lake_root=tmp_path)
    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-15")],
            "asset": ["A"],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="production 禁止 direct-local"):
        target.write_factor_frame("prod_local_target", frame)
    monkeypatch.delenv("FACTOR_ENGINE_RUN_MODE", raising=False)


# ---------------------------------------------------------------------------
# P0-09 native Polars long 走 governed store.scan()（不再 scan_polars 旁路）
# ---------------------------------------------------------------------------


def _spy_store(tmp_path):
    from data_access.core.engine import DuckDBEngine
    from data_access.registry import load_registry

    root = tmp_path / "d"
    root.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist(
            [
                {"d": pd.Timestamp("2024-01-02").date(), "s": "AAA", "v": 1.0},
                {"d": pd.Timestamp("2024-01-03").date(), "s": "AAA", "v": 2.0},
            ]
        ),
        str(root / "part.parquet"),
    )
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        f"""
ds:
  kind: static
  access_mode: published
  layout: plain
  root: {root}
  glob: "*.parquet"
  time_column: d
  instrument_column: s
  schema:
    d: date
    s: string
    v: double
""",
        encoding="utf-8",
    )
    real = __import__("data_access.store", fromlist=["DataAccessStore"]).DataAccessStore(
        registry=load_registry(cfg), engine=DuckDBEngine(threads=2)
    )

    class Spy:
        scan_calls = 0
        scan_polars_calls = 0

        def __getattr__(self, name):
            return getattr(real, name)

        def scan(self, dataset, **kwargs):
            self.scan_calls += 1
            return real.scan(dataset, **kwargs)

        def scan_polars(self, dataset, **kwargs):
            self.scan_polars_calls += 1
            return real.scan_polars(dataset, **kwargs)

    return Spy()


def test_polars_long_native_uses_governed_scan(tmp_path):
    """native-long helper 与 governed-lazy 同源：走 store.scan()（ScanHandle），
    不再优先 store.scan_polars() 拿裸 LazyFrame。"""
    from backend.polars_lazy import build_scan_polars_long

    store = _spy_store(tmp_path)
    lf = build_scan_polars_long(
        store,
        "ds",
        logical_columns=["v"],
        physical_columns=["v"],
        output_names={},
        time_column="d",
        instrument_column="s",
        time_range=("2024-01-02", "2024-01-03"),
        instrument_filter=None,
        params={},
        frequency="daily",
    )
    assert store.scan_calls >= 1, "polars-long 必须走 store.scan()（ScanHandle）"
    assert store.scan_polars_calls == 0, (
        "native-long 不应再走 store.scan_polars() 裸 LazyFrame 旁路"
    )
    assert lf.collect().height == 2
