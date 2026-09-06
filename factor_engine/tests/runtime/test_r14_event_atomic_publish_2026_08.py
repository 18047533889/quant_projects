# -*- coding: utf-8 -*-
"""R14 #5a：production DataEvent 两阶段发布（stage all → publish all）+ R14 复查
P0-1 方案 A feature gate。

外部 AI 复查 P0：逐 factor 物化时 factor1 成功、factor2 失败 → published 因子湖
出现「6 新 + 4 旧」的 mixed generation；且跨 factor 不整体 rollback。上一轮
「stage all → publish all」只解决了 stage 阶段——**publish 阶段仍是逐 factor**：
f1 成功、f2 失败 → f1 已可见、事件 rejected（mixed published state）。

R14 复查 P0-1（方案 A）修复：
  * **production DataEvent 自动发布默认关闭**（``DATA_EVENT_PRODUCTION_AUTO_PUBLISH``
    未设置）→ production 事件直接拒绝（``ProductionEventAutoPublishDisabled``），
    不落任何 staging/published → published 湖零 mixed（读者看不到任何新因子）；
  * production 一律**强制** ``write_target="staging"``——不尊重调用方覆盖
    （``staging_clickhouse``/``clickhouse`` 会在 stage-all 阶段产生 published/CH
    side effect → CH mixed）；
  * 显式 ``DATA_EVENT_PRODUCTION_AUTO_PUBLISH=1`` 启用两阶段（实验性）：
    任一 stage 失败 → publish 阶段不进入 → 零 mixed；全部 stage 成功 → 逐因子
    publish；**publish 阶段**失败仍是已知限制（已发布因子可见、事件 rejected，
    真正 visibility transaction 留待后续）；
  * 同 ``event_id`` 重试幂等收敛，最终只 commit 一次。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from factor_engine.runtime.dependency_catalog import DependencyCatalog, FactorDependencyEdge
from factor_engine.runtime.incremental_scheduler import (
    DataEvent,
    PartialIncrementalFailureError,
    ProductionEventAutoPublishDisabled,
    execute_incremental_updates_from_event,
)
from factor_engine.storage.materializer import ParquetMaterializer
from factor_engine.storage.trading_calendar import (
    TradingCalendar,
    clear_trading_calendar_cache,
    register_trading_calendar,
)


@pytest.fixture(autouse=True)
def _production_calendar():
    """Install a deterministic, explicitly sourced A-share calendar for R14."""
    clear_trading_calendar_cache()
    register_trading_calendar(
        "ashare",
        TradingCalendar(
            list(pd.bdate_range("2026-07-01", "2026-09-30")),
            anchor_policy="previous_trade_day",
            source="explicit_test_fixture",
            snapshot="r14-2026-08",
            version="1",
            timezone="Asia/Shanghai",
        ),
    )
    yield
    clear_trading_calendar_cache()


def _ser() -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-15"]), ["A"]], names=["timestamp", "instrument"]
    )
    return pd.Series([1.0], index=idx)


class _FakeStore:
    """fake DA store：记录 staged / published，可让指定 factor 的 publish 失败。"""

    def __init__(self, *, fail_publish_on: str | None = None):
        self.staged: list[str] = []
        self.published: list[str] = []
        self._fail_publish_on = fail_publish_on

    def upsert(self, dataset, table, upsert_on=None, partition_by=None,
               factor_id=None, **kw):
        self.staged.append(factor_id)
        return {"rows_upserted": table.num_rows}

    def resolve_dataset_path(self, dataset, factor_id=None):
        # 返回不存在的 Path：advance_published_watermark 对 published 目录调
        # ``.is_dir()``，不存在 → 水位线保持原样（None），publish 流程照常。
        return Path("/tmp/fake_published")

    def publish_from_staging(self, *args, **kwargs):
        fid = kwargs.get("factor_id")
        if self._fail_publish_on == fid:
            raise RuntimeError(f"publish failed for {fid}")
        self.published.append(fid)
        return {"rows": 1}


def _install_fake_publish(monkeypatch, store: _FakeStore) -> None:
    def _publish(*, factor_id, expected_staging_generation=None,
                 expected_manifest_digest=None, expected_run_id=None,
                 frequency=None, **kwargs):
        assert expected_staging_generation == f"gen-{factor_id}"
        assert expected_manifest_digest == f"digest-{factor_id}"
        assert expected_run_id == f"run-{factor_id}"
        assert frequency == "1d"
        store.publish_from_staging(factor_id=factor_id)
        return {
            "factor_id": factor_id,
            "approved": True,
            "generation_id": expected_staging_generation,
            "manifest_digest": expected_manifest_digest,
            "run_id": expected_run_id,
        }

    monkeypatch.setattr(
        "factor_engine.storage.materialize.lake_publish.publish_factor_lake", _publish
    )


def _setup_lake(tmp_path, *, fail_stage_on: str | None = None) -> tuple:
    """注册 2 个因子（f_a / f_b）依赖 ``shared_ds.close``；返回 (lake, control)。

    因子表达式用 ``field("close")``（production 强制 field 引用 + 可 lower），并
    注册**真实** IR hash——否则 production 下 ``_verify_factor_semantic_identity``
    的 ast_hash 校验会失败（R14 #4 后按 Expr→IR lower 计算）。
    """
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.storage.catalog import compute_ir_hash

    _ir = Analyzer(production=True, market="ashare").lower(
        parse_factor('field("close")').expr
    ).ir
    ast_hash = compute_ir_hash(_ir)
    lake = tmp_path / "lake"
    catalog = ParquetMaterializer(lake_root=lake).catalog
    dep = DependencyCatalog(catalog)
    control = {"fail_stage_on": fail_stage_on, "staged": []}
    for fid in ("f_a", "f_b"):
        catalog.register(
            fid,
            author="tester",
            frequency="1d",
            ast_hash=ast_hash,
            expression='field("close")',
            data_source_config={"type": "parquet", "dataset": "ds_x"},
        )
        dep.record_factor_edges(
            fid,
            edges=[FactorDependencyEdge(fid, "shared_ds", "close")],
            lookback=2,
            frequency="1d",
            source_dataset="shared_ds",
        )
        dep.record_full_factor_definition(
            fid,
            expression='field("close")',
            frequency="1d",
            data_source_config={"type": "parquet", "dataset": "ds_x"},
        )

    def fake_engine_factory(full_def, *, lake_root=None, market=None, run_mode=None):
        fid = full_def["factor_id"]
        eng = MagicMock()

        def _mi(factor, *, factor_id, since=None, end_date=None, expression=None,
                frequency=None, description=None, deleted_keys=None, **kw):
            control["staged"].append(factor_id)
            assert kw.get("write_target") == "staging", (
                "production 事件必须写 staging，收到 write_target="
                f"{kw.get('write_target')!r}"
            )
            assert kw.get("value_dtype") is None
            if control["fail_stage_on"] == factor_id:
                raise RuntimeError(f"stage failed for {factor_id}")
            return {
                "materialization": {
                    "factor_id": factor_id,
                    "rows_written": 1,
                    "staging": {
                        "dataset": "factor_lake_staging",
                        "identity": {
                            "generation_id": f"gen-{factor_id}",
                            "manifest_digest": f"digest-{factor_id}",
                            "run_id": f"run-{factor_id}",
                        },
                    },
                }
            }

        eng.materialize_incremental.side_effect = _mi
        return eng

    return lake, control, fake_engine_factory


def _prod_event() -> DataEvent:
    return DataEvent(
        dataset="shared_ds",
        column="close",
        updated_date="2026-08-09",
        field_id="close",
        revision_kind="update",
        event_id="ev_atomic",
    )


def _force_production(monkeypatch) -> None:
    monkeypatch.setattr(
        "factor_engine.runtime.production_policy.is_production_mode", lambda: True
    )


def _enable_auto_publish(monkeypatch) -> None:
    """显式开启 production DataEvent 自动发布（R14 复查 P0-1 方案 A opt-in）。"""
    monkeypatch.setenv("DATA_EVENT_PRODUCTION_AUTO_PUBLISH", "1")


def test_r14_production_event_stage_all_then_publish_all(tmp_path, monkeypatch):
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, control, factory = _setup_lake(tmp_path)
    store = _FakeStore()
    monkeypatch.setattr("data_access.get_store", lambda: store)
    _install_fake_publish(monkeypatch, store)

    out = execute_incremental_updates_from_event(
        None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
    )
    # 两阶段：全部 stage → 全部 publish → commit 一次
    assert sorted(control["staged"]) == ["f_a", "f_b"]
    assert sorted(store.published) == ["f_a", "f_b"]
    assert out["ledger_status"] == "committed"
    assert out["succeeded"] == 2
    assert out["published"] == ["f_a", "f_b"]


def test_r14_production_event_stage_failure_no_publish(tmp_path, monkeypatch):
    """factor2 stage 失败 → 直接 reject，publish 阶段不进入，published 湖零 mixed。"""
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, control, factory = _setup_lake(tmp_path, fail_stage_on="f_b")
    store = _FakeStore()
    monkeypatch.setattr("data_access.get_store", lambda: store)
    _install_fake_publish(monkeypatch, store)

    with pytest.raises(PartialIncrementalFailureError):
        execute_incremental_updates_from_event(
            None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
        )
    # f_a 已 stage（暂存区），但**没有任何 publish** → published 湖完全未动
    assert "f_a" in control["staged"]
    assert store.published == []
    # 事件保持 rejected，重试可收敛
    ledger_records = [l for l in (lake / ".event_ledger.jsonl").read_text().splitlines()]
    assert any("ev_atomic" in l and "rejected" in l for l in ledger_records)


def test_r14_production_event_retry_converges_once(tmp_path, monkeypatch):
    """失败后重跑同一 event_id → 幂等收敛，最终只 commit 一次。"""
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, control, factory = _setup_lake(tmp_path, fail_stage_on="f_b")
    store = _FakeStore()
    monkeypatch.setattr("data_access.get_store", lambda: store)
    _install_fake_publish(monkeypatch, store)

    with pytest.raises(PartialIncrementalFailureError):
        execute_incremental_updates_from_event(
            None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
        )
    # 修复 f_b 后重跑
    control["fail_stage_on"] = None
    out = execute_incremental_updates_from_event(
        None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
    )
    assert out["ledger_status"] == "committed"
    assert sorted(store.published) == ["f_a", "f_b"]
    # commit 记录只有一条 committed
    records = [l for l in (lake / ".event_ledger.jsonl").read_text().splitlines()
               if "ev_atomic" in l and "committed" in l]
    assert len(records) == 1


def test_r14_production_optin_publish_failure_rejects_and_raises(tmp_path, monkeypatch):
    """opt-in 下全部 stage 成功但 f_b publish 失败 → rejected + raise。

    R14 复查 P0-1：这是 opt-in 路径的**已知限制**——f_a 已发布但事件整体
    rejected（mixed published state），正是生产默认关闭自动发布的原因。真正的
    visibility transaction 留待后续；此测试只验证 fail-closed 拒绝 + 重试收敛。
    """
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, control, factory = _setup_lake(tmp_path)
    store = _FakeStore(fail_publish_on="f_b")
    monkeypatch.setattr("data_access.get_store", lambda: store)
    _install_fake_publish(monkeypatch, store)

    with pytest.raises(PartialIncrementalFailureError):
        execute_incremental_updates_from_event(
            None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
        )
    # f_a 已发布但事件整体 rejected（opt-in 已知限制）→ 重试按同一 event_id 收敛
    assert "f_a" in store.published
    assert "f_b" not in store.published
    # 重跑（修复 publish）→ 收敛、只 commit 一次
    store._fail_publish_on = None
    control["staged"] = []
    out = execute_incremental_updates_from_event(
        None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
    )
    assert out["ledger_status"] == "committed"


def test_r14_production_disabled_by_default_rejects_no_publish(tmp_path, monkeypatch):
    """R34 P0-039：production 事件**默认**走原子两阶段（不再默认拒绝）。

    旧 R14 语义：production 默认不自动发布（env bypass 不存在），事件直接拒绝。
    R34 修正：escape hatch 已删除——production DataEvent 唯一路径就是原子两阶段
    （全部 stage → 全部 publish），因此默认（无 ``DATA_EVENT_PRODUCTION_AUTO_PUBLISH``）
    事件现在正常原子发布，不再 ``ProductionEventAutoPublishDisabled``。
    """
    _force_production(monkeypatch)
    lake, control, factory = _setup_lake(tmp_path)
    store = _FakeStore()
    monkeypatch.setattr("data_access.get_store", lambda: store)
    _install_fake_publish(monkeypatch, store)

    out = execute_incremental_updates_from_event(
        None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
    )
    # 默认即原子两阶段：全部 stage → 全部 publish，ledger committed
    assert out["ledger_status"] == "committed"
    assert set(store.published) == {"f_a", "f_b"}  # 无 env 也原子发布
    raw = (lake / ".event_ledger.jsonl").read_text()
    assert "ev_atomic" in raw and "committed" in raw


def test_r14_production_gate_applies_without_event_id(tmp_path, monkeypatch):
    """production 事件即便没有 event_id 也不得自动发布（gate 不依赖 ledger）。"""
    _force_production(monkeypatch)
    ev = DataEvent(
        dataset="shared_ds",
        column="close",
        updated_date="2026-08-09",
        field_id="close",
        revision_kind="update",
    )
    with pytest.raises(ProductionEventAutoPublishDisabled):
        execute_incremental_updates_from_event(
            None, ev, lake_root=str(tmp_path / "lake"), market="ashare"
        )


def test_r14_production_forces_staging_target(tmp_path, monkeypatch):
    """opt-in 下 production 事件强制 ``write_target=staging``——调用方无法覆盖成
    ``staging_clickhouse``/``clickhouse``（stage-all 阶段产生 published/CH side
    effect → 后续 factor 失败 → CH mixed）。"""
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, control, factory = _setup_lake(tmp_path)
    store = _FakeStore()
    monkeypatch.setattr("data_access.get_store", lambda: store)
    _install_fake_publish(monkeypatch, store)

    out = execute_incremental_updates_from_event(
        None,
        _prod_event(),
        lake_root=lake,
        market="ashare",
        engine_factory=factory,
        materialize_kwargs={"write_target": "clickhouse"},
    )
    # fake engine 的 _mi 已断言收到 write_target == "staging"（否则本测试即失败）
    assert out["ledger_status"] == "committed"
    assert sorted(store.published) == ["f_a", "f_b"]


@pytest.mark.parametrize(
    "publish_result",
    [
        None,
        {},
        {
            "factor_id": "wrong-factor",
            "approved": True,
            "generation_id": "gen-f_a",
            "manifest_digest": "digest-f_a",
            "run_id": "run-f_a",
        },
        {
            "factor_id": "f_a",
            "approved": False,
            "generation_id": "gen-f_a",
            "manifest_digest": "digest-f_a",
            "run_id": "run-f_a",
        },
        {
            "factor_id": "f_a",
            "approved": True,
            "generation_id": "stale",
            "manifest_digest": "stale",
            "run_id": "stale",
        },
    ],
)
def test_r14_unproven_publish_completion_is_not_visible(
    tmp_path, monkeypatch, publish_result
):
    """None or stale completion cannot close the event or enter published[]."""
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, _control, factory = _setup_lake(tmp_path)
    monkeypatch.setattr(
        "factor_engine.storage.materialize.lake_publish.publish_factor_lake",
        lambda **kwargs: publish_result,
    )

    with pytest.raises(PartialIncrementalFailureError, match="publish failed"):
        execute_incremental_updates_from_event(
            None, _prod_event(), lake_root=lake, market="ashare", engine_factory=factory
        )
    raw = (lake / ".event_ledger.jsonl").read_text()
    assert "ev_atomic" in raw and "rejected" in raw


def test_r14_missing_staged_identity_blocks_publish_call(tmp_path, monkeypatch):
    """A stage summary without bound identity must fail before publisher mutation."""
    _force_production(monkeypatch)
    _enable_auto_publish(monkeypatch)
    lake, _control, factory = _setup_lake(tmp_path)

    def missing_identity_factory(*args, **kwargs):
        engine = factory(*args, **kwargs)
        original = engine.materialize_incremental.side_effect

        def _materialize(*call_args, **call_kwargs):
            result = original(*call_args, **call_kwargs)
            result["materialization"]["staging"].pop("identity")
            return result

        engine.materialize_incremental.side_effect = _materialize
        return engine

    monkeypatch.setattr(
        "factor_engine.storage.materialize.lake_publish.publish_factor_lake",
        lambda **kwargs: pytest.fail("publisher must not run without staged identity"),
    )
    with pytest.raises(PartialIncrementalFailureError, match="publish failed"):
        execute_incremental_updates_from_event(
            None,
            _prod_event(),
            lake_root=lake,
            market="ashare",
            engine_factory=missing_identity_factory,
        )
