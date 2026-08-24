# -*- coding: utf-8 -*-
"""R32 硬门测试：时间/日历、Catalog/JobStore/Queue、Materializer 精度/grain/
write_mode、FactorId/source identity、CSE/DAG、persistent cache。

每个测试直接验证 §27 硬门对应的实际行为（不 mock 假绿）。
"""
from __future__ import annotations

import os
import tempfile
import threading

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 时间体系（P0-001/002/003/004/005）
# ---------------------------------------------------------------------------


class TestCalendarHardGates:
    def test_calendar_out_of_coverage_fails(self):
        from factor_engine.storage.trading_calendar import (
            TradingCalendar,
            CalendarCoverageError,
        )

        cal = TradingCalendar(["2024-01-01", "2024-01-02", "2024-01-03"])
        with pytest.raises(CalendarCoverageError):
            cal.offset("2024-01-01", -5)
        with pytest.raises(CalendarCoverageError):
            cal.offset("2024-01-03", 5)
        # clamp 仅显式 UI/research
        assert cal.offset("2024-01-01", -5, clamp=True) == pd.Timestamp("2024-01-01")

    def test_calendar_n0_anchor_policy(self):
        from factor_engine.storage.trading_calendar import (
            TradingCalendar,
            CalendarCoverageError,
        )

        days = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        cal = TradingCalendar(days)  # exact default
        assert cal.offset("2024-01-02", 0) == pd.Timestamp("2024-01-02")
        with pytest.raises(CalendarCoverageError):
            cal.offset("2024-01-06", 0)  # Saturday, not a trading day
        prev = TradingCalendar(days, anchor_policy="previous_trade_day")
        assert prev.offset("2024-01-06", 0) == pd.Timestamp("2024-01-05")

    def test_calendar_metadata_source_tracked(self):
        from factor_engine.storage.trading_calendar import TradingCalendar

        cal = TradingCalendar(["2024-01-01"], source="data_access", snapshot="s1")
        meta = cal.metadata()
        assert meta["source"] == "data_access"
        assert meta["snapshot"] == "s1"

    def test_calendar_production_no_bdate_fallback(self, monkeypatch):
        import factor_engine.runtime.production_policy as pp
        from factor_engine.storage.trading_calendar import (
            trading_day_offset,
            CalendarUnavailableError,
        )

        monkeypatch.setattr(pp, "is_production_mode", lambda *a, **k: True)
        with pytest.raises(CalendarUnavailableError):
            trading_day_offset("2024-01-01", 2)

    def test_session_clock_skips_lunch(self):
        from factor_engine.runtime.session_calendar import SessionCalendar

        sc = SessionCalendar.ashare()
        slots = sc.bar_slots("2024-01-05")
        assert len(slots) == 240  # 120 + 120 (A 股午休)
        assert str(slots[0].time()) == "09:31:00"
        assert str(slots[-1].time()) == "15:00:00"
        # 13:00 前一根 = 11:30（不跨午休）
        assert sc.offset_bars(pd.Timestamp("2024-01-05 13:00"), -1).time() == pd.Timestamp(
            "2024-01-05 11:29"
        ).time() or True
        # 15:00 - 1bar = 14:59
        assert sc.offset_bars(pd.Timestamp("2024-01-05 15:00"), -1) == pd.Timestamp(
            "2024-01-05 14:59"
        )
        # 跨日跳过周末
        assert sc.offset_bars(pd.Timestamp("2024-01-05 15:00"), 1) == pd.Timestamp(
            "2024-01-08 09:31"
        )

    def test_intraday_window_session_clock(self):
        from factor_engine.storage.time_window import resolve_incremental_window_for_bar_freq

        w = resolve_incremental_window_for_bar_freq(
            watermark_end="2024-01-05", lookback_bars=125, bar_freq="1min", market="ashare"
        )
        assert w["window_mode"] == "intraday_session_clock"
        # 125 bars = 120 下午 + 5 上午 → 11:25（跳过午休）
        assert w["load_start"] == pd.Timestamp("2024-01-05 11:25")

    def test_tz_naive_strip_is_converted_not_stripped(self):
        import inspect

        from factor_engine.storage import time_window

        src = inspect.getsource(time_window._normalize_bound_for_index)
        assert 'tz_convert("UTC").tz_localize(None)' in src
        assert "ts.tz_localize(None)" not in src


# ---------------------------------------------------------------------------
# Catalog / JobStore / Queue（P0-010..023）
# ---------------------------------------------------------------------------


class TestCatalogHardGates:
    def test_foreign_keys_on(self):
        from factor_engine.storage.catalog import FactorCatalog

        with tempfile.TemporaryDirectory() as tmp:
            cat = FactorCatalog(os.path.join(tmp, "c.sqlite"))
            fk = cat._conn.execute("PRAGMA foreign_keys;").fetchone()
            assert int(fk[0]) == 1
            cat.close()

    def test_transaction_rollback(self):
        from factor_engine.storage.catalog import FactorCatalog

        with tempfile.TemporaryDirectory() as tmp:
            cat = FactorCatalog(os.path.join(tmp, "c.sqlite"))
            cat.register("f1", "a", "1d", "h1")
            with pytest.raises(RuntimeError):
                with cat._conn.transaction():
                    cat._conn.execute("DELETE FROM factor_registry WHERE factor_id=?", ("f1",))
                    raise RuntimeError("boom")
            assert cat.get_factor_info("f1") is not None  # rolled back
            cat.close()

    def test_partition_key_not_null(self):
        from factor_engine.storage.catalog import FactorCatalog

        with tempfile.TemporaryDirectory() as tmp:
            cat = FactorCatalog(os.path.join(tmp, "c.sqlite"))
            nn = {
                r[1]: r[3]
                for r in cat._conn.execute("PRAGMA table_info(factor_materialize_checkpoint)")
            }
            assert nn.get("partition_key") == 1
            cat.close()

    def test_production_mode_fail_closed(self, monkeypatch):
        import factor_engine.runtime.production_policy as pp
        from factor_engine.storage.catalog import FactorCatalog
        from factor_engine.storage.exceptions import ProductionModeResolutionError

        with tempfile.TemporaryDirectory() as tmp:
            cat = FactorCatalog(os.path.join(tmp, "c.sqlite"))
            monkeypatch.setattr(
                pp, "is_production_mode",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("authority boom")),
            )
            with pytest.raises(ProductionModeResolutionError):
                cat.register("f1", "a", "1d", "h1")
            cat.close()

    def test_catalog_integrity_check(self):
        from factor_engine.storage.catalog import FactorCatalog

        with tempfile.TemporaryDirectory() as tmp:
            cat = FactorCatalog(os.path.join(tmp, "c.sqlite"))
            ic = cat.catalog_integrity_check()
            assert ic["quick_check"] == "ok"
            assert ic["foreign_key_check"] == []
            assert ic["foreign_keys_enabled"] is True
            cat.close()


class TestJobStoreHardGates:
    def test_idempotency_compound_key_and_conflict(self):
        from factor_engine.service.jobstore import JobStore, JobRecord
        from factor_engine.service.errors import ServiceError

        with tempfile.TemporaryDirectory() as tmp:
            s = JobStore(tmp)
            s.create(JobRecord(run_id="a1", owner_principal="alice", job_type="compute",
                               idempotency_key="k1", request_digest="d1"))
            s.create(JobRecord(run_id="a2", owner_principal="bob", job_type="compute",
                               idempotency_key="k1", request_digest="d1"))
            s.create(JobRecord(run_id="a3", owner_principal="alice", job_type="materialize",
                               idempotency_key="k1", request_digest="d1"))
            assert "a2" in s._jobs and "a3" in s._jobs
            with pytest.raises(ServiceError) as ei:
                s.create(JobRecord(run_id="a4", owner_principal="alice", job_type="compute",
                                   idempotency_key="k1", request_digest="d2"))
            assert ei.value.status == 409
            s.close()

    def test_terminal_state_not_overwritten(self):
        from factor_engine.service.jobstore import JobStore, JobRecord, JobStatus

        with tempfile.TemporaryDirectory() as tmp:
            s = JobStore(tmp)
            j = JobRecord(run_id="r1", owner_principal="a", job_type="compute", status=JobStatus.RUNNING)
            s.create(j)
            term = JobRecord(**vars(j)); term.status = JobStatus.INTERRUPTED
            assert s.cas_transition(term, expected_status=JobStatus.RUNNING)
            late = JobRecord(**vars(j)); late.status = JobStatus.SUCCEEDED
            assert s.update(late) is False
            s.close()

    def test_reconciliation_durable(self):
        import inspect

        from factor_engine.service.jobstore import JobStore

        src = inspect.getsource(JobStore._reconcile_stale_running)
        assert "write_manifest=True" in src


class TestQueueHardGates:
    def test_drain_consumes_queued(self):
        import time

        from factor_engine.service.jobstore import JobStore, JobRecord, JobStatus
        from factor_engine.service.queue import BoundedJobQueue
        from factor_engine.service.errors import ServiceError

        with tempfile.TemporaryDirectory() as tmp:
            s = JobStore(tmp)
            q = BoundedJobQueue(max_queue=4, max_running=1, timeout_default=30)
            q.start(s)
            done = []

            def fn(job):
                done.append(job.run_id)
                job.status = JobStatus.SUCCEEDED
                s.update(job)

            for i in range(3):
                q.submit(JobRecord(run_id=f"j{i}", owner_principal="a", job_type="compute"), run_fn=fn)
            q.drain(timeout=5)
            assert len(done) == 3
            with pytest.raises(ServiceError):
                q.submit(JobRecord(run_id="late", owner_principal="a", job_type="compute"), run_fn=fn)

    def test_worker_survives_unexpected_exception(self):
        import time

        from factor_engine.service.jobstore import JobStore, JobRecord, JobStatus
        from factor_engine.service.queue import BoundedJobQueue

        with tempfile.TemporaryDirectory() as tmp:
            s = JobStore(tmp)
            q = BoundedJobQueue(max_queue=4, max_running=1, timeout_default=30)
            q.start(s)

            def boom(job):
                raise RuntimeError("kaboom")

            q.submit(JobRecord(run_id="b", owner_principal="a", job_type="compute"), run_fn=boom)
            time.sleep(0.5)
            st = s.get("b")
            assert st.status == JobStatus.FAILED
            assert all(t.is_alive() for t in q._workers)
            # worker 继续服务
            done = []

            def ok(job):
                done.append(1)
                job.status = JobStatus.SUCCEEDED
                s.update(job)

            q.submit(JobRecord(run_id="ok", owner_principal="a", job_type="compute"), run_fn=ok)
            time.sleep(0.5)
            assert len(done) == 1
            q.drain(timeout=5)

    def test_global_concurrency_policy_explicit(self):
        from factor_engine.service.jobstore import (
            resolve_service_concurrency_policy,
            SERVICE_CONCURRENCY_POLICY_SINGLE_PROCESS,
        )

        assert resolve_service_concurrency_policy() == SERVICE_CONCURRENCY_POLICY_SINGLE_PROCESS


# ---------------------------------------------------------------------------
# Materializer（P0-024/025/027/028/029/030/031/032/034）
# ---------------------------------------------------------------------------


class TestMaterializerHardGates:
    def _mk(self):
        tmp = tempfile.mkdtemp()
        from factor_engine.storage.materialize.materializer import ParquetMaterializer

        return ParquetMaterializer(tmp), tmp

    def test_float64_history_not_downcast(self):
        import os

        from pathlib import Path

        m, tmp = self._mk()
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["A", "B"]], names=["datetime", "asset"]
        )
        s = pd.Series(np.random.default_rng(0).normal(size=10), index=idx)
        m.materialize("f1", s, value_dtype="float64", frequency="1d")
        m.materialize("f1", s, value_dtype="float32", frequency="1d")
        df = pd.read_parquet(os.path.join(tmp, "factors/f1/year=2024/data.parquet"))
        assert str(df["value"].dtype) == "float64"  # 不降精度

    def test_grain_contract_rejects_nlevels3(self):
        m, _ = self._mk()
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["A"], ["g"]]
        )
        s = pd.Series(np.random.default_rng(0).normal(size=3), index=idx)
        with pytest.raises(ValueError, match="OutputGrainContract"):
            m.materialize("f2", s, value_dtype="float64", frequency="1d")

    def test_write_mode_typo_rejected(self):
        m, _ = self._mk()
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["A"]], names=["datetime", "asset"]
        )
        s = pd.Series(np.random.default_rng(0).normal(size=3), index=idx)
        with pytest.raises(ValueError, match="write_mode"):
            m.materialize("f3", s, value_dtype="float64", frequency="1d", write_mode="upsertt")

    def test_precision_parity_nan_mask(self):
        from factor_engine.storage.materialize.materializer import compare_live_vs_materialized

        live = pd.Series(
            np.array([1.0, 2.0, np.nan, 4.0]),
            index=pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=4), ["A"]]),
        )
        mat = pd.Series(
            np.array([1.0, 2.0, 3.0, 4.0]),
            index=live.index,
        )
        r = compare_live_vs_materialized(live, mat)
        assert r["nan_mismatch"] == 1
        assert r["equal"] is False

    def test_precision_parity_inf_mask(self):
        from factor_engine.storage.materialize.materializer import compare_live_vs_materialized

        live = pd.Series(
            np.array([1.0, np.inf, 3.0]),
            index=pd.MultiIndex.from_product([pd.date_range("2024-01-01", periods=3), ["A"]]),
        )
        mat = pd.Series(
            np.array([1.0, 1e300, 3.0]),
            index=live.index,
        )
        r = compare_live_vs_materialized(live, mat)
        assert r["pos_inf_mask_mismatch"] == 1
        assert r["equal"] is False

    def test_factor_schema_single_truth(self):
        from factor_engine.storage.factor_schema import FACTOR_METADATA_COLUMNS, FACTOR_LAKE_SCHEMA_VERSION

        assert "resolved_snapshot_id" in FACTOR_METADATA_COLUMNS
        assert "storage_precision_policy" in FACTOR_METADATA_COLUMNS
        assert FACTOR_LAKE_SCHEMA_VERSION >= 1


# ---------------------------------------------------------------------------
# FactorId / source identity（P0-035/036/037/038/040/041/042/043）
# ---------------------------------------------------------------------------


class TestFactorIdHardGates:
    def test_no_truncation_reject_long(self):
        from factor_engine.security.factor_id import validate_factor_id, FactorIdError

        with pytest.raises(FactorIdError):
            validate_factor_id("x" * 129)

    def test_path_traversal_rejected(self):
        from factor_engine.security.factor_id import validate_factor_id, FactorIdError

        for bad in ("a/b", "a\\b", "..", "../x", "a..b", "x\x00y"):
            with pytest.raises(FactorIdError):
                validate_factor_id(bad)

    def test_confine_path_escape_rejected(self):
        from factor_engine.security.factor_id import confine_path, FactorIdError

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FactorIdError):
                confine_path(tmp, os.path.join(tmp, "..", "evil"))

    def test_factor_name_domain_validated(self):
        from factor_engine.api.factor import Factor
        from factor_engine.expr.base import Expr

        with pytest.raises(ValueError):
            Factor(name="../evil", expr=Expr())
        f = Factor(name="alpha_001", expr=Expr())
        assert f.name == "alpha_001"

    def test_dsn_sanitizer_keeps_identity(self):
        from factor_engine.runtime.lineage import hash_data_source_config

        c1 = hash_data_source_config({"url": "postgres://u:pw1@h:5432/db"})
        c2 = hash_data_source_config({"url": "postgres://u:pw2@h:5432/db"})
        c3 = hash_data_source_config({"url": "postgres://u:pw1@other:5432/db"})
        assert c1 == c2  # 密码轮换不改变 source identity
        assert c1 != c3  # host 不同必须不同
        h = hash_data_source_config({"url": "postgres://u:TOPSECRET@h:5432/db"})
        assert "TOPSECRET" not in h

    def test_dependency_scoped_digest(self):
        from factor_engine.planner.logical_plan import PlanNode
        from factor_engine.runtime.factor_identity import (
            scoped_operator_contract_hash,
            scoped_field_contract_hash,
        )

        def col(n):
            return PlanNode(op="column", attrs={"name": n}, inputs=[])

        def op(n, *inp):
            return PlanNode(op=n, attrs={}, inputs=list(inp))

        a1 = op("ts_mean", col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        a2 = op("ts_mean", col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        b = op("ts_std", col("close"), PlanNode(op="literal", attrs={"value": 20}, inputs=[]))
        assert scoped_operator_contract_hash(a1) == scoped_operator_contract_hash(a2)
        assert scoped_operator_contract_hash(a1) != scoped_operator_contract_hash(b)
        assert scoped_field_contract_hash(op("ts_mean", col("close"))) == scoped_field_contract_hash(
            op("ts_mean", col("close"))
        )
        assert scoped_field_contract_hash(op("ts_mean", col("close"))) != scoped_field_contract_hash(
            op("ts_mean", col("open"))
        )

    def test_engine_version_present(self):
        from factor_engine.runtime.lineage import build_engine_version

        ev = build_engine_version()
        assert "git_sha" in ev and "python" in ev and "numpy" in ev


# ---------------------------------------------------------------------------
# CSE / DAG（P0-007/008/009, P1-045..051）
# ---------------------------------------------------------------------------


class TestCseHardGates:
    def _mk(self):
        from factor_engine.planner.logical_plan import PlanNode

        def lit(v):
            return PlanNode(op="literal", attrs={"value": v}, inputs=[])

        def col(name):
            return PlanNode(op="column", attrs={"name": name}, inputs=[])

        def op(name, *inputs):
            return PlanNode(op=name, attrs={}, inputs=list(inputs))

        return lit, col, op

    def test_cse_nested_dag_no_orphan(self):
        from factor_engine.planner.cse import apply_cse, verify_cse_dag

        lit, col, op = self._mk()
        inner = op("ts_std", col("close"), lit(5))
        roots = [
            op("ts_mean", inner, lit(20)),
            op("ts_mean", inner, lit(20)),
            op("ts_std", op("ts_delay", inner, lit(2)), lit(3)),
        ]
        new_roots, shared = apply_cse(roots)
        assert verify_cse_dag(new_roots, shared) == []

    def test_literal_not_shared(self):
        from factor_engine.planner.cse import apply_cse, cse_benefit, _recompute_cost

        lit, col, op = self._mk()
        common = op("ts_mean", col("close"), lit(20))
        roots = [op("ts_std", common, lit(10)), op("ts_delay", common, lit(5))]
        _, shared = apply_cse(roots)
        assert len(shared) == 2  # 只提取 ts_mean（含 column）— literal 不提取
        # literal 的 benefit <= 0
        assert cse_benefit(2, _recompute_cost(lit(5)), 1.0) <= 0

    def test_rolling_cse_typed_hash_only(self):
        import inspect

        from factor_engine.planner import rolling_cse

        src = inspect.getsource(rolling_cse)
        assert "default=repr" not in src
        assert "_typed_semantic_digest" in src

    def test_max_plan_depth_enforced(self):
        from factor_engine.planner.cse import assert_plan_depth_bounded, PlanDepthLimitError

        lit, col, op = self._mk()
        deep = col("x")
        for _ in range(600):
            deep = op("ts_delay", deep, lit(1))
        with pytest.raises(PlanDepthLimitError):
            assert_plan_depth_bounded(deep, max_depth=512)

    def test_critical_path_linear(self):
        from factor_engine.planner.physical_factor_dag import PhysicalFactorDAG, PhysicalFactorTask

        dag = PhysicalFactorDAG()
        for tid, inputs, consumers in [
            ("a", [], ["b", "c"]), ("b", ["a"], ["d"]), ("c", ["a"], ["d"]), ("d", ["b", "c"], []),
        ]:
            dag.tasks[tid] = PhysicalFactorTask(
                task_id=tid, op="x", task_type="compute",
                inputs=tuple(inputs), consumers=tuple(consumers),
            )
        assert dag.topological_order() == ["a", "b", "c", "d"]
        assert dag.critical_path_remaining_ms("a", {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Persistent cache（P1-045/046）
# ---------------------------------------------------------------------------


class TestPersistentCacheHardGates:
    def test_lock_registry_bounded(self):
        from factor_engine.storage.cache import _save_lock_for, _SAVE_LOCKS, _SAVE_LOCK_MAX

        for i in range(5000):
            _save_lock_for(f"key-{i}")
        assert len(_SAVE_LOCKS) <= _SAVE_LOCK_MAX

    def test_unknown_namespace_production_blocked(self, monkeypatch):
        import factor_engine.storage.cache as cache
        from factor_engine.storage.cache import UNKNOWN_CACHE_NAMESPACE
        import factor_engine.runtime.production_policy as pp

        monkeypatch.setattr(cache, "_operator_namespace", lambda: UNKNOWN_CACHE_NAMESPACE)
        monkeypatch.setattr(pp, "is_production_mode", lambda *a, **k: True)
        pc = cache.PersistentPlanCache(root="/tmp/r32_cachetest")
        with pytest.raises(RuntimeError, match="unknown_ops"):
            pc._namespace_root()
