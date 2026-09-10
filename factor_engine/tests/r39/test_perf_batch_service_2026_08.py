# -*- coding: utf-8 -*-
"""R39 性能整改 —— runtime.batch_service 相关真实路径测试。

覆盖：
- PERF-001：adaptive-scheduler 路径不调用 ``_maybe_prepare_batch_data``
  （Gate-01：``legacy_union_prefetch_count == 0``）。
- PERF-013：ReadOnlyOverlayMap 读写/隔离语义 + ``_execute_root_with_path`` 不再
  全量复制共享 dict（Gate-09）。
- PERF-014 / PERF-015：BatchWarmupPlan 全 batch 只算一次 + 扫描成本感知分组。
- PERF-017：OutputSlice 输出切片 plumbing（零拷贝位置视图 / 已携带 slice 不重切）。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import factor_engine.runtime.batch_service as bs


# ---------------------------------------------------------------------------
# 测试用最小桩（避免拉起完整 FactorEngine / DataAccess）
# ---------------------------------------------------------------------------


class _FakeFactor:
    def __init__(self, name: str, freq: str = "D", universe=None):
        self.name = name
        self.freq = freq
        self.universe = universe


class _FakeDag:
    def __init__(self, roots=()):
        self.roots = list(roots)
        self.shared_nodes = {}


class _FakeEngine:
    run_mode = "research"

    def __init__(self):
        self._prefetch_calls = 0

    def _dag_from_factors(self, factors, **kw):
        return _FakeDag(), {}

    def _prepare_batch_data(self, *a, **k):
        self._prefetch_calls += 1
        return None


# ---------------------------------------------------------------------------
# PERF-001：adaptive-scheduler 不提前 full-union prefetch
# ---------------------------------------------------------------------------


def test_scheduler_path_skips_legacy_prefetch(monkeypatch):
    monkeypatch.delenv("FACTOR_ENGINE_LAYER_LOOP", raising=False)
    monkeypatch.setattr(
        bs, "assert_production_run_flags", lambda **k: None
    )
    monkeypatch.setattr(
        bs, "assert_production_factors", lambda *a, **k: None
    )
    calls = {"prepared": 0, "scheduled": 0}

    def _spy_prepare(engine, dag, analyses, **kw):
        calls["prepared"] += 1
        return {"passed": True}

    def _spy_scheduler(*a, **k):
        calls["scheduled"] += 1
        return {"results": {}, "scheduler_stats": {}}

    monkeypatch.setattr(bs, "_maybe_prepare_batch_data", _spy_prepare)
    monkeypatch.setattr(bs, "_execute_run_many_scheduler", _spy_scheduler)
    monkeypatch.setattr(
        bs, "_maybe_prepare_batch_warmup", lambda engine, *a, **k: (engine, {})
    )
    monkeypatch.setattr(bs, "_batch_source_bars_per_day", lambda e: 1)

    engine = _FakeEngine()
    before = bs.legacy_union_prefetch_count()
    out = bs.execute_run_many(
        engine,
        [_FakeFactor("a"), _FakeFactor("b")],
        perf=None,
    )
    assert calls["prepared"] == 0, "scheduler path must NOT call _maybe_prepare_batch_data"
    assert calls["scheduled"] == 1
    assert out["control_plane"] == "adaptive_scheduler"
    assert out["legacy_union_prefetch_count"] == 0
    assert engine._prefetch_calls == 0, "engine._prepare_batch_data must never run before scheduler"
    assert bs.legacy_union_prefetch_count() == before, "process-level legacy counter unchanged"


def test_legacy_layer_loop_still_prefetches(monkeypatch):
    # FACTOR_ENGINE_LAYER_LOOP=1 → legacy 路径保留 union prefetch（行为不变）。
    # 用 sentinel 截断 legacy 后续执行（_make_context），只验证 control-plane 分流。
    monkeypatch.setenv("FACTOR_ENGINE_LAYER_LOOP", "1")
    monkeypatch.setattr(
        bs, "assert_production_run_flags", lambda **k: None
    )
    monkeypatch.setattr(
        bs, "assert_production_factors", lambda *a, **k: None
    )
    calls = {"prepared": 0, "scheduled": 0}

    def _spy_prepare(engine, dag, analyses, **kw):
        calls["prepared"] += 1
        return None

    def _spy_scheduler(*a, **k):
        calls["scheduled"] += 1
        return {}

    class _LegacyReached(Exception):
        pass

    monkeypatch.setattr(bs, "_maybe_prepare_batch_data", _spy_prepare)
    monkeypatch.setattr(bs, "_execute_run_many_scheduler", _spy_scheduler)
    monkeypatch.setattr(
        bs, "_maybe_prepare_batch_warmup", lambda engine, *a, **k: (engine, {})
    )
    monkeypatch.setattr(bs, "_batch_source_bars_per_day", lambda e: 1)

    engine = _FakeEngine()
    # 进入 layer-loop 的 ctx 构造即触发 sentinel，截断后续执行
    engine._make_context = lambda *a, **k: (_ for _ in ()).throw(_LegacyReached())

    assert bs.resolve_batch_execution_control_plane() == "legacy"
    with pytest.raises(_LegacyReached):
        bs.execute_run_many(engine, [_FakeFactor("a")], perf=None)
    assert calls["prepared"] == 1, "legacy path keeps full-union prefetch"
    assert calls["scheduled"] == 0


# ---------------------------------------------------------------------------
# PERF-013：ReadOnlyOverlayMap 语义 + _execute_root_with_path 不复制共享 dict
# ---------------------------------------------------------------------------


def test_overlay_map_read_write_isolation_semantics():
    from factor_engine.runtime.readonly_overlay_map import ReadOnlyOverlayMap

    base = {"a": 1, "b": 2}
    ov = ReadOnlyOverlayMap(base, {})
    # read through base
    assert ov["a"] == 1
    assert "a" in ov
    # write goes to local only
    ov["c"] = 3
    assert ov["c"] == 3
    assert "c" in ov
    # keys/items/contains reflect local+base
    assert set(ov.keys()) == {"a", "b", "c"}
    assert set(iter(ov)) == {"a", "b", "c"}
    assert dict(ov) == {"a": 1, "b": 2, "c": 3}
    assert len(ov) == 3
    # local shadows base without mutating it
    ov["a"] = 99
    assert ov["a"] == 99
    assert base["a"] == 1
    # per-root isolation: a second overlay does NOT see the first root's local
    ov2 = ReadOnlyOverlayMap(base, {})
    assert ov2["a"] == 1
    assert "c" not in ov2
    assert len(ov2) == 2
    # base-only pop returns value without deleting shared state
    assert ov2.pop("b") == 2
    assert "b" in ov2
    assert base["b"] == 2
    # deleting a base-only key is forbidden (fail-closed)
    with pytest.raises(KeyError):
        del ov2["a"]


def test_execute_root_with_path_uses_overlay_no_full_copy(monkeypatch):
    from dataclasses import dataclass, field, replace

    from factor_engine.runtime.readonly_overlay_map import ReadOnlyOverlayMap

    monkeypatch.setattr(
        bs, "assert_production_fastpath_runtime", lambda *a, **k: None
    )
    monkeypatch.setattr(
        bs, "_assert_no_native_certified_fallback", lambda *a, **k: None
    )

    @dataclass
    class _Ctx:
        task_id: str | None = None
        factor_id: str | None = None
        runtime_stats: dict = field(default_factory=dict)
        shared_result_cache: dict | None = None
        shared_long_lazy_cache: dict | None = None
        materialized_long_lazy: dict | None = None
        materialized_series: dict | None = None
        panel_cache: dict | None = None
        run_mode: str = "research"

    base_shared = {f"s{i}": i for i in range(1000)}
    base_long = {"shared_lazy": "L"}
    ctx = _Ctx(
        runtime_stats={},
        shared_result_cache=base_shared,
        shared_long_lazy_cache=base_long,
        materialized_long_lazy={},
        materialized_series={},
        panel_cache={},
    )
    seen: dict = {}

    class _Backend:
        def execute(self, plan, local_ctx):
            seen["local_ctx"] = local_ctx
            # every shared mapping is an overlay, not a dict copy
            assert isinstance(local_ctx.shared_result_cache, ReadOnlyOverlayMap)
            assert isinstance(local_ctx.shared_long_lazy_cache, ReadOnlyOverlayMap)
            assert isinstance(local_ctx.materialized_long_lazy, ReadOnlyOverlayMap)
            assert isinstance(local_ctx.materialized_series, ReadOnlyOverlayMap)
            # overlay shares the SAME base object (no clone)
            assert local_ctx.shared_result_cache.base is base_shared
            assert local_ctx.shared_long_lazy_cache.base is base_long
            assert local_ctx.shared_result_cache["s500"] == 500
            # telemetry proves zero entries were copied into the root ctx
            assert local_ctx.runtime_stats["dict_entry_copy_count"] == 0
            idx = pd.MultiIndex.from_product(
                [pd.to_datetime(["2024-01-01"]), ["A"]],
                names=["timestamp", "instrument"],
            )
            return pd.Series([1.0], index=idx)

    result, path = bs._execute_root_with_path(
        _Backend(), object(), ctx, run_mode="research", factor_name="x"
    )
    assert seen.get("local_ctx") is not None
    assert result.iloc[0] == 1.0
    # base dicts are untouched / not copied
    assert len(base_shared) == 1000 and base_shared["s0"] == 0
    assert base_long == {"shared_lazy": "L"}
    # a root-local write does NOT leak into base
    local_ctx = seen["local_ctx"]
    local_ctx.shared_result_cache["s_new"] = 123
    assert "s_new" not in base_shared


# ---------------------------------------------------------------------------
# PERF-014 / PERF-015：BatchWarmupPlan 只算一次 + 扫描成本感知分组
# ---------------------------------------------------------------------------


def _fake_warmup_ctx(engine, rw, *, full=False):
    from factor_engine.runtime.run_window import RunWindow
    from factor_engine.runtime.warmup_service import WarmupContext

    if rw is None:
        rw = RunWindow(
            requested_start="2024-01-01",
            requested_end="2024-12-31",
            actual_load_start="2024-01-01",
            actual_load_end="2024-12-31",
            warmup_bars=0,
            trim_output=True,
            full_history_required=full,
            full_history_start="2020-01-01" if full else None,
        )
    return WarmupContext(
        engine=engine,
        run_window=rw,
        source_bar_freq="D",
        history_buffer=0,
        bars_per_day=1,
        full_history_required=full,
    )


class _PlanSrc:
    dataset = "ashare_daily"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    instrument_filter = ["A", "B", "C"]
    bar_freq = "D"
    freq = "D"
    instrument_count = 3

    def time_range(self):
        return ("2024-01-01", "2024-12-31")


class _PlanEngine:
    run_mode = "research"
    data_source = _PlanSrc()


def _analysis(name: str, cols, full: bool = False, lookback: int = 5):
    return SimpleNamespace(
        referenced_columns=set(cols),
        lookback=lookback,
        requires_full_history=full,
        ir=None,
        node_count=5,
    )


def test_batch_warmup_plan_computed_once_and_groups_by_scan_cost(monkeypatch):
    import factor_engine.runtime.batch_warmup_plan as bwp

    calls: dict[str, int] = {"prepare_run_warmup": 0}

    def _spy_prepare_run_warmup(engine, factor, analysis, **kw):
        calls["prepare_run_warmup"] += 1
        full = bool(getattr(analysis, "requires_full_history", False))
        return _fake_warmup_ctx(engine, None, full=full)

    monkeypatch.setattr("factor_engine.runtime.warmup_service.prepare_run_warmup", _spy_prepare_run_warmup)

    factors = [
        _FakeFactor("short_a"),
        _FakeFactor("short_b"),
        _FakeFactor("long_c"),
        _FakeFactor("full_d"),
    ]
    analyses = {
        "short_a": _analysis("short_a", ["close", "volume"]),
        "short_b": _analysis("short_b", ["close", "volume"]),
        "long_c": _analysis("long_c", ["close", "open"], lookback=120),
        "full_d": _analysis("full_d", ["close"], full=True),
    }
    engine = _PlanEngine()
    plan = bwp.compute_batch_warmup_plan(
        analyses,
        None,
        engine,
        factors=factors,
        auto_warmup=True,
        trim_warmup=True,
        market=None,
        strict=True,
    )
    # 每个因子 run_window 只算一次（PERF-014 的「compute once」在单次调用内成立）
    assert calls["prepare_run_warmup"] == len(factors)
    assert set(plan.per_factor.keys()) == {"short_a", "short_b", "long_c", "full_d"}
    # 扫描成本感知分组（PERF-015）：同窗短因子合并；full_history 独立组置末
    all_names = [g.factor_names for g in plan.groups]
    flat = [n for grp in all_names for n in grp]
    assert set(flat) == {"short_a", "short_b", "long_c", "full_d"}
    full_group = [g for g in plan.groups if "full_d" in g.factor_names]
    assert len(full_group) == 1
    assert full_group[0].factor_names == ("full_d",)
    assert all(g.scan_bytes > 0 for g in plan.groups)
    assert plan.to_dict()["group_count"] >= 2


def test_maybe_prepare_batch_warmup_consumes_plan_no_recompute(monkeypatch):
    import factor_engine.runtime.batch_service as bs_mod
    import factor_engine.runtime.batch_warmup_plan as bwp

    calls: dict[str, int] = {"prepare_run_warmup": 0}

    def _spy_prepare_run_warmup(engine, factor, analysis, **kw):
        calls["prepare_run_warmup"] += 1
        return _fake_warmup_ctx(engine, None)

    monkeypatch.setattr("factor_engine.runtime.warmup_service.prepare_run_warmup", _spy_prepare_run_warmup)

    factors = [_FakeFactor("a"), _FakeFactor("b")]
    analyses = {
        "a": _analysis("a", ["close"]),
        "b": _analysis("b", ["close"]),
    }
    engine = _PlanEngine()
    plan = bwp.compute_batch_warmup_plan(
        analyses,
        None,
        engine,
        factors=factors,
        auto_warmup=True,
        trim_warmup=True,
        market=None,
        strict=True,
    )
    first_count = calls["prepare_run_warmup"]
    assert first_count == 2
    # 消费既有 plan → 不再调用 prepare_run_warmup
    engine_to_use, per_windows = bs_mod._maybe_prepare_batch_warmup(
        engine,
        factors,
        analyses,
        auto_warmup=True,
        trim_warmup=True,
        market=None,
        plan=plan,
    )
    assert calls["prepare_run_warmup"] == first_count, "plan must be consumed, not recomputed"
    assert set(per_windows.keys()) == {"a", "b"}
    assert engine_to_use is engine  # load_start == requested_start → no narrowing


def test_cluster_factors_by_cost_uses_scan_cost_groups(monkeypatch):
    import factor_engine.runtime.batch_warmup_plan as bwp

    monkeypatch.setattr(
        "factor_engine.runtime.warmup_service.prepare_run_warmup",
        lambda engine, factor, analysis, **kw: _fake_warmup_ctx(
            engine,
            None,
            full=bool(getattr(analysis, "requires_full_history", False)),
        ),
    )
    factors = [
        _FakeFactor("s1"),
        _FakeFactor("s2"),
        _FakeFactor("fh"),
    ]
    analyses = {
        "s1": _analysis("s1", ["close"]),
        "s2": _analysis("s2", ["close"]),
        "fh": _analysis("fh", ["close"], full=True),
    }
    engine = _PlanEngine()
    plan = bwp.compute_batch_warmup_plan(
        analyses,
        None,
        engine,
        factors=factors,
        auto_warmup=True,
        trim_warmup=True,
        market=None,
        strict=True,
    )
    waves = bs._cluster_factors_by_cost(engine, factors, analyses, plan=plan)
    flat = {f.name for wave in waves for f in wave}
    assert flat == {"s1", "s2", "fh"}
    # full-history factor 与短因子不在同一 wave
    for wave in waves:
        names = {f.name for f in wave}
        assert not ({"fh"} & names and {"s1", "s2"} & names)


# ---------------------------------------------------------------------------
# PERF-017：OutputSlice 输出切片 plumbing
# ---------------------------------------------------------------------------


def _result_series():
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            ["A", "B"],
        ],
        names=["timestamp", "instrument"],
    )
    return pd.Series(np.arange(8.0), index=idx)


def test_output_slice_zero_copy_positional_trim():
    from factor_engine.planner.output_slice import (
        OutputSlice,
        apply_output_slice,
        carried_output_slice,
        compute_output_slice,
    )
    from factor_engine.storage.time_window import slice_series_time_window

    s = _result_series()
    rw = SimpleNamespace(
        requested_start="2024-01-02", requested_end="2024-01-03", trim_output=True
    )
    sl = compute_output_slice(s, rw, bars_per_day=1)
    assert sl == OutputSlice(2, 6)
    trimmed = apply_output_slice(s, sl)
    expected = slice_series_time_window(
        s,
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-03"),
    )
    assert trimmed.equals(expected)
    assert trimmed.index.names == ["timestamp", "instrument"]
    # 连续 block 位置切片零拷贝
    assert np.shares_memory(s.values, trimmed.values)
    assert carried_output_slice(trimmed) is not None


def test_trim_batch_result_respects_carried_slice_and_falls_back():
    from factor_engine.planner.output_slice import compute_output_slice
    from factor_engine.storage.time_window import slice_series_time_window

    s = _result_series()
    rw = SimpleNamespace(
        requested_start="2024-01-02", requested_end="2024-01-03", trim_output=True
    )
    expected = slice_series_time_window(
        s,
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-03"),
    )
    # 无 slice 时正常裁剪（且与 mask 语义一致）
    out = bs._trim_batch_result(s, rw, bars_per_day=1)
    assert out.equals(expected)
    assert np.shares_memory(s.values, out.values)
    # 已携带 slice → 不再二次裁剪（返回同一对象）
    carried = bs._trim_batch_result(s, rw, bars_per_day=1)
    rw2 = SimpleNamespace(
        requested_start="2024-01-03", requested_end="2024-01-03", trim_output=True
    )
    again = bs._trim_batch_result(carried, rw2, bars_per_day=1)
    assert again is carried


def test_trim_batch_result_skips_without_trim():
    s = _result_series()
    rw = SimpleNamespace(requested_start=None, trim_output=True)
    assert bs._trim_batch_result(s, rw, bars_per_day=1) is s
    rw2 = SimpleNamespace(
        requested_start="2024-01-02", requested_end=None, trim_output=False
    )
    assert bs._trim_batch_result(s, rw2, bars_per_day=1) is s


def test_output_slice_falls_back_on_non_monotonic():
    from factor_engine.planner.output_slice import compute_output_slice

    # 非单调索引 → 返回 None（调用方走旧 mask 语义，行为不变）
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-01"), "B"),
            (pd.Timestamp("2024-01-02"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    rw = SimpleNamespace(
        requested_start="2024-01-02", requested_end="2024-01-03", trim_output=True
    )
    assert compute_output_slice(s, rw, bars_per_day=1) is None
    from factor_engine.storage.time_window import slice_series_time_window

    out = bs._trim_batch_result(s, rw, bars_per_day=1)
    expected = slice_series_time_window(
        s,
        start=pd.Timestamp("2024-01-02"),
        end=pd.Timestamp("2024-01-03"),
    )
    assert out.equals(expected)


def test_output_slice_validation():
    from factor_engine.planner.output_slice import OutputSlice

    assert OutputSlice(0, None).is_full
    assert OutputSlice(2, 6).to_dict() == {"start_offset": 2, "end_offset": 6}
    with pytest.raises(ValueError):
        OutputSlice(-1, 5)
    with pytest.raises(ValueError):
        OutputSlice(6, 2)
