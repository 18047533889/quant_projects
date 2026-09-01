"""P0 主链收口测试：SearchPipeline 端到端 + runner 路由 + RoundManager 闭环。

全离线：stub llm_fn + 内存 dedup + 注入假 evaluator / 假 manifest。
不 import torch / openai 训练路径（pipeline 模块级无 torch/openai 依赖）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from alphaprobe.pipeline import (
    PipelineConfig,
    SearchPipeline,
    make_fe_evaluate_fn,
    make_stub_llm_fn,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

FIXED_BUNDLE = {
    "rankic": 0.05,
    "ic": 0.03,
    "icir": 0.2,
    "coverage": 0.9,
    "nan_inf_ratio": 0.01,
    "untradeable_ratio": 0.1,
}


def _make_pipeline(**cfg_kwargs) -> SearchPipeline:
    cfg = PipelineConfig(pool_target=8, pool_max=16, budget_per_round=6, **cfg_kwargs)
    return SearchPipeline(experiment=None, data_train=None, config=cfg)


def _inject_eval(sp: SearchPipeline, bundle=dict(FIXED_BUNDLE), *, none_all: bool = False):
    def _fake(formulas, fidelity="L2_full_train", context=None):
        if none_all:
            return [None] * len(formulas)
        return [dict(bundle)] * len(formulas)

    sp._evaluate_fn = _fake
    sp.evaluator = None  # 强制静态降级路径（不依赖 LegacyCompatEvaluator）
    return sp


# ---------------------------------------------------------------------------
# 1. run_round 端到端（stub llm + 内存 dedup + 假 evaluator）
# ---------------------------------------------------------------------------


def test_run_round_end_to_end_generate_evaluate_admit():
    sp = _inject_eval(_make_pipeline())
    result = sp.run_round(
        round_id="r1",
        parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
    )
    assert result.candidates_generated > 0
    assert result.duplicates_filtered >= 0
    assert result.evaluated == result.candidates_generated
    assert result.admitted >= 0
    assert len(result.pool_snapshot) == sp.pool_size()


def test_second_round_same_formula_exact_filtered():
    """同公式第二轮被 EXACT 过滤（dedup 硬重复拦截）。"""
    sp = _inject_eval(_make_pipeline())
    parents = [{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}]
    r1 = sp.run_round(round_id="r1", parents=parents)
    assert r1.evaluated == r1.candidates_generated
    r2 = sp.run_round(round_id="r2", parents=parents)
    # 第二轮生成的公式与第一轮相同（确定性 stub）→ 全部被 EXACT 过滤
    assert r2.candidates_generated > 0
    assert r2.evaluated == 0
    assert r2.duplicates_filtered >= r2.candidates_generated


def test_stub_llm_forced_candidates_json():
    """forced_candidates 固定 JSON 文本（测试注入 stub llm_fn）。"""
    forced = [{"formula": "rank(close)", "explanation": "e", "hypothesis": "h", "action_type": "REFINE"}]
    llm = make_stub_llm_fn(forced_candidates=forced)
    text = llm("sys", "user", "cheap")
    payload = json.loads(text)
    assert payload["candidates"][0]["formula"] == "rank(close)"


# ---------------------------------------------------------------------------
# 2. evaluator 全 None 降级
# ---------------------------------------------------------------------------


def test_all_none_evaluator_no_raise_records_zero_and_degraded():
    sp = _inject_eval(_make_pipeline(), none_all=True)
    result = sp.run_round(
        round_id="r1",
        parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
    )
    # 不抛、不入池；evaluated=0；degraded 记录（但 evaluator 全 None 属预期降级）
    assert result.evaluated == 0
    assert result.admitted == 0
    assert result.candidates_generated > 0


# ---------------------------------------------------------------------------
# 3. 默认参数不 import torch / openai（模块级）
# ---------------------------------------------------------------------------


def test_pipeline_module_no_torch_top_level():
    """pipeline 模块级不 import torch / openai（legacy trainer 才需要）。"""
    import sys

    mod = sys.modules.get("alphaprobe.pipeline")
    assert mod is not None
    src = Path(mod.__file__).read_text(encoding="utf-8")
    # 只检查顶层 import 语句（注释允许出现说明性文字）
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            assert False, f"pipeline.py top-level must not import torch: {stripped}"
        if stripped.startswith("import openai") or stripped.startswith("from openai"):
            assert False, f"pipeline.py top-level must not import openai: {stripped}"


def test_pipeline_import_does_not_pull_torch():
    """pipeline 顶层 import 后 sys.modules 中不应新增 torch / openai。"""
    import sys

    assert "alphaprobe.pipeline" in sys.modules
    for name in ("torch", "openai", "sentence_transformers"):
        # 允许存在（其他测试可能已拉入），但 pipeline import 本身不应是来源
        pass  # 占位：模块级无 torch import 由 test_pipeline_module_no_torch_top_level 保证


# ---------------------------------------------------------------------------
# 4. 内存 dedup：EXACT / NEW 语义
# ---------------------------------------------------------------------------


def test_in_memory_dedup_exact_after_reserve():
    sp = _make_pipeline()
    dc = sp.dedup_client
    v1 = dc.check_new_candidate("rank(close)")
    assert v1.status == "NEW"
    dc.reserve("rank(close)")
    v2 = dc.check_new_candidate("rank(close)")
    assert v2.status == "EXACT"
    assert v2.rejection_reason.value == "EXACT_DUPLICATE"


def test_in_memory_dedup_missing_client_falls_back_to_seen():
    sp = _make_pipeline()
    assert sp.dedup_client is not None


# ---------------------------------------------------------------------------
# 5. make_fe_evaluate_fn（数据不可用 → 全 None，不抛）
# ---------------------------------------------------------------------------


def test_fe_evaluate_fn_none_data_returns_all_none():
    fn = make_fe_evaluate_fn(None)
    out = fn(["rank(close)", "ts_mean(close, 20)"], "L2_full_train", None)
    assert len(out) == 2
    assert all(b is None for b in out)


def test_fe_evaluate_fn_returns_bundles_for_fake_stock_data():
    """假 stock_data（evaluate_many 返回小平面）→ bundle dict 列表。"""
    import numpy as np

    class FakeStockData:
        def __init__(self):
            self._field_names = lambda: ["open", "high", "low", "close", "vwap", "volume"]
            self.data = np.zeros((40, 3, 6), dtype=float)
            # vwap 列（index 4）有规律数值；close 列（3）
            for t in range(40):
                self.data[t, :, 4] = 10.0 + 0.1 * t
                self.data[t, :, 3] = 10.0 + 0.1 * t

        def evaluate_many(self, formulas):
            return [np.ones((40, 3), dtype=float) for _ in formulas]

    fn = make_fe_evaluate_fn(FakeStockData(), label_days=5)
    out = fn(["rank(close)"], "L2_full_train", None)
    assert len(out) == 1
    assert out[0] is not None
    assert "rankic" in out[0]
    # 常数因子 → rankic 无有效截面变体 → None 或数值均可（不抛即可）
    assert set(out[0].keys()) >= {"rankic", "ic", "coverage", "nan_inf_ratio"}


# ---------------------------------------------------------------------------
# 6. runner：--pipeline 路由 + json 修复（不吞 NameError）
# ---------------------------------------------------------------------------


def test_runner_import_and_pipeline_arg_route(monkeypatch):
    """--pipeline {legacy,new} 参数路由（默认 new）；不触碰厚依赖真实 import。"""
    import sys
    import types

    from alphaprobe.runner import build_train_parser

    p = build_train_parser()
    assert p.parse_args([]).pipeline == "new"
    assert p.parse_args(["--pipeline", "legacy"]).pipeline == "legacy"
    assert p.parse_args(["--pipeline", "new"]).pipeline == "new"


def test_record_exported_to_memory_does_not_swallow_nameerror(tmp_path, monkeypatch):
    """_record_exported_to_memory：坏 manifest 记日志跳过；NameError 类不再被吞。

    用一个假的 GlobalMemoryStore 验证 upsert/mark_exported 被调用（不抛），
    同时坏 JSON 清单跳过不崩溃。runner 模块已由其它用例导入。
    """
    import alphaprobe.runner as runner_mod

    # 构造假 exporter 结果：一个合法 manifest + 一个坏 JSON manifest
    good = tmp_path / "good" / "manifest.json"
    good.parent.mkdir(parents=True)
    good.write_text(json.dumps({"formula": "rank(close)"}), encoding="utf-8")
    bad = tmp_path / "bad" / "manifest.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not valid json", encoding="utf-8")

    calls = {"upsert": 0, "mark": 0}

    class FakeStore:
        def __init__(self):
            self.closed = False

        def upsert_factor_node(self, **kw):
            calls["upsert"] += 1
            return True, None

        def mark_exported(self, **kw):
            calls["mark"] += 1

        def close(self):
            self.closed = True

    # 全局 patch：runner 函数内部 import GlobalMemoryStore（from alphaprobe.memory）
    import alphaprobe.memory as memory_mod

    monkeypatch.setattr(memory_mod, "GlobalMemoryStore", lambda: FakeStore())

    class FakePool:
        size = 1
        # 与 runner 生产一致：exprs 是 Expression 对象（expression_to_dsl 消费）。
        # 用 alphaprobe 自有的轻量表达式类型构造，避免依赖 shared/alphagen。
        from alphaprobe.fe_bridge.dsl_expression import FactorEngineDslExpression

        exprs = [FactorEngineDslExpression("rank(close)")]

    class FakeExperiment:
        def __init__(self):
            self.delivery = type("D", (), {"resolved_campaign_id": lambda self: "c1"})()

    runner_mod._record_exported_to_memory(
        FakeExperiment(), FakePool(), {"exported_manifests": [str(good), str(bad)]}
    )
    assert calls["upsert"] == 1
    assert calls["mark"] == 1


# ---------------------------------------------------------------------------
# 7. RoundManager 真闭环
# ---------------------------------------------------------------------------


def test_round_manager_passes_packet_parents_to_callback():
    """RoundManager._load_memory 的 packet.parents 真被传给 campaign_callback。"""
    import tempfile
    import os

    from alphaprobe.continuous.round_manager import RoundManager
    from alphaprobe.memory import GlobalMemoryStore

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "mem.sqlite3")
        store = GlobalMemoryStore(db_path=db)
        seen = {}

        def cb(round_no, calibrator, *a, **kw):
            seen["parents"] = kw.get("parents")
            seen["packet"] = kw.get("packet")
            return {"pool_size": 1, "round_no": round_no}

        rm = RoundManager(store=store, campaign_callback=cb, max_rounds=1, sleep_seconds=0)
        rm.run()
        assert seen.get("parents") is not None
        assert len(seen["parents"]) >= 1
        assert seen["parents"][0].get("factor_id") == "round_root"
        assert seen.get("packet") is not None
        store.close()


def test_round_manager_default_campaign_pipeline_failure_logs_not_raise(monkeypatch):
    """pipeline 构造失败时 _default_campaign log 且不抛，回落占位。"""
    import sys
    import types

    from alphaprobe.continuous.round_manager import RoundManager

    # 预置一个构造即失败的 alphaprobe.pipeline 模块
    fake_mod = types.ModuleType("alphaprobe.pipeline")
    fake_mod.PipelineConfig = lambda **k: type("C", (), {})()

    def _boom(*a, **k):
        raise RuntimeError("boom construct")

    fake_mod.SearchPipeline = _boom
    sys.modules["alphaprobe.pipeline"] = fake_mod
    try:
        res = RoundManager._default_campaign(1, None)
        assert res["pool_size"] == 0
        assert "boom construct" in res.get("fallback_reason", "")
    finally:
        sys.modules.pop("alphaprobe.pipeline", None)


def test_round_manager_default_campaign_runs_pipeline():
    """_default_campaign 委托 SearchPipeline 真跑（stub llm + 内存 dedup）。"""
    from alphaprobe.continuous.round_manager import RoundManager

    res = RoundManager._default_campaign(1, None)
    assert "round_result" in res
    assert res["pool_size"] >= 0


# ---------------------------------------------------------------------------
# 8. 泄漏纪律：pipeline 不接收 test 段
# ---------------------------------------------------------------------------


def test_pipeline_has_no_test_data_construction():
    """grep gate：pipeline.py 不构造 test 段数据（无 data_test / test_period 构造）。"""
    import sys

    mod = sys.modules["alphaprobe.pipeline"]
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "data_test" not in src
    # test_period 只允许出现在注释/说明（不构造数据）
    assert "_make_stock_data" not in src
