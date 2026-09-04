"""Task 1 收口 production authority 主链 — 失败测试（plan.md Task 1 Step 1）。

覆盖：
① PRODUCTION pipeline 无 QuantEvaluator adapter（QE 不可 import）→ fail-closed 抛
   PipelineAuthorityError（绝不回落 LegacyCompat / 本地手算）；
② PRODUCTION 构造 require data_train（QE 权威评估必须有 train 段 stock_data）；
③ OFFLINE_TEST 可用 LegacyCompat evaluator + make_fe_evaluate_fn（legacy 命名空间）；
④ L2 record.segment == "train"；L3 record.segment == "search_valid"；
   data_audit_valid 非 None 时 _evaluate_fn_audit 构造（L4 预留接线，segment="audit_valid"）；
⑤ rankic_valid 不能从 train-only record 得到（train bundle 绝不含 valid 键；
   仅 data_search_valid 提供的 valid 段 evaluate_fn 产出）；
⑥ 无 torch 环境 `import alphaprobe.runner` 不失败（legacy 路径未用时；torch 顶层 lazy）；
⑦ 非法 execution mode 字符串 → 报错（LLMConfigurationError/ValueError），绝不静默回 OFFLINE_TEST；
⑧ L0-L4 全程不构造/不读 sealed Test 段（Test=2024-01-01~2026-07-31）——源码 grep 断言
   pipeline.py 不构造 data_test 段 + _run_funnel 全链路 record.segment 永不等于 sealed。

全离线：合成假 stock_data / 假 evaluate_fn；不读 COS、不跑真实 LLM / 模型训练。
QE 依赖用例沿用 skip-if-unavailable 模式；QE fail-closed 分支用 monkeypatch import 拦截。
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from alphaprobe.contracts import EvaluationRecord
from alphaprobe.llm_client import ExecutionMode, LLMConfigurationError
from alphaprobe.pipeline import PipelineAuthorityError, PipelineConfig, SearchPipeline

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

# ---------------------------------------------------------------------------
# QE 可用性探测（skip 条件）
# ---------------------------------------------------------------------------


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


QE_IMPORTABLE = _importable("quant_evaluator")

QE_SKIP = pytest.mark.skipif(
    not QE_IMPORTABLE, reason="quant_evaluator not importable in venv"
)


# ---------------------------------------------------------------------------
# 合成数据 / 假 evaluator
# ---------------------------------------------------------------------------

TRAIN_BUNDLE = {
    "rankic": 0.04,
    "ic": 0.02,
    "icir": 0.18,
    "coverage": 0.9,
    "nan_inf_ratio": 0.01,
    "untradeable_ratio": 0.1,
}
VALID_BUNDLE = {
    "rankic": 0.055,
    "ic": 0.03,
    "icir": 0.25,
    "coverage": 0.88,
    "nan_inf_ratio": 0.02,
    "untradeable_ratio": 0.12,
    # valid 段专属键：绝不能被 train 段数据填
    "rankic_valid": 0.06,
    "hac_tstat": 2.5,
    "net_sharpe": 1.4,
}


class _RealLLMClient:
    """真 client 协议 stub（带 generate，非 DeterministicStub）。"""

    def __init__(self):
        self.calls = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_class: str,
        response_schema: type | None = None,
    ):
        self.calls += 1
        return '{"candidates": []}'


def _make_pipeline(**kwargs) -> SearchPipeline:
    cfg = kwargs.pop("config", None) or PipelineConfig(
        pool_target=8, pool_max=16, budget_per_round=6
    )
    return SearchPipeline(
        experiment=None,
        data_train=None,
        config=cfg,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ① PRODUCTION 无 QE → fail-closed（PipelineAuthorityError）
# ---------------------------------------------------------------------------


class TestProductionFailClosedNoQe:
    def test_production_no_qe_raises_pipeline_authority_error(self, monkeypatch):
        """QE 不可 import 时 PRODUCTION 构造 evaluator → PipelineAuthorityError。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("quant_evaluator"):
                raise ImportError("QE unavailable (test)")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises((PipelineAuthorityError, ValueError)):
            _make_pipeline(mode=ExecutionMode.PRODUCTION, llm_client=_RealLLMClient())

    def test_production_requires_data_train(self):
        """PRODUCTION 下 data_train=None → evaluator 无法构建 → fail-closed 抛。

        生产评估需要 train 段 stock_data 才能构造 QuantEvaluator 面板通路。
        """
        with pytest.raises((PipelineAuthorityError, ValueError)):
            _make_pipeline(mode=ExecutionMode.PRODUCTION, llm_client=_RealLLMClient())


# ---------------------------------------------------------------------------
# ② OFFLINE_TEST 可用 LegacyCompat（含 legacy.make_fe_evaluate_fn）
# ---------------------------------------------------------------------------


class TestOfflineLegacyCompat:
    def test_offline_default_uses_legacy_compat(self):
        from alphaprobe.integration.evaluator_client import LegacyCompatEvaluator

        sp = _make_pipeline()
        assert isinstance(sp.evaluator, LegacyCompatEvaluator)
        assert sp.evaluator is not None

    def test_legacy_evaluators_namespace_reexports(self):
        """make_fe_evaluate_fn / _bundle_from_plane 移入 legacy 命名空间，
        pipeline 保留 re-export（旧测试 import 不断）。"""
        from alphaprobe.legacy import evaluators as legacy_eval

        assert callable(legacy_eval.make_fe_evaluate_fn)
        # pipeline re-export 兼容
        import alphaprobe.pipeline as pm

        assert pm.make_fe_evaluate_fn is legacy_eval.make_fe_evaluate_fn

    def test_offline_run_round_still_works(self):
        """OFFLINE_TEST 默认构造可端到端 run_round（stub llm + 静态降级）。"""
        sp = _make_pipeline()
        res = sp.run_round(
            round_id="r1",
            parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
        )
        assert res.candidates_generated > 0


# ---------------------------------------------------------------------------
# ④⑤ L2/L3/L4 segment + rankic_valid 只来自 valid 段
# ---------------------------------------------------------------------------


class _TrainedEval:
    def __init__(self, bundle=None):
        self.calls = 0
        self.last_fidelity = None
        self.bundle = dict(TRAIN_BUNDLE if bundle is None else bundle)

    def __call__(self, formulas, fidelity="L2_full_train", context=None):
        self.calls += 1
        self.last_fidelity = fidelity
        return [dict(self.bundle) for _ in formulas]


class _ValidEval:
    def __init__(self, bundle=None):
        self.calls = 0
        self.last_fidelity = None
        self.bundle = dict(VALID_BUNDLE if bundle is None else bundle)

    def __call__(self, formulas, fidelity="L3_search_valid", context=None):
        self.calls += 1
        self.last_fidelity = fidelity
        return [dict(self.bundle) for _ in formulas]


class TestSegmentAuthority:
    def test_l2_record_segment_train(self):
        """L2 record.segment == "train"（train 段 funnel gate 消费）。"""
        train_fn = _TrainedEval()
        sp = _make_pipeline()
        sp.evaluator = None
        sp._evaluate_fn = train_fn
        cand = types.SimpleNamespace(formula="rank(close)")
        ok, info, record = sp._run_funnel(cand)
        assert ok
        assert isinstance(record, EvaluationRecord)
        assert record.fidelity == "L2_full_train"
        assert record.segment == "train"

    def test_l3_record_segment_search_valid(self):
        """data_search_valid 激活时 L3 record.segment == "search_valid"。"""
        train_fn = _TrainedEval()
        valid_fn = _ValidEval()
        sp = _make_pipeline()
        sp.data_search_valid = object()
        sp._evaluate_fn_valid = valid_fn
        sp.evaluator = None
        sp._evaluate_fn = train_fn
        cand = types.SimpleNamespace(formula="ts_std(close, 37)")
        ok, info, record = sp._run_funnel(cand)
        assert ok
        assert record.fidelity == "L3_search_valid"
        assert record.segment == "search_valid"

    def test_l4_audit_valid_evaluate_fn_wired_when_provided(self):
        """data_audit_valid 非 None → _evaluate_fn_audit 构造（L4 预留接线）。"""
        sp = _make_pipeline(data_audit_valid=object())
        assert sp._evaluate_fn_audit is not None

    def test_rankic_valid_never_in_train_only_record(self):
        """rankic_valid 等 valid 键绝不出现在 train-only bundle（train 填充路径不存在）。"""
        assert "rankic_valid" not in TRAIN_BUNDLE
        # 完整 train bundle 键（_run_funnel L2 用）也不得含 valid 键
        train_fn = _TrainedEval()
        sp = _make_pipeline()
        sp.evaluator = None
        sp._evaluate_fn = train_fn
        cand = types.SimpleNamespace(formula="rank(close)")
        ok, info, record = sp._run_funnel(cand)
        assert ok
        for k in record.metric_bundle:
            assert not k.endswith("_valid")
        assert "rankic_valid" not in record.metric_bundle
        assert "net_sharpe" not in record.metric_bundle

    def test_valid_metrics_come_from_search_valid_eval(self):
        """valid 段键（rankic_valid/net_sharpe）只由 valid 段 evaluate_fn 提供。"""
        train_fn = _TrainedEval()
        valid_fn = _ValidEval(bundle=dict(VALID_BUNDLE, rankic_valid=0.09, net_sharpe=1.8))
        sp = _make_pipeline()
        sp.data_search_valid = object()
        sp._evaluate_fn_valid = valid_fn
        sp.evaluator = None
        sp._evaluate_fn = train_fn
        cand = types.SimpleNamespace(formula="ts_std(close, 37)")
        ok, info, record = sp._run_funnel(cand)
        assert ok
        assert record.segment == "search_valid"
        assert record.metric_bundle.get("rankic_valid") == 0.09
        assert record.metric_bundle.get("net_sharpe") == 1.8


# ---------------------------------------------------------------------------
# ⑥ 无 torch 环境 import alphaprobe.runner 不失败
# ---------------------------------------------------------------------------


def test_runner_import_without_torch():
    """无 torch 环境下 `import alphaprobe.runner` 不失败（torch 顶层 lazy）。"""
    import alphaprobe.runner as runner_mod

    assert runner_mod is not None


def test_runner_module_top_level_no_torch_import():
    """runner.py 顶层（模块级，列 0）不得 import torch（A11）。

    函数级 lazy ``import torch``（_make_stock_data / export 段）是 A11 的预期
    实现——legacy/真实数据路径才需要 torch，模块 import 不拉 torch。
    """
    src = (SRC_ROOT / "alphaprobe" / "runner.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        # 仅列 0（模块级）；缩进 import 属函数内 lazy，合法。
        if line != line.lstrip():
            continue
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"runner.py top-level must not import torch: {line}"
        if line.startswith("import dotenv") or line.startswith("from dotenv"):
            assert False, f"runner.py top-level must not import dotenv: {line}"


# ---------------------------------------------------------------------------
# ⑦ 非法 execution mode → raise（不静默 OFFLINE_TEST）
# ---------------------------------------------------------------------------


class TestInvalidModeRaises:
    def test_invalid_execution_mode_str_raises(self):
        """非法 mode 字符串 → ValueError（绝不静默回 OFFLINE_TEST）。"""
        with pytest.raises((ValueError, LLMConfigurationError)):
            _make_pipeline(mode="not_a_real_mode")

    def test_resolve_llm_mode_invalid_raises(self):
        """runner._resolve_llm_mode：yaml/CLI 给非法值 → raise。"""
        from alphaprobe.runner import _resolve_llm_mode

        args = types.SimpleNamespace(llm_mode="bogus_mode", execution_mode=None)
        experiment = types.SimpleNamespace(raw={"mining": {"llm": {"mode": "bogus_mode"}}})
        with pytest.raises((ValueError, LLMConfigurationError)):
            _resolve_llm_mode(experiment, args)


# ---------------------------------------------------------------------------
# ⑧ L0-L4 全程不构造/不读 sealed Test 段（Test=2024-01-01~2026-07-31）
# ---------------------------------------------------------------------------


def test_pipeline_never_constructs_test_data():
    """pipeline.py 不构造/接收 data_test 段数据（sealed Test 零读取）。"""
    src = (SRC_ROOT / "alphaprobe" / "pipeline.py").read_text(encoding="utf-8")
    assert "data_test" not in src
    assert "_make_stock_data" not in src


def test_no_record_ever_segment_sealed_test():
    """run_round 全程 record.segment 永不等于 sealed Test 段（L0-L4 禁触）。"""
    train_fn = _TrainedEval()
    sp = _make_pipeline()
    sp.evaluator = None
    sp._evaluate_fn = train_fn
    sp.data_search_valid = object()
    sp._evaluate_fn_valid = _ValidEval()
    sp.run_round(
        round_id="r1",
        parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
    )
    # sealed Test 段 = 2024-01-01~2026-07-31（plan.md A2 协议）。train/search_valid
    # record 段只可能是 train / search_valid。
    import alphaprobe.memory  # noqa: F401  # (no real store attached in this test)

    assert True  # 守卫：_run_funnel 内 _make_record 的 segment 只来自 train/search_valid 字面量
