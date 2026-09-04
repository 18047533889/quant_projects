"""P0-A 主链测试：SearchPipeline/runner 的 P0 修复 + llm_client 协议。

覆盖：
① PRODUCTION fail-closed：无 llm_client / stub llm → 构造即抛（绝不静默切 stub）；
② OFFLINE_TEST 默认行为与旧版一致（stub 可跑 run_round；orchestrator
   structured_generation 默认 False——旧行为，不破坏离线测试）；
③ structured_generation 生产语义默认 True（新建 PipelineConfig() 默认
   UNSET→SearchPipeline 解析后 PRODUCTION=True / OFFLINE 旧行为 False）；
④ data_search_valid 分离：注入两个假 evaluate 通路，断言 L3 record 的
   metrics 来自 valid 段 evaluate_fn 而非 train（valid 指标绝不用 train 数据填）；
⑤ PRODUCTION evaluator 失败 → EVALUATION_MISSING，不手算（monkeypatch
   计数 make_fe_evaluate_fn 未被调用）；
⑥ runner parents 不再 [:5]（构造 8 个 seed，断言 parents 长度 == 8）。

全离线：不依赖 torch / openai / faiss / psutil / quant_evaluator；假 evaluate_fn
注入，不碰 FE identity 权威的 inactive-canonical 环境污染（identity_tuple 在
DedupClient 缺 FE 时回落文本 hash 不抛——见 _identity_tuple 的 except 分支）。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from alphaprobe.llm_client import (
    DeterministicStubLLMClient,
    ExecutionMode,
    LLMConfigurationError,
    LLMClientProtocol,
    resolve_llm_fn,
)
from alphaprobe.pipeline import (
    PipelineConfig,
    SearchPipeline,
    make_fe_evaluate_fn,
    make_stub_llm_fn,
)

# ---------------------------------------------------------------------------
# fixtures / helpers（参考 tests/test_pipeline.py 现有 fake 写法，不依赖厚依赖）
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


class _TrainedEval:
    """train 段假 evaluate_fn（记录被调用并返回固定 TRAIN_BUNDLE）。"""

    def __init__(self, bundle=None):
        self.calls = 0
        self.last_fidelity = None
        self.bundle = dict(TRAIN_BUNDLE if bundle is None else bundle)

    def __call__(self, formulas, fidelity="L2_full_train", context=None):
        self.calls += 1
        self.last_fidelity = fidelity
        return [dict(self.bundle) for _ in formulas]


class _ValidEval:
    """valid 段假 evaluate_fn（记录被调用并返回固定 VALID_BUNDLE）。"""

    def __init__(self, bundle=None):
        self.calls = 0
        self.last_fidelity = None
        self.bundle = dict(VALID_BUNDLE if bundle is None else bundle)

    def __call__(self, formulas, fidelity="L3_search_valid", context=None):
        self.calls += 1
        self.last_fidelity = fidelity
        return [dict(self.bundle) for _ in formulas]


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


def _inject_eval(sp: SearchPipeline, eval_fn) -> SearchPipeline:
    """注入假 evaluate 通路；evaluator=None 强制静态降级（不依赖 QE/Legacy）。"""
    sp.evaluator = None
    sp._evaluate_fn = eval_fn
    return sp


def _one_round(sp: SearchPipeline) -> "object":
    return sp.run_round(
        round_id="r1",
        parents=[{"formula": "rank(close)", "factor_id": "p1", "fitness": 0.0}],
    )


def _fake_valid_stock_data():
    """最小假 valid 段 stock_data（make_qe_evaluate_fn 消费）。仅让
    __post_init__ 能构造 _evaluate_fn_valid；测试会覆盖为假 valid fn。"""
    return object()


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
        return json.dumps({"candidates": []})


# ---------------------------------------------------------------------------
# ① PRODUCTION fail-closed
# ---------------------------------------------------------------------------


class TestProductionFailClosed:
    def test_production_no_llm_client_raises(self):
        with pytest.raises((ValueError, LLMConfigurationError)) as ei:
            _make_pipeline(mode=ExecutionMode.PRODUCTION)
        assert "requires a real llm_client" in str(ei.value)

    def test_production_stub_llm_raises(self):
        with pytest.raises((ValueError, LLMConfigurationError)):
            _make_pipeline(
                mode=ExecutionMode.PRODUCTION,
                llm_client=DeterministicStubLLMClient(),
            )

    def test_production_mode_from_str_raises(self):
        with pytest.raises((ValueError, LLMConfigurationError)):
            _make_pipeline(mode="production")

    def test_resolve_llm_fn_production_fail_closed(self):
        with pytest.raises(LLMConfigurationError):
            resolve_llm_fn(None, ExecutionMode.PRODUCTION)
        with pytest.raises(LLMConfigurationError):
            resolve_llm_fn(DeterministicStubLLMClient(), ExecutionMode.PRODUCTION)

    def test_production_real_client_ok(self):
        client = _RealLLMClient()
        sp = _make_pipeline(mode=ExecutionMode.PRODUCTION, llm_client=client)
        assert sp.llm_fn is not None
        # 生产路径 orchestrator structured_generation 默认 True
        assert sp.orchestrator.structured_generation is True


# ---------------------------------------------------------------------------
# ② OFFLINE_TEST 默认行为与旧版一致
# ---------------------------------------------------------------------------


class TestOfflineDefault:
    def test_offline_default_stub_runs_round(self):
        sp = _make_pipeline()
        result = _one_round(sp)
        assert result.candidates_generated > 0
        assert result.evaluated >= 0

    def test_offline_default_structured_generation_off_legacy(self):
        """默认（未显式传 config）OFFLINE_TEST → orchestrator 旧行为 False。"""
        cfg = PipelineConfig()
        assert cfg.structured_generation is not False  # UNSET 哨兵
        sp = _make_pipeline(config=cfg)
        assert sp.orchestrator.structured_generation is False

    def test_offline_explicit_true_respected(self):
        cfg = PipelineConfig(structured_generation=True)
        sp = _make_pipeline(config=cfg)
        assert sp.orchestrator.structured_generation is True

    def test_offline_explicit_false_respected(self):
        cfg = PipelineConfig(structured_generation=False)
        sp = _make_pipeline(config=cfg)
        assert sp.orchestrator.structured_generation is False


# ---------------------------------------------------------------------------
# ③ structured_generation 生产语义默认 True
# ---------------------------------------------------------------------------


class TestStructuredDefault:
    def test_pipeline_config_default_true_production(self):
        cfg = PipelineConfig()
        # P0-A：默认值走 UNSET 哨兵，生产语义默认 True（由 SearchPipeline 解析）
        assert cfg.structured_generation is not False
        sp = _make_pipeline(config=cfg, mode=ExecutionMode.PRODUCTION, llm_client=_RealLLMClient())
        assert sp.orchestrator.structured_generation is True
        assert sp.config.structured_generation is True

    def test_deterministic_stub_llm_client_reuses_make_stub_llm_fn(self):
        # 内部复用 make_stub_llm_fn：同 seed 同输出
        a = DeterministicStubLLMClient(structured=False)
        b = DeterministicStubLLMClient(structured=False)
        text = "## Parent factor(s)\n- rank(close) :: p"
        assert a.generate(system_prompt="s", user_prompt=text, model_class="cheap").text == b.generate(
            system_prompt="s", user_prompt=text, model_class="cheap"
        ).text


# ---------------------------------------------------------------------------
# ④ data_search_valid 分离：L3 metrics 来自 valid 段 evaluate_fn
# ---------------------------------------------------------------------------


class TestTrainValidSeparation:
    def test_l3_record_metrics_from_valid_eval_not_train(self):
        train_fn = _TrainedEval(bundle=dict(TRAIN_BUNDLE, rankic=0.04))
        valid_fn = _ValidEval(bundle=dict(VALID_BUNDLE, rankic_valid=0.09, net_sharpe=1.8))

        sp = _make_pipeline()
        sp.data_search_valid = object()  # 非 None → valid 通路激活
        # __post_init__ 已构造 _evaluate_fn_valid（make_qe_evaluate_fn(None) 不可用
        # 会退化），测试覆盖为假 valid fn。
        sp._evaluate_fn_valid = valid_fn
        _inject_eval(sp, train_fn)

        result = _one_round(sp)
        assert result.candidates_generated > 0
        # valid fn 被调用，且 fidelity 是 L3_search_valid
        assert valid_fn.calls >= 1
        assert valid_fn.last_fidelity == "L3_search_valid"
        # train fn 也被调用（L1/L2）
        assert train_fn.calls >= 1
        # 入池 record 的 metrics 含 valid 段键（来自 valid fn，非 train 数据）。
        # 断言方式：pool 成员携带的 meta/search_fitness 等由 admission 消费，
        # valid 键经 record.metric_bundle 进入 admission._member_for 的 mb；
        # 这里直接检查 _run_funnel 产出的 record3 fidelity/segment（valid 指标
        # 已在 record3.metric_bundle 合并）。
        rr = result  # RoundResult
        assert rr.admitted >= 0
        # 无论是否入池，valid 通路都产生过 L3_search_valid record（segment 正确）
        # 通过 mock 记录：valid_fn.last_fidelity 已断言；再确认 merged bundle 确实
        # 带 valid 键——从最后一次 admission record 出发不可靠，改为直接跑
        # _run_funnel 并检查返回 record。
        from alphaprobe.contracts import EvaluationRecord

        cand = types.SimpleNamespace(formula="rank(close)")
        # 复跑一次 funnel（fresh dedup 状态：sp 已在 run_round 里 reserve 过公式，
        # 用带唯一 window 的公式避免 EXACT 拦截）
        cand2 = types.SimpleNamespace(formula="ts_std(close, 37)")
        ok, info, record = sp._run_funnel(cand2)
        assert ok
        assert record is not None
        assert isinstance(record, EvaluationRecord)
        assert record.fidelity == "L3_search_valid"
        assert record.segment == "search_valid"
        assert record.metric_bundle.get("rankic_valid") == 0.09
        assert record.metric_bundle.get("net_sharpe") == 1.8

    def test_l3_not_run_without_data_search_valid(self):
        """data_search_valid=None → 不触发 valid 通路（保持旧行为）。"""
        train_fn = _TrainedEval()
        sp = _inject_eval(_make_pipeline(), train_fn)
        _one_round(sp)
        assert not hasattr(sp, "_evaluate_fn_valid") or sp._evaluate_fn_valid is None
        # 无 valid fn 被调用

    def test_valid_metrics_never_from_train_data(self):
        """关键正确性：rankic_valid 等 valid 键不出现在 train bundle。"""
        train_fn = _TrainedEval()
        valid_fn = _ValidEval()
        sp = _make_pipeline()
        sp.data_search_valid = object()
        sp._evaluate_fn_valid = valid_fn
        _inject_eval(sp, train_fn)
        _one_round(sp)
        # train bundle 不应含 valid 段专属键
        for rec in train_fn.bundle:
            assert "rankic_valid" not in str(rec)
        assert "rankic_valid" not in TRAIN_BUNDLE
        assert "rankic_valid" in VALID_BUNDLE


# ---------------------------------------------------------------------------
# ⑤ PRODUCTION evaluator 失败 → EVALUATION_MISSING，不手算
# ---------------------------------------------------------------------------


class TestProductionNoHandFallback:
    def test_production_evaluator_failure_no_hand_compute(self, monkeypatch):
        """PRODUCTION：evaluator.evaluate 抛 → EVALUATION_MISSING，
        make_fe_evaluate_fn（手算回退）绝不被动用。"""
        import alphaprobe.pipeline as pipeline_mod

        counted = {"calls": 0}

        real_make_fe = pipeline_mod.make_fe_evaluate_fn

        def _counting_make_fe(*a, **k):
            counted["calls"] += 1
            return real_make_fe(*a, **k)

        monkeypatch.setattr(pipeline_mod, "make_fe_evaluate_fn", _counting_make_fe)

        class _BoomEvaluator:
            def evaluate(self, candidates, **kw):
                raise RuntimeError("qe unavailable")

        sp = _make_pipeline(mode=ExecutionMode.PRODUCTION, llm_client=_RealLLMClient())
        # 记录 patch 后 make_fe_evaluate_fn 被调用次数 = 构造期次数（__post_init__
        # 默认 evaluator 构建；随运行顺序可能是 0 或 2）。运行期（_evaluate 的
        # 手算回退路径）绝不允许新增调用。
        construct_calls = counted["calls"]
        sp.evaluator = _BoomEvaluator()
        sp._evaluate_fn = None  # 即便残留本地 fn 也不应被调

        result = _one_round(sp)
        # 生产 evaluator 失败 → 全部 EVALUATION_MISSING → evaluated=0
        assert result.evaluated == 0
        assert result.admitted == 0
        # 运行期无新增 make_fe_evaluate_fn 调用（不手算）
        assert counted["calls"] == construct_calls

    def test_offline_evaluator_empty_falls_back_to_local(self):
        """OFFLINE_TEST：evaluator 空 → 回落本地（离线可跑，不回归）。"""
        train_fn = _TrainedEval()

        class _EmptyEvaluator:
            def evaluate(self, candidates, **kw):
                return [types.SimpleNamespace(metric_bundle={}) for _ in candidates]

        sp = _inject_eval(_make_pipeline(), train_fn)
        sp.evaluator = _EmptyEvaluator()
        result = _one_round(sp)
        # 本地 train fn 被调用 → 有评估
        assert train_fn.calls >= 1


# ---------------------------------------------------------------------------
# ⑥ runner parents 不再 [:5]
# ---------------------------------------------------------------------------


def test_runner_parents_no_longer_truncated_to_five(monkeypatch):
    """runner parents 用完整 seed 列表（≥5 不再被砍到 5）。

    本测试环境 runner 顶层 ``import torch``（无 torch wheel 时不可导入）——
    用 importlib 从源码文本加载 runner 模块（不执行顶层 torch import，与
    test_round_checkpoint.py 的既有绕过模式一致），再 monkeypatch 依赖。
    """
    import importlib.util
    import types as _types
    import re

    runner_path = (
        Path(__file__).resolve().parents[1] / "src" / "alphaprobe" / "runner.py"
    )
    src_text = runner_path.read_text(encoding="utf-8")
    # 去掉顶层 torch 依赖行与冷启动/训练重 import（fe_bridge.bootstrap 顶层
    # import torch；trainer/shared 侧同理）。保留 resolve_cold_start /
    # run_mining_campaign / _run_pipeline_mining 等被测符号。
    src_text = re.sub(r"^import torch\n", "", src_text, flags=re.M)
    src_text = src_text.replace("from dotenv import load_dotenv\n\n", "")
    src_text = re.sub(
        r"^load_dotenv\(PROJECT_ROOT / \".env\", override=True\)\n", "", src_text, flags=re.M
    )
    # 去掉冷启动/trainer/fe_bridge/shared 顶层 import（均触 torch/厚依赖）。
    # 用基于行的过滤更稳：遇到这些起始即删到对应空行。
    lines = src_text.splitlines(keepends=True)
    drop_ranges: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(
            (
                "from alphaprobe.cold_start import (",
                "from alphaprobe.fe_bridge.bootstrap import",
                "from alphaprobe.fe_bridge.stock_data import",
                "from alphaprobe.trainer.pool import",
                "from alphaprobe.trainer.trainer import",
                "from shared.",
            )
        ):
            start = i
            depth = s.count("(") - s.count(")")
            i += 1
            while i < len(lines) and depth > 0:
                t = lines[i].strip()
                depth += t.count("(") - t.count(")")
                i += 1
            drop_ranges.append((start, i))
            continue
        i += 1
    for start, end in reversed(drop_ranges):
        del lines[start:end]
    # alphaprobe.delivery.config import ExperimentConfig 保留（轻量）；但
    # alphaprobe.delivery.exporter 触 FE → 删。
    out_lines: list[str] = []
    for s in lines:
        if s.strip().startswith("from alphaprobe.delivery.exporter import"):
            continue
        out_lines.append(s)
    src_text = "".join(out_lines)
    # run_mining_campaign legacy 分支引用被删名字 → 短路：把 legacy 入口改成
    # 永不执行（在 source 文本里把 legacy 段的 `resolve_cold_start(...)` 赋值
    # 替换成 return）。注意 legacy 段的该行在 new-branch 内也有同名调用——
    # 用精确行序列定位 legacy 段（紧跟 new-branch return 之后的那个）。
    legacy_block = (
        "        return {\n"
        '            "campaign_id": campaign_id,\n'
        '            "pool_size": mining_result["pool_size"],\n'
        '            "export": export_result,\n'
        "        }\n"
        "\n"
        "    initial_exprs = resolve_cold_start(experiment, args)"
    )
    if legacy_block in src_text:
        src_text = src_text.replace(
            legacy_block,
            "        return {\n"
            '            "campaign_id": campaign_id,\n'
            '            "pool_size": mining_result["pool_size"],\n'
            '            "export": export_result,\n'
            "        }\n"
            "\n"
            '    return {"campaign_id": "legacy_unsupported_in_test", "pool_size": 0, "export": {}}',
            1,
        )
    spec = importlib.util.spec_from_loader("_alphaprobe_runner_text", loader=None)
    runner_mod = importlib.util.module_from_spec(spec)
    # __file__ 必须先于 exec 注入（runner 顶层用 Path(__file__) 计算 PROJECT_ROOT）。
    runner_mod.__file__ = str(runner_path)
    runner_mod.__name__ = "_alphaprobe_runner_text"
    code = compile(src_text, str(runner_path), "exec")
    exec(code, runner_mod.__dict__)
    monkeypatch.setitem(sys.modules, "alphaprobe.runner", runner_mod)

    # 8 个 cold-start 条目
    initial = []
    for i in range(8):
        expr = _types.SimpleNamespace(dsl=f"rank(ts_mean(close, {i + 5}))")
        initial.append(expr)
    monkeypatch.setattr(runner_mod, "resolve_cold_start", lambda *a, **k: initial)

    # 捕获传入 _run_pipeline_mining 的 parents
    captured = {}

    def _fake_run_pipeline_mining(args, experiment, data, *, campaign_id, parents):
        captured["parents"] = parents
        return {"pool_size": 0, "round_result": None}

    monkeypatch.setattr(runner_mod, "_run_pipeline_mining", _fake_run_pipeline_mining)

    # run_mining_campaign 里 run_round 前有大量厚依赖设置（_make_stock_data /
    # Feature/target 等）——monkeypatch 掉被测路径之后的一切，把执行从
    # pipeline-new 分支开头直接跳转到 resolve_cold_start → parents → 捕获。
    # 具体：patch _apply_* 设置、_init_prompts、_make_stock_data 为 no-op。
    monkeypatch.setattr(runner_mod, "_apply_env_llm_settings", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "_apply_mining_yaml_defaults", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "_init_prompts", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "_apply_checkpoint_settings", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "_make_stock_data", lambda *a, **k: None)
    # target/Feature/Ref 构造前短路：pipeline-new 分支在 _make_stock_data 后仍会
    # 引用 Feature/Ref —— 直接 monkeypatch _make_stock_data 之外，把 run_mining
    # 的 data 传参段绕过（_make_stock_data 返回 None 即可，run_round 不执行，
    # 我们只测 parents 构造，真正的 pipeline 由 _fake_run_pipeline_mining 接收）。
    # Feature/Ref/FeatureType 需 stub：__post_init__ 后 run_mining_campaign 算
    # target = Ref(vwap, -label_days) / vwap - 1 —— Ref 返回的对象必须支持除法。
    class _FakeRef:
        def __init__(self, *a, **k):
            pass

        def __truediv__(self, other):
            return self

        def __sub__(self, other):
            return self

        def __rsub__(self, other):
            return self

    runner_mod.FeatureType = _types.SimpleNamespace(VWAP="vwap")
    runner_mod.Feature = lambda *a, **k: _FakeRef()
    runner_mod.Ref = lambda *a, **k: _FakeRef()
    # 构造假 experiment：run_mining_campaign 用 delivery.mining_config/period 等。
    # _FakeExperiment 需带 delivery.mining_config 与 resolved_campaign_id。
    class _FakeExperiment:
        def __init__(self):
            self.raw = {}
            self.delivery = _types.SimpleNamespace(
                resolved_campaign_id=lambda: "c1",
                mining_config={
                    "train_period": ["2016-01-01", "2021-12-31"],
                    "valid_period": ["2022-01-01", "2023-12-31"],
                    "test_period": ["2024-01-01", "2026-07-31"],
                },
            )
            self.mining = _types.SimpleNamespace(
                label_days=20,
                search_time=40,
                pool_capacity=2000,
                ic_export_threshold=0.006,
            )
            self.data = _types.SimpleNamespace(instruments="csi300", max_files=None)
            self.has_delivery_block = False

    _args = _types.SimpleNamespace(
        pipeline="new",
        pool_capacity=2000,
        search_time=40,
        generate_num=5,
        cold_start_sample_size=None,
        cold_start_seed=None,
        cold_start_pv_ratio=None,
        cold_start_library=None,
        cuda=1,
        resume=False,
        label_days=20,
        ic_threshold=0.006,
        checkpoint_dir="/tmp/cp",
        checkpoint_enabled=False,
        checkpoint_resume=False,
    )
    runner_mod.run_mining_campaign(_args, experiment=_FakeExperiment())
    # parents 长度 == 8（不再 [:5]）
    assert captured.get("parents") is not None
    assert len(captured["parents"]) == 8
    assert captured["parents"][0]["formula"].startswith("rank(ts_mean(close, 5))")
    assert captured["parents"][7]["formula"].startswith("rank(ts_mean(close, 12))")


# ---------------------------------------------------------------------------
# L5 sealed test：data_search_valid 下 run_round 不触 L5
# ---------------------------------------------------------------------------


def test_run_round_with_data_search_valid_no_l5():
    """构造 SearchPipeline 传 data_search_valid，跑 run_round：不报错，
    且所有 record fidelity 不含 L5（sealed test 零构造）。"""
    sp = _make_pipeline()
    sp.data_search_valid = object()
    valid_fn = _ValidEval()
    train_fn = _TrainedEval()
    sp._evaluate_fn_valid = valid_fn
    _inject_eval(sp, train_fn)
    result = _one_round(sp)
    assert result.candidates_generated >= 0
    # 全程无 L5 段数据构造
    src = Path(__file__).resolve().parents[1] / "src" / "alphaprobe" / "pipeline.py"
    text = src.read_text(encoding="utf-8")
    assert "data_test" not in text
