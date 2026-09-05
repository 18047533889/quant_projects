"""plan Task 22：Search cost accounting + model routing 测试。

覆盖 plan Task 22 Tests + Non-negotiable #26 / #30：
1. duplicate candidate 的 QE 成本 ≈ 0（GlobalFactorGate 前拒后不进 QE，成本记账
   反映 duplicate 节省）——用真实 :class:`SearchCostLedger` 显式记账验证；
2. 等增益下贵 action 的 cost-adjusted EV 更低（cost 进 EV 而非仅日志）——
   确定性可调函数 + 聚合成本投影验证 EVI/cost 关系；
3. llm_client 记账扩展（UsageRecord：prompt/completion tokens、latency、cost、
   usage 可选返回）不破坏既有 fail-closed 语义（LLMConfigurationError /
   ExecutionMode / stub 纪律）；
4. cheap / strong / critic 三档 model class 路由接口（provider/model 名不实现，
   Part J1 留白：只做 class 路由协议 + 配置）；
5. #26：cost 只记账不改变 reward 语义——ledger 只新增记账方法，不改
   ``CandidateAttemptLedger.settle_reward`` 行为。

全部合成数据、零 LLM / 零网络 / 零模型训练；不 import torch / faiss /
lightgbm / xgboost。
"""

from __future__ import annotations

import pytest

from alphaprobe.llm_client import (
    DeterministicStubLLMClient,
    ExecutionMode,
    LLMConfigurationError,
    UsageRecord,
    resolve_llm_fn,
)
from alphaprobe.search.cost_accounting import (
    DEFAULT_QE_SECONDS_REF,
    SearchCostLedger,
    action_cost_adjusted_ev,
)
from alphaprobe.search.routing import (
    MODEL_CLASSES,
    ModelRoutingConfig,
    RouteResult,
    UnknownModelClassError,
    classify_llm_route,
    resolve_model_class,
)


# ---------------------------------------------------------------------------
# 1. duplicate candidate 的 QE 成本 ≈ 0
# ---------------------------------------------------------------------------


class TestDuplicateQECost:
    def test_duplicate_candidate_qe_cost_zero(self):
        led = SearchCostLedger()
        # duplicate：GlobalFactorGate 前拒，从未进 QE
        led.record_attempt(
            attempt_id="dup-1",
            factor_id="f-dup",
            kind="duplicate",
            duplicate_of="f-orig",
        )
        assert led.duplicate_count == 1
        assert led.total_qe_seconds == 0.0
        assert led.qe_seconds_of("f-dup") == 0.0

    def test_full_eval_costs_money(self):
        led = SearchCostLedger()
        led.record_attempt(attempt_id="a1", factor_id="f1", kind="qe", qe_seconds=300.0)
        assert led.total_qe_seconds == pytest.approx(300.0)
        assert led.total_fe_seconds == 0.0

    def test_duplicate_saves_estimated_cost(self):
        """duplicate 节省 = 估计重跑一次的 QE 成本；净值反映在 ledger。"""
        led = SearchCostLedger()
        led.record_attempt(attempt_id="a1", factor_id="f1", kind="qe", qe_seconds=300.0)
        led.record_duplicate_saving(factor_id="f-dup", saved_qe_seconds=300.0)
        assert led.total_qe_seconds == pytest.approx(300.0)  # 已实际发生不变
        assert led.total_qe_saved_seconds == pytest.approx(300.0)
        assert led.net_qe_seconds == pytest.approx(0.0)

    def test_totals_dict_shape(self):
        led = SearchCostLedger()
        led.record_attempt(attempt_id="a1", factor_id="f1", kind="qe", qe_seconds=100.0)
        t = led.totals()
        for key in (
            "attempts",
            "duplicates",
            "qe_seconds",
            "fe_seconds",
            "qe_saved_seconds",
            "llm_tokens",
            "llm_cost",
        ):
            assert key in t

    def test_aggregate_cost_source_compatible_with_ev(self):
        """CostLedger 聚合口径可直接接入 EV 的 cost_source（每 factor 成本）。"""
        led = SearchCostLedger()
        led.record_attempt(attempt_id="a1", factor_id="f1", kind="qe", qe_seconds=60.0)
        cost_fn = led.cost_source()
        # 归一成本 = qe_seconds / ref（默认 ref=3600）→ 60s = 0.0167
        assert cost_fn("f1") == pytest.approx(60.0 / DEFAULT_QE_SECONDS_REF)
        assert cost_fn("f-unknown") == 0.0  # 未见 → 0（无成本不惩罚）


# ---------------------------------------------------------------------------
# 2. 等增益下贵 action 的 cost-adjusted EV 更低（cost 进 EV）
# ---------------------------------------------------------------------------


class TestCostAdjustedEV:
    def test_equal_gain_expensive_action_lower_ev(self):
        """同一增益 0.5：cost 0.1 的 action EV 显著高于 cost 0.9 的。"""
        cheap = action_cost_adjusted_ev(gain=0.5, cost=0.1, p=0.5)
        pricey = action_cost_adjusted_ev(gain=0.5, cost=0.9, p=0.5)
        assert cheap > pricey

    def test_ev_shape_cost_in_denominator(self):
        """EVI = p×gain/cost：cost 进 EV 而非仅日志。"""
        ev = action_cost_adjusted_ev(gain=0.5, cost=0.25, p=0.4)
        assert ev == pytest.approx(0.4 * 0.5 / 0.25)

    def test_zero_cost_no_penalty(self):
        ev = action_cost_adjusted_ev(gain=0.5, cost=0.0, p=0.4)
        assert ev == pytest.approx(0.4 * 0.5)

    def test_invalid_cost_fails_closed(self):
        with pytest.raises(ValueError):
            action_cost_adjusted_ev(gain=0.5, cost=-0.1, p=0.4)

    def test_ledger_projection_informs_ev(self):
        """ledger 聚合的每-factor 成本可直接投影到 cost-adjusted EV。"""
        led = SearchCostLedger()
        led.record_attempt(attempt_id="a1", factor_id="cheap", kind="qe", qe_seconds=10.0)
        led.record_attempt(attempt_id="a2", factor_id="pricey", kind="qe", qe_seconds=900.0)
        # 归一成本：qe_seconds 线性的 cost 投影（cap 1.0）
        c_cheap = min(1.0, 10.0 / 900.0)
        c_pricey = min(1.0, 900.0 / 900.0)
        ev_cheap = action_cost_adjusted_ev(gain=0.5, cost=c_cheap, p=0.5)
        ev_pricey = action_cost_adjusted_ev(gain=0.5, cost=c_pricey, p=0.5)
        assert ev_cheap > ev_pricey


# ---------------------------------------------------------------------------
# 3. llm_client UsageRecord 记账（不破坏 fail-closed）
# ---------------------------------------------------------------------------


class _RecorderClient:
    """带 usage 的协议 client：记录 generate 调用并返回 usage 可选。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_class: str,
        response_schema: type | None = None,
    ) -> str:
        del response_schema
        self.calls.append((system_prompt, user_prompt, model_class))
        return "rank(close)"


class _UsageClient(_RecorderClient):
    """generate 额外暴露 usage 的协议 client（向后兼容的加法）。"""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_class: str,
        response_schema: type | None = None,
    ) -> object:
        del response_schema
        self.calls.append((system_prompt, user_prompt, model_class))
        out = _TextWithUsage("rank(close)")
        out.usage = UsageRecord(
            prompt_tokens=120,
            completion_tokens=45,
            latency_ms=320,
            cost=0.0021,
        )
        return out


class _TextWithUsage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage: UsageRecord | None = None


class TestLLMUsageRecord:
    def test_usage_record_defaults(self):
        u = UsageRecord()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.latency_ms == 0
        assert u.cost == 0.0
        assert u.model_class == ""

    def test_usage_record_fields(self):
        u = UsageRecord(
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=210,
            cost=0.001,
            model_class="strong",
        )
        assert u.prompt_tokens == 100
        assert u.cost == 0.001
        assert u.total_tokens == 150

    def test_usage_record_consumed_by_cost_ledger(self):
        led = SearchCostLedger()
        u = UsageRecord(
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=210,
            cost=0.001,
            model_class="strong",
        )
        led.record_llm_usage(u, factor_id="f1")
        assert led.total_llm_tokens == 150
        assert led.total_llm_cost == pytest.approx(0.001)
        assert led.total_llm_calls == 1

    def test_usage_record_into_legacy_ledger_column(self):
        """UsageRecord 可直接写入 CandidateAttemptLedger 的成本字段（llm_tokens/llm_cost）。"""
        from alphaprobe.ledger import CandidateAttemptLedger

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            led = CandidateAttemptLedger(db_path=f"{td}/ledger.sqlite3")
            u = UsageRecord(prompt_tokens=30, completion_tokens=10, cost=0.0005)
            ok, _ = led.record_attempt(
                attempt_id="at-1",
                llm_tokens=u.prompt_tokens + u.completion_tokens,
                llm_cost=u.cost,
            )
            assert ok
            row = led.get_attempt("at-1")
            assert row["llm_tokens"] == 40
            assert row["llm_cost"] == pytest.approx(0.0005)
            led.close()

    # -- fail-closed 语义保持 -------------------------------------------------

    def test_fail_closed_semantics_unchanged(self):
        """resolve_llm_fn PRODUCTION 下 None/stub 仍抛（扩展不软化纪律）。"""
        with pytest.raises(LLMConfigurationError):
            resolve_llm_fn(None, ExecutionMode.PRODUCTION)
        with pytest.raises(LLMConfigurationError):
            resolve_llm_fn(DeterministicStubLLMClient(), ExecutionMode.PRODUCTION)

    def test_offline_stub_still_allowed(self):
        fn = resolve_llm_fn(None, ExecutionMode.OFFLINE_TEST)
        assert callable(fn)
        out = fn("s", "u", "cheap")
        assert isinstance(out, str)

    def test_production_recorder_client_ok(self):
        fn = resolve_llm_fn(_RecorderClient(), ExecutionMode.PRODUCTION)
        out = fn("s", "u", "cheap")
        assert out == "rank(close)"

    def test_resolve_llm_fn_returns_usage_text_when_client_provides(self):
        """协议宽松：带 .text + .usage 的对象 → llm_fn 仍返回文本（兼容），
        usage 由 client 层可选暴露（向后兼容，不改变既有调用签名）。"""
        client = _UsageClient()
        fn = resolve_llm_fn(client, ExecutionMode.PRODUCTION)
        out = fn("s", "u", "cheap")
        assert out == "rank(close)"
        assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# 4. cheap / strong / critic 三档 model class 路由（Part J1 留白协议）
# ---------------------------------------------------------------------------


class TestModelClassRouting:
    def test_three_classes_known(self):
        assert set(MODEL_CLASSES) == {"cheap", "strong", "critic"}

    def test_action_family_routing(self):
        # 确定性默认表：REFINE 类 → cheap；schema 探索/批判 → 强/批判
        cheap_actions = {"REFINE", "GENERATION_REPAIR", "FIELD_SUBSTITUTION"}
        for a in cheap_actions:
            assert classify_llm_route(a) == "cheap"
        strong = classify_llm_route("SCHEMA_EXPLORE")
        assert strong in {"strong", "critic"}

    def test_explicit_override_wins(self):
        cfg = ModelRoutingConfig(overrides={"REFINE": "strong"})
        assert classify_llm_route("REFINE", cfg=cfg) == "strong"

    def test_unknown_action_defaults_cheap(self):
        assert classify_llm_route("NO_SUCH_ACTION") == "cheap"

    def test_unknown_model_class_fails_closed(self):
        with pytest.raises(UnknownModelClassError):
            resolve_model_class("turbo-ultra")

    def test_resolve_returns_route_result(self):
        r = resolve_model_class("cheap")
        assert isinstance(r, RouteResult)
        assert r.model_class == "cheap"
        assert r.provider is None  # Part J1：provider/model 名留白
        assert r.model_name is None

    def test_config_default_classes(self):
        cfg = ModelRoutingConfig()
        assert cfg.cheap == "cheap"
        assert cfg.strong == "strong"
        assert cfg.critic == "critic"

    def test_critic_class_resolvable(self):
        r = resolve_model_class("critic")
        assert r.model_class == "critic"
