"""plan.md Task 4 / A8：CandidateAttemptLedger + settle_reward 幂等。

覆盖验收点：
1. 同一 attempt_id settle 两次只更新 scheduler 一次（DB 仲裁 + 内存守卫）；
2. 评估后、结算前 crash → 重载后 resume 能且只能结算一次（新进程新 step、
   复用持久化 attempt_id → 账本拦第二次）；
3. FE 校验失败也建 ledger row（failure_reason 有值，行存在）；
4. duplicate factor 记录 rediscovery attempt 但无评估成本（qe_seconds 等
   为 0/None）；
5. reward 可追溯到子代评估（attempt → record_event event_id → settle 用同
   一 reward_event_id 的链完整）。

测试纪律：DB 一律 tmp_path，绝不写当前目录；ledger=None 时 settle 退化为
内存幂等守卫（不回退到旧版重复计数语义）。
"""

from __future__ import annotations

import json

import pytest

from alphaprobe.ledger import CandidateAttemptLedger
from alphaprobe.search import ACTION_FAMILIES
from alphaprobe.search.orchestrator import SearchOrchestrator

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stub_llm_text(system_prompt: str, user_prompt: str, model_class: str) -> str:
    """单候选 stub：action_type REFINE、一个 parent。"""
    return json.dumps(
        {
            "candidates": [
                {
                    "formula": "rank(ts_std(close, 20))",
                    "explanation": "stub",
                    "hypothesis": "stub hypothesis",
                    "action_type": "REFINE",
                    "parent_ids": ["p1"],
                }
            ]
        }
    )


def _parent() -> dict:
    return {
        "factor_id": "p1",
        "formula": "rank(ts_mean(close, 20))",
        "explanation": "parent",
    }


def _family_of(orch: SearchOrchestrator, step) -> str:
    return step.action.action_type.value


def _attempts_of(orch: SearchOrchestrator, family: str) -> int:
    return int(orch.scheduler.stats[family].attempts)


# ---------------------------------------------------------------------------
# 1. 同一 attempt_id settle 两次只更新 scheduler 一次
# ---------------------------------------------------------------------------


class TestSettleTwice:
    def test_no_ledger_second_settle_is_noop(self):
        """内存守卫：无账本时同一 step settle 两次也只更新 scheduler 一次。"""
        orch = SearchOrchestrator()
        step = orch.step_with_llm(_parent(), _stub_llm_text)
        family = _family_of(orch, step)
        orch.settle_reward(step, 0.5)
        assert _attempts_of(orch, family) == 1
        assert step.reward_settled is True
        assert step.reward_event_id is not None
        # 第二次 settle：no-op，attempts 不 +1
        orch.settle_reward(step, 0.5)
        assert _attempts_of(orch, family) == 1
        assert orch.scheduler.state()[family]["mean"] == pytest.approx(0.5)

    def test_ledger_same_attempt_id_second_settle_blocked(self, tmp_path):
        """账本仲裁：同一 attempt_id 用**同一 event** 重复 settle 只赢一次。

        等价重放：worker retry 用相同 event id 重放结算。
        """
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "ledger.sqlite3"))
        orch = SearchOrchestrator(ledger=ledger)
        step = orch.step_with_llm(_parent(), _stub_llm_text, attempt_id="at_twice")
        family = _family_of(orch, step)
        orch.settle_reward(step, 0.4)
        assert _attempts_of(orch, family) == 1
        assert ledger.is_settled("at_twice") is True
        # 第二次（同一 step 对象）→ 内存守卫直接 no-op
        orch.settle_reward(step, 0.4)
        assert _attempts_of(orch, family) == 1

    def test_ledger_same_attempt_two_workers_two_events_only_one_wins(self, tmp_path):
        """两个 worker 抢同一 attempt：都用账本仲裁，只有一个赢家。

        worker A：新 step（attempt_id=at_race）、新 event → 赢；
        worker B：另一进程视角的新 step（同 attempt_id）、另一 event → 输。
        """
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "ledger.sqlite3"))
        orch_a = SearchOrchestrator(ledger=ledger)
        step_a = orch_a.step_with_llm(
            _parent(), _stub_llm_text, attempt_id="at_race"
        )
        orch_a.settle_reward(step_a, 0.3)
        family_a = _family_of(orch_a, step_a)
        assert _attempts_of(orch_a, family_a) == 1

        # worker B：crash 后新进程（新 orchestrator、新 step），复用 attempt_id
        orch_b = SearchOrchestrator(ledger=ledger)
        step_b = orch_b.step_with_llm(
            _parent(), _stub_llm_text, attempt_id="at_race"
        )
        orch_b.settle_reward(step_b, 0.9)
        family_b = _family_of(orch_b, step_b)
        assert _attempts_of(orch_b, family_b) == 0  # 输家绝不更新 scheduler
        # 账本里仍是 A 的 reward（先写先赢）
        row = ledger.get_attempt("at_race")
        assert row["reward"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 2. 评估后、结算前 crash → resume 能且只能结算一次
# ---------------------------------------------------------------------------


class TestCrashResume:
    def test_resume_settles_once_across_processes(self, tmp_path):
        """完整生命周期：评估完成（l2/l3 状态落库）→ 结算前 crash → resume。

        - worker 1：record_attempt 占位 + 评估（l2_status/l3_status 落库）+
          record_event(evaluation)；结算前进程死亡 → DB 留下未结算 attempt；
        - worker 2（新 ledger 连接 = 新进程）：unsettled_attempt_ids 找到它 →
          settle 一次成功；再 settle（重放）失败；scheduler 只 +1。
        """
        db = str(tmp_path / "crash.sqlite3")
        # --- worker 1（crash 前）---
        ledger1 = CandidateAttemptLedger(db_path=db)
        created, _ = ledger1.record_attempt(
            attempt_id="at_crash1",
            round_id="r1",
            generation=2,
            parent_ids=["p1"],
            action_type="REFINE",
            compiled_factor_id="f_crash1",
            dedup_status="NEW",
            l1_status="PASS",
            l2_status="PASS",
            l3_status="PASS",
            fitness_before=0.10,
            fitness_after=0.16,
            delta_fitness=0.06,
            qe_seconds=2.5,
            fe_seconds=0.4,
        )
        assert created is True
        ledger1.record_event(
            event_id="ev_crash1_eval",
            attempt_id="at_crash1",
            event_kind="evaluation_completed",
            payload={"fitness_after": 0.16},
        )
        # crash 模拟：不 settle，直接丢进程（close 连接即可）
        ledger1.close()

        # --- worker 2（resume）---
        ledger2 = CandidateAttemptLedger(db_path=db)
        unsettled = ledger2.unsettled_attempt_ids(round_id="r1")
        assert "at_crash1" in unsettled
        orch = SearchOrchestrator(ledger=ledger2)
        # resume：构造同一 attempt 的 step（复用 attempt_id + 评估 event id）
        step = orch.step_with_llm(
            _parent(), _stub_llm_text, attempt_id="at_crash1"
        )
        step.reward_event_id = "ev_crash1_eval"  # reward 溯源到子代评估事件
        orch.settle_reward(step, 0.7)
        family = _family_of(orch, step)
        assert _attempts_of(orch, family) == 1  # 结算一次
        assert ledger2.is_settled("at_crash1")
        row = ledger2.get_attempt("at_crash1")
        assert row["reward"] == pytest.approx(0.7)
        assert row["reward_event_id"] == "ev_crash1_eval"
        assert row["l2_status"] == "PASS"  # 评估信息保留

        # resume 后 worker retry 重放 settle（同 attempt、同 event）→ 被拦
        orch.settle_reward(step, 0.7)
        assert _attempts_of(orch, family) == 1  # 不重复计

    def test_settle_arbitration_after_placeholder_row(self, tmp_path):
        """仅 record_attempt 占位（无 reward 字段）→ settle 走 UPDATE 命中。"""
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "p.sqlite3"))
        ledger.record_attempt(attempt_id="at_ph", round_id="r1", action_type="REFINE")
        won, reason = ledger.settle_reward(
            attempt_id="at_ph", reward=0.2, reward_event_id="ev_ph"
        )
        assert won is True
        assert reason is None
        # 再结算（新 event）→ 输
        won2, reason2 = ledger.settle_reward(
            attempt_id="at_ph", reward=0.8, reward_event_id="ev_ph2"
        )
        assert won2 is False
        assert "already settled" in (reason2 or "")
        assert ledger.get_attempt("at_ph")["reward"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 3. FE 校验失败也建 ledger row（failure_reason 有值）
# ---------------------------------------------------------------------------


class TestFeFailureRow:
    def test_failed_fe_validation_creates_row(self, tmp_path):
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "fe.sqlite3"))
        created, _ = ledger.record_attempt(
            attempt_id="at_fe_fail",
            round_id="r1",
            generation=1,
            parent_ids=["p1"],
            action_type="REFINE",
            raw_llm_output_ref="llm_out_1",
            compiled_factor_id=None,
            failure_reason="INVALID_DSL: unbalanced parentheses",
            fe_seconds=0.05,
            qe_seconds=None,  # FE 失败 → 无 QE 成本
            dedup_status="FE_REJECTED",
        )
        assert created is True
        row = ledger.get_attempt("at_fe_fail")
        assert row is not None
        assert row["failure_reason"] == "INVALID_DSL: unbalanced parentheses"
        assert row["compiled_factor_id"] is None
        assert row["qe_seconds"] is None
        assert row["fe_seconds"] == pytest.approx(0.05)
        # FE 失败 → 无 reward（未评估）→ 不应出现在 settled
        assert row["reward"] is None
        assert ledger.is_settled("at_fe_fail") is False

    def test_record_attempt_idempotent_no_overwrite(self, tmp_path):
        """同 attempt_id 重复落行（crash 重放）不覆盖已有字段。"""
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "idem.sqlite3"))
        ledger.record_attempt(
            attempt_id="at_fix", failure_reason="INVALID_DSL", qe_seconds=None
        )
        created2, existing = ledger.record_attempt(
            attempt_id="at_fix", failure_reason="PIT_VIOLATION", qe_seconds=9.9
        )
        assert created2 is False
        assert existing == "at_fix"
        row = ledger.get_attempt("at_fix")
        assert row["failure_reason"] == "INVALID_DSL"  # 先写先赢，不被覆盖
        assert row["qe_seconds"] is None


# ---------------------------------------------------------------------------
# 4. duplicate factor 记录 rediscovery attempt 但无评估成本
# ---------------------------------------------------------------------------


class TestDuplicateRediscovery:
    def test_duplicate_records_rediscovery_without_eval_cost(self, tmp_path):
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "dup.sqlite3"))
        # 占位行（生成后即发现是 duplicate：无 FE/QE）
        created, _ = ledger.record_attempt(
            attempt_id="at_dup1",
            round_id="r1",
            parent_ids=["p1"],
            action_type="REFINE",
            compiled_factor_id=None,
            dedup_status="NEW",
        )
        assert created is True
        # §41 硬重复命中 → mark_duplicate：rediscovery，不产生评估成本
        updated = ledger.mark_duplicate(
            attempt_id="at_dup1", duplicate_of="f_seed_001", dedup_status="REDISCOVERY"
        )
        assert updated is True
        row = ledger.get_attempt("at_dup1")
        assert row["dedup_status"] == "REDISCOVERY"
        assert row["failure_reason"] == "f_seed_001"  # duplicate_of 记录
        # 无评估成本：qe_seconds/fe_seconds/l2/l3 全 None，reward 未结算
        assert row["qe_seconds"] is None
        assert row["fe_seconds"] is None
        assert row["l2_status"] is None
        assert row["l3_status"] is None
        assert row["reward"] is None
        assert ledger.is_settled("at_dup1") is False

    def test_mark_duplicate_never_overwrites_settled(self, tmp_path):
        """先结算后才发现 duplicate → mark_duplicate 不覆盖 reward 字段。"""
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "dup2.sqlite3"))
        ledger.settle_reward(
            attempt_id="at_dup2", reward=0.6, reward_event_id="ev_dup2"
        )
        ledger.mark_duplicate(attempt_id="at_dup2", duplicate_of="f_x")
        row = ledger.get_attempt("at_dup2")
        assert row["reward"] == pytest.approx(0.6)  # 结算信息保留
        assert row["reward_event_id"] == "ev_dup2"


# ---------------------------------------------------------------------------
# 5. reward 可追溯到子代评估（reward_source / reward_event_id 链）
# ---------------------------------------------------------------------------


class TestRewardTraceability:
    def test_reward_event_chain_to_child_evaluation(self, tmp_path):
        """attempt → record_event(评估事件) → settle 同 event id → 可溯源。"""
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "trace.sqlite3"))
        ledger.record_attempt(
            attempt_id="at_trace1",
            round_id="r1",
            parent_ids=["p1"],
            action_type="REFINE",
            compiled_factor_id="f_trace1",
            dedup_status="NEW",
            l1_status="PASS",
            l2_status="PASS",
            fitness_before=0.2,
            fitness_after=0.26,
        )
        # 子代评估事件（evaluation event）带 metric 载荷
        ledger.record_event(
            event_id="ev_child_eval_42",
            attempt_id="at_trace1",
            event_kind="evaluation_completed",
            payload={"metric_bundle": {"rankic": 0.05}, "fidelity": "L2_full_train"},
        )
        # settle：reward_event_id 用同一子代评估事件 id
        won, _ = ledger.settle_reward(
            attempt_id="at_trace1", reward=0.31, reward_event_id="ev_child_eval_42"
        )
        assert won is True
        row = ledger.get_attempt("at_trace1")
        # 链完整：attempt 行 reward_event_id == 子代评估 event_id
        assert row["reward_event_id"] == "ev_child_eval_42"
        assert row["reward"] == pytest.approx(0.31)
        assert row["reward_settled_at"] is not None
        assert row["settled"] is True
        # 事件表可查：同 event id 存在且属同一 attempt（事件 → attempt 反查）
        ev = ledger.get_event("ev_child_eval_42")
        assert ev is not None
        assert ev["attempt_id"] == "at_trace1"
        assert ev["event_kind"] == "evaluation_completed"
        assert "rankic" in json.dumps(ev.get("payload") or {})

    def test_orchestrator_settle_stamps_event_and_timestamp(self, tmp_path):
        """orchestrator settle 成功 → step 带 reward_event_id/reward_settled_at。"""
        ledger = CandidateAttemptLedger(db_path=str(tmp_path / "tr2.sqlite3"))
        orch = SearchOrchestrator(ledger=ledger)
        step = orch.step_with_llm(_parent(), _stub_llm_text, attempt_id="at_tr2")
        step.reward_event_id = "ev_tr2_eval"
        orch.settle_reward(step, 0.42)
        assert step.reward_settled is True
        assert step.pending_reward is False
        assert step.reward_event_id == "ev_tr2_eval"
        assert step.reward_settled_at is not None
        row = ledger.get_attempt("at_tr2")
        assert row["reward_event_id"] == "ev_tr2_eval"


# ---------------------------------------------------------------------------
# 辅助：get_event 查询接口（reward 溯源反查）
# ---------------------------------------------------------------------------
