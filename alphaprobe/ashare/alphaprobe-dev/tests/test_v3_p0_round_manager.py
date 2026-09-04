"""V3 P0-E：RoundManager._update_memory 三修 + GlobalMemoryStore 搜索轮事件。

P0-E 范围（任务书 §28/§74/§77）：
- §74 事件驱动：round 事件写独立 search_run_events，regime 表不再被污染；
- pool_snapshot 全量兜底对账（去掉 [:50] 截断）；
- identity 只认 FactorEngine 权威（build_identity_view），不回落文本 regex；
- LineagePatienceTracker round 级 dummy 现状仅注释（行为不变）。
"""

from __future__ import annotations

import pytest

from alphaprobe.memory import GlobalMemoryStore


# ---------------------------------------------------------------------------
# §74：GlobalMemoryStore.add_search_run_event（独立表 + 幂等）
# ---------------------------------------------------------------------------


def test_add_search_run_event_idempotent_and_regime_clean(tmp_path):
    """round 事件写 search_run_events；market_regime_events 保持空。"""
    store = GlobalMemoryStore(tmp_path / "memory.sqlite3")
    try:
        store.add_search_run_event(
            event_id="round_r1_complete",
            round_id="r1",
            description="round r1 complete",
            payload={"pool_size": 3},
        )
        # 幂等：同 event_id 再次写入不报错、不产生第二行
        store.add_search_run_event(
            event_id="round_r1_complete",
            round_id="r1",
            description="round r1 complete",
            payload={"pool_size": 3},
        )
        assert store.search_run_event_exists("round_r1_complete") is True

        rows = store._conn.execute(
            "SELECT event_id, round_id, description, payload, created_at"
            " FROM search_run_events"
        ).fetchall()
        assert len(rows) == 1
        event_id, round_id, description, payload, created_at = rows[0]
        assert event_id == "round_r1_complete"
        assert round_id == "r1"
        assert description == "round r1 complete"
        assert created_at  # created_at TEXT 非空

        # regime 表零污染：没有任何 round_* 事件
        regime_rows = store._conn.execute(
            "SELECT event_id FROM market_regime_events WHERE event_id LIKE 'round\\_%' ESCAPE '\\'"
        ).fetchall()
        assert regime_rows == []
        n_regime = store._conn.execute(
            "SELECT COUNT(*) FROM market_regime_events"
        ).fetchone()[0]
        assert n_regime == 0
    finally:
        store.close()


def test_search_run_events_separate_from_regime(tmp_path):
    """同 event_id 分别写两表，互不串扰（round 事件不进 regime）。"""
    store = GlobalMemoryStore(tmp_path / "memory.sqlite3")
    try:
        store.add_regime_event(
            event_id="decay_2026_jun_jul",
            start_date="2026-06-01",
            end_date="2026-07-31",
            description="market regime",
            payload={"kind": "massive_decay"},
        )
        store.add_search_run_event(
            event_id="round_r1_summary",
            round_id="r1",
            description="round r1 pipeline summary",
            payload={"evaluated": 2, "admitted": 1},
        )
        # 两表独立存在
        assert store.regime_event_exists("decay_2026_jun_jul") is True
        assert store.regime_event_exists("round_r1_summary") is False
        assert store.search_run_event_exists("round_r1_summary") is True
        assert store.search_run_event_exists("decay_2026_jun_jul") is False
    finally:
        store.close()


# ---------------------------------------------------------------------------
# RoundManager._update_memory：全量对账 + FE 权威身份
# ---------------------------------------------------------------------------


class _RecordingStore:
    """记录调用的最小 stub（不落 SQLite）。"""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.upserted: list[dict] = []
        self.packets = 0

    def add_search_run_event(self, **kwargs) -> None:
        self.events.append(dict(kwargs))

    def build_memory_packet(self, **kwargs):
        self.packets += 1
        return {"parent": {}}

    def upsert_factor_node(self, **kwargs) -> tuple[bool, None]:
        self.upserted.append(dict(kwargs))
        return True, None


class _FakeCalibrator:
    def freeze(self, round_id: str) -> None:
        pass


def _mk_member(i: int) -> dict:
    return {"factor_id": f"f{i:03d}", "formula": f"rank(ts_mean(close, {i}))"}


def _mk_result(n_members: int, *, evaluated: int = 0, admitted: int = 0) -> dict:
    class _RR:
        pass

    rr = _RR()
    rr.pool_snapshot = [_mk_member(i) for i in range(n_members)]
    rr.evaluated = evaluated
    rr.admitted = admitted
    return {"pool_size": n_members, "round_result": rr}


def _new_rm(store) -> "RoundManager":
    from alphaprobe.continuous.round_manager import RoundManager

    return RoundManager(
        store=store,
        calibrator=_FakeCalibrator(),
        campaign_callback=lambda **kw: {"pool_size": 0},
        state_dir="/tmp/p0e_state",
        sleep_seconds=0,
    )


def test_update_memory_full_pool_no_50_truncation(monkeypatch):
    """60 个成员全量 upsert（不再是 [:50] 截断）。"""
    from alphaprobe.continuous.round_manager import RoundManager

    store = _RecordingStore()
    rm = _new_rm(store)
    monkeypatch.setattr(RoundManager, "run", lambda self: 0)  # 不进入主循环

    fake_view = {
        "canonical_formula": "CANON",
        "canonical_ast_hash": "AST_HASH",
        "signal_equivalence_id": "SIG_ID",
        "parameter_family_id": "FAM",
    }

    def fake_build(formula, provider=None):  # noqa: ARG001
        fake_view["formula"] = formula
        return dict(fake_view)

    # _update_memory 函数内 from alphaprobe.authority import ... as _build_identity_view
    # → 直接 patch alphaprobe.authority.build_identity_view（函数调用时读取）
    from alphaprobe import authority

    monkeypatch.setattr(authority, "build_identity_view", fake_build)

    rm._update_memory("r1", _mk_result(60, evaluated=5, admitted=2))

    # 60 个成员全量写（非 50 截断）
    assert len(store.upserted) == 60
    fids = {u["factor_id"] for u in store.upserted}
    assert fids == {f"f{i:03d}" for i in range(60)}
    # 身份字段来自 FE 权威 view
    for u in store.upserted:
        assert u["canonical_formula"] == "CANON"
        assert u["canonical_ast_hash"] == "AST_HASH"
        assert u["signal_equivalence_id"] == "SIG_ID"
        assert u["parameter_family_id"] == "FAM"
        assert u["source_system"] == "alphaprobe"
        assert u["source_type"] == "MINED"
        assert u["exportable"] is False

    # 两处 round 事件均写 add_search_run_event（非 add_regime_event）
    assert len(store.events) == 2
    assert {e["event_id"] for e in store.events} == {"round_r1_complete", "round_r1_summary"}
    for e in store.events:
        assert "round_id" in e


def test_update_memory_authority_error_skips_member(monkeypatch):
    """build_identity_view 抛 FactorIdentityAuthorityError → 跳过该成员不阻塞。"""
    from alphaprobe.authority import FactorIdentityAuthorityError

    store = _RecordingStore()
    rm = _new_rm(store)

    def failing_build(formula, provider=None):  # noqa: ARG001
        if formula.startswith("rank(ts_mean(close, 7))"):
            raise FactorIdentityAuthorityError("FE identity unavailable")
        return {
            "canonical_formula": formula,
            "canonical_ast_hash": f"h:{formula}",
            "signal_equivalence_id": f"s:{formula}",
            "parameter_family_id": None,
        }

    from alphaprobe import authority

    monkeypatch.setattr(authority, "build_identity_view", failing_build)

    rm._update_memory("r1", _mk_result(10))

    # 9 个成员写入，bad 成员被跳过；不抛异常
    assert len(store.upserted) == 9
    fids = {u["factor_id"] for u in store.upserted}
    assert "f007" not in fids
