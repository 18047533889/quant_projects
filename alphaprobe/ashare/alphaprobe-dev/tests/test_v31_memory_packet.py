"""plan.md Task 21 / Part A12：MemoryPacket V2（multi-parent、多段、预算截断）。

覆盖 plan 三条 + Part A12：
1. multi-parent packet 包含全部 parents 与 roles（不再单 parent 偏置）；
2. sealed survival 段被排除（版本隔离，#23）；
3. 超大记忆确定性截断（token budget 2k-4k 可配置，截断优先级固定不随机）；
4. 缺失段优雅省略（None → 不进 prompt，不硬造内容）；
5. retriever 组装点改造保持既有调用兼容（V2 封装）。

实现文件：memory/memory_packet.py（新建）+ memory/retriever.py（组装点改造）。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from alphaprobe.contracts import EvaluationRecord
from alphaprobe.memory import GlobalMemoryStore, MemoryRetriever
from alphaprobe.memory.memory_packet import (
    DEFAULT_TOKEN_BUDGET_MAX,
    DEFAULT_TOKEN_BUDGET_MIN,
    MemoryPacketV2,
    PacketBuilder,
    Section,
    sealed_survival_rows,
    simple_tokens,
)


@pytest.fixture()
def store(tmp_path):
    s = GlobalMemoryStore(tmp_path / "memory.sqlite3")
    yield s
    s.close()


def _upsert(store, fid, formula, signal_id, family_id=None, schema_json=None):
    return store.upsert_factor_node(
        factor_id=fid,
        canonical_formula=formula,
        canonical_ast_hash=f"h:{fid}",
        signal_equivalence_id=signal_id,
        parameter_family_id=family_id,
        schema_json=schema_json,
    )


def _parent(fid="p1", formula="rank(ts_mean(close, 20))", fitness=None):
    d = {"factor_id": fid, "formula": formula}
    if fitness is not None:
        d["fitness"] = fitness
    return d


# ---------------------------------------------------------------------------
# 1. multi-parent packet：全部 parents 与 roles（不再单 parent 偏置）
# ---------------------------------------------------------------------------


def test_v2_packet_includes_all_parents_and_roles():
    packet = MemoryPacketV2(
        parents=[_parent("A", "rank(close)"), _parent("B", "ts_std(volume, 20)")],
        parent_roles={"A": "main", "B": "context"},
    )
    text = packet.to_prompt_text()
    assert "[Parent set]" in text
    assert "rank(close)" in text
    assert "ts_std(volume, 20)" in text
    assert "main=" in text
    assert "context=" in text
    # 没有丢弃第二个 parent（不再单 parent 偏置）
    assert text.index("rank(close)") < text.index("ts_std(volume, 20)")


def test_v2_primary_parent_is_first_for_backcompat():
    packet = MemoryPacketV2(
        parents=[_parent("A", "rank(close)"), _parent("B", "ts_std(volume, 20)")],
    )
    # 旧版兼容属性 .parent 仍指向第一个 parent
    assert packet.parent["factor_id"] == "A"


def test_v2_prompt_len_under_budget_when_oversized():
    """multi-parent + 多邻居 + 多段 → 总 prompt 长度受 budget 上限约束。"""
    parents = [_parent(f"p{i}", f"rank(ts_mean(close, {20 + i}))", fitness=0.0 + i / 100)
               for i in range(6)]
    packet = MemoryPacketV2(
        parents=parents,
        parent_roles={f"p{i}": ("main" if i == 0 else "parent") for i in range(6)},
        structural_neighbors=[
            {"factor_id": f"s{i}", "formula": f"ts_std(close, {10 + i})", "fitness": 0.02 * i}
            for i in range(8)
        ],
        numerical_neighbors=[
            {"factor_id": f"n{i}", "formula": f"ts_mean(close, {5 + i})", "fitness": 0.01 * i}
            for i in range(8)
        ],
        successful_actions=["REFINE", "WINDOW_SCALE", "CROSSOVER"] * 3,
        failed_actions=["OPERATOR_SUBSTITUTION", "STATE_CONDITION"],
        allowed_fields=[f"field_{i}" for i in range(20)],
        allowed_operators=[f"op_{i}" for i in range(20)],
        research_objective="liquidity absorption reversal with volume divergence",
    )
    text = packet.build_packet_text()
    assert simple_tokens(text) <= DEFAULT_TOKEN_BUDGET_MAX


# ---------------------------------------------------------------------------
# 2. sealed survival 段被排除（版本隔离，#23）
# ---------------------------------------------------------------------------


def test_sealed_survival_rows_excluded_by_version():
    rows = [
        {"factor_id": "f1", "formula": "rank(close)", "survival_rate": 0.9,
         "version_key": "2025"},
        {"factor_id": "f2", "formula": "ts_std(close, 20)", "survival_rate": 0.8,
         "research_version": "2025"},
        {"factor_id": "f3", "formula": "ts_mean(close, 10)", "survival_rate": 0.7,
         "segment": "2024"},
        # 无版本字段 → 当前版本合法（不伪造排除）
        {"factor_id": "f4", "formula": "delta(close, 5)", "survival_rate": 0.6},
    ]
    kept = sealed_survival_rows(rows, version_key="2026r2")
    ids = {r["factor_id"] for r in kept}
    assert ids == {"f4"}


def test_retriever_excludes_sealed_survival_when_version_given(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    _upsert(store, "p2", "ts_std(close, 20)", "sig-p2", family_id="fam-1")
    # 写入带版本字段的 survival（payload 里带 research_version/segment）
    _write_survival_v2(
        store,
        factor_id="p1",
        confidence=0.9,
        version_key="2025",  # 已封存旧版本
    )
    _write_survival_v2(
        store,
        factor_id="p2",
        confidence=0.8,
        version_key="2026r2",  # 当前版本
    )
    retriever = MemoryRetriever(store)
    packet = retriever.build_packet("p1", research_version="2026r2")
    assert packet is not None
    # 只有当前版本的 survival 出现
    ids = {e["factor_id"] for e in packet.survival_exemplars}
    assert ids == {"p2"}


# ---------------------------------------------------------------------------
# 3. 超大记忆确定性截断（budget 可配置 + 优先级固定）
# ---------------------------------------------------------------------------


def test_deterministic_truncation_same_input_same_output():
    """同一输入 + 同一 budget → 截断结果确定（无随机）。"""
    long_parents = [
        _parent(f"p{i}", f"rank(ts_mean(close, {i})) * cs_rank(volume, {i})", fitness=i)
        for i in range(12)
    ]
    kw = {
        "parents": long_parents,
        "parent_roles": {f"p{i}": "parent" for i in range(12)},
        "numerical_neighbors": [
            {"factor_id": f"n{i}", "formula": f"ts_mean(close, {i})" + " + " * (i % 3), "fitness": i}
            for i in range(12)
        ],
    }
    a = MemoryPacketV2(**kw).build_packet_text()
    b = MemoryPacketV2(**kw).build_packet_text()
    assert a == b


def test_truncation_respects_budget():
    big = MemoryPacketV2(
        parents=[_parent(f"p{i}", f"rank(ts_mean(close, {i}))") for i in range(8)],
        structural_neighbors=[
            {"factor_id": f"s{i}", "formula": f"ts_std(close, {i}) * {i}"}
            for i in range(10)
        ],
    )
    small_budget = 400
    text = big.build_packet_text(max_tokens=small_budget)
    assert simple_tokens(text) <= small_budget


def test_budget_configurable_range_default():
    """默认预算 2k-4k；可配置（ablation switch）。"""
    assert DEFAULT_TOKEN_BUDGET_MIN == 2000
    assert DEFAULT_TOKEN_BUDGET_MAX == 4000
    p = MemoryPacketV2(
        parents=[_parent("p1", "rank(close)")],
        research_objective="x",
        budget_max=800,
    )
    assert p.budget_max == 800
    assert simple_tokens(p.to_prompt_text()) <= 800


def test_tiny_budget_still_keeps_research_objective_and_parent_set():
    """极小 budget 下 research objective / parent set 段保留（固定优先级）。"""
    packet = MemoryPacketV2(
        parents=[_parent("p1", "rank(ts_mean(close, 20))", fitness=0.03)],
        research_objective="overnight gap mean reversion",
        structural_neighbors=[{"factor_id": f"s{i}", "formula": f"ts_std(close, {i})"} for i in range(6)],
        numerical_neighbors=[{"factor_id": f"n{i}", "formula": f"ts_mean(close, {i})"} for i in range(6)],
    )
    text = packet.build_packet_text(max_tokens=80)
    assert "[Research objective]" in text or "overnight gap mean reversion" in text
    assert "rank(ts_mean(close, 20))" in text  # parent set 保留


# ---------------------------------------------------------------------------
# 4. 缺失段优雅省略（None → 不进 prompt，不硬造）
# ---------------------------------------------------------------------------


def test_missing_sections_omitted():
    packet = MemoryPacketV2(
        parents=[_parent("p1", "rank(close)")],
        research_objective="liquidity absorption",
    )
    text = packet.to_prompt_text()
    # 未提供 → 段不存在（不出现空标题/占位符）
    assert "[Numerical nearest]" not in text
    assert "[Structural nearest]" not in text
    assert "[Best offspring]" not in text
    assert "[Failures to avoid]" not in text
    assert "[Cluster saturation]" not in text
    assert "[Allowed surface]" not in text
    assert "[Legal survival exemplars]" not in text
    assert "[Avoid repeating]" not in text
    assert "[Research objective]" in text
    assert "[Parent set]" in text


def test_builder_missing_sections_empty():
    """13 段全部缺省 → build_text 空字符串（不硬造内容）。"""
    b = PacketBuilder()
    assert b.build_text() == ""


def test_all_thirteen_sections_rendered_when_present():
    packet = MemoryPacketV2(
        parents=[_parent("p1", "rank(close)", fitness=0.03)],
        parent_roles={"p1": "main"},
        logic_schema={"logic": "LQ1", "schema": "SCH_ABC"},
        lineage_actions=[{"action_type": "REFINE", "action": "transform", "status": "ok"}],
        best_offspring=[{"factor_id": "o1", "formula": "rank(ts_mean(close, 5))"}],
        representative_failures=[{"reason": "lookahead", "formula": "close"}],
        structural_neighbors=[{"factor_id": "s1", "formula": "ts_std(close, 20)"}],
        numerical_neighbors=[{"factor_id": "n1", "formula": "ts_mean(close, 5)"}],
        cluster_saturation={"cluster_id": "c1", "saturation": 0.9},
        successful_actions=["REFINE"],
        failed_actions=["CROSSOVER"],
        allowed_operators=["ts_rank"],
        allowed_fields=["close"],
        survival_exemplars=[{"factor_id": "f1", "formula": "rank(close)", "survival_rate": 0.8}],
        avoid_repeat=[{"factor_id": "old1", "reason": "stale"}],
        research_objective="volume-price divergence",
    )
    text = packet.to_prompt_text()
    for marker in (
        "[Research objective]", "[Parent set]", "[Logic/Schema]",
        "[Last lineage actions]", "[Best offspring]", "[Failures to avoid]",
        "[Structural nearest]", "[Numerical nearest]", "[Cluster saturation]",
        "[Actions tried]", "[Allowed surface]", "[Legal survival exemplars]",
        "[Avoid repeating]",
    ):
        assert marker in text, f"missing section marker {marker}"


# ---------------------------------------------------------------------------
# 5. retriever 组装点改造（V2 封装，保持既有调用兼容）
# ---------------------------------------------------------------------------


def test_retriever_build_packet_v2_wrapper(store):
    _upsert(store, "p1", "rank(ts_mean(close, 20))", "sig-p", family_id="fam-1")
    _upsert(store, "p2", "ts_std(volume, 20)", "sig-p2", family_id="fam-1")
    _upsert(store, "n1", "ts_mean(close, 5)", "sig-n1", family_id="fam-1")
    retriever = MemoryRetriever(store)
    # V2 组装：传入 parent 池 + roles
    packet = retriever.build_packet(
        "p1",
        parents=[store.get_node("p1"), store.get_node("p2")],
        roles={"p1": "main", "p2": "context"},
        research_version="2026r2",
        allowed_fields=["close", "volume"],
        allowed_operators=["ts_rank", "ts_mean"],
    )
    assert packet is not None
    assert isinstance(packet, MemoryPacketV2)
    text = packet.to_prompt_text()
    assert "[Parent set]" in text
    assert "ts_std(volume, 20)" in text  # 第二 parent 也进入 prompt
    assert "context=" in text


def test_retriever_build_packet_v1_style_still_works(store):
    """既有调用方式（无 parents/roles）仍返回兼容 packet（.parent 可用）。"""
    _upsert(store, "p1", "rank(ts_mean(close, 20))", "sig-p", family_id="fam-1")
    _upsert(store, "n1", "ts_mean(close, 5)", "sig-n1", family_id="fam-1")
    retriever = MemoryRetriever(store)
    packet = retriever.build_packet("p1")
    assert packet is not None
    assert packet.parent["factor_id"] == "p1"
    assert hasattr(packet, "to_prompt_text")
    assert "[Parent set]" in packet.to_prompt_text()
    # 数字邻居等 V1 字段照常
    assert len(packet.numerical_neighbors) <= 5


def test_retriever_unknown_parent_still_returns_none(store):
    retriever = MemoryRetriever(store)
    assert retriever.build_packet("NOT_A_REAL_ID") is None


# ---------------------------------------------------------------------------
# helpers（survival profile 带版本字段）
# ---------------------------------------------------------------------------


def _write_survival_v2(
    store,
    *,
    factor_id: str,
    confidence: float,
    version_key: str = "",
):
    """直接往 survival_profiles 写带 research_version 的行（绕过 SurvivalProfile
    无版本字段的局限——V2 的版本字段由调用方在 payload 里给）。"""
    import json
    import time

    payload = json.dumps(
        {"factor_id": factor_id, "research_version": version_key,
         "version_key": version_key, "status": "HEALTHY"}
    )
    store._conn.execute(
        "INSERT OR REPLACE INTO survival_profiles VALUES (?,?,?,?,?,?)",
        (factor_id, "HEALTHY", payload, confidence, 12, time.time()),
    )
    store._conn.commit()


def _survival_profile(factor_id, status, confidence, support_periods=12, version_key=""):
    class _Profile:
        def __init__(self):
            self.factor_id = factor_id
            self.status = status
            self.confidence = confidence
            self.support_periods = support_periods
            self.version_key = version_key

    return _Profile()
