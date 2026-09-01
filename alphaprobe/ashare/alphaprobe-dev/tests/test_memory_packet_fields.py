"""P1-mem：MemoryPacket 全字段真进 Prompt + Retriever 升级（§41/§43/§48）。

覆盖：
- build_memory_packet 填充 numerical_neighbors（同 family 两条因子）
- survival_profiles 有数据时 survival_exemplars 非空
- to_prompt_text 包含 [Numerical neighbors]/[2026 survival exemplars] 段，
  且空数据时不输出空段
- allowed_fields/allowed_operators 传入即出现在 prompt
- _fitness_from_evaluations 优先于 schema_json（两种来源值不同时取 evaluations）
"""

from __future__ import annotations

from datetime import datetime

import pytest

from alphaprobe.contracts import EvaluationRecord, SurvivalProfile
from alphaprobe.memory import GlobalMemoryStore, MemoryPacket, MemoryRetriever


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


def _record_eval(store, factor_id, ic):
    store.record_evaluation(
        EvaluationRecord(
            factor_id=factor_id,
            segment="2026",
            fidelity="full",
            metric_bundle={"single_ic": ic},
            artifact_refs={},
            evaluator_version="t",
            data_snapshot_id="d",
            universe_snapshot_id="u",
            label_spec_hash="l",
            created_at=datetime(2026, 1, 1),
        )
    )


def _survival_profile(factor_id, status, confidence, support_periods=12):
    return SurvivalProfile(
        factor_id=factor_id,
        status=status,
        confidence=confidence,
        support_periods=support_periods,
    )


# -- build_memory_packet 填充 numerical_neighbors -----------------------------


def test_build_packet_fills_numerical_neighbors(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    _upsert(store, "n1", "ts_mean(close, 20)", "sig-n1", family_id="fam-1")
    _upsert(store, "n2", "ts_std(close, 20)", "sig-n2", family_id="fam-1")

    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    assert packet.numerical_neighbors, "同 family 有邻居却未填充"
    formulas = {n["formula"] for n in packet.numerical_neighbors}
    assert "ts_mean(close, 20)" in formulas
    assert "ts_std(close, 20)" in formulas
    # 排除自身
    assert all(n["factor_id"] != "p1" for n in packet.numerical_neighbors)
    # 每个邻居都有 fitness 键（值可 None）
    assert all("fitness" in n for n in packet.numerical_neighbors)


def test_retriever_build_packet_fills_numerical_neighbors(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    _upsert(store, "n1", "ts_mean(close, 20)", "sig-n1", family_id="fam-1")
    retriever = MemoryRetriever(store)
    packet = retriever.build_packet("p1")
    assert packet is not None
    assert len(packet.numerical_neighbors) == 1


# -- survival_exemplars --------------------------------------------------------


def test_build_packet_fills_survival_exemplars(store):
    _upsert(store, "f1", "rank(close)", "sig-1", family_id="fam-1")
    store.update_survival(_survival_profile("f1", "HEALTHY", 0.92))
    store.update_survival(_survival_profile("other", "BROKEN", 0.1))

    packet = store.build_memory_packet(parent_node=store.get_node("f1"))
    assert packet.survival_exemplars, "survival_profiles 有数据却未填充"
    top = packet.survival_exemplars[0]
    assert top["factor_id"] == "f1"
    assert top["survival_rate"] == 0.92
    assert top["formula"] == "rank(close)"
    # 最多 3 个 exemplar
    assert len(packet.survival_exemplars) <= 3


def test_build_packet_empty_survival_is_empty(store):
    _upsert(store, "f1", "rank(close)", "sig-1", family_id="fam-1")
    packet = store.build_memory_packet(parent_node=store.get_node("f1"))
    assert packet.survival_exemplars == []


# -- to_prompt_text：真输出 + 空数据不出空段 -----------------------------------


def test_prompt_text_contains_new_sections(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    _upsert(store, "n1", "ts_mean(close, 20)", "sig-n1", family_id="fam-1")
    _record_eval(store, "n1", 0.03)
    store.update_survival(_survival_profile("p1", "HEALTHY", 0.9))
    store.update_survival(_survival_profile("n1", "HEALTHY", 0.85))

    packet = store.build_memory_packet(
        parent_node=store.get_node("p1"),
        allowed_fields=["close", "volume"],
        allowed_operators=["ts_rank", "ts_mean"],
    )
    text = packet.to_prompt_text()
    assert "[Numerical neighbors]" in text
    assert "ts_mean(close, 20)(fitness=0.03)" in text
    assert "[2026 survival exemplars]" in text
    assert "survival_rate=0.9" in text
    assert "[Cluster context]" in text
    assert "[Allowed fields]" in text and "close,volume" in text
    assert "[Allowed operators]" in text and "ts_rank,ts_mean" in text


def test_prompt_text_no_empty_sections(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    text = packet.to_prompt_text()
    assert "[Numerical neighbors]" not in text
    assert "[2026 survival exemplars]" not in text
    assert "[Allowed fields]" not in text
    assert "[Allowed operators]" not in text


# -- allowed_fields / allowed_operators 传入即出现 -----------------------------


def test_build_packet_passes_allowed_lists(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    packet = store.build_memory_packet(
        parent_node=store.get_node("p1"),
        allowed_fields=["close", "open", "high", "low", "volume"],
        allowed_operators=["ts_rank", "ts_mean", "delta"],
    )
    assert packet.allowed_fields == ["close", "open", "high", "low", "volume"]
    assert packet.allowed_operators == ["ts_rank", "ts_mean", "delta"]
    text = packet.to_prompt_text()
    assert "[Allowed fields]" in text
    assert "[Allowed operators]" in text

    # retriever.build_packet 透传
    retriever = MemoryRetriever(store)
    built = retriever.build_packet("p1", allowed_fields=["close"], allowed_operators=["rank"])
    assert built is not None
    assert built.allowed_fields == ["close"]
    assert built.allowed_operators == ["rank"]


def test_build_packet_default_allowed_empty(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    assert packet.allowed_fields == []
    assert packet.allowed_operators == []


# -- cluster_context -----------------------------------------------------------


def test_cluster_context_summary(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    store.upsert_direction(cluster_id="c1", member_count=5, survival_rate=0.8)
    store.upsert_direction(cluster_id="c2", member_count=2, survival_rate=0.5)
    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    assert packet.cluster_context is not None
    ctx = packet.cluster_context
    assert ctx["total_clusters"] == 2
    assert ctx["total_members"] == 7
    assert ctx["top"][0]["cluster_id"] == "c1"


def test_cluster_context_empty_is_noneish(store):
    _upsert(store, "p1", "rank(close)", "sig-p", family_id="fam-1")
    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    ctx = packet.cluster_context
    assert ctx is not None
    assert ctx["total_clusters"] == 0
    assert ctx["top"] == []


# -- _fitness_from_evaluations 优先于 schema_json -------------------------------


def test_evaluations_fitness_beats_schema_json(store):
    _upsert(
        store,
        "p1",
        "rank(close)",
        "sig-p",
        family_id="fam-1",
        schema_json={"single_ic": 0.05},
    )
    _upsert(
        store,
        "n1",
        "ts_mean(close, 20)",
        "sig-n1",
        family_id="fam-1",
        schema_json={"single_ic": 0.99},  # schema_json 里的“高值”
    )
    _record_eval(store, "n1", 0.04)  # evaluations 里的真值（低值）

    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    n1 = [n for n in packet.numerical_neighbors if n["factor_id"] == "n1"][0]
    assert n1["fitness"] == 0.04, "evaluations 应优先于 schema_json"

    # retriever 路径同样优先 evaluations
    retriever = MemoryRetriever(store)
    nn = retriever.numerical_neighbors("p1")
    n1b = [n for n in nn if n["factor_id"] == "n1"][0]
    assert n1b["fitness"] == 0.04


def test_schema_json_fallback_when_no_evaluations(store):
    _upsert(
        store,
        "p1",
        "rank(close)",
        "sig-p",
        family_id="fam-1",
        schema_json={"single_ic": 0.05},
    )
    _upsert(
        store,
        "n1",
        "ts_mean(close, 20)",
        "sig-n1",
        family_id="fam-1",
        schema_json={"single_ic": 0.03},
    )
    packet = store.build_memory_packet(parent_node=store.get_node("p1"))
    n1 = [n for n in packet.numerical_neighbors if n["factor_id"] == "n1"][0]
    assert n1["fitness"] == 0.03
