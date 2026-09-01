"""Global Memory / Persistent Lineage / Source Ingestion（任务书 §33-§43 / §76 / §89）。

§89.4 同 signal rediscovery 不建新节点；§89.5 action_edges 多父 lineage；
§35 冷启动兼容（yaml 不存在 → 0 不抛）；§41 MemoryPacket 定长（<4000 字符）。
"""

from __future__ import annotations

import sqlite3

import pytest

from alphaprobe.memory import GlobalMemoryStore, MemoryPacket, MemoryRetriever
from alphaprobe.memory.seed_ingestion import ingest_cold_start_yaml


@pytest.fixture()
def store(tmp_path):
    s = GlobalMemoryStore(tmp_path / "memory.sqlite3")
    yield s
    s.close()


def _upsert(store, fid, formula, signal_id, family_id=None):
    return store.upsert_factor_node(
        factor_id=fid,
        canonical_formula=formula,
        canonical_ast_hash=f"h:{fid}",
        signal_equivalence_id=signal_id,
        parameter_family_id=family_id,
    )


# -- §89.4：同 signal rediscovery 不建新节点 -----------------------------------


def test_upsert_rediscovery_same_signal_reuses_node(store):
    created, existing = _upsert(store, "f1", "rank(close)", "sig-1")
    assert created is True
    assert existing is None
    # 同 signal、不同 factor_id → rediscovery 合并，不新建
    created2, existing2 = _upsert(store, "f2", "rank(close)", "sig-1")
    assert created2 is False
    assert existing2 == "f1"
    # 节点数仍是 1
    n = store._conn.execute("SELECT COUNT(*) FROM factor_nodes").fetchone()[0]
    assert n == 1
    row = store._conn.execute(
        "SELECT times_seen, rediscovery_count FROM factor_nodes WHERE factor_id='f1'"
    ).fetchone()
    assert row == (2, 1)
    # f2 未建成节点，但 f1 多了 alias 记录
    aliases = store._conn.execute(
        "SELECT COUNT(*) FROM factor_aliases WHERE factor_id='f1'"
    ).fetchone()[0]
    assert aliases >= 1


def test_upsert_distinct_signal_creates_new_node(store):
    _upsert(store, "f1", "rank(close)", "sig-1")
    _upsert(store, "f2", "rank(open)", "sig-2")
    n = store._conn.execute("SELECT COUNT(*) FROM factor_nodes").fetchone()[0]
    assert n == 2


# -- §89.5：action_edges 多父 lineage ------------------------------------------


def test_lineage_multi_parent(store):
    store.insert_action_edge(
        action_id="a1",
        action_type="CROSSOVER",
        parent_factor_ids=["A", "B"],
        child_factor_id="C",
        round_id="r1",
        campaign_id="c1",
        generation=1,
    )
    parents = store.lineage_parents("C")
    assert len(parents) == 2
    pid_set = {p["parent_id"] for p in parents}
    assert pid_set == {"A", "B"}
    for p in parents:
        assert p["action_type"] == "CROSSOVER"
        assert p["generation"] == 1
    # 无子节点 → 空
    assert store.lineage_parents("NOPE") == []


# -- §35：seed_ingestion 冷启动兼容 --------------------------------------------


def test_ingest_cold_start_yaml_missing_file_returns_zero(store, tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    # 不存在 → 0，不抛异常
    assert ingest_cold_start_yaml(missing, store) == 0
    # 空目录（存在但不含 yaml 文件）→ 0，不抛异常
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert ingest_cold_start_yaml(empty_dir, store) == 0
    # 未灌入任何节点
    n = store._conn.execute("SELECT COUNT(*) FROM factor_nodes").fetchone()[0]
    assert n == 0


def test_ingest_cold_start_yaml_entries_with_dedup_identity(store, tmp_path):
    yaml_file = tmp_path / "cold_start.yaml"
    yaml_file.write_text(
        "entries:\n"
        "  - expr: rank(close)\n"
        "    topic: momentum\n"
        "    description: rank close\n"
        "  - expr: ts_mean(close, 20)\n"
        "    topic: trend\n"
        "    description: mean 20\n",
        encoding="utf-8",
    )
    from alphaprobe.dedup import canonical_ast_hash, canonicalize_dsl, signal_equivalence_id

    def identity_fn(formula):
        return (
            canonicalize_dsl(formula),
            canonical_ast_hash(formula),
            signal_equivalence_id(formula),
            None,
        )

    n = ingest_cold_start_yaml(yaml_file, store, identity_fn)
    assert n == 2
    nodes = store._conn.execute(
        "SELECT factor_id, is_seed, lineage_root, exportable FROM factor_nodes"
    ).fetchall()
    assert len(nodes) == 2
    for fid, is_seed, lineage_root, exportable in nodes:
        assert fid.startswith("seed_")
        assert is_seed == 1
        assert lineage_root == 1
        assert exportable == 0
    # 重复摄入同文件 → 全部 rediscovery，created=0
    n2 = ingest_cold_start_yaml(yaml_file, store, identity_fn)
    assert n2 == 0


def test_ingest_cold_start_yaml_relaxed_keys(store, tmp_path):
    """宽松解析：expression/formula + topic + explanation keys。"""
    yaml_file = tmp_path / "relaxed.yaml"
    yaml_file.write_text(
        "entries:\n"
        "  - formula: rank(close)\n"
        "    topic: momentum\n"
        "    explanation: rank close\n"
        "  - expression: ts_mean(close, 20)\n"
        "    topic: trend\n"
        "    explanation: mean 20\n",
        encoding="utf-8",
    )
    n = ingest_cold_start_yaml(yaml_file, store)
    assert n == 2


# -- §41：MemoryPacket 定长（<4000 字符）---------------------------------------


def test_memory_packet_prompt_len_under_4000(store):
    # 造一些邻居/失败/方向数据，把 packet 撑满
    _upsert(store, "p1", "rank(close)", "sig-parent", family_id="fam-1")
    for i in range(8):
        _upsert(
            store,
            f"n{i}",
            f"ts_mean(close, {20 + i})",
            f"sig-n{i}",
            family_id="fam-1",
        )
    for i in range(3):
        store.record_failure(
            factor_id="p1",
            action_family="REFINE",
            reason=f"reason-{i}",
            structural=False,
        )
    for i in range(6):
        store.bump_exploration(
            factor_id="p1",
            action_family=f"ACTION_{i}",
            success=(i % 2 == 0),
            elite=False,
        )
    store.bump_exploration(factor_id="p1", action_family="REFINE", success=False, elite=False)
    for i in range(3):
        store.upsert_direction(cluster_id=f"c{i}", member_count=i + 1)

    retriever = MemoryRetriever(store)
    parent = store.get_node("p1")
    packet = store.build_memory_packet(parent_node=parent)
    packet.numerical_neighbors = retriever.numerical_neighbors("p1")
    text = packet.to_prompt_text()
    assert len(text) < 4000, f"prompt too long: {len(text)}"
    assert "[Parent]" in text

    # build_packet 兼容：未知 factor_id → None
    assert retriever.build_packet("NOT_A_REAL_ID") is None
    built = retriever.build_packet("p1")
    assert built is not None
    assert len(built.numerical_neighbors) <= 5


def test_memory_packet_unknown_parent_returns_none(store):
    retriever = MemoryRetriever(store)
    assert retriever.build_packet("missing") is None
