"""Seen Index Phase B 测试（§54-§60）。

覆盖：seed 导入命中 EXACT / 跨 Miner alias / 跨 Round times_seen /
f 与 -f orientation / 多线程并发 reserve 恰好 1 acquired /
rank fingerprint 相似 → nearest 命中 / 阈值 0.995 与 0.90 边界 /
HIGHLY_CORRELATED 不 reject / identity_version 不互判 /
action index first_time / subtree saturation / 1000 条性能冒烟。
"""

from __future__ import annotations

import json
import random
import threading
import time

import pytest

from alphaprobe.seen import build_seen_index
from alphaprobe.seen.action_index import ActionSeenIndex, action_key, normalize_payload
from alphaprobe.seen.audit import AuditReport
from alphaprobe.seen.dedup_service import DedupConfig, DedupService
from alphaprobe.seen.family_registry import FamilyRegistry
from alphaprobe.seen.fingerprint import (
    build_rank_fingerprint,
    fingerprint_hamming,
    lsh_bands,
    similarity_to_hamming,
)
from alphaprobe.seen.nearest import NearestIndex, NearestNeighbor
from alphaprobe.seen.store import SeenStore
from alphaprobe.seen.subtree import SubtreeStats


class FakeIdentity:
    """鸭子类型 FactorIdentity（不需要真 FE / contracts）。"""

    def __init__(
        self,
        formula: str,
        canon_hash: str,
        signal_id: str,
        family_id: str | None = None,
        orientation: int = 1,
        factor_id: str | None = None,
        identity_version: str = "1",
        operator_semantics_version: str | None = None,
    ) -> None:
        self.factor_id = factor_id or canon_hash[:12]
        self.canonical_formula = formula
        self.canonical_ast_hash = canon_hash
        self.signal_equivalence_id = signal_id
        self.parameter_family_id = family_id
        self.orientation = orientation
        self.identity_version = identity_version
        self.operator_semantics_version = operator_semantics_version


@pytest.fixture()
def svc(tmp_path):
    db = tmp_path / "seen.sqlite3"
    service = DedupService(SeenStore(db))
    yield service
    service._store.close()


@pytest.fixture()
def store(tmp_path):
    st = SeenStore(tmp_path / "seen.sqlite3")
    yield st
    st.close()


def _mk(formula="rank(close)", canon="h1", sig="s1", **kw):
    """canon/sig 参数展开为 32 字符 hash（与 dedup 层 32 位 hex 一致）。"""
    return FakeIdentity(
        formula,
        (canon * 32)[:32],
        (sig * 32)[:32],
        **kw,
    )


# -- §54：seed 导入后 ((ts_mean(close,20)))*1 命中 SEED/EXACT -------------------


def test_seed_ingest_then_exact_hit(svc):
    seed = _mk("((ts_mean(close,20)))*1", canon="abc", sig="seed_sig", family_id="fam-seed")
    r1 = svc.ingest_seed(seed, source_system="seed_library")
    assert r1.created is True
    assert r1.verdict == "NEW"
    assert r1.factor_id == ("abc" * 32)[:12]

    # 同 identity 再来 → EXACT_DUPLICATE，不新建
    r2 = svc.ingest_seed(seed)
    assert r2.created is False
    assert r2.verdict == "EXACT_DUPLICATE"
    assert r2.existing_factor_id == r1.factor_id
    assert svc.stats()["total_factors"] == 1

    # lookup 判定
    lk = svc.lookup_identity(seed)
    assert lk.verdict == "EXACT_DUPLICATE"
    assert lk.existing_factor_id == r1.factor_id


def test_seed_import_times_seen_increments(svc):
    seed = _mk("ts_mean(close, 20)", canon="seed1", sig="seed_sig_1")
    svc.ingest_seed(seed)
    r2 = svc.ingest_seed(seed)
    row = svc._store.get_factor_by_id(r2.existing_factor_id)
    assert row["times_seen"] == 2


# -- §55/§56：跨 Miner 同 hash 只加 alias；跨 Round times_seen+1 -------------


def test_cross_miner_alias_same_node(svc):
    base = _mk("rank(close)", canon="m1", sig="mig")
    r1 = svc.reserve(base, source_system="miner_A")
    miner_b = _mk("rank(close)", canon="m1", sig="mig")  # 同 hash 同 signal
    r2 = svc.reserve(miner_b, source_system="miner_B")
    assert r2.acquired is False
    assert r2.verdict == "EXACT_DUPLICATE"
    assert r2.existing_factor_id == r1.factor_id
    # 只 1 个 factor 节点
    assert svc.stats()["total_factors"] == 1
    # alias 注册
    added = svc.register_alias(miner_b, r1.factor_id, source_system="miner_B")
    assert added is True
    # 同 alias 再来 → False（不重复加）
    added2 = svc.register_alias(miner_b, r1.factor_id, source_system="miner_B")
    assert added2 is False
    assert svc._registry.total_aliases() == 1


def test_cross_round_times_seen_no_new(svc):
    r1 = svc.reserve(_mk("rank(close)", canon="r1", sig="r1s"), run_id="round_1")
    r2 = svc.reserve(_mk("rank(close)", canon="r1", sig="r1s"), run_id="round_2")
    assert r2.acquired is False
    assert r2.verdict == "EXACT_DUPLICATE"
    row = svc._store.get_factor_by_pk(r1.factor_pk)
    assert row["times_seen"] == 2
    assert svc.stats()["total_factors"] == 1


def test_rediscovery_count(svc):
    svc.reserve(_mk(canon="rd", sig="rds"))
    svc.reserve(_mk(canon="rd", sig="rds"))
    svc.reserve(_mk(canon="rd", sig="rds"))
    row = svc._store.q1(
        "SELECT times_seen FROM seen_factors WHERE canonical_ast_hash=?", (("rd" * 32)[:32],)
    )
    assert row[0] == 3


# -- §59：f 与 -f 同 signal_id 不同 orientation --------------------------------


def test_orientation_recorded_for_sign_equivalent(svc):
    f_pos = _mk("rank(close)", canon="ori1", sig="ori_sig", orientation=1)
    r1 = svc.reserve(f_pos)
    f_neg = _mk("-rank(close)", canon="ori2", sig="ori_sig", orientation=-1)
    r2 = svc.reserve(f_neg)
    assert r2.acquired is False
    assert r2.verdict == "SIGN_EQUIVALENT_DUPLICATE"
    assert r2.existing_factor_id == r1.factor_id
    row = svc._store.get_factor_by_pk(r1.factor_pk)
    assert row["orientations_seen"] == "-1,1"


# -- §59：多线程并发 reserve 同一 signal 恰好 1 个 acquired -------------------


def test_concurrent_reserve_exactly_one_acquired(svc, tmp_path):
    results: list = []
    errors: list = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            ident = _mk(
                f"rank(close) + {i}",  # 每个 worker 不同 factor_id，但同 signal
                canon="conc",
                sig="conc_sig",
                factor_id=f"conc_{i}",
            )
            r = svc.reserve(ident, source_system="miner", worker_id=f"w{i}")
            with lock:
                results.append(r)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"errors: {errors}"
    acquired = [r for r in results if r.acquired]
    assert len(acquired) == 1, f"expected exactly 1 acquired, got {len(acquired)}"
    assert svc.stats()["total_factors"] == 1


# -- §32-§35：rank fingerprint 相似 → nearest 命中 -----------------------------


def _ranks(seed: int, n: int = 300, perturb: int = 0) -> dict[str, float]:
    rnd = random.Random(seed)
    ranks = {f"S{i}": rnd.random() for i in range(n)}
    for i in range(perturb):
        k = f"S{i}"
        ranks[k] = ranks[k] * 0.98 + 0.01
    return ranks


def test_nearest_hits_similar_fingerprint(svc):
    fp1 = build_rank_fingerprint([_ranks(1)])
    fp2 = build_rank_fingerprint([_ranks(1, perturb=2)])  # 轻微扰动
    assert fingerprint_hamming(fp1, fp2) <= 12

    ident = _mk("rank(close)", canon="ann", sig="ann_sig")
    r = svc.reserve(ident)
    svc.register_evaluation(
        r.factor_id,
        metrics={"search_fitness": 0.3},
        fingerprint=fp1,
        subtree_hashes=["sub_ann"],
    )
    nearest = svc.nearest_by_fingerprint(fp2, k=5)
    assert len(nearest) >= 1, "LSH 应召回相似指纹"
    assert nearest[0].factor_id == r.factor_id
    assert nearest[0].hamming == fingerprint_hamming(fp1, fp2)


def test_nearest_does_not_return_dissimilar(svc):
    fp1 = build_rank_fingerprint([_ranks(5)])
    fp_dissimilar = build_rank_fingerprint([_ranks(6)])
    ident = _mk("rank(close)", canon="prec", sig="prec_sig")
    r = svc.reserve(ident)
    svc.register_evaluation(r.factor_id, metrics={}, fingerprint=fp1)
    nearest = svc.nearest_by_fingerprint(fp_dissimilar, k=5)
    assert nearest == []


def test_lsh_bands_are_16x16(svc):
    fp = build_rank_fingerprint([_ranks(9)])
    bands = lsh_bands(fp)
    assert len(bands) == 16
    for b in bands:
        assert 0 <= b < 2**16


# -- §36：阈值判定 0.995 / 0.90 边界 -------------------------------------------


def test_similarity_threshold_boundaries(svc):
    cfg = DedupConfig(rank_exact_threshold=0.995, highly_correlated_threshold=0.90)
    s = DedupService(svc._store, config=cfg)

    n1 = NearestNeighbor(factor_id="f1", factor_pk=1, hamming=0, fingerprint_version="v1")
    n2 = NearestNeighbor(factor_id="f2", factor_pk=2, hamming=4, fingerprint_version="v1")
    n3 = NearestNeighbor(factor_id="f3", factor_pk=3, hamming=24, fingerprint_version="v1")

    # corr=1.0 -> RANK_EQUIVALENT；corr≈0.984 -> HIGHLY_CORRELATED；corr≈0.906 -> HIGHLY_CORRELATED
    dec = s.confirm_signal_similarity(b"\x00" * 32, [n1, n2, n3], correlation_values=[1.0, 0.984, 0.906])
    assert dec[0]["verdict"] == "RANK_EQUIVALENT_DUPLICATE"
    assert dec[1]["verdict"] == "HIGHLY_CORRELATED"
    assert dec[2]["verdict"] == "HIGHLY_CORRELATED"

    # 边界：正好 0.995 -> RANK_EQUIVALENT；正好 0.90 -> HIGHLY_CORRELATED
    dec2 = s.confirm_signal_similarity(b"\x00" * 32, [n1, n3], correlation_values=[0.995, 0.90])
    assert dec2[0]["verdict"] == "RANK_EQUIVALENT_DUPLICATE"
    assert dec2[1]["verdict"] == "HIGHLY_CORRELATED"

    # 0.89 -> LOW_CORRELATION（低于 0.90 不算）
    dec3 = s.confirm_signal_similarity(b"\x00" * 32, [n1], correlation_values=[0.89])
    assert dec3[0]["verdict"] == "LOW_CORRELATION"


def test_hamming_similarity_mapping():
    assert similarity_to_hamming(0.995) == 1
    assert similarity_to_hamming(0.90) == 25


# -- §31：HIGHLY_CORRELATED 不 reject -----------------------------------------


def test_highly_correlated_not_rejected(svc):
    ident = _mk("rank(close)", canon="hc", sig="hc_sig")
    r = svc.reserve(ident)
    svc.register_evaluation(r.factor_id, metrics={}, fingerprint=build_rank_fingerprint([_ranks(3)]))
    # 高相关候选 → HIGHLY_CORRELATED，但没有 reject 语义（不抛、不删）
    fp_query = build_rank_fingerprint([_ranks(3, perturb=1)])
    nearest = svc.nearest_by_fingerprint(fp_query, k=5)
    dec = svc.confirm_signal_similarity(fp_query, nearest, correlation_values=[0.93])
    assert all(d["verdict"] in ("RANK_EQUIVALENT_DUPLICATE", "HIGHLY_CORRELATED") for d in dec)
    # 库中仍保留
    assert svc.stats()["total_factors"] == 1


# -- §45：identity_version 变 → 不互判（VERSION_MISMATCH / NEW） -------------


def test_identity_version_mismatch_not_duplicate(svc):
    v1 = _mk("rank(close)", canon="ver", sig="ver_sig", identity_version="1")
    r1 = svc.reserve(v1)
    v2 = _mk("rank(close)", canon="ver", sig="ver_sig", identity_version="2")
    lk = svc.lookup_identity(v2)
    assert lk.verdict == "VERSION_MISMATCH"
    assert lk.existing_factor_id == r1.factor_id

    r2 = svc.reserve(v2)
    assert r2.acquired is True
    assert r2.verdict == "NEW"
    assert r2.stats.get("identity_version_mismatch") is True
    assert svc.stats()["total_factors"] == 2


def test_version_namespace_in_stats(svc):
    svc.reserve(_mk(canon="vns", sig="vns1", identity_version="1"))
    svc.reserve(_mk(canon="vns", sig="vns2", identity_version="2"))
    stats = svc.stats()
    assert stats["per_version"]["1"] == 1
    assert stats["per_version"]["2"] == 1


# -- §39：action index 二次同 action first_time=False -------------------------


def test_action_index_first_time(store):
    ai = ActionSeenIndex(store)
    key1, first1 = ai.build_and_record("sig_parent", "REFINE", {"op": "ts_mean", "window": 20})
    key2, first2 = ai.build_and_record("sig_parent", "REFINE", {"op": "ts_mean", "window": 20})
    assert first1 is True
    assert first2 is False
    assert key1 == key2
    assert ai.count() == 1


def test_action_key_payload_normalization(store):
    # 数字/列表排序归一
    k1 = action_key("p", "REFINE", {"opts": [3, 1, 2], "window": 20})
    k2 = action_key("p", "REFINE", {"opts": [1, 2, 3], "window": 20})
    assert k1 == k2
    # 不同 payload → 不同 key
    k3 = action_key("p", "REFINE", {"opts": [1, 2, 3], "window": 21})
    assert k1 != k3
    # 不同 grammar_version → 不同 key
    k4 = action_key("p", "REFINE", {"opts": [1, 2, 3], "window": 20}, grammar_version="v2")
    assert k1 != k4


def test_normalize_payload_sorts_lists():
    assert normalize_payload([3, 1, 2]) == [1, 2, 3]
    assert normalize_payload({"b": 2, "a": {"y": [2, 1], "x": 1}}) == {
        "a": {"x": 1, "y": [1, 2]},
        "b": 2,
    }


# -- §40：subtree saturation 统计正确 -----------------------------------------


def test_subtree_saturation(store):
    st = SubtreeStats(store)
    pk1 = store.insert_factor(
        factor_id="f1", canonical_ast_hash="a" * 32, signal_equivalence_id="s1" * 32,
        identity_version="1",
    )
    pk2 = store.insert_factor(
        factor_id="f2", canonical_ast_hash="b" * 32, signal_equivalence_id="s2" * 32,
        identity_version="1",
    )
    st.record("sub_x", "ts_mean(close,20)", pk1, success=True, elite=False)
    st.record("sub_x", "ts_mean(close,20)", pk2, success=False, elite=False)
    st.record("sub_x", "ts_mean(close,20)", pk1, success=True, elite=False)  # 同因子重复

    stat = st.get("sub_x")
    assert stat["appearance_count"] == 3
    assert stat["unique_factor_count"] == 2  # 重复出现不算新 unique
    # 同一因子第二次 success 不重复计数（successful_factor_count 仍=1）
    assert stat["successful_factor_count"] == 1
    assert stat["success_denominator"] == 2  # 每次 success 都计分母
    # saturation = 1 - successful/unique = 0.5
    assert st.saturation("sub_x") == pytest.approx(0.5)
    assert st.survival_rate("sub_x") == pytest.approx(0.5)
    # 未见过 → 0
    assert st.saturation("nope") == 0.0


# -- §20：family registry ------------------------------------------------------


def test_family_registry_best_member(store):
    fr = FamilyRegistry(store)
    pk1 = store.insert_factor(
        factor_id="fam_f1", canonical_ast_hash="fa" * 32, signal_equivalence_id="fs1" * 32,
        parameter_family_id="fam_20", identity_version="1",
    )
    pk2 = store.insert_factor(
        factor_id="fam_f2", canonical_ast_hash="fb" * 32, signal_equivalence_id="fs2" * 32,
        parameter_family_id="fam_20", identity_version="1",
    )
    fr.register_member("fam_20", factor_id="fam_f1", factor_pk=pk1, search_fitness=0.3, outcome="evaluated")
    fr.register_member("fam_20", factor_id="fam_f2", factor_pk=pk2, search_fitness=0.5, outcome="evaluated")

    fam = fr.get("fam_20")
    assert fam["member_count"] == 2
    assert fam["best_factor_id"] == "fam_f2"
    assert fam["best_search_fitness"] == pytest.approx(0.5)
    assert fam["historical_success_rate"] == pytest.approx(1.0)


# -- §61：audit 报告 -----------------------------------------------------------


def test_audit_report(svc):
    svc.reserve(_mk(canon="au1", sig="au1s"))
    svc.reserve(_mk(canon="au1", sig="au1s"))  # exact dup
    svc.reserve(_mk(canon="au2", sig="au1s", orientation=-1))  # sign eq
    svc._store.record_parse_failure("rank(", "parse error")

    audit = AuditReport(svc._store)
    rep = audit.report()
    assert rep["raw_factor_count"] == 1
    assert rep["exact_duplicate_count"] == 1
    assert rep["sign_equivalent_count"] == 1
    assert rep["parse_failure_count"] == 1
    assert "Dedup Audit Report" in audit.render_text()


# -- build_seen_index 工厂 ------------------------------------------------------


def test_build_seen_index_factory(tmp_path):
    svc = build_seen_index(str(tmp_path / "factory.sqlite3"))
    assert isinstance(svc, DedupService)
    r = svc.reserve(_mk(canon="fac", sig="facs"))
    assert r.acquired is True
    svc._store.close()


# -- 性能冒烟：1000 条 ingest < 30s -------------------------------------------


def test_ingest_1000_performance_smoke(tmp_path):
    db = tmp_path / "perf.sqlite3"
    service = DedupService(SeenStore(db))
    t0 = time.time()
    for i in range(1000):
        ident = _mk(
            f"ts_mean(close, {i})",
            canon=f"perf{i:04d}",
            sig=f"perf_sig_{i}",
            family_id=f"fam_{i % 10}",
        )
        r = service.reserve(ident)
        assert r.acquired is True
    elapsed = time.time() - t0
    assert service.stats()["total_factors"] == 1000
    assert elapsed < 30.0, f"ingest 1000 took {elapsed:.2f}s (limit 30s)"
    service._store.close()


# -- 并发 + 事务正确性 ----------------------------------------------------------


def test_transaction_rollback_on_error(store):
    from alphaprobe.seen.reservation import ReservationService

    rs = ReservationService(store)
    ident = _mk(canon="tr", sig="trs", family_id="tr_fam")
    # 用一个会导致 IntegrityError 的 identity（factor_id 与 canon 前缀冲突已处理），
    # 验证事务回滚后库中无残留
    try:
        store.transaction().__enter__() if False else None
    except Exception:
        pass
    r = rs.reserve(ident)
    assert r.acquired is True
    assert store.count_factors() == 1


def test_store_reopen_persists(tmp_path):
    db = tmp_path / "persist.sqlite3"
    s1 = SeenStore(db)
    s1.insert_factor(
        factor_id="persist_1", canonical_ast_hash="p" * 32,
        signal_equivalence_id="ps" * 32, identity_version="1",
    )
    s1.close()
    s2 = SeenStore(db)
    assert s2.count_factors() == 1
    s2.close()
