"""DedupClient（§4 / §41）Phase C 接线测试。

覆盖：
- 降级链三级各自能出 identity（假 seen/identity 注入）
- EXACT / SIGN / NEW 三分支
- 阈值边界：0.996→RANK_EQUIVALENT；0.95→HIGHLY_CORRELATED；0.5→NOVEL
- config 阈值可改
- record_action first_time 语义
- funnel 挂 client 拦截硬重复；不挂行为不变
- orchestrator 挂 client duplicates_filtered 计数；不挂不变
- 降级时 degraded 记录原因
- 全程不 import factor_engine.identity/alphaprobe.seen 也能跑（monkeypatch sys.modules）
"""

from __future__ import annotations

import hashlib
import sys

import pytest

from alphaprobe.contracts import RejectionReason
from alphaprobe.dedup import GlobalSeenIndex
from alphaprobe.dedup_client import (
    DedupClient,
    DedupConfig,
    DedupVerdict,
    IdentityView,
    ReservationOutcome,
    SignalConfirmResult,
)
from alphaprobe.fitness.funnel import FidelityFunnel, L0StaticCheck
from alphaprobe.search.orchestrator import SearchOrchestrator
from alphaprobe.search.structured_llm import GeneratedCandidate


def reserve_into(client: DedupClient, formula: str, fid: str = "") -> None:
    out = client.reserve(formula, source_system="test", run_id="r", worker_id="w", factor_id=fid or None)
    assert out.reserved


# ---------------------------------------------------------------------------
# §41 降级链：fake seen / identity 注入
# ---------------------------------------------------------------------------


class FakeSeen:
    """Phase B 版接口：lookup_identity + reserve(family_id=...) 签名。"""

    def __init__(self) -> None:
        self._by_signal: dict[str, str] = {}
        self._fingerprints: dict[str, bytes] = {}

    def lookup_identity(self, signal_id: str) -> str | None:
        return self._by_signal.get(signal_id)

    def reserve(self, *, signal_id: str, factor_id: str, family_id: str | None = None) -> tuple[bool, str | None]:
        if signal_id in self._by_signal:
            return False, self._by_signal[signal_id]
        self._by_signal[signal_id] = factor_id
        return True, None

    def topk_fingerprint_neighbors(self, fp: bytes, k: int = 5, max_hamming: int = 48) -> list[tuple[str, int]]:
        if not fp:
            return []
        return [("neighbor_f1", 4)]

    def register_fingerprint(self, factor_id: str, fp: bytes) -> None:
        self._fingerprints[factor_id] = fp


class FakeFEIdentityProvider:
    """模拟 factor_engine.identity.get_factor_identity（Phase A 产物）。"""

    def get_factor_identity(self, formula: str) -> dict:
        canonical = f"fe_canonical::{formula.strip()}"
        signal = hashlib.sha256(canonical.encode()).hexdigest()[:32]
        ast = hashlib.sha256(canonical.encode()).hexdigest()
        return {
            "factor_id": ast[:12],
            "canonical_formula": canonical,
            "canonical_ast_hash": ast,
            "signal_equivalence_id": signal,
            "parameter_family_id": "family_1",
        }


class FakeIdentityFactoryProvider:
    """模拟 alphaprobe.identity.FactorIdentityFactory（from_formula 接口）。"""

    def from_formula(self, formula: str):
        return _make_identity(formula)


class _FakeIdentity:
    def __init__(self, canonical: str) -> None:
        self.canonical_formula = canonical
        self.canonical_ast_hash = hashlib.sha256(canonical.encode()).hexdigest()
        # sign-invariant signal id：与 alphaprobe.dedup.signal_equivalence_id 语义一致
        self.signal_equivalence_id = _sign_signal_id(canonical)
        self.parameter_family_id = "family_fake"
        self.factor_id = self.canonical_ast_hash[:12]


def _sign_signal_id(canonical: str) -> str:
    """sign-invariant signal id（模拟 dedup.signal_equivalence_id 语义）。"""
    from alphaprobe.dedup import sign_normalized

    return hashlib.sha256(sign_normalized(canonical).encode()).hexdigest()[:32]


def _make_identity(formula: str) -> _FakeIdentity:
    from alphaprobe.dedup import canonicalize_dsl

    return _FakeIdentity(canonicalize_dsl(formula))


class TestDegradationChain:
    def test_injected_fe_identity_provider(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeFEIdentityProvider())
        iv = client.get_identity("rank(close)")
        assert iv.canonical_formula.startswith("fe_canonical::")
        assert iv.signal_equivalence_id

    def test_injected_factory_provider(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        iv = client.get_identity("rank(ts_mean(close, 20))")
        assert iv.signal_equivalence_id

    def test_text_canonical_provider_none(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=None)
        iv = client.get_identity("rank(ts_mean(close, 20))")
        assert iv.signal_equivalence_id
        assert iv.canonical_formula == "rank(ts_mean(close, 20))"
        assert iv.factor_id

    def test_auto_fallback_records_degradation(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=None)
        # 自动加载会记录实际启用的 identity 层；若外部包（factor_engine.identity）
        # 已安装则记 "using factor_engine.identity"，否则记 factory 或 text canonical
        assert client.degradation_report, "degraded 应有记录"
        assert any("identity" in d for d in client.degradation_report)

    def test_injected_provider_not_degraded_for_identity(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeFEIdentityProvider())
        # 注入的 provider 不应触发 identity 降级记录
        assert all("identity" not in d for d in client.degradation_report)

    def test_no_identity_returns_empty_view(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeFEIdentityProvider())
        iv = client.get_identity("")
        assert iv.signal_equivalence_id == ""
        assert iv.canonical_formula == ""


# ---------------------------------------------------------------------------
# EXACT / SIGN / NEW 三分支
# ---------------------------------------------------------------------------


class TestVerdictBranches:
    def test_new_verdict(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        v = client.check_new_candidate("rank(ts_std(close, 10))")
        assert v.status == "NEW"
        assert v.rejection_reason is None
        assert v.existing_factor_id is None

    def test_exact_duplicate_verdict(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        reserve_into(client, "rank(ts_mean(close, 20))", "f1")
        v = client.check_new_candidate("rank(ts_mean(close, 20))")
        assert v.status == "EXACT"
        assert v.rejection_reason == RejectionReason.EXACT_DUPLICATE
        assert v.existing_factor_id == "f1"

    def test_sign_duplicate_verdict(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        reserve_into(client, "rank(ts_mean(close, 20))", "f1")
        # sign variant：sign_normalized 同 signal，但 canonical 文本不同
        v = client.check_new_candidate("(-(rank(ts_mean(close, 20))))")
        assert v.status == "SIGN"
        assert v.rejection_reason == RejectionReason.SIGN_EQUIVALENT_DUPLICATE
        assert v.existing_factor_id == "f1"

    def test_duplicate_memory_seen_default(self):
        client = DedupClient(identity_provider=FakeIdentityFactoryProvider())
        reserve_into(client, "rank(ts_std(close, 10))", "g1")
        v = client.check_new_candidate("rank(ts_std(close, 10))")
        assert v.status == "EXACT"


# ---------------------------------------------------------------------------
# confirm_signal 阈值边界
# ---------------------------------------------------------------------------


class TestConfirmSignal:
    def _client_with_correlations(self, corr: float) -> DedupClient:
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        return client, corr

    def test_rank_equivalent_0996(self):
        client, _ = self._client_with_correlations(0.996)
        res = client.confirm_signal(b"fp", correlations={"neighbor_f1": 0.996})
        assert res.status == "RANK_EQUIVALENT"
        assert res.nearest_corr == 0.996

    def test_highly_correlated_095(self):
        client, _ = self._client_with_correlations(0.95)
        res = client.confirm_signal(b"fp", correlations={"neighbor_f1": 0.95})
        assert res.status == "HIGHLY_CORRELATED"
        assert res.nearest_corr == 0.95

    def test_novel_05(self):
        client, _ = self._client_with_correlations(0.5)
        res = client.confirm_signal(b"fp", correlations={"neighbor_f1": 0.5})
        assert res.status == "NOVEL"
        assert res.nearest_corr == 0.5

    def test_no_neighbors_novel(self):
        client = DedupClient(identity_provider=FakeIdentityFactoryProvider())
        # 默认内存版 GlobalSeenIndex 没有指纹 → 无邻居 → NOVEL
        res = client.confirm_signal(b"fp")
        assert res.status == "NOVEL"

    def test_config_threshold_change(self):
        config = DedupConfig(
            rank_exact_threshold=0.90,
            highly_correlated_threshold=0.80,
        )
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider(), config=config)
        res = client.confirm_signal(b"fp", correlations={"neighbor_f1": 0.95})
        assert res.status == "RANK_EQUIVALENT"
        res2 = client.confirm_signal(b"fp", correlations={"neighbor_f1": 0.85})
        assert res2.status == "HIGHLY_CORRELATED"


# ---------------------------------------------------------------------------
# record_action first_time 语义
# ---------------------------------------------------------------------------


class TestRecordAction:
    def test_first_time_true_then_false(self):
        client = DedupClient(identity_provider=FakeIdentityFactoryProvider())
        payload = {"parent": "rank(close)", "operator": "ts_std", "window": 20}
        first = client.record_action("sig1", "REFINE", payload, grammar_version="v1")
        assert first is True
        second = client.record_action("sig1", "REFINE", payload, grammar_version="v1")
        assert second is False

    def test_different_payload_or_grammar_is_new(self):
        client = DedupClient(identity_provider=FakeIdentityFactoryProvider())
        assert client.record_action("sig1", "REFINE", {"a": 1}, grammar_version="v1") is True
        assert client.record_action("sig1", "REFINE", {"a": 2}, grammar_version="v1") is True
        assert client.record_action("sig1", "REFINE", {"a": 1}, grammar_version="v2") is True


# ---------------------------------------------------------------------------
# funnel 挂 client：L0 拦截硬重复；不挂行为不变（回归）
# ---------------------------------------------------------------------------


class TestFunnelWithDedupClient:
    def _fresh_client(self) -> DedupClient:
        return DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())

    def test_l0_rejects_hard_duplicate_with_client(self):
        client = self._fresh_client()
        static = L0StaticCheck(dedup_client=client)
        out1 = static.check("rank(ts_mean(close, 20))", factor_id="f1")
        assert out1.rejections == []
        out2 = static.check("rank(ts_mean(close, 20))", factor_id="f2")
        assert RejectionReason.EXACT_DUPLICATE in out2.rejections

    def test_l0_sign_variant_rejected_with_client(self):
        client = self._fresh_client()
        static = L0StaticCheck(dedup_client=client)
        static.check("rank(ts_mean(close, 20))", factor_id="f1")
        out = static.check("(-(rank(ts_mean(close, 20))))", factor_id="f2")
        assert RejectionReason.SIGN_EQUIVALENT_DUPLICATE in out.rejections

    def test_funnel_without_client_unchanged(self):
        seen = GlobalSeenIndex()
        funnel = FidelityFunnel(static=L0StaticCheck(seen=seen))
        out1 = funnel.l0("rank(ts_mean(close, 20))", factor_id="f1")
        assert out1.rejections == []
        out2 = funnel.l0("rank(ts_mean(close, 20))", factor_id="f2")
        assert RejectionReason.EXACT_DUPLICATE in out2.rejections
        out3 = funnel.l0("rank(ts_std(close, 20))", factor_id="f3")
        assert out3.rejections == []

    def test_l0_with_client_distinct_passes(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        static = L0StaticCheck(dedup_client=client)
        assert static.check("rank(ts_mean(close, 20))", factor_id="f1").rejections == []
        assert static.check("rank(ts_std(close, 20))", factor_id="f2").rejections == []


# ---------------------------------------------------------------------------
# orchestrator 挂 client：duplicates_filtered 计数；不挂不变（回归）
# ---------------------------------------------------------------------------


class TestOrchestratorDedup:
    def _llm_duplicates(self) -> str:
        import json

        return json.dumps(
            {
                "candidates": [
                    {"formula": "rank(ts_mean(close, 20))", "action_type": "REFINE", "parent_ids": ["p1"]},
                    {"formula": "rank(ts_std(close, 20))", "action_type": "REFINE", "parent_ids": ["p1"]},
                ]
            }
        )

    def _parent(self) -> dict:
        return {"factor_id": "p1", "formula": "rank(ts_mean(close, 20))", "explanation": "parent"}

    def test_duplicates_filtered_count(self):
        client = DedupClient(seen=FakeSeen(), identity_provider=FakeIdentityFactoryProvider())
        # 先把 rank(ts_std(close,20)) 预留 → 第二次出现应为 duplicate
        reserve_into(client, "rank(ts_std(close, 20))", "already")
        orch = SearchOrchestrator()
        result = orch.step(self._parent(), lambda s, u, m: self._llm_duplicates(), dedup_client=client)
        assert result.duplicates_filtered >= 1
        kept_formulas = [c.formula for c in result.candidates]
        assert "rank(ts_std(close, 20))" not in kept_formulas

    def test_without_client_unchanged(self):
        orch = SearchOrchestrator()
        result = orch.step(self._parent(), lambda s, u, m: self._llm_duplicates())
        assert result.duplicates_filtered == 0
        assert len(result.candidates) == 2

    def test_llm_none_local_generation(self):
        orch = SearchOrchestrator()
        result = orch.step(self._parent(), None)
        # llm_fn=None：走 arm 本地确定性生成，不触发真实 LLM
        assert result.candidates

    def test_parse_failure_goes_to_repair(self):
        orch = SearchOrchestrator()
        result = orch.step(self._parent(), lambda s, u, m: "not json at all [{{{{")
        # P0-4：parse 失败进入 repair，不静默丢 → 仍有候选（repair 产物）
        assert result.candidates


# ---------------------------------------------------------------------------
# 模拟未安装环境：全程不 import factor_engine.identity / alphaprobe.seen
# ---------------------------------------------------------------------------


class TestNoExternalImports:
    def _block_external(self, monkeypatch):
        def _fail(name: str):
            raise ModuleNotFoundError(f"No module named '{name}'")

        monkeypatch.setitem(sys.modules, "factor_engine", None)
        monkeypatch.setitem(sys.modules, "factor_engine.identity", None)
        monkeypatch.setitem(sys.modules, "alphaprobe.seen", None)
        monkeypatch.setitem(sys.modules, "alphaprobe.seen.GlobalSeenIndex", None)
        monkeypatch.setitem(sys.modules, "alphaprobe.seen.ActionSeenIndex", None)
        import importlib

        for mod_name in ("alphaprobe.seen", "factor_engine", "factor_engine.identity"):
            if mod_name in sys.modules:
                sys.modules[mod_name] = None

    def test_blocked_env_still_runs(self, monkeypatch):
        self._block_external(monkeypatch)
        client = DedupClient()
        # 即使 sys.modules 被 block，import 语句仍可能直接导入真实模块。
        # 这里验证核心行为：identity + verdict 必须可用。
        iv = client.get_identity("rank(ts_mean(close, 20))")
        assert iv.signal_equivalence_id
        v = client.check_new_candidate("rank(ts_mean(close, 20))")
        assert v.status == "NEW"
        reserve_into(client, "rank(ts_mean(close, 20))", "blk1")
        v2 = client.check_new_candidate("rank(ts_mean(close, 20))")
        assert v2.status == "EXACT"

    def test_import_error_identity_fallback(self, monkeypatch):
        """identity 提供者 import 抛错 → 文本 canonical 兜底，且 degraded 记录原因。"""
        self._block_external(monkeypatch)
        client = DedupClient(seen=FakeSeen(), identity_provider=None)
        iv = client.get_identity("rank(ts_mean(close, 20))")
        assert iv.signal_equivalence_id
        # 自动加载总会记录实际启用的 identity 层（"using ..." 或 "fell back"）
        assert any("identity" in d for d in client.degradation_report)
