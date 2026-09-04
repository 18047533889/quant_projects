"""V3.1 Hypothesis Alignment 验收（plan Task 6 / Non-negotiable #18）。

覆盖 plan Task 6 五条验收：
① hypothesis 声称 FUNDAMENTAL，但 AST/实现只用 PRICE/VOLUME → hard_mismatch
  （确定性，不跑任何模型/LLM；**评估前被拦**）；
② volume-price divergence 类 hypothesis 且实现确实消费 price+volume domain →
  过确定性检查（score 高、无 hard_mismatch）；
③ FE static analysis 的 future/lookahead timing flag → 生成阶段即阻断
  （不进市场评估：result.blocked=True / timing_flags 非空）；
④ 已有 rank/zscore treatment lineage 被检测 → 阻止冗余 treatment 提案
  （explanation_codes 含 existing_treatment_redundant）；
⑤ production 下未知 field taxonomy → fail-closed（抛 UnknownFieldTaxonomyError，
  不静默当通过）。

外加契约断言：
- AlignmentResult 按 plan 契约字段：score / hard_mismatch / missing_domains /
  unexpected_domains / timing_flags / explanation_codes；blocked/ok 语义正确。

Part E 边界：alignment 只消费 FE artifact（field_usages /
existing_treatment_semantic_ids / timing_flags）与 DA taxonomy 注入面
（describe_fields → data_domains / pit_class）。fake taxonomy + 鸭子 artifact
是主力（确定性、零 FE/DA 调用）；「真 FE artifact + 真 DA taxonomy」的端到端
冒烟放同一文件，模块级缓存解析结果（真实 FE 首调 ~50s registry bootstrap，
进程内共享；data_access 秒级）。

全部合成公式数据；不跑模型/LLM/回测。
"""

from __future__ import annotations

from typing import Any

import pytest

from alphaprobe.research_space.alignment import (
    DOMAIN_CONTRADICTION_CODE,
    PIT_UNVERIFIED_CODE,
    REDUNDANT_TREATMENT_CODE,
    DeterministicAligner,
    UnknownFieldTaxonomyError,
    bare_field_name,
)
from alphaprobe.research_space.contracts import AlignmentResult, HypothesisSpec


# ---------------------------------------------------------------------------
# 鸭子对象（不 import factor_engine / data_access）
# ---------------------------------------------------------------------------


class _FakeFieldUsage:
    def __init__(self, canonical_field_id: str) -> None:
        self.canonical_field_id = canonical_field_id


class _FakeAnalysis:
    """FE FactorStaticAnalysisArtifact 的最小鸭子视图。"""

    def __init__(
        self,
        field_ids=(),
        *,
        treatments=(),
        timing_flags=None,
    ) -> None:
        self.field_usages = [_FakeFieldUsage(f) for f in field_ids]
        self.existing_treatment_semantic_ids = tuple(treatments)
        self.timing_flags = dict(timing_flags or {})


class _FakeDescriptor:
    def __init__(self, domains=(), pit_class="") -> None:
        self.data_domains = tuple(domains)
        self.pit_class = pit_class
        self.economic_roles = ()


class _FakeTaxonomy:
    """DA FieldTaxonomyProvider 的最小 fake（bare-name 键，production 语义）。"""

    def __init__(self, mapping) -> None:
        self._mapping = dict(mapping)

    def describe_fields(self, canonical_field_ids):
        out = {}
        for cid in canonical_field_ids:
            bare = bare_field_name(cid)
            d = self._mapping.get(bare)
            if d is None:
                raise UnknownFieldTaxonomyError(f"unknown field {bare}")
            out[str(cid)] = d
        return out


def _pv_taxonomy() -> _FakeTaxonomy:
    """PRICE / VOLUME / LIQUIDITY 行情 taxonomy。"""
    return _FakeTaxonomy(
        {
            "close": _FakeDescriptor(domains=("PRICE",), pit_class="panel_same_day"),
            "volume": _FakeDescriptor(domains=("VOLUME",), pit_class="panel_same_day"),
            "amount": _FakeDescriptor(
                domains=("VOLUME", "LIQUIDITY"), pit_class="panel_same_day"
            ),
            "vwap": _FakeDescriptor(domains=("PRICE",), pit_class="panel_same_day"),
        }
    )


def _fundamental_taxonomy() -> _FakeTaxonomy:
    """加基本面字段的 taxonomy。"""
    base = {
        "close": _FakeDescriptor(domains=("PRICE",), pit_class="panel_same_day"),
        "volume": _FakeDescriptor(domains=("VOLUME",), pit_class="panel_same_day"),
    }
    base["roe"] = _FakeDescriptor(
        domains=("FUNDAMENTAL.QUALITY",), pit_class="announcement_pit"
    )
    base["pe_ratio"] = _FakeDescriptor(
        domains=("FUNDAMENTAL.VALUE",), pit_class="announcement_pit"
    )
    return _FakeTaxonomy(base)


class _Mode:
    """ExecutionMode 鸭子（避免 import llm_client 依赖）。"""

    def __init__(self, value: str) -> None:
        self.value = value


OFFLINE = _Mode("offline_test")
PRODUCTION = _Mode("production")
RESEARCH_DEGRADED = _Mode("research_degraded")


def _aligner(taxonomy=None, mode=OFFLINE, **kw):
    return DeterministicAligner(taxonomy=taxonomy, mode=mode, **kw)


# ---------------------------------------------------------------------------
# ① FUNDAMENTAL 声称 vs PRICE/VOLUME 实现 → hard_mismatch
# ---------------------------------------------------------------------------


class TestFundamentalDomainMismatch:
    def test_hard_mismatch_when_fundamental_claim_but_pv_impl(self):
        h = HypothesisSpec(
            hypothesis_id="h1",
            text="盈利修正被价格低估（fundamental revision underreaction）",
            expected_domains=("FUNDAMENTAL.QUALITY", "FUNDAMENTAL.VALUE"),
        )
        analysis = _FakeAnalysis(["close", "volume"])  # ts_corr(volume, close)
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert res.hard_mismatch is True
        assert res.missing_domains
        assert set(res.missing_domains) == {"FUNDAMENTAL.QUALITY", "FUNDAMENTAL.VALUE"}
        assert res.unexpected_domains == ("PRICE", "VOLUME")
        assert res.blocked is True
        assert res.ok is False
        assert DOMAIN_CONTRADICTION_CODE in res.explanation_codes
        # 确定性：两次调用完全一致
        res2 = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert res == res2

    def test_same_check_is_deterministic_across_instances(self):
        h = HypothesisSpec(hypothesis_id="h1", expected_domains=("FUNDAMENTAL.QUALITY",))
        analysis = _FakeAnalysis(["close"])
        r1 = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        r2 = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert (r1.score, r1.hard_mismatch, r1.explanation_codes) == (
            r2.score,
            r2.hard_mismatch,
            r2.explanation_codes,
        )

    def test_mismatch_also_in_production(self):
        """production 下同款矛盾同样 hard_mismatch（不依赖 OFFLINE 宽松）。"""
        h = HypothesisSpec(hypothesis_id="h1", expected_domains=("FUNDAMENTAL.QUALITY",))
        analysis = _FakeAnalysis(["close"])
        res = _aligner(taxonomy=_pv_taxonomy(), mode=PRODUCTION).align(h, analysis)
        assert res.hard_mismatch is True
        assert res.blocked is True


# ---------------------------------------------------------------------------
# ② volume-price divergence 类 hypothesis 且实现消费 price+volume → 过
# ---------------------------------------------------------------------------


class TestVolumePriceDivergencePasses:
    def test_matching_pv_hypothesis_passes_deterministic_check(self):
        h = HypothesisSpec(
            hypothesis_id="h2",
            text="volume-price divergence: 放量滞涨后均值回复",
            expected_domains=("VOLUME", "PRICE"),
        )
        analysis = _FakeAnalysis(["close", "volume"])
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert res.hard_mismatch is False
        assert res.blocked is False
        assert res.missing_domains == ()
        assert res.unexpected_domains == ()
        assert res.timing_flags == ()
        assert res.score > 0.9
        assert res.ok is True

    def test_unexpected_extra_domain_reported_not_blocking(self):
        """实现多用了 amount（VOLUME+LIQUIDITY）而假设只声称 PRICE/VOLUME：
        liquidity 是「额外域」被报告但不一票否决（非 hard）。"""
        h = HypothesisSpec(
            hypothesis_id="h2b", expected_domains=("PRICE", "VOLUME")
        )
        analysis = _FakeAnalysis(["close", "amount"])
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert res.hard_mismatch is False
        assert res.missing_domains == ()
        assert "LIQUIDITY" in res.unexpected_domains

    def test_partial_missing_degrades_score_not_hard(self):
        """声称 PRICE+VOLUME 但实现只有 PRICE：部分缺失 → 中低分，非 hard。"""
        h = HypothesisSpec(hypothesis_id="h2c", expected_domains=("PRICE", "VOLUME"))
        analysis = _FakeAnalysis(["close"])
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert res.hard_mismatch is False
        assert res.missing_domains == ("VOLUME",)
        assert res.blocked is False
        assert res.score < 1.0


# ---------------------------------------------------------------------------
# ③ future/lookahead timing flag → 生成阶段即阻断（不进市场评估）
# ---------------------------------------------------------------------------


class TestTimingFlagsBlockBeforeMarketEval:
    def test_extra_timing_flag_blocks(self):
        """调用方显式注入 lookahead 位（FE 静态分析扩展未来可用）→ blocked。"""
        h = HypothesisSpec(hypothesis_id="h3", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"])
        res = _aligner(taxonomy=_pv_taxonomy(), mode=PRODUCTION).align(
            h, analysis, extra_timing_flags={"leaks_future": True}
        )
        assert res.blocked is True
        assert "leaks_future" in res.timing_flags
        assert res.ok is False

    def test_false_extra_flag_not_blocking(self):
        analysis = _FakeAnalysis(["close"])
        h = HypothesisSpec(hypothesis_id="h3", expected_domains=("PRICE",))
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(
            h, analysis, extra_timing_flags={"leaks_future": False}
        )
        assert res.timing_flags == ()
        assert res.ok is True

    def test_artifact_embedded_risk_key_blocks(self):
        """FE artifact 自带 risk 键（未来 FE 版本若扩展）也会被识别。"""
        h = HypothesisSpec(hypothesis_id="h3b", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"], timing_flags={"lookahead": True})
        res = _aligner(taxonomy=_pv_taxonomy(), mode=PRODUCTION).align(h, analysis)
        assert "lookahead" in res.timing_flags
        assert res.blocked is True

    def test_non_risk_fe_flags_do_not_block(self):
        """FE 现有四个 timing 位不是风险位（has_time_series_op 等）→ 不拦。"""
        h = HypothesisSpec(hypothesis_id="h3c", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(
            ["close"],
            timing_flags={
                "has_time_series_op": True,
                "has_cross_section_op": True,
                "has_group_cross_section_op": False,
                "requires_full_history": True,
            },
        )
        res = _aligner(taxonomy=_pv_taxonomy(), mode=PRODUCTION).align(h, analysis)
        assert res.timing_flags == ()
        assert res.blocked is False

    def test_unverified_pit_class_flags_timing(self):
        """字段 taxonomy 的 pit_class 无法确证 PIT-safe → 防御性 timing 标记。"""
        tax = _FakeTaxonomy(
            {
                "close": _FakeDescriptor(domains=("PRICE",), pit_class="unknown"),
            }
        )
        h = HypothesisSpec(hypothesis_id="h3d", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"])
        res = _aligner(taxonomy=tax, mode=OFFLINE).align(h, analysis)
        assert PIT_UNVERIFIED_CODE in res.timing_flags
        assert res.blocked is True


# ---------------------------------------------------------------------------
# ④ 已有 rank/zscore treatment lineage 被检测 → 阻止冗余 treatment 提案
# ---------------------------------------------------------------------------


class TestRedundantTreatmentDetection:
    def test_existing_rank_lineage_reported_redundant(self):
        """实现已含 rank（lineage CS_RANK:pct）→ 冗余 code + 分数降（非 hard）。"""
        h = HypothesisSpec(hypothesis_id="h4", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"], treatments=("CS_RANK:pct",))
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert REDUNDANT_TREATMENT_CODE in res.explanation_codes
        assert res.hard_mismatch is False
        assert res.blocked is False
        assert res.score < 1.0

    def test_redundant_can_be_hard_when_configured(self):
        h = HypothesisSpec(hypothesis_id="h4b", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"], treatments=("ZSCORE:cs",))
        res = _aligner(
            taxonomy=_pv_taxonomy(),
            mode=OFFLINE,
            reject_redundant_treatment_as_hard=True,
        ).align(h, analysis)
        assert REDUNDANT_TREATMENT_CODE in res.explanation_codes
        assert res.hard_mismatch is True
        assert res.blocked is True

    def test_no_lineage_no_redundancy(self):
        h = HypothesisSpec(hypothesis_id="h4c", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"], treatments=())
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        assert REDUNDANT_TREATMENT_CODE not in res.explanation_codes


# ---------------------------------------------------------------------------
# ⑤ production 下未知 field taxonomy → fail-closed
# ---------------------------------------------------------------------------


class TestUnknownFieldTaxonomyFailClosed:
    def test_production_unknown_field_raises(self):
        h = HypothesisSpec(hypothesis_id="h5", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close", "some_unknown_field"])
        with pytest.raises(UnknownFieldTaxonomyError):
            _aligner(taxonomy=_pv_taxonomy(), mode=PRODUCTION).align(h, analysis)

    def test_research_degraded_unknown_field_raises(self):
        h = HypothesisSpec(hypothesis_id="h5", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close", "unknown_field_2"])
        with pytest.raises(UnknownFieldTaxonomyError):
            _aligner(taxonomy=_pv_taxonomy(), mode=RESEARCH_DEGRADED).align(h, analysis)

    def test_offline_unknown_field_does_not_raise_and_marks_unavailable(self):
        h = HypothesisSpec(hypothesis_id="h5", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close", "unknown_field_3"])
        res = _aligner(taxonomy=_pv_taxonomy(), mode=OFFLINE).align(h, analysis)
        # 未知字段被跳过（不硬拦），且不产生错误结果
        assert "domains_unavailable_offline" in res.explanation_codes
        assert res.hard_mismatch is False

    def test_production_no_taxonomy_provider_raises(self):
        """未注入 taxonomy provider + production → fail-closed（不静默通过）。"""
        h = HypothesisSpec(hypothesis_id="h5b", expected_domains=("PRICE",))
        analysis = _FakeAnalysis(["close"])
        with pytest.raises(UnknownFieldTaxonomyError):
            _aligner(taxonomy=None, mode=PRODUCTION).align(h, analysis)


# ---------------------------------------------------------------------------
# 契约：AlignmentResult 字段 / blocked / ok
# ---------------------------------------------------------------------------


class TestAlignmentResultContract:
    def test_fields_present_and_defaults(self):
        r = AlignmentResult()
        assert r.score == 1.0
        assert r.hard_mismatch is False
        assert r.missing_domains == ()
        assert r.unexpected_domains == ()
        assert r.timing_flags == ()
        assert r.explanation_codes == ()
        assert r.ok is True
        assert r.blocked is False

    def test_hard_and_timing_set_blocked(self):
        assert AlignmentResult(score=0.0, hard_mismatch=True).blocked is True
        assert AlignmentResult(timing_flags=("leaks_future",)).blocked is True
        assert AlignmentResult(hard_mismatch=True).ok is False


# ---------------------------------------------------------------------------
# 真实 FE + 真实 DA taxonomy 端到端冒烟（可选；进程内共享 registry bootstrap）
# ---------------------------------------------------------------------------


def _real_stack_available() -> bool:
    try:
        import alphaprobe.fe_bridge.paths as _p  # noqa: F401
        from alphaprobe.fe_bridge import paths  # noqa: F401

        paths.ensure_factor_engine_importable()
        import factor_engine.api  # noqa: F401

        return True
    except Exception:
        return False


STACK_OK = _real_stack_available()
STACK_REQUIRED = pytest.mark.skipif(not STACK_OK, reason="FE/DA not importable")


class _RealDescriptor:
    """FieldSemanticDescriptor 鸭子（只暴露本模块消费的字段）。"""

    def __init__(self, d) -> None:
        self.data_domains = tuple(getattr(d, "data_domains", ()) or ())
        self.pit_class = getattr(d, "pit_class", None)


_REAL_ARTIFACTS: dict[str, Any] = {}
_REAL_STACK = {"taxonomy": None}


@pytest.fixture(scope="module")
def real_stack():
    """模块级 fixture：真实 FE artifact 分析器 + 真实 DA taxonomy provider。

    真实 FE 首调 ~50s（registry bootstrap）；模块内共享缓存，进程内跨模块也
    共享（FE bootstrap 进程级一次）。data_access 秒级。
    """
    from factor_engine.api.static_analysis import analyze_factor_definition
    from data_access.read.semantic_catalog import SemanticFieldTaxonomyProvider

    def analyze(formula: str):
        if formula not in _REAL_ARTIFACTS:
            _REAL_ARTIFACTS[formula] = analyze_factor_definition(formula)
        return _REAL_ARTIFACTS[formula]

    if _REAL_STACK["taxonomy"] is None:
        _REAL_STACK["taxonomy"] = SemanticFieldTaxonomyProvider()
    return analyze, _REAL_STACK["taxonomy"]


@STACK_REQUIRED
class TestRealStackEndToEnd:
    def test_real_pv_artifact_passes_pv_hypothesis(self, real_stack):
        analyze, taxonomy = real_stack
        art = analyze("ts_corr(volume, close, 20)")
        h = HypothesisSpec(
            hypothesis_id="r1",
            text="volume-price divergence",
            expected_domains=("PRICE", "VOLUME"),
        )
        aligner = DeterministicAligner(taxonomy=taxonomy, mode=PRODUCTION)
        res = aligner.align(h, art)
        assert res.hard_mismatch is False
        assert res.blocked is False
        assert set(res.missing_domains) <= set()

    def test_real_fundamental_artifact_accepts_fundamental_hypothesis(self, real_stack):
        analyze, taxonomy = real_stack
        art = analyze("rank(roe)")
        h = HypothesisSpec(
            hypothesis_id="r2",
            expected_domains=("FUNDAMENTAL.QUALITY",),
        )
        aligner = DeterministicAligner(taxonomy=taxonomy, mode=PRODUCTION)
        res = aligner.align(h, art)
        assert res.hard_mismatch is False
        assert res.missing_domains == ()
        assert res.blocked is False

    def test_real_fundamental_claim_rejects_pv_formula(self, real_stack):
        """真实现（PV 公式）+ 真 DA taxonomy + production → hard mismatch。"""
        analyze, taxonomy = real_stack
        art = analyze("rank(ts_mean(close, 20))")
        h = HypothesisSpec(
            hypothesis_id="r3",
            text="fundamental revision underreaction",
            expected_domains=("FUNDAMENTAL.QUALITY",),
        )
        aligner = DeterministicAligner(taxonomy=taxonomy, mode=PRODUCTION)
        res = aligner.align(h, art)
        assert res.hard_mismatch is True
        assert res.blocked is True

    def test_real_pv_artifact_with_real_lineage_reports_rank_redundant(self, real_stack):
        analyze, taxonomy = real_stack
        art = analyze("rank(close)")  # lineage CS_RANK:pct
        assert "CS_RANK:pct" in art.existing_treatment_semantic_ids
        h = HypothesisSpec(hypothesis_id="r4", expected_domains=("PRICE",))
        aligner = DeterministicAligner(taxonomy=taxonomy, mode=PRODUCTION)
        res = aligner.align(h, art)
        assert REDUNDANT_TREATMENT_CODE in res.explanation_codes
