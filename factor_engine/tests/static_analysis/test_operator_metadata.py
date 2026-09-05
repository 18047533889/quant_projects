# -*- coding: utf-8 -*-
"""R61-FI-012 — FE operator semantic metadata bridge 测试。

覆盖:
    a) 已知算子元数据正确且冻结（frozen dataclass / 只读查询，无 registry 变更）；
    b) 未知算子显式 UNKNOWN、不抛异常（禁猜）；
    c) 元数据可从 analyze_factor_definition 的 operator usage 链路取到；
    d) 全量算子遍历无 crash（registry 实际登记为准）。

运行:
    cd /home/sunhaiwei/quant_projects/factor_engine
    python -m pytest tests/static_analysis/test_operator_metadata.py -q --timeout=300

注意:
    FE registry 是 production-governed frozen 的 —— 本测试只做**只读**查询，
    绝不 thaw，绝不在 import 期触发 @register_operator 到 frozen registry。
"""

from __future__ import annotations

import dataclasses

import pytest

from factor_engine.api.static_analysis import (
    UNKNOWN_CAUSALITY,
    UNKNOWN_FITTED,
    UNKNOWN_STAGE,
    UNKNOWN_TAG,
    OperatorSemanticMetadata,
    analyze_factor_definition,
    operator_semantic_metadata,
)

#: 声明表覆盖的已知 canonical（与 _OPERATOR_SEMANTIC_TAGS / _STAGE_HINT 对齐）。
_KNOWN_ANNOTATED = (
    "rank", "zscore", "ts_mean", "group_neutralize", "winsorize",
    "ts_ema", "ts_delta", "ts_var", "cs_demean", "cs_pct_rank",
    "intra_amihud", "intra_realized_variance", "intra_bipower_variation",
    "ts_mean_reversion_half_life", "ts_corr",
)


# ---------------------------------------------------------------------------
# a) 已知算子元数据正确且冻结
# ---------------------------------------------------------------------------

class TestKnownOperatorMetadata:
    def test_rank_metadata_correct(self):
        m = operator_semantic_metadata("rank")
        assert m.operator_id == "rank"
        assert m.semantic_tags == ("rank",)
        assert m.stage_hint == "representation"
        assert m.causality_class == "cross_sectional"
        assert m.fitted_or_stateless == "stateless"
        assert m.coverage == "KNOWN"

    def test_ts_ema_metadata_correct(self):
        m = operator_semantic_metadata("ts_ema")
        assert m.operator_id == "ts_ema"
        assert "smoothing" in m.semantic_tags
        assert m.stage_hint == "smoothing"
        assert m.causality_class == "causal"
        assert m.fitted_or_stateless == "stateless"

    def test_ema_alias_resolves_to_canonical(self):
        m = operator_semantic_metadata("EMA")
        assert m.operator_id == "ts_ema"
        assert m.semantic_tags == ("smoothing",)
        assert m.coverage == "KNOWN"

    def test_group_neutralize_metadata_correct(self):
        m = operator_semantic_metadata("group_neutralize")
        assert m.operator_id == "group_neutralize"
        assert "neutralize" in m.semantic_tags
        assert m.stage_hint == "neutralize"
        assert m.causality_class == "cross_sectional"

    def test_volatility_tag_present(self):
        m = operator_semantic_metadata("ts_var")
        assert "volatility" in m.semantic_tags
        assert m.causality_class == "causal"

    def test_metadata_frozen(self):
        """OperatorSemanticMetadata 为冻结 dataclass，只读。"""
        m = operator_semantic_metadata("rank")
        assert dataclasses.is_dataclass(m)
        with pytest.raises(Exception):
            m.semantic_tags = ("mutated",)  # frozen

    def test_known_annotated_all_known(self):
        """声明表覆盖的 canonical 必须全部 KNOWN（不允许漏标成 UNKNOWN）。"""
        for name in _KNOWN_ANNOTATED:
            m = operator_semantic_metadata(name)
            assert m.coverage == "KNOWN", f"{name} should be KNOWN, got {m}"


# ---------------------------------------------------------------------------
# b) 未知算子显式 UNKNOWN、不抛异常
# ---------------------------------------------------------------------------

class TestUnknownOperatorMetadata:
    def test_unregistered_operator_explicit_unknown(self):
        m = operator_semantic_metadata("no_such_operator_xyz")
        assert m.coverage == "UNKNOWN"
        assert m.semantic_tags == (UNKNOWN_TAG,)
        assert m.stage_hint == UNKNOWN_STAGE
        assert m.causality_class == UNKNOWN_CAUSALITY
        assert m.fitted_or_stateless == UNKNOWN_FITTED

    def test_unannotated_registered_operator_is_unknown(self):
        """registered 但未在声明表覆盖的算子：tags/stage 显式 UNKNOWN（不猜）。"""
        m = operator_semantic_metadata("floor")
        assert m.coverage == "UNKNOWN" or m.semantic_tags == (UNKNOWN_TAG,)
        # 即使 causality 来自 AxisEffectContract（真实声明），tags/stage 仍 UNKNOWN
        assert UNKNOWN_TAG in m.semantic_tags
        assert m.stage_hint == UNKNOWN_STAGE

    def test_unknown_does_not_raise(self):
        for name in ("totally_unknown", "", "rank(", "123"):
            operator_semantic_metadata(name)  # 不应抛错

    def test_unknown_to_dict_survives(self):
        m = operator_semantic_metadata("no_such_operator_xyz")
        d = m.to_dict()
        assert d["coverage"] == "UNKNOWN"
        assert d["semantic_tags"] == [UNKNOWN_TAG]


# ---------------------------------------------------------------------------
# c) 元数据可从 operator usage 链路取到
# ---------------------------------------------------------------------------

class TestOperatorMetadataFromUsageChain:
    def test_metadata_reachable_from_usage(self):
        """analyze_factor_definition 的 operator_usages 可反查语义元数据。"""
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        by_id = {o.operator_id: o for o in a.operator_usages}
        assert "rank" in by_id and "ts_mean" in by_id
        m_rank = operator_semantic_metadata(by_id["rank"].operator_id)
        assert m_rank.coverage == "KNOWN"
        assert m_rank.semantic_tags == ("rank",)
        m_ts = operator_semantic_metadata(by_id["ts_mean"].operator_id)
        assert m_ts.causality_class == "causal"

    def test_usage_axis_effect_aligns_with_metadata_causality(self):
        """operator usage 的 axis_effect 与 metadata.causality_class 对齐。"""
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        by_id = {o.operator_id: o for o in a.operator_usages}
        m_ts = operator_semantic_metadata("ts_mean")
        assert by_id["ts_mean"].axis_effect == "time_series"
        assert m_ts.causality_class == "causal"
        m_rank = operator_semantic_metadata("rank")
        assert by_id["rank"].axis_effect == "cross_section"
        assert m_rank.causality_class == "cross_sectional"

    def test_metadata_to_dict_for_downstream(self):
        """FO/FA 可消费 to_dict（可 JSON 序列化）。"""
        import json

        m = operator_semantic_metadata("ts_ema")
        d = m.to_dict()
        json.dumps(d)  # 必须可 JSON 序列化
        assert set(d) == {
            "operator_id", "semantic_tags", "stage_hint", "causality_class",
            "fitted_or_stateless", "surface_known", "coverage",
        }


# ---------------------------------------------------------------------------
# d) 全量算子遍历无 crash（registry 实际登记为准）
# ---------------------------------------------------------------------------

class TestFullTraversal:
    def test_all_registered_canonicals_no_crash(self):
        """遍历 registry 全部 canonical，查询不抛错、返回冻结对象。"""
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

        ensure_cleaned_loaded()
        canonicals = OperatorRegistry.list_canonical()
        assert len(canonicals) > 1000
        for canonical in canonicals:
            m = operator_semantic_metadata(canonical)
            assert isinstance(m, OperatorSemanticMetadata)
            assert m.operator_id  # 非空
            assert m.fitted_or_stateless in ("stateless", UNKNOWN_FITTED)
            assert m.coverage in ("KNOWN", "UNKNOWN")

    def test_all_daily_surface_no_crash_and_reasonable_causality(self):
        """daily 生产面算子：causality 必须能被 AxisEffectContract 推导（非 UNKNOWN）。"""
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.cleaned_operators.operator_surface import classify_canonical
        from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

        ensure_cleaned_loaded()
        daily = [
            c for c in OperatorRegistry.list_canonical()
            if classify_canonical(c) == "daily"
        ]
        unknown_causality = []
        for canonical in daily:
            m = operator_semantic_metadata(canonical)
            if m.causality_class == UNKNOWN_CAUSALITY:
                unknown_causality.append(canonical)
        # 声明表外未标注算子允许 UNKNOWN tags，但 daily 面算子至少应有
        # 明确的 causality（AxisEffectContract 全覆盖声明）。
        assert len(unknown_causality) == 0, (
            f"{len(unknown_causality)} daily canonicals lack causality: "
            f"{unknown_causality[:20]}"
        )
