# -*- coding: utf-8 -*-
"""R61-FI-011/012 — factor_engine 公共静态分析清单测试（plan §62.2 必测用例）。

运行:
    cd /home/sunhaiwei/quant_projects/factor_engine
    python -m pytest tests/static_analysis/test_static_analysis.py -q --timeout=300

范围: 仅 parse/analyze 层 —— 不物化数据、不执行算子。首次进程冷启动会触发
cleaned_operators 注册表加载（约 10-50s，pytest sessionstart 已把 late-surface
模块 staged，测试期不再 thaw 生产 registry —— P0-14）。
"""

from __future__ import annotations

import json

import pytest

from factor_engine.api.static_analysis import (
    FactorStaticAnalysisArtifact,
    FieldUsage,
    OperatorUsage,
    analyze_factor_definition,
    causality_class_of,
    existing_treatment_semantic_id_of,
    resolve_canonical_operator,
    semantic_stage_hint_of,
)


# ---------------------------------------------------------------------------
# §62.2 field / operator usage（嵌套表达式）
# ---------------------------------------------------------------------------

class TestFieldAndOperatorUsage:
    def test_field_usage_nested_expression(self):
        """嵌套表达式：rank(ts_mean(close, 5)) 只引用 StockDailyBarAdj.close。"""
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        fids = [f.canonical_field_id for f in a.field_usages]
        assert fids == ["StockDailyBarAdj.close"]
        usage = a.field_usages[0]
        assert isinstance(usage, FieldUsage)
        assert usage.dataset_id == "ashare_stock_daily_adj"
        assert usage.domain == "price_volume"
        assert usage.occurrence_count == 1

    def test_operator_usage_nested_expression(self):
        """嵌套表达式算子统计：rank + ts_mean canonical，各出现一次。"""
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        ops = {o.operator_id: o for o in a.operator_usages}
        assert set(ops) == {"rank", "ts_mean"}
        assert ops["rank"].occurrence_count == 1
        assert ops["ts_mean"].occurrence_count == 1
        # 别名语义已在 canonical 层统一
        assert a.operator_usages[0].operator_version == "1.0"

    def test_field_usage_occurrence_counts_repeated_field(self):
        """同一字段出现多次 occurrence_count 累加。"""
        a = analyze_factor_definition("ts_mean(close, 5) - ts_mean(close, 20)")
        close = [f for f in a.field_usages if f.canonical_field_id.endswith(".close")][0]
        assert close.occurrence_count == 2

    def test_operator_alias_canonical_dedupe(self):
        """cs_rank / RANK / rank 全部归一到 canonical 'rank'。"""
        a = analyze_factor_definition("cs_rank(close) + RANK(high) + rank(low)")
        ops = {o.operator_id: o.occurrence_count for o in a.operator_usages}
        assert ops["rank"] == 3

    def test_mixed_domain_field_usage(self):
        """混合 domain 字段：price_volume(close) + fundamental(roe) 各自独立。"""
        a = analyze_factor_definition("rank(ts_mean(close, 5)) + roe")
        by_id = {f.canonical_field_id: f for f in a.field_usages}
        assert set(by_id) == {"StockDailyBarAdj.close", "StockIndicator.roe"}
        assert by_id["StockDailyBarAdj.close"].domain == "price_volume"
        assert by_id["StockIndicator.roe"].domain == "fundamental"
        assert by_id["StockIndicator.roe"].table == "StockIndicator"


# ---------------------------------------------------------------------------
# §62.2 existing-treatment lineage（RANKED / ZSCORED / WINSORIZED /
# SMOOTHED_EWMA / NEUTRALIZED_*）
# ---------------------------------------------------------------------------

class TestExistingTreatmentLineage:
    def test_existing_rank_lineage(self):
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        assert "CS_RANK:pct" in a.existing_treatment_semantic_ids

    def test_existing_zscore_lineage(self):
        a = analyze_factor_definition("zscore(winsorize(close, 0.01, 0.99))")
        assert "ZSCORE:cs" in a.existing_treatment_semantic_ids
        assert "WINSOR:cs" in a.existing_treatment_semantic_ids
        # 两个处理都出现；排序按 AST 后序（不做语义先后强断言）

    def test_existing_neutralization_lineage(self):
        a = analyze_factor_definition(
            "industry_neutralize(rank(ts_mean(close, 20)), industry_code)"
        )
        ids = a.existing_treatment_semantic_ids
        assert "INDUSTRY_NEUTRAL:sw_l1" in ids
        assert "CS_RANK:pct" in ids

    def test_existing_size_neutralization_and_dual(self):
        a1 = analyze_factor_definition("size_neutralize(rank(close))")
        assert "SIZE_NEUTRAL:log_mktcap" in a1.existing_treatment_semantic_ids
        a2 = analyze_factor_definition("industry_size_neutralize(close, industry_code)")
        assert "DUAL_NEUTRAL:industry_size" in a2.existing_treatment_semantic_ids

    def test_existing_smoothing_ewma_lineage(self):
        a = analyze_factor_definition("EMA(close, 10)")
        assert "SMOOTH:ewma" in a.existing_treatment_semantic_ids

    def test_untreated_formula_has_no_lineage(self):
        # ts_mean/ts_std 是特征构造（rolling mean/std），非预处理平滑处理
        a = analyze_factor_definition("ts_mean(close, 5) / ts_std(close, 5)")
        assert a.existing_treatment_semantic_ids == ()

    def test_semantic_bridge_direct_mapping(self):
        # FE 侧语义桥直接映射（canonical / alias）
        assert existing_treatment_semantic_id_of("cs_rank") == "CS_RANK:pct"
        assert existing_treatment_semantic_id_of("RANK") == "CS_RANK:pct"
        assert existing_treatment_semantic_id_of("cs_zscore") == "ZSCORE:cs"
        assert existing_treatment_semantic_id_of("cap") == "WINSOR:cs"  # cap→clip
        assert existing_treatment_semantic_id_of("ewma") == "SMOOTH:ewma"
        assert existing_treatment_semantic_id_of("no_such_operator") is None
        # 通用可选元数据
        assert semantic_stage_hint_of("rank") == "representation"
        assert semantic_stage_hint_of("winsorize") == "outlier"
        assert causality_class_of("ts_mean") == "causal"
        assert causality_class_of("rank") == "cross_sectional"
        assert causality_class_of("abs") == "elementwise"


# ---------------------------------------------------------------------------
# §62.2 lookback / canonical hash / fail-closed
# ---------------------------------------------------------------------------

class TestLookbackAndCanonicalHash:
    def test_max_lookback(self):
        a = analyze_factor_definition("ts_delay(close, 5) + ts_mean(high, 10)")
        # history_requirement 行数（window-1 + lag）≤ heuristic 参数值
        assert a.max_lookback == 9  # ts_mean(10)→9, ts_delay 5 不增（单权威 9）
        b = analyze_factor_definition("rank(ts_mean(close, 20))")
        assert b.max_lookback == 19

    def test_canonical_hash_format_invariant(self):
        """canonical_dsl_hash 是 64 位 SHA-256；且同义公式（别名/空格）同 hash。"""
        a1 = analyze_factor_definition("rank(ts_mean(close, 5))")
        a2 = analyze_factor_definition("rank( ts_mean( close , 5 ) )")
        assert len(a1.canonical_dsl_hash) == 64
        assert a1.canonical_dsl_hash == a2.canonical_dsl_hash
        # canonical_dsl_hash == identity canonical_ast_hash
        from factor_engine.identity import get_factor_identity

        ident = get_factor_identity("rank(ts_mean(close, 5))")
        assert a1.canonical_dsl_hash == ident.canonical_ast_hash

    def test_factor_definition_id_prefix(self):
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        assert a.factor_definition_id.startswith("F_")
        # F_ + 11 base32（derive_factor_id 口径）
        assert len(a.factor_definition_id) == 13
        assert a.factor_definition_id == "F_" + a.factor_definition_id[2:]

    def test_content_hash_deterministic_and_payload_bound(self):
        a1 = analyze_factor_definition("rank(ts_mean(close, 5))")
        a2 = analyze_factor_definition("rank(ts_mean(close, 5))")
        assert len(a1.content_hash) == 64
        assert a1.content_hash == a2.content_hash
        # canonical_dsl 是载荷的一部分：include_canonical_dsl=False 时 content_hash
        # 必须随之变化（载荷不同 → 哈希不同）
        a3 = analyze_factor_definition(
            "rank(ts_mean(close, 5))", include_canonical_dsl=False
        )
        assert a3.canonical_dsl == ""
        assert a3.content_hash != a1.content_hash

    def test_unknown_operator_fail_closed(self):
        """未注册算子：DSL 解析直接失败（fail-closed）。"""
        with pytest.raises(Exception) as excinfo:
            analyze_factor_definition("no_such_operator(close, 20)")
        assert "no_such_operator" in str(excinfo.value)

    def test_expr_input_and_payload_input_equivalent(self):
        """Expr 入参与 DSL 文本入参、canonical_dsl JSON 入参等价。"""
        from factor_engine.api.dsl_parser import parse_expr
        from factor_engine.identity.api import _payload_to_expr

        a_str = analyze_factor_definition("rank(ts_mean(close, 5))")
        a_expr = analyze_factor_definition(parse_expr("rank(ts_mean(close, 5))"))
        assert a_expr.content_hash == a_str.content_hash
        a_payload = analyze_factor_definition(_payload_to_expr(a_str.canonical_dsl))
        assert a_payload.content_hash == a_str.content_hash
        assert a_payload.factor_definition_id == a_str.factor_definition_id

    def test_artifact_frozen_and_to_dict(self):
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        with pytest.raises(Exception):
            a.content_hash = "x"  # frozen dataclass
        d = a.to_dict()
        assert set(d) == {
            "factor_definition_id", "canonical_dsl", "canonical_dsl_hash",
            "field_usages", "operator_usages", "existing_treatment_semantic_ids",
            "max_lookback", "complexity", "timing_flags", "content_hash", "surface",
        }
        # content_payload 排除 content_hash（防自指）
        payload = a.content_payload()
        assert "content_hash" not in payload


# ---------------------------------------------------------------------------
# §62.2 complexity / timing_flags / DSL-only mutation 无回归
# ---------------------------------------------------------------------------

class TestComplexityAndTiming:
    def test_complexity_counts(self):
        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        assert a.complexity["nodes"] == 4          # rank + ts_mean + close + 5
        assert a.complexity["depth"] == 3          # rank -> ts_mean -> leaf
        assert a.complexity["operators"] == 2
        assert a.complexity["fields"] == 1
        assert a.complexity["literals"] == 1
        assert a.complexity["max_arity"] == 2

    def test_timing_flags(self):
        a_ts = analyze_factor_definition("EMA(close, 10)")
        assert a_ts.timing_flags["has_time_series_op"] is True
        assert a_ts.timing_flags["has_cross_section_op"] is False
        a_cs = analyze_factor_definition("winsorize(close, 0.01, 0.99)")
        assert a_cs.timing_flags["has_cross_section_op"] is True
        a_both = analyze_factor_definition("rank(ts_mean(close, 5))")
        assert a_both.timing_flags["has_time_series_op"] is True
        assert a_both.timing_flags["has_cross_section_op"] is True


class TestDslOnlyMutationNoRegression:
    """静态分析不应改变 FE 的 parse/compile 行为（纯只读）。"""

    def test_parse_still_works_after_analysis(self):
        from factor_engine.api.dsl_parser import parse_expr

        a = analyze_factor_definition("rank(ts_mean(close, 5))")
        assert a.canonical_dsl_hash
        # DSL-only mutation：解析 + canonicalize + 身份仍可编译（无 runtime 需要）
        e = parse_expr("rank(ts_mean(close, 5))")
        from factor_engine.identity.canonicalizer import canonicalize

        canon = canonicalize(e)
        assert canon is not None
        from factor_engine.identity import get_factor_identity

        ident = get_factor_identity(canon)
        assert ident.canonical_ast_hash == a.canonical_dsl_hash

    def test_analyze_does_not_materialize_or_require_backend(self):
        """清单只走 parse/analyze —— 不触 runtime 算子 calculate。"""
        a = analyze_factor_definition("rank(ts_mean(close, 5)) + roe")
        assert a.complexity["fields"] == 2
        assert a.max_lookback >= 0


# ---------------------------------------------------------------------------
# 公开契约冻结（防止无意改字段名/结构破坏 FO/FA 消费方）
# ---------------------------------------------------------------------------

class TestFrozenPublicContract:
    def test_contract_field_lists(self):
        import dataclasses

        fu_fields = [f.name for f in dataclasses.fields(FieldUsage)]
        assert fu_fields == [
            "canonical_field_id", "dataset_id", "table", "domain", "role",
            "occurrence_count", "direct_or_derived", "timing_class",
        ]
        ou_fields = [f.name for f in dataclasses.fields(OperatorUsage)]
        assert ou_fields == [
            "operator_id", "operator_version", "occurrence_count", "stage_hint",
            "causality_class", "axis_effect", "category",
        ]
        sa_fields = [f.name for f in dataclasses.fields(FactorStaticAnalysisArtifact)]
        assert sa_fields == [
            "factor_definition_id", "canonical_dsl", "canonical_dsl_hash",
            "field_usages", "operator_usages", "existing_treatment_semantic_ids",
            "max_lookback", "complexity", "timing_flags", "content_hash", "surface",
        ]

    def test_public_module_symbols(self):
        import factor_engine.api.static_analysis as m

        for sym in (
            "analyze_factor_definition",
            "FieldUsage",
            "OperatorUsage",
            "FactorStaticAnalysisArtifact",
            "resolve_canonical_operator",
            "existing_treatment_semantic_id_of",
            "semantic_stage_hint_of",
            "causality_class_of",
        ):
            assert hasattr(m, sym)
