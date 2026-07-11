"""evaluation.label 的规则贴标与语义标签标准化逻辑。

模块: evaluation.label
职责:
1. 根据评估输出生成确定性的本地规则标签。
2. 为 DeepSeek 构造语义贴标上下文。
3. 基于标准标签注册表对语义标签进行标准化。

本文件只保留计算逻辑: 不执行文件 I/O，也不遍历路径。
"""

from __future__ import annotations

import json
from typing import Any

from .deepseek_client import DeepSeekClient
from .logging_utils import get_label_logger
from .schemas import (
    DeepSeekTags,
    EvaluationSummary,
    FactorMeta,
    JsonDict,
    LabelGovernance,
    RuleTags,
    TagPackage,
    _validate_label_key,
)


LOGGER = get_label_logger()


def generate_tags(
    factor_meta: FactorMeta,
    evaluation_summary: EvaluationSummary,
    tag_policy: JsonDict,
    label_registry: JsonDict,
    deepseek_client: DeepSeekClient,
) -> TagPackage:
    """生成单个因子的完整标签包。

    已实现的评审要求:
    1. v1_review.md 4.3-(3): 语义标签由一个标准标签注册表统一管理，
       新标签以治理记录方式落盘。
    2. v1_review.md 4.3-(7): 仅将已校验的白名单结构传入 LLM 边界。
    3. LLM 选择已有标签时直接使用；提出新标签时直接加入注册表并记录决策。
    """

    LOGGER.info("generate_tags_start factor_id=%s", factor_meta["factor_id"])
    rule_tags = collect_rule_tags(factor_meta, evaluation_summary, tag_policy)
    semantic_context = build_semantic_context(factor_meta, evaluation_summary)
    LOGGER.info(
        "semantic_context_built factor_id=%s context_keys=%s",
        factor_meta["factor_id"],
        sorted(semantic_context.keys()),
    )
    deepseek_raw_result = deepseek_client.request_semantic_tags(
        expression=factor_meta["expression"],
        charts_summary=semantic_context,
        label_registry_excerpt=label_registry,
    )
    deepseek_tags, label_governance = normalize_semantic_label(
        deepseek_result=deepseek_raw_result,
        label_registry=label_registry,
    )
    LOGGER.info(
        "generate_tags_done factor_id=%s rule_tag_counts=%s primary_label=%s "
        "registry_action=%s",
        factor_meta["factor_id"],
        {key: len(value) for key, value in rule_tags.items()},
        deepseek_tags["primary_label"],
        label_governance["registry_action"],
    )
    return {
        "factor_id": factor_meta["factor_id"],
        "rule_tags": rule_tags,
        "deepseek_tags": deepseek_tags,
        "label_governance": label_governance,
    }


def collect_rule_tags(
    factor_meta: FactorMeta,
    evaluation_summary: EvaluationSummary,
    tag_policy: JsonDict,
) -> RuleTags:
    """根据配置的标签策略生成规则标签。

    已实现的评审要求:
    1. v1_review.md 4.3-(10): 规则值异常时直接暴露问题，而不是静默丢弃标签。
    2. v1_review.md 4.3-(3): 规则标签保持确定性和标准化。
    """

    scorecard = evaluation_summary["summary_scorecard"]
    performance_tags: list[str] = []
    style_tags: list[str] = []
    lifecycle_tags: list[str] = []
    action_tags: list[str] = []

    for rule in tag_policy["performance_rules"]:
        metric_name = rule["metric"]
        if metric_name in scorecard and compare_threshold(
            scorecard[metric_name], rule["op"], rule["threshold"]
        ):
            performance_tags.append(rule["tag"])
            LOGGER.info(
                "rule_tag_matched section=performance metric=%s tag=%s",
                metric_name,
                rule["tag"],
            )

    for rule in tag_policy["trading_rules"]:
        metric_name = rule["metric"]
        if metric_name in scorecard and compare_threshold(
            scorecard[metric_name], rule["op"], rule["threshold"]
        ):
            performance_tags.append(rule["tag"])
            action_tags.append(rule["action_tag"])
            LOGGER.info(
                "rule_tag_matched section=trading metric=%s tag=%s action_tag=%s",
                metric_name,
                rule["tag"],
                rule["action_tag"],
            )

    for rule in tag_policy["lifecycle_rules"]:
        field_name = rule["field"]
        if evaluation_summary.get(field_name) == rule["equals"]:
            lifecycle_tags.append(rule["tag"])
            LOGGER.info(
                "rule_tag_matched section=lifecycle field=%s value=%s tag=%s",
                field_name,
                rule["equals"],
                rule["tag"],
            )

    searchable_text = _build_style_search_text(factor_meta)
    for rule in tag_policy["style_rules"]:
        if any(token in searchable_text for token in rule["contains_any"]):
            style_tags.append(rule["tag"])
            LOGGER.info(
                "rule_tag_matched section=style tag=%s tokens=%s",
                rule["tag"],
                rule["contains_any"],
            )

    rule_tags = {
        "source_tags": _build_source_tags(factor_meta),
        "performance_tags": _unique_tags(performance_tags),
        "style_tags": _unique_tags(style_tags),
        "lifecycle_tags": _unique_tags(lifecycle_tags),
        "action_tags": _unique_tags(action_tags),
    }
    LOGGER.info(
        "collect_rule_tags_done factor_id=%s counts=%s",
        factor_meta["factor_id"],
        {key: len(value) for key, value in rule_tags.items()},
    )
    return rule_tags


def build_semantic_context(
    factor_meta: FactorMeta,
    evaluation_summary: EvaluationSummary,
) -> JsonDict:
    """构造传入 DeepSeek 的已清洗语义上下文。

    已实现的评审要求:
    1. v1_review.md 4.3-(17): 仅将白名单内且通过结构校验的字段传入 LLM 边界。
    """

    return {
        "factor_id": factor_meta["factor_id"],
        "candidate_id": factor_meta["candidate_id"],
        "signal_structure": factor_meta["signal_structure"],
        "asset_class": factor_meta["asset_class"],
        "frequency_bucket": factor_meta["frequency_bucket"],
        "domain_root": factor_meta["domain_root"],
        "ast": factor_meta["ast"],
        "summary_scorecard": evaluation_summary["summary_scorecard"],
        "survival_view": evaluation_summary["survival_view"],
        "usage_role": evaluation_summary["usage_role"],
        "route_recommendation": evaluation_summary["route_recommendation"],
        "fragility_tags": list(evaluation_summary["fragility_tags"]),
        "incremental_value_summary": dict(
            evaluation_summary["incremental_value_summary"]
        ),
        "charts_summary": dict(evaluation_summary["charts_summary"]),
    }


def normalize_semantic_label(
    deepseek_result: JsonDict,
    label_registry: JsonDict,
) -> tuple[DeepSeekTags, LabelGovernance]:
    """基于标准标签注册表标准化一个 LLM 语义结果。

    当前逻辑:
    1. 如果 LLM 选择已有标签，直接使用该标签。
    2. 如果 LLM 提出新标签，直接将其加入注册表。
    3. similar_candidates 仅在新标签场景中记录。
    """

    primary_label = deepseek_result["primary_label"]
    explanation = deepseek_result["explanation"]
    suggested_new_label = deepseek_result["suggested_new_label"]
    similar_candidates = deepseek_result["similar_candidates"]
    confidence = deepseek_result["confidence"]

    registry_labels = label_registry["labels"]
    existing_label_keys = {item["label_key"] for item in registry_labels}

    if suggested_new_label == "":
        if primary_label not in existing_label_keys:
            raise ValueError(
                "deepseek_result.primary_label must exist in label_registry when "
                f"suggested_new_label is empty, got: {primary_label}"
            )
        if similar_candidates:
            raise ValueError(
                "deepseek_result.similar_candidates must be empty when the LLM "
                "selected an existing label"
            )
        deepseek_tags: DeepSeekTags = {
            "primary_label": primary_label,
            "normalized_primary_label": primary_label,
            "confidence": confidence,
            "explanation": explanation,
            "suggested_new_label": "",
            "similar_candidates": [],
        }
        governance: LabelGovernance = {
            "registry_hit": True,
            "matched_label_key": primary_label,
            "registry_action": "use_existing",
            "similar_candidates": [],
            "review_notes": [f"used existing registry label: {primary_label}"],
        }
        LOGGER.info(
            "semantic_label_use_existing primary_label=%s confidence=%.4f",
            primary_label,
            confidence,
        )
        return deepseek_tags, governance

    if primary_label != suggested_new_label:
        raise ValueError(
            "deepseek_result.primary_label must equal "
            "deepseek_result.suggested_new_label when creating a new label"
        )
    # 新标签在写入注册表前必须通过标准 snake_case 校验（FID §2.2/§2.3）；
    # 否则非法 label_key 会污染全局 label_registry.json，导致下次所有因子的
    # validate_label_registry 整体失败。
    _validate_label_key(primary_label, "deepseek_result.suggested_new_label")
    if primary_label in existing_label_keys:
        raise ValueError(
            "deepseek_result.suggested_new_label must not already exist in the "
            f"registry, got: {primary_label}"
        )

    for candidate in similar_candidates:
        if candidate not in existing_label_keys:
            raise ValueError(
                "deepseek_result.similar_candidates contains a label that is not "
                f"in the current registry: {candidate}"
            )

    new_label_record = {
        "label_key": primary_label,
        "display_name": _display_name_from_label_key(primary_label),
        "category": "semantic",
        "description": explanation or "Added by evaluation.label semantic tagging.",
        "aliases": [],
        "status": "active",
    }
    registry_labels.append(new_label_record)
    LOGGER.info(
        "semantic_label_add_new primary_label=%s similar_candidates=%s "
        "registry_size=%s",
        primary_label,
        similar_candidates,
        len(registry_labels),
    )

    deepseek_tags = {
        "primary_label": primary_label,
        "normalized_primary_label": primary_label,
        "confidence": confidence,
        "explanation": explanation,
        "suggested_new_label": primary_label,
        "similar_candidates": list(similar_candidates),
    }
    governance = {
        "registry_hit": False,
        "matched_label_key": primary_label,
        "registry_action": "add_new",
        "similar_candidates": list(similar_candidates),
        "review_notes": [
            f"added new registry label: {primary_label}",
            f"similar existing labels: {', '.join(similar_candidates) if similar_candidates else 'none'}",
        ],
    }
    return deepseek_tags, governance


def compare_threshold(left: Any, operator: str, right: Any) -> bool:
    """比较一个指标值是否满足一条规则阈值。

    已实现的评审要求:
    1. v1_review.md 4.3-(10): 类型错误直接暴露，不转换为静默的 False 结果。
    """

    if not _is_number(left):
        raise TypeError(f"rule comparison left value must be numeric, got: {left!r}")
    if not _is_number(right):
        raise TypeError(f"rule comparison right value must be numeric, got: {right!r}")
    left_value = float(left)
    right_value = float(right)
    if operator == "==":
        return left_value == right_value
    if operator == ">":
        return left_value > right_value
    if operator == ">=":
        return left_value >= right_value
    if operator == "<":
        return left_value < right_value
    if operator == "<=":
        return left_value <= right_value
    raise ValueError(f"unsupported rule comparison operator: {operator}")


def _build_source_tags(factor_meta: FactorMeta) -> list[str]:
    tags: list[str] = []
    generator_name = factor_meta.get("generator_name", "")
    if generator_name:
        tags.append(generator_name)
    domain_root = factor_meta.get("domain_root", "")
    if domain_root:
        tags.append(domain_root)
    asset_class = factor_meta.get("asset_class", "")
    frequency_bucket = factor_meta.get("frequency_bucket", "")
    if asset_class and frequency_bucket:
        tags.append(f"{asset_class}_{frequency_bucket}")
    return _unique_tags(tags)


def _build_style_search_text(factor_meta: FactorMeta) -> str:
    ast_text = json.dumps(factor_meta["ast"], ensure_ascii=False, sort_keys=True)
    return " ".join(
        part
        for part in (
            factor_meta["expression"],
            ast_text,
            factor_meta["domain_root"],
            factor_meta["generator_name"],
        )
        if part
    )


def _display_name_from_label_key(label_key: str) -> str:
    return " ".join(part.capitalize() for part in label_key.split("_"))


def _unique_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_tags: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    return unique_tags


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
