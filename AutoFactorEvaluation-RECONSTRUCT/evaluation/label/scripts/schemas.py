"""Evaluation label 服务的数据契约与默认配置。

模块: evaluation.label
职责:
1. 定义贴标和路由使用的公共 JSON 契约。
2. 定义仅用于正式流程的路径和环境配置入口。
3. 为所有外部输入保留严格校验入口。

本文件只保留契约和配置逻辑。业务计算放在 tagging.py / routing.py 中。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, TypedDict

from .logging_utils import get_label_logger


JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JsonDict = dict[str, Any]
LOGGER = get_label_logger()

SURVIVAL_VIEWS = {
    "gross_only",
    "net_survive",
    "taker_survive",
    "maker_only",
    "failed",
}

USAGE_ROLES = {
    "core_signal",
    "satellite_signal",
    "gate",
    "context_feature",
    "model_feature",
    "archive_only",
}

ROUTE_RECOMMENDATIONS = {
    "tier3a_core",
    "tier3b_satellite",
    "tier3c_feature",
    "tier3d_optimized_reserve",
    "tier2_incubator",
    "tier2x_optimization_factory",
    "tier4_archive",
}

ROUTE_TIERS = {
    "Tier3A",
    "Tier3B",
    "Tier3C",
    "Tier3D",
    "Tier2",
    "Tier2X",
    "Tier4",
}


class EvaluationSummary(TypedDict, total=False):
    """label 模块消费的上游评估摘要。

    首个正式版本必需字段:
    - summary_scorecard
    - survival_view
    - usage_role
    - route_recommendation
    """

    evaluation_protocol_key: str
    summary_scorecard: JsonDict
    survival_view: str
    usage_role: str
    fragility_tags: list[Any]
    incremental_value_summary: JsonDict
    route_recommendation: str | JsonDict
    admission_confidence: float
    primary_failure_reasons: list[Any]
    detail_page_template_key: str
    charts_summary: JsonDict


class FactorMeta(TypedDict, total=False):
    """从上游纯化输出加载的因子元数据。"""

    factor_id: str
    candidate_id: str
    expression: str
    formula_ast: JsonDict
    ast: JsonDict
    generator_name: str
    campaign_id: str
    iteration_id: str
    batch_id: str
    signal_structure: str
    asset_class: str
    frequency_bucket: str
    domain_root: str
    domain: str
    lib_coordinates: JsonDict
    source_library: str
    tier: str
    status: str


class RuleTags(TypedDict, total=False):
    """本地策略引擎输出的规则标签。"""

    source_tags: list[str]
    performance_tags: list[str]
    style_tags: list[str]
    lifecycle_tags: list[str]
    action_tags: list[str]


class DeepSeekTags(TypedDict, total=False):
    """基于标签注册表标准化后的语义贴标结果。"""

    primary_label: str
    normalized_primary_label: str
    confidence: float
    explanation: str
    suggested_new_label: str
    similar_candidates: list[str]


class LabelGovernance(TypedDict, total=False):
    """标准化标签和增量标签管理的治理结果。"""

    registry_hit: bool
    matched_label_key: str
    registry_action: str
    similar_candidates: list[str]
    review_notes: list[str]


class TagPackage(TypedDict, total=False):
    """返回给路由逻辑并持久化到磁盘的贴标输出。"""

    factor_id: str
    rule_tags: RuleTags
    deepseek_tags: DeepSeekTags
    label_governance: LabelGovernance


class AdmissionDecision(TypedDict, total=False):
    """label 模块输出的路由决策。"""

    factor_id: str
    tier: str
    reason: str
    incubation_goal: str
    actions: list[str]
    decided_by: str


class RouteRecord(TypedDict, total=False):
    """单次路由事件的生命周期记录。"""

    factor_id: str
    from_state: str
    to_state: str
    from_tier: str
    to_tier: str
    trigger_type: str
    trigger_ref: str
    operator: str
    created_at: str
    target_factor_dir: str


class LifecycleEvent(TypedDict, total=False):
    """与路由记录一同输出的审计事件。"""

    factor_id: str
    event_type: str
    operator: str
    created_at: str
    payload: JsonDict


class LabelRunResult(TypedDict, total=False):
    """模块入口函数的返回载荷。"""

    factor_id: str
    tag_package: TagPackage
    admission_decision: AdmissionDecision
    route_record: RouteRecord
    lifecycle_event: LifecycleEvent


EVALUATION_SUMMARY_OPTIONAL_DEFAULTS: dict[str, JsonValue] = {
    "evaluation_protocol_key": "",
    "fragility_tags": [],
    "incremental_value_summary": {},
    "admission_confidence": 0.0,
    "primary_failure_reasons": [],
    "detail_page_template_key": "",
    "charts_summary": {},
}

FACTOR_META_OPTIONAL_DEFAULTS: dict[str, JsonValue] = {
    "expression": "",
    "formula_ast": {},
    "ast": {},
    "generator_name": "",
    "campaign_id": "",
    "iteration_id": "",
    "batch_id": "",
    "signal_structure": "",
    "asset_class": "",
    "frequency_bucket": "",
    "domain_root": "",
    "domain": "",
    "lib_coordinates": {},
    "source_library": "",
    "tier": "Tier1",
    "status": "pure",
}

ALLOWED_RULE_OPERATORS = {"==", ">", ">=", "<", "<="}
LABEL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(slots=True)
class DeepSeekConfig:
    """DeepSeek 正式调用配置。

    真实 API key 必须来自环境变量或注入的密钥配置。
    新实现不保留 mock profile。
    """

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str | None = None
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 60
    max_retries: int = 3
    backoff_seconds: list[int] = field(default_factory=lambda: [1, 2, 4])


@dataclass(slots=True)
class LabelModuleConfig:
    """单次 label 模块运行使用的正式流程配置。

    所有路径均可注入，使模块可接入更大的流水线，
    无需改写代码级常量。
    """

    project_root: Path
    tag_policy_path: Path
    route_policy_path: Path
    label_registry_path: Path
    state_dir: Path
    log_dir: Path
    tier1_pure_factor_base_dir: Path
    tier2_fix_base_dir: Path
    tier2x_llm_mutation_base_dir: Path
    tier3a_core_base_dir: Path
    tier3b_satellite_base_dir: Path
    tier3c_feature_material_base_dir: Path
    tier3d_operation_storage_base_dir: Path
    tier4_anti_sample_base_dir: Path
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)


def default_label_module_config(project_root: str | Path = ".") -> LabelModuleConfig:
    """从仓库根目录构造默认正式配置。

    这是唯一声明默认项目路径的位置。调用方可在调用入口函数前覆盖任意字段。
    """

    root = Path(project_root)
    return LabelModuleConfig(
        project_root=root,
        tag_policy_path=root / "evaluation" / "label" / "configs" / "tag_policy.json",
        route_policy_path=root / "evaluation" / "label" / "configs" / "route_policy.json",
        label_registry_path=root
        / "database"
        / "cache"
        / "evaluation"
        / "label"
        / "label_registry.json",
        state_dir=root / "database" / "cache" / "evaluation" / "label",
        log_dir=root / "database" / "log" / "evaluation" / "label",
        tier1_pure_factor_base_dir=root
        / "database"
        / "tier1"
        / "purification_pure_factor_base",
        tier2_fix_base_dir=root / "database" / "tier2" / "2_fix_base",
        tier2x_llm_mutation_base_dir=root
        / "database"
        / "tier2"
        / "2x_llm_mutation_base",
        tier3a_core_base_dir=root
        / "database"
        / "tier3"
        / "3a_core_production_base",
        tier3b_satellite_base_dir=root
        / "database"
        / "tier3"
        / "3b_satellite_production_base",
        tier3c_feature_material_base_dir=root
        / "database"
        / "tier3"
        / "3c_feature_matierial_base",
        tier3d_operation_storage_base_dir=root
        / "database"
        / "tier3"
        / "3d_operation_storage_base",
        tier4_anti_sample_base_dir=root / "database" / "tier4" / "anti_sample_base",
    )


def default_mock_label_module_config(
    project_root: str | Path = ".",
    mock_root: str | Path = "database/mock",
) -> LabelModuleConfig:
    """构造与正式环境拥有相同 tier/cache/log 结构的 mock 配置。

    参数:
        project_root: 仓库根目录。项目应从该根目录运行，对应评审文档中的
            /AutoFactorEvaluation。
        mock_root: mock 数据库根目录。将其切换为结构一致的真实数据库根目录，
            即为预期的上线切换方式。

    返回:
        输入/输出数据库路径指向 database/mock、共享策略文件仍位于
        evaluation/label 下的 LabelModuleConfig。
    """

    root = Path(project_root)
    mock_db = root / mock_root
    return LabelModuleConfig(
        project_root=root,
        tag_policy_path=root / "evaluation" / "label" / "configs" / "tag_policy.json",
        route_policy_path=root / "evaluation" / "label" / "configs" / "route_policy.json",
        label_registry_path=mock_db / "cache" / "evaluation" / "label" / "label_registry.json",
        state_dir=mock_db / "cache" / "evaluation" / "label",
        log_dir=mock_db / "log" / "evaluation" / "label",
        tier1_pure_factor_base_dir=mock_db
        / "tier1"
        / "purification_pure_factor_base",
        tier2_fix_base_dir=mock_db / "tier2" / "2_fix_base",
        tier2x_llm_mutation_base_dir=mock_db / "tier2" / "2x_llm_mutation_base",
        tier3a_core_base_dir=mock_db / "tier3" / "3a_core_production_base",
        tier3b_satellite_base_dir=mock_db / "tier3" / "3b_satellite_production_base",
        tier3c_feature_material_base_dir=mock_db / "tier3" / "3c_feature_matierial_base",
        tier3d_operation_storage_base_dir=mock_db / "tier3" / "3d_operation_storage_base",
        tier4_anti_sample_base_dir=mock_db / "tier4" / "anti_sample_base",
    )


def validate_evaluation_summary(payload: EvaluationSummary) -> EvaluationSummary:
    """严格校验上游评估摘要。

    已实现的评审要求:
    1. v1_review.md 4.3-(1): 所有路径和文件处理均保留在本函数之外。
    2. v1_review.md 4.3-(7): 评估摘要执行严格校验，不只做静默默认值填充。
    3. v1_review.md 共性问题(3): 服务入口依赖显式、带类型的边界校验。

    实现说明:
    1. 必需业务字段必须存在，并具备预期类型。
    2. 枚举值会提前校验，使未知上游状态快速失败。
    3. route_recommendation 会标准化为普通字符串，因为路由逻辑只消费一种稳定结构。
    4. 可选字段在校验后显式标准化。
    """

    normalized: EvaluationSummary = dict(payload)
    for key in ("summary_scorecard", "survival_view", "usage_role", "route_recommendation"):
        if key not in normalized:
            raise ValueError(f"evaluation_summary missing required field: {key}")

    if not isinstance(normalized["summary_scorecard"], dict):
        raise TypeError("evaluation_summary.summary_scorecard must be a JSON object")
    if normalized["survival_view"] not in SURVIVAL_VIEWS:
        raise ValueError(
            f"evaluation_summary.survival_view must be one of {sorted(SURVIVAL_VIEWS)}"
        )
    if normalized["usage_role"] not in USAGE_ROLES:
        raise ValueError(
            f"evaluation_summary.usage_role must be one of {sorted(USAGE_ROLES)}"
        )

    route_recommendation = _normalize_route_recommendation(
        normalized["route_recommendation"]
    )
    if route_recommendation not in ROUTE_RECOMMENDATIONS:
        raise ValueError(
            "evaluation_summary.route_recommendation must be a supported "
            f"value in {sorted(ROUTE_RECOMMENDATIONS)}"
        )
    normalized["route_recommendation"] = route_recommendation

    for key, default_value in EVALUATION_SUMMARY_OPTIONAL_DEFAULTS.items():
        if key not in normalized:
            normalized[key] = _copy_default(default_value)

    if not isinstance(normalized["fragility_tags"], list):
        raise TypeError("evaluation_summary.fragility_tags must be a list")
    if not isinstance(normalized["incremental_value_summary"], dict):
        raise TypeError(
            "evaluation_summary.incremental_value_summary must be a JSON object"
        )
    if not isinstance(normalized["primary_failure_reasons"], list):
        raise TypeError(
            "evaluation_summary.primary_failure_reasons must be a list"
        )
    if not isinstance(normalized["charts_summary"], dict):
        raise TypeError("evaluation_summary.charts_summary must be a JSON object")
    if not _is_number(normalized["admission_confidence"]):
        raise TypeError(
            "evaluation_summary.admission_confidence must be an int or float"
        )

    LOGGER.info(
        "evaluation_summary_validated survival_view=%s usage_role=%s "
        "route_recommendation=%s",
        normalized["survival_view"],
        normalized["usage_role"],
        normalized["route_recommendation"],
    )
    return normalized


def validate_factor_meta(payload: FactorMeta) -> FactorMeta:
    """严格校验源因子元数据。

    已实现的评审要求:
    1. v1_review.md 共性问题(3): 为流水线嵌入提供显式类型边界。
    2. v1_review.md 4.3-(3): 首个版本结构保持简单、可读。

    实现说明:
    1. 当前纯化样例在 manifest.json 中携带 factor_id/candidate_id，
       但不携带 expression；因此该校验器在存储边界将 expression 视为可选。
    2. AST 别名会标准化到一个 ast 字段，使下游代码不需要处理多个结构分支。
    3. 函数会立即拒绝缺失标识符和格式错误的容器字段，不增加业务 fallback 行为。
    """

    normalized: FactorMeta = dict(payload)
    for key in ("factor_id", "candidate_id"):
        if key not in normalized or not isinstance(normalized[key], str) or not normalized[key]:
            raise ValueError(f"factor_meta missing required non-empty field: {key}")

    for key, default_value in FACTOR_META_OPTIONAL_DEFAULTS.items():
        if key not in normalized:
            normalized[key] = _copy_default(default_value)

    if not isinstance(normalized["expression"], str):
        raise TypeError("factor_meta.expression must be a string")
    if not isinstance(normalized["formula_ast"], dict):
        raise TypeError("factor_meta.formula_ast must be a JSON object")
    if not isinstance(normalized["ast"], dict):
        raise TypeError("factor_meta.ast must be a JSON object")

    if not normalized["ast"] and normalized["formula_ast"]:
        normalized["ast"] = dict(normalized["formula_ast"])
    if normalized["domain_root"] == "" and isinstance(normalized["domain"], str):
        normalized["domain_root"] = normalized["domain"]
    if not isinstance(normalized["lib_coordinates"], dict):
        raise TypeError("factor_meta.lib_coordinates must be a JSON object")

    for key in (
        "generator_name",
        "campaign_id",
        "iteration_id",
        "batch_id",
        "signal_structure",
        "asset_class",
        "frequency_bucket",
        "domain_root",
        "domain",
        "source_library",
        "tier",
        "status",
    ):
        if not isinstance(normalized[key], str):
            raise TypeError(f"factor_meta.{key} must be a string")

    LOGGER.info(
        "factor_meta_validated factor_id=%s candidate_id=%s tier=%s status=%s",
        normalized["factor_id"],
        normalized["candidate_id"],
        normalized["tier"],
        normalized["status"],
    )
    return normalized


def validate_tag_policy(payload: JsonDict) -> JsonDict:
    """在规则贴标前校验本地标签策略。

    已实现的评审要求:
    1. v1_review.md 4.3-(3): 标签系统必须标准化，避免临时标签命名。
    2. v1_review.md 4.3-(10): 比较输入格式错误时应显式暴露，不静默忽略。

    实现说明:
    1. 每个规则分组都必须存在且必须是列表。
    2. 指标规则会提前校验运算符和阈值结构。
    3. 策略标签名限制为标准 snake_case 约定。
    """

    if "tag_rule_version" not in payload or not isinstance(payload["tag_rule_version"], str):
        raise ValueError("tag_policy.tag_rule_version must be a string")

    normalized = dict(payload)
    for key in (
        "performance_rules",
        "trading_rules",
        "lifecycle_rules",
        "style_rules",
    ):
        rules = normalized.get(key)
        if not isinstance(rules, list):
            raise TypeError(f"tag_policy.{key} must be a list")

    if not any(normalized[key] for key in (
        "performance_rules",
        "trading_rules",
        "lifecycle_rules",
        "style_rules",
    )):
        raise ValueError(
            "tag_policy has no rules; add formal rule config before running "
            "evaluation.label"
        )

    for rule in normalized["performance_rules"]:
        _validate_metric_rule(rule, "performance_rules")
    for rule in normalized["trading_rules"]:
        _validate_metric_rule(rule, "trading_rules", require_action_tag=True)
    for rule in normalized["lifecycle_rules"]:
        _validate_lifecycle_rule(rule)
    for rule in normalized["style_rules"]:
        _validate_style_rule(rule)

    LOGGER.info(
        "tag_policy_validated version=%s performance_rules=%s trading_rules=%s "
        "lifecycle_rules=%s style_rules=%s",
        normalized["tag_rule_version"],
        len(normalized["performance_rules"]),
        len(normalized["trading_rules"]),
        len(normalized["lifecycle_rules"]),
        len(normalized["style_rules"]),
    )
    return normalized


def validate_route_policy(payload: JsonDict) -> JsonDict:
    """在生成决策前校验路由策略。

    已实现的评审要求:
    1. v1_review.md 4.3-(8): 安全覆盖必须来自配置，而不是硬编码特殊分支。
    2. v1_review.md 4.3-(9): 未知推荐值的路由必须在决策逻辑前显式校验结构。

    实现说明:
    1. route_recommendation_mapping 为必需字段，只能包含支持的推荐键和目标层级。
    2. tier_status_mapping 必须定义本模块使用的每一个路由层级。
    3. safety_override_rules 作为一等路由配置进行校验，而不是注释或隐藏默认值。
    """

    if "policy_name" not in payload or not isinstance(payload["policy_name"], str):
        raise ValueError("route_policy.policy_name must be a string")

    mapping = payload.get("route_recommendation_mapping")
    if not isinstance(mapping, dict):
        raise TypeError("route_policy.route_recommendation_mapping must be an object")
    if not mapping:
        raise ValueError(
            "route_policy.route_recommendation_mapping is empty; add formal "
            "routing config before running evaluation.label"
        )
    for route_key, tier in mapping.items():
        if route_key not in ROUTE_RECOMMENDATIONS:
            raise ValueError(
                "route_policy.route_recommendation_mapping contains unsupported "
                f"recommendation: {route_key}"
            )
        if tier not in ROUTE_TIERS:
            raise ValueError(
                "route_policy.route_recommendation_mapping contains unsupported "
                f"tier: {tier}"
            )

    tier_status_mapping = payload.get("tier_status_mapping")
    if not isinstance(tier_status_mapping, dict):
        raise TypeError("route_policy.tier_status_mapping must be an object")
    for tier in ROUTE_TIERS:
        if tier not in tier_status_mapping:
            raise ValueError(
                f"route_policy.tier_status_mapping missing tier status for {tier}"
            )
        if not isinstance(tier_status_mapping[tier], str) or not tier_status_mapping[tier]:
            raise ValueError(
                f"route_policy.tier_status_mapping[{tier}] must be a non-empty string"
            )

    safety_override_rules = payload.get("safety_override_rules", [])
    if not isinstance(safety_override_rules, list):
        raise TypeError("route_policy.safety_override_rules must be a list")
    for rule in safety_override_rules:
        _validate_route_override_rule(rule, "safety_override_rules")

    fallback_rules = payload.get("fallback_rules", [])
    if not isinstance(fallback_rules, list):
        raise TypeError("route_policy.fallback_rules must be a list")
    for rule in fallback_rules:
        _validate_route_override_rule(rule, "fallback_rules")

    LOGGER.info(
        "route_policy_validated policy_name=%s route_mappings=%s "
        "safety_overrides=%s fallback_rules=%s",
        payload["policy_name"],
        len(mapping),
        len(safety_override_rules),
        len(fallback_rules),
    )
    return dict(payload)


def validate_label_registry(payload: JsonDict) -> JsonDict:
    """校验语义贴标使用的标准标签注册表。

    已实现的评审要求:
    1. v1_review.md 4.3-(3): 标准标签治理必须避免同一语义概念出现多种拼写。
    2. v1_review.md 4.3-(17): LLM 输出应基于严格本地结构标准化，
       而不是自由生成标签。

    实现说明:
    1. 标准标签键和别名必须遵循 snake_case。
    2. 标签键和别名必须全局唯一。
    3. 现在即校验注册表条目，使后续语义标准化可依赖一个确定性结构。
    """

    if "schema_version" not in payload or not isinstance(payload["schema_version"], str):
        raise ValueError("label_registry.schema_version must be a string")
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise TypeError("label_registry.labels must be a list")
    if not labels:
        raise ValueError(
            "label_registry.labels is empty; add formal labels before running "
            "evaluation.label"
        )

    seen_names: set[str] = set()
    normalized_labels: list[JsonDict] = []
    for item in labels:
        if not isinstance(item, dict):
            raise TypeError("label_registry.labels items must be objects")
        label_key = item.get("label_key")
        if not isinstance(label_key, str) or not label_key:
            raise ValueError("label_registry label_key must be a non-empty string")
        _validate_label_key(label_key, "label_registry.labels[].label_key")
        if label_key in seen_names:
            raise ValueError(f"duplicate label_registry label_key: {label_key}")
        seen_names.add(label_key)

        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            raise TypeError(
                f"label_registry label {label_key} aliases must be a list"
            )
        for alias in aliases:
            if not isinstance(alias, str) or not alias:
                raise ValueError(
                    f"label_registry label {label_key} alias must be a non-empty string"
                )
            _validate_label_key(alias, f"label_registry alias for {label_key}")
            if alias in seen_names:
                raise ValueError(
                    f"label_registry alias collides with existing key/alias: {alias}"
                )
            seen_names.add(alias)

        for key in ("display_name", "category", "description", "status"):
            if key not in item or not isinstance(item[key], str) or not item[key]:
                raise ValueError(
                    f"label_registry label {label_key} missing non-empty string field: {key}"
                )

        normalized_label = dict(item)
        normalized_label["aliases"] = list(aliases)
        normalized_labels.append(normalized_label)

    validated_registry = {
        "schema_version": payload["schema_version"],
        "labels": normalized_labels,
    }
    LOGGER.info(
        "label_registry_validated schema_version=%s labels=%s",
        validated_registry["schema_version"],
        len(validated_registry["labels"]),
    )
    return validated_registry


def _normalize_route_recommendation(value: str | JsonDict) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        route_value = value.get("value", value.get("route_recommendation"))
        if isinstance(route_value, str):
            return route_value
    raise TypeError(
        "evaluation_summary.route_recommendation must be a string or an object "
        "containing a string value field"
    )


def _validate_metric_rule(
    rule: JsonDict,
    section_name: str,
    require_action_tag: bool = False,
) -> None:
    if not isinstance(rule, dict):
        raise TypeError(f"tag_policy.{section_name} items must be objects")
    for key in ("metric", "op", "threshold", "tag"):
        if key not in rule:
            raise ValueError(f"tag_policy.{section_name} rule missing field: {key}")
    if not isinstance(rule["metric"], str) or not rule["metric"]:
        raise ValueError(f"tag_policy.{section_name}.metric must be a non-empty string")
    if rule["op"] not in ALLOWED_RULE_OPERATORS:
        raise ValueError(
            f"tag_policy.{section_name}.op must be one of {sorted(ALLOWED_RULE_OPERATORS)}"
        )
    if not _is_number(rule["threshold"]):
        raise TypeError(f"tag_policy.{section_name}.threshold must be a number")
    _validate_label_key(str(rule["tag"]), f"tag_policy.{section_name}.tag")
    if require_action_tag:
        if "action_tag" not in rule:
            raise ValueError(f"tag_policy.{section_name} rule missing field: action_tag")
        _validate_label_key(
            str(rule["action_tag"]),
            f"tag_policy.{section_name}.action_tag",
        )


def _validate_lifecycle_rule(rule: JsonDict) -> None:
    if not isinstance(rule, dict):
        raise TypeError("tag_policy.lifecycle_rules items must be objects")
    for key in ("field", "equals", "tag"):
        if key not in rule:
            raise ValueError(f"tag_policy.lifecycle_rules rule missing field: {key}")
    if rule["field"] not in {"survival_view", "usage_role"}:
        raise ValueError(
            "tag_policy.lifecycle_rules.field must be survival_view or usage_role"
        )
    if not isinstance(rule["equals"], str) or not rule["equals"]:
        raise ValueError("tag_policy.lifecycle_rules.equals must be a non-empty string")
    _validate_label_key(str(rule["tag"]), "tag_policy.lifecycle_rules.tag")


def _validate_style_rule(rule: JsonDict) -> None:
    if not isinstance(rule, dict):
        raise TypeError("tag_policy.style_rules items must be objects")
    if "contains_any" not in rule or "tag" not in rule:
        raise ValueError("tag_policy.style_rules rule must contain contains_any and tag")
    contains_any = rule["contains_any"]
    if not isinstance(contains_any, list) or not contains_any:
        raise ValueError("tag_policy.style_rules.contains_any must be a non-empty list")
    for token in contains_any:
        if not isinstance(token, str) or not token:
            raise ValueError(
                "tag_policy.style_rules.contains_any items must be non-empty strings"
            )
    _validate_label_key(str(rule["tag"]), "tag_policy.style_rules.tag")


def _validate_route_override_rule(rule: JsonDict, section_name: str) -> None:
    if not isinstance(rule, dict):
        raise TypeError(f"route_policy.{section_name} items must be objects")
    for key in ("field", "equals", "tier", "reason"):
        if key not in rule:
            raise ValueError(f"route_policy.{section_name} rule missing field: {key}")
    if rule["field"] not in {"survival_view", "usage_role"}:
        raise ValueError(
            f"route_policy.{section_name}.field must be survival_view or usage_role"
        )
    if not isinstance(rule["equals"], str) or not rule["equals"]:
        raise ValueError(f"route_policy.{section_name}.equals must be a non-empty string")
    if rule["tier"] not in ROUTE_TIERS:
        raise ValueError(
            f"route_policy.{section_name}.tier must be one of {sorted(ROUTE_TIERS)}"
        )
    if not isinstance(rule["reason"], str) or not rule["reason"]:
        raise ValueError(f"route_policy.{section_name}.reason must be a non-empty string")


def _validate_label_key(value: str, field_name: str) -> None:
    if not LABEL_KEY_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must follow canonical snake_case, got: {value}"
        )


def _copy_default(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
