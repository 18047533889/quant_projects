"""evaluation.label 的准入路由与生命周期规划。

模块: evaluation.label
职责:
1. 将评估推荐和标签转换为最终路由决策。
2. 构造供下游持久化使用的路由记录和生命周期记录。
3. 保持路由逻辑与磁盘 I/O 分离。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .logging_utils import get_label_logger
from .schemas import (
    AdmissionDecision,
    EvaluationSummary,
    FactorMeta,
    JsonDict,
    LabelModuleConfig,
    LifecycleEvent,
    RouteRecord,
    TagPackage,
)

LOGGER = get_label_logger()

INCUBATION_GOALS = {
    "Tier2": "fix factor defects and resubmit for production review",
    "Tier2X": "mutate or optimize the factor before production admission",
    "Tier3D": "store as operational reserve and monitor for future reuse",
}


def route_factor(
    factor_meta: FactorMeta,
    evaluation_summary: EvaluationSummary,
    tag_package: TagPackage,
    route_policy: JsonDict,
    config: LabelModuleConfig,
    operator: str,
) -> tuple[AdmissionDecision, RouteRecord, LifecycleEvent]:
    """构造单个因子的路由输出。

    已实现的评审要求:
    1. v1_review.md 共性问题(2): 模块一次只处理一个因子，
       不执行外层遍历或流水线编排。
    2. v1_review.md 4.3-(12): 生命周期事件携带时间戳和操作者信息，
       不再是不完整的状态迁移载荷。

    实现说明:
    1. 路由保持纯计算逻辑，不创建目录。
    2. 目标路径由 config + factor_id 确定性解析。
    3. 一个准入决策对应一条路由记录和一条生命周期事件。
    """

    LOGGER.info("route_factor_start factor_id=%s", factor_meta["factor_id"])
    decision = decide_admission(evaluation_summary, tag_package, route_policy)
    factor_id = factor_meta["factor_id"]
    target_factor_dir = _target_factor_dir_for_tier(decision["tier"], factor_id, config)
    route_record = build_route_record(
        factor_meta=factor_meta,
        decision=decision,
        target_factor_dir=target_factor_dir,
        operator=operator,
    )
    lifecycle_event = build_lifecycle_event(
        factor_meta=factor_meta,
        decision=decision,
        operator=operator,
    )
    LOGGER.info(
        "route_factor_done factor_id=%s tier=%s target_factor_dir=%s",
        factor_id,
        decision["tier"],
        target_factor_dir,
    )
    return decision, route_record, lifecycle_event


def decide_admission(
    evaluation_summary: EvaluationSummary,
    tag_package: TagPackage,
    route_policy: JsonDict,
) -> AdmissionDecision:
    """决定一个已评估因子的目标层级。

    已实现的评审要求:
    1. v1_review.md 4.3-(8): 安全覆盖规则由配置驱动。
    2. v1_review.md 4.3-(9): 未知路由推荐值直接抛出异常，
       不隐式落入 Tier4 归档路径。
    3. v1_review.md 4.3-(11): Tier2 / Tier2X 路由到真实输出层级，
       不再作为空操作分支处理。

    实现说明:
    1. 安全规则优先于常规推荐映射。
    2. 路由器不为未知推荐值使用通用 fallback；应修复配置或上游评估。
    3. 仅对表示后续工作流的层级附加孵化元数据。
    """

    overridden = apply_safety_overrides(evaluation_summary, route_policy)
    if overridden is not None:
        LOGGER.info(
            "admission_decision_safety_override tier=%s reason=%s",
            overridden["tier"],
            overridden["reason"],
        )
        return overridden

    route_recommendation = evaluation_summary["route_recommendation"]
    if not isinstance(route_recommendation, str):
        raise TypeError(
            "evaluation_summary.route_recommendation must already be normalized "
            "to a string before routing"
        )

    mapping = route_policy["route_recommendation_mapping"]
    if route_recommendation not in mapping:
        raise ValueError(
            "route_policy.route_recommendation_mapping is missing recommendation: "
            f"{route_recommendation}"
        )

    tier = mapping[route_recommendation]
    decision: AdmissionDecision = {
        "tier": tier,
        "reason": f"route_recommendation={route_recommendation}",
        "actions": [],
        "decided_by": "evaluation.label.route_policy",
    }
    if "factor_id" in evaluation_summary:
        decision["factor_id"] = str(evaluation_summary["factor_id"])
    if tier in INCUBATION_GOALS:
        decision["incubation_goal"] = INCUBATION_GOALS[tier]
    LOGGER.info(
        "admission_decision_mapping route_recommendation=%s tier=%s",
        route_recommendation,
        tier,
    )
    return decision


def apply_safety_overrides(
    evaluation_summary: EvaluationSummary,
    route_policy: JsonDict,
) -> AdmissionDecision | None:
    """在常规路由前应用可配置安全规则。

    已实现的评审要求:
    1. v1_review.md 4.3-(8): 安全覆盖规则从配置读取，
       不在代码中重复硬编码。

    实现说明:
    1. 规则按 field/equals 精确匹配。
    2. 第一条命中规则生效，使策略顺序可审计。
    """

    for rule in route_policy.get("safety_override_rules", []):
        field_name = rule["field"]
        expected_value = rule["equals"]
        current_value = evaluation_summary.get(field_name)
        if current_value == expected_value:
            LOGGER.info(
                "safety_override_matched field=%s value=%s tier=%s reason=%s",
                field_name,
                expected_value,
                rule["tier"],
                rule["reason"],
            )
            decision: AdmissionDecision = {
                "tier": rule["tier"],
                "reason": rule["reason"],
                "actions": [],
                "decided_by": "evaluation.label.safety_override",
            }
            if rule["tier"] in INCUBATION_GOALS:
                decision["incubation_goal"] = INCUBATION_GOALS[rule["tier"]]
            return decision
    return None


def build_route_record(
    factor_meta: FactorMeta,
    decision: AdmissionDecision,
    target_factor_dir: Path,
    operator: str,
) -> RouteRecord:
    """为一个准入决策构造持久化路由记录。

    已实现的评审要求:
    1. v1_review.md 4.3-(12): 路由/生命周期审计包含时间戳和操作者，
       不再只有局部状态元数据。

    实现说明:
    1. 本模块来源始终为纯化后的 Tier1 因子库。
    2. 状态命名优先来自因子元数据；不存在时使用正式来源默认值。
    """

    from_tier = factor_meta.get("tier", "Tier1")
    from_state = factor_meta.get("status", "pure")
    to_tier = decision["tier"]
    to_state = _state_for_tier(to_tier)
    return {
        "factor_id": factor_meta["factor_id"],
        "from_state": from_state,
        "to_state": to_state,
        "from_tier": from_tier,
        "to_tier": to_tier,
        "trigger_type": "admission_decision",
        "trigger_ref": decision["reason"],
        "operator": operator,
        "created_at": _utc_now(),
        "target_factor_dir": str(target_factor_dir),
    }


def build_lifecycle_event(
    factor_meta: FactorMeta,
    decision: AdmissionDecision,
    operator: str,
) -> LifecycleEvent:
    """构造与路由记录配套的生命周期事件。

    已实现的评审要求:
    1. v1_review.md 4.3-(12): 生命周期事件包含时间戳、操作者和完整迁移载荷。
    """

    return {
        "factor_id": factor_meta["factor_id"],
        "event_type": "label_route_applied",
        "operator": operator,
        "created_at": _utc_now(),
        "payload": {
            "from_tier": factor_meta.get("tier", "Tier1"),
            "from_state": factor_meta.get("status", "pure"),
            "to_tier": decision["tier"],
            "to_state": _state_for_tier(decision["tier"]),
            "reason": decision["reason"],
            "incubation_goal": decision.get("incubation_goal", ""),
            "actions": list(decision.get("actions", [])),
        },
    }


def _target_factor_dir_for_tier(
    tier: str,
    factor_id: str,
    config: LabelModuleConfig,
) -> Path:
    return _target_base_dir_for_tier(tier, config) / factor_id


def _target_base_dir_for_tier(tier: str, config: LabelModuleConfig) -> Path:
    if tier == "Tier3A":
        return config.tier3a_core_base_dir
    if tier == "Tier3B":
        return config.tier3b_satellite_base_dir
    if tier == "Tier3C":
        return config.tier3c_feature_material_base_dir
    if tier == "Tier3D":
        return config.tier3d_operation_storage_base_dir
    if tier == "Tier2":
        return config.tier2_fix_base_dir
    if tier == "Tier2X":
        return config.tier2x_llm_mutation_base_dir
    if tier == "Tier4":
        return config.tier4_anti_sample_base_dir
    raise ValueError(f"unsupported routing tier: {tier}")


def _state_for_tier(tier: str) -> str:
    if tier in {"Tier3A", "Tier3B", "Tier3C"}:
        return "production"
    if tier == "Tier3D":
        return "reserve"
    if tier == "Tier2":
        return "incubating"
    if tier == "Tier2X":
        return "optimizing"
    if tier == "Tier4":
        return "archived"
    raise ValueError(f"unsupported routing tier: {tier}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
