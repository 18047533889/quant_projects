"""Evaluation label 模块的单因子入口。

模块: evaluation.label
职责:
1. 编排单个因子的贴标签、路由和产物持久化流程。
2. 保持流程编排、计算逻辑和磁盘 I/O 辅助函数分离。
3. 暴露一个用于接入流水线的入口函数。
"""

from __future__ import annotations

from pathlib import Path

from .deepseek_client import DeepSeekClient
from .logging_utils import configure_label_logger
from .routing import route_factor
from .schemas import (
    LabelModuleConfig,
    LabelRunResult,
    default_label_module_config,
    validate_evaluation_summary,
    validate_factor_meta,
    validate_label_registry,
    validate_route_policy,
    validate_tag_policy,
)
from .storage import (
    load_evaluation_summary,
    load_factor_meta,
    load_json_file,
    materialize_routed_factor_package,
    persist_label_state,
    prepare_target_factor_dir,
)
from .tagging import generate_tags


def run_label_pipeline(
    source_factor_dir: str | Path,
    evaluation_summary_path: str | Path,
    config: LabelModuleConfig | None = None,
    operator: str = "evaluation.label",
    deepseek_client: DeepSeekClient | None = None,
    materialize: bool = True,
) -> LabelRunResult:
    """运行单个因子的正式贴标流程。

    函数功能:
        作为流水线嵌入时使用的唯一公共入口。

    已实现的评审要求:
    1. v1_review.md 共性问题(1): 所有文件路径均通过 LabelModuleConfig 由调用方配置，
       不在服务层硬编码。
    2. v1_review.md 共性问题(2): 入口函数只处理单个因子，不负责遍历流水线。
    3. v1_review.md 共性问题(3): 在 evaluation.label.__init__ 中暴露一个带类型的公共入口。

    参数:
        source_factor_dir: 单个因子目录，位于
            database/tier1/purification_pure_factor_base/<factor_id> 下。
        evaluation_summary_path: 同一因子的上游评估摘要 JSON 文件。
        config: 正式贴标服务使用的路径和环境配置。
        operator: 写入生命周期记录的操作者名称。
        deepseek_client: 可选语义贴标客户端。正式调用方不传入该参数，服务会使用
            DeepSeekClient(config.deepseek)；测试和演示可注入具备同名方法的确定性客户端。
        materialize: 是否把因子产物物化到目标 tier 目录并持久化标签状态。
            默认 True（独立调用时执行完整入库落地）。当上层评估 pipeline 只需要
            "标签 + 入库推荐路由信息"、由更高级管道统一落地入库时，应传 False，
            此时仅做纯计算（贴标 + 路由），不写目标目录、不更新共享标签注册表。

    返回:
        包含当前因子的 tag_package、admission_decision、route_record 和
        lifecycle_event 的字典。
    """

    if config is None:
        raise ValueError(
            "LabelModuleConfig 不能为空，请通过配置文件指定 label 参数。"
            "参考: all_configs/auto_factor_evaluation/config.yaml"
        )
    active_config = config
    logger = configure_label_logger(active_config.log_dir)
    logger.info(
        "label_pipeline_start source_factor_dir=%s evaluation_summary_path=%s "
        "operator=%s",
        source_factor_dir,
        evaluation_summary_path,
        operator,
    )

    try:
        logger.info("load_inputs_start")
        factor_meta = validate_factor_meta(load_factor_meta(source_factor_dir))
        evaluation_summary = validate_evaluation_summary(
            load_evaluation_summary(evaluation_summary_path)
        )
        logger.info(
            "load_inputs_done factor_id=%s candidate_id=%s route_recommendation=%s",
            factor_meta["factor_id"],
            factor_meta["candidate_id"],
            evaluation_summary["route_recommendation"],
        )

        logger.info(
            "load_config_start tag_policy=%s route_policy=%s label_registry=%s",
            active_config.tag_policy_path,
            active_config.route_policy_path,
            active_config.label_registry_path,
        )
        tag_policy = validate_tag_policy(load_json_file(active_config.tag_policy_path))
        route_policy = validate_route_policy(
            load_json_file(active_config.route_policy_path)
        )
        label_registry = validate_label_registry(
            load_json_file(active_config.label_registry_path)
        )
        logger.info(
            "load_config_done tag_rules=%s route_mappings=%s registry_labels=%s",
            sum(
                len(tag_policy[key])
                for key in (
                    "performance_rules",
                    "trading_rules",
                    "lifecycle_rules",
                    "style_rules",
                )
            ),
            len(route_policy["route_recommendation_mapping"]),
            len(label_registry["labels"]),
        )

        semantic_client = deepseek_client or DeepSeekClient(active_config.deepseek)
        tag_package = generate_tags(
            factor_meta=factor_meta,
            evaluation_summary=evaluation_summary,
            tag_policy=tag_policy,
            label_registry=label_registry,
            deepseek_client=semantic_client,
        )
        logger.info(
            "tagging_done factor_id=%s primary_label=%s registry_action=%s",
            factor_meta["factor_id"],
            tag_package["deepseek_tags"]["primary_label"],
            tag_package["label_governance"]["registry_action"],
        )

        admission_decision, route_record, lifecycle_event = route_factor(
            factor_meta=factor_meta,
            evaluation_summary=evaluation_summary,
            tag_package=tag_package,
            route_policy=route_policy,
            config=active_config,
            operator=operator,
        )
        admission_decision["factor_id"] = factor_meta["factor_id"]
        logger.info(
            "routing_done factor_id=%s tier=%s reason=%s",
            factor_meta["factor_id"],
            admission_decision["tier"],
            admission_decision["reason"],
        )

        if not materialize:
            # 路由信息计算模式：仅返回标签与入库推荐路由，不物化目标目录、不持久化
            # 共享标签注册表；具体入库落地由更高级别调用管道统一负责。
            logger.info(
                "label_pipeline_done_route_only factor_id=%s target_factor_dir=%s",
                factor_meta["factor_id"],
                route_record["target_factor_dir"],
            )
            return {
                "factor_id": factor_meta["factor_id"],
                "tag_package": tag_package,
                "admission_decision": admission_decision,
                "route_record": route_record,
                "lifecycle_event": lifecycle_event,
            }

        target_factor_dir = prepare_target_factor_dir(
            factor_id=factor_meta["factor_id"],
            decision=admission_decision,
            config=active_config,
        )
        logger.info(
            "prepare_target_factor_dir_done factor_id=%s target_factor_dir=%s",
            factor_meta["factor_id"],
            target_factor_dir,
        )
        materialize_routed_factor_package(
            source_factor_dir=source_factor_dir,
            target_factor_dir=target_factor_dir,
            tag_package=tag_package,
            decision=admission_decision,
            route_record=route_record,
            evaluation_summary=evaluation_summary,
        )
        logger.info(
            "materialize_routed_factor_package_done factor_id=%s target_factor_dir=%s",
            factor_meta["factor_id"],
            target_factor_dir,
        )
        persist_label_state(
            label_registry=label_registry,
            label_governance=tag_package["label_governance"],
            factor_id=factor_meta["factor_id"],
            lifecycle_event=lifecycle_event,
            config=active_config,
        )
        logger.info(
            "persist_label_state_done factor_id=%s registry_labels=%s",
            factor_meta["factor_id"],
            len(label_registry["labels"]),
        )
        logger.info(
            "label_pipeline_done factor_id=%s target_factor_dir=%s",
            factor_meta["factor_id"],
            target_factor_dir,
        )

        return {
            "factor_id": factor_meta["factor_id"],
            "tag_package": tag_package,
            "admission_decision": admission_decision,
            "route_record": route_record,
            "lifecycle_event": lifecycle_event,
        }
    except Exception:
        logger.exception(
            "label_pipeline_failed source_factor_dir=%s evaluation_summary_path=%s",
            source_factor_dir,
            evaluation_summary_path,
        )
        raise
