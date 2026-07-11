"""evaluation.label 的文件系统边界。

模块: evaluation.label
职责:
1. 读取 label 模块所需的单因子输入。
2. 原子化持久化标签和路由产物。
3. 将路由后的因子包物化到目标层级目录。

本文件负责所有磁盘 I/O。业务决策保留在 tagging.py / routing.py 中。
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
import tempfile

from .logging_utils import get_label_logger
from .schemas import (
    AdmissionDecision,
    EvaluationSummary,
    FactorMeta,
    JsonDict,
    LabelGovernance,
    LabelModuleConfig,
    LifecycleEvent,
    RouteRecord,
    TagPackage,
)


LOGGER = get_label_logger()


def load_factor_meta(source_factor_dir: str | Path) -> FactorMeta:
    """从一个纯化因子目录加载源因子元数据。

    已实现的评审要求:
    1. v1_review.md 共性问题(1): 源路径由外部传入，不在模块内部硬编码。
    2. v1_review.md 共性问题(2): 加载器一次只处理一个因子目录。

    实现说明:
    1. 当前项目样例在 manifest.json 中提供 factor_id/candidate_id，
       在 candidate.json 中提供流程状态分段。
    2. 本加载器提取当前实际可用的元数据；当上游文件尚未携带语义字段时保持为空。
    3. 文件夹名必须与 manifest 中的 factor_id 完全一致。
    """

    factor_dir = Path(source_factor_dir)
    candidate_path = factor_dir / "candidate.json"
    manifest_path = factor_dir / "manifest.json"

    candidate_payload = load_json_file(candidate_path)
    manifest_payload = load_json_file(manifest_path)

    factor_id = manifest_payload.get("factor_id")
    candidate_id = manifest_payload.get("candidate_id")
    if not isinstance(factor_id, str) or not factor_id:
        raise ValueError("manifest.json missing non-empty factor_id")
    if factor_dir.name != factor_id:
        raise ValueError(
            f"source factor directory name {factor_dir.name} does not match factor_id {factor_id}"
        )
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("manifest.json missing non-empty candidate_id")

    assetization = candidate_payload.get("Assetization", {})
    purification = candidate_payload.get("Purification", {})
    if not isinstance(assetization, dict):
        assetization = {}

    # 语义字段优先从 candidate.json 读取（顶层或 Assetization 段），缺失时保持为空。
    # 这些字段驱动 style_rules / source_tags（§2.2）与 LLM 语义上下文（§2.3）；
    # 全部缺失时贴标能力降级，故在下方显式打 WARNING，使口径退化可见。
    def _pick_str(*keys: str) -> str:
        for container in (candidate_payload, assetization):
            for key in keys:
                value = container.get(key)
                if isinstance(value, str) and value:
                    return value
        return ""

    def _pick_dict(*keys: str) -> dict:
        for container in (candidate_payload, assetization):
            for key in keys:
                value = container.get(key)
                if isinstance(value, dict) and value:
                    return value
        return {}

    expression = _pick_str("expression", "formula")
    formula_ast = _pick_dict("ast", "formula_ast")
    generator_name = _pick_str("generator_name")
    asset_class = _pick_str("asset_class")

    factor_meta: FactorMeta = {
        "factor_id": factor_id,
        "candidate_id": candidate_id,
        "expression": expression,
        "formula_ast": formula_ast,
        "ast": formula_ast,
        "generator_name": generator_name,
        "campaign_id": "",
        "iteration_id": str(candidate_payload.get("iteration_id", "")),
        "batch_id": str(candidate_payload.get("batch_id", "")),
        "signal_structure": "cross_sectional",
        "asset_class": asset_class,
        "frequency_bucket": str(assetization.get("frequency_bucket", "")),
        "domain_root": str(assetization.get("domain_root", "")),
        "domain": str(assetization.get("domain", "")),
        "lib_coordinates": {},
        "source_library": str(manifest_payload.get("library", "")),
        "tier": "Tier1",
        "status": "pure",
    }
    if factor_meta["domain_root"] == "" and factor_meta["domain"]:
        factor_meta["domain_root"] = factor_meta["domain"]
    if isinstance(purification, dict) and "Label" in purification:
        factor_meta["status"] = str(purification["Label"]).lower()
    if not expression:
        LOGGER.warning(
            "factor_meta_expression_missing factor_id=%s: style/source/semantic tagging "
            "will be degraded because candidate.json carries no factor expression",
            factor_id,
        )
    LOGGER.info(
        "factor_meta_loaded factor_id=%s candidate_id=%s source_factor_dir=%s",
        factor_id,
        candidate_id,
        factor_dir,
    )
    return factor_meta


def load_evaluation_summary(evaluation_summary_path: str | Path) -> EvaluationSummary:
    """加载单个因子的上游评估摘要。

    已实现的评审要求:
    1. v1_review.md 共性问题(1): 输入路径始终由调用方传入。
    2. v1_review.md 4.3-(7): 原始加载后再执行严格结构校验，
       因此 I/O 代码不会静默修改结构。
    """

    payload = load_json_file(evaluation_summary_path)
    LOGGER.info("evaluation_summary_loaded path=%s", evaluation_summary_path)
    return dict(payload)


def load_json_file(path: str | Path) -> JsonDict:
    """从磁盘读取一个 JSON 文件。

    实现说明:
    1. 无效 JSON 允许直接抛出异常，因为本模块应暴露损坏的上游产物，
       而不是掩盖问题。
    """

    json_path = Path(path)
    raw_text = json_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise ValueError(
            f"configuration JSON file is empty: {json_path}. Add formal config "
            "before running evaluation.label"
        )
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {json_path}")
    LOGGER.info("json_file_loaded path=%s keys=%s", json_path, sorted(payload.keys()))
    return payload


def write_json_atomic(path: str | Path, payload: JsonDict | list[object]) -> None:
    """原子化写入一个 JSON 产物。

    已实现的评审要求:
    1. v1_review.md 4.3-(13): 多产物写入依赖原子 JSON 写入原语，
       不再直接使用 open(..., "w")。
    2. v1_review.md 4.3-(14): 文件通过临时文件 + replace 顺序写入，
       避免进程中断后留下半截 JSON。
    """

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=json_path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.write("\n")
        temp_path = Path(tmp_file.name)
    temp_path.replace(json_path)
    LOGGER.info("json_file_written path=%s", json_path)


def prepare_target_factor_dir(
    factor_id: str,
    decision: AdmissionDecision,
    config: LabelModuleConfig,
) -> Path:
    """解析并创建路由后的目标因子目录。

    已实现的评审要求:
    1. v1_review.md 共性问题(1): 磁盘契约路径通过 config 传入。
    2. v1_review.md 4.3-(11): Tier2 / Tier2X 会解析到真实物化目录，
       不再成为消失分支。
    """

    tier = decision["tier"]
    if tier == "Tier3A":
        base_dir = config.tier3a_core_base_dir
    elif tier == "Tier3B":
        base_dir = config.tier3b_satellite_base_dir
    elif tier == "Tier3C":
        base_dir = config.tier3c_feature_material_base_dir
    elif tier == "Tier3D":
        base_dir = config.tier3d_operation_storage_base_dir
    elif tier == "Tier2":
        base_dir = config.tier2_fix_base_dir
    elif tier == "Tier2X":
        base_dir = config.tier2x_llm_mutation_base_dir
    elif tier == "Tier4":
        base_dir = config.tier4_anti_sample_base_dir
    else:
        raise ValueError(f"unsupported routing tier: {tier}")

    factor_dir = base_dir / factor_id
    factor_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "target_factor_dir_prepared factor_id=%s tier=%s target_factor_dir=%s",
        factor_id,
        tier,
        factor_dir,
    )
    return factor_dir


def materialize_routed_factor_package(
    source_factor_dir: str | Path,
    target_factor_dir: str | Path,
    tag_package: TagPackage,
    decision: AdmissionDecision,
    route_record: RouteRecord,
    evaluation_summary: EvaluationSummary | None = None,
) -> None:
    """将最终路由后的因子包写入目标层级目录。

    已实现的评审要求:
    1. v1_review.md 4.3-(11): Tier2 / Tier2X 会接收真实磁盘产物。
    2. v1_review.md 共性问题(1): 产物文件名与项目层级契约一致，
       并写入调用方配置的路径下。

    实现说明:
    1. candidate.json / manifest.json 作为上游事实源复制。
    2. 当源目录存在 data.parquet 时同步复制。
    3. label 模块将自身产物写在复制后的源文件旁边，
       方便下游模块消费一个自包含的因子目录。
    """

    source_dir = Path(source_factor_dir)
    target_dir = Path(target_factor_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "materialize_package_start factor_id=%s source_dir=%s target_dir=%s",
        tag_package["factor_id"],
        source_dir,
        target_dir,
    )

    for filename in ("candidate.json", "manifest.json"):
        copy2(source_dir / filename, target_dir / filename)
        LOGGER.info("source_artifact_copied filename=%s target_dir=%s", filename, target_dir)

    data_path = source_dir / "data.parquet"
    if data_path.is_file():
        copy2(data_path, target_dir / "data.parquet")
        LOGGER.info("source_artifact_copied filename=data.parquet target_dir=%s", target_dir)

    write_json_atomic(target_dir / "tag_package.json", tag_package)
    write_json_atomic(target_dir / "factor_taxonomy_tags.json", tag_package["rule_tags"])
    write_json_atomic(
        target_dir / "factor_semantic_profile.json",
        tag_package["deepseek_tags"],
    )
    write_json_atomic(target_dir / "admission_decision.json", decision)
    write_json_atomic(target_dir / "route_record.json", route_record)
    if evaluation_summary is not None:
        write_json_atomic(
            target_dir / "summary_scorecard.json",
            evaluation_summary["summary_scorecard"],
        )
    LOGGER.info(
        "materialize_package_done factor_id=%s tier=%s target_dir=%s",
        tag_package["factor_id"],
        decision["tier"],
        target_dir,
    )


def persist_label_state(
    label_registry: JsonDict,
    label_governance: LabelGovernance,
    factor_id: str,
    lifecycle_event: LifecycleEvent,
    config: LabelModuleConfig,
) -> None:
    """持久化模块级标签治理状态。

    已实现的评审要求:
    1. v1_review.md 4.3-(12): 生命周期审计以完整事件载荷存储。
    2. v1_review.md 4.3-(13): 状态文件通过原子写入更新。
    3. 标签事件记录: 已有标签集合和新增/使用决策记录均持久化到正式模块缓存路径下。
    """

    config.state_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "persist_label_state_start factor_id=%s state_dir=%s registry_action=%s",
        factor_id,
        config.state_dir,
        label_governance["registry_action"],
    )
    write_json_atomic(config.state_dir / "label_registry.json", label_registry)

    governance_records_path = config.state_dir / "label_governance_records.json"
    if governance_records_path.is_file():
        governance_records_payload = _load_json_value(governance_records_path)
        if not isinstance(governance_records_payload, list):
            raise TypeError(
                "state label_governance_records.json must contain a list: "
                f"{governance_records_path}"
            )
        governance_records = list(governance_records_payload)
    else:
        governance_records = []
    governance_records.append(
        {
            "factor_id": factor_id,
            "governance": dict(label_governance),
        }
    )
    write_json_atomic(governance_records_path, governance_records)

    lifecycle_events_path = config.state_dir / "lifecycle_events.json"
    if lifecycle_events_path.is_file():
        existing_payload = _load_json_value(lifecycle_events_path)
        if not isinstance(existing_payload, list):
            raise TypeError(
                f"state lifecycle_events.json must contain a list: {lifecycle_events_path}"
            )
        lifecycle_events = list(existing_payload)
    else:
        lifecycle_events = []
    lifecycle_events.append(dict(lifecycle_event))
    write_json_atomic(lifecycle_events_path, lifecycle_events)
    LOGGER.info(
        "persist_label_state_done factor_id=%s governance_records=%s "
        "lifecycle_events=%s",
        factor_id,
        len(governance_records),
        len(lifecycle_events),
    )


def _load_json_value(path: str | Path) -> object:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)
