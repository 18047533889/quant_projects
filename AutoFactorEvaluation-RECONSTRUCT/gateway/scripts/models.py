"""
数据模型定义 - 对应 Disk Contracts 中的数据结构

包含:
- GatewayLabel: 网关路由标签枚举
- Candidate: 候选因子数据结构（对应 candidate.json）
- GatewayResult: 网关判断结果
- Manifest: 清单文件结构（对应 manifest.json）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class DataQualityLabel(str, Enum):
    """数据质量标签"""
    PASS = "Pass"
    WARNING = "Warning"
    FAIL = "Fail"


class GatewayLabel(str, Enum):
    """网关路由标签"""
    PASS = "Pass"  # 通过，进入资产化
    DUPLICATED = "Duplicated"  # 临时观察，需要人工审核或引用历史报告
    REJECTED = "Rejected"  # 拒绝，进入反样本库


class ValidationStatus(str, Enum):
    """验证状态"""
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class GatewaySegment:
    """
    candidate.json 中的 Gateway 段落

    对应 Disk Contracts 要求：各模块只追加自己的阶段段落
    """
    label: GatewayLabel
    reason: str  # Temp/Rejected 时必填
    run_id: str  # 本次运行的唯一标识
    checked_at: str  # ISO 8601 格式时间戳

    # 可选：诊断信息（用于缓存报告）
    is_legal: Optional[bool] = None
    has_future: Optional[bool] = None
    complexity_score: Optional[float] = None
    is_within_budget: Optional[bool] = None
    is_duplicate: Optional[bool] = None
    historical_report_id: Optional[str] = None
    error_message: Optional[str] = None
    # 数据质量检测结果
    data_quality_label: Optional[str] = None
    data_quality_score: Optional[float] = None
    data_quality_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatewaySegment":
        """从字典创建实例"""
        return cls(
            label=GatewayLabel(data["Label"]),
            reason=data.get("Reason", ""),
            run_id=data["run_id"],
            checked_at=data["checked_at"],
            is_legal=data.get("is_legal"),
            has_future=data.get("has_future"),
            complexity_score=data.get("complexity_score"),
            is_within_budget=data.get("is_within_budget"),
            is_duplicate=data.get("is_duplicate"),
            historical_report_id=data.get("historical_report_id"),
            error_message=data.get("error_message"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于 JSON 序列化"""
        result = {
            "Label": self.label.value,
            "Reason": self.reason,
            "run_id": self.run_id,
            "checked_at": self.checked_at,
        }
        # 添加非 None 的可选字段
        if self.is_legal is not None:
            result["is_legal"] = self.is_legal
        if self.has_future is not None:
            result["has_future"] = self.has_future
        if self.complexity_score is not None:
            result["complexity_score"] = self.complexity_score
        if self.is_within_budget is not None:
            result["is_within_budget"] = self.is_within_budget
        if self.is_duplicate is not None:
            result["is_duplicate"] = self.is_duplicate
        if self.historical_report_id is not None:
            result["historical_report_id"] = self.historical_report_id
        if self.error_message is not None:
            result["error_message"] = self.error_message
        return result


@dataclass
class Candidate:
    """
    候选因子数据结构 - 对应 candidate.json

    符合 Disk Contracts 要求：
    - factor_id 在 Assetization 前必须为 None
    - 各阶段段落（Gateway, Assetization, Purification, Evaluation）作为独立字段
    """
    schema_version: str
    candidate_id: str
    expr: str  # 因子表达式（原 Expr 字段）
    config: Dict[str, Any]  # 因子配置（原 Config 字段）
    born_timestamp: str  # ISO 8601 格式
    basic_info: Dict[str, Any]  # 基础信息

    # 可选字段
    factor_id: Optional[str] = None  # Assetization 阶段生成，网关阶段为 None
    campaign_id: Optional[str] = None
    batch_id: Optional[str] = None

    # 各阶段段落（各模块只追加自己的段落）
    gateway: Optional[GatewaySegment] = None
    assetization: Optional[Dict[str, Any]] = None
    purification: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """从 JSON 字典创建 Candidate 实例"""
        # 解析 Gateway 段落（如果存在）
        gateway_segment = None
        if "Gateway" in data and data["Gateway"]:
            gateway_segment = GatewaySegment.from_dict(data["Gateway"])

        return cls(
            schema_version=data["schema_version"],
            candidate_id=data["candidate_id"],
            expr=data["Expr"],
            config=data["Config"],
            born_timestamp=data["BornTimestamp"],
            basic_info=data["BasicInfo"],
            factor_id=data.get("factor_id"),  # 可能为 None
            campaign_id=data.get("campaign_id"),
            batch_id=data.get("batch_id"),
            gateway=gateway_segment,
            assetization=data.get("Assetization"),
            purification=data.get("Purification"),
            evaluation=data.get("Evaluation"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于 JSON 序列化"""
        result = {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "Expr": self.expr,
            "Config": self.config,
            "BornTimestamp": self.born_timestamp,
            "BasicInfo": self.basic_info,
        }

        # 添加可选字段（如果存在）
        if self.factor_id is not None:
            result["factor_id"] = self.factor_id
        if self.campaign_id is not None:
            result["campaign_id"] = self.campaign_id
        if self.batch_id is not None:
            result["batch_id"] = self.batch_id

        # 添加各阶段段落（如果存在）
        if self.gateway is not None:
            result["Gateway"] = self.gateway.to_dict()
        if self.assetization is not None:
            result["Assetization"] = self.assetization
        if self.purification is not None:
            result["Purification"] = self.purification
        if self.evaluation is not None:
            result["Evaluation"] = self.evaluation

        return result


@dataclass
class GatewayResult:
    """
    网关判断结果（内部使用，用于在模块间传递）

    注意：这个不是直接写入 JSON 的结构，而是用于 Gateway Core 内部传递
    """
    label: GatewayLabel
    reason: str
    run_id: str
    checked_at: str

    # 诊断信息
    is_legal: Optional[bool] = None
    has_future: Optional[bool] = None
    complexity_score: Optional[float] = None
    is_within_budget: Optional[bool] = None
    is_duplicate: Optional[bool] = None
    historical_report_id: Optional[str] = None
    error_message: Optional[str] = None
    # 数据质量检测结果
    data_quality_label: Optional[str] = None
    data_quality_score: Optional[float] = None
    data_quality_reason: Optional[str] = None

    def to_gateway_segment(self) -> GatewaySegment:
        """转换为 GatewaySegment，用于写入 candidate.json"""
        return GatewaySegment(
            label=self.label,
            reason=self.reason,
            run_id=self.run_id,
            checked_at=self.checked_at,
            is_legal=self.is_legal,
            has_future=self.has_future,
            complexity_score=self.complexity_score,
            is_within_budget=self.is_within_budget,
            is_duplicate=self.is_duplicate,
            historical_report_id=self.historical_report_id,
            error_message=self.error_message,
            data_quality_label=self.data_quality_label,
            data_quality_score=self.data_quality_score,
            data_quality_reason=self.data_quality_reason,
        )


@dataclass
class Manifest:
    """
    清单文件结构 - 对应 manifest.json

    符合 Disk Contracts 要求：
    - 每个交付目录必须有 manifest.json
    - 包含必需字段：schema_version, library, artifact_type, producer, run_id, created_at, files, validation
    """
    schema_version: str
    library: str  # 静态库名称（如 gateway_pass_base）
    artifact_type: str  # 制品类型（如 GatewayCandidate, GatewayReport）
    candidate_id: str
    producer: str  # 生产者模块名称（如 gateway）
    run_id: str
    created_at: str  # ISO 8601 格式
    files: Dict[str, str]  # 文件映射，如 {"candidate": "candidate.json"}
    validation: Dict[str, Any]  # 验证信息，包含 status 和 errors

    # 可选字段
    factor_id: Optional[str] = None  # 仅当因子已生成时存在

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于 JSON 序列化"""
        result = {
            "schema_version": self.schema_version,
            "library": self.library,
            "artifact_type": self.artifact_type,
            "candidate_id": self.candidate_id,
            "producer": self.producer,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "files": self.files,
            "validation": self.validation,
        }
        if self.factor_id is not None:
            result["factor_id"] = self.factor_id
        return result

    @classmethod
    def for_gateway_output(
            cls,
            candidate_id: str,
            library: str,
            run_id: str,
            validation_status: ValidationStatus,
            errors: List[str] = None,
    ) -> "Manifest":
        """
        创建网关输出的 manifest.json

        Args:
            candidate_id: 候选因子 ID
            library: 输出库名称（gateway_pass_base / temp_base / anti_sample_base）
            run_id: 本次运行 ID
            validation_status: 验证状态（passed / failed）
            errors: 错误列表（可选）
        """
        from datetime import datetime, timezone

        return cls(
            schema_version="disk.v1",
            library=library,
            artifact_type="GatewayCandidate",
            candidate_id=candidate_id,
            producer="gateway",
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            files={"candidate": "candidate.json"},
            validation={
                "status": validation_status.value,
                "errors": errors or [],
            },
            factor_id=None,
        )

    @classmethod
    def for_cache_report(
            cls,
            candidate_hash: str,
            expr_hash: str,
            run_id: str,
            cache_key_version: str = "v1",
    ) -> "Manifest":
        """
        创建缓存报告的 manifest.json

        Args:
            candidate_hash: 候选因子的哈希值（用作目录名）
            expr_hash: 表达式的哈希值
            run_id: 本次运行 ID
            cache_key_version: 缓存键版本
        """
        from datetime import datetime, timezone

        return cls(
            schema_version="disk.v1",
            library="gateway_report_cache",
            artifact_type="GatewayReport",
            candidate_id=candidate_hash,  # 缓存目录使用 hash 作为标识
            producer="gateway",
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            files={"report": "gateway_report.md"},
            validation={"status": "passed", "errors": []},
            factor_id=None,
        )


# 在文件末尾添加，不修改原有类

class DataQualityLabel(str, Enum):
    PASS = "Pass"
    TEMP = "Temp"


@dataclass
class DataQualitySegment:
    label: DataQualityLabel
    reason: str
    run_id: str
    checked_at: str

    distribution_drift_score: Optional[float] = None
    ks_p_value: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    liquidity_anomaly_score: Optional[float] = None
    spread_spike_count: Optional[int] = None
    clock_jitter_count: Optional[int] = None
    timestamp_out_of_order: Optional[int] = None
    overall_quality_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "Label": self.label.value,
            "Reason": self.reason,
            "run_id": self.run_id,
            "checked_at": self.checked_at,
        }
        if self.distribution_drift_score is not None:
            result["distribution_drift_score"] = self.distribution_drift_score
        if self.ks_p_value is not None:
            result["ks_p_value"] = self.ks_p_value
        if self.wasserstein_distance is not None:
            result["wasserstein_distance"] = self.wasserstein_distance
        if self.liquidity_anomaly_score is not None:
            result["liquidity_anomaly_score"] = self.liquidity_anomaly_score
        if self.spread_spike_count is not None:
            result["spread_spike_count"] = self.spread_spike_count
        if self.clock_jitter_count is not None:
            result["clock_jitter_count"] = self.clock_jitter_count
        if self.timestamp_out_of_order is not None:
            result["timestamp_out_of_order"] = self.timestamp_out_of_order
        if self.overall_quality_score is not None:
            result["overall_quality_score"] = self.overall_quality_score
        return result