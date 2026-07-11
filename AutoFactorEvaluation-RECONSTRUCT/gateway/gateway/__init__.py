"""
网关模块 - 因子评估流水线的第一道关口

Gateway 模块负责对候选因子进行快速合法性校验、未来函数拦截、
复杂度预算控制和语义去重，并决定因子的后续路由。

路由决策:
    Pass (通过)       → tier1/gateway_pass_base/
    Duplicated (重复)  → tier1/gateway_duplicated_base/
    Rejected (拒绝)    → tier4/anti_sample_base/gateway/

检查流程:
    Schema 校验 → 算子白名单验证 → 未来函数拦截 → 复杂度评估
    → 语义去重 → [可选]数据质量检测

入口函数:
    GatewayCore.process(candidate_dict) -> GatewayResult
        处理单个候选因子，返回路由结果。

配置:
    GatewayConfig(config_dir=None)  — 从 configs/ 目录加载 YAML+JSON 配置。
        支持环境变量覆盖路径和阈值。

使用示例:
    from gateway import GatewayCore, GatewayConfig

    config = GatewayConfig()
    gateway = GatewayCore(config, market_data_path="...")
    result = gateway.process(candidate_dict)
    if result.label == "Pass":
        print("Candidate passed all checks")
"""

from .scripts import (
    GatewayCore, GatewayConfig, KafkaConfig, FeaturesConfig, PathsConfig,
    GatewayLabel, DataQualityLabel, Candidate, GatewayResult,
    GatewaySegment, Manifest, ValidationStatus,
    StaticValidator, SchemaValidator, OperatorValidator, ValidationResult,
    FutureFunctionScanner, FutureScanResult,
    ComplexityEvaluator, ComplexityReport,
    SemanticDeduplicator, DeduplicationResult, HashDeduplicator,
    DataQualityRunner, QualityReport,
    IOManager,
    KafkaProducer, KafkaMessageBuilder, SendResult, SendStatus,
    GatewayError, SchemaValidationError, OperatorNotAllowedError,
    DataTypeError, FutureFunctionError, ComplexityError, ConfigError,
)

__version__ = "1.0.0"

__all__ = [
    "GatewayCore",
    "GatewayConfig", "KafkaConfig", "FeaturesConfig", "PathsConfig",
    "Candidate", "GatewayResult", "GatewayLabel", "GatewaySegment",
    "DataQualityLabel", "Manifest", "ValidationStatus",
    "StaticValidator", "SchemaValidator", "OperatorValidator", "ValidationResult",
    "FutureFunctionScanner", "FutureScanResult",
    "ComplexityEvaluator", "ComplexityReport",
    "SemanticDeduplicator", "DeduplicationResult", "HashDeduplicator",
    "DataQualityRunner", "QualityReport",
    "IOManager",
    "KafkaProducer", "KafkaMessageBuilder", "SendResult", "SendStatus",
    "GatewayError", "SchemaValidationError", "OperatorNotAllowedError",
    "DataTypeError", "FutureFunctionError", "ComplexityError", "ConfigError",
]
