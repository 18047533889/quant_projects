"""
配置管理模块 - 加载 YAML 和 JSON 配置文件

负责:
- 加载 gateway_config.yaml 主配置
- 加载算子白名单、未来函数黑名单、复杂度权重表
- 提供统一的配置访问接口
- 支持从环境变量覆盖配置（可选）
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
from dataclasses import dataclass, field

import yaml


class ConfigError(Exception):
    """配置错误异常"""
    pass


@dataclass
class PathsConfig:
    """磁盘路径配置"""
    candidate_pool: str
    gateway_pass_base: str
    duplicated_base: str
    anti_sample_base: str
    report_cache: str

    def validate(self) -> List[str]:
        """验证路径配置，返回缺失的路径列表（不自动创建）"""
        missing = []
        for name, path in self.__dict__.items():
            if not path:
                missing.append(f"{name} is empty")
        return missing


@dataclass
class KafkaConfig:
    """Kafka 配置"""
    enabled: bool
    bootstrap_servers: str
    topic: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KafkaConfig":
        return cls(
            enabled=data.get("enabled", False),
            bootstrap_servers=data.get("bootstrap_servers", "localhost:9092"),
            topic=data.get("topic", "factor_candidates_compute"),
        )


@dataclass
class FeaturesConfig:
    """功能开关配置"""
    enable_complexity_check: bool = True
    enable_semantic_dedup: bool = True
    enable_future_scan: bool = True
    enable_kafka: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturesConfig":
        return cls(
            enable_complexity_check=data.get("enable_complexity_check", True),
            enable_semantic_dedup=data.get("enable_semantic_dedup", True),
            enable_future_scan=data.get("enable_future_scan", True),
            enable_kafka=data.get("enable_kafka", False),
        )


class GatewayConfig:
    """
    网关配置类 - 加载和管理所有配置

    使用示例:
        cfg = GatewayConfig()
        print(cfg.paths.candidate_pool)
        print(cfg.operator_whitelist)  # Set of allowed operators
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        初始化配置

        Args:
            config_dir: 配置文件目录，默认为共享配置目录
                all_configs/auto_factor_evaluation/
        """
        if config_dir is None:
            # 默认配置目录：项目级共享配置
            _project = Path(__file__).resolve().parents[3]
            _candidate = _project / "all_configs" / "auto_factor_evaluation"
            if not _candidate.exists():
                _candidate = _project.parent / "all_configs" / "auto_factor_evaluation"
            self.config_dir = _candidate
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise ConfigError(f"Config directory not found: {self.config_dir}")

        # 加载主配置
        self._load_main_config()

        # 加载子配置文件
        self._load_operator_whitelist()
        self._load_future_blacklist()
        self._load_complexity_weights()

        # 应用环境变量覆盖
        self._apply_env_overrides()

    def _load_main_config(self) -> None:
        """加载主配置文件 gateway_config.yaml"""
        main_config_path = self.config_dir / "gateway_config.yaml"
        if not main_config_path.exists():
            raise ConfigError(f"Main config file not found: {main_config_path}")

        try:
            with open(main_config_path, "r", encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML config: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read config file: {e}")

        # 解析路径配置
        paths_data = self._raw_config.get("paths", {})
        self.paths = PathsConfig(
            candidate_pool=paths_data["candidate_pool"],
            gateway_pass_base=paths_data["gateway_pass_base"],
            duplicated_base=paths_data["duplicated_base"],
            anti_sample_base=paths_data["anti_sample_base"],
            report_cache=paths_data["report_cache"],
        )

        # 验证路径配置
        missing = self.paths.validate()
        if missing:
            raise ConfigError(f"Missing required path configurations: {missing}")

        # 解析市场数据路径（绝对路径直用，相对路径基于项目根）
        _project_root = Path(__file__).resolve().parents[3]
        _mkt = self._raw_config["market_data_path"]
        _mkt_path = Path(_mkt)
        self.market_data_path = str(_mkt_path) if _mkt_path.is_absolute() else str((_project_root / _mkt).resolve())

        # 解析阈值配置
        self.max_complexity = self._raw_config["max_complexity"]
        self.semantic_similarity_threshold = self._raw_config.get(
            "semantic_similarity_threshold", 0.95
        )
        self.max_ast_depth = self._raw_config["max_ast_depth"]

        # 解析 Kafka 配置
        kafka_data = self._raw_config.get("kafka", {})
        self.kafka = KafkaConfig.from_dict(kafka_data)

        # 解析功能开关
        features_data = self._raw_config.get("features", {})
        self.features = FeaturesConfig.from_dict(features_data)

        # 记录子配置文件路径（用于加载）
        self._operator_whitelist_file = self._raw_config["operator_whitelist_file"]
        self._future_blacklist_file = self._raw_config["future_blacklist_file"]
        self._complexity_weights_file = self._raw_config["complexity_weights_file"]

    def _load_operator_whitelist(self) -> None:
        """加载算子白名单"""
        whitelist_path = self.config_dir / self._operator_whitelist_file
        if not whitelist_path.exists():
            raise ConfigError(f"Operator whitelist file not found: {whitelist_path}")

        try:
            with open(whitelist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse operator whitelist JSON: {e}")

        operators = data.get("operators", [])
        if not operators:
            raise ConfigError("Operator whitelist is empty")

        self.operator_whitelist = set(operators)

        # 字段类型规则（用于算子-字段兼容性校验）
        self.field_type_rules = data.get("field_type_rules", {})
        # 算子字段限制（哪些算子只能用于哪些字段类型）
        self.operator_restrictions = data.get("operator_restrictions", {})
        # 构建算子→允许字段类型的快速映射
        self._operator_field_map: dict[str, set[str]] = {}
        for group_name, group in self.operator_restrictions.items():
            allowed = set(group.get("allowed_types", []))
            for op in group.get("operators", []):
                self._operator_field_map[op] = allowed

    def _load_future_blacklist(self) -> None:
        """加载未来函数黑名单"""
        blacklist_path = self.config_dir / self._future_blacklist_file
        if not blacklist_path.exists():
            raise ConfigError(f"Future blacklist file not found: {blacklist_path}")

        try:
            with open(blacklist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse future blacklist JSON: {e}")

        self.future_blacklist = set(data.get("functions", []))
        self.future_regex_patterns = data.get("regex_patterns", [])
        self.time_offset_keywords = data.get("time_offset_keywords", [])

    def _load_complexity_weights(self) -> None:
        """加载复杂度权重表"""
        weights_path = self.config_dir / self._complexity_weights_file
        if not weights_path.exists():
            raise ConfigError(f"Complexity weights file not found: {weights_path}")

        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse complexity weights JSON: {e}")

        self.complexity_weights = data.get("base_weights", {})

        # 验证：白名单中的算子都应该有对应的权重
        missing_weights = self.operator_whitelist - set(self.complexity_weights.keys())
        if missing_weights:
            # 警告但不报错，使用默认权重 1.0
            for op in missing_weights:
                self.complexity_weights[op] = 1.0

        # 深度惩罚配置
        depth_penalty_data = data.get("depth_penalty", {})
        self.depth_penalty_enabled = depth_penalty_data.get("enabled", True)
        self.depth_penalty_per_level = depth_penalty_data.get("penalty_per_level", 0.5)

        # 额外规则
        self.complexity_additional_rules = data.get("additional_rules", {})

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖（可选）"""
        # 支持通过环境变量覆盖路径
        if os.environ.get("GATEWAY_CANDIDATE_POOL"):
            self.paths.candidate_pool = os.environ["GATEWAY_CANDIDATE_POOL"]
        if os.environ.get("GATEWAY_PASS_BASE"):
            self.paths.gateway_pass_base = os.environ["GATEWAY_PASS_BASE"]
        if os.environ.get("GATEWAY_DUPLICATED_BASE"):
            self.paths.duplicated_base = os.environ["GATEWAY_DUPLICATED_BASE"]
        if os.environ.get("GATEWAY_ANTI_SAMPLE_BASE"):
            self.paths.anti_sample_base = os.environ["GATEWAY_ANTI_SAMPLE_BASE"]
        if os.environ.get("GATEWAY_REPORT_CACHE"):
            self.paths.report_cache = os.environ["GATEWAY_REPORT_CACHE"]

        # 覆盖市场数据路径
        if os.environ.get("MARKET_DATA_PATH"):
            self.market_data_path = os.environ["MARKET_DATA_PATH"]

        # 覆盖阈值
        if os.environ.get("GATEWAY_MAX_COMPLEXITY"):
            self.max_complexity = float(os.environ["GATEWAY_MAX_COMPLEXITY"])

        # 覆盖 Kafka 配置
        if os.environ.get("GATEWAY_KAFKA_ENABLED"):
            self.kafka.enabled = os.environ["GATEWAY_KAFKA_ENABLED"].lower() == "true"
        if os.environ.get("GATEWAY_KAFKA_BOOTSTRAP"):
            self.kafka.bootstrap_servers = os.environ["GATEWAY_KAFKA_BOOTSTRAP"]
        if os.environ.get("GATEWAY_KAFKA_TOPIC"):
            self.kafka.topic = os.environ["GATEWAY_KAFKA_TOPIC"]

    def get_operator_weight(self, operator: str) -> float:
        """
        获取算子的复杂度权重

        Args:
            operator: 算子名称（如 "MA", "STD"）

        Returns:
            权重值，如果不在表中返回默认值 1.0
        """
        return self.complexity_weights.get(operator, 1.0)

    def is_operator_allowed(self, operator: str) -> bool:
        """
        检查算子是否在白名单中

        Args:
            operator: 算子名称

        Returns:
            True 表示允许使用
        """
        return operator in self.operator_whitelist

    def is_future_function(self, function_name: str) -> bool:
        """
        检查函数是否在未来函数黑名单中

        Args:
            function_name: 函数名称

        Returns:
            True 表示是未来函数
        """
        return function_name in self.future_blacklist

    def get_allowed_operators(self) -> Set[str]:
        """获取所有允许的算子集合"""
        return self.operator_whitelist.copy()

    def is_operator_allowed_for_field(self, operator: str, field_type: str) -> bool:
        """检查算子是否允许用于指定字段类型。

        若算子无字段限制（不在 operator_restrictions 中），视为通用算子，允许所有类型。
        """
        allowed = self._operator_field_map.get(operator)
        if allowed is None:
            return True  # 无限制的通用算子
        return field_type in allowed

    def get_future_functions(self) -> Set[str]:
        """获取未来函数黑名单集合"""
        return self.future_blacklist.copy()

    def get_complexity_weights(self) -> Dict[str, float]:
        """获取复杂度权重表副本"""
        return self.complexity_weights.copy()

    def get_path(self, name: str) -> str:
        """
        获取指定路径配置

        Args:
            name: 路径名称（candidate_pool, gateway_pass_base, duplicated_base, anti_sample_base, report_cache）

        Returns:
            路径字符串
        """
        return getattr(self.paths, name, "")

    def print_summary(self) -> None:
        """打印配置摘要（用于调试）"""
        print("=" * 50)
        print("Gateway Configuration Summary")
        print("=" * 50)
        print("\n[Paths]")
        for name, value in self.paths.__dict__.items():
            print(f"  {name}: {value}")

        print("\n[Thresholds]")
        print(f"  max_complexity: {self.max_complexity}")
        print(f"  semantic_similarity_threshold: {self.semantic_similarity_threshold}")
        print(f"  max_ast_depth: {self.max_ast_depth}")

        print("\n[Kafka]")
        print(f"  enabled: {self.kafka.enabled}")
        print(f"  bootstrap_servers: {self.kafka.bootstrap_servers}")
        print(f"  topic: {self.kafka.topic}")

        print("\n[Features]")
        for name, value in self.features.__dict__.items():
            print(f"  {name}: {value}")

        print(f"\n[Operator Whitelist]")
        print(f"  count: {len(self.operator_whitelist)}")
        print(f"  operators: {', '.join(sorted(self.operator_whitelist)[:10])}...")

        print(f"\n[Future Blacklist]")
        print(f"  functions: {len(self.future_blacklist)}")
        print(f"  regex patterns: {len(self.future_regex_patterns)}")

        print(f"\n[Complexity Weights]")
        print(f"  operators with weights: {len(self.complexity_weights)}")
        print("=" * 50)


# 便捷函数：创建默认配置实例
_default_config: Optional[GatewayConfig] = None


def get_config(config_dir: Optional[Path] = None) -> GatewayConfig:
    """
    获取配置实例（单例模式，可选使用）

    Args:
        config_dir: 配置文件目录

    Returns:
        GatewayConfig 实例
    """
    global _default_config
    if config_dir is not None:
        return GatewayConfig(config_dir)
    if _default_config is None:
        _default_config = GatewayConfig()
    return _default_config