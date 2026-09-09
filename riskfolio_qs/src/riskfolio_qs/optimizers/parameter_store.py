"""
参数仓库：从 YAML 加载冻结参数，提供参数合并与白名单校验。

核心职责：
1. 加载 parameters.v0.2.0.yaml 中的全局/风险/求解器/优化器参数
2. resolve() 按优先级合并：global → risk → solver → optimizer → runtime_override
3. 校验 runtime 覆盖是否在白名单内，值是否满足 validation_rules

参数优先级（从低到高）：
    Global 默认 < Risk 默认 < Solver 默认 < Optimizer 默认 < Runtime 覆盖

关联文档：riskfolio_qs_v0.2_冻结参数规范.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sysconfig
from typing import Any, Dict

import yaml


@dataclass(slots=True)
class ParameterStore:
    """参数仓库：加载并合并多层冻结参数，提供白名单覆盖校验。

    Attributes:
        config_path: 参数 YAML 文件路径，None 则使用默认路径
        parameter_version: 参数版本号（从 YAML 读取）
        overrides_whitelist: 允许运行时覆盖的参数名集合
        validation_rules: 参数值校验规则（gt/ge/lt/le）
    """
    config_path: Path | None = None
    parameter_version: str = ""
    overrides_whitelist: set[str] = field(default_factory=set, repr=False)
    validation_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    _global_defaults: Dict[str, Any] = field(default_factory=dict, repr=False)
    _risk_defaults: Dict[str, Any] = field(default_factory=dict, repr=False)
    _solver_defaults: Dict[str, Any] = field(default_factory=dict, repr=False)
    _optimizer_defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """加载 YAML 并初始化各层参数。"""
        path = self.config_path or self._default_path()
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        self.parameter_version = str(payload.get("parameter_version", "v0.0.0"))
        self._global_defaults = dict(payload.get("global", {}))
        self._risk_defaults = dict(payload.get("risk", {}))
        self._solver_defaults = dict(payload.get("solver", {}))
        self._optimizer_defaults = dict(payload.get("optimizers", {}))
        self.overrides_whitelist = set(payload.get("overrides_whitelist", []))
        self.validation_rules = dict(payload.get("validation_rules", {}))

    def resolve(
        self,
        optimizer_name: str,
        runtime_overrides: Dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """按优先级合并各层参数并应用 runtime 覆盖。

        合并顺序：global → risk → solver → optimizer → runtime override
        （后面的覆盖前面的）。

        Args:
            optimizer_name: 优化器名称（必须在 YAML 中定义）
            runtime_overrides: 运行期覆盖参数（只允许白名单内的 key）

        Returns:
            (resolved_params, applied_overrides)：
            - resolved_params：合并后的最终参数字典
            - applied_overrides：本次实际应用的覆盖参数

        Raises:
            KeyError: optimizer_name 不在参数 YAML 中
            ValueError: 覆盖的 key 不在白名单中或值校验失败
        """
        if optimizer_name not in self._optimizer_defaults:
            raise KeyError(f"Unknown optimizer for parameters: {optimizer_name}")

        # 按优先级逐层合并参数
        resolved: Dict[str, Any] = {}
        resolved.update(self._global_defaults)                    # 第 1 层：全局默认
        resolved.update(self._risk_defaults)                      # 第 2 层：风险默认
        resolved.update(self._solver_defaults)                    # 第 3 层：求解器默认
        resolved.update(self._optimizer_defaults[optimizer_name]) # 第 4 层：优化器默认

        # 冻结配置本身也必须通过同一套校验，不能只校验
        # runtime override 而信任 YAML 中的非法默认值。
        for key, value in resolved.items():
            self._validate_override_value(key, value)

        # 第 5 层：运行时覆盖（仅白名单 + 规则校验）
        overrides = dict(runtime_overrides or {})
        self._validate_override_keys(overrides)
        for key, value in overrides.items():
            self._validate_override_value(key, value)
            resolved[key] = value

        return resolved, overrides

    def _validate_override_keys(self, overrides: Dict[str, Any]) -> None:
        """校验覆盖参数的 key 是否在白名单内。

        Raises:
            ValueError: 有不在白名单中的 key。
        """
        for key in overrides:
            if key not in self.overrides_whitelist:
                raise ValueError(f"Override key not allowed: {key}")

    def _validate_override_value(self, key: str, value: Any) -> None:
        """校验覆盖参数的值是否满足 validation_rules。

        支持的规则：gt(大于)、ge(大于等于)、lt(小于)、le(小于等于)。

        Raises:
            ValueError: 值不满足规则。
        """
        if value is None:
            return
        rule = self.validation_rules.get(key)
        if not rule:
            return
        if "gt" in rule and not (value > rule["gt"]):
            raise ValueError(f"Override {key} must be > {rule['gt']}, got {value}")
        if "ge" in rule and not (value >= rule["ge"]):
            raise ValueError(f"Override {key} must be >= {rule['ge']}, got {value}")
        if "lt" in rule and not (value < rule["lt"]):
            raise ValueError(f"Override {key} must be < {rule['lt']}, got {value}")
        if "le" in rule and not (value <= rule["le"]):
            raise ValueError(f"Override {key} must be <= {rule['le']}, got {value}")
        if "in" in rule and value not in rule["in"]:
            raise ValueError(f"Override {key} must be one of {rule['in']}, got {value}")

    @staticmethod
    def _default_path() -> Path:
        """获取默认参数 YAML 路径。

        优先读取随 wheel 分发的包内配置，并兼容旧版 data-files 安装路径。
        """
        package_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "optimizer"
            / "parameters.v0.2.0.yaml"
        )
        if package_path.exists():
            return package_path
        installed_path = (
            Path(sysconfig.get_path("data"))
            / "riskfolio_qs"
            / "configs"
            / "optimizer"
            / "parameters.v0.2.0.yaml"
        )
        if installed_path.exists():
            return installed_path
        raise FileNotFoundError(
            "Cannot find parameters.v0.2.0.yaml; pass OptimizationPipeline(parameter_path=...)"
        )
