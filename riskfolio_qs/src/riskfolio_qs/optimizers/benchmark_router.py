"""
基准路由器：从 YAML 映射文件加载优化器配置，提供按名称/场景的查找能力。

核心数据结构：
- BenchmarkSpec：单个优化器的完整映射信息（名称、目标、约束集、风险口径、后端等）
- BenchmarkRouter：管理所有 BenchmarkSpec 并提供查找/路由能力

数据来源：configs/optimizer/mapping.v0.2.0.yaml

关联文档：riskfolio_qs_v0.2_优化器映射规范.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sysconfig
from typing import Any, Dict, List

import yaml


@dataclass(slots=True)
class BenchmarkSpec:
    """单个优化器的完整映射描述。

    每个字段直接对应 mapping YAML 中的优化器条目。

    Attributes:
        name: 优化器名称（如 meanvar_enhance_index）
        status: 状态（active / draft / deprecated）
        objective_id: 目标函数模板标识（如 OBJ_MEANVAR_ACTIVE）
        hard_constraint_set: 硬约束集合标识（如 HC_INDEX_ENHANCE_CORE）
        soft_constraint_set: 软约束集合标识（如 SC_TC_TURNOVER）
        risk_mode: 风险口径（barra_factor / historical_cov / none）
        backend: 求解后端（riskfolio_backend / rule_backend）
        solver_default: 默认求解器（如 CLARABEL）
        fallback_policy: 降级策略（fail_fast / degrade_to_rule）
        required_inputs: 强/弱依赖输入字段列表
        params: 原始 YAML 参数字典（透传）
    """
    name: str
    status: str
    objective_id: str
    hard_constraint_set: str
    soft_constraint_set: str
    risk_mode: str
    backend: str
    solver_default: str
    fallback_policy: str
    fallback_optimizer: str = ""
    required_inputs: Dict[str, List[str]] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)


class BenchmarkRouter:
    """基准路由器：管理所有优化器映射并提供查找能力。

    在初始化时从 YAML 加载映射，后续可通过 get() 按名称查找、
    通过 get_default_by_scenario() 按场景查找默认优化器。
    """

    def __init__(self, mapping_path: Path | None = None) -> None:
        """初始化路由器并加载映射 YAML。

        Args:
            mapping_path: 映射 YAML 文件路径，None 则使用默认路径。
        """
        path = mapping_path or self._default_path()
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        self.mapping_version = str(payload.get("mapping_version", "v0.0.0"))
        self.default_optimizer_set = list(payload.get("default_optimizer_set", []))
        self.default_routing = dict(payload.get("default_routing", {}))
        self._specs = self._build_specs(payload.get("optimizers", {}))

    def list_specs(self) -> List[BenchmarkSpec]:
        """列出所有已加载的 BenchmarkSpec。"""
        return list(self._specs)

    def get(self, benchmark_name: str) -> BenchmarkSpec:
        """按名称精确查找 BenchmarkSpec。

        Args:
            benchmark_name: 优化器名称

        Returns:
            对应的 BenchmarkSpec

        Raises:
            KeyError: 未找到指定名称的优化器
        """
        for spec in self._specs:
            if spec.name == benchmark_name:
                return spec
        raise KeyError(f"Unknown benchmark: {benchmark_name}")

    def get_default_by_scenario(self, scenario: str = "index_enhancement") -> BenchmarkSpec:
        """按场景名查找默认优化器。

        先查 default_routing 映射，若未配置则取 default_optimizer_set 的第一个。

        Args:
            scenario: 场景名（如 index_enhancement / absolute_return / fallback）

        Returns:
            对应场景的默认 BenchmarkSpec

        Raises:
            KeyError: 无默认优化器配置
        """
        if scenario in self.default_routing:
            return self.get(self.default_routing[scenario])
        if self.default_optimizer_set:
            return self.get(self.default_optimizer_set[0])
        raise KeyError("No default optimizer configured in mapping")

    @staticmethod
    def _build_specs(raw_optimizers: Dict[str, Dict[str, Any]]) -> List[BenchmarkSpec]:
        """从 YAML 原始字典构建 BenchmarkSpec 列表。

        对每个优化器条目提取 required_inputs 中的 strong/weak 列表，
        并将完整原始字典存入 params。

        Args:
            raw_optimizers: YAML 中 optimizers 节点的原始字典

        Returns:
            BenchmarkSpec 列表
        """
        specs: List[BenchmarkSpec] = []
        for name, raw in raw_optimizers.items():
            required_inputs = raw.get("required_inputs", {})
            spec = BenchmarkSpec(
                name=name,
                status=str(raw.get("status", "draft")),
                objective_id=str(raw.get("objective_id", "")),
                hard_constraint_set=str(raw.get("hard_constraint_set", "")),
                soft_constraint_set=str(raw.get("soft_constraint_set", "")),
                risk_mode=str(raw.get("risk_mode", "none")),
                backend=str(raw.get("backend", "rule_backend")),
                solver_default=str(raw.get("solver_default", "none")),
                fallback_policy=str(raw.get("fallback_policy", "fail_fast")),
                fallback_optimizer=str(raw.get("fallback_optimizer", "")),
                required_inputs={
                    "strong": list(required_inputs.get("strong", [])),
                    "weak": list(required_inputs.get("weak", [])),
                },
                params=dict(raw),
            )
            specs.append(spec)
        return specs

    @staticmethod
    def _default_path() -> Path:
        """获取默认映射 YAML 路径。

        优先读取随 wheel 分发的包内配置，并兼容旧版 data-files 安装路径。
        """
        package_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "optimizer"
            / "mapping.v0.2.0.yaml"
        )
        if package_path.exists():
            return package_path
        installed_path = (
            Path(sysconfig.get_path("data"))
            / "riskfolio_qs"
            / "configs"
            / "optimizer"
            / "mapping.v0.2.0.yaml"
        )
        if installed_path.exists():
            return installed_path
        raise FileNotFoundError(
            "Cannot find mapping.v0.2.0.yaml; pass OptimizationPipeline(mapping_path=...)"
        )
