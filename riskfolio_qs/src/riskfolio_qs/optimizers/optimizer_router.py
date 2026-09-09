"""
优化器路由器：BenchmarkRouter 的上层封装。

提供更简洁的接口：
- resolve(optimizer_name)：按名称查找 BenchmarkSpec
- resolve_default(scenario)：按场景查找默认优化器
- mapping_version：读取当前映射版本号
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .benchmark_router import BenchmarkRouter, BenchmarkSpec


@dataclass(slots=True)
class OptimizerRouter:
    """优化器路由器：封装 BenchmarkRouter，提供场景路由和版本查询。"""

    mapping_path: Path | None = None
    router: BenchmarkRouter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """自动初始化 BenchmarkRouter（使用默认 YAML 路径）。"""
        self.router = BenchmarkRouter(mapping_path=self.mapping_path)

    def resolve(self, optimizer_name: str) -> BenchmarkSpec:
        """按优化器名称精确查找。

        Args:
            optimizer_name: 优化器名称（如 meanvar_enhance_index）

        Returns:
            对应的 BenchmarkSpec
        """
        return self.router.get(optimizer_name)

    def resolve_default(self, scenario: str = "index_enhancement") -> BenchmarkSpec:
        """按场景名查找默认优化器。

        Args:
            scenario: 场景名（index_enhancement / absolute_return / fallback 等）

        Returns:
            对应场景的默认 BenchmarkSpec
        """
        return self.router.get_default_by_scenario(scenario)

    def default_benchmark_names(self) -> list[str]:
        """列出所有已注册的优化器名称。"""
        return [spec.name for spec in self.router.list_specs()]

    @property
    def mapping_version(self) -> str:
        """当前加载的映射 YAML 版本号。"""
        return self.router.mapping_version
