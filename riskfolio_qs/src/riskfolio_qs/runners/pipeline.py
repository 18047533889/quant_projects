"""
优化管线：v0.2 配置驱动的一键组合优化主流程。

Pipeline 串联以下步骤：
1. 输入校验（SchemaValidator）
2. 信号平滑（SignalSmoother）
3. 优化器路由（OptimizerRouter）——按 optimizer_name 或 scenario 查找映射
4. 参数合并（ParameterStore）——冻结参数 + runtime 覆盖 + 白名单校验
5. 依赖校验（find_missing_inputs）——强缺失 fail_fast / degrade_to_rule
6. 组合优化（PortfolioOptimizer）——cvxpy 凸求解或 rule backend
7. 输出校验（SchemaValidator）
8. 可选：落盘 resolved_mapping.yaml / resolved_params.yaml

关联文档：
- riskfolio_qs_v0.2_优化器映射规范.md
- riskfolio_qs_v0.2_冻结参数规范.md
- riskfolio_qs_v0.2_优化器与参数总表.md
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..adapters.output_adapter import OutputAdapter
from ..core.contracts import InputBundle, OutputBundle
from ..core.validator import SchemaValidator
from ..constraints.constraint_builder import ConstraintBuilder
from ..optimizers.optimizer_router import OptimizerRouter
from ..optimizers.parameter_store import ParameterStore
from ..optimizers.portfolio_optimizer import PortfolioOptimizer
from ..smoothers.signal_smoother import SignalSmoother


@dataclass(slots=True)
class OptimizationPipeline:
    """配置驱动的组合优化管线。

    使用 YAML 映射和冻结参数配置，按 optimizer_name 或 scenario 路由
    到对应的优化器，执行完整的"输入→求解→输出"流程。

    Attributes:
        smoother: 信号平滑器（可注入）
        output_adapter: 输出适配器（可注入）
        validator: Schema 校验器（可注入）
        router: 优化器路由器（自动从 YAML 初始化）
        parameter_store: 冻结参数仓库（自动从 YAML 初始化）
        optimizer: 组合优化器
    """
    smoother: SignalSmoother | None = None
    output_adapter: OutputAdapter | None = None
    validator: SchemaValidator | None = None
    mapping_path: Path | None = None
    parameter_path: Path | None = None
    router: OptimizerRouter = field(init=False, repr=False)
    parameter_store: ParameterStore = field(init=False, repr=False)
    optimizer: PortfolioOptimizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化管线的默认组件。"""
        self.smoother = self.smoother or SignalSmoother()
        self.output_adapter = self.output_adapter or OutputAdapter()
        self.validator = self.validator or SchemaValidator()

        # 从 configs/optimizer/ 下的 YAML 加载映射和参数
        self.router = OptimizerRouter(mapping_path=self.mapping_path)
        self.parameter_store = ParameterStore(config_path=self.parameter_path)

        self.optimizer = PortfolioOptimizer(
            smoother=self.smoother,
            output_adapter=self.output_adapter,
            constraint_builder=ConstraintBuilder(),
        )

    def run(
        self,
        bundle: InputBundle,
        optimizer_name: str | None = None,
        scenario: str | None = None,
        runtime_overrides: dict[str, Any] | None = None,
        data_version_hash: str = "",
        output_dir: str | None = None,
    ) -> OutputBundle:
        """执行一次完整的组合优化。

        流程：
        1. 校验输入数据包
        2. 信号平滑
        3. 按 optimizer_name 或 scenario 解析 BenchmarkSpec
        4. 检查强/弱依赖输入缺失
        5. 强缺失时按 fallback_policy 降级或 fail_fast
        6. 合并冻结参数 + runtime 覆盖
        7. 执行优化求解
        8. 校验输出
        9. 可选：落盘 resolved 配置用于审计

        Args:
            bundle: 输入数据包
            optimizer_name: 显式指定优化器名称（优先级高于 scenario）
            scenario: 场景名（index_enhancement / conservative_index_enhancement /
                      absolute_return / fallback）。None 时根据 bundle.metadata
                      .alpha_is_absolute_return 选择 absolute_return 或 index_enhancement。
            runtime_overrides: 运行期参数覆盖（仅白名单内参数可覆盖）
            data_version_hash: 输入数据版本哈希，写入审计元数据
            output_dir: 可选，指定 resolved 配置落盘目录

        Returns:
            OutputBundle：包含目标持仓、交易明细、摘要和审计元数据。
        """
        # 先解析路由，使输入依赖能按具体优化器判定。
        resolved_scenario = scenario or (
            "absolute_return" if bundle.metadata.alpha_is_absolute_return else "index_enhancement"
        )
        routing_reason = ""
        if optimizer_name:
            spec = self.router.resolve(optimizer_name)
            requested_optimizer_name = spec.name
            routing_reason = f"explicit optimizer_name={optimizer_name}"
        elif resolved_scenario in {
            "index_enhancement",
            "conservative_index_enhancement",
        }:
            conservative = resolved_scenario == "conservative_index_enhancement"
            if (
                bundle.F is not None
                and bundle.factor_cov is not None
                and bundle.specific_var is not None
            ):
                selected = (
                    "minvar_enhance_barra_precomputed"
                    if conservative
                    else "meanvar_enhance_barra_precomputed"
                )
                routing_reason = "precomputed Barra B/F/D inputs are available"
            elif (
                bundle.F is not None
                and bundle.F_ret is not None
                and bundle.F_spec is not None
            ):
                selected = (
                    "minvar_enhance_index"
                    if conservative
                    else "meanvar_enhance_index"
                )
                routing_reason = "raw Barra exposure/factor/specific returns are available"
            elif bundle.market is not None:
                selected = (
                    "minvar_enhance_hist"
                    if conservative
                    else "meanvar_enhance_hist"
                )
                routing_reason = "historical market data is the best available risk input"
            else:
                raise ValueError(
                    "No usable index-enhancement risk input: require either "
                    "F+factor_cov+specific_var, F+F_ret+F_spec, or market"
                )
            spec = self.router.resolve(selected)
            requested_optimizer_name = f"scenario:{resolved_scenario}"
        else:
            spec = self.router.resolve_default(resolved_scenario)
            requested_optimizer_name = f"scenario:{resolved_scenario}"
            routing_reason = f"default route for scenario={resolved_scenario}"

        # 步骤 1：输入校验
        self.validator.validate_input_bundle(bundle)

        # 步骤 2：信号平滑
        smoothed_alpha, diagnostics = self.smoother.transform(bundle.alpha)

        # 步骤 4：依赖输入缺失检查
        strong_missing, weak_missing = self.validator.find_missing_inputs(bundle, spec)
        fallback_used = False
        fallback_reason = ""

        # 步骤 5：强缺失处理 —— fail_fast 或 degrade_to_rule
        if strong_missing:
            if spec.fallback_policy in {"degrade_to_rule", "degrade_to_optimizer"}:
                fallback_reason = f"missing strong inputs for {spec.name}: {strong_missing}"
                if spec.fallback_policy == "degrade_to_optimizer":
                    if not spec.fallback_optimizer:
                        raise ValueError(
                            f"Optimizer {spec.name} has no configured fallback_optimizer"
                        )
                    spec = self.router.resolve(spec.fallback_optimizer)
                else:
                    spec = self.router.resolve_default("fallback")
                fallback_used = True
                strong_missing, weak_missing = self.validator.find_missing_inputs(bundle, spec)
                if strong_missing:
                    raise ValueError(
                        f"Fallback optimizer {spec.name} still has missing strong inputs: "
                        f"{strong_missing}; original reason: {fallback_reason}"
                    )
            else:
                raise ValueError(f"Missing strong inputs for optimizer {spec.name}: {strong_missing}")

        # 步骤 6：合并参数（冻结 + 覆盖）
        resolved_params, applied_overrides = self.parameter_store.resolve(spec.name, runtime_overrides)

        # 可选：落盘 resolved 配置用于审计回放
        if output_dir:
            self._dump_resolved_configs(
                output_dir=output_dir,
                spec=spec,
                resolved_params=resolved_params,
                runtime_overrides=applied_overrides,
            )

        # 步骤 7：执行优化
        output = self.optimizer.optimize(
            bundle=bundle,
            benchmark_spec=spec,
            smoothed_alpha=smoothed_alpha,
            resolved_params=resolved_params,
            parameter_version=self.parameter_store.parameter_version,
            mapping_version=self.router.mapping_version,
            fallback_used=fallback_used,
            requested_optimizer_name=requested_optimizer_name,
            fallback_reason=fallback_reason,
            routing_reason=routing_reason,
            weak_missing_inputs=weak_missing,
            applied_overrides=applied_overrides,
            data_version_hash=data_version_hash,
            diagnostics=diagnostics,
        )

        # 步骤 8：输出校验
        self.validator.validate_output_bundle(output)
        return output

    def _dump_resolved_configs(
        self,
        output_dir: str,
        spec: Any,
        resolved_params: dict[str, Any],
        runtime_overrides: dict[str, Any],
    ) -> None:
        """将本次运行的 resolved 配置落盘到 YAML 文件。

        生成两个文件：
        - resolved_mapping.yaml：本次使用的优化器映射快照
        - resolved_params.yaml：本次使用的冻结参数 + 覆盖快照

        Args:
            output_dir: 输出目录路径
            spec: BenchmarkSpec 对象
            resolved_params: 合并后的最终参数字典
            runtime_overrides: 本次应用的覆盖参数
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        mapping_payload = {
            "mapping_version": self.router.mapping_version,
            "optimizer": asdict(spec),
        }
        params_payload = {
            "parameter_version": self.parameter_store.parameter_version,
            "optimizer_name": spec.name,
            "resolved_params": resolved_params,
            "runtime_overrides": runtime_overrides,
        }

        with (out / "resolved_mapping.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(mapping_payload, f, sort_keys=False, allow_unicode=True)
        with (out / "resolved_params.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(params_payload, f, sort_keys=False, allow_unicode=True)
