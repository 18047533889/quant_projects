"""
约束模板：定义各优化器对应的默认约束参数集合。

ConstraintTemplate 为 dataclass，包含各约束维度的默认取值。
get_benchmark_constraint_template 根据优化器名称返回对应的模板实例。

注意：此模块是 v0.1 的约束管理方式，v0.2 已改为 YAML 参数驱动。
当前保留用于 rule_backend 优化器的约束参数回退。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ConstraintTemplate:
    """约束参数模板：定义一组组合约束的默认参数。

    各字段含义：
    - budget: 预算约束（权重之和），默认 1.0（满仓）
    - max_single_name_weight: 个股最大权重，默认 0.05
    - turnover_cap: 换手上限（单边），默认 0.20
    - turnover_definition: 换手率计算公式
    - topn_n_long: 做多股票数
    - topn_n_short: 做空股票数
    - long_weight_per_name: 每只做多个股权重
    - short_weight_per_name: 每只做空个股权重
    - gross_exposure_cap: 总敞口上限
    - net_exposure_target: 净敞口目标
    """
    budget: float = 1.0
    max_single_name_weight: float = 0.05
    turnover_cap: float = 0.20
    turnover_definition: str = "0.5 * sum(|w_t - w_{t-1}|)"
    topn_n_long: int = 50
    topn_n_short: int = 10
    long_weight_per_name: float = 0.02
    short_weight_per_name: float = -0.05
    gross_exposure_cap: float = 1.0
    net_exposure_target: float = 0.0


def get_benchmark_constraint_template(benchmark_name: str) -> ConstraintTemplate:
    """根据优化器名称返回对应的约束模板。

    Args:
        benchmark_name: 优化器名称（如 "topn_long_short_equal_weight"）

    Returns:
        对应的 ConstraintTemplate 实例。
    """
    benchmark_name = benchmark_name.lower()
    # 多空等权优化器：覆盖默认参数
    if benchmark_name == "topn_long_short_equal_weight":
        return ConstraintTemplate(
            topn_n_long=10,
            topn_n_short=10,
            long_weight_per_name=0.05,
            short_weight_per_name=-0.05,
            gross_exposure_cap=1.0,
            net_exposure_target=0.0,
        )
    # 其他优化器使用默认模板
    return ConstraintTemplate()
