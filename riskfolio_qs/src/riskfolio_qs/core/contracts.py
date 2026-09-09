"""
核心数据契约：定义系统中所有模块之间传递的不可变数据结构。

四个核心 dataclass：
- RunMetadata：单次优化运行的审计元数据（v0.2 版本审计字段）
- OptimizationContext：优化上下文（场景、基准等信息）
- InputBundle：输入数据包（alpha、行情、Barra 外采因子等）
- OutputBundle：输出数据包（目标权重、交易表、摘要、元数据）

关联文档：riskfolio_qs_v0.2_数据链路初版.md（输入输出字段契约）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(slots=True)
class RunMetadata:
    """单次优化运行的完整审计元数据。

    对齐 v0.2 文档要求的审计字段：
    optimizer_name、objective_id、risk_mode、parameter_version、
    mapping_version、data_version_hash 等。
    """
    # ---- v0.2 审计核心字段 ----
    optimizer_name: str = ""          # 使用的优化器名称
    objective_id: str = ""            # 目标函数模板标识
    hard_constraint_set: str = ""     # 硬约束集合标识
    soft_constraint_set: str = ""     # 软约束集合标识
    risk_mode: str = ""               # 风险口径：barra_factor / historical_cov / none
    parameter_version: str = ""       # 冻结参数版本号
    mapping_version: str = ""         # 优化器映射版本号
    data_version_hash: str = ""       # 输入数据版本哈希

    # ---- 兼容旧字段 ----
    model_id: str = ""                # 模型标识（现已等同于 optimizer_name）
    alpha_version: str = ""           # alpha 数据版本
    constraint_profile: str = ""      # 约束模板标识
    solver: str = ""                  # 求解器名称
    solve_status: str = ""            # 求解状态：success / failed / infeasible
    solve_time_ms: float = 0.0        # 求解耗时（毫秒）
    fallback_used: bool = False       # 是否使用了降级策略
    extras: Dict[str, Any] = field(default_factory=dict)  # 扩展字段（含 backend、overrides 等）


@dataclass(slots=True)
class OptimizationContext:
    """优化运行上下文信息。

    用于标识本次优化的场景类型和基准信息，
    由 InputAdapter 在构建 InputBundle 时填充。
    """
    alpha_is_absolute_return: bool = False  # True 表示绝对收益场景，False 表示相对收益（指增）场景
    benchmark_name: str = ""                # 基准名称
    benchmark_family: str = ""              # 基准家族
    rebalance_id: str = ""                  # 调仓批次标识
    calendar_name: str = "XNYS"             # exchange_calendars 的交易所日历名
    decision_time: str = "close"            # 组合决策时点
    execution_time: str = "next_open"       # 目标权重执行时点
    market_data_lag_periods: int = 0         # 决策时可用行情的滞后期数
    exposure_data_lag_periods: int = 0       # 决策时可用暴露的滞后期数
    alpha_input_type: str = "score"          # score / expected_return
    alpha_horizon_days: int = 1              # alpha 预测期限
    market_input_type: str = "price"          # price / return
    risk_covariance_units: str = "daily_variance"  # daily/horizon/annual variance
    risk_horizon_days: int = 1               # horizon_variance 对应的自然日/交易日数
    annualization_factor: float = 252.0       # 年化使用的交易期数
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputBundle:
    """优化输入数据包：包含一次组合优化所需的所有输入。

    核心字段：
    - alpha：alpha 信号矩阵，行=日期(DatetimeIndex)，列=资产代码
    - market：行情价格矩阵，同结构。TopN 可为 None，历史协方差路径必填

    可选字段（含 Barra 外采因子，v0.2 新增）：
    - F：因子暴露，MultiIndex(date, asset) x factor，外采输入
    - G：行业暴露矩阵，外采输入
    - F_mcap：市值暴露矩阵，外采输入
    - F_ret：因子收益矩阵，外采输入（用于派生 Σ_f）
    - F_spec：特质收益矩阵，外采输入（用于派生 D）
    - factor_cov：预计算因子协方差，MultiIndex(date, factor_i) x factor_j
    - specific_var：预计算特异方差，date x asset
    - industry：行业分类标签
    - style：风格因子暴露
    - benchmark：基准权重矩阵
    - prev_positions：上期持仓权重矩阵
    - tradable：可交易标志矩阵
    - metadata：优化上下文
    """
    # 必填
    alpha: pd.DataFrame
    market: Optional[pd.DataFrame] = None

    # Barra 外采三输入（v0.2）
    F: Optional[pd.DataFrame] = None          # MultiIndex(date, asset) x factor
    G: Optional[pd.DataFrame] = None          # 行业暴露
    F_mcap: Optional[pd.DataFrame] = None     # 市值暴露
    F_ret: Optional[pd.DataFrame] = None      # 因子收益
    F_spec: Optional[pd.DataFrame] = None     # 特质收益
    factor_cov: Optional[pd.DataFrame] = None # MultiIndex(date, factor_i) x factor_j
    specific_var: Optional[pd.DataFrame] = None  # date x asset，方差而非波动率

    # 其他可选输入
    industry: Optional[pd.DataFrame] = None
    style: Optional[pd.DataFrame] = None
    benchmark: Optional[pd.DataFrame] = None
    prev_positions: Optional[pd.DataFrame] = None
    tradable: Optional[pd.DataFrame] = None
    risk_covered: Optional[pd.DataFrame] = None  # point-in-time risk-model coverage
    linear_cost_bps: Optional[pd.DataFrame] = None  # 单边线性成本，bps
    impact_cost: Optional[pd.DataFrame] = None      # 二次冲击成本系数
    metadata: OptimizationContext = field(default_factory=OptimizationContext)


@dataclass(slots=True)
class OutputBundle:
    """优化输出数据包：一次组合优化的完整产出。

    四个 DataFrame 必须使用 DatetimeIndex（或 trades 的多级索引首级为日期）。
    """
    target_positions: pd.DataFrame  # 目标持仓权重，行=日期，列=资产
    trades: pd.DataFrame            # 交易明细，MultiIndex(date, asset) 或 DatetimeIndex
    summary: pd.DataFrame           # 组合摘要，含 gross_exposure / net_exposure / turnover
    metadata: pd.DataFrame          # 审计元数据，一行一条运行记录
