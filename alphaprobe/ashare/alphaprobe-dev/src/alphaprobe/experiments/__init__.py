"""alphaprobe.experiments —— 算法实验 / ablation harness（plan.md Task 24）。

只做「配置矩阵装配 + 指标聚合」，不做真实 campaign：
- ``ablation.AblationConfig``：10 个累积配置工厂（legacy reference →
  + next-version Survival Memory），装配后产出可注入真实链的
  ``AblationAssembly``（各特性模块 config 实例）。
- ``ablation.AblationMetrics``（即 ``compute_metrics``）：从一次 run 的
  attempt 序列聚合 EliteYield / CostAdjustedEliteYield / duplicate rate /
  coverage / cost-per-elite（只聚合已评估事实，空 run 中性，不炸）。
- ``ablation.AblationRunner``：配置 → 开关装配 → run 回调注入 → 指标聚合；
  dry-run（OFFLINE_TEST）语义下不真跑 LLM/FE/QE campaign。

本包不 import torch / faiss / openai / factor_engine / data_access /
quant_evaluator（模块 import 即安全；survival.attribution 的 sklearn 为
try-import 可选）。
"""

from __future__ import annotations

from alphaprobe.experiments.ablation import (
    ABLATION_CONFIG_NAMES,
    DEFAULT_ELITE_YIELD_DENOMINATOR,
    ELITE_YIELD_DENOMINATOR,
    FEATURE_ORDER,
    METRIC_KEYS,
    AblationAssembly,
    AblationAttempt,
    AblationConfig,
    AblationReport,
    AblationRunResult,
    AblationRunner,
    FeatureSwitch,
    ablation_config_matrix,
    all_ablation_configs,
    build_ablation_config,
    compute_metrics,
    configs_diff,
    make_feature_switch,
)

__all__ = [
    "ABLATION_CONFIG_NAMES",
    "DEFAULT_ELITE_YIELD_DENOMINATOR",
    "ELITE_YIELD_DENOMINATOR",
    "FEATURE_ORDER",
    "METRIC_KEYS",
    "AblationAssembly",
    "AblationAttempt",
    "AblationConfig",
    "AblationReport",
    "AblationRunResult",
    "AblationRunner",
    "FeatureSwitch",
    "ablation_config_matrix",
    "all_ablation_configs",
    "build_ablation_config",
    "compute_metrics",
    "configs_diff",
    "make_feature_switch",
]
