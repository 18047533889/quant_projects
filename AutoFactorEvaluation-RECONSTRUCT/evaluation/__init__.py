"""Evaluation 评估总模块：因子评估一条龙服务。

架构:
    tier1/purification_pure_factor_base/
        ↓ [Evaluation Worker Pool]
    tier0/evaluation_temp/
        ↓ [Evaluation Router]
    tier2/2_fix_base/          (route_recommendation=tier2_incubator)
    tier2/2x_llm_mutation_base/  (route_recommendation=tier2x_factory)
    tier3/3a_core_production_base/  (route_recommendation=tier3a_core)
    tier3/3b_satellite_production_base/  (route_recommendation=tier3b_satellite)
    tier3/3c_feature_matierial_base/  (route_recommendation=tier3c_feature)
    tier3/3d_operation_storage_base/  (route_recommendation=tier3d_optimized_reserve)
    tier4/anti_sample_base/    (route_recommendation=tier4_archive)

子模块与上下游顺序
------------------
1. ``evaluation.timeseries``（4.1）：因子绩效时序计算（IC/分层/多空/换手/覆盖率）。
2. ``evaluation.indicator``（4.2）：消费绩效时序，计算统计指标矩阵、评分卡与评估摘要。
3. ``evaluation.label``（4.3）：依据评估摘要贴标，并给出入库推荐路由信息。

公共入口
--------
通过 worker 模块的 process_factor_dir() 函数入口调用，后续路由由 EvaluationRouter 完成。
    - pure_factor_dir: 因子资产目录（含因子表达式/配置/历史因子暴露值三件套）。
    - market_data_path: 历史行情数据路径。
    - universe_path: 市场基本信息（行业、可交易约束、市值）。
    其余为运行 ID、各子模块配置与主 horizon 等可选项。

出参（EvaluationPipelineResult）：
    - timeseries: 4.1 绩效时序产物索引。
    - indicator_by_horizon: 各 horizon 的统计指标与评分卡。
    - summary_scorecard / evaluation_summary: 主 horizon 的评分卡与评估摘要。
    - tag_package: 因子标签包。
    - admission_decision / route_record / lifecycle_event: 入库推荐路由信息
      （不含真正入库落地，由更高级别管道统一执行）。

各子模块入口亦在对应包的 ``__init__`` 中独立暴露：
    - evaluation.timeseries.run_timeseries_evaluation
    - evaluation.indicator.run_stage4_2
    - evaluation.label.run_label_pipeline
"""

from evaluation.pipeline import EvaluationPipelineResult, run_evaluation_pipeline
from evaluation.scripts.worker import process_factor_dir
from evaluation.scripts.router import EvaluationRouter
from evaluation.config import EvaluationConfig

__all__ = [
    "EvaluationPipelineResult", "run_evaluation_pipeline",
    "process_factor_dir", "EvaluationRouter", "EvaluationConfig",
]
