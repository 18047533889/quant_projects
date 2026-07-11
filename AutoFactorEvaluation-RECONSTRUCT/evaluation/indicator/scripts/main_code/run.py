"""阶段 4-2 A 轨时序指标计算模块入口。

所属模块:
    evaluation/indicator

对应文档:
    docs/FID_stage4_2_indicator_metrics.md

文件职责:
    暴露 `run_stage4_2` 作为 pipeline 和本地脚本调用的唯一推荐入口。
    该入口只负责接收输入、输出和日志路径，并把完整计算交给
    `main_code.calculator.run_metric_calculation`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .calculator import MetricCalculationResult, run_metric_calculation


def run_stage4_2(input_dir: str | Path, output_dir: str | Path, log_dir: str | Path | None = None) -> dict[str, Any]:
    """阶段 4-2 的项目级主入口函数。

    这是本模块对外推荐使用的唯一 run 函数，负责把阶段 4-1 的绩效序列输入目录
    交给指标计算器，并把阶段 4-2 的所有标准产物写到输出目录。

    入参：
        input_dir: 阶段 4-1 输出目录，必须包含 `performance_series_bundle.json`。
            如果目录内有 `metric_method_config.json`，会使用其中配置；否则使用默认配置。
        output_dir: 阶段 4-2 输出目录，会生成指标矩阵、评分卡、指标包、校验报告等文件。

    出参：
        返回 dict，字段包括：
        - `metrics_matrix`: 内存态指标矩阵，便于测试或后续处理。
        - `summary_scorecard`: 标准评分卡 JSON 结构。
        - `dev4_evaluation_summary`: 适配下游 Dev4/Yurui 的评估摘要。
        - `validation_events`: 输入与计算过程中的校验事件。
        - `output_dir`: 输出目录字符串。
    """

    return run_metric_calculation(Path(input_dir), Path(output_dir), Path(log_dir) if log_dir is not None else None)


__all__ = ["MetricCalculationResult", "run_stage4_2"]
