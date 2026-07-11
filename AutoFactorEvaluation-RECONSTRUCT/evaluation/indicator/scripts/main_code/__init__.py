"""阶段 4-2 A 轨时序指标计算模块的公共 API。

推荐入口函数：
    run_stage4_2(input_dir: str | Path, output_dir: str | Path, log_dir: str | Path | None = None) -> dict

输入目录格式：
    - performance_series_bundle.json: 必需文件，记录逻辑序列名到数据文件的映射。
    - metric_method_config.json: 可选文件，用于覆盖 DEFAULT_METHOD_CONFIG。

输出目录格式：
    - metrics_matrix.parquet/json
    - summary_scorecard.json
    - dev4_evaluation_summary.json
    - metrics_package.json
    - metric_validation_report.json
    - metric_manifest.json
"""

from .calculator import MetricCalculationResult, run_metric_calculation
from .run import run_stage4_2

__all__ = ["MetricCalculationResult", "run_metric_calculation", "run_stage4_2"]
