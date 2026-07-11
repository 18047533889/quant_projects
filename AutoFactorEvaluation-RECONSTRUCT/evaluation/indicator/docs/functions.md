# Indicator 子模块功能概述（4.2）

## 一、模块目的

消费 Timeseries 产出的绩效时序数据，计算统计指标矩阵、评分卡和 Dev4 评估摘要。

## 二、核心函数

### run_stage4_2

**签名**: `run_stage4_2(input_dir, output_dir, log_dir=None) → dict`

**入参**:
- `input_dir`: 含 `performance_series_bundle.json` 的目录
- `output_dir`: 指标产出目录
- `log_dir`: 可选日志目录

**出参**:
| 字段 | 说明 |
|------|------|
| `summary_scorecard` | 评分卡（rank_ic_mean, long_short_sharpe 等） |
| `metrics_matrix` | 指标矩阵 |
| `dev4_evaluation_summary` | Dev4 评估摘要 |
