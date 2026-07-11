# Timeseries 子模块功能概述（4.1）

## 一、模块目的

计算因子绩效时序序列，包括 IC、分层回测、多空收益、换手率、覆盖率等。

## 二、核心函数

### run_timeseries_evaluation

**签名**: `run_timeseries_evaluation(*, eval_run_id, pure_factor_dir, market_data_path, universe_path, ...) → TimeseriesRunResult`

**入参**:
- `eval_run_id`: 运行唯一 ID
- `pure_factor_dir`: PureFactor 目录（含三件套）
- `market_data_path`: 历史行情路径
- `universe_path`: 市场基本信息路径

**产物**:
| 逻辑名 | 说明 |
|--------|------|
| `daily_rank_ic` | 日度 Rank IC 序列 |
| `daily_ic` | 日度 Pearson IC 序列 |
| `quantile_backtest` | 分层回测收益面板 |
| `top_minus_bottom` | 多空收益序列 |
| `long_short_returns` | 多空收益序列（带成本） |
| `turnover_series` | 换手率序列 |
| `coverage_series` | 覆盖率序列 |
