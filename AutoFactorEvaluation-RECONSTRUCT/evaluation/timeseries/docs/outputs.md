# Timeseries 产物结构

## 输出目录

```
{out_dir}/
├── manifest.json
├── validation_report.json
├── daily_rank_ic.parquet
├── daily_ic.parquet
├── quantile_backtest.parquet
├── top_minus_bottom.parquet
├── long_short_returns.parquet
├── turnover_series.parquet
├── coverage_series.parquet
└── logs/
    └── {eval_run_id}.log
```

## data.parquet 公共列

所有时序 parquet 均含 `datetime`, `factor_id`, `horizon` 列及对应指标列。
