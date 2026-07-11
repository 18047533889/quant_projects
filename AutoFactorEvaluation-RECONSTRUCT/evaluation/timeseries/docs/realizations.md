# Timeseries 工程实现

## 核心流程

```
PureFactor 输入 (data.parquet)
+ 行情数据 (market_data)
+ Universe (行业/可交易/市值)
    → PIT 时间对齐 (pit_align)
    → 前向收益计算 (forward_returns)
    → IC 计算 (rank_ic, pearson_ic)
    → 分层回测 (quantile_backtest)
    → 多空组合 (long_short)
    → 换手率 (turnover)
    → 覆盖率 (coverage)
    → 产物落盘 (parquet + manifest)
```

## 关键实现

- **PIT 对齐**: 确保因子数据与行情数据的时间点对齐，避免前视偏差
- **多 horizon 支持**: 同时计算多个持有期的绩效
- **Validation 缓冲**: 所有校验事件统一收集，不阻塞计算流程
