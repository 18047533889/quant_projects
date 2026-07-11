# Indicator 工程实现

## 核心流程

```
performance_series_bundle.json
    → 加载各序列 parquet
    → 计算指标矩阵（IC 统计、分位收益统计、多空统计等）
    → 生成评分卡（加权综合评分）
    → 生成 Dev4 评估摘要
    → 写入 output_dir
```

## 关键设计

- 单 horizon 契约：每次只处理一个 horizon 的时序数据
- `net_return` 桥接：从 `gross_return` 补 `net_return = gross_return`（分位层不计成本）
- 日志追加模式：多次运行的日志保留在同一文件中
