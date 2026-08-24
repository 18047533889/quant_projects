# 美股财务 PIT Serving Layout（#8）

## 问题

A 股财务 E1 文件按 `PubDate` 命名，PIT 也是 `PubDate`，按 knowledge time 做
路径裁剪天然很好。美股 E2 完全不同：

| 项 | A 股 E1 | 美股 E2 |
|---|---|---|
| 物理文件 | `{PubDate}.parquet` | `{period_end}.parquet` |
| PIT（可知时间） | `PubDate` | 文件内的 `filing_date` |
| 路径裁剪 | 按 PubDate 直接裁剪 | **无法**按 filing_date 裁剪 |

查询「2024~2026 的 filing_date」时，按 period_end 命名会让路径裁剪访问到
大量不相干文件。

## 两层方案

### 1. 二级 PITEventIndex（已落地）

`data_access.read.pit_event_index` 维护
`(ticker, filing_date, period_end, timeframe, file_path)`：

- `store.pit_event_index(dataset, ...)`：扫描一次，落盘 `_pit_event_index.parquet`
  sidecar（记录每行源文件，用 DuckDB `glob()` 枚举 + pyarrow 逐文件读列）。
- `store.prune_pit_paths(dataset, filing_range=, timeframe=, tickers=)`：用索引
  按 filing_date 精准返回文件子集。

用法：

```python
store.pit_event_index("us_stock_balance")                      # 有 E2 契约，自动取列
store.prune_pit_paths("us_stock_balance",
                      filing_range=("2024-01-01", "2026-12-31"),
                      timeframe="annual")
# -> [".../us_stock_balance/2023-12-31.parquet", ".../2024-12-31.parquet", ...]
```

无契约数据集可显式传列名：

```python
store.pit_event_index("us_fin", filing_column="filing_date",
                      period_column="period_end", timeframe_column="timeframe")
```

### 2. 物化 PIT 优化湖（建议，未落地）

对研究热路径进一步物化一份 **filing 分区、按 (ticker, filing_date, period_end)
排序**的优化湖：

```
us_fin_pit_serving/
  filing_year=2024/
    filing_month=03/
      <ticker>_<filing_date>_<period_end>.parquet
```

- COS 原始布局保持不动（可追溯性）；
- 研究热路径走优化湖（filing 分区裁剪 + 排序便于 ASOF）；
- 由 `compute_and_write` / `publish_from_staging` 维护，`missing_partition_semantics`
  控制缺失行为。

**原则**：Source Layer 只读原始 COS；Serving Layer 为性能改变
partition/sort/rowgroup；Semantic View 负责 PIT/单位/市场适配。
