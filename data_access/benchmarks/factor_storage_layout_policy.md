# Factor 值存储布局策略（Storage Layout Policy）

> 对应平台总规范 §21（COS 数据湖布局）、§54（性能与规模测试）、§55（"每个日期
> 一个 factor 文件 → object explosion → bucket/partition"）。
> 实证数据：`data_access/benchmarks/benchmark_factor_storage_layout.py`。

## 结论（一句话）

**禁止**永久使用 "100k 因子 × 每天 × 每因子一个小 parquet" 布局。生产布局 =
**日频分片 delta（factor_bucket 1024 分片 + year/month 分片）+ 定期 compaction**，
对象数被 1024 × 12 × 2 ≈ 24.6k/年 封顶，与因子数无关。

---

## 1. 为什么禁止 per-factor-per-day 小对象

A 股 ~5000 标的/日、~250 交易日/年。若每因子每天一个小 parquet：

```
对象数 = factor_count × 250 天/年
```

| 因子数 | 对象数/年 | 评估翻倍后 | 预算判定 |
|---|---|---|---|
| 10k | 2.5M | 5M | OVER_BUDGET |
| 50k | 12.5M | 25M | OVER_BUDGET |
| 100k | **25M** | **50M** | **OVER_BUDGET** |

对象数预算（`factor_storage_layout.py`）：硬上限 `1_000_000`，预警线 `100_000`。
100k 因子布局 A 达 25M，**超预算 25 倍**。COS/S3 单 prefix 对象数超过 ~1M 后
LIST 延迟线性恶化、DuckDB 规划需打开海量 footer、compaction 需遍历海量小对象。

## 2. 生产布局（布局 B）

```text
factor_values/
  market=ashare/
    freq=1d/
      factor_bucket=0012/            # shard_key(factor_id, 1024)
        year=2026/
          month=01/
            delta.parquet            # 日频 delta（增量写入）
            compacted.parquet       # compaction 产物（raw_factor_delta → raw_factor_compacted）
```

- **分片**：`shard_key(factor_id, 1024)` 用 blake2b 前 4 字节取模，确定性、同一
  factor 永远落同一分片 → 增量写入与 compaction 都按分片聚合，不跨分片扫描。
- **分区**：`partition_key(asset, region, freq, year, month)` 生成
  `ashare/cn/1d/year=2026/month=01`。
- **对象数**：`bucket_count × months × 2`（delta + compacted）= 1024 × 12 × 2 =
  **24,576/年**，与因子数无关。

## 3. Benchmark 实证（采样外推）

采样 500 标的/日、25 天、2000 因子，用 `LocalObjectStore` 提供真实对象数语义，
再按采样比例外推到全规模（不物化真实 25M 对象）。

| 布局 | 因子数 | 采样对象数 | 采样 PUT(ms) | 采样 LIST(ms) | 采样读(ms) | 外推对象数 | 预算 |
|---|---|---|---|---|---|---|---|
| A | 10k | 50,000 | 6,285 | 2,526 | 2,979 | 2.5M | OVER |
| A | 50k | 50,000 | 7,161 | 2,646 | 2,946 | 12.5M | OVER |
| A | 100k | 50,000 | 7,434 | 2,892 | 2,969 | **25M** | **OVER** |
| B | 10k | 1,754 | 351 | 2,622 | 142 | 24,576 | UNDER |
| B | 50k | 1,754 | 308 | 2,626 | 141 | 24,576 | UNDER |
| B | 100k | 1,754 | 327 | 2,644 | 140 | **24,576** | **UNDER** |

要点：
- 布局 B 的**对象数不随因子数增长**（分片封顶），100k 下仍 24,576，远低于预算。
- 布局 B 的 PUT 成本比 A 低 ~20×（分片聚合，每 (shard, month) 一个对象）。
- 布局 B 的读延迟比 A 低 ~20×（读 1,754 个对象 vs 50,000 个）。
- LIST 延迟两者相近（都遍历 factor_values/ prefix），但 A 在真实 25M 对象下会
  线性恶化，B 恒为 ~24.6k。

## 4. 参数校准

- **bucket_count = 1024**：分片数取 2 的幂，blake2b 取模均匀；1024 分片 × 12 月
  × 2 = 24.6k 对象/年，远低于预算，且每分片每月一个 parquet 落在 8MB-128MB
  目标区间（`target_parquet_size(5000, 100) = 2MB`，compaction 后合并多日更接近
  目标）。
- **compaction cadence**：`compaction_decision(delta_object_count)` —— 当某
  (shard, month) 的 delta 对象数达到阈值即 compaction。原子发布：先写
  `compacted.parquet` + old→new lineage，**grace period（默认 7 天）后**才删除
  delta 旧对象，绝不在 compaction 完成前删除。
- **evaluation_daily vs evaluation_bundle 分离**（规范 §14-15）：评估结果按
  `evaluation_daily`（单日）与 `evaluation_bundle`（批量）分离，避免评估对象
  与因子值对象混在同一 prefix 造成 LIST 放大。

## 5. 落地

- 纯函数助手：`data_access/benchmarks/factor_storage_layout.py`
  - `shard_key(factor_id, bucket_count)` → `[0, bucket_count)`
  - `partition_key(asset, region, freq, year, month)`
  - `factor_bucket_dir(factor_id, bucket_count)`
  - `check_object_count_budget(factor_count, layout)` → 预算判定
  - `compaction_decision(...)` → delta→compacted 决策
  - `target_parquet_size(...)` → 目标 parquet 字节
- 真实 writer 应 import 这些纯函数复用，不重复实现分片/分区逻辑。
- 测试：`data_access/tests/unit/test_factor_storage_layout.py`（确定性、分片范围、
  预算判定）。
