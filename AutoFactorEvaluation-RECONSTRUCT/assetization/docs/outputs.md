# Assetization 模块产物结构

## 一、输出目录结构

```
{raw_factor_base_dir}/{factor_id}/
├── afv.json               # 元数据（含 Assetization 段）
├── data/
│   ├── {YYYY-MM-DD}.parquet  # 每日因子值切片
│   ├── {YYYY-MM-DD}.parquet
│   └── ...
└── manifest.json          # 上游候选池元信息（可选）
```

目录结构说明：
- 使用扁平 `{factor_id}/` 结构，不再嵌套 `{domain_root}/{frequency_bucket}/`。
- 因子值数据按交易日分片存储在 `data/` 子目录下，每个文件为单日截面。
- 元数据统一使用 `afv.json`（Alpha Factor Verification），替代旧版 `candidate.json`。

## 二、data/{YYYY-MM-DD}.parquet 数据结构

| 列名 | 类型 | 说明 |
|------|------|------|
| `TradeDate` | datetime64 | 交易日（每个文件内相同） |
| `Symbol` | string | 资产代码 |
| `factor_id` | string | 全局唯一因子 ID |
| `factor_value` | float64 | 因子值 |

## 三、afv.json Assetization 段

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "cand_20260528_a1b2c3d4",
  "campaign_id": "campaign_alpha_001",
  "formula": "ts_mean(close, 10) / ts_std(close, 10)",
  "gateway_version": "1.0.0",
  "Gateway": {
    "Label": "Pass",
    "Reason": "",
    "run_id": "gateway_20260528_083000_a1b2c3d4",
    "checked_at": "2026-05-28T08:30:00.123456+00:00",
    "recommendation": "pass",
    "checks": {
      "schema_valid": true,
      "operator_allowed": true,
      "has_future": false,
      "complexity_score": 3.5,
      "is_within_budget": true,
      "is_duplicate": false
    }
  },
  "factor_id": "equity_1d_cross_sectional_a3f2b1c0",
  "Assetization": {
    "Label": "Materialized",
    "status": "passed",
    "factor_id": "equity_1d_cross_sectional_a3f2b1c0",
    "run_id": "assetization_20260528_083000_a3f2b1c0",
    "materialized_at": "2026-05-28T08:30:00Z",
    "coordinates": {
      "signal_structure": "cross_sectional",
      "asset_class": "equity",
      "frequency_bucket": "1d",
      "domain_root": "us_stock",
      "domain": "price_volume"
    },
    "market_period": {
      "start": "2026-01-01",
      "end": "2026-05-28"
    },
    "row_count": 509099,
    "formula": "ts_mean(close, 10) / ts_std(close, 10)",
    "candidate_id": "cand_20260528_a1b2c3d4"
  }
}
```

### Assetization 段字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `Label` | string | 是 | `Materialized` |
| `status` | string | 是 | `passed` |
| `factor_id` | string | 是 | 全局唯一因子 ID |
| `run_id` | string | 是 | 运行标识 |
| `materialized_at` | string | 是 | ISO 8601 资产化时间 |
| `coordinates` | dict | 是 | 库坐标 |
| `market_period` | dict | 是 | 行情数据时间区间 `{start, end}` |
| `row_count` | int | 是 | 输出因子数据行数 |
| `formula` | string | 是 | DSL 表达式 |
| `candidate_id` | string | 是 | 上游候选因子 ID |

## 四、manifest.json 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | string | `disk.v1` |
| `candidate_id` | string | 候选 ID |
| `formula` | string | DSL 表达式 |
| `domain_root` | string | 域根 |
| `frequency_bucket` | string | 频率桶 |
| `...` | ... | 上游候选池保留的其他元信息 |
