# Purification 模块产物结构

## 一、输出目录结构

```
{pure_factor_dir}/{factor_id}/
├── afv.json               # 元数据（含 Purification 段）
├── data/
│   ├── {YYYY-MM-DD}.parquet  # 每日纯化后因子值切片
│   ├── {YYYY-MM-DD}.parquet
│   └── ...
└── manifest.json          # 完整元数据
```

## 二、data/{YYYY-MM-DD}.parquet 数据结构

与 RawFactor 相同结构，但 `factor_value` 列经过纯化处理：

| 列名 | 类型 | 说明 |
|------|------|------|
| `TradeDate` | datetime64 | 交易日 |
| `Symbol` | string | 资产代码 |
| `factor_id` | string | 因子 ID |
| `factor_value` | float64 | 纯化后的因子值 |

> **注意**：内部处理时列名映射为 `datetime` / `asset`，输出时恢复为 `TradeDate` / `Symbol`。

## 三、afv.json Purification 段

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "cand_20260528_a1b2c3d4",
  "factor_id": "equity_1d_cross_sectional_a3f2b1c0",
  "formula": "ts_mean(close, 10) / ts_std(close, 10)",
  "Gateway": { ... },
  "Assetization": { ... },
  "Purification": {
    "Label": "Pure",
    "run_id": "purification_20260528_083000",
    "purified_at": "2026-05-28T08:30:00Z",
    "input_raw_factor_uri": "database/tier1/assetization_raw_factor_base/equity_1d_cross_sectional_a3f2b1c0",
    "orthogonal_basis_uris": [
      "database/tier3/3a_core_production_base/risk_exposures",
      "database/tier3/3b_satellite_production_base/risk_exposures"
    ],
    "steps": [
      "imputation",
      "winsorization",
      "orthogonalization"
    ],
    "row_count": 510286,
    "industry_standard": "zjw"
  }
}
```

### Purification 段字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `Label` | string | 是 | `Pure` |
| `run_id` | string | 是 | 运行 ID |
| `purified_at` | string | 是 | ISO 8601 纯化时间 |
| `input_raw_factor_uri` | string | 是 | 上游 RawFactor 完整路径 |
| `orthogonal_basis_uris` | list | 是 | 正交化依赖路径列表 |
| `steps` | list | 是 | 执行的纯化步骤 |
| `row_count` | int | 是 | 纯化后因子数据行数 |
| `industry_standard` | string | 是 | 行业分类口径（如 `zjw`） |

## 四、manifest.json 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | string | `disk.v1` |
| `library` | string | `pure_factor_base` |
| `artifact_type` | string | `PureFactor` |
| `producer` | string | `purification` |
| `factor_id` | string | 因子 ID |
| `run_id` | string | 运行 ID |
| `created_at` | string | 创建时间 |
| `source.raw_factor_uri` | string | 上游 RawFactor 路径 |
| `source.orthogonal_basis_uris` | list | 正交化依赖路径 |
| `row_count` | int | 因子数据行数 |
| `validation.status` | string | `passed` |
