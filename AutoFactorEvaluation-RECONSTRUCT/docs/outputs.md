# AutoFactorEvaluation 产物结构

## 数据库目录结构

```
database/
├── tier1/                          # 一级数据（原始/中间层）
│   ├── gateway_pass_base/{candidate_id}/
│   │   ├── afv.json                # 含 Gateway 段标记
│   │   └── manifest.json
│   ├── gateway_duplicated_base/{candidate_id}/
│   │   ├── afv.json
│   │   └── manifest.json
│   ├── assetization_raw_factor_base/{factor_id}/
│   │   ├── afv.json                # 含 Assetization 段标记
│   │   ├── data/                   # 因子值（按日分片 parquet）
│   │   │   ├── 2016-01-04.parquet
│   │   │   ├── 2016-01-05.parquet
│   │   │   └── ...
│   │   └── manifest.json
│   └── purification_pure_factor_base/{factor_id}/
│       ├── afv.json                # 含 Purification 段标记
│       ├── data/                   # 纯化因子值（按日分片 parquet）
│       │   ├── 2016-01-04.parquet
│       │   ├── 2016-01-05.parquet
│       │   └── ...
│       └── manifest.json
├── tier2/
│   ├── 2_fix_base/{factor_id}/
│   └── 2x_llm_mutation_base/{factor_id}/
├── tier3/
│   ├── 3a_core_production_base/{factor_id}/
│   ├── 3b_satellite_production_base/{factor_id}/
│   ├── 3c_feature_matierial_base/{factor_id}/
│   └── 3d_operation_storage_base/{factor_id}/
├── tier4/
│   └── anti_sample_base/gateway/{candidate_id}/
│       └── afv.json
├── real/
│   ├── candidate_pool/             # 候选因子输入
│   ├── candidate_pool_record/      # 空池归档（因子全部抽离后）
│   └── market_data/                # 行情数据
├── cache/
│   ├── gateway_report_cache/
│   ├── market_data/                # 合并 parquet 缓存（约 388MB）
│   └── timeseries_forward_return/  # 前向收益率缓存
├── mock/                           # Mock 测试数据
└── log/                            # 日志
    ├── evaluation/
    │   ├── evaluation_pipeline.log
    │   ├── timeseries/
    │   ├── indicator/
    │   └── label/
    ├── assetization/
    ├── gateway/
    └── purification/
```

**关键变更（相对于旧版）**：

| 旧版 | 新版 | 原因 |
|------|------|------|
| `candidate.json` | `afv.json` | 统一命名，明确标识全链路元数据 |
| `data.parquet` （单一文件） | `data/{YYYY-MM-DD}.parquet` （每日分片） | 支持增量追加、按日查询 |
| `{domain}/{freq}/{factor_id}/` | `{factor_id}/` （扁平结构） | 简化路径，无需坐标编码 |
| 无 `market_period` 字段 | Assetization 段含 `market_period` | 记录因子覆盖的时间范围 |

---

## 各模块产物的标记字段

### afv.json 阶段标记

每个模块在 `afv.json` 中追加自己的阶段段落：

**Gateway 段**（追加在 Gateway 阶段后）：

```json
{
  "Gateway": {
    "Label": "Pass",
    "Reason": "",
    "run_id": "gateway_20260528_083000_a1b2",
    "checked_at": "2026-05-28T08:30:00Z",
    "recommendation": "pass",
    "checks": {
      "schema_valid": true,
      "operator_allowed": true,
      "has_future": false,
      "complexity_score": 3.5,
      "is_within_budget": true,
      "is_duplicate": false
    }
  }
}
```

**Assetization 段**（追加在 Assetization 阶段后）：

```json
{
  "Assetization": {
    "Label": "Materialized",
    "status": "passed",
    "factor_id": "equity_daily_cross_sectional_0b649ced",
    "run_id": "assetization_20260528_083000_a3f2",
    "materialized_at": "2026-05-28T08:30:00Z",
    "coordinates": {
      "signal_structure": "cross_sectional",
      "asset_class": "equity",
      "frequency_bucket": "daily",
      "domain_root": "us_stock",
      "domain": "price_volume"
    },
    "market_period": {
      "start": "2016-01-04",
      "end": "2026-06-15"
    },
    "row_count": 498669,
    "formula": "...",
    "candidate_id": "..."
  }
}
```

**Purification 段**（追加在 Purification 阶段后）：

```json
{
  "Purification": {
    "Label": "Pure",
    "run_id": "purification_20260528_083000",
    "purified_at": "2026-05-28T08:30:00Z",
    "steps": ["imputation", "winsorization", "orthogonalization"],
    "industry_standard": "zjw",
    "row_count": 499872
  }
}
```

---

## data/ 目录数据格式

所有模块输出的因子 parquet 均遵循以下 schema：

| 列名 | 类型 | 说明 | 出现阶段 |
|------|------|------|---------|
| `TradeDate` | datetime64 | 交易日 | Assetization+ |
| `Symbol` | string | 资产代码 | Assetization+ |
| `factor_id` | string | 全局唯一因子 ID | Assetization+ |
| `factor_value` | float64 | 因子值（原始/纯化/评估） | Assetization+ |

每文件为一个交易日的截面切片：

```
data/2016-01-04.parquet → 包含 2016-01-04 全部资产的因子值
data/2016-01-05.parquet → 包含 2016-01-05 全部资产的因子值
```

---

## Evaluation 产物目录结构

```
{work_dir}/
├── indicator_input_h{horizon}/     # 按 horizon 组织的 indicator 输入 bundle
│   ├── performance_series_bundle.json
│   ├── rank_ic_series.parquet
│   ├── quantile_return_panel.parquet
│   └── ...
├── indicator_output_h{horizon}/    # indicator 输出
│   ├── summary_scorecard.json
│   ├── metrics_matrix.parquet
│   └── dev4_evaluation_summary.json
└── label/                          # label 产物
    ├── tag_package.json
    ├── admission_decision.json
    ├── route_record.json
    └── lifecycle_event.json
```

---

## 市场数据缓存

系统采用三级缓存优化市场数据访问：

| 层级 | 大小 | 说明 |
|------|------|------|
| **原始 parquet** | — | 按日分片的原始行情数据 |
| **合并 parquet** | ~388MB | 合并后的缓存文件（6.8x 压缩比） |
| **进程内存** | ~388MB | 通过 `preload_market_data()` 预加载，fork 子进程 COW 共享 |

缓存位于 `database/cache/market_data/`，通过 `cache_utils.py` 管理。
