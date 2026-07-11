# AutoFactorEvaluation 接口规范

> 定义服务的外部输入/输出契约、各阶段产物格式、以及模块间数据交换标准。

---

## 1. 阶段间数据交换接口

### 1.1 通用产物结构

每个阶段输出到 temp 的因子目录包含：

```
{stage}_temp/
└── {factor_id}/
    ├── afv.json              ← 全链路元数据（各阶段追加标识）
    ├── data/                 ← 因子值数据（按日分片 parquet，仅最后阶段含评估标识）
    │   ├── 2016-01-04.parquet
    │   ├── 2016-01-05.parquet
    │   └── ...
    └── manifest.json         ← 产物清单（部分阶段可选）
```

**注意**：因子目录结构是**扁平**的（`{factor_id}/`），无 `{domain}/{freq}/` 前缀。
因子值文件以每日截面切片存储（`data/{YYYY-MM-DD}.parquet`），而非单一 `data.parquet`。

### 1.2 afv.json 全链路 Schema

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "raw_qa_hash_3e4d5f",
  "campaign_id": "llm_us_pv_quantaalpha_2026q2",
  "formula": "ts_cov(rank(close - open), rank(close), 5) / (ts_std(close, 5) + 1e-8)",
  "factor_id": "equity_daily_cross_sectional_0b649ced",

  "Gateway": {
    "Label": "Pass",
    "Reason": "",
    "run_id": "gateway_20260615_203319",
    "checked_at": "2026-06-15T12:33:19Z",
    "recommendation": "pass",
    "checks": {
      "schema_valid": true,
      "operator_allowed": true,
      "has_future": false,
      "complexity_score": 8.0,
      "is_within_budget": true,
      "is_duplicate": false
    }
  },

  "Assetization": {
    "Label": "Materialized",
    "status": "passed",
    "factor_id": "equity_daily_cross_sectional_0b649ced",
    "run_id": "assetization_20260615_141535",
    "materialized_at": "2026-06-15T14:15:35Z",
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
  },

  "Purification": {
    "Label": "Pure",
    "run_id": "purification_20260615_150000",
    "purified_at": "2026-06-15T15:00:00Z",
    "steps": ["imputation", "winsorization", "orthogonalization"],
    "industry_standard": "zjw",
    "row_count": 499872
  },

  "Evaluation": {
    "Label": "Evaluated",
    "eval_run_id": "evaluation_20260615_160000",
    "evaluated_at": "2026-06-15T16:00:00Z",
    "route_recommendation": "tier4_archive",
    "target_tier": "Tier4",
    "horizons": [1, 5, 20],
    "key_metrics": {
      "rank_ic_mean": -0.003,
      "rank_ic_ir": -0.776,
      "rank_ic_win_rate": 0.479,
      "top_minus_bottom_mean": -0.0001,
      "long_short_sharpe": -0.576,
      "turnover": 0.863,
      "universe_coverage_mean": 0.999
    },
    "tags": {
      "performance_tags": ["high_turnover"],
      "action_tags": ["needs_smoothing"],
      "semantic_label": "intraday_momentum",
      "confidence": 0.85
    },
    "admission_reason": "survival_view=failed",
    "row_count": 499872
  }
}
```

#### Assetization 段新增字段（相比旧版）

| 字段 | 类型 | 说明 |
|------|------|------|
| `market_period.start` | string | 市场数据覆盖起始日期（`YYYY-MM-DD`） |
| `market_period.end` | string | 市场数据覆盖结束日期（`YYYY-MM-DD`） |
| `row_count` | int | 因子数据总行数 |
| `formula` | string | 因子公式（冗余，便于独立查看） |
| `candidate_id` | string | 上游候选 ID（冗余，便于独立查看） |

### 1.3 data/ 目录规范

```
data/
├── 2016-01-04.parquet    ← 每个文件为一个交易日的截面因子值
├── 2016-01-05.parquet
└── ...

每文件 schema:
  TradeDate    datetime64    ← 交易日
  Symbol       string        ← 资产代码
  factor_id    string        ← 因子 ID
  factor_value float64       ← 因子值（经纯化/正交化等处理后）
```

---

## 2. 候选因子池输入规范（Gateway 输入）

### 2.1 目录结构

```
candidate_pool/
└── {campaign_id}/                    ← 挖掘算法批次标识
    └── {candidate_id}/               ← 因子唯一标识
        ├── manifest.json             ← 因子元数据（必需）
        ├── formula.txt               ← 因子公式文本（可选，与 manifest 二选一）
        └── config.json               ← Campaign 级配置（可选）
```

### 2.2 manifest.json

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "raw_qa_hash_3e4d5f",
  "campaign_id": "llm_us_pv_quantaalpha_2026q2",
  "generator_name": "quantaalpha",
  "formula": "ts_cov(rank(close - open), rank(close), 5) / (ts_std(close, 5) + 1e-8)",
  "domain_root": "price_volume",
  "frequency_bucket": "daily",
  "universe_id": "A_SHARE_LQTP",
  "depth": 5,
  "parent_local_hashes": ["raw_qa_hash_8f9a2c"],
  "metrics": {
    "train": {"ic": 0.051, "icir": 2.38}
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `schema_version` | string | 是 | 固定 `disk.v1` |
| `candidate_id` | string | 是 | 候选因子唯一 ID |
| `campaign_id` | string | 是 | 挖掘批次 ID |
| `formula` | string | 是 | 因子表达式 |
| `domain_root` | string | 是 | 数据域（`price_volume` / `fundamental` 等） |
| `frequency_bucket` | string | 是 | 频率桶（`daily` / `weekly` / `monthly`） |

---

## 3. 各阶段输入/输出合约

### 3.1 Gateway 阶段

| 项目 | 说明 |
|------|------|
| **输入** | `candidate_pool/{campaign}/{candidate}/manifest.json` + formula |
| **产出** | `gateway_temp/{candidate}/afv.json` + `afv.json["Gateway"]` |
| **Router 目标** | `gateway_pass_base/{candidate}/` （Label=Pass） |
| | `gateway_duplicated_base/{candidate}/` （Label=Duplicated） |
| | `anti_sample_base/gateway/{candidate}/` （Label=Rejected） |
| **配置引用** | `gateway_config.yaml` → 白名单/黑名单/复杂度权重/阈值 |
| **外部依赖** | `operator_whitelist.json`, `future_blacklist.json`, `complexity_weights.json` |
| **处理流程** | 6 步顺序检查：Schema 校验 → 算子白名单 → 未来函数检测 → 复杂度评估 → 语义去重 |

### 3.2 Assetization 阶段

| 项目 | 说明 |
|------|------|
| **输入** | `gateway_pass_base/{candidate}/afv.json` （含 Gateway.Pass 标识） |
| **产出** | `assetization_temp/{factor_id}/afv.json` + `data/` （按日 parquet） |
| **Router 目标** | `assetization_raw_factor_base/{factor_id}/` |
| **外部依赖** | `MARKET_DATA_PATH/StockDailyBar/` （行情数据） |
| **因子目录** | 扁平结构：`{factor_id}/`，无 `{domain}/{freq}/` 前缀 |

### 3.3 Purification 阶段

| 项目 | 说明 |
|------|------|
| **输入** | `assetization_raw_factor_base/{factor_id}/afv.json` + `data/` |
| **产出** | `purification_temp/{factor_id}/afv.json` + `data/` （值已纯化） |
| **Router 目标** | `purification_pure_factor_base/{factor_id}/` |
| **外部依赖** | `StockIndustry/` + `zjw` 分类口径 |
| | `tier3/*/` 已有因子库（可选，正交化依赖） |

### 3.4 Evaluation 阶段

| 项目 | 说明 |
|------|------|
| **输入** | `purification_pure_factor_base/{factor_id}/afv.json` + `data/` |
| **产出** | `evaluation_temp/{factor_id}/afv.json` + `data/` |
| **Router 目标** | 根据 `route_recommendation` 路由至对应 tier |
| **外部依赖** | `MARKET_DATA_PATH/StockDailyBar/` （行情，用于 forward return 计算） |
| | `all_configs/auto_factor_evaluation/configs/.env` （DeepSeek V4 Flash API key） |
| | `configs/tag_policy.json` （标签策略） |
| | `configs/route_policy.json` （路由策略） |

### 3.5 Route Recommendation → 目标 Tier 映射

| `route_recommendation` | 目标目录 | ConfigManager 访问方式 |
|------------------------|---------|----------------------|
| `tier3a_core` | `tier3/3a_core_production_base/` | `output_tier("tier3a_core")` |
| `tier3b_satellite` | `tier3/3b_satellite_production_base/` | `output_tier("tier3b_satellite")` |
| `tier3c_feature` | `tier3/3c_feature_matierial_base/` | `output_tier("tier3c_feature")` |
| `tier3d_optimized_reserve` | `tier3/3d_operation_storage_base/` | `output_tier("tier3d_optimized_reserve")` |
| `tier2_incubator` | `tier2/2_fix_base/` | `output_tier("tier2_incubator")` |
| `tier2x_optimization_factory` | `tier2/2x_llm_mutation_base/` | `output_tier("tier2x_optimization_factory")` |
| `tier4_archive` | `tier4/anti_sample_base/` | `output_tier("tier4_archive")` |

---

## 4. 外部配置 YAML 规范

### 4.1 Gateway 配置

```yaml
# all_configs/auto_factor_evaluation/gateway_config.yaml
paths:
  candidate_pool: "database/real/candidate_pool"
  candidate_pool_record: "database/real/candidate_pool_record"
  gateway_pass_base: "database/tier1/gateway_pass_base"
  duplicated_base: "database/tier1/gateway_duplicated_base"
  anti_sample_base: "database/tier4/anti_sample_base"

market_data_path: "/path/to/lqtp_data"

max_complexity: 20.0
semantic_similarity_threshold: 0.95
max_ast_depth: 10

operator_whitelist_file: "configs/operator_whitelist.json"
future_blacklist_file: "configs/future_blacklist.json"
complexity_weights_file: "configs/complexity_weights.json"
```

### 4.2 Purification 配置

```yaml
# all_configs/auto_factor_evaluation/purification_config.yaml
paths:
  raw_factor_base: "database/tier1/assetization_raw_factor_base"
  temp_base: "database/tier0/purification_temp"
  pure_factor_base: "database/tier1/purification_pure_factor_base"

industry_data:
  path: "/path/to/lqtp_data/StockIndustry"
  standard: "zjw"

purification:
  imputation_method: "industry_weighted"
  max_delay: 5
  winsorization_mad_multiplier: 3.148
```

### 4.3 Evaluation 配置

```yaml
# all_configs/auto_factor_evaluation/evaluation_config.yaml
paths:
  pure_factor_base: "database/tier1/purification_pure_factor_base"
  temp_base: "database/tier0/evaluation_temp"
  output:
    tier3a_core: "database/tier3/3a_core_production_base"
    tier3b_satellite: "database/tier3/3b_satellite_production_base"
    tier3c_feature: "database/tier3/3c_feature_matierial_base"
    tier3d_optimized_reserve: "database/tier3/3d_operation_storage_base"
    tier2_incubator: "database/tier2/2_fix_base"
    tier2x_optimization_factory: "database/tier2/2x_llm_mutation_base"
    tier4_archive: "database/tier4/anti_sample_base"

market_data_path: "/path/to/lqtp_data/StockDailyBar"

evaluation:
  horizons: [1, 5, 20]
  min_assets: 30
  n_quantiles: 5
  factor_direction: 1
  weighting: "equal"
  cost_bps: 0.0
```

---

## 5. 配置加载机制

### 5.1 ConfigManager（统一配置总线）

```python
# pipeline.py 中的 _get_config() 单例模式
_CONFIG: Optional[Any] = None

def _get_config() -> Any:
    global _CONFIG
    if _CONFIG is None:
        from config_manager import ConfigManager
        _CONFIG = ConfigManager()
    return _CONFIG
```

所有路径通过 `ConfigManager` 访问：
- `_cfg.path("gateway_pass_base")` → 解析为绝对路径
- `_cfg.output_tier("tier3b_satellite")` → 解析输出 tier 路径
- `_cfg.market_data_path` → 行情数据根目录
- `_cfg.gateway_param("max_complexity")` → Gateway 参数

### 5.2 环境变量 `AFVCONFIG`

```
export AFVCONFIG=/path/to/custom_profile
```

优先级：显式 `config_dir` 参数 > `AFVCONFIG` 环境变量 > 内置默认路径 (`all_configs/auto_factor_evaluation/`)

---

## 6. 错误处理合约

| 场景 | 行为 | 产物状态 |
|------|------|---------|
| **候选池 manifest 格式错误** | Gateway 记录 ERROR，跳过该候选 | 候选保留在原位 |
| **算子不在白名单中** | Gateway Label=Rejected，路由至 tier4 | 候选保留在原位 |
| **Assetization 市场数据缺失** | Worker 抛出异常，因子留在 pass_base | pass_base 保留 |
| **Purification 行业数据缺失** | 退化为 forward_fill_decay 填补，跳过正交化 | 仍标记 Pure |
| **Evaluation 计算超时** | Worker 超时退出，因子留在 pure_factor_base | pure_factor_base 保留 |
| **Router move 失败** | 重试 3 次，失败后因子留在 temp | temp 保留，人工介入 |
