# AutoFactorEvaluation 数据产物流动全览

> 本文档完整描述一个因子从 `candidate_pool` 出发，流经 Gateway → Assetization → Purification → Evaluation 四个阶段，最终路由至目标 tier 库的全过程中，**封板池**与**因子目录**的信息/数据结构如何变化。

---

## 1. 总览：数据流管道

```
candidate_pool/{campaign}/              ← 外部挖掘算法写入
    │  config.json (campaign 元信息)
    │  {candidate_id}/manifest.json (因子配方)
    ▼
[Gateway 审查] → afv.json 追加 Gateway 段
    │
tier0/gateway_temp/{candidate_id}/
    │  afv.json (含 Gateway 段)
    │  manifest.json
    ▼
[Gateway Router]
    │
    ├─ Pass      → tier1/gateway_pass_base/{candidate_id}/
    ├─ Duplicated → tier1/gateway_duplicated_base/{candidate_id}/
    └─ Rejected   → tier4/anti_sample_base/gateway/{candidate_id}/
    │
    ▼
[Assetization 计算] → 生成 factor_id + 因子值 + 按日 parquet
    │
tier0/assetization_temp/{factor_id}/
    │  afv.json (含 Assetization 段)
    │  data/{YYYY-MM-DD}.parquet (因子值)
    │  manifest.json
    ▼
[Assetization Router]
    │
    └─ → tier1/assetization_raw_factor_base/{factor_id}/
    │
    ▼
[Purification 纯化] → 填补/去极值/正交化
    │
tier0/purification_temp/{factor_id}/
    │  afv.json (含 Purification 段)
    │  data/{YYYY-MM-DD}.parquet (纯化后值)
    ▼
[Purification Router]
    │
    └─ → tier1/purification_pure_factor_base/{factor_id}/
    │
    ▼
[Evaluation 评估] → 时序绩效 + 统计指标 + 语义标签
    │
tier0/evaluation_temp/{factor_id}/
    │  afv.json (含 Evaluation 段)
    │  data/{YYYY-MM-DD}.parquet
    │  timeseries_output/ (绩效时序产物)
    │  pipeline_work/ (中间产物)
    ▼
[Evaluation Router]   ← 根据 route_recommendation
    │
    ├─ tier3a_core           → tier3/3a_core_production_base/{factor_id}/
    ├─ tier3b_satellite      → tier3/3b_satellite_production_base/{factor_id}/
    ├─ tier3c_feature        → tier3/3c_feature_matierial_base/{factor_id}/
    ├─ tier3d_optimized_reserve → tier3/3d_operation_storage_base/{factor_id}/
    ├─ tier2_incubator       → tier2/2_fix_base/{factor_id}/
    ├─ tier2x_factory        → tier2/2x_llm_mutation_base/{factor_id}/
    └─ tier4_archive         → tier4/anti_sample_base/{factor_id}/
```

---

## 2. 封板池生命周期

### 2.1 初始状态：活池

```
candidate_pool/
└── llm_us_pv_2026q2/                    ← 外部写入的 campaign 池
    ├── config.json                      ← 池自身元信息（始终保留）
    ├── raw_hash_a1b2c3/
    │   └── manifest.json
    ├── raw_hash_d4e5f6/
    │   └── manifest.json
    └── ...                              ← N 个候选因子
```

### 2.2 处理中：因子逐渐被抽离

```
candidate_pool/llm_us_pv_2026q2/
    ├── config.json
    ├── raw_hash_a1b2c3/      ← 已处理（Gateway 审查通过，移至 pass_base）
    ├── raw_hash_d4e5f6/      ← 正在处理中（在 gateway_temp）
    └── raw_hash_x9y8z7/      ← 尚未处理
```

### 2.3 空池：所有因子被抽离

```
candidate_pool/llm_us_pv_2026q2/
    └── config.json            ← 仅剩池配置，无任何因子子目录
```

### 2.4 归档：转移至 record 路径

```
candidate_pool_record/                   ← 由 _archive_empty_campaigns() 触发
└── llm_us_pv_2026q2/
    └── config.json            ← 原封不动保留，用于追溯
```

> 空池归档在 Gateway 阶段结束后自动执行。识别标准：campaign 目录中仅有 `config.json` 而无任何含 `manifest.json` 的子目录。

---

## 3. 因子目录结构演变

### 3.1 输入阶段：candidate_pool

```
{pool}/{campaign}/{candidate_id}/
├── manifest.json              # 因子配方（必填）
│   ├── schema_version         # "disk.v1"
│   ├── candidate_id           # 候选因子 ID
│   ├── campaign_id            # 所属批次
│   ├── formula                # 因子表达式
│   ├── domain_root            # 数据域
│   ├── frequency_bucket       # 频率
│   ├── depth                  # AST 深度
│   └── metrics                # 训练集指标（可选）
└── (无 afv.json，无 data/)
```

### 3.2 Gateway 通过：gateway_pass_base

```
tier1/gateway_pass_base/{candidate_id}/
├── afv.json                   # ← 首次生成
│   ├── formula                # 源自 manifest
│   ├── candidate_id           # 源自 manifest
│   ├── campaign_id            # 源自 manifest
│   └── Gateway:               # ← 新增
│       ├── Label              # "Pass"
│       ├── run_id
│       ├── checked_at
│       ├── recommendation
│       └── checks: {...}      # 各项审查结果
├── manifest.json              # 保留上游元信息
└── (无 data/)
```

### 3.3 Assetization 产出：assetization_temp → raw_factor_base

```
tier1/assetization_raw_factor_base/{factor_id}/
├── afv.json
│   ├── ...                    # 保留上游所有字段
│   ├── factor_id              # 全局唯一 ID（首次出现）
│   └── Assetization:          # ← 新增
│       ├── Label              # "Materialized"
│       ├── factor_id
│       ├── run_id
│       ├── materialized_at
│       ├── coordinates        # {signal_structure, asset_class, ...}
│       ├── market_period      # ← 行情时间区间
│       │   ├── start          # 数据最早日期
│       │   └── end            # 数据最晚日期
│       ├── row_count
│       └── formula
├── data/                      # ← 首次出现
│   ├── 2026-01-19.parquet     # 每日一个文件
│   ├── 2026-01-20.parquet
│   └── ...                    # TradeDate, Symbol, factor_id, factor_value
└── manifest.json
```

**data/ 目录 parquet schema：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `TradeDate` | datetime64 | 交易日 |
| `Symbol` | string | 资产代码 |
| `factor_id` | string | 全局唯一因子 ID |
| `factor_value` | float64 | 因子值 |

### 3.4 Purification 产出：purification_temp → pure_factor_base

```
tier1/purification_pure_factor_base/{factor_id}/
├── afv.json
│   ├── ...                    # 保留上游所有字段
│   └── Purification:          # ← 新增
│       ├── Label              # "Pure"
│       ├── run_id
│       ├── purified_at
│       ├── steps              # ["imputation", "winsorization", "orthogonalization"]
│       ├── industry_standard  # 行业分类口径
│       └── row_count
├── data/                      # 值已被纯化处理
│   ├── 2026-01-19.parquet     # 同 schema，factor_value 为纯化后值
│   └── ...
└── manifest.json
```

**纯化处理的内容：**
- 缺失值填补（行业加权 / 时序衰减）
- 去极值（MAD 方法）
- 风险正交化（WLS 回归剥离风险因子暴露，可选）

### 3.5 Evaluation 产出：evaluation_temp → 目标 tier

```
tier3/3b_satellite_production_base/{factor_id}/   (示例)
├── afv.json
│   ├── ...                    # 保留上游所有字段
│   └── Evaluation:            # ← 最终阶段
│       ├── Label              # "Evaluated"
│       ├── eval_run_id
│       ├── evaluated_at
│       ├── route_recommendation  # 路由推荐
│       ├── target_tier           # 目标层级
│       ├── horizons              # [1, 5, 20]
│       ├── key_metrics           # 绩效指标矩阵
│       │   ├── rank_ic_mean
│       │   ├── rank_ic_ir
│       │   ├── rank_ic_win_rate
│       │   ├── top_minus_bottom_mean
│       │   ├── long_short_sharpe
│       │   ├── turnover
│       │   └── universe_coverage_mean
│       ├── tags                  # 标签
│       │   ├── performance_tags  # 绩效标签
│       │   ├── action_tags       # 操作标签
│       │   ├── semantic_label    # DeepSeek 语义标签
│       │   └── confidence        # 置信度
│       ├── admission_reason      # 准入/拒绝理由
│       └── row_count
├── data/                      # 因子值（同 schema）
└── manifest.json
```

---

## 4. 各阶段 afv.json 字段累积

| 字段 | candidate_pool | gateway_pass | raw_factor | pure_factor | 最终 tier |
|------|:---:|:---:|:---:|:---:|:---:|
| `schema_version` | — | ✅ | ✅ | ✅ | ✅ |
| `formula` | manifest | ✅ | ✅ | ✅ | ✅ |
| `candidate_id` | manifest | ✅ | ✅ | ✅ | ✅ |
| `campaign_id` | manifest | ✅ | ✅ | ✅ | ✅ |
| `factor_id` | — | — | ✅ | ✅ | ✅ |
| `Gateway{}` | — | ✅ | ✅ | ✅ | ✅ |
| `Assetization{}` | — | — | ✅ | ✅ | ✅ |
| `market_period{}` | — | — | ✅ | ✅ | ✅ |
| `Purification{}` | — | — | — | ✅ | ✅ |
| `Evaluation{}` | — | — | — | — | ✅ |

---

## 5. 数据目录（`data/`）在各阶段的变化

| 阶段 | 数据目录 | factor_value 含义 | 行数变化 |
|------|---------|-------------------|---------|
| Assetization | `data/{date}.parquet` | 原始因子值 | 原始 N 行 |
| Purification | `data/{date}.parquet` | 纯化后因子值 | 填补后 N' ≥ N |
| Evaluation | `data/{date}.parquet` | 纯化后因子值（不变） | 同 purification |

---

## 6. 关键变更点总结

| 时间点 | 发生什么 | 产物位置 |
|--------|---------|---------|
| Gateway 审查完成 | 生成 afv.json，追加 Gateway 段 | `tier0/gateway_temp/` |
| Gateway Router 执行 | 按 Label 分发到不同基座 | `tier1/{pass,duplicated}_base/` 或 `tier4/` |
| **所有因子抽离后** | **campaign 目录仅剩 config.json → 归档** | **→ `candidate_pool_record/`** |
| Assetization 计算完成 | 生成 factor_id + data/ 目录 | `tier0/assetization_temp/` |
| Assetization Router | 移至 RawFactor 基座 | `tier1/assetization_raw_factor_base/` |
| Purification 纯化完成 | 因子值经填补/去极值/正交化 | `tier0/purification_temp/` |
| Purification Router | 移至 PureFactor 基座 | `tier1/purification_pure_factor_base/` |
| Evaluation 评估完成 | 时序绩效 + 指标 + 语义标签 | `tier0/evaluation_temp/` |
| Evaluation Router | 按 route_recommendation 路由 | `tier{2,3,4}/.../` |
