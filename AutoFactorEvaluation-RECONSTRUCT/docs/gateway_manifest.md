# Gateway 审查模块规范文档（Gateway Manifest）

> 定义输入/输出格式、配置规范、字段兼容性规则及跨市场适配机制。

---

## 1. 架构总览

```
                    ┌──────────────────────────────────────────────────┐
                    │              Gateway 审查引擎                     │
  candidate_pool/   │  ┌──────────┐  ┌──────────┐  ┌────────────┐    │  tier0/gateway_temp/
  ────────────────→ │  │ Schema   │→│ 算子白   │→│ 字段兼容   │    │ ────────────────→
  {campaign}/       │  │ 校验     │  │ 名单校验 │  │ 性校验     │    │  {candidate_id}/
  {factor}/         │  └──────────┘  └──────────┘  └────────────┘    │  ├── afv.json
  └── manifest.json │        ↓            ↓              ↓          │  └── manifest.json
                    │  ┌──────────┐  ┌──────────┐  ┌────────────┐    │
                    │  │ 未来函数 │  │ 复杂度   │  │ 语义去重   │    │
                    │  │ 拦截     │→│ 评估     │→│（可选）    │    │
                    │  └──────────┘  └──────────┘  └────────────┘    │
                    │        ↓            ↓              ↓          │
                    │  ┌────────────────────────────────────────┐    │
                    │  │  数据质量检测（可选，需行情数据路径）   │    │
                    │  └────────────────────────────────────────┘    │
                    └──────────────────────────────────────────────────┘
```

**设计原则**：
- **模块化**：每项检查独立，可插拔
- **可移植**：不绑定特定市场/资产
- **标准化**：输入/输出格式统一

---

## 2. 输入规范

### 2.1 候选因子目录结构

```
{candidate_pool}/{campaign_id}/
├── config.json                 # 因子池宏观配置（可选）
└── {candidate_id}/             # 单个因子目录
    └── manifest.json           # 因子元信息（必需）
```

### 2.2 manifest.json

```json
{
  "schema_version": "disk.v1",
  "candidate_id": "raw_qa_hash_8f9a2c",
  "campaign_id": "llm_us_pv_quantaalpha_2026q2",
  "formula": "ts_mean(close / open - 1, 3)",
  "generator_name": "quantaalpha",
  "iteration_id": "iter_03",
  "batch_id": "20260531_01",
  "domain_root": "us_stock",
  "frequency_bucket": "daily",
  "universe_id": "US_MAIN_STOCKS_PIT",
  "validation": { "status": "pending" }
}
```

必需字段：`schema_version`、`candidate_id`、`campaign_id`、`formula`、`domain_root`、`frequency_bucket`

---

## 3. 输出规范

### 3.1 审查后目录

```
tier0/gateway_temp/{candidate_id}/
├── afv.json                  # 审查结果
└── manifest.json              # 原始元信息
```

### 3.2 afv.json

```json
{
  "candidate_id": "raw_qa_hash_8f9a2c",
  "formula": "...",
  "Gateway": {
    "Label": "Pass",
    "run_id": "gateway_...",
    "checked_at": "2026-06-15T08:30:00Z",
    "recommendation": "pass",
    "checks": {
      "schema_valid": true,
      "operator_allowed": true,
      "has_future": false,
      "complexity_score": 3.5,
      "is_within_budget": true,
      "is_duplicate": false,
      "field_compatible": true
    }
  }
}
```

### 3.3 路由目标

| Label | 目标 | 说明 |
|-------|------|------|
| `Pass` | `tier1/gateway_pass_base/` | 全部通过 |
| `Temp` | `tier1/gateway_temp_base/` | 数据质量警告 |
| `Duplicated` | `tier1/gateway_duplicated_base/` | 重复 |
| `Rejected` | `tier4/anti_sample_base/gateway/` | 非法 |

---

## 4. 配置规范

### 4.1 gateway_config.yaml

```yaml
paths:
  candidate_pool: "database/real/candidate_pool"
  gateway_pass_base: "database/tier1/gateway_pass_base"
  duplicated_base: "database/tier1/gateway_duplicated_base"
  anti_sample_base: "database/tier4/anti_sample_base"
  report_cache: "database/cache/gateway_report_cache"

market_data_path: "/path/to/market/data"

max_complexity: 20.0
semantic_similarity_threshold: 0.95
max_ast_depth: 10

features:
  enable_complexity_check: true
  enable_semantic_dedup: true
  enable_future_scan: true
  enable_kafka: false

operator_whitelist_file: "path/to/operator_whitelist.json"
future_blacklist_file: "path/to/future_blacklist.json"
complexity_weights_file: "path/to/complexity_weights.json"
```

### 4.2 跨市场适配

更换市场只需替换配置文件，**不修改代码**：

| 配置 | 美股示例 |
|------|---------|
| `operator_whitelist_file` | `us_equity_operators.json` |
| `future_blacklist_file` | `us_future_blacklist.json` |
| `complexity_weights_file` | `us_complexity_weights.json` |
| `market_data_path` | `/data/us_market/` |

---

## 5. 算子白名单规范

### 5.1 结构

```json
{
  "version": "v1.0",
  "field_type_rules": { "type_name": { "match_patterns": [], "table_patterns": [] } },
  "operator_restrictions": { "group_name": { "operators": [], "allowed_types": [] } },
  "operators": ["TS_MEAN", "RANK", ...]
}
```

### 5.2 字段类型匹配

| 属性 | 匹配方式 | 优先级 |
|------|---------|--------|
| `match_patterns` | `re.fullmatch(field_name, pattern, re.I)` | 高 |
| `table_patterns` | `re.search(table.field, pattern, re.I)` | 低 |

### 5.3 算子限制

不在 `operator_restrictions` 中的算子视为**通用算子**，不限制字段类型。

### 5.4 预定义字段类型

| 类型 | 示例字段 |
|------|---------|
| `price` | Close, Open, High, Low, Vwap |
| `return` | Return, ret |
| `volume_amount` | Volume, Amount |
| `capital` | MarketCap, TotalCapital |
| `valuation_ratio` | PeRatio, PbRatio |
| `financial` | StockBalance.*, StockIncome.* |
| `industry` | IndustryCode |
| `status` | IsSuspend, PublicStatusCode |
| `datetime` | TradeDate, PubDate |
| `identifier` | Symbol, Name |
| `l2` | tick_price, tick_volume |
| `minute_bar` | minute_open, minute_close |

### 5.5 新增市场适配

1. 定义 `field_type_rules`（字段命名模式）
2. 定义 `operator_restrictions`（算子兼容性）
3. 配置新市场的三份 JSON 文件
4. **无需修改 Gateway 代码**

---

## 6. 审查流程

### 6.1 执行顺序

```
① Schema 校验    → manifest.json 格式和必需字段
② 算子白名单     → 算子在 whitelist 中
③ 字段兼容性     → 算子与字段类型匹配
④ 未来函数拦截   → 正则匹配禁止模式
⑤ 复杂度评估     → 权重累加 + 深度惩罚
⑥ 语义去重       → 哈希匹配
⑦ 数据质量       → 可选，标签不阻断流程
```

### 6.2 失败结果

| 阶段 | 结果 |
|------|------|
| ①-⑤ | Rejected |
| ⑥ | Duplicated |
| ⑦ | Pass + Temp 标记 |

---

## 7. 扩展指南

### 7.1 添加新算子

1. `operator_whitelist.json` → `operators` 列表 + `operator_restrictions`
2. `complexity_weights.json` → `base_weights`
3. **无需修改代码**

### 7.2 添加新字段类型

1. `field_type_rules` 添加新类型
2. `operator_restrictions` 中相关算子添加该类型
3. **无需修改代码**

### 7.3 适配新市场

```
创建三份 JSON 配置文件 → 更新 gateway_config.yaml 引用 → 重启
```
