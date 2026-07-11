# AutoFactorEvaluation 配置文档

## 配置总览

所有外部传参集中在一个 `config.yaml` 文件中，通过环境变量 `AFVCONFIG` 或 `ConfigManager(config_dir=...)` 传入。

### 配置分类

| 类别 | 说明 | 必填 |
|------|------|:----:|
| **基座路径** | 三大基座路径，自动推导所有子路径 | ✅ |
| **数据源路径** | 行情表、行业表的路径及字段映射 | ✅ |
| **DeepSeek 密钥** | API 密钥字符串 | ✅ |
| **模块参数** | Gateway/Purification/Evaluation 参数 | ✅（暂保留） |

---

## 一、三大基座路径

### 1. `candidate_pool` — 候选因子池（COS）

A 股样例：`cos://qs-cold/candidate_pool/ashare/`

```
{candidate_pool}/
├── candidate/         ← 封板因子投放入口（系统从此读取待处理因子）
└── archive/           ← 已处理完成的 campaign 归档（空池移入）
```

**读取逻辑变更**：原系统直接从 `candidate_pool/` 读取，现改为从 `candidate_pool/candidate/` 读取。空的 campaign 从 `candidate/` 移至 `archive/`。

### 2. `factor_pool` — 因子持久化基座（COS）

A 股样例：`cos://qs-cold/factor_pool/ashare/`

```
{factor_pool}/
├── tier1/
│   ├── gateway_pass_base/
│   ├── gateway_duplicated_base/
│   ├── gateway_anti_sample_base/
│   ├── assetization_raw_factor_base/
│   └── purification_pure_factor_base/
├── tier2/
│   ├── 2_fix_base/
│   └── 2x_llm_mutation_base/
├── tier3/
│   ├── 3a_core_production_base/
│   ├── 3b_satellite_production_base/
│   ├── 3c_feature_matierial_base/
│   └── 3d_operation_storage_base/
└── tier4/
    └── anti_sample_base/
```

### 3. `local_tmp` — 本地临时工作区

默认：`/tmp/auto_factor_evaluation_tmp/`

```
{local_tmp}/
├── tier0/
│   ├── gateway_temp/
│   ├── assetization_temp/
│   ├── purification_temp/
│   └── evaluation_temp/
├── cache/
│   ├── market_data/
│   ├── timeseries_forward_return/
│   ├── evaluation_label/
│   └── gateway_report/
└── logs/
    ├── pipeline/
    ├── gateway/
    ├── assetization/
    ├── purification/
    └── evaluation/
```

---

## 二、数据源路径

### `market_data` — 行情数据

```yaml
market_data:
  path: cos://qs-cold/clean_data/ashare/lqtp_data/StockDailyBar/
  forward_return_field: Vwap       # 用于计算 forward return 的字段
  adjustment_field: Factor          # 后复权调整字段
```

| 字段 | 必填 | 说明 | A 股样例值 |
|------|:----:|------|-----------|
| `path` | ✅ | 行情数据表完整路径（COS 或本地） | `cos://.../StockDailyBar/` |
| `forward_return_field` | ✅ | 用于计算前向收益率的字段名 | `Vwap` |
| `adjustment_field` | ✅ | 后复权调整使用的字段名 | `Factor` |

### `industry_data` — 行业分类数据

```yaml
industry_data:
  path: cos://qs-cold/clean_data/ashare/lqtp_data/StockIndustry/
  source_field: IndustrySource      # 识别分类口径的字段
  code_field: IndustryCode           # 表示行业代码的字段
  standard: sws2021                  # 行业分类口径
```

| 字段 | 必填 | 说明 | A 股样例值 |
|------|:----:|------|-----------|
| `path` | ✅ | 行业分类表完整路径（COS 或本地） | `cos://.../StockIndustry/` |
| `source_field` | ✅ | 识别行业分类口径的字段名 | `IndustrySource` |
| `code_field` | ✅ | 表示行业代码的字段名 | `IndustryCode` |
| `standard` | ✅ | 行业分类口径名称 | `sws2021` |

---

## 三、DeepSeek 密钥

```yaml
deepseek:
  api_key: sk-your-api-key-here
```

直接传入密钥字符串，无需 `.env` 文件。

---

## 四、COS 校验工具

```bash
# 校验 candidate_pool
python -m cos_utils candidate_pool cos://qs-cold/candidate_pool/ashare/ --fix

# 校验 factor_pool
python -m cos_utils factor_pool cos://qs-cold/factor_pool/ashare/ --fix
```

自动创建缺失的子目录（candidate/, archive/ 及所有 tier 目录）。

---

## 五、配置示例（完整 A 股）

完整样例请见 `config.yaml`。
