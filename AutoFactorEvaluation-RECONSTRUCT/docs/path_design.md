# AutoFactorEvaluation 路径设计方案（v2 — 三基座）

## 概览

系统采用 **三基座路径模式**，将不同性质的路径分离管理：

| 基座 | 用途 | 存储类型 | 自动推导的子路径 |
|------|------|----------|-----------------|
| `candidate_pool` | 候选因子池 | COS | `candidate/`, `archive/` |
| `factor_pool` | 因子持久化存储 | COS | tier1~tier4 共 12 个子目录 |
| `local_tmp` | 本地临时工作区 | 本地磁盘 | tier0、缓存、日志共 13 个子路径 |

---

## 1. `candidate_pool` — COS 候选因子池

**配置**: `candidate_pool: cos://qs-cold/candidate_pool/ashare/`

```
{candidate_pool}/
├── candidate/         ← 封板因子投放入口（系统从此读取）
└── archive/           ← 已处理完成的 campaign 归档
```

### 读取逻辑

- **原逻辑**: 系统直接从 `candidate_pool/` 读取因子，空池移至 `candidate_pool/candidate_pool_record/`
- **新逻辑**: 系统从 `candidate_pool/candidate/` 读取因子，空池移至 `candidate_pool/archive/`

---

## 2. `factor_pool` — COS 因子持久化基座

**配置**: `factor_pool: cos://qs-cold/factor_pool/ashare/`

```
{factor_pool}/
├── tier1/gateway_pass_base/
├── tier1/gateway_duplicated_base/
├── tier1/gateway_anti_sample_base/
├── tier1/assetization_raw_factor_base/
├── tier1/purification_pure_factor_base/
├── tier2/2_fix_base/
├── tier2/2x_llm_mutation_base/
├── tier3/3a_core_production_base/
├── tier3/3b_satellite_production_base/
├── tier3/3c_feature_matierial_base/
├── tier3/3d_operation_storage_base/
└── tier4/anti_sample_base/
```

---

## 3. `local_tmp` — 本地临时工作区

**配置**: `local_tmp: /tmp/auto_factor_evaluation_tmp`（默认值）

```
{local_tmp}/
├── tier0/gateway_temp/
├── tier0/assetization_temp/
├── tier0/purification_temp/
├── tier0/evaluation_temp/
├── cache/market_data/
├── cache/timeseries_forward_return/
├── cache/evaluation_label/
├── cache/gateway_report/
└── logs/{pipeline,gateway,assetization,purification,evaluation}/
```

---

## 4. 路径推导对照表

### CANDIDATE_TREE

| 逻辑键名 | 相对路径 | 用途 |
|----------|----------|------|
| `candidate` | `candidate/` | 封板因子投放入口 |
| `archive` | `archive/` | 已处理归档 |

### FACTOR_TREE

| 逻辑键名 | 相对路径 | 用途 |
|----------|----------|------|
| `gateway_pass_base` | `tier1/gateway_pass_base/` | Gateway Pass 输出 |
| `gateway_duplicated_base` | `tier1/gateway_duplicated_base/` | 重复因子输出 |
| `gateway_anti_sample_base` | `tier1/gateway_anti_sample_base/` | 反样本 |
| `assetization_raw_factor_base` | `tier1/assetization_raw_factor_base/` | 原始因子基座 |
| `purification_pure_factor_base` | `tier1/purification_pure_factor_base/` | 纯化因子基座 |
| `tier2_incubator` | `tier2/2_fix_base/` | 孵化工段 |
| `tier2x_optimization_factory` | `tier2/2x_llm_mutation_base/` | LLM 变异 |
| `tier3a_core` | `tier3/3a_core_production_base/` | 核心生产 |
| `tier3b_satellite` | `tier3/3b_satellite_production_base/` | 卫星生产 |
| `tier3c_feature` | `tier3/3c_feature_matierial_base/` | Feature |
| `tier3d_optimized_reserve` | `tier3/3d_operation_storage_base/` | 运营储备 |
| `tier4_archive` | `tier4/anti_sample_base/` | 归档 |

### LOCAL_TREE

| 逻辑键名 | 相对路径 | 用途 |
|----------|----------|------|
| `gateway_temp` | `tier0/gateway_temp/` | Gateway 临时区 |
| `assetization_temp` | `tier0/assetization_temp/` | Assetization 临时区 |
| `purification_temp` | `tier0/purification_temp/` | Purification 临时区 |
| `evaluation_temp` | `tier0/evaluation_temp/` | Evaluation 临时区 |
| `market_data_cache` | `cache/market_data/` | 行情数据缓存 |
| `timeseries_forward_return_cache` | `cache/timeseries_forward_return/` | 时序收益缓存 |
| `evaluation_label_cache` | `cache/evaluation_label/` | 评估标签缓存 |
| `gateway_report_cache` | `cache/gateway_report/` | 报告缓存 |
| `pipeline_log` | `logs/pipeline/` | 流水线日志 |
| `gateway_log` | `logs/gateway/` | Gateway 日志 |
| `assetization_log` | `logs/assetization/` | Assetization 日志 |
| `purification_log` | `logs/purification/` | Purification 日志 |
| `evaluation_log` | `logs/evaluation/` | Evaluation 日志 |

---

## 5. 自动创建机制

- **ConfigManager 初始化时**: 自动调用 `cos_utils.ensure_candidate_pool_dirs()` 和 `cos_utils.ensure_factor_pool_dirs()` 确保 COS 目录完整。
- **pipeline 启动时**: `_validate_all_paths()` 自动创建本地 `local_tmp` 下的所有子目录。
- COS 目录通过上传 `.empty` 标记文件保持可见。

## 6. 路径类型

- **绝对路径**: 以 `/` 或 `cos://` 开头 — 直接使用
- **相对路径**: 不再支持（移除了 `project_root`）
