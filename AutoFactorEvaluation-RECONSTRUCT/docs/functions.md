# AutoFactorEvaluation 功能概述

## 项目目的

**AutoFactorEvaluation** 是一个自动化量化因子评估全流程系统。系统从候选因子出发，依次经过**网关校验 → 因子资产化 → 因子纯化 → 绩效评估与贴标**四个阶段，最终给出因子的统计评分、语义标签和入库推荐方向。

---

## 模块总览

```
                        AutoFactorEvaluation
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
    Gateway              Assetization              Purification
    (极速网关)            (因子资产化)              (因子纯化)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                           Evaluation
                         (绩效评估与贴标)
                                │
                                ▼
                    统计指标 + 标签 + 路由
```

---

## 各模块功能

### 1. Gateway（极速网关）

位于 `gateway/gateway/` 子包，通过 `gateway/gateway_core.py` 中的 `GatewayCore` 类实现。

| 功能 | 说明 |
|------|------|
| Schema 校验 | 检查候选因子 manifest.json 格式和必需字段 |
| 算子白名单 | 验证表达式中的算子在 `operator_whitelist.json` 允许列表中 |
| 未来函数拦截 | 通过正则匹配 + `future_blacklist.json` 检测并拒绝引用未来数据的因子 |
| 复杂度评估 | 根据 `complexity_weights.json` 评估计算开销，超出预算则拒绝 |
| 语义去重 | 基于 SHA256 哈希检测重复因子，避免重复计算 |
| 路由决策 | Pass → `tier1/gateway_pass_base/`；Duplicated → `tier1/gateway_duplicated_base/`；Rejected → `tier4/anti_sample_base/` |
| Kafka（可选） | 可将 Pass 因子发送到消息队列（默认关闭） |

**处理流程**：

```
manifest.json → ① Schema 校验 → ② 算子白名单 → ③ 未来函数检测
    → ④ 复杂度评估 → ⑤ 语义去重 → ⑥ 路由决策
```

### 2. Assetization（因子资产化）

位于 `assetization/scripts/` 中。

| 功能 | 说明 |
|------|------|
| `resolve_factor_id` | 从 Config 提取坐标，生成全局唯一 `factor_id`（格式：`{asset_class}_{frequency}_{signal}_{suffix8}`） |
| `compute_factor_values` | 解析 DSL 公式，计算因子值（支持 pandas/polars 双后端） |
| `run_assetization_pipeline` | 完整编排：读 Gateway 输出 → 生成 ID → 读行情 → 计算 → 写入 RawFactorBase |
| 输出 | `afv.json` + `data/{YYYY-MM-DD}.parquet`（按日分片，非单一 `data.parquet`） |

**因子目录结构**（扁平，无 `{domain}/{freq}/` 前缀）：

```
assetization_raw_factor_base/
└── {factor_id}/
    ├── afv.json
    └── data/
        ├── 2016-01-04.parquet
        ├── 2016-01-05.parquet
        └── ...
```

### 3. Purification（因子纯化）

位于 `purification/scripts/` 中。

| 功能 | 说明 |
|------|------|
| 动态缺失值填补 | `industry_weighted`（行业加权均值填补）或 `forward_fill_decay`（时序前向填充+衰减） |
| 稳健去极值 | MAD 方法，乘数 3.148（对应 3-Sigma），numpy 向量化实现 |
| 风险正交化（可选） | WLS 回归剥离风险因子暴露，无辅助数据时跳过 |
| 输入输出校验 | 列完整性、主键唯一、factor_id 一致、时间序正确、row_count |

### 4. Evaluation（绩效评估与贴标）

位于 `evaluation/scripts/` 中。

| 阶段 | 功能 | 说明 |
|------|------|------|
| 4.1 timeseries | 绩效时序计算 | IC、分层回测、多空、换手、覆盖率（多 horizon：[1, 5, 20]） |
| 4.2 indicator | 统计指标与摘要 | 指标矩阵、评分卡、Dev4 评估摘要 |
| 4.3 label | 贴标与路由推荐 | 规则标签 + DeepSeek V4 Flash 语义标签 + 入库推荐 |

**优化**：
- 分位数计算使用 vectorbt，约 2.1x 加速
- IC 计算使用 vectorbt，约 1.4x 加速
- 前向收益率使用共享缓存目录 `database/cache/timeseries_forward_return/`

---

## 磁盘静态库接力

各模块通过标准化的目录结构和文件契约传递数据：

```
database/
├── tier1/
│   ├── gateway_pass_base/{candidate_id}/       ← Gateway 输出（通过）
│   ├── gateway_duplicated_base/{candidate_id}/ ← Gateway 输出（重复）
│   ├── assetization_raw_factor_base/{factor_id}/  ← Assetization 输出
│   └── purification_pure_factor_base/{factor_id}/ ← Purification 输出
├── tier2/
│   ├── 2_fix_base/{factor_id}/                 ← Evaluation 推荐目标
│   └── 2x_llm_mutation_base/{factor_id}/
├── tier3/
│   ├── 3a_core_production_base/{factor_id}/
│   ├── 3b_satellite_production_base/{factor_id}/
│   ├── 3c_feature_matierial_base/{factor_id}/
│   └── 3d_operation_storage_base/{factor_id}/
└── tier4/
    └── anti_sample_base/gateway/{candidate_id}/
```

每个因子目录包含：
- `afv.json` — 含各模块的阶段标记（**非** `candidate.json`）
- `data/{YYYY-MM-DD}.parquet` — 因子值数据，按日分片（**非** `data.parquet`）
- `manifest.json` — 完整元数据

---

## 配置管理

所有外部输入通过配置文件统一管理，不提供任何代码级默认值。

### 配置目录

```
all_configs/auto_factor_evaluation/
├── config.yaml                   ← 主配置
├── gateway_config.yaml
├── purification_config.yaml
├── evaluation_config.yaml
└── configs/
    ├── operator_whitelist.json
    ├── future_blacklist.json
    ├── complexity_weights.json
    ├── tag_policy.json
    ├── route_policy.json
    ├── label_registry.json
    └── .env                      ← DeepSeek V4 Flash API key
```

### 环境变量

```bash
export AFVCONFIG=/path/to/custom_profile    # 配置目录
```

### pipeline.py 单例

```python
from config_manager import ConfigManager

def _get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = ConfigManager()
    return _CONFIG
```

---

## 运行方式

```bash
# 全流程（Gateway → Assetization → Purification → Evaluation）
python pipeline.py --all

# 单阶段运行
python pipeline.py --gateway-only
python pipeline.py --assetization-only
python pipeline.py --purification-only
python pipeline.py --evaluation-only

# 指定 Worker 数
python pipeline.py --all --gw-workers 4 --ev-workers 8

# 禁用内存预加载
python pipeline.py --all --no-preload

# 指定配置目录
AFVCONFIG=/path/to/custom_config python pipeline.py --all
```

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.13 | 运行环境 |
| Anaconda | 环境管理 |
| multiprocessing | Worker 进程池 |
| pandas / polars | 因子计算引擎 |
| vectorbt | 绩效指标加速计算 |
| numpy | 向量化运算 |
| parquet | 因子值存储格式 |
| yaml | 配置文件解析 |
| DeepSeek V4 Flash | 语义标签生成 |
| Linux fork + COW | 进程内存预加载 |
