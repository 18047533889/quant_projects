# AutoFactorEvaluation 因子评估流水线

> **⚠️ 服务化持续运行模式（Service Mode）尚未实现。**
> 当前 `pipeline.py` 是唯一的编排入口，以**单次执行**或**分阶段执行**模式运行。
> 以下文档描述当前架构和未来规划。

---

## 1. 当前架构：pipeline.py 编排

`pipeline.py` 是当前唯一的编排入口，以多进程方式运行四阶段流水线。

### 1.1 架构总览

```
                          pipeline.py (orchestrator)
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
    Gateway Worker Pool    Assetization Worker     Purification Worker
    (多进程)               Pool (多进程)            Pool (多进程)
              │                   │                   │
              ▼                   ▼                   ▼
         tier0/               tier0/               tier0/
      gateway_temp/       assetization_temp/   purification_temp/
              │                   │                   │
              ▼                   ▼                   ▼
        Router → tier1/      Router → tier1/      Router → tier1/
                                              │
                                              ▼
                                     Evaluation Worker
                                     Pool (多进程)
                                              │
                                              ▼
                                         tier0/
                                      evaluation_temp/
                                              │
                                              ▼
                                        Router → tier2/3/4/
```

### 1.2 核心组件

| 组件 | 职责 | 运行模式 |
|------|------|---------|
| **Gateway Worker Pool** | 多进程并行审查候选因子 | `multiprocessing.Process` |
| **Assetization Worker Pool** | 多进程并行计算因子值 | `multiprocessing.Process` |
| **Purification Worker Pool** | 多进程并行纯化因子 | `multiprocessing.Process` |
| **Evaluation Worker Pool** | 多进程并行评估因子 | `multiprocessing.Process` |
| **Router**（每个阶段一个） | 阶段性输出 temp → 目标 base | 单线程串行 |
| **ConfigManager** | 统一管理所有路径/参数/白名单 | 单例，YAML 加载 |

### 1.3 数据流

```
candidate_pool/               ← 外部因子挖掘算法写入
    ↓ [Gateway Workers]
gateway_temp/ → [Router] → gateway_pass_base/   (或 duplicated_base / anti_sample_base)
    ↓ [Assetization Workers]
assetization_temp/ → [Router] → assetization_raw_factor_base/
    ↓ [Purification Workers]
purification_temp/ → [Router] → purification_pure_factor_base/
    ↓ [Evaluation Workers]
evaluation_temp/ → [Router] → tier2/3/4/ (根据 route_recommendation)
```

---

## 2. 命令行接口

### 2.1 CLI 参数

```bash
python pipeline.py [options]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gateway-only` | flag | false | 仅 Gateway 审查 + 路由 |
| `--assetization-only` | flag | false | 仅 Assetization 计算 + 路由 |
| `--purification-only` | flag | false | 仅 Purification 纯化 + 路由 |
| `--evaluation-only` | flag | false | 仅 Evaluation 评估 + 路由 |
| `--all` | flag | false | 全流程（带基准计时） |
| `--workers` | int | 0 | 全局 Worker 数（0=CPU-1） |
| `--gw-workers` | int | None | Gateway Worker 数 |
| `--as-workers` | int | None | Assetization Worker 数 |
| `--pu-workers` | int | None | Purification Worker 数 |
| `--ev-workers` | int | 4 | Evaluation Worker 数 |
| `--market-data` | str | None | 行情数据路径覆盖 |
| `--gateway-config` | str | None | Gateway 配置目录覆盖 |
| `--preload` | flag | true | 预加载行情到进程内存（fork 继承） |
| `--no-preload` | flag | false | 禁用进程内存预加载 |

### 2.2 使用示例

```bash
# 全流程（Gateway → Assetization → Purification → Evaluation）
python pipeline.py --all

# 仅 Gateway 审查
python pipeline.py --gateway-only

# 仅 Evaluation 评估
python pipeline.py --evaluation-only --ev-workers 8

# 指定 Gateway 配置目录
python pipeline.py --gateway-only --gateway-config /path/to/config_dir

# 禁用内存预加载
python pipeline.py --all --no-preload
```

### 2.3 全流程基准计时

执行 `--all` 时，pipeline.py 自动对各阶段计时并输出性能报告：

```
🏁 性能基准报告
  Gateway         12.3秒 ( 15.2%)  处理 5 个因子
  Assetization    45.6秒 ( 56.3%)  处理 3 个因子
  Purification    10.2秒 ( 12.6%)  处理 3 个因子
  Evaluation      12.9秒 ( 15.9%)  处理 3 个因子
  总耗时          81.0秒 (总计)
```

---

## 3. 配置策略

### 3.1 配置层次

所有外部输入统一管理，分为三层：

```
层级1: YAML 配置文件（静态，启动时加载）
  ├── all_configs/auto_factor_evaluation/config.yaml       ← 主配置
  ├── all_configs/auto_factor_evaluation/gateway_config.yaml  ← Gateway 参数
  ├── all_configs/auto_factor_evaluation/purification_config.yaml  ← Purification 参数
  ├── all_configs/auto_factor_evaluation/evaluation_config.yaml  ← Evaluation 参数
  └── all_configs/auto_factor_evaluation/configs/.env      ← DeepSeek V4 Flash API key

层级2: CLI 参数（覆盖 YAML 默认值）
  ├── --market-data PATH                    ← 行情数据根路径
  ├── --gateway-config PATH                 ← Gateway 配置目录
  └── --workers N                           ← 全局 Worker 数

层级3: 环境变量
  ├── AFVCONFIG                             ← 配置目录（替代默认 all_configs/...）
  └── DEEPSEEK_V4_FLASH_API_KEY             ← DeepSeek API key（从 .env 加载）
```

### 3.2 配置传递方式

```
启动 → ConfigManager() 加载 config.yaml → CLI 参数覆盖
     → 解析路径为绝对路径 → 分发给各 Worker 进程
```

---

## 4. 并发模型

### 4.1 多进程 Worker

每个阶段使用独立的 `multiprocessing.Process` Worker 池：

- **任务队列**：`multiprocessing.Queue`，主进程推送因子目录路径
- **结果队列**：`multiprocessing.Queue`，Worker 返回处理结果
- **并行度**：每阶段可独立配置（`--gw-workers`, `--as-workers` 等）
- **默认值**：0 = CPU 核数 - 1（Evaluation 默认 4）

### 4.2 进程内存预加载

`--preload`（默认开启）利用 Linux fork 的 COW（Copy-on-Write）特性：

```
1. 主进程 preload_market_data() 加载合并 parquet 到内存
   ↓
2. fork 子进程（继承内存映像，零 I/O）
   ↓
3. 每个子进程直接访问预加载的 DataFrame
```

优势：
- 子进程无需重复读取磁盘
- 合并 parquet（约 388MB，6.8x 压缩比）减少 I/O
- COW 保证子进程修改不互相影响

**缓存层级**：
```
市场数据原始 parquet (原始分片)
    ↓ [merge]
合并 parquet (约 388MB, 6.8x 压缩)
    ↓ [preload → fork]
进程内存 (COW 共享)
```

### 4.3 并行度控制

- 每个阶段独立配置 Worker 数
- 默认：`CPU 核数 - 1`
- 可通过 `--workers` 全局设定或各阶段独立覆盖

---

## 5. 目录结构预期

```
database/
├── real/
│   └── candidate_pool/          ← 外部写入点（不可由服务清理）
│   └── candidate_pool_record/  ← 空池归档（仅保留 config.json）
│       └── {campaign}/
│           └── {candidate}/     ← manifest.json + formula
├── tier0/                       ← 临时区（服务自动管理）
│   ├── gateway_temp/
│   ├── assetization_temp/
│   ├── purification_temp/
│   └── evaluation_temp/
├── tier1/                       ← 中间交换区（模块间传递）
│   ├── gateway_pass_base/
│   ├── gateway_duplicated_base/
│   ├── assetization_raw_factor_base/
│   └── purification_pure_factor_base/
├── tier2/                       ← Evaluation 路由目标
│   ├── 2_fix_base/
│   └── 2x_llm_mutation_base/
├── tier3/
│   ├── 3a_core_production_base/
│   ├── 3b_satellite_production_base/
│   ├── 3c_feature_matierial_base/
│   └── 3d_operation_storage_base/
├── tier4/
│   └── anti_sample_base/
├── cache/                       ← 计算缓存
│   ├── gateway_report_cache/
│   ├── market_data/             ← 合并 parquet 缓存（约 388MB）
│   └── timeseries_forward_return/  ← 前向收益率缓存
└── log/                         ← 运行日志
    ├── gateway/
    ├── assetization/
    ├── purification/
    └── evaluation/
```

---

## 6. 未来规划：服务化模式

> 以下为计划中的功能，**尚未实现**。

### 6.1 持续监控模式

计划将 pipeline.py 封装为 7×24 持续运行的后台服务，自动发现并处理候选因子池中的新因子：

```
每个阶段独立运行一个 Monitor Thread:

Monitor(gateway_pass_base)        Monitor(raw_factor_base)
  ├── 每隔 T 秒扫描目录              ├── 每隔 T 秒扫描目录
  ├── 发现新因子 → 加入队列          ├── 发现新因子 → 加入队列
  ├── Worker Pool 消费处理            ├── Worker Pool 消费处理
  ├── 处理完成 → 触发 Router          ├── 处理完成 → 触发 Router
  └── 更新已处理集合（防重复）        └── 更新已处理集合（防重复）
```

### 6.2 部署建议

| 维度 | 建议 |
|------|------|
| **运行环境** | Python 3.13+, Anaconda, 与行情数据 `lqtp_data/` 同机部署 |
| **资源要求** | 至少 16GB 内存（Evaluation 加载全量行情约需 8-10GB） |
| **并发模型** | 多进程 Worker (multiprocessing.Process) |
| **监控告警** | 建议接入 Prometheus + Grafana |
| **日志采集** | 各 Stage 日志按日期滚动，`database/log/{stage}/` |
