# factor_engine IT 对接说明

## 1. 这是什么

`factor_engine` 当前是一个可编译、可执行、可落盘的因子计算引擎，不是服务。

它现在已经实现了四类能力：
- 因子表达式构建与 DSL 解析。
- 因子编译、执行、多因子批量执行。
- 因子结果物化到 factor lake，并维护 catalog 与 watermark。
- 一层薄的 pipeline 编排，可统一输出 `run_summary.json`、单项结果 JSON 和配置快照。

它当前没有实现：
- 没有 HTTP / RPC 服务入口。
- 没有 Job API。
- 没有 Query API。
- 没有统一 `run_id`。
- 没有标准 Manifest。
- 没有事件通知。
- 没有调用方权限控制。

## 2. 现在已经实现的功能

### A. 表达式与 DSL

- 支持用 Python API 构造因子表达式。
- 支持从 YAML / DSL 字符串解析表达式。
- 支持算子白名单与算子注册表。
- 支持常用 Arithmetic / Logical / TS / CS / Group / Cleaning / Technical / Context 算子。
- DSL 白名单仅含 **`cleaned_operators` 已注册且可执行** 的算子；未实装算子不会进入 `parse_expr`。

### B. 编译与执行

- `Expr -> IR -> Logical Plan -> Optimized Plan` 完整编译链已经实现。
- 支持单因子执行。
- 支持多因子执行。
- 支持多因子公共子表达式消除 `CSE`。
- 支持 `joblib` 并行执行多因子根节点。
- 支持列级 cache。

### C. 数据源与后端

当前已支持的数据源类型：

| data_source.type | 用途 |
|---|---|
| **`data_access`** | **推荐**：`datasets.yaml` 登记数据集（含 `composite` 子源） |
| `composite` | 多源 asof/exact 对齐 |
| `clickhouse` | ClickHouse 长表 |
| `parquet_kline` | legacy K 线 parquet |
| `parquet` | legacy 通用 parquet |
| `multi_parquet` | legacy 多文件 parquet |
| `cleaned_parquet` | legacy cleaned parquet（`max_files` 调试） |

当前已支持的后端类型：

| backend.type | 用途 |
|---|---|
| `pandas` | 默认执行后端 |
| `pandas_modin` | 兼容 pandas API，底层走 modin |
| `polars` | 别名入口，实际委托 `pandas` + cleaned_operators |
| `polars_lazy` | 同上 |
| `debug` | 只看计划，不执行真实数据 |

### D. 结果物化

- 支持把因子执行结果落盘为分区 Parquet。
- 支持按年分区写入。
- 支持幂等 upsert。
- 支持原子写入。
- 支持 SQLite catalog 注册。
- 支持 watermark 更新。
- 支持 AST Hash 校验，防止“公式变了但 factor_id 没变”。

## 3. 当前建议对外入口

当前建议分两层使用：
- IT 或上层编排模块优先调用 `pipeline.py`。
- 研发内部若要直接控制执行细节，再调用 `runtime.engine.FactorEngine`。

也就是说：
- `pipeline.py` 是当前最接近“模块级交付接口”的入口。
- `runtime.engine.FactorEngine` 仍然是执行内核，不是最终服务接口。

### Pipeline API

推荐入口：
- `pipeline.run(config)`
- `pipeline.run_pipeline(config)`
- `pipeline.run_from_config(config_path)`
- `pipeline.run_config_directory(config_dir)`

### CLI Wrapper

当前已经提供一个很薄的根目录 CLI 包装：
- `run_pipeline.py`

单配置运行：

```bash
python factor_layer/factor_engine/run_pipeline.py config \
  /abs/path/to/factor.yaml \
  --output-root /abs/path/to/output_root
```

目录批量运行：

```bash
python factor_layer/factor_engine/run_pipeline.py config-dir \
  /abs/path/to/config_dir \
  --output-root /abs/path/to/output_root
```

CLI 当前只是对 `pipeline.run_from_config` 和 `pipeline.run_config_directory` 的薄包装。
它负责参数解析和 JSON 打印，不重新实现执行逻辑。

这层 pipeline 已经负责：
- 统一 `output_root`
- 统一 `run_summary.json`
- 统一单项结果 JSON
- 统一 `config_snapshot.yaml` 或 `config_snapshots/*.yaml`
- 将 runtime 的内部对象压缩成可序列化摘要

### Python API

若调用方明确需要直接控制引擎行为，可继续使用：
- `FactorEngine.from_config(config_path)`
- `FactorEngine.run_from_config(config_path)`
- `FactorEngine.materialize_from_config(config_path, ...)`
- `engine.run(factor)`
- `engine.materialize(factor, ...)`
- `engine.run_many(factors)`
- `engine.run_many_parallel(factors)`
- `FactorEngine.run_many_from_config(config_paths, pipeline_overrides=...)`
- `FactorEngine.run_many_from_config_parallel(config_paths, n_jobs=...)`
- `FactorEngine.materialize_many_from_config(config_paths, batch_run=True, pipeline_overrides=...)`
- `FactorEngine.materialize_many_from_config_parallel(config_paths, n_jobs=...)`
- `FactorEngine.materialize_incremental_many_from_config(config_paths, pipeline_overrides=...)`

目录批量（`run_config_directory`）在 `n_jobs=1` 且 ≥2 配置时会自动委托上述 batch API；CLI 参数（DQ、写目标、增量 since/end 等）统一封装为 `PipelineConfigOverrides` 传入，不再阻断 batch 路径。

### 手工脚本入口

当前仓库里有可直接运行的样例脚本：
- `run_pipeline.py`
- `examples/materialize_from_config.py`

注意：
- `run_pipeline.py` 是当前推荐的模块级 CLI 包装。
- 它仍然不是服务契约，只是对 pipeline 的命令行入口。
- `examples/*.py` 仍然更偏研发示例，不应视为稳定对外入口。

## 4. 返回结果怎么理解

### `pipeline.run_from_config(config_path)` / `pipeline.run_pipeline(config)` 返回

```python
{
  "summary": {...},
  "results": [...],
  "output_root": "...",
  "run_summary_path": "...",
  "config_snapshot": "..." | None,
}
```

其中：
- `summary`：整次运行摘要，包含成功失败数、结果 JSON 列表、输出目录。
- `results`：每个配置一条结果记录。
- `output_root`：本次 pipeline 输出根目录。
- `run_summary_path`：总摘要文件路径。
- `config_snapshot`：单配置运行时的配置快照路径。

### `pipeline.run_config_directory(config_dir)` 返回

```python
{
  "summary": {...},
  "results": [...],
  "output_root": "...",
  "run_summary_path": "...",
  "config_snapshots": [...],
}
```

### Pipeline 单项结果 JSON 当前包含

- `config_name`
- `factor_name`
- `status`
- `mode`，当前为 `run` 或 `materialize`
- `analysis`
- `plan`
- `result`
- `materialization`
- `error`
- `traceback`
- `result_json_path`

### `run(factor)` 返回

```python
{
  "factor": Factor(...),
  "analysis": AnalysisResult(...),
  "plan": PlanNode(...),
  "result": <pandas Series, MultiIndex(timestamp, instrument)>,
}
```

这表示：
- `factor`：因子定义。
- `analysis`：编译分析结果。
- `plan`：优化后的执行计划。
- `result`：真正的因子值结果。

### `run_from_config(config_path)` 返回

在 `run(...)` 基础上多一个：

```python
{
  ...,
  "config": FactorEngineConfig(...)
}
```

### `materialize(...)` / `materialize_from_config(...)` 返回

在 `run(...)` 基础上多一个：

```python
{
  ...,
  "materialization": {
    "factor_id": "...",
    "rows_written": 123,
    "partitions": [2024, 2025],
    "watermark": {...},
    "lake_root": "..."
  }
}
```

### `run_many(...)` / `run_many_parallel(...)` 返回

```python
{
  "results": {"factor_a": ..., "factor_b": ...},
  "dag": DAGPlan(...),
  "analyses": {"factor_a": ..., "factor_b": ...}
}
```

## 5. 上游怎么接

上游需要准备两部分输入：
- 因子定义。
- 数据源配置。

### A. 因子定义

最小字段：

```yaml
factor:
  name: smoke_rank_ts_mean_3
  expr: rank(ts_mean(col("close"), 3))
  freq: 1d
  universe: equities
  description: demo factor
```

支持两种接法：
- 直接构造 `Factor` 对象。
- 通过 YAML / DSL 字符串交给 `FactorEngine.from_config`。

### B. 数据源配置

**推荐**（`data_access` + 登记数据集）：

```yaml
data_source:
  type: data_access
  dataset: us_stocks_sip_day_aggs
  fields:
    close: close
  start_date: 2024-01-01
  end_date: 2024-01-31
```

Composite 示例（价量 anchor + 基本面 asof）见 `examples/profiles/us_sip_fundamental.yaml`。

Legacy 直连 parquet（仅调试）：

```yaml
data_source:
  type: parquet_kline
  root: /abs/path/to/day_aggs_v1
  instrument_column: ticker
  timestamp_column: window_start
  fields:
    close: close
  max_files: 5
```

上游需要保证：
- 数据路径真实存在。
- 列映射正确。
- 时间列和标的列可被标准化。
- 因子表达式里引用的字段，数据源真的能提供。

### C. backend / engine 配置

常见最小配置：

```yaml
backend:
  type: pandas

engine:
  enable_cache: true
```

### D. 如果要直接落盘

配置中额外增加：

```yaml
materialization:
  lake_root: /abs/path/to/factor_lake
  factor_id: my_factor_v1
  description: factor demo
  expression: rank(ts_mean(col("close"), 3))
```

### E. 如果要走当前模块级 pipeline

最简单接法是准备单个 YAML 或一个 YAML 目录，然后调用：
- `pipeline.run_from_config(config_path)`
- `pipeline.run_config_directory(config_dir)`

如果 YAML 里带 `materialization`，pipeline 会走 `materialize` 模式；否则默认走 `run` 模式。

## 6. 下游怎么接

下游接法现在分三类。

### A. 优先消费 pipeline 产物

如果调用方走的是 `pipeline.py`，建议优先消费 pipeline 输出目录，而不是直接解析 runtime 内部对象。

单配置运行默认目录语义：

```text
<output_root>/
├── run_summary.json
├── config_snapshot.yaml
└── results/
    └── <config_name>.json
```

目录批量运行默认目录语义：

```text
<output_root>/
├── run_summary.json
├── config_snapshots/
│   ├── <config_1>.yaml
│   └── <config_2>.yaml
└── results/
    ├── <config_1>.json
    └── <config_2>.json
```

下游读取顺序建议：
- 先读 `run_summary.json`
- 再按 `result_json_files` 逐个读取单项 JSON
- 若结果是 `materialize` 模式，再继续读取 factor lake

### B. 只消费 runtime 计算结果

直接使用 `run(...)` 或 `run_from_config(...)` 的返回值：
- 重点消费 `result`
- 调试时可看 `analysis` 和 `plan`

这里的 `result` 当前通常是：
- `pandas Series`
- `MultiIndex(timestamp, instrument)`

### C. 消费已物化因子

建议优先消费 factor lake，而不是依赖运行进程内存对象。

默认物化后目录拓扑：

```text
<lake_root>/
├── _catalog.sqlite
└── factors/
    └── <factor_id>/
        ├── year=2024/
        │   └── data.parquet
        └── year=2025/
            └── data.parquet
```

下游应重点对接：
- `_catalog.sqlite`：因子注册信息和 watermark
- `factors/<factor_id>/year=YYYY/data.parquet`：实际因子值

当前 catalog 已经能表达：
- `factor_id`
- `author`
- `frequency`
- `description`
- `ast_hash`
- `expression`
- `created_at`
- `start_date`
- `end_date`
- `last_updated`
- `row_count`

### C. 物化结果的关键语义

- 同一个 `factor_id` 如果 AST Hash 变了，会拒绝覆盖，要求升级版本号。
- 同一年分区重复写入时，按 `[datetime, asset]` 做幂等去重，保留最新值。
- 水位线会自动更新。

## 7. 当前模块边界

当前模块负责：
- 因子 DSL / API 表达式构建
- 编译与执行
- 数据源读取适配
- 多因子 CSE 和并行
- 结果物化到 factor lake
- catalog 与 watermark 维护
- 薄 pipeline 编排与统一摘要输出

当前模块不负责：
- 任务提交与排队
- 任务状态查询
- 统一服务响应协议
- 统一错误码体系
- 鉴权与调用方隔离
- 事件广播
- 平台级 artifact registry

## 8. 如果要包装成符合契约的服务，建议怎么做

最简单的做法不是改写 factor_engine，而是在外面包一层更薄的 service adapter，把 `pipeline.py` 当模块级编排入口，把 `FactorEngine` 当执行内核，把 `ParquetMaterializer` 当产物落盘器。

### 建议的服务分层

第 1 层：Spec Adapter
- 接收 YAML / JSON 请求。
- 校验 `factor`、`data_source`、`backend`、`engine`、`materialization`。
- 转成 `FactorEngineConfig`。

第 2 层：Execution Service
- 根据请求选择 `pipeline.run_from_config`、`pipeline.run_config_directory` 或更底层的 `run_many`、`materialize`。
- 生成 `run_id`。
- 记录开始时间、结束时间、状态。

第 3 层：Manifest / Artifact Layer
- 为每次运行落一份 manifest。
- 登记计算结果、物化路径、catalog 路径、watermark。

第 4 层：Query / Event
- 提供状态查询。
- 在任务成功或失败后发事件。

## 9. 最小服务化方案

如果 IT 现在就要接，最小实现建议如下。

### A. Query API

至少补两个只读接口：

```text
GET /factor-engine/operators
POST /factor-engine/validate-spec
```

含义：
- `operators`：返回当前支持的算子名和分类。
- `validate-spec`：只做 DSL / 配置合法性校验，不执行任务。

注意：
- 当前仓库里有 `operator_registry` 和 `parse_factor` 这些 Python 能力。
- 但 HTTP 级 Query API 现在还没有，需要外包 service adapter。

### B. Job API

建议把计算和物化分成两个任务入口，避免语义混淆：

```text
POST /factor-engine/jobs/compute
POST /factor-engine/jobs/materialize
```

请求体核心字段：
- `requested_by`
- `factor`
- `data_source`
- `backend`
- `engine`
- `materialization`，仅物化任务必填

返回至少包含：
- `run_id`
- `status`
- `submitted_at`

### C. Query Job Status

至少补下面两个查询接口：

```text
GET /factor-engine/jobs/{run_id}
GET /factor-engine/jobs/{run_id}/artifacts
```

### D. Manifest

每个任务至少落一份 manifest：

```json
{
  "run_id": "...",
  "service": "factor_engine",
  "requested_by": "...",
  "job_type": "compute|materialize",
  "status": "submitted|running|succeeded|failed",
  "submitted_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "request": {...},
  "factor_name": "...",
  "factor_id": "...",
  "artifact_uris": [...],
  "summary": {...},
  "error": {...}
}
```

### E. Artifact 契约

建议统一登记下面几类产物：
- 运行请求快照
- 编译分析摘要
- 执行计划摘要
- 结果预览或临时结果文件
- `_catalog.sqlite`
- `factors/<factor_id>/year=YYYY/data.parquet`
- watermark 摘要

### F. Error Contract

建议先收敛成少量服务错误码：
- `CONFIG_INVALID`
- `DSL_INVALID`
- `DATA_SOURCE_INVALID`
- `BACKEND_UNSUPPORTED`
- `EXECUTION_FAILED`
- `MATERIALIZATION_FAILED`
- `FACTOR_HASH_MISMATCH`

## 10. 推荐落地顺序

1. 保留现有 `FactorEngine` 和 `ParquetMaterializer` 不动。
2. 新增 service adapter，把 `run_from_config` / `materialize_from_config` 包起来。
3. 先做同步接口，先打通 `validate -> compute -> materialize -> query artifacts`。
4. 稳定后再加异步调度、事件通知和权限控制。

## 11. IT 同事可以直接采用的结论

- 现在 factor_engine 已经能稳定做因子编译、执行、批量执行和因子落盘。
- 现在 factor_engine 还已经有一层薄 pipeline，可统一输出 summary、结果 JSON 和配置快照。
- 现在它还不是服务，不能直接按 Job / Query / Manifest 契约接入平台。
- 上游应提供因子 DSL 和数据源配置；若走 pipeline，下游应先消费 pipeline 输出，再决定是否继续读取 factor lake。
- 最低成本服务化路线是外包一层 service adapter，不要重写 pipeline 或执行内核。
- 真正需要补齐的只有 6 个点：`run_id`、job status、manifest、query API、artifact registry、error/event 统一化。

## 12. 挖掘投递（disk.v1，与 §2 引擎 Job 不同）

自动化挖掘框架（QuantaAlpha、CogAlpha 等）产出的 **candidate manifest** 不走本章 Job API；契约与校验见：

| 文档 | 用途 |
|------|------|
| [`docs/miner_delivery_spec.md`](docs/miner_delivery_spec.md) | **组员必读**：`config.json` / `manifest.json` 字段、§5.4 投递前 CLI |
| [`docs/miner_delivery_spec.md`](docs/miner_delivery_spec.md) | disk.v1 投递契约 |
| [`docs/算子与导入教程.md`](docs/算子与导入教程.md) | factor_engine DSL 写法与校验 |
| [`docs/changelog_shw.md`](docs/changelog_shw.md) | 版本沿革（含第 31 版 cleaned 架构） |
| [`../data/factor_pools/candidate_pool/README.md`](../data/factor_pools/candidate_pool/README.md) | 本地落盘目录约定 |

**执行分工（摘要）：** A 股 / 美股 PV + `expression_type=dsl` → **统一 factor_engine**（仅数据源与 canonical 字段不同）。投递前：`check_manifest_fields.py` + `validate_delivery_formula.py`（不必安装 `platforms.lqtp`）。