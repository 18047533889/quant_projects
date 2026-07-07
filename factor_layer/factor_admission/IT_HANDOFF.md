# factor_admission IT 对接说明

## 1. 这是什么

`factor_admission` 当前是一个准入决策与登记模块，不是服务。

它负责消费已经存在的 `factor_evaluation` 运行结果，生成准入结论，并把评估 run 与决策结果写入 catalog。

它现在已经实现的事情：
- 读取 YAML 配置。
- 校验目标 `factor_id` 与 `run_id` 的评估目录是否存在。
- 读取评估运行目录下的 `summary.json`。
- 按规则阈值或人工指定配置生成 `approved` / `rejected` 决策。
- 将评估摘要索引到 SQLite catalog。
- 将 admission 决策写入 SQLite catalog。
- 更新 `factor_admission_status`。
- 可选写出 `admission_decision.json`。
- 提供一层薄 pipeline，统一输出 `run_summary.json`、单项结果 JSON 和配置快照。

它当前没有实现的事情：
- 没有 HTTP / RPC 服务入口。
- 没有 Job API。
- 没有 Query API。
- 没有统一 `run_id`。
- 没有独立 Manifest。
- 没有事件通知。
- 没有调用方权限控制。

## 2. 当前建议对外入口

当前建议分两层使用：
- IT 或上层编排模块优先调用 `pipeline.py`。
- 若只需要复用纯决策逻辑，再调用 `admission.py` 或 `config_runner.py`。

### Pipeline API

推荐入口：
- `pipeline.run(config)`
- `pipeline.run_pipeline(config)`
- `pipeline.run_from_config(config_path)`
- `pipeline.run_config_directory(config_dir)`

这层 pipeline 已经负责：
- 统一 `output_root`
- 统一 `run_summary.json`
- 统一单项结果 JSON
- 统一 `config_snapshot.yaml` 或 `config_snapshots/*.yaml`
- 将 admission 结果、catalog 路径、decision file 路径压缩成可序列化摘要

### CLI Wrapper

当前已经提供一个薄的根目录 CLI 包装：
- `run_pipeline.py`

单配置运行：

```bash
python factor_layer/factor_admission/run_pipeline.py config \
  /abs/path/to/factor_admission.yaml \
  --output-root /abs/path/to/output_root
```

目录批量运行：

```bash
python factor_layer/factor_admission/run_pipeline.py config-dir \
  /abs/path/to/config_dir \
  --output-root /abs/path/to/output_root
```

兼容的单配置 CLI 入口仍保留：
- `run_from_config.py`

它会继续打印：
- `factor_id`
- `run_id`
- `decision`

## 3. 上游怎么接

上游需要准备两类输入：
- admission 配置 YAML
- 已存在的 factor_evaluation 运行目录

### A. Admission 配置

最小配置示例：

```yaml
meta:
  factor_id: daily_quality_v1
  run_id: 20240501_120000

source:
  factor_lake_root: /abs/path/to/factor_lake
  evaluation_root: /abs/path/to/evaluations

decision:
  mode: rule_based
  decided_by: system
  policy_name: default_daily_v1
  primary_horizon: 1
  thresholds:
    min_rank_ic_mean: 0.03
    min_long_short_sharpe: 0.5

output:
  write_decision_file: true
```

### B. 上游评估目录契约

当前代码要求：

```text
<evaluation_root>/<factor_id>/<run_id>/summary.json
```

这份 `summary.json` 至少要能提供：
- `factor_id`
- `run_id`
- `primary_horizon`
- `summary`
- `config_hash`
- `created_at`

如果上游没有先完成 factor_evaluation，这个模块不能独立运行。

## 4. 下游怎么接

下游接法分三类。

### A. 优先消费 pipeline 产物

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

建议读取顺序：
- 先读 `run_summary.json`
- 再按 `result_json_files` 逐个读单项 JSON

### B. 消费 side effect 产物

当前真正业务产物仍然在评估目录与 factor lake：
- `<evaluation_root>/<factor_id>/<run_id>/admission_decision.json`
- `<factor_lake_root>/_catalog.sqlite`

### C. 消费 catalog

当前 SQLite catalog 中已实现的核心表有：
- `factor_evaluation_runs`
- `factor_evaluation_summary`
- `factor_admission_status`
- `factor_admission_decisions`

下游如果要查“当前因子是否可用”，应优先读取：
- `factor_admission_status`

如果要审计某次决策，应读取：
- `factor_admission_decisions`
- `factor_evaluation_summary`

## 5. 当前返回结构怎么理解

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
- `factor_id`
- `run_id`
- `status`
- `decision`
- `approved`
- `decision_id`
- `primary_horizon`
- `reason`
- `diagnostics`
- `factor_status`
- `catalog_db_path`
- `evaluation_run_dir`
- `summary_path`
- `decision_file`
- `error`
- `traceback`

## 6. 当前模块边界

当前模块负责：
- 读取和校验 factor_evaluation 结果
- 生成准入决策
- 将 run 和 decision 索引到 catalog
- 更新因子最新状态
- 可选写 decision 文件
- 薄 pipeline 编排与统一摘要输出

当前模块不负责：
- 因子计算
- 评估指标生成
- 任务调度与排队
- 统一服务错误码
- 鉴权与调用方隔离
- 事件广播
- 平台级 artifact registry

## 7. 如果要包装成符合契约的服务，建议怎么做

最简单的做法不是改写 factor_admission，而是在外面包一层更薄的 service adapter，把 `pipeline.py` 当模块级编排入口。

建议的服务分层：

第 1 层：Request Adapter
- 接收 YAML / JSON 请求
- 校验 `factor_id`、`run_id`、路径和 decision 配置
- 转成 `FactorAdmissionConfig`

第 2 层：Execution Service
- 调用 `pipeline.run_from_config` 或 `pipeline.run_config_directory`
- 生成外层 `run_id`
- 记录开始时间、结束时间、状态

第 3 层：Query / Catalog Adapter
- 查询某次 admission 结果
- 查询 `factor_admission_status`
- 查询 `factor_admission_decisions`

## 8. IT 同事可以直接采用的结论

- 现在 factor_admission 已经能稳定做 admission 决策，并把结果写入 catalog。
- 现在它还不是服务，不能直接按 Job / Query / Manifest 契约接入平台。
- 若当前只需要模块级集成，优先调用 `pipeline.py` 或 `run_pipeline.py`。
- 上游必须先有 factor_evaluation 运行目录；这个模块不会自己计算评估指标。
- 最低成本服务化路线是在 pipeline 之上包 service adapter，不要重写 admission 逻辑。