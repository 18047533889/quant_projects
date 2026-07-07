# factor_evaluation_alphapurify IT 对接说明

## 1. 这是什么

`factor_evaluation_alphapurify` 当前是一个同步执行的因子分析库，不是服务。

它已经实现的事情：
- 读取 YAML 配置或内存配置对象。
- 校验 `data_root`、目标因子目录、暴露因子目录和年份可用性。
- 把原始行情与因子 parquet 适配成分析面板。
- 对每个目标因子执行一整套暴露分析和因子分析。
- 为每个目标因子输出 1 份 JSON 和一组 PNG。
- 为整次运行输出 `run_summary.json`。
- 可选保存 `config_snapshot.yaml`。
- 可按配置切换 `Database`、`Exposures`、`FactorAnalyzer` 的历史版本实现。

它当前没有实现的事情：
- 没有 Job API。
- 没有 Query API。
- 没有 `run_id`。
- 没有 Manifest。
- 没有事件通知。
- 没有调用方权限控制。
- 没有统一错误码和服务级状态机。

## 2. 当前对外入口

当前建议只使用下面两个入口：

### Python API

- `pipeline.run_from_config(config_path)`
- `pipeline.run_pipeline(config)`

返回结构固定为：

```python
{
  "summary": {...},
  "results": [...],
  "output_root": "...",
  "run_summary_path": "...",
  "config_snapshot": "..." | None,
}
```

### CLI

```bash
python factor_layer/factor_evaluation_alphapurify/run_from_config.py \
  factor_layer/factor_evaluation_alphapurify/examples/run_config.example.yaml
```

CLI 最终会打印：
- `output_root`
- `run_summary_path`
- `summary`
- `config_snapshot`

## 3. 上游怎么接

上游只需要准备两类输入。

### 数据目录

`data_root` 下至少要有：

```text
data_root/
├── daily_market_summary/
│   └── daily_market_summary_<year>.parquet
└── factors/
    └── <factor_name>/
        └── year=<year>/
            └── *.parquet
```

当前代码默认依赖这些语义：
- 行情目录是 `daily_market_summary/`
- 因子目录是 `factors/<factor_name>/year=YYYY/`
- 目标因子和暴露因子都按目录名指定

### 运行配置

最小配置字段：

```yaml
data_root: /abs/path/to/data_root
output_root: /abs/path/to/output_root
year: 2025
min_symbols_per_day: 10
target_factors:
  - factor_a
exposure_factors:
  - factor_x
  - factor_y
save_config_snapshot: true
```

可选扩展字段：
- `database_module`
- `exposures_module`
- `factor_analyzer_module`

这些字段用于切换到 `history_module` 下的历史实现做回归或对照。

## 4. 下游怎么接

下游不要直接扫整棵输出目录，建议按下面顺序消费。

### 第一步：读 run_summary.json

这是整次运行的总索引，当前包含：
- `timestamp`
- `output_root`
- `year`
- `selected_target_factors`
- `selected_exposure_factors`
- `factors_total`
- `factors_success`
- `factors_partial_failed`
- `factors_failed`
- `json_files`

### 第二步：按 json_files 逐个读取单因子 JSON

单因子结果里重点关注：
- `factor_name`
- `status`
- `errors`
- `json_path`
- `images`
- `database`
- `exposures`
- `factor_analyzer`

当前单因子 `status` 语义：
- `success`：该因子完整成功
- `partial_failed`：部分子阶段失败，但仍产出了部分结果
- `failed`：该因子失败

### 第三步：按 images 字段或标准目录找图

标准目录结构：

```text
<output_root>/
├── run_summary.json
├── config_snapshot.yaml
├── json/
│   └── <factor>.json
└── images/
    └── <factor>/
        ├── portfolio_exposures/
        ├── pure_exposures/
        └── factor_analyzer/
```

## 5. 当前模块的边界

当前模块负责：
- 数据适配
- 因子暴露分析
- 因子统计分析
- 结果落盘

当前模块不负责：
- 任务提交与排队
- 任务查询
- 用户身份和权限
- 统一服务错误语义
- 跨系统通知
- 产物注册到统一元数据中心

## 6. 如果要包装成符合契约的服务，建议怎么做

最简单的做法不是改写分析核心，而是在外面包一层很薄的 service adapter，把当前 `run_from_config` 和 `run_pipeline` 当成执行引擎。

### 建议的服务分层

第 1 层：Service Entry
- 对外只暴露服务接口，不让外部直接调 `pipeline.py` 内部函数。

第 2 层：Job 管理
- 生成 `run_id`。
- 落一份 job manifest。
- 记录 `submitted`、`running`、`succeeded`、`failed`。

第 3 层：Execution Adapter
- 把服务请求转成 `AlphaPurifyAdapterConfig`。
- 调用 `run_pipeline(config)`。
- 把返回值转换成统一服务响应。

第 4 层：Artifact Registry
- 把 `run_summary.json`、单因子 JSON、图片目录登记成 artifact 列表。

第 5 层：Query / Event
- 提供状态查询。
- 在任务结束时发事件或通知。

## 7. 最小服务化方案

如果 IT 现在就要接，最小实现建议是下面这套。

### A. Job API

提供一个提交接口，例如：

```text
POST /alphapurify/jobs
```

请求体至少包含：
- `requested_by`
- `data_root`
- `output_root`
- `year`
- `target_factors`
- `exposure_factors`
- `min_symbols_per_day`

返回至少包含：
- `run_id`
- `status`
- `submitted_at`

### B. Query API

至少补两个查询接口：

```text
GET /alphapurify/jobs/{run_id}
GET /alphapurify/jobs/{run_id}/artifacts
```

第一个查任务状态，第二个查产物清单。

### C. Manifest

每个任务至少落一份 manifest，建议字段：

```json
{
  "run_id": "...",
  "service": "factor_evaluation_alphapurify",
  "requested_by": "...",
  "status": "submitted|running|succeeded|failed",
  "submitted_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "request": {...},
  "summary": {...},
  "artifact_uris": [...],
  "error": {...}
}
```

### D. Artifact 契约

建议统一登记这几类产物：
- `run_summary.json`
- `json/<factor>.json`
- `images/<factor>/...`
- `config_snapshot.yaml`

### E. Error Contract

建议先把当前散落异常收敛成少量服务错误码：
- `CONFIG_INVALID`
- `DATA_ROOT_NOT_FOUND`
- `FACTOR_DIR_NOT_FOUND`
- `FACTOR_YEAR_NOT_FOUND`
- `PIPELINE_EXECUTION_FAILED`
- `ARTIFACT_EXPORT_FAILED`

## 8. 推荐的落地顺序

1. 保留现有 `pipeline.py` 不动，把它当执行内核。
2. 新增一个 service adapter，负责请求转配置、生成 `run_id`、保存 manifest。
3. 先做同步 Job API，再补 Query API。
4. 稳定后再加异步队列和事件通知。

## 9. IT 同事可以直接采用的结论

- 现在它能跑，而且能稳定产出 JSON、PNG 和 `run_summary.json`。
- 现在它还不是服务，不能直接按 Job/Query/Manifest 契约接入平台。
- 最低成本方案是外包一层 service adapter，不要重写分析核心。
- 上游按目录和 YAML 喂数据，下游按 `run_summary.json -> 单因子 JSON -> 图片目录` 消费。
- 真正的服务化补齐点只有 5 个：`run_id`、manifest、query、artifact registry、error/event 统一化。