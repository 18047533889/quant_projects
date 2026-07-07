# INTERFACE

当前文件记录 factor_engine 已实现的模块级接口、输入输出和产物。

说明：
- 已实现的入口与返回结构写在对应章节。
- 没有做的功能在对应章节直接留白。

## 1. Module Responsibility

负责：
- 因子表达式与 DSL 解析。
- `Expr -> IR -> Logical Plan -> Optimized Plan` 编译链。
- 单因子执行、多因子执行、并行多因子执行。
- 因子结果物化到 factor lake，并维护 catalog 与 watermark。
- 通过 `pipeline.py` 输出统一的运行摘要、结果 JSON 和配置快照。

不负责：
- HTTP / RPC 服务入口。
- 异步调度与任务队列。
- 平台级 Query API。
- 调用方鉴权与配额控制。

## 2. Public Service Entries

- `pipeline.run`
- `pipeline.run_pipeline`
- `pipeline.run_from_config`
- `pipeline.run_config_directory`
- `run_pipeline.py config`
- `run_pipeline.py config-dir`
- `runtime.engine.FactorEngine.from_config`
- `runtime.engine.FactorEngine.run_from_config`
- `runtime.engine.FactorEngine.materialize_from_config`
- `runtime.engine.FactorEngine.run`
- `runtime.engine.FactorEngine.materialize`
- `runtime.engine.FactorEngine.run_many`
- `runtime.engine.FactorEngine.run_many_parallel`

## 3. Request / Response Contract

### `pipeline.run_from_config(config_path)`

Request:
- `config_path`
- `output_root` optional
- `materialize` optional
- `preview_rows` optional

Response:
- `summary`
- `results`
- `output_root`
- `run_summary_path`
- `config_snapshot`

### `pipeline.run_config_directory(config_dir)`

Request:
- `config_dir`
- `pattern` optional
- `output_root` optional
- `materialize` optional
- `preview_rows` optional
- `stop_on_error` optional

Response:
- `summary`
- `results`
- `output_root`
- `run_summary_path`
- `config_snapshot_root`

### `FactorEngine.run(factor)`

Request:
- `factor`

Response:
- `factor`
- `analysis`
- `plan`
- `result`

### `FactorEngine.run_from_config(config_path)`

Request:
- `config_path`

Response:
- `factor`
- `analysis`
- `plan`
- `result`
- `config`

### `FactorEngine.materialize_from_config(config_path)`

Request:
- `config_path`
- `lake_root` optional
- `factor_id` optional
- `author` optional
- `frequency` optional
- `description` optional
- `expression` optional

Response:
- `factor`
- `analysis`
- `plan`
- `result`
- `materialization`
- `config`

### `FactorEngine.run_many(factors)` / `FactorEngine.run_many_parallel(factors)`

Request:
- `factors`
- `enable_cse` optional
- `perf` optional
- `n_jobs` optional for `run_many_parallel`

Response:
- `results`
- `dag`
- `analyses`

## 4. Backing Python APIs

- `runtime.config.load_config`
- `runtime.engine.FactorEngine`
- `api.dsl_parser.parse_factor`
- `backend.factory.build_backend`
- `storage.factory.build_data_source`
- `storage.materializer.ParquetMaterializer`

## 5. Artifacts

- `<output_root>/run_summary.json`
- `<output_root>/config_snapshot.yaml`
- `<output_root>/config_snapshots/*.yaml`
- `<output_root>/results/*.json`
- `<lake_root>/_catalog.sqlite`
- `<lake_root>/factors/<factor_id>/year=<YYYY>/data.parquet`

## 6. Manifest Contract

## 7. Events

## 8. Error Contract

- 配置文件解析失败。
- 因子表达式解析或编译失败。
- `backend.type` 或 `data_source.type` 不支持。
- 数据路径不存在或字段映射错误。
- 执行结果不满足物化输入格式要求。
- `factor_id` 沿用但 AST Hash 不一致，物化拒绝覆盖。

## 9. Upstream / Downstream

Upstream:
- 因子 YAML / DSL 配置。
- parquet / multi_parquet / composite 数据源。
- backend 配置与 engine 配置。
- 可选 materialization 配置。

Downstream:
- pipeline 产出的 `run_summary.json` 与 `results/*.json`。
- factor lake 分区 Parquet。
- SQLite catalog 与 watermark。
- 后续消费 factor lake 的评估、研究与策略模块。

## 10. Forbidden Direct Calls

## 11. Unresolved Issues
