# INTERFACE

当前文件记录 factor_evaluation_alphapurify 已实现的模块级接口、输入输出和产物。

说明：
- 已实现的入口与返回结构写在对应章节。
- 没有做的功能在对应章节直接留白。

## 1. Module Responsibility

负责：
- 读取 YAML 配置或内存配置对象。
- 校验 `data_root`、目标因子目录、暴露因子目录和年份可用性。
- 将行情与因子 parquet 适配为分析面板。
- 对每个目标因子执行 `Database -> Exposures -> FactorAnalyzer` 分析链路。
- 输出单因子 JSON、图片目录和整次运行的 `run_summary.json`。
- 可选保存 `config_snapshot.yaml`。
- 允许通过模块选择配置切换历史实现版本。

不负责：
- HTTP / RPC 服务入口。
- 任务调度与异步队列。
- 平台级 Query API。
- 调用方鉴权与权限控制。

## 2. Public Service Entries

- `pipeline.run_pipeline`
- `pipeline.run_from_config`
- `run_from_config.py`
- `__init__.run_pipeline`
- `__init__.run_from_config`

## 3. Request / Response Contract

### `pipeline.run_pipeline(config)`

Request:
- `config.data_root`
- `config.output_root`
- `config.year` optional
- `config.min_symbols_per_day`
- `config.target_factors`
- `config.exposure_factors`
- `config.save_config_snapshot`
- `config.database_module` optional
- `config.exposures_module` optional
- `config.factor_analyzer_module` optional
- `config_path` optional

Response:
- `summary`
- `results`
- `output_root`
- `run_summary_path`
- `config_snapshot`

### `pipeline.run_from_config(config_path)`

Request:
- `config_path`

Response:
- `summary`
- `results`
- `output_root`
- `run_summary_path`
- `config_snapshot`

### `run_from_config.py <config_path>`

Request:
- 命令行参数 `config`

Stdout JSON:
- `output_root`
- `run_summary_path`
- `summary`
- `config_snapshot`

### `summary`

当前包含：
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

### `results[]`

单因子结果当前包含：
- `factor_name`
- `factor_dir`
- `exposure_dirs`
- `timestamp`
- `status`
- `database`
- `exposures`
- `factor_analyzer`
- `images`
- `errors`
- `json_path`
- `traceback` optional

`status` 当前语义：
- `success`
- `partial_failed`
- `failed`

## 4. Backing Python APIs

- `config.load_config`
- `module_selector.select_runtime_modules`
- `Database.DataBase.build_alphapurify_symbol_parquet_input`
- `Exposures.PortfolioExposures`
- `Exposures.PureExposures`
- `FactorAnalyzer.FactorAnalyzer`

## 5. Artifacts

- `<output_root>/run_summary.json`
- `<output_root>/config_snapshot.yaml`
- `<output_root>/json/<factor>.json`
- `<output_root>/images/<factor>/portfolio_exposures/*.png`
- `<output_root>/images/<factor>/pure_exposures/*.png`
- `<output_root>/images/<factor>/factor_analyzer/*.png`

## 6. Manifest Contract

## 7. Events

## 8. Error Contract

- `data_root` 不存在或目录结构不符合预期。
- 指定目标因子目录不存在。
- 指定暴露因子目录不存在。
- 指定年份没有可用 parquet。
- 运行时模块选择失败或历史模块导入失败。
- 单因子分析链路部分失败，结果记为 `partial_failed`。
- 单因子分析链路整体失败，结果记为 `failed` 并写入 `traceback`。
- 图片导出失败。

## 9. Upstream / Downstream

Upstream:
- `data_root/daily_market_summary/daily_market_summary_<year>.parquet`
- `data_root/factors/<factor_name>/year=<year>/*.parquet`
- alphapurify YAML 配置。
- 可选历史模块选择配置。

Downstream:
- `run_summary.json`
- `json/<factor>.json`
- `images/<factor>/...`
- 后续消费分析结果的服务包装、研究流程或报表模块。

## 10. Forbidden Direct Calls

## 11. Unresolved Issues