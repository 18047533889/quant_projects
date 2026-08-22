# `util` — 通用工具

与业务无关的小工具：日志、路径解析。根目录也有同名模块（`logging_utils.py`、`workspace_paths.py`），
包内 `util/` 供 `from util.xxx import ...` 相对导入使用。

## 文件说明

| 文件 | 作用 |
|------|------|
| `logging_utils.py` | `configure_logging`、`get_logger`、`ProgressLogger`（无 tqdm 的进度条） |
| `workspace_paths.py` | `quant_projects_root()`、`resolve_path()`、`default_factor_lake_root()` |

## 环境变量

| 变量 | 用途 |
|------|------|
| `QUANT_PROJECTS_ROOT` | 覆盖 monorepo 根路径 |
| `FACTOR_LAKE_ROOT` | 因子湖根目录 |
| `QUANTSOCIETY_WORKSPACE_DATA_ROOT` | workspace 数据根 |
| `FACTOR_ENGINE_LOG_LEVEL` | 默认日志级别 |
