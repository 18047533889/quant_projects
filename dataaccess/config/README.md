# dataaccess/config/ 说明

本目录核心文件：`datasets.yaml`。它是**团队读数据的唯一真源**。

> 磁盘路径：`dataaccess/config/`｜包内仍称 data_access 配置。

## 为什么要这份 YAML

业务代码只写数据集名，不硬编码绝对路径：

```python
from data_access import get_store
store = get_store()
store.read_frame("us_stocks_sip_day_aggs", columns=[...], time_range=(...))
```

路径、分区、时间列、标的列都在 YAML 里；写路径走 staging → publish。

## 字段说明

### 必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | `static` / `parametric` | 路径是否含运行时参数（如 `factor_id`） |
| `access_mode` | `published` / `namespaced` / `staging` | 三态权限模型（见下） |
| `layout` | `plain` / `hive` | 目录结构类型 |
| `time_column` | str | 数据集里"时间列"的真实列名（如 `align_time`、`datetime`） |
| `instrument_column` | str | 数据集里"标的列"的真实列名（如 `ticker`、`asset`） |

### 静态数据集（`kind: static`）

| 字段 | 说明 |
|---|---|
| `root` | 根目录绝对路径（可用 `${ENV_VAR}` 占位） |
| `glob` | 相对 root 的通配符，默认 `"**/*.parquet"` |

### 参数化数据集（`kind: parametric`）

| 字段 | 说明 |
|---|---|
| `root_template` | 含 `{参数名}` 的根路径模板 |
| `glob_template` | 含 `{参数名}` 的 glob 模板 |
| `params_schema` | `{参数名: 类型名}` 字典；类型名目前仅作文档，不强制校验 |

### 可选字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `hive_partitioning` | `layout == hive` 时为 true | 传给 DuckDB `read_parquet(hive_partitioning=)`；对 `year=YYYY/` 这种分区做剪枝时必开 |
| `union_by_name` | `false` | schema 演进安全；不同文件列集不完全相同时开（推荐开） |

## access_mode 三态

| 模式 | 读权限 | 写权限 | 路径规则 |
|---|---|---|---|
| `published` | 全员 | 仅经 `publish()` 晋升 | 固定路径，多人共读同一份 |
| `namespaced` | 仅本 `RUN_NAMESPACE` | 仅本 `RUN_NAMESPACE` | 路径自动拼 `${RUN_NAMESPACE}` |
| `staging` | 仅本 `RUN_NAMESPACE` | 仅本 `RUN_NAMESPACE` | 发布前暂存，通过 `publish()` 晋升到 `published` |

> 读 / 写 / `publish()` 均已落地；细则见 [用户使用手册](../docs/用户使用手册.md)。

## 环境变量

可以在 `root` / `root_template` 里用 `${VAR}` 或 `${VAR:-default}`（bash 风格默认值）：

- `MASSIVE_PARQUET_ROOT` —— 大数据落盘根，默认 `/home/yluel/share/projects/massive_parquet`
- `FACTOR_LAKE_ROOT` —— 因子湖根
- `QUANTSOCIETY_WORKSPACE_DATA_ROOT` —— workspace_data 根
- `${RUN_NAMESPACE}` —— 特殊占位，加载 YAML 时会替换为 `resolve_namespace()` 的值
- `QUANT_RUN_NAMESPACE` —— 用户显式设置的 namespace（影响 `${RUN_NAMESPACE}` 的值）

## 加新数据集的 checklist

1. 在 `datasets.yaml` 加一条
2. 在 `dataaccess/tests/fixtures/golden/` 放一个小样本（<1 MB）
3. 在 `dataaccess/tests/contract/test_store_golden.py` 加 golden 断言
4. 跑 `pytest dataaccess/tests/` 全绿
5. PR 描述里写明：数据来源、负责人、磁盘估算
