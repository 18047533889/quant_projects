# `examples/configs` — 按数据集分类的 YAML 模板（详尽说明)

每个文件对应 **一种数据源形态**（与 **`storage/factory.build_data_source`** 的 **`type`** 一致），用于 **`FactorEngine.run_from_config(path)`** 或复制后改路径。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**按数据集分类的 YAML 模板**（日 K、分钟、基本面多表等），与根 **`README`「数据集列表」** 对齐。
2. **怎么用**：复制一份 → 改 **`data_source`** 与列映射 → 改 **`factor.expr`** → 单文件 **`run_from_config`** 或批量 **`run_many_from_config` / `materialize_many_from_config`**（见根 [`README.md`](../../README.md)「企业级批量配置」）。
3. **生产 profile**：YAML 顶行 `profile: prod` 或见 [`examples/profiles/prod.yaml`](../profiles/prod.yaml)。

---

## 1. 使用步骤

1. 复制一份 YAML 到任意路径。  
2. 修改 **`data_source.root`**（及 `instrument_column`、`timestamp_column`、`fields` 等）指向本地 parquet。路径支持 **`~`** 与 **`QUANT_PROJECTS_ROOT`** 环境变量；默认约定 **`~/quant_projects/data/...`**（各组员 home 不同，monorepo 目录名相同）。  
3. 修改 **`factor.expr`** 为合法 DSL（函数名在 [`docs/dsl_operators_reference.md`](../../docs/dsl_operators_reference.md) 白名单内）。  
4. 单文件运行：  
   `PYTHONPATH=. python -c "from runtime.engine import FactorEngine; print(FactorEngine.run_from_config('你的.yaml'))"`
5. 多文件批量（同数据源推荐）：  
   `FactorEngine.run_many_from_config(['a.yaml','b.yaml'])` 或 `FactorEngine.materialize_many_from_config(..., batch_run=True)`

---

## 2. 文件与数据集对应表

| YAML 文件 | 典型 `data_source.type` | 内容说明 |
|-----------|-------------------------|----------|
| `data_access_auto_smoke.yaml` | `data_access` | **推荐**：`backend: auto`（DuckDB 读 + hybrid 算） |
| `data_access_duckdb_sql_smoke.yaml` | `data_access` | DuckDB SQL 下推 smoke |
| `us_stocks_sip_day_aggs_v1.yaml` | `data_access` | 日 K 线（``us_stocks_sip_day_aggs``） |
| `day_aggs_v1_price_volume/*.yaml` | `data_access` | SIP 日 K 量价因子 |
| `day_aggs_v1_fundamental/*.yaml` | `composite` + `data_access` | SIP 日 K + 基本面 asof（balance_sheet / cash_flow / ratios 等） |

**Mining preset 对照**（Python：`default_mining_data_source_presets()`）：

| Preset 名 | 用途 |
|-----------|------|
| `us_stocks_sip_day_aggs` | SIP 日 K 单表 |
| `us_sip_day_ratios` | 日 K + 财务比率 composite |
| `us_sip_balance_sheet` / `us_sip_cash_flow` / `us_sip_income_statement` / `us_sip_floats` | 日 K + 各基本面 composite |
| `us_stocks_sip_quotes` / `us_stocks_sip_trades` | SIP tick |
| `us_polygon_floats` | Polygon 日线 + 流通股 |

完整 JSON：[`docs/mining_data_source_presets.json`](../docs/mining_data_source_presets.json)

| `fundamentals_*.yaml`（6 个） | `data_access` | cleaned massive 基本面 |
| `us_stocks_sip_minute_aggs_v1.yaml` | `data_access` | SIP 分钟 K（``us_stocks_sip_minute_aggs``） |
| `fundamentals_stocks_floats.yaml` | `data_access` | 流通股（``stocks_floats``） |
| `clickhouse_panel_rank.yaml` | `clickhouse` + `clickhouse_sql` | ClickHouse 长表 + SQL 因子下推 |
| `us_stocks_sip_quotes_v1.yaml` | `data_access` | SIP 报价（``us_stocks_sip_quotes``） |
| `us_stocks_sip_trades_v1.yaml` | `data_access` | SIP 逐笔成交（``us_stocks_sip_trades``） |

根目录 **`config_driven_factor.yaml`** / **`notebook_config_smoke.yaml`** 等 smoke 配置也已迁移至 ``data_access`` + ``us_stocks_sip_day_aggs``。

**字段含义** 见 [`docs/massive_parquet_data_dictionary.md`](../../docs/massive_parquet_data_dictionary.md)。

---

## 3. 与 `DataSourceConfig` 的映射

- YAML 中 **`data_source:`** 下 **`type:`** 以外的键 **全部** 进入 **`DataSourceConfig.options`**。  
- **`factory.py`** 用 **`type`** 选择类，用 **`**options`** 实例化。

---

## 4. 常见修改项

| 键 | 说明 |
|----|------|
| `dataset` | ``data_access`` 数据集名（见 ``datasets.yaml`` / ``docs/mining_data_source_presets.json``） |
| `start_date` / `end_date` | 行级时间过滤（``data_access`` 推荐） |
| `root` | 直连 parquet 根路径（legacy） |
| `max_files` | 限制扫描文件数（调试） |
| `instrument_column` / `timestamp_column` | K 线源列名映射 |
| `fields` | 逻辑名 → 文件列名 |

---

## 5. 延伸阅读

- [`storage/README.md`](../../storage/README.md)  
- [`runtime/config.py`](../../runtime/config.py)  
- 根 [`README.md`](../../README.md)「数据集列表」  
