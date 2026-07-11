# `examples/configs` — 按数据集分类的 YAML 模板（详尽说明)

每个文件对应 **一种数据源形态**（与 **`storage/factory.build_data_source`** 的 **`type`** 一致），用于 **`FactorEngine.run_from_config(path)`** 或复制后改路径。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**按数据集分类的 YAML 模板**（日 K、分钟、基本面多表等），与根 **`README`「数据集列表」** 对齐。
2. **怎么用**：复制一份 → 改 **`data_source.root`** 与列映射 → 改 **`factor.expr`**（DSL 白名单）→ **`FactorEngine.run_from_config('路径')`**。
3. **字段含义**：见 **`docs/massive_parquet_data_dictionary.md`**；**`type`** 以外键全部进 **`options`**（见 **`storage/factory.py`**）。

---

## 1. 使用步骤

1. 复制一份 YAML 到任意路径。  
2. 修改 **`data_source.root`**（及 `instrument_column`、`timestamp_column`、`fields` 等）指向本地 parquet。路径支持 **`~`** 与 **`QUANT_PROJECTS_ROOT`** 环境变量；默认约定 **`~/quant_projects/data/...`**（各组员 home 不同，monorepo 目录名相同）。  
3. 修改 **`factor.expr`** 为合法 DSL（函数名在 [`docs/dsl_operators_reference.md`](../../docs/dsl_operators_reference.md) 白名单内）。  
4. 在仓库根执行：  
   `PYTHONPATH=. python -c "from runtime.engine import FactorEngine; print(FactorEngine.run_from_config('你的.yaml'))"`

---

## 2. 文件与数据集对应表

| YAML 文件 | 典型 `data_source.type` | 内容说明 |
|-----------|-------------------------|----------|
| `us_stocks_sip_day_aggs_v1.yaml` | `parquet_kline` | 日 K 线 |
| `us_stocks_sip_minute_aggs_v1.yaml` | `parquet_kline` | 分钟 K 线 |
| `us_stocks_sip_quotes_v1.yaml` | `parquet_kline` 或扩展 | 报价 |
| `us_stocks_sip_trades_v1.yaml` | 同上 | 逐笔成交 |
| `fundamentals_balance_sheet.yaml` | `multi_parquet` | 资产负债表 |
| `fundamentals_cash_flow_statement.yaml` | `multi_parquet` | 现金流量表 |
| `fundamentals_income_statement.yaml` | `multi_parquet` | 利润表 |
| `fundamentals_financials_ratios.yaml` | `multi_parquet` | 财务比率 |
| `fundamentals_short_interest.yaml` | `multi_parquet` | 融券兴趣 |
| `fundamentals_short_volume.yaml` | `multi_parquet` | 融券成交量 |
| `fundamentals_stocks_floats.yaml` | `multi_parquet` | 流通股 |

**字段含义** 见 [`docs/massive_parquet_data_dictionary.md`](../../docs/massive_parquet_data_dictionary.md)。

---

## 3. 与 `DataSourceConfig` 的映射

- YAML 中 **`data_source:`** 下 **`type:`** 以外的键 **全部** 进入 **`DataSourceConfig.options`**。  
- **`factory.py`** 用 **`type`** 选择类，用 **`**options`** 实例化。

---

## 4. 常见修改项

| 键 | 说明 |
|----|------|
| `root` | 数据根路径 |
| `max_files` | 限制扫描文件数（调试） |
| `instrument_column` / `timestamp_column` | K 线源列名映射 |
| `fields` | 逻辑名 → 文件列名 |

---

## 5. 延伸阅读

- [`storage/README.md`](../../storage/README.md)  
- [`runtime/config.py`](../../runtime/config.py)  
- 根 [`README.md`](../../README.md)「数据集列表」  
