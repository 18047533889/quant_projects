# `storage/sources` — 数据源实现

`DataSource` 的具体实现：从 parquet、`data_access`、ClickHouse 等读列，
统一输出 `(timestamp, instrument)` MultiIndex Series 或宽表 panel。

## 协作者速览

| 类 | 何时用 |
|----|--------|
| `DataAccessSource` | **默认生产**：`datasets.yaml` 登记数据集 |
| `ClickHouseSource` | ClickHouse 长表 panel |
| `LongTableSource` | 已是 long 格式的 parquet |
| `ParquetSource` | 遗留直读 parquet（新代码勿用） |
| `CompositeSource` | 多数据源合并 |

工厂入口：[`../factory.py`](../factory.py) 的 `build_data_source({"type": "data_access", ...})`。

## 文件说明

| 文件 | 作用 |
|------|------|
| `data_access_source.py` | `get_store().load_columns` + panel 缓存 |
| `clickhouse_source.py` | ClickHouse `read_columns` → Series |
| `long_table_source.py` | datetime/asset/value 长表 |
| `parquet_source.py` | 历史直读（allowlist 豁免） |
| `kline_parquet_source.py` | K 线目录布局 |
| `cleaned_parquet_source.py` | 清洗后固定布局 |
| `composite_source.py` | 多 source 按列名路由 |
| `staging_loader.py` | 从 `factor_lake_staging` 读因子 |

## 与 factor_engine 的契约

- `load_column(name)` / `load_columns(names)`：必须返回 MultiIndex Series
- `prefetch_columns` / `prefetch_panels`：批量预加载，减少重复 IO
- `dataset_axis_columns()`：返回 registry 时间列 / 标的列名（SQL 下推用）
