# `storage` — 数据源与缓存（详尽说明）

抽象 **「因子需要的数据从哪来」**：统一为按 **列名**、在 **时间 × 标的** MultiIndex 上提供 **Series/DataFrame**，供 **`backend`** 执行 `PlanNode` 时拉取。

> **数据读取**：`ParquetSource` 优先尝试可选包 ``data_access`` 批量读；**未安装时自动回退 pandas**（monorepo 默认路径）。列名 ``Symbol``/``Ticker`` 等可自动回退。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**`DataSource`** 实现类 + **`factory.build_data_source`**；**推荐** 经 **`DataAccessSource`** 读 `datasets.yaml` 登记数据集；legacy parquet 类保留兼容。
2. **和 YAML 的关系**：`data_source.type` + 其余键 → **`DataSourceConfig.options`** → **工厂** 选类并实例化（与 **`examples/configs/`** 模板一一对应）。
3. **读代码顺序**：**`datasource.py`**（接口）→ **`factory.py`** → 具体 **`*_source.py`**；物化见 **`materializer.py`** / **`write_targets.py`**。
4. **边界**：**不负责** 因子编译与 **`PlanNode` 执行**（那是 **`backend`**）。

---

## 1. 抽象与实现

### [`datasource.py`](datasource.py)

- **`DataSource`** 基类：定义 **`load`/`get_series`** 等接口（以源码为准）；**执行上下文** 通过 **`ExecutionContext`** 传入 `backend`。

### [`data_access_source.py`](data_access_source.py)

- **`DataAccessSource`**：**企业级默认读通道**，经 `data_access.get_store()` 读 `datasets.yaml` 登记数据集；支持 `params`/`kind`（参数化数据集）、`normalize_timestamp`、`instrument_filter`。

### [`factory.py`](factory.py)

| `data_source.type` | 类 |
|--------------------|-----|
| **`data_access`** | **`DataAccessSource(**options)`** |
| `composite` | `CompositeDataSource` |
| `clickhouse` | `ClickHouseSource` |
| `parquet` | `ParquetSource(**options)` |
| `parquet_kline` | `KlineParquetSource(**options)` |
| `multi_parquet` / `cleaned_parquet` | legacy parquet 源 |

- **`options`**：`YAML` 里除 **`type`** 外的全部键值。

### [`kline_parquet_source.py`](kline_parquet_source.py)（legacy）

- **`KlineParquetSource`**：直连 K 线 parquet 目录；新配置请优先 **`data_access`** + `us_stocks_sip_day_aggs`。

### [`parquet_source.py`](parquet_source.py)（legacy）

- **`ParquetSource`**：单文件或简单目录的 **通用 parquet**；`type: parquet` / `multi_parquet`。

---

## 2. [`cache.py`](cache.py)

- **`CacheManager`**：**列级** 缓存，避免同一列在多因子或多节点中重复 IO。  
- 由 **`FactorEngine`** 构造时传入 **`ExecutionContext`**；**`enable_cache: false`** 时不创建。

---

## 3. [`materializer.py`](materializer.py) / 写目标 / schema

### [`materializer.py`](materializer.py) — `ParquetMaterializer`

- 因子 **MultiIndex Series → 长表** → 按 **`year=YYYY/data.parquet`** 分区 upsert + SQLite **`_catalog.sqlite`** 水位线。
- **`write_target`**：`local` / `staging` / `clickhouse` / `both`；staging 经 **`resolve_write_target("staging")`** upsert 到 `data_access`。
- **`defer_watermark`**：CH 双写成功后再提交 catalog（见 **`runtime/dual_write_service.py`**）。

### [`write_targets.py`](write_targets.py) — `FactorWriteTarget`

| 类 / 函数 | 说明 |
|-----------|------|
| `LocalParquetWriteTarget` | 本地因子湖分区 Parquet |
| `StagingWriteTarget` | `data_access` upsert → `factor_lake_staging` |
| `ClickHouseWriteTarget` | CH ReplacingMergeTree；`write_factor_series()` 供双写 |
| `resolve_write_target(name, ...)` | `local` / `staging` / `staging:<dataset>` / `clickhouse` |

公开 export：`from storage import resolve_write_target, LocalParquetWriteTarget, ...`

### [`factor_schema.py`](factor_schema.py) / [`schema_migration.py`](schema_migration.py)

- 因子湖长表 metadata 列契约（`calc_time`、`factor_version`、`is_valid`、`invalid_reason` 等）。
- CLI：`scripts/migrate_factor_lake_schema.py --lake-root ... --apply`

### [`result_store.py`](result_store.py)

- 中间结果持久化扩展点。

---

## 4. 与 YAML 的对应关系

根目录 [`README.md`](../README.md)「配置驱动运行」中的 **`data_source`** 示例：

- **`data_access`**：`dataset`、`fields`、`start_date`/`end_date`、`params`/`kind`（参数化数据集）  
- **`composite`**：`anchor`、`sources`、`joins`（非 anchor 源默认 `asof_backward`）  
- **legacy**：`parquet_kline` / `multi_parquet` — 仅调试或无 registry 时使用  
- 字段语义见 [`docs/massive_parquet_data_dictionary.md`](../docs/massive_parquet_data_dictionary.md) 与 [`data_access/config/datasets.yaml`](../../data_access/config/datasets.yaml)

---

## 5. 测试

- `tests/test_config_runtime.py`  
- `tests/test_factor_templates.py`  
- `tests/test_enterprise_p0.py` … `test_enterprise_p6.py`（写目标 / 批量 config / 物化 batch / Pipeline 覆盖）  
- `tests/test_real_data_factor_smoke.py`（需数据与环境变量）

---

## 6. 延伸阅读

- [`runtime/README.md`](../runtime/README.md)  
- [`backend/README.md`](../backend/README.md)  
- [`examples/configs/README.md`](../examples/configs/README.md)  
- [`docs/enterprise_factor_engine_roadmap.md`](../docs/enterprise_factor_engine_roadmap.md)
