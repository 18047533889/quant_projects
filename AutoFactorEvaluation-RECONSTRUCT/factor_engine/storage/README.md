# `storage` — 数据源与缓存（详尽说明）

抽象 **「因子需要的数据从哪来」**：统一为按 **列名**、在 **时间 × 标的** MultiIndex 上提供 **Series/DataFrame**，供 **`backend`** 执行 `PlanNode` 时拉取。

> **数据读取**：`ParquetSource` 优先尝试可选包 ``data_access`` 批量读；**未安装时自动回退 pandas**（monorepo 默认路径）。列名 ``Symbol``/``Ticker`` 等可自动回退。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**`DataSource`** 实现类（Parquet K 线、通用 parquet、multi_parquet 等）+ **`factory.build_data_source`**；**`CacheManager`** 给 **`ExecutionContext`** 做列级缓存。
2. **和 YAML 的关系**：`data_source.type` + 其余键 → **`DataSourceConfig.options`** → **工厂** 选类并实例化（与 **`examples/configs/`** 模板一一对应）。
3. **读代码顺序**：**`datasource.py`**（接口）→ **`factory.py`** → 具体 **`*_source.py`**。
4. **边界**：**不负责** 因子编译与 **`PlanNode` 执行**（那是 **`backend`**）。

---

## 1. 抽象与实现

### [`datasource.py`](datasource.py)

- **`DataSource`** 基类：定义 **`load`/`get_series`** 等接口（以源码为准）；**执行上下文** 通过 **`ExecutionContext`** 传入 `backend`。

### [`kline_parquet_source.py`](kline_parquet_source.py)

- **`KlineParquetSource`**：适用于 **日/分钟 K 线** 规范目录（如 `window_start`、**`instrument_column`**：`ticker`）。  
- 构造函数参数与 YAML 中 **`parquet_kline`** 的字段一一对应（见 `examples/configs/us_stocks_sip_*.yaml`）。

### [`parquet_source.py`](parquet_source.py)

- **`ParquetSource`**：单文件或简单目录的 **通用 parquet**；`type: parquet`。

### [`factory.py`](factory.py)

| `data_source.type` | 类 |
|--------------------|-----|
| `parquet` | `ParquetSource(**options)` |
| `parquet_kline` | `KlineParquetSource(**options)` |
| `multi_parquet` | **`MultiParquetSeriesSource`**（定义在 `runtime/real_data_factor_smoke.py`，多文件基本面等） |

- **`options`**：`YAML` 里除 **`type`** 外的全部键值。

---

## 2. [`cache.py`](cache.py)

- **`CacheManager`**：**列级** 缓存，避免同一列在多因子或多节点中重复 IO。  
- 由 **`FactorEngine`** 构造时传入 **`ExecutionContext`**；**`enable_cache: false`** 时不创建。

---

## 3. [`materializer.py`](materializer.py) / [`result_store.py`](result_store.py)

- **物化中间结果**、**持久化因子结果** 的扩展点；主路径可读源码确认当前使用程度。

---

## 4. 与 YAML 的对应关系

根目录 [`README.md`](../README.md)「配置驱动运行」中的 **`data_source`** 示例：

- **`parquet_kline`**：`root`、`instrument_column`、`timestamp_column`、`fields`、`max_files` 等。  
- **`multi_parquet`**：`root`、`timestamp_col`、`instrument_col`、`max_files` 等。  
- 字段语义见 [`docs/massive_parquet_data_dictionary.md`](../docs/massive_parquet_data_dictionary.md)。

---

## 5. 测试

- `tests/test_config_runtime.py`  
- `tests/test_factor_templates.py`  
- `tests/test_real_data_factor_smoke.py`（需数据与环境变量）

---

## 6. 延伸阅读

- [`runtime/README.md`](../runtime/README.md)  
- [`backend/README.md`](../backend/README.md)  
- [`examples/configs/README.md`](../examples/configs/README.md)  
