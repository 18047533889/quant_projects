# `backend/sql_pushdown` — SQL 子树下推

把因子计划里 **可编译为 SQL 的子树** 交给 DuckDB / ClickHouse 在数据库侧算完，
Python 只处理 fallback 分支或结果拼接。

## 协作者速览（约 5 分钟）

```
PlanNode 子树
  → emitter.py：生成 SELECT（CTE、窗口、JOIN）
  → sql_registry.py：算子 → SQL 模板注册表
  → executor.py：执行 SQL，返回 Arrow / LazyFrame
  → strict.py：禁止危险关键字、路径逃逸
```

- **DuckDB**：读 `data_access` 登记的 parquet（默认生产路径）
- **ClickHouse**：读已同步的长表 panel（可选，见 `storage/sources/clickhouse_source.py`）
- **覆盖率**：[`docs/sql_pushdown_coverage.md`](../../docs/sql_pushdown_coverage.md)

## 文件说明

| 文件 | 作用 |
|------|------|
| `emitter.py` | 核心：PlanNode → SQL 字符串（最大文件，按 tier 分批支持算子） |
| `executor.py` | 执行 SQL、绑定参数、与 `ExecutionContext` 对接 |
| `sql_registry.py` | canonical 算子名 → emit 函数；与 manifest 同步 |
| `duckdb_capabilities.py` | DuckDB 方言能力探测（窗口、UNNEST 等） |
| `clickhouse_capabilities.py` | ClickHouse 方言能力 |
| `strict.py` | SQL 安全：仅 SELECT、表名白名单 |
| `source_resolver.py` | 从 data_source 解析 scan 根路径 / 列名 |
| `plan_fixtures.py` | 测试用固定计划片段 |

## 入口

- `build_backend("duckdb_sql")` / `"clickhouse_sql"` → [`duckdb_pushdown_backend.py`](../duckdb_pushdown_backend.py)
- `build_backend("auto")` → [`hybrid_backend.py`](../hybrid_backend.py) 自动选 SQL 可编译子树

## 维护者

新增 SQL 下推算子：在 `sql_registry.py` 注册 emit 函数，并跑 `tests/backend_sql/` 与 parity 套件。
