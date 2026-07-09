# 代码目录结构说明

> 2026-07 重组：按职责分子包，**旧 import 路径保留兼容 shim**（根目录 `pipeline.py`、`runtime/dq_gates.py`、`storage/materializer.py` 等）。

## factor_engine 顶层

```
factor_engine/
├── api/              # DSL 公开 API（Factor、col、算子工厂）
├── expr/ ir/         # 表达式 AST → IR
├── planner/          # 逻辑/物理计划、CSE、依赖图
├── backend/          # 执行后端（pandas / polars / sql_pushdown/）
├── cleaned_operators/# 算子实现（按 domain 分子目录）
├── cache/            # L0–L3 执行期缓存
├── runtime/          # 引擎编排（见下）
├── storage/          # 读写与物化（见下）
├── util/             # 日志、工作区路径
├── scripts/          # 维护/文档生成 CLI
├── tests/            # 测试（按域分子目录，见下）
├── run_pipeline.py   # 主 CLI 入口
├── pipeline.py       # shim → runtime.pipeline.batch
└── pipeline_event.py # shim → runtime.pipeline.event
```

## runtime/

| 子目录 | 职责 | 典型模块 |
|--------|------|----------|
| `pipeline/` | 配置批跑、数据事件 | `batch.py`, `event.py` |
| `reconcile/` | 双写对账、快照 | `dual_write_*`, `snapshot_reconcile` |
| `quality/` | DQ、input_dq、PIT | `dq_gates`, `input_dq`, `pit_audit` |
| （根） | 核心引擎 | `engine.py`, `config.py`, `incremental*.py`, `materialize_service.py` |

## storage/

| 子目录 | 职责 | 典型模块 |
|--------|------|----------|
| `sources/` | 各类 DataSource | `data_access_source`, `composite_source`, `read_session` |
| `materialize/` | 落盘与写目标 | `materializer`, `write_targets`, `factor_matrix_materializer` |
| （根） | 目录/契约 | `catalog`, `factory`, `result_store`, `partition_policy` |

## tests/

| 子目录 | 覆盖 |
|--------|------|
| `backend/` | Polars/SQL/ClickHouse 后端 |
| `runtime/` | pipeline、增量、DQ、phase 平台 |
| `storage/` | 数据源、物化、分区 |
| `planner/` | 计划、CSE、依赖图 |
| `integration/` | enterprise、golden、mining |
| `operators/` | 算子语义、causal、phase ops |
| `util/` | workspace_paths 等 |
| `perf/` | 性能 smoke |
| `fixtures/` | golden 数据 |

## data_access

```
data_access/
├── clickhouse/       # CH 读写（panel.py, write.py）
├── config/           # datasets.yaml
├── scripts/          # stats 刷新、bucket 迁移规划
├── tests/unit|contract|concurrency/
└── store.py          # 主入口（尚未拆分，后续可抽 core/）
```

根目录 `clickhouse_write.py` / `clickhouse_panel.py` 为 **shim**，指向 `clickhouse/`。

## 迁移原则

1. **新代码**优先写子包路径（如 `from runtime.pipeline import run_data_event`）。
2. **旧路径**通过 shim 保留，避免大规模改 import。
3. 下一步可选：`backend/pandas|polars/`、`storage/catalog/`、`tests/` 细分子目录。
