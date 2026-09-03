# `runtime` — 运行时编排（详尽说明)

> **零基础读者**：请先读 **[`docs/FactorEngine完全指南.md`](../docs/FactorEngine完全指南.md)**。

本目录提供 **因子引擎对外主入口 `FactorEngine`**：组装 **`backend` + `data_source` + 可选 cache**，完成 **compile → execute**，并支持 **YAML 一键运行** 与 **多因子并行**。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**`FactorEngine`** —— **`compile`/`run`**、**`run_from_config`**、**`run_many`/`run_many_parallel`**、**`run_many_from_config`** / **`materialize_many_from_config`**；从 YAML 构建 **`backend`** + **`data_source`**（见 **`config.py`**）。
2. **性能与资源**：**`perf_config.py`** —— CSE、并行 worker、**`FACTOR_BACKTEST_EXECUTION_ENGINE`**（多资产回测内核）等 **环境变量** 的单一入口。
3. **从哪读**：下面 **§1** 生命周期；配置字段表见 **§2**。
4. **边界**：**回测业务**在 monorepo **`backtest_layer/single_asset_backtest/`**（不在本目录 `backtest/`）；本目录只管 **因子引擎主链路**。

---

## 1. `FactorEngine` 生命周期

### 构造

```python
FactorEngine(backend=..., data_source=..., cache=None)
```

- **`cache`**：通常为 **`CacheManager()`**（来自 `storage/cache.py`），由配置 **`engine.enable_cache`** 控制是否创建。

### 单次因子：`compile` + `run`

1. **`compile(factor)`**  
   - `Analyzer.lower(factor.expr)` → IR + `AnalysisResult`  
   - `Lowerer.to_logical_plan` → `PlanNode`  
   - `Optimizer.optimize` → 优化后 `PlanNode`  
2. **`run(factor)`**  
   - 调用 `compile`  
   - 构造 **`ExecutionContext(data_source=..., cache=...)`**  
   - **`backend.execute(plan, ctx)`** → **`result`**：一般为 **pandas Series（MultiIndex）**  
3. **返回 dict**：`factor`、`analysis`、`plan`、`result`。

### 多因子：`compile_many` / `run_many` / `run_many_parallel`

- **`compile_many(factors)`** → **`DAGPlan`**（含 **`shared_nodes`**），可选 CSE。  
- **`run_many`**：先算 **共享子式** 缓存，再算各因子根。  
- **`run_many_parallel`**：共享子式仍串行；根节点 **Joblib** `Parallel(..., backend="threading")`；需 **`factor-engine[parallel]`**（joblib）。

---

## 2. 配置文件入口

### [`config.py`](config.py)

| 数据类 | YAML 键 | 说明 |
|--------|---------|------|
| **`FactorDefinitionConfig`** | `factor` | **`name`**、**`expr`**（字符串）、`freq`、`universe`、`description` |
| **`DataSourceConfig`** | `data_source` | **`type`**（**`data_access`** / `composite` / legacy parquet）+ 其余进 **`options`** |
| **`BackendConfig`** | `backend` | **`type`**：默认 `pandas`，见 `backend/factory.py` |
| **`EngineConfig`** | `engine` | **`enable_cache`**：默认 True |
| **`FactorEngineConfig`** | 根 | 上述四块组合 |

- **`load_config(path)`**：`yaml.safe_load` → 校验 **`factor.name/expr`**、**`data_source.type`** 存在；路径字段经 [`workspace_paths`](../workspace_paths.py) 展开 `~` / 环境变量。

### 类方法

- **`FactorEngine.from_config(path)`** → **`(engine, factor, FactorEngineConfig)`**  
  - `build_backend(config.backend.type)`  
  - `build_data_source(config.data_source)`  
  - `parse_factor(config.factor.expr, name=..., ...)`  
- **`FactorEngine.run_from_config(path)`** → **`run` 结果 + 附带 `config` 对象**

---

## 3. [`perf_config.py`](perf_config.py)

- **`PerfConfig.from_env()`**：读取 **`FACTOR_ENGINE_DISABLE_CSE`**、**`FACTOR_ENGINE_MAX_WORKERS`**、chunk、内存上限、**`FACTOR_BACKTEST_EXECUTION_ENGINE`**（`python` / `numpy` / `numba` / `auto`，多资产回测执行内核；当 **`BacktestConfig.portfolio_execution_engine == "python"`** 时由 **`single_asset_backtest.runner`** 用此值替换请求）等。  
- 被 **`compile_many`**、**`run_many`**、**`run_many_parallel`** 与 **多资产回测** 使用。

---

## 4. [`exceptions.py`](exceptions.py)

- **`FactorEngineError`** 等，供上层捕获。

---

## 5. 已移除模块

- **`runtime/registry.py`** 已删除（无引用）。扩展注册请用 `api/operator_registry.build_dsl_allowlist()` 与 `cleaned_operators` 注册表。

---

## 6. [`real_data_factor_smoke.py`](real_data_factor_smoke.py)

- **真实 parquet** 冒烟：**`DatasetSpec`**、**`MultiParquetSeriesSource`** 等，供 `tests/test_real_data_factor_smoke.py` 与 `storage/factory` 的 **`multi_parquet`** 使用。

---

## 7. 与相邻目录

| 目录 | 关系 |
|------|------|
| `api` | `Factor` / `parse_factor` |
| `storage` | `build_data_source`、`CacheManager` |
| `backend` | `build_backend` |
| `planner` | `compile` 内部使用 |

---

## 8. 企业级服务层（2026-07）

| 模块 | 职责 |
|------|------|
| [`config_runtime.py`](config_runtime.py) | `resolve_run_kwargs` / `resolve_materialize_kwargs`；`PipelineConfigOverrides`；`config_*_batch_key` |
| [`warmup_service.py`](warmup_service.py) | `prepare_run_warmup()` — 扩窗与 trim |
| [`session_calendar.py`](session_calendar.py) | 分钟频 bar 级 session 日历 |
| [`intraday_aggregator.py`](intraday_aggregator.py) | `IntradayAggregator`（分钟→日频聚合）+ `IntradayFeatureCompiler.compute_many`（P0#5 一次 scan 批量算分钟特征） |
| [`lineage_service.py`](lineage_service.py) | 物化 lineage（`source_expr`、composite join） |
| [`materialize_service.py`](materialize_service.py) | `execute_materialize()` / `execute_materialize_from_resolved()` |
| [`dual_write_service.py`](dual_write_service.py) | staging → ClickHouse 双写 |

### 批量配置入口

- **`FactorEngine.run_many_from_config(paths, parallel=False, profile=..., pipeline_overrides=...)`** / **`run_many_from_config_parallel`**：按 scope + `config_run_batch_key` 子分组。
- **`FactorEngine.materialize_many_from_config(paths, batch_run=True, enable_cse=..., pipeline_overrides=...)`** / **`materialize_many_from_config_parallel`**：同 scope 共享 **`run_many`**（可并行根节点），再逐因子落盘。
- **`FactorEngine.materialize_incremental_many_from_config(paths, pipeline_overrides=...)`**：多 YAML 增量物化。
- **`pipeline.run_config_directory`**（`n_jobs=1`、≥2 配置）：自动委托上述批量 API（含 `--incremental` 与 CLI 覆盖；结果 JSON 含 `batched_engine: true`）。
- **`ResolvedMaterializeKwargs.to_engine_materialize_kwargs()`** / **`to_incremental_materialize_kwargs()`**：pipeline 与 engine 统一传参。

写目标见 [`storage/write_targets.py`](../storage/write_targets.py)：`local` / `staging` / `staging:<dataset>` / `clickhouse`。

---

## 9. 延伸阅读

- 根目录 [`README.md`](../README.md)「配置驱动运行」  
- [`storage/README.md`](../storage/README.md)  
- [`backend/README.md`](../backend/README.md)  
- [`examples/config_driven_factor.yaml`](../examples/config_driven_factor.yaml)  
