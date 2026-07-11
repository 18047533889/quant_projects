# `runtime` — 运行时编排（详尽说明)

本目录提供 **因子引擎对外主入口 `FactorEngine`**：组装 **`backend` + `data_source` + 可选 cache**，完成 **compile → execute**，并支持 **YAML 一键运行** 与 **多因子并行**。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**`FactorEngine`** —— **`compile`/`run`**、**`run_from_config`**、**`run_many`/`run_many_parallel`**；从 YAML 构建 **`backend`** + **`data_source`**（见 **`config.py`**）。
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
| **`DataSourceConfig`** | `data_source` | **`type`**（`parquet` / `parquet_kline` / `multi_parquet`）+ 其余进 **`options`** |
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

## 8. 延伸阅读

- 根目录 [`README.md`](../README.md)「配置驱动运行」  
- [`storage/README.md`](../storage/README.md)  
- [`backend/README.md`](../backend/README.md)  
- [`examples/config_driven_factor.yaml`](../examples/config_driven_factor.yaml)  
