# `tests` — 测试套件

本目录为 **pytest** 单元与集成测试，覆盖 `expr` → `ir` → `planner` → `backend`（**cleaned_operators**）→ `runtime` 全链路。

### 协作者速览

1. **本目录在干什么**：行为规格 — 改 DSL / cleaned 算子 / backend 时先看或补这里。  
2. **怎么跑**：`cd factor_engine && PYTHONPATH=. pytest tests/ -q`  
3. **主测 cleaned**：`test_cleaned_operators_comprehensive.py`（~88 用例）、`test_cleaned_integration.py`  
4. **回测专项**：monorepo [`../../../backtest_layer/tests/`](../../../backtest_layer/tests/)

---

## 1. 运行方式

```bash
cd /path/to/factor_engine
PYTHONPATH=. pytest tests/ -q
```

- 部分用例 `pytest.importorskip`（polars、modin 等），未装依赖会 **跳过**。  
- 真实数据：`test_real_data_factor_smoke.py` 需 **`RUN_REAL_PARQUET_SMOKE=1`** 与 `MASSIVE_PARQUET_ROOT`（默认 `~/quant_projects/data/us_stock/cleaned_massive_data`）。  
- 本地 COS 镜像：`test_local_massive_smoke.py` 需 **`RUN_LOCAL_MASSIVE_SMOKE=1`** 与 `~/quant_projects/data/us_stock/massive_data/StockDailyBar`。

---

## 2. 按文件分类

| 文件 | 覆盖范围 |
|------|----------|
| `test_cleaned_operators_comprehensive.py` | DSL 白名单、IR 降级、alias、四则、代表性算子执行、边界 |
| `test_cleaned_integration.py` | cleaned 端到端冒烟 |
| `test_expr.py` | `ColumnRef` / `Literal` / `CleanedCall`、运算符重载 |
| `test_planner.py` | Lowerer、Optimizer、常量折叠 |
| `test_backend.py` | Backend 抽象与入口 |
| `test_pandas_backend.py` | Pandas + cleaned_bridge 主路径 |
| `test_polars_backend.py` | Polars 委托路径 |
| `test_pandas_compat.py` | Modin / pandas 兼容 |
| `test_dsl_parser.py` | `parse_expr` 白名单与语法 |
| `test_config_runtime.py` | YAML → `FactorEngineConfig` |
| `test_cse_run_many.py` | 多因子 CSE 与 `run_many` |
| `test_end_to_end.py` | 小样本端到端 |
| `test_factor_templates.py` | 多数据集模板参数化 |
| `test_pipeline.py` / `test_pipeline_cli.py` | AFV pipeline 集成 |
| `test_parquet_source.py` / `test_composite_source.py` | 存储与数据源（含 Symbol→Ticker 列名回退） |
| `test_cleaned_bridge.py` | cleaned_bridge `d→window` 重试逻辑 |
| `test_storage_factory_paths.py` / `test_workspace_paths.py` | 路径展开与 canonical 元数据 |
| `test_mining_integration.py` | 挖掘对接 API、白名单导出 |
| `test_canonical_us_stock_daily_bar.py` | US StockDailyBar OHLCV 注册表回归 |
| `test_real_data_factor_smoke.py` | 原始 massive parquet（可选） |

> **第 31 版起** 已删除 `test_operators_*.py`、`test_intraday_stub.py` 等旧 native-operator 单测；能力由 **`test_cleaned_operators_comprehensive.py`** 承接。

---

## 3. [`helpers.py`](helpers.py)

内存数据源、合成 MultiIndex 等夹具。

---

## 4. 编写新测试的建议

1. **新增 cleaned 算子**：在 `test_cleaned_operators_comprehensive.py` 增加最小 DSL + 执行用例。  
2. **DSL / 白名单变更**：更新 `test_dsl_parser.py`。  
3. **计划 / CSE 变更**：`test_cse_run_many.py`、`test_planner.py`。

---

## 5. 延伸阅读

- 根 [`README.md`](../README.md)「运行测试」  
- [`docs/算子与导入教程.md`](../docs/算子与导入教程.md)  
- [`docs/changelog_shw.md`](../docs/changelog_shw.md)
