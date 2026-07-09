# `scripts` — 性能剖析与对比脚本（详尽说明）

本目录脚本 **不是** 包内可导入 API，用于本地 **性能诊断** 与 **可选依赖对比**。  
必须在 **`factor_engine` 仓库根** 执行，并设置 **`PYTHONPATH=.`**，或先 **`pip install -e .`**。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**非包内 API** 的 **性能剖析**（`profile_pandas_backend`）与 **可选依赖对比**（`bench_pandas_vs_modin`）；**不参与** `pytest` 默认集。
2. **什么时候用**：优化 **`kernels`/`pandas_backend`** 前后对比；评估 **Modin** 是否值得接 CI。
3. **和 `tests/` 的区别**：测试断言 **正确性**；脚本输出 **耗时/内存/热点**，需人工读。

---

## 1. [`profile_pandas_backend.py`](profile_pandas_backend.py)

- **目的**：对 **`PandasBackend`** 执行路径做 **cProfile**（或类似）热点分析，定位 **慢算子 / 重复分配**。  
- **典型用法**：

```bash
cd /path/to/factor_engine
PYTHONPATH=. python scripts/profile_pandas_backend.py
```

- **输出**：通常生成 **排序后的耗时统计** 或 **pstats** 可读报告；**具体参数与输出路径以脚本内 `argparse`/注释为准**（随版本可能调整）。

- **适用场景**：单因子或固定计划下的 **回归对比**（优化 `kernels` 前后）。

---

## 2. [`bench_pandas_vs_modin.py`](bench_pandas_vs_modin.py)

- **目的**：在 **同一套计划/数据** 上对比 **纯 pandas** 与 **Modin（`modin.pandas`）** 的 **耗时与峰值内存**（若脚本内实现）。  
- **依赖**：**`pip install "factor-engine[modin]"`**，并理解 **Modin 与 pandas 在 rolling/MultiIndex 上行为差异**（见 `tests/test_pandas_compat.py`）。

```bash
cd /path/to/factor_engine
PYTHONPATH=. python scripts/bench_pandas_vs_modin.py
```

- **注意**：Modin 需 **Ray 或 Dask** 后端时，环境需自行配置；脚本内可能有说明。

---

## 3. 挖掘 / 投递校验脚本

| 脚本 | 用途 |
|------|------|
| [`validate_delivery_formula.py`](validate_delivery_formula.py) | 美股 manifest / 公式 DSL 语法校验（`api/mining_integration`） |
| [`export_dsl_allowlist.py`](export_dsl_allowlist.py) | 导出 `docs/dsl_allowlist.json`（AFV 白名单对齐） |
| [`build_canonical_fields.py`](build_canonical_fields.py) | 生成 `canonical_data_fields.json`（在 `quant_projects` 根执行） |

```bash
cd ~/quant_projects/factor_engine
PYTHONPATH=. python scripts/validate_delivery_formula.py --manifest path/to/manifest.json
PYTHONPATH=. python scripts/export_dsl_allowlist.py
```

规范与命令说明见 [`docs/miner_delivery_spec.md`](../docs/miner_delivery_spec.md) §5.4。

---

## 4. 企业级运维 / 文档脚本

| 脚本 | 用途 |
|------|------|
| [`check_enterprise_docs.py`](check_enterprise_docs.py) | CI：roadmap Phase 14–16 标记 + LLM prompt 同步 |
| [`sync_factor_engine_llm_prompt_txt.py`](sync_factor_engine_llm_prompt_txt.py) | `factor_engine_llm_prompt.md` → `.txt` |
| [`migrate_factor_lake_schema.py`](migrate_factor_lake_schema.py) | 旧因子湖 parquet 补 metadata 列 |
| [`report_backend_coverage.py`](report_backend_coverage.py) | Polars/SQL 覆盖报告 + `sql_pushdown_coverage.md` |

```bash
cd ~/quant_projects/factor_engine
PYTHONPATH=. python scripts/check_enterprise_docs.py
PYTHONPATH=. python scripts/migrate_factor_lake_schema.py --lake-root /path/to/lake --dry-run
```

---

## 5. 与 `tests/` 的区别

| | `scripts/` | `tests/` |
|---|------------|----------|
| 目的 | 人工调优、可选长耗时 | CI、断言、回归 |
| 失败 | 不阻断发布 | 阻断 |

---

## 6. 延伸阅读

- [`backend/pandas_compat.py`](../backend/pandas_compat.py)  
- [`runtime/perf_config.py`](../runtime/perf_config.py)  
- [`docs/enterprise_factor_engine_roadmap.md`](../docs/enterprise_factor_engine_roadmap.md)  
- 根 [`README.md`](../README.md) 第 23 版关于 Modin/CSE 的说明  
