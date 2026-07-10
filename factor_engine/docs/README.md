# `docs` — 文档中心

> **零基础读者**：请先读 **[`FactorEngine完全指南.md`](FactorEngine完全指南.md)** — 总文档，涵盖架构、上手、目录地图、角色分流与全书目。

**规范正文**以本目录 Markdown / JSON 为准；算子 **唯一合法写法** = `build_dsl_allowlist()` + `parse_expr`（见 [`算子与导入教程.md`](算子与导入教程.md)）。

### 协作者速览（约 5 分钟）

1. **完全不懂 factor_engine**：[`FactorEngine完全指南.md`](FactorEngine完全指南.md) ← **从这里开始**
2. **挖掘 / 投递**：[`miner_delivery_spec.md`](miner_delivery_spec.md) → [`算子与导入教程.md`](算子与导入教程.md) → [`dsl_operators_reference.md`](dsl_operators_reference.md)
3. **算子语义**：[`operators_semantics.md`](operators_semantics.md)
4. **字段命名**：[`canonical_data_fields.md`](canonical_data_fields.md)
5. **版本沿革**：[`changelog_shw.md`](changelog_shw.md)
6. **文档检查**：`scripts/check_enterprise_docs.py`（LLM prompt md/txt 同步）
7. **源码导读**：[`源码注释导读.md`](源码注释导读.md) · 各包 [`README.md`](../README.md)（见下表）

---

## 1. 必读（零基础与入门）

| 文档 | 用途 |
|------|------|
| **[`FactorEngine完全指南.md`](FactorEngine完全指南.md)** | **总文档**：架构、上手、目录、角色表、FAQ、全书目 |
| [`源码注释导读.md`](源码注释导读.md) | 读源码时的注释约定与模块索引 |
| [`算子与导入教程.md`](算子与导入教程.md) | import、DSL 写法、校验 CLI |

## 2. 必读（挖掘与投递）

| 文档 | 用途 |
|------|------|
| [`miner_delivery_spec.md`](miner_delivery_spec.md) | **disk.v1 投递 JSON 契约**（组员必读） |
| [`dsl_operators_reference.md`](dsl_operators_reference.md) | 白名单枚举、不可 parse 名与替代写法 |
| [`dsl_allowlist.json`](dsl_allowlist.json) | 机器可读白名单（`export_dsl_allowlist.py` 生成） |
| [`canonical_data_fields.md`](canonical_data_fields.md) | manifest 字段 ↔ parquet 列 |
| [`../api/mining_integration.py`](../api/mining_integration.py) | 投递校验 API、默认 data_source |
| [`../data_access/config/datasets.yaml`](../../data_access/config/datasets.yaml) | 登记数据集 schema（读端契约） |
| [`mining_data_source_presets.json`](mining_data_source_presets.json) | mining preset 机器可读快照 |
| [`sql_pushdown_coverage.md`](sql_pushdown_coverage.md) | SQL 下推 canonical 清单（CI 自动生成） |
| [`backend_coverage.md`](backend_coverage.md) | Backend 三层覆盖（脚本自动生成） |

---

## 3. 算子语义与 LLM

| 文档 | 用途 |
|------|------|
| [`operators_semantics.md`](operators_semantics.md) | 参数、列依赖、DSL 限制 |
| [`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) | 给模型的算子摘要 |
| [`factor_engine_llm_prompt.txt`](factor_engine_llm_prompt.txt) | 同上（由 `sync_factor_engine_llm_prompt_txt.py` 从 md 生成） |
| [`cleaned_operators/operators_catalog.md`](../cleaned_operators/docs/operators_catalog.md) | 全量审计（含 stub） |

---

## 4. ADR 与数据字典

| 文档 | 用途 |
|------|------|
| [`adr_backtest_target_position.md`](adr_backtest_target_position.md) | 回测 `target_position` 协议 |
| [`adr_trade_when.md`](adr_trade_when.md) | `trade_when` 语义 |
| [`massive_parquet_data_dictionary.md`](massive_parquet_data_dictionary.md) | Massive 原始 parquet 字段 |
| [`changelog_shw.md`](changelog_shw.md) | 分版变更记录 |

---

## 5. 包内 README 导航

| 目录 | README |
|------|--------|
| `api/` | [`api/README.md`](../api/README.md) |
| `cleaned_operators/` | [`cleaned_operators/README.md`](../cleaned_operators/README.md) |
| `backend/` | [`backend/README.md`](../backend/README.md) |
| `runtime/` | [`runtime/README.md`](../runtime/README.md) |
| `storage/` | [`storage/README.md`](../storage/README.md)（含因子落盘） |
| `tests/` | [`tests/README.md`](../tests/README.md) |
| `examples/` | [`examples/README.md`](../examples/README.md) |
| `scripts/` | [`scripts/README.md`](../scripts/README.md) |
| `cache/` | [`cache/README.md`](../cache/README.md) |
| `util/` | [`util/README.md`](../util/README.md) |
| `backend/sql_pushdown/` | [`backend/sql_pushdown/README.md`](../backend/sql_pushdown/README.md) |
| `storage/sources/` | [`storage/sources/README.md`](../storage/sources/README.md) |
| `benchmarks/` | [`benchmarks/README.md`](../benchmarks/README.md) |
| `cleaned_operators/common/` 等 | 各子目录 [`README.md`](../cleaned_operators/README.md) |

回测长篇说明：monorepo [`../../backtest_layer/single_asset_backtest/README.md`](../../backtest_layer/single_asset_backtest/README.md)

---

## 6. 已移除文档（勿再引用）

2026-07 清理：删除已完成或过期的 **规划/对照/重复结构** 文档，内容已并入 [`FactorEngine完全指南.md`](FactorEngine完全指南.md)、各包 `README.md` 或 `changelog_shw.md`：

`enterprise_factor_engine_roadmap.md` · `performance_platform_plan.md` · `STRUCTURE.md` · `lqtp_vs_factor_engine_operators.md`（可用 `scripts/generate_lqtp_comparison.py` 重新生成）

2026-06 整理时删除：

`miner_api_import_guide.md` · `miner_ecosystem_integration.md` · `挖掘对接与本轮优化说明.md` · `operators_roadmap.md` · `huatai_factor_factory_operator_catalog.md` · `adr_huatai_factor_factory_operators.md` · `adr_context_benchmark.md` · `adr_ts_step_hump.md` · `factor_materializer.md` · `docs/_refs/`
