# `docs` — 文档中心

> **零基础读者**：先读 **[`FactorEngine完全指南.md`](FactorEngine完全指南.md)**；HTTP 直接看 [`../service/README.md`](../service/README.md)。  
> **仓库**：https://github.com/HKUST-QUANT-SOCIETY/factor_engine · 读数 https://github.com/HKUST-QUANT-SOCIETY/data_access

**规范正文**以本目录 Markdown / JSON 为准；算子 **唯一合法写法** = `build_dsl_allowlist()` + `parse_expr`（见 [`算子与导入教程.md`](算子与导入教程.md)）。

### 协作者速览（约 5 分钟）

1. **仓库**：https://github.com/HKUST-QUANT-SOCIETY/factor_engine · 读数 https://github.com/HKUST-QUANT-SOCIETY/data_access
2. **完全不懂 factor_engine**：[`FactorEngine完全指南.md`](FactorEngine完全指南.md)（含 §6.5 HTTP）
3. **HTTP 服务完整说明**：[`../service/README.md`](../service/README.md)
4. **挖掘 / 投递**：[`miner_delivery_spec.md`](miner_delivery_spec.md) → [`算子与导入教程.md`](算子与导入教程.md) → [`dsl_operators_reference.md`](dsl_operators_reference.md)
5. **算子语义**：[`operators_semantics.md`](operators_semantics.md)
6. **字段命名**：[`canonical_data_fields.md`](canonical_data_fields.md)
7. **版本沿革**：[`changelog_shw.md`](changelog_shw.md)
8. **源码导读**：[`源码注释导读.md`](源码注释导读.md)

---

## 1. 必读（零基础与入门）

| 文档 | 用途 |
|------|------|
| **[`FactorEngine完全指南.md`](FactorEngine完全指南.md)** | 本模块总文档：架构、上手、HTTP、目录、角色表、FAQ |
| **[`../service/README.md`](../service/README.md)** | HTTP 服务完整使用（clone / 接口 / curl） |
| [`源码注释导读.md`](源码注释导读.md) | 读源码时的注释约定与模块索引 |
| [`算子与导入教程.md`](算子与导入教程.md) | import、DSL 写法、校验 CLI |
| https://github.com/HKUST-QUANT-SOCIETY/data_access | 读数层与其 HTTP `/v1/read` |

## 2. 必读（挖掘与投递）

| 文档 | 用途 |
|------|------|
| [`miner_delivery_spec.md`](miner_delivery_spec.md) | **disk.v1 投递 JSON 契约**（组员必读） |
| [`dsl_operators_reference.md`](dsl_operators_reference.md) | 白名单枚举、不可 parse 名与替代写法 |
| [`dsl_allowlist.json`](dsl_allowlist.json) | 机器可读白名单 |
| [`canonical_data_fields.md`](canonical_data_fields.md) | manifest 字段 ↔ parquet 列 |
| [`../api/mining_integration.py`](../api/mining_integration.py) | 投递校验 API、默认 data_source（含 A 股 preset） |
| https://github.com/HKUST-QUANT-SOCIETY/data_access/blob/main/config/datasets.yaml | 登记数据集 schema（读端契约） |
| [`mining_data_source_presets.json`](mining_data_source_presets.json) | mining preset 快照 |
| [`sql_pushdown_coverage.md`](sql_pushdown_coverage.md) | SQL 下推清单 |
| [`backend_coverage.md`](backend_coverage.md) | Backend 覆盖 |

---

## 3. 算子语义与 LLM

| 文档 | 用途 |
|------|------|
| [`operators_semantics.md`](operators_semantics.md) | 参数、列依赖、DSL 限制 |
| [`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) | 给模型的算子摘要 |
| [`factor_engine_llm_prompt.txt`](factor_engine_llm_prompt.txt) | 同上（txt 同步） |
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
| `storage/` | [`storage/README.md`](../storage/README.md) |
| `tests/` | [`tests/README.md`](../tests/README.md) |
| `examples/` | [`examples/README.md`](../examples/README.md) |
| `scripts/` | [`scripts/README.md`](../scripts/README.md) |

读数层文档：https://github.com/HKUST-QUANT-SOCIETY/data_access/tree/main/docs

---

## 6. 已移除文档（勿再引用）

2026-07 清理：`enterprise_factor_engine_roadmap.md` · `performance_platform_plan.md` · `STRUCTURE.md` · `lqtp_vs_factor_engine_operators.md` 等（内容已并入完全指南 / changelog）。

2026-06 整理时删除的旧 miner_* / operators_roadmap 等：见 git 历史，勿再链接。
