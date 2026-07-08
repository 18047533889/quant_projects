# `docs` — 文档中心

**规范正文**以本目录 Markdown / JSON 为准；算子 **唯一合法写法** = `build_dsl_allowlist()` + `parse_expr`（见 [`算子与导入教程.md`](算子与导入教程.md)）。

### 协作者速览（约 5 分钟）

1. **挖掘 / 投递**：[`miner_delivery_spec.md`](miner_delivery_spec.md) → [`算子与导入教程.md`](算子与导入教程.md) → [`dsl_operators_reference.md`](dsl_operators_reference.md)
2. **算子语义**：[`operators_semantics.md`](operators_semantics.md)
3. **字段命名**：[`canonical_data_fields.md`](canonical_data_fields.md)
4. **版本沿革**：[`changelog_shw.md`](changelog_shw.md)（第 1–30 版为历史架构描述）
5. **源码导读**：各包 [`README.md`](../README.md)（见下表）

---

## 1. 必读（挖掘与投递）

| 文档 | 用途 |
|------|------|
| [`miner_delivery_spec.md`](miner_delivery_spec.md) | **disk.v1 投递 JSON 契约**（组员必读） |
| [`算子与导入教程.md`](算子与导入教程.md) | import、factor_engine DSL 写法、校验 CLI |
| [`dsl_operators_reference.md`](dsl_operators_reference.md) | 白名单枚举、不可 parse 名与替代写法 |
| [`dsl_allowlist.json`](dsl_allowlist.json) | 机器可读白名单（`export_dsl_allowlist.py` 生成） |
| [`canonical_data_fields.md`](canonical_data_fields.md) | manifest 字段 ↔ parquet 列 |
| [`../api/mining_integration.py`](../api/mining_integration.py) | 投递校验 API、默认 data_source |
| [`../data_access/config/datasets.yaml`](../../data_access/config/datasets.yaml) | 登记数据集 schema（读端契约） |
| [`mining_data_source_presets.json`](mining_data_source_presets.json) | mining preset 机器可读快照 |
| [`enterprise_factor_engine_roadmap.md`](enterprise_factor_engine_roadmap.md) | 企业级路线图与验收表 |

---

## 2. 算子语义与 LLM

| 文档 | 用途 |
|------|------|
| [`operators_semantics.md`](operators_semantics.md) | 参数、列依赖、DSL 限制 |
| [`factor_engine_llm_prompt.md`](factor_engine_llm_prompt.md) | 给模型的算子摘要 |
| [`factor_engine_llm_prompt.txt`](factor_engine_llm_prompt.txt) | 同上（由 `sync_factor_engine_llm_prompt_txt.py` 从 md 生成） |
| [`cleaned_operators/operators_catalog.md`](../cleaned_operators/docs/operators_catalog.md) | 全量审计（含 stub） |

---

## 3. ADR 与数据字典

| 文档 | 用途 |
|------|------|
| [`adr_backtest_target_position.md`](adr_backtest_target_position.md) | 回测 `target_position` 协议 |
| [`adr_trade_when.md`](adr_trade_when.md) | `trade_when` 语义 |
| [`massive_parquet_data_dictionary.md`](massive_parquet_data_dictionary.md) | Massive 原始 parquet 字段 |
| [`changelog_shw.md`](changelog_shw.md) | 分版变更记录 |

---

## 4. 包内 README 导航

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

回测长篇说明：monorepo [`../../backtest_layer/single_asset_backtest/README.md`](../../backtest_layer/single_asset_backtest/README.md)

---

## 5. 已移除文档（勿再引用）

2026-06 整理时删除重复或过时正文，内容已并入上表或 `changelog_shw.md`：

`miner_api_import_guide.md` · `miner_ecosystem_integration.md` · `挖掘对接与本轮优化说明.md` · `operators_roadmap.md` · `huatai_factor_factory_operator_catalog.md` · `adr_huatai_factor_factory_operators.md` · `adr_context_benchmark.md` · `adr_ts_step_hump.md` · `factor_materializer.md` · `docs/_refs/`
