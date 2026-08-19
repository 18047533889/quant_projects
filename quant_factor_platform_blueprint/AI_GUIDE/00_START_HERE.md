# 00 — Start Here：AI 开发总指令

> **Context budget:** skim this file for orientation. Do **not** load the entire `AI_GUIDE/` tree (or `docs/R2_HISTORY_ARCHIVE.md`) into a session/subagent. Open only the chapter for the package you are changing.

## 0.1 目标

把现有 `quant_projects` 从“多个历史模块并存、能力重复、接口分散”的状态，整理为一套可以独立分发、企业级可维护、适配数万到十万因子规模的模型前因子平台。

本次只建设四个新运行库：

1. `quant_evaluator`
2. `factor_optimizer`
3. `factor_assets`
4. `factor_preprocess`

现有 `dataaccess` 和 `factor_engine` 是底座，原则上修补接口/bug 可以，但不得用新库复制它们的职责。

## 0.2 先迁移，后新增；先正确，后加速；先 Contract，后并行

必须严格按以下顺序：

1. **Server Audit**：确认服务器 working tree 当前真实代码。
2. **Legacy Mining**：逐函数盘点旧代码。
3. **Migration Matrix**：每个候选函数分类。
4. **Contract Freeze**：冻结跨库 schema / public API / dependency DAG。
5. **Reference Migration**：先迁可信 reference。
6. **Golden Tests**：证明数学/行为保持正确。
7. **Fast Kernels**：再写 NumPy/Numba/Polars 快路径。
8. **New Capabilities**：增加旧仓没有的高级功能。
9. **Cross-library Integration**。
10. **Independent Audit**。
11. **Extraction Test**。
12. **Legacy Archive**。

任何 Agent 如果跳过 1–4 直接造新平台，都应被停止。

## 0.3 五类迁移决定

Legacy Miner 对每个文件/函数必须给出以下之一：

- `REUSE_AFTER_TEST`：代码结构可直接迁，但必须先加测试并移除耦合。
- `REWRITE`：数学/业务语义有价值，但实现有 bug、性能、耦合或边界问题，需要重写。
- `REFERENCE_ONLY`：只作为数学/reference oracle 或行为对照，不进入 production runtime。
- `CORPUS_ONLY`：作为真实公式、真实 workload、回归/性能测试语料。
- `DISCARD`：能力已被 DA/FE 覆盖、属于旧平台壳、明显 stub/错误/重复，无迁移价值。

## 0.4 当前已知高价值代码矿区（必须在服务器重新确认）

基于现有仓库历史审计，优先搜索：

- `factor_layer/factor_evaluation/pipeline.py`
- `AutoFactorEvaluation-RECONSTRUCT/evaluation/batch_metrics.py`
- `FactorAnalyzer.py`
- `Exposures.py`
- `AlphaPurifier.py`
- `APr_utils.py`
- `toolkit/cross_sectional.py`
- `toolkit/registry.py`
- `alpha_tools/registry.py`
- `alpha_tools/generated_library.py`
- `factor_admission/catalog.py`
- Gateway 的 Candidate / Complexity / Deduplicator / Assetization 相关实现
- `factor_agent/`
- `gtja191/`
- `week2_pv_factors/`
- `ashare_lqtp_kit/`
- `raw_data_layer/`

**注意：路径可能已在服务器发生变化。禁止因 GitHub/旧文档存在某路径就假设服务器仍一样。使用 `find`, `rg`, Python AST/import scan` 重新定位。**

## 0.5 当前建议的职责

```text
DataAccess
  owns: data semantics, PIT data access, universe/calendar/snapshot, I/O

FactorEngine
  owns: factor definition, DSL/AST/IR, operator semantics, computation/materialization

QuantEvaluator
  owns: evidence, metrics, diagnostics

FactorOptimizer
  owns: diagnosis-guided mutation and search

FactorAssets
  owns: factor identity, relations, decisions, organization, aggregation, lifecycle

FactorPreprocess
  owns: representation/neutralization/transforms immediately before model input

Research Control
  owns: campaign/trial/event ledger only
```

## 0.6 不得重复建设

详见 `13_DECISIONS_AND_DO_NOT_BUILD.md`。尤其禁止二次实现 PIT、storage、factor compiler、materialization、universe/calendar、forward platform。

## 0.7 开发完成定义

“代码能跑”不是完成。必须同时满足：

- 数学正确
- 时间正确
- 可独立安装
- 批量可扩展
- 旧代码依赖清零
- 不重复 DA/FE
- 真实因子 corpus 回归通过
- 性能有 benchmark
- 关键路径有独立 audit
- 迁移来源与新归属可追溯

从下一份文档开始按专题执行。
