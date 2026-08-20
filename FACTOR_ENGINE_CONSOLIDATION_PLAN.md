# FactorEngine 架构收口整改计划（round-6 最终收口）

基于 main `6c1453f6de4794130524373e2bd147a50d6d3843`，对第 5 版 AI 审查（P0-01..P0-35 / P1-01..P1-24 / P2-01..P2-06 + 系统级测试 A-J）的落地计划。

## 总原则

> 同一个 canonical，在任意合法 backend、任意 full/incremental 分段、任意合法数据读取路径上，都表示同一个严格 PIT、安全、有明确定义的数学对象。

- **本轮不加新算子**（review 明确要求先收口再扩面）。
- 不重复修已确认修好的项（review §四 20 条 + 前几轮已验的 P0/P1）。
- 并发 session 正在改同一棵树：每个 finding 先 `git status` + 重读，已被并发修好的跳过；重读被 Edit 拒绝=并发在改，重读后最小化改。
- 证据链（factor→primitive→recipe→3 manifests→catalog）在树稳定前只跑 fast 生成器，最后统一重跑。

## 探索结论摘要（8 个 mapping agent 已回报，2026-08-08）

| 工作流 | 状态 | 缺口 |
|---|---|---|
| P0-01/02/03/25 backend 调用合同 | ✅ 已做 | 仅 24 个 polars bridge `param_names=[]`（cosmetic） |
| P0-04/05/06 composite lowering | 部分 | `_helpers.float_attr/literal` 裸 `float()`；optimizer 无 pre-lowering ParamSpec 校验；70 复合算子 deps=[] 全走假 probe；`_LOWERING_REPLACEMENTS` 死代码 |
| P0-07/08/09/35 minute→daily grain | 缺 | 全在 SeriesOperator、无 SessionAggregationOperator/grain 字段/SessionGrid/SessionCoveragePolicy；6 处 concat+dropna；realized skew/kurtosis 无 min-count |
| P0-10/11/12 source 排序/字段计划 | 缺 | scan 无排序；无 NormalizedFieldPlan；`_resolve_columns:257` broad-except → 静默降级；clickhouse 无 scale 归一 |
| P0-13/P1-15/16 stateful 唯一 kernel | 部分 | 两套实现已 bit-exact 但无共享 RecursiveKernel；7 处 stateful 集合分散；STATEFUL_DEFERRED 命名 `ewm_std`≠`ts_ewm_std` |
| P0-18..23 六证独立 + hash 覆盖 | 部分/缺 | semantic/temporal 门被 implementation 间接授信；alias 全继承；inherited audit 只比 count；stateful_runtime/contract 不入 hash；override 是 source-wide 白名单；kernel hash 哈希 wrapper |
| P0-24 polars native/fallback | 缺 | 无三分类 capability；polars_geometry_math/polars_ops 真 materialize pandas 却标 `backend="polars"`；成本平摊 |
| P0-26..29/34 市场 provider + UniverseMask | 部分/缺 | US dividend 两层冲突(provider=declaration_date vs contract=effective_only)；US tradability transform=_identity；A 股 `_not_bool` NaN→False 不保 unknown；US mktcap coverage_gate 只 warning；UniverseMask 定义未接线 |
| P0-30..33 typed IR + flow | 部分 | Schema 无 price_basis；analyzer 用字符串匹配；无 flow_semantics 枚举；`check_financial_grain_contract` 是 AST name-guess |
| P1-01..14/17..24 算子+治理 | 混合 | P1-01 reindex 静默、02 负权剔除、03 doc/impl 不一致、06 gap 后重 0、07 peak 跨 gap 保留、13 required=0 静默、14 固定 80 行、18 broad-except、19/20 无 canonicalizer/sensitivity、24 lineage 无 coverage；P1-08/17/21/22/23 已做 |

## 实施状态（2026-08-08 更新）

| 批次 | 状态 | 交付 |
|---|---|---|
| WS1 composite (主会话) | ✅ | `strict_float`/`validate_plan_params`(pre-lowering ParamSpec 校验)/replacement manifest+source 追踪/honest-probe+deps 回填/LoweringContract/ParameterCanonicalizer/STATEFUL_DEFERRED `ts_ewm_*` 命名；tests/planner 27+20 过（1 evidence-stale 除外） |
| WS2 evidence (agent) | ✅ | P0-18 六证独立、P0-19 alias 不升级 parent、P0-20 exact canonical set+hash、P0-21 execution hash 覆盖 stateful/calendar/lowerings、P0-22 per-canonical override manifest、P0-23 code-object+closure kernel hash |
| WS3 source (agent) | 进行中 | P0-10 强制排序、P0-11 NormalizedFieldPlan、P0-12 catalog 错误层次+production fatal |
| WS4 grain (agent) | ✅ | P0-07 SessionAggregationOperator+grain 字段、P0-08 SessionGrid+require_same_session_grid、P0-09 SessionCoveragePolicy+realized min-count、P1-18 DataDegeneracy、P1-17 PANEL_SKIP_COLUMNS；final_pack 64 过 |
| WS5 provider (agent) | ✅ | P0-26 US dividend strict-PIT(declaration_date)两层层级统一、P0-27 US tradability 真实 CS+DailyBar、P0-28 A 股 `_not_bool` NaN-preserve、P0-29 mktcap coverage production hard-fail、P0-34 UniverseMask 接线+lineage、P0-35 required_grain+market-mechanism fail-closed、P1-24 coverage lineage；tests/market 113 过 |
| WS6 CS/P1 (agent) | ✅ | P1-01 _align 拒绝 reindex、P1-02 负权 fail-closed、P1-03 NaN weight fail-closed、P1-06 gap 后 censored、P1-07 peak 跨 gap 重置、P1-13 HistoryRequirementError |
| WS7 typed IR (agent) | ✅ | P0-30 SemanticType+Schema price_basis/flow_semantics、P0-31 PriceBasis compile-time、P0-32 flow_semantics 枚举、P0-33 重写 check_financial_grain_contract(同比)、P1-14 fiscal_event_lookback；38 过 |
| 系统测试 C/D (主会话) | ✅ | test_r6_stateful_split_parity.py 14 过（全 split 全 segmented canonical + checkpoint revision） |
| 系统测试 A (主会话) | ✅ | test_r6_backend_call_contract.py 20 过（pandas/polars 一致 + sql planning 一致） |
| P0-13 RecursiveKernel (主会话) | ✅ | stateful_kernel.py（bootstrap_full/step/serialize/restore 共享接口） |

## 实施批次（6 个并行 agent + 主会话，文件不重叠）

### WS1（主会话）— composite lowering 硬化 + ParameterCanonicalizer
planner/lowerings/_helpers.py、planner/lowerings/technical.py、planner/composite_lowering.py、planner/optimizer.py、planner/canonicalize_params.py、backend/parameter_aliases.py
→ P0-04(G1/G2)、P0-05(G3/G4)、P0-06(G5/G6)、P1-19、STATEFUL_DEFERRED 命名

### WS2（agent A）— evidence 独立 + hash 覆盖
semantic_certification.py、factor_operator_evidence.py、evidence_provenance.py、registry.py、production_certification_overlay.py
→ P0-18/19/20/21/22/23

### WS3（agent B）— source 排序 + NormalizedFieldPlan + 错误层次
backend/polars_lazy.py、storage/sources/data_access_source.py、clickhouse_source.py、long_table_source.py、backend/long_frame.py
→ P0-10/11/12

### WS4（agent C）— minute grain + SessionGrid + coverage
cleaned_operators/intraday/_core.py、time_structure.py、time_structure_v2.py、higher_moments.py、realized_beta.py、cleaned_operators/base.py(grain 字段)、operator_policy.py
→ P0-07/08/09、P1-18(DataDegeneracy)、P1-17(统一 _SKIP_PANEL)

### WS5（agent D）— providers + UniverseMask + lineage
fields/providers.py、market/universe.py、market/capability_resolver.py、runtime/lineage.py、runtime/lineage_service.py、../dataaccess/cos_contract_us.py
→ P0-26/27/28/29/34/35、P1-24

### WS6（agent E）— CS/算子 P1
cleaned_operators/gather_ext.py、weighted_moment_ext.py、fundamental/expectation_v2.py、downside_risk.py、ir/analyzer.py(history req)
→ P1-01/02/03/06/07/13

### WS7（agent F）— typed IR + PriceBasis + flow semantics
ir/schema.py、ir/analyzer.py、ir/types.py、fields/spec.py、fields/concepts.py、cleaned_operators/operator_spec.py
→ P0-30/31/32/33、P1-14

## 系统级测试（A-J）
P0-20 StatefulnessRegistry CI · P0-30 typed IR 类型 · P0-31 PriceBasis · P0-32 flow grammar ·
P0-34 UniverseMask · P1-16 唯一 stateful registry · P1-20 sensitivity gate ·
系统级: backend call contract / composite / stateful split parity / checkpoint revision /
minute session / grain transition / source parity / ordering / provider contract / financial flow

## 实施批次（文件不重叠，可并行）

### Batch A — 中央调用合同（P0-01/02/03/25）
### Batch B — composite lowering 硬化（P0-04/05/06）
### Batch C — minute→daily grain 拆型 + SessionGrid（P0-07/08/09/35）
### Batch D — source 排序 + NormalizedFieldPlan + 分辨率失败硬化（P0-10/11/12）
### Batch E — RecursiveKernel + StatefulnessRegistry（P0-13/P1-15/16）
### Batch F — 六证独立 + execution-hash + override manifest + kernel-hash（P0-18..23）
### Batch G — polars capability 三分类 + 成本（P0-24 + P2-03）
### Batch H — US/A provider 合同统一 + UniverseMask（P0-26..29/34 + P1-24）
### Batch I — typed IR + PriceBasis + flow semantics（P0-30..33）
### Batch J — CS/算子 P1 修复（P1-01..14, P1-17..23 + P2-01/02/04/05/06）

## 系统级测试（A-J）

A backend call contract · B composite validation · C stateful split parity ·
D checkpoint revision · E minute session · F grain transition · G source backend parity ·
H ordering shuffle · I provider contract · J financial flow grammar

## 最终门禁

```
pytest tests/ -q
audit_all_factor_production.py --strict
evidence: factor → primitive → recipe → 3 manifests → catalog（树稳定后一次跑完）
```

## 交付

中文报告：每项 finding 的 [已修-我/已修-并发/部分/未做] + 真实测试结果 + 遗留 + 证据重生成顺序。
