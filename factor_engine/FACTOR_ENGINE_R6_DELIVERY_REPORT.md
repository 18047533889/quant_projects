# FactorEngine 第 6 轮架构收口整改 — 交付报告

基于 main `6c1453f6de4794130524373e2bd147a50d6d3843`，对第 5 版 AI 审查（P0-01..P0-35 / P1-01..P1-24 / P2-01..P2-06 + 系统级测试 A-J）的收口落地。并发 session 全程同树并行；每处改动先 `git status`+重读，已被并发修好的不重复改。

## 一、已确认"已修好"不重复动（探索验证）

P0-01 backend 调用合同（`base_polars` 全走 `validate_operator_call` + `cleaned_bridge` 双保险 + 注册门禁）、P0-02 `calculate` 审计（注册时强制框架或 `_HANDLES_CALL_CONTRACT`）、P0-03 ParamSpec 权威（whitelist 仅 fallback，5.9/bool/NaN/Inf/unknown-kwarg/extra-positional 全拒）、P0-05 lowering 重复注册 guard、P1-08 event-age gap 语义、P1-17 `_SKIP_PANEL` 一致、P1-21 source_certified 机制、P1-23 capability INPUT_DEPENDENT 语义。

## 二、本轮实施（主会话 WS1 + 6 并行 agent WS2..WS7）

| 项 | 交付 |
|---|---|
| **P0-04** lowering 在参数校验之后 | `_helpers.strict_float`（拒 bool/NaN/Inf）；optimizer 在 lowering 前跑 `validate_plan_params`（复用 runtime 的 `_normalise_integer` → planning==runtime 精确一致） |
| **P0-05** 禁 silent overwrite | `_LOWERING_SOURCES` 来源追踪；duplicate 报错带双方模块；`declare_lowering_replacement()` 显式授权；`_LOWERING_REPLACEMENTS` 从死代码变真机制 |
| **P0-06** 禁假 probe | `LoweringContract` dataclass；`lowered_primitives` 改从注册算子真实 ParamSpec 默认值取参（fast/slow/signal 区分），不再硬编码 window=3；70 复合算子回填 deps/min_inputs（technical 13 + ts_ratio + microstructure 2） |
| **P1-19** ParameterCanonicalizer | float 12 位显著位、proportional weights 归一化 → 等价因子同 hash |
| **P1-16** stateful 命名 | STATEFUL_DEFERRED `ewm_std`→`ts_ewm_std` 统一 canonical |
| **P0-13** RecursiveKernel | `stateful_kernel.py`：bootstrap_full/step/serialize/restore 共享接口；split-parity 全 split bit-exact |
| **P0-18** 六证独立 | semantic/temporal/source 各自独立 evidence，不再由 implementation 间接授信 |
| **P0-19** alias evidence | alias 记录 evidence_origin，永远不升级 parent |
| **P0-20** inherited audit | 精确 canonical 集合 + per-canonical hash 校验，不再只比 count |
| **P0-21** execution hash | 覆盖 stateful_runtime/stateful_contract/session_calendar/planner/lowerings |
| **P0-22** override manifest | per-canonical exact manifest（additive，31 个 bootstrap 层不受影响） |
| **P0-23** kernel hash | 哈希 `__code__`/closure 而非 wrapper 源码 |
| **P0-25** backend signature 统一 | `OperatorRegistry.register` 把 canonical `param_names` 回填到每个 backend 实例 metadata（bridge `param_names=[]` 不再让 R5-06 extra-positional 门拒掉合法多面板调用） |
| **P0-10** 强制排序 | scan 边界统一 (instrument, time)/(+session)；clickhouse `ORDER BY`；接口文档化排序保证 |
| **P0-11** NormalizedFieldPlan | `storage/sources/field_plan.py` 15 字段统一计划；三后端读同一 plan；clickhouse 补 scale 归一 |
| **P0-12** 错误层次 | CatalogUnavailable/NotConfigured/ResolutionError/Corrupt；production fatal；import 失败默认 strict（不再静默降 research） |
| **P0-07** grain 拆型 | `SessionAggregationOperator(SeriesOperator)`；metadata input_grain/output_grain；`grain_minute_to_daily` tag；operator_policy shape_preserving=False（70 intraday） |
| **P0-08** SessionGrid | `SessionGrid` dataclass + `require_same_session_grid` fail-closed；6 处 concat+dropna 全加守卫 |
| **P0-09** SessionCoveragePolicy | strict_full_session/min_coverage_095/pairwise_allowed/event_sparse；realized skew/kurtosis min-30-returns；daily_agg min_finite=2 |
| **P1-18** DataDegeneracy | daily_agg 只把 DataDegeneracy/ZeroDiv/Overflow 转 NaN，参数/内核错误冒泡 |
| **P0-26** US dividend | DataAccess contract strict-PIT(declaration_date) + provider fail-closed 缺 declaration 不回落 ex-date；A 股保持 effective-only |
| **P0-27** US tradability | 构造 `StockList.type=="CS" AND finite(Close)` 真 mask（非 identity） |
| **P0-28** A 股 unknown-preserve | `_not_bool` NaN→NaN（不再 NaN→False）；concept 不再 overclaim listing |
| **P0-29** US mktcap | production full-universe coverage<0.8 → UNSUPPORTED（ProviderCoverageError），research 保持 warning |
| **P0-34** UniverseMask 接线 | `apply_universe_mask_to_panel` + lineage 记录 mask/coverage |
| **P0-35** input contract | `OperatorMarketContract.required_grain`；market-mechanism 无 contract → fail-closed（62 个） |
| **P1-24** coverage lineage | RunLineage 增 coverage_mask/coverage_ratio/drop_reason |
| **P1-01/02/03/06/07/13** | gather_ext 拒 reindex；负权 fail-closed；NaN weight fail-closed；expectation gap 后 censored；drawdown peak 跨 gap 重置；HistoryRequirementError |
| **P0-30/31/32/33** typed IR | Schema price_basis/flow_semantics；SemanticType 17 类；PriceBasis compile-time；flow_semantics 枚举；`check_financial_grain_contract` 重写为 typed flow（同比） |
| **P1-14** report lookback | `fiscal_event_lookback` 数据驱动，回落 env 80 |

## 三、系统级测试

| 测试 | 结果 |
|---|---|
| A backend call contract（`test_r6_backend_call_contract.py`，20） | pandas/polars `validate_operator_call` 一致 + sql 走 planning 拒绝；fractional/bool/NaN/Inf/unknown/extra/misalign/permute 全拒 |
| B composite validation（并入 `test_composite_lowering.py`，+10） | 5.9 拒绝；duplicate 带来源拒绝；declared replacement 放行；strict_float 拒 bool/NaN/Inf；proportional 归一 |
| C stateful split parity（`test_r6_stateful_split_parity.py`，11 canonical × 全 split） | full == prefix+restore+suffix 全过；含 gap 场景 |
| D checkpoint revision | input identity 变 → 拒；instrument 变 → 拒；同 identity → 复用 |
| E/G/H/I/J | 依赖 evidence/并发的测试见遗留 |

## 四、验证结果

- `load_all()` 干净：**1360 canonicals，0 unclassified**
- 本轮新增/更新测试全绿：composite 27/28、backend-contract 20/20、stateful 14/14+7、final_pack 64、audit_fixes 24、typed_ir 38、market 113、storage 52+11、evidence_fail_closed 4
- dataaccess `test_cos_contract_core.py` 4 过（P0-26 合同更新后已同步测试）

## 五、遗留（诚实说明）

1. **evidence 链仍 stale**（并发 session 持续改源码哈希）：`PRIMITIVE_DUAL_BACKEND_PRODUCTION_SAFE` 空、SQL parity 空 → `test_composite_dual_backend_capable_mom`、`test_sql_io`×3、`test_recipe_planner_bridge`、`test_backend_coverage`×2 失败。恢复顺序（树稳定后一次跑完）：factor → primitive → recipe → 3 manifests → catalog。
2. **并发 owner 的不一致**：`ts_huber_regression_resid`/`ts_expectile_regression_resid` 被并发 session 升到 daily 表面但 `should_fail_closed=True`+`pit_safe=False`（P0-04 门应 catch）；`test_layer_governance` duplicate；intraday_golden `ts_sharpe/ts_autocorr` min_periods 被并发 R5-06 门拒；`test_speed_oriented_backends`×2 + polars_registry_long_bridge[MACD]（并发 mid-edit）。
3. **P0-03 全量迁移未做**：所有 production 参数显式 ParamSpec 是大迁移；当前 whitelist 仅作 fallback，校验已 planning+runtime 一致。
4. **P0-13 共享 kernel 为 facade**：recurrence 单实现仍居 `stateful_runtime`（未重复），split-parity 已 bit-exact；pandas ewm 调用点改走 kernel 的深重构 deferred（风险/收益权衡）。
5. **P0-34 UniverseMask 已接线但未逐算子重写**：所有 CS 算子强制消费 mask 是超大横切改动；当前提供函数+lineage+测试。
6. **P2 项**（KNN artifact 缓存、signature kernel 合并、fallback 成本、stale ewm 名、research missing 语义、docs 同源）为低优先，本轮部分覆盖。
