# FactorEngine 第 7 轮架构收口整改 — 执行计划

基于最新 main `608f602800887998587d723564107ed813a5d80c`（含并发 session 的 round-7 工作）。
本轮按 AI 审查 #219–#317 + 8 个统一重构包 A–H + 17 个 Release Gates 收口。
**原则：不做"每算子一个 patch / hardcoded whitelist"，只修共享契约与执行链根因。**

## 基线（实测确认）

- `ensure_cleaned_loaded()` 后 **1362 canonicals / 0 unclassified**（`list_canonical()` 直接调只有 417——必须先走 cleaned_bridge）。
- 并发 session 已修（不再重复）：Gemini V2 时间轴列、ParamSpec choices、unknown kwargs、部分 `active_when`、`_auto_pin_declared_bootstrap` 链 pinning、surface 的 `extend_extended_only` 活 mutator。
- 工作树 122 个文件 dirty（并发 session 未提交）。**每处改动先 `git status --short <file>` + 重读；Edit stale-read 拒绝 = 冲突检测；并发已修好则不重复。**
- 证据链仍 stale（concurrent 持续改哈希）；evidence 重生成放最后。

## 确认为 live 的 review 项（gap analysis 实测）

| # | 状态 |
|---|---|
| #219 `ParamSpec(int, min=None)` 被 `_normalise_integer` 兜底成 ≥1 | LIVE（base.py:251 `spec.min is not None` 才用，否则 name-whitelist） |
| #222 `ParamSpec.default=None` 无 MISSING sentinel | LIVE（base.py 无 `MISSING = object()`） |
| #224 arity 未拆 panel/scalar | LIVE（OperatorSpec 无 panel_arity/scalar_params 细分） |
| #226 impl hash 优先级可能 hash 到 base `calculate` | LIVE（registry 已有 `_impl_source_hash`，需补优先级与 framework_hash 组合） |
| #234 first_registered_status/source/hash 无 | LIVE（registry 无 lifecycle snapshot 前置） |
| #240 `_install_polars_validation` 仍在 registry 尾段装 | LIVE（contract_hardening.py:352） |
| #248 `_MIN_PERIODS_OVERRIDES.get(canonical, 1)` blanket | LIVE（production_hardening.py:345） |
| #250 edge undeclared → vacuous pass | LIVE（edge_requirements.py `edge_gate_strict=False` 默认） |
| #251 手工 NAN_REQUIRED(36)/INF_REQUIRED(17) | LIVE（edge_requirements.py） |
| #253/#254 evidence TCB 不全 / 无 invalidate_all_evidence_caches | LIVE（evidence_provenance 无 ExecutionTCB、无统一 invalidation） |
| #274 `_AVAILABILITY_RANK` "unknown" 排最前（应 fail-closed +∞） | LIVE（ir/schema.py:110 `"unknown"` 在 index 0） |
| #277 `_EOD_AVAILABILITY_MARKERS` substring "pe"/"pb" | LIVE（ir/schema.py:162） |
| #294 `MAX_OPS = 60` | LIVE（scripts/parameter_sensitivity_audit.py） |
| #300/#301 dedup `rank(axis=0)` + `round(9)`/`-1e300` | LIVE（factor_dedup.py:51/59 `axis=0`、`nan_to_num(...,-1e300)`） |
| #309 `PANDAS_FIRST_PRODUCTION_CANONICALS = frozenset(EXTENDED_ONLY...)` 快照 | LIVE（production_tiers.py size 1173） |
| #312/#313/#314 `_PANEL_NAMES`/`_QUANTILE_Q`/`_TOPK`/`_WINDOW_NAMES` | LIVE（pandas_first_signature.py） |
| #256/#257 `_STATEFUL_CANONICALS`+`_LOOKBACK_PARAM_PRIORITY` 窗口名猜测 | LIVE（contract_hardening.py:37/85） |

## 八重构包 → 工作流划分（文件不重叠）

### WS-A 「LogicalOperatorContract 单一真相源」（我直改，核心契约层）
**改**：`cleaned_operators/base.py`、`cleaned_operators/registry.py`、`cleaned_operators/contract_hardening.py`、`backend/production_signature.py`、`backend/pandas_first_signature.py`
**内容**：#219 min=None 用显式值、#220 统一 `_validate_param_spec`（int/float/bool/str/enum 严格）、#222 `MISSING` sentinel、#223 alias→canonical 绑定验证、#224 arity 拆 panel/total、#226–#227 implementation hash（类定义 `_calculate_series` 优先 + deterministic closure serializer，禁 `id()`/`repr`）、#228–#229 override 精确 `(canonical,backend)` + 删 trusted-source 宽覆盖、#230 LogicalOperatorContract 一份 immutable contract 三 backend 引用、#231 rename 迁移全部治理数据、#232 logical signature equivalence 审计、#233 replacement history 记 impl/contract hash、#234 first_registered_status 永久快照、#240 删 `_install_polars_validation`（Pandas/Polars 统一 `validate_operator_call`）、#248 删 blanket `min_periods=1`、#249/255/258 late sync 全量 contract + 防 split-brain、#257 `history_requirement(params)` 唯一 authority、#259 三套签名合并、#312–#314 删 pandas_first_signature 启发式名单。
**验收**：`_normalise_integer` 三连（#219）；`ParamSpec(default=MISSING)` 区分；`ts_mean/pandas_numpy`、`ts_mean/polars` 独立 override；`rename_canonical` 前后 contract hash 一致；Polars/Pandas 同一 validator。

### WS-B 「PanelSchema / PanelIdentity」（agent）
**改**：`cleaned_operators/base_polars.py`、`cleaned_operators/gemini_v2_common.py`、`cleaned_operators/rolling_pack.py`、`cleaned_operators/composite_fastpath*.py`、`cleaned_operators/layer_*`（仅 panel 对齐处）、`backend/panel_polars.py`、`backend/cleaned_bridge.py`
**内容**：#235–#238 元数据列/索引语义统一、#241 缺失语义 `MissingNumeric = null OR NaN` 统一 + NaN/Null 混合 parity、#242 `_align()` 禁取列交集、#243–#244 PanelIdentity（time_index_hash/instrument_axis_hash/grain/frequency）进 validator、#245 `panel_to_polars` 时间 index 保留（`__fe_time__` 列或 PanelIdentity token）、#246 `strict_polars_long_fallback` 先读 `ctx.run_mode`、#247 optional polars ImportError 只吞 `ModuleNotFoundError: name=="polars"`。
**验收**：#243 日期轴移动一天 → fail；列置换 → fail；NaN/Null 三 backend parity。

### WS-C 「Typed IR 2.0 + AvailabilityExpr」（agent）
**改**：`ir/types.py`、`ir/schema.py`、`ir/analyzer.py`、`fields/spec.py`
**内容**：#265–#268 semantic lattice（domain/frequency/source_vintage/universe_id/semantic_kind 五维）+ per-argument type contract、#269–#273 `input_types={...}` 逐参数声明 + FieldSpec `semantic_kind`（Volume→NonNegativeActivity、UniverseMask→EventBool、GroupId→GroupKey）、#274–#277 `AvailabilityExpr`（SessionClose(TradeDate)/NextTradingOpen(PubDate)/TimestampColumn(...)），unknown → production 根 unknown/fail-closed，禁 substring 猜（`"pe"`/`"pb"`）、#278 `SourceVintageSpec` 结构化。
**验收**：Volume↔Return 换位编译拒绝；unknown availability fail-closed；`available_at` 无字符串全序。

### WS-D 「History/Stateful + Composite Contract」（agent）
**改**：`runtime/warmup_service.py`、`cleaned_operators/production_policy_extensions_v2.py`、`planner/composite_lowering.py`、`planner/lowerings/_helpers.py`、`planner/optimizer.py`
**内容**：#256 `ExecutionContract(state_model/chunking/checkpoint_schema)` 单一真相、删 1e9 sentinel、#257 `HistoryRequirement(kind/rows)` 唯一 authority、#260–#261 `register_lowering(contract=...)` 完整保存 + expected old hash、#262 composite certification 枚举全部 `param_branches`、#263–#264 capability probe 按 panel_params 真实输入 + 多 regime。
**验收**：`ts_ema` state_model=recursive、chunking=checkpoint；MACD fast/slow/signal 各分支全认证；full==chunked==incremental。

### WS-E 「Field/Source Contract 执行」（agent）
**改**：`fields/catalog.py`、`fields/providers.py`、`storage/sources/field_plan.py`、`storage/sources/data_access_source.py`、`storage/sources/lqtp_logical_source_v2.py`、`api/columns.py`、`api/source_ref.py`、`storage/sources/financial.py`、`storage/sources/relation.py`、`pit_contract.py`
**内容**：#279 `mining_allowed` production hard gate、#280 `current_snapshot_only` → 禁止历史挖因子、#281 `required_filters/applicability/allowed_operator_families/null_policy/semantic_kind` 贯穿到 IR leaf、#282 四层 PIT（field∧table∧dataset∧operator）、#283 replace 清理旧 alias、#284 `FieldRole` Enum、#289 SourceRef scalar 拒 NaN/Inf、#290 minute→daily 禁 asof carry（exact TradeDate×Symbol）、#291 Intermediate/DerivedField `join_policy=exact` 默认、#292 all-missing volume/amount → NaN（sum min_count=1）、#293 timestamp convention 来自 DatasetContract、#315 HistoricalCoverageContract、#316 MissingSemantic（UNKNOWN/NO_EVENT/NOT_APPLICABLE/NOT_TRADING/STRUCTURAL_ZERO）、#317 source_dependency_hash 入 materialization identity。
**验收**：`mining_allowed=False` 字段 mining 拒；minute→daily 缺日 NaN 不 carry；`role="knowledge_tiem"` 拼错 catalog fail。

### WS-F 「Evidence TCB + EdgeContract」（agent）
**改**：`backend/evidence_provenance.py`、`backend/factor_operator_evidence.py`、`backend/primitive_evidence.py`、`cleaned_operators/edge_requirements.py`、`backend/production_certification_overlay.py`
**内容**：#250 undeclared=edge FAIL（删 `edge_gate_strict=False` vacuous pass）、#251 OperatorContract `EdgeContract(nan/inf/zero/domain)` 取代手工名单、#252 edge evidence 按 backend（pandas/polars/duckdb 三份）、#253 ExecutionTCB（operator/contract/catalog/IR/planner/bridge/router/panel_conversion/source_alignment/calendar）进 factor evidence hash、#254 `invalidate_all_evidence_caches()` 统一出口。
**验收**：改 bridge/router → 旧 evidence 失效；undeclared edge → 不认证；三 backend 独立 edge evidence。

### WS-G 「Search-space Audit / Dedup」（agent）
**改**：`scripts/parameter_sensitivity_audit.py`、`cleaned_operators/search/factor_dedup.py`、`api/mining_integration.py`
**内容**：#294 默认全量（CI 用 stable hash sharding 16 分片）、#295 用 `panel_params/scalar_params` 非猜测、#296 choices 枚举采样、#297 active branch 敏感性、#298 exception → FAIL/AUDIT_UNEXECUTABLE 不 silent continue、#299 typed fixture、#300 `rank(axis=1)`、#301 probe ≥128 stocks + 多 regime（Gaussian/heavy-tail/trend/mean-revert/ties/gaps/positive/event/group/OHLC）、#302 exact signature 不 round(9)、#303 NaN validity mask 独立 hash 不替换 -1e300、#304 algebraic canonicalization 真实现或删除表述、#305 ALPHA/CONDITION/EVENT/GLOBAL_STATE 不同 signature、#306 sign invariance 显式 DedupPolicy。
**验收**：`rank(axis=1)` 全部；NaN mask hash；dedup 不再 `axis=0`。

### WS-H 「Surface / Production Tier 正交」（agent）
**改**：`cleaned_operators/operator_surface.py`、`cleaned_operators/production_tiers.py`、`cleaned_operators/registration_audit.py`、`cleaned_operators/__init__.py`
**内容**：#309 `pandas_first_production_canonicals()` 动态派生（不 snapshot）、#310 AuthoringTier / ProductionCertification / BackendCapability 三正交维度、#311 `DAILY_FACTOR_MIGRATED` 由版本绑定 reviewed manifest 生成。
**验收**：`PANDAS_FIRST_PRODUCTION_CANONICALS` 非 frozenset 快照；surface 成员变动不隐式改 certification。

## 最终 Release Gates（全部加 CI 测试）

1. Contract Single-Source：同 canonical 无互相独立参数定义。
2. Panel Identity：随机移动输入日期轴一天 → 多 panel 算子 fail。
3. Column Permutation：打乱再还原 → 一致。
4. NaN/Null/Inf 三 backend 独立 evidence。
5. Typed Input：Volume↔Return 编译拒绝。
6. Unknown Field：拼错 production 编译失败。
7. Coverage：current snapshot / partial history / sparse event 按合同。
8. History：实际 warmup == planner history。
9. Chunk：full == chunked == incremental。
10. Composite Branch：全参数分支认证。
11. Backend TCB：改 bridge/router → evidence 失效。
12. Search Equivalence：searchable 参数产生真正不同 cs-rank。
13. Cross-sectional Dedup：统一 `axis=1`。
14. Import Failure：backend 模块内部 ImportError 不被 optional loader 吞。
15. Registry Freeze：freeze 后 contract/backend/surface mutation 全失败。
16. ParamSpec/Runtime/Planner Consistency。
17. First-Registration Lifecycle。

## 实施顺序（严格遵守）

1. WS-A LogicalOperatorContract（我）→ 2. WS-B PanelSchema → 3. 删 late polars validator（WS-A 内 #240）→ 4. WS-D History → 5. WS-C Typed IR → 6. AvailabilityExpr（WS-C）→ 7. WS-E Field/Source → 8. WS-D Composite → 9. WS-F Evidence TCB → 10. WS-G Search audit → 11. WS-H Surface → 12. 全量 evidence/catalog/manifest 重生（树稳定后）。

## 验证

- 每 WS 完成后跑其验收点 + `ensure_cleaned_loaded()` 后 load_all（目标 1362/0）。
- 最终：新增 gate 测试全绿；operators 套件失败数较基线收敛；17 gates 逐条通过。
- evidence 重生成按序：factor → primitive → recipe → 3 manifests → catalog。

## 关键文件

新增测试：`tests/operators/test_r7_param_spec_binder.py`、`tests/operators/test_r7_panel_identity.py`、`tests/operators/test_r7_availability_expr.py`、`tests/operators/test_r7_execution_contract.py`、`tests/operators/test_r7_field_contract_gates.py`、`tests/operators/test_r7_evidence_tcb.py`、`tests/operators/test_r7_search_axis1.py`、`tests/operators/test_r7_surface_orthogonal.py`、`tests/operators/test_r7_release_gates.py`。
