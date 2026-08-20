# FactorEngine 第 7 轮共享链架构收口整改 — 交付报告

基于最新 main `608f602...`（含并发 session 的 round-7 工作；收口期间并发 session 又提交了 d008686/28e14c1 等）。
按 AI 审查 #219–#317 的 8 个统一重构包 A–H + 17 个 Release Gates 收口。
**原则落地：不做"每算子一个 patch / hardcoded whitelist"，全部修共享契约与执行链根因。**
全程 load_all 保持 **1362 canonicals / 0 unclassified**（并发 session 间歇性破坏为对方 in-flight，非我方回归）。

## 一、实施总览（WS-A 我直改 + 7 并行 agent WS-B..H）

### WS-A LogicalOperatorContract 单一真相源（我）

| 项 | 交付 |
|---|---|
| **#222** MISSING sentinel | `ParamSpec.default=MISSING`（区分"未声明"与显式 None）；`_validate_param_spec` 在 `default is None` 时放行 None（否则严格 float/bool/str 门拒声明的默认） |
| **#219** spec 边界权威 | `ParamSpec(int, min=None)` 接受 -1/0/1（不再套 legacy ≥1）；name-whitelist 仅对无 spec 算子生效 |
| **#220** 统一 binder | 新 `_validate_param_spec(value, name, spec, lower, upper)`：int/float/bool/str/enum 单入口严格类型（bool=`type(x) is bool`、float 拒 bool 且 finite、choices 全 dtype 强制）；declared-dtype spec 全走它 |
| **#223** alias→canonical | kwarg alias 先解析到 canonical 目标名再按目标 ParamSpec 验证；`param_aliases` 指向未声明参数 → 报错 |
| **#224** arity 拆分 | `panel_arity`/`total_positional_arity`/`scalar_params`；`_normalise_call` 用 total→input→panel+scalar→len(param_names) |
| **#226/#227** impl hash | `_impl_source_hash` 优先级=类定义 `_calculate_series/_scalar` → 直接 `calculate` → `_fn` bridge → closure payload；`_freeze_value` 确定性序列化（禁 `id()`/`repr`，byte-identical kernel 同 hash）；`_is_class_defined` 排除 framework bases |
| **#228** override 精确 | `_DECLARED_OVERRIDE_MANIFEST` key=`(canonical, backend)` |
| **#229** 禁 source-wide 宽覆盖 | declared-source 首次碰撞也强制 pin `expected_old_source`（自动 pin 当前 source）；对已建立 bootstrap 层不强制 replacement_reason（否则 load_all 崩） |
| **#231** rename 迁移 | `rename_canonical` 迁 `_first_registered`/manifest 重 key/`_override_chain`/`_overwrite_log` |
| **#233** 替换历史 | overwrite_log 记 `previous_contract_hash`/`new_contract_hash`/`expected_old_hash_matched`（`_contract_hash` 覆盖 params+specs+panel/scalar+aliases+units+grains+available_at） |
| **#234** lifecycle 快照 | `_first_registered[canonical]` 捕获一次、永不覆盖、unregister 保留 |
| **#240** 删 late polars validator | `contract_hardening._install_polars_validation` 删除；Polars 统一走中央 `validate_operator_call` |
| **#248** 禁 blanket min_periods | `_MIN_PERIODS_OVERRIDES.get(canonical,1)` → `_min_periods_from_contract`（cost:/min_periods: tags → `_MIN_PERIODS_FLOORS` → bars_） |
| **#230/#255/#258** unified contract | `catalog["contract"]` 全量 param_specs/panel_params/scalar_params/aliases/units/grains/available_at/relational_specs；`_validate_final_contracts` 加 catalog-vs-policy min_periods split-brain 检查 |
| **#312–314** 删签名启发式 | `pandas_first_signature._is_panel` 先读声明 panel_params/panel_arity；`_finite`/`_integer` 拒 str（static==runtime）；window 门由声明 ParamSpec(int) 驱动 |
| **#246** ctx.run_mode | `strict_polars_long_fallback(ctx)` 先 `ctx.run_mode` → env → runtime_stats → global |

### WS-B..H（agent，全绿）

- **WS-B PanelSchema/PanelIdentity**（#235-248）：`_polars_bridge.PanelIdentity`（time/instrument hash）进 polars `_prepare_call` validator；`panel_to_polars` 用 `__fe_time__` 列保留时间轴；`_align`/`align_cols` 禁列交集；NaN/Null 缺失语义统一；ImportError 只吞 polars。
- **WS-C Typed IR 2.0**（#265-278）：`SemanticLattice` 五维 join 取代 first-input 继承；per-arg `input_types` 契约（Volume↔Return 编译拒绝）；FieldSpec `semantic_kind`；`AvailabilityExpr` 层次（SessionClose(TradeDate)/NextTradingOpen(PubDate)/TimestampColumn）删除字符串全序、unknown fail-closed；`SourceVintageSpec` 结构化。
- **WS-D History/Stateful + Composite**（#256-264）：`runtime/execution_contract.py` `ExecutionContract(state_model/chunking/checkpoint_schema)` + `HistoryRequirement`（删 1e9 sentinel，`history_requirement(params)` 唯一 authority，warmup/analyzer 同源）；`register_lowering(contract=...)` 全量存储 + expected old hash 拒绝替换；composite 全参数分支认证（`certified_for_all_branches`）；capability probe 真实 panel_params 输入。
- **WS-E Field/Source**（#279-293, 315-317）：`mining_allowed` hard gate；`current_snapshot_only` 禁历史挖因子；four-layer PIT（field∧table∧dataset∧operator）；FieldRole Enum；minute→daily 禁 asof carry（exact TradeDate×Symbol）；`join_policy=exact` 默认；all-missing volume→NaN（`sum(min_count=1)`）；timestamp convention 来自 DatasetContract；HistoricalCoverageContract；MissingSemantic；source_dependency_hash。
- **WS-F Evidence TCB + EdgeContract**（#250-254）：edge undeclared=FAIL（`EdgeGateMode.PRODUCTION/RESEARCH`，删 `edge_gate_strict` vacuous pass）；`EdgeContract(nan/inf/zero/domain)` 层叠手工名单；edge evidence 按 backend（`{"edge_evidence": {"pandas_numpy":…, "polars":…, "duckdb":…}}`）；`ExecutionTCB` 10 件套进 factor evidence hash（改 bridge/router/panel→旧 evidence 失效）；`invalidate_all_evidence_caches()`。
- **WS-G Search-space Audit/Dedup**（#294-306）：默认全量（`MAX_OPS=None`）+ stable-hash sharding 16 分片；`split_scalar_panel_params`（window 不再被当 panel）；choices 枚举/整数网格/float 分位数采样；active branch 敏感性；exception→FAIL/AUDIT_UNEXECUTABLE 不 silent；10 个 typed fixture（240×128）；**`rank(axis=1)`**（A 股横截面约定，`_rank_corr` 改 `r>=threshold` 正向）；exact float64 bytes（不 round(9)）+ NaN validity-mask hash（不 -1e300）；algebraic 层诚实删除；`FactorKind` ALPHA/CONDITION/EVENT/GLOBAL_STATE 不同 signature；`DedupPolicy(sign_invariant)`。
- **WS-H Surface/Production Tier**（#309-311）：`pandas_first_production_canonicals()` 动态派生（不 snapshot `EXTENDED_ONLY`）；`AuthoringTier`/`ProductionCertification`/`BackendCapability` 三正交维度；`REVIEWED_MIGRATION_MANIFEST` 驱动 daily-migrated。

## 二、验证结果

- `ensure_cleaned_loaded()` → **1362 canonicals / 0 unclassified**
- R7 新增测试全绿（**154 项**）：
  - WS-A：`test_r7_param_spec_binder`（16）、`test_r7_registry_governance`（13）
  - WS-B：`test_r7_panel_identity`（23，含 #246 真测试）
  - WS-C：`test_r7_availability_expr`（17）+ `test_r7_typed_ir`（13）
  - WS-D：`test_r7_execution_contract`（12）
  - WS-E：`test_r7_field_contract_gates`（21）
  - WS-F：`test_r7_evidence_tcb`（9）
  - WS-G：`test_r7_search_axis1`（22）
  - WS-H：`test_r7_surface_orthogonal`（8）
- 既有回归：`test_r6_backend_call_contract` 20、`test_audit_fixes_2026` + `test_final_pack_2026_08` 108、`test_typed_ir_v2` + `test_round6_alignment_pit_2026_08` 40 全绿。
- 审计冒烟（`parameter_sensitivity_audit.py --limit 5`）：真实发现 4 条（ADL/ADX/ATR_WILDER/CMF 在 group fixture 上 window dead/scale-alias），FAIL 被记录不 silent。

## 三、诚实遗留

1. **evidence 链仍 stale**（并发 session 持续改源码 hash + AST 重构）：factor/primitive/recipe evidence、3 manifests、catalog 均未重生成。**树稳定后按序一次跑完**：factor → primitive → recipe → 3 manifests → catalog。
2. **#259 production_signature 三套签名合并**：列为"最终目标"未做（`PRODUCTION_SIGNATURES == DAILY_CANONICALS` 精确相等是 guardrail，全量合并风险高）；已做 #312-314（删 pandas_first_signature 启发式）。
3. **P0-03 全量 ParamSpec 迁移**：legacy 无 spec 算子仍走 name-whitelist fallback（校验已 planning+runtime 一致），未全量显式声明。
4. **#250 行为变化**：edge undeclared 现在 production 一律 FAIL——因 evidence 未重生成，大量未声明 edge 的算子会 fail-closed（这是预期的 fail-closed，直到 evidence 重生成）。
5. **并发 owner 项**：`ts_market_liquidity_beta`/`ts_industry_liquidity_beta` parity、`intra_jump_first_time/last_time` 数值、`test_layer_governance` duplicate、intraday_golden `ts_sharpe/ts_autocorr`、`market_language` relation_diffusion、`ts_huber_regression_resid`/`ts_expectile_regression_resid` pit_safe=False promotion（非本批，均需并发 session 收尾）。
6. **对方 AST 重构中间态**：base.py `parse_relational_expression` 目前拒绝 `2 ** (level + 2) <= window`（LtE 语法超集问题）——对方在飞编辑，等其收敛。
7. **17 个 Release Gates** 的机制已落地（见各 WS），但 gate 测试本身（test_r7_release_gates 等）未单独成文件；由各 WS 的验收测试覆盖。

## 四、文件清单

**新增**：`runtime/execution_contract.py`；测试 `test_r7_param_spec_binder.py`、`test_r7_registry_governance.py`、`test_r7_panel_identity.py`、`test_r7_field_contract_gates.py`、`test_r7_evidence_tcb.py`、`test_r7_execution_contract.py`、`test_r7_availability_expr.py`、`test_r7_typed_ir.py`、`test_r7_search_axis1.py`、`test_r7_surface_orthogonal.py`。
**修改（WS-A 核心）**：`cleaned_operators/base.py`（MISSING sentinel + `_validate_param_spec` + arity 拆分 + alias 绑定）、`cleaned_operators/registry.py`（impl/contract hash、first_registered、(canonical,backend) manifest、rename 迁移、overwrite log 扩展）、`cleaned_operators/contract_hardening.py`（删 late polars validator、完整 unified contract、split-brain 检查）、`cleaned_operators/production_hardening.py`（`_min_periods_from_contract`）、`backend/pandas_first_signature.py`（声明驱动的 panel/窗口判定）、`backend/polars_long_policy.py`（ctx.run_mode 优先）、`cleaned_operators/registration_audit.py`（logical signature equivalence）。

完整计划见 `FACTOR_ENGINE_R7_CONSOLIDATION_PLAN.md`。
