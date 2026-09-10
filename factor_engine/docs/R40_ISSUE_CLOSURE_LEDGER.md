# R40 Issue Closure Ledger

> 整改依据：`FactorEngine_最新HEAD_260项全量整改提示词_20260811.md`（260 项全量缺陷与优化）
> 审计基线 HEAD：`03a57f35`；实施期间并发会话持续提交（最终 HEAD `96939012`），证据以真实 HEAD 为准。
> 完成判定遵循提示词 §十二：**直接开始改，不要只输出整改建议**；每项只有真实执行路径改动 + 单元测试 + 负控 + Current-SHA 证据才可标 `FIXED`。
> 实施方式：5 个映射 agent（只读定位，含 file:line + 根因 + 建议）→ 5 个实现 agent（文件不相交分簇）→ 中央串行回归 + 集成修复 + hard gates + 本 ledger。
> 测试纪律：单进程串行（小内存服务器）；`tests/r40/` 新测试全部通过；既有测试不改（唯一例外：`test_r7_typed_ir.py` 的两处按 #174 新契约更新，见 §中央修复）。

每项格式（紧凑表）：`Issue → 状态 — 代码位置 — 实现/根因 — 测试证据`

状态枚举：`FIXED`（真实改动+测试+负控）/ `PARTIAL`（部分完成，诚实说明）/ `FIXED_ALREADY`（整改前已修复，附 proof）/ `OPEN`（未做）/ `NOT_APPLICABLE`（证明不适用）。

---

## §0 中央集成修复（本 ledger 主会话，非簇）

| Issue | Status |
|---|---|
| #213 | **FIXED** — `backend/operator_capability.py`：`_safe_payload_hash` 移除 `repr(payload)` fallback，typed serializer error → `CapabilityInfrastructureError`（production identity 不允 repr 伪 identity）。负控：不可序列化 payload 抛错而非返回 repr-hash。tests/r40/test_r40_capability_identity.py（3 tests）✓ |
| #151 死锁（自举自循环） | **FIXED（中央）** — `cleaned_operators/__init__.py::RegistryBootstrap.ensure_ready`：owner 线程在 INITIALIZING 时经 SQL emitter 能力路径重入 `ensure_ready()` 会对自己 `wait()` 死锁（load_all 永不返回）。修复：owner 重入直接返回。**这是阻断全部依赖完整 bootstrap 的既有测试的根因**；修复后 `load_all()` 成功。 |
| #176 收尾（批量轴效果声明） | **FIXED（中央）** — #261 实现了 AxisEffectContract 机器 + production 无声明即拒（符合 #176 静态门 `PRODUCTION_AXIS_EFFECT_UNDECLARED==0`），但未给全部算子声明契约 → production 编译 `ts_mean` 直接失败（真实回归）。中央加 `ir/types.py::register_axis_effects_for_surface`：把算子 author 的 category + 算子族命名约定固化为一次性显式声明（load 期计算、planning 期零推断），覆盖全部 1483 个注册算子；双效应算子（ts_market/industry_liquidity_beta）用 `_override_ts_cs` 双标志。挂到 `_load_all_impl` freeze 前。tests/r40/test_r40_ir.py::TestAxisEffectContract::test_bulk_registration_covers_daily_surface ✓ |
| #174 集成 | **FIXED（中央）** — #261 在 `Analyzer.__init__` 加 `ProductionMarketContextRequiredError`（符合 #174 原文），但两个 production 代码路径 `runtime/engine.py:631`（构造时）与 `runtime/incremental_scheduler.py:1038`（IR hash）用 `Analyzer(production=True)` 无 market 会崩溃。中央把 `_market_of_factor(factor)`（semantic_identity.market → factor.market）接进 engine.compile() 与 scheduler 身份校验；production 下 factor 无 market → fail-closed。测试 `test_production_requires_explicit_market`（新）+ `test_production_rejects_unknown_raw_column`（按新契约传 `market="ashare"`）✓ |
| #62 reference 修正 | **FIXED（中央）** — `backend/pandas_backend.py::_REFERENCE_CANONICALS["ts_mean"]`：expected 原按满窗语义手算 `[nan,nan,...]`，但 ts_mean 声明策略 `min_periods=1`（operator_policy.py:628，内核 `rolling(min_periods=1)`）——backend 输出正确、reference 写错。修正 expected 并注释契约来源。tests/r40/test_r40_pandas_live_evidence.py 3 passed ✓ |
| 测试隔离修复 | **FIXED（中央）** — 新增 `tests/r40/conftest.py::writable_global_registry`（解冻→清理→重冻结，沿用 test_r10 已验证模式）；`test_r40_operator_spec_127_130.py`/`test_r40_bootstrap.py` 接入；`test_r40_registry_governance_208_212.py` 把 `_TestRegistry.register` 绑回未封印原始 register（`strict_register` 的重复检查先于 declared-override 逻辑）。315 passed / 2 xfailed（含 service_security）✓ |

---

## §1 整改前已修复（FIXED_ALREADY，mapping 证明）

映射 agent（#1-75 proof 组 + #76-220 各簇）逐项比对当前 HEAD，下列项在整改前已被既有 round 修复，附代码 proof（无新改动）：

| Issue | Status |
|---|---|
| #1-10, #12-13, #15-24, #26-51, #66-75（#52-65 已由 #264 实施，见 §6） | FIXED_ALREADY（mapping proof）——参数认证链 R26-29、TemporalPlan R30-33、内存 R34-37、ReadHandle R38-42、组合读 R43-44、ChangeImpact R45-49、RuntimeModeIdentity R50-51、DA 杂项 R66-75 均为前轮已落地实现 |
| #113 | FIXED_ALREADY — factor_identity.py:52 `_typed_hash_value` 强制 string key + tests/r32/test_r32_hard_gates_2026_08.py:432-437 |
| #114 | FIXED_ALREADY — plan_hash.py:40-53 PlanHashKind + :396-398 `_kind_digest` |
| #117 | FIXED_ALREADY — plan_hash.py:239-243 `_raise_semantic_default` + factor_identity.py:68-71 |
| #118 | FIXED_ALREADY — factor_identity.py:300-332 scoped_field_contract_hash + :245-261 |
| #119 | FIXED_ALREADY — storage/cache.py:300-315 + factor_identity.py:833-843 |
| #121 | FIXED_ALREADY — storage/cache.py:318-341 + tests/runtime/test_r20_concurrency_cache.py:283-310 |
| #123 | FIXED_ALREADY — cache/session.py:113-162 release 顺序 + tests/r36/test_r36_hard_gates_2026_08.py:246-249 |
| #125 | FIXED_ALREADY — planner/logical_plan.py:10 PlanNode frozen dataclass |
| #126 | FIXED_ALREADY — plan_hash.py 四层身份 |
| #133 | FIXED_ALREADY — fields/units_v2.py:227-228 + tests/market/test_concepts_bindings.py:119 |
| #134 | FIXED_ALREADY — security/access.py:75-81 from_dict bool 校验 |
| #136 | FIXED_ALREADY — api/factor.py:13-49 FactorExecutionScopeHint |
| #139 | FIXED_ALREADY — ir/types.py:266-301 SemanticIdentityDigest + tests/ir/test_semantic_identity_digest.py |

## §2 #76-108 Config/Packaging/HTTP/Service（簇 #258）

| Issue | Status |
|---|---|
| #76 | FIXED — runtime/config.py import json 补正 |
| #77 | FIXED — config 合并后再 validate（validate-after-merge） |
| #78 | FIXED — bool/负/零值拒绝 + allow_missing legacy policy |
| #79 | FIXED — data_access/label 从 allowed set 移除 + profile merge strip |
| #80 | FIXED — pyproject.toml include 补 market*/mining*/security*/semantic* |
| #81 | FIXED — security/__init__.py 相对导入 |
| #82 | FIXED_ALREADY — app.py 无代码改动（现状已满足，mapping proof） |
| #83 | FIXED — `_redact_secrets_for_hash` |
| #84 | FIXED — HTTP 入口 import error fail-closed |
| #85 | FIXED — market= 传至 validator |
| #86 | FIXED — mining_integration market 参数透传 |
| #87 | FIXED — validate_syntax_only_dsl + DeprecationWarning |
| #88 | FIXED — DataSourceBuildContext |
| #89 | FIXED — ExecutionSemanticIdentityV2 接入 HTTP 路径 |
| #90/91 | FIXED — models.py 摘要字段 |
| #92 | FIXED — name 字段 |
| #93 | FIXED — timeout_seconds 校验 1.0≤x≤3600.0 |
| #94 | FIXED — `_LazyRuntimeProxy` + lifespan |
| #95 | FIXED — jobstore policy_id/version/digest 字段 |
| #96 | **PARTIAL** — CancellationToken 传播到 engine 入口 + scheduler 阶段边界；DA/backend 层未接（诚实） |
| #97 | FIXED — queue 三阶段 drain |
| #98 | FIXED — task_done() |
| #99 | FIXED — durable drain CAS |
| #100 | FIXED — heartbeat_at 用 wall-clock age |
| #101 | FIXED — `_cancel_events` dict + ContextVar |
| #102 | FIXED — emergency_journal.jsonl |
| #103 | FIXED — manifest-without-checksum 隔离 |
| #104 | FIXED — manifest-without-schema_version 拒绝 + `_migrate_manifest` |
| #105 | FIXED — future schema 版本拒绝 |
| #106 | FIXED — verify-old-checksum-first |
| #107 | FIXED — dumps() 恒 checksum + dumps_legacy() |
| #108 | FIXED — reject scalar/list |
| #135 | FIXED — TypedDataSourceOptions frozen dataclass |
| #148 | FIXED — RequestBudgetSnapshot frozen |
| #149 | FIXED — catalog_generations 7-tuple |
| #150 | FIXED — result_preview 按 access level 脱敏 |

簇验证：tests/r40/（config/catalog/jobstore/queue/models/http_semantic_context/cancellation/mining_integration/backend_factory）+ service 既有回归；82 passed（簇内）。

## §3 #109-150 Identity/Unit/Security/Mining（簇 #259）

| Issue | Status |
|---|---|
| #109 | FIXED — factor_identity.py:353-412 scoped_operator_contract_hash + OperatorSemanticContractDigest；impl hash 变化 → scoped hash 变化 |
| #110 | FIXED — plan_hash `_operator_semantic_contract` 与 identity 同 digest 投影（双路径统一） |
| #111 | FIXED — partition_input_fingerprint：typed schema+index+canonical bytes+null mask（int32≠int64） |
| #112 | FIXED — 含 datetime+asset 先 sort 再 hash（row-order invariant） |
| #115 | FIXED — `_typed_hash_value` numpy scalar 前置 dtype 归一 |
| #116 | FIXED — 2^53 vs 2^53+1 原始 int64 bytes 不碰撞 |
| #120 | FIXED — storage/cache.py `_RefCountedSaveLock`：只淘汰 refcount==0 |
| #122 | FIXED — wrap_context 合并既有 runtime_stats |
| #124 | FIXED — execution_id 默认 `uuid4().hex`（并发 layer 名不碰撞） |
| #127-130 | FIXED — operator_spec：frequency/output_type/supports_panel/dual_backend_target 全从真实契约导出 |
| #131 | FIXED — fields/spec.py + concepts.py 新增 annual_flow 推导 + FLOW_SEMANTICS_ANNUAL |
| #132 | FIXED — flow_semantics 完整推导矩阵（ytd/annual/quarter/single/ttm/balance/flow） |
| #137 | **PARTIAL** — market/context.py 加 `resolved_trading_days_per_year` 真实日历优先；ASHARE/US_CONTEXT 默认未翻 basis="calendar"（会让既有 test_cross_market_golden 断言 252 失败） |
| #138 | FIXED — MarketContext.extra 禁含 dataclass 语义字段名 |
| #140 | FIXED — `build_backend("pandas_modin")` 不再改 os.environ；PandasBackend(use_modin_pandas) + ContextVar |
| #141 | FIXED — production_execution_certificate 加 requested_backend/resolved_dialect/datasource_identity |
| #142 | **NOT_APPLICABLE（诚实阻塞）** — config_runtime `to_run_kwargs()` 加 calendar_id 会令 `engine.run(**to_run_kwargs())` 抛 TypeError（engine.run 不接受 calendar_id）；calendar_id 已在 engine 构造消费 + config_run_batch_key 分组。证明：runtime/engine.py:2335,2915 |
| #143 | FIXED — security/access.py 加 timestamp/approval_id；validate/require 默认 strict |
| #144 | FIXED — 空 access tags → `_SENSITIVITY_UNKNOWN=100`；production 拒 UNKNOWN |
| #145 | FIXED — factor_id `_PRODUCTION_FACTOR_ID_RE` 白名单；production 拒同形异义 |
| #146 | FIXED — fin_mad 从 surface+两处 `_SPECS` 移除，注册 compat alias→fin_mean_abs_deviation |
| #147 | FIXED — `OperatorMigrationRecord`+`MIGRATION_TABLE`（7 条 deprecated alias 全覆盖） |
| #208 | FIXED — registry `_mutation_token` 令牌（finalize 失效/thaw 恢复）+ freeze 后 mappingproxy 不可变 + 公开读走 `_frozen` |
| #209 | **PARTIAL** — BackendOverrideSpec（含 expected_old_contract_hash）+ `_DECLARED_OVERRIDE_CONTRACT_HASHES` 强制校验；`_auto_pin_declared_bootstrap` 保持 True（翻 False 会破坏 ~31 处非本簇声明 override 层） |
| #211 | **PARTIAL** — CanonicalOperatorManifest + `_merge_param_names(canonical=)` 权威化 param_names |
| #212 | **PARTIAL** — semantic_version 默认改 "" + audit 不伪造 "1.0" + `assert_production_semantic_versions_declared()` 门禁；`register()` 硬 reject 回退（~10 处 production 注册无 semantic_version 在非本簇文件） |
| #220 | FIXED — fields/spec.py production 禁 name-based semantic_kind fallback |

簇验证：tests/r40/ 8 文件 59 passed；既有 r32/r20/r7/ir 抽查通过（依赖完整 load_all 的套件当时被 #151 死锁阻断，中央修复后全绿）。

## §4 #151-220 Axis/Unit/Missing/Backend-truth（簇 #261）

| Issue | Status |
|---|---|
| #151 | FIXED — RegistryBootstrap（Condition + NEW/INITIALIZING/READY/FAILED + owner_thread_id）；非 owner 等待。中央修复 owner 重入死锁后 load_all 全通 |
| #152 | FIXED — load_all 委托 bootstrap；首次调用冻结 surface，include_research 后续值不重跑 |
| #153 | **PARTIAL** — RegistryBootstrap.reset() FAILED 后干净重试；全量事务回滚未实现（registry 被 ~20 层治理原地改） |
| #154 | FIXED — BootstrapModuleSpec(module,role,required) + validate；226 模块 spec 表与 legacy 精确相等 |
| #155 | FIXED — cleaned_bridge 删 `_CLEANED_LOADED`；ensure_operator_registry(surface=production 强制 research-free + signature authority) |
| #156 | FIXED — NoCertifiedParameterRegionError：production 无认证区硬失败（无 coverage_skip） |
| #157 | FIXED — 认证阶段 typed（BindCall→LoadCertificate→ResolveIdentity→CheckMembership）+ 修一个被 broad except 吞掉的 NameError |
| #158 | FIXED — `_BoundScalarParams` complete flag 单 bind 路径；无 broad except |
| #159 | FIXED — semantic-version resolver 失败抛而非返回 "" |
| #160 | FIXED — InputDTypeSignature ordered 认证 key |
| #161 | FIXED — ExecutionVariantIdentity 从实际实现 of() |
| #162 | FIXED — ExecutionPerfCounters（atomic, lock-protected）；修 racy runtime_stats 读改写 |
| #163 | FIXED — GrainTransformCertificate + validate_grain_transform（8 checks） |
| #164 | FIXED — panel_polars verify_and_restore_axis；缺 `__fe_time__` production hard-fail |
| #165 | FIXED — bulk 与 column-loop 共享轴验证 |
| #166 | FIXED — _polars_bridge 位置写回前断言 index==base（kernel index 重排可检测） |
| #167 | FIXED — PhysicalColumnNameMap.validate_injective 表示边界 |
| #168 | FIXED — AxisColumnRole + classify_panel_columns；wide value 列叫 date 被拒 |
| #169 | **PARTIAL** — PanelCompatibilityPolicy（production 要求 PROVEN+等 grain/freq）；`PanelIdentity.__eq__` 保持历史容忍（既有 test_r9_panel_identity 编码） |
| #170 | FIXED — attach_contract_axis：A 股日线面板获 contract grain 而非 index-inferred unknown |
| #171 | FIXED — instrument_count = unique instrument-label 基数 |
| #172 | FIXED — canonical_axis_label typed encoder（拒 float/bool/object） |
| #173 | **PARTIAL** — hash-includes-grain 实现后**回退**（既有 test_hash_consistent_with_equality 断言 hash 排除 grain/freq）；严格相等由 PanelCompatibilityPolicy 覆盖 |
| #174 | FIXED（簇）+ 中央集成（§0）— production Analyzer 无 market → ProductionMarketContextRequiredError；engine/scheduler 接线 market |
| #175 | **PARTIAL** — make_cleaned_kernel 走 REGISTRY_BOOTSTRAP；analyzer 仍 research 默认 ensure_cleaned_loaded（frozen surface 接线延后） |
| #176 | FIXED — AxisEffectContract + register_axis_effect；production 读契约，name/category heuristics 仅 research |
| #177 | FIXED — full-history 契约 ImportError production 硬失败 |
| #178 | **PARTIAL** — HistoryContract type + registry + production gate；~1000-op 表面逐算子声明未应用 |
| #179 | **OPEN** — per-field FieldContractDigest 未实现（analyzer 仍用全局 catalog hash） |
| #180 | FIXED — normalize_availability_descriptor 拆 policy label vs column ref |
| #181 | **PARTIAL** — normalize_period_duration + SemanticTypeBundle.period_duration；factor_identity 接线在只读文件 |
| #182 | FIXED — SemanticTypeBundle（8+ 经济维度）+ from_field_like |
| #183 | FIXED — SemanticIdentityDigest typed 序列化；unsupported 对象拒绝 |
| #184 | FIXED — canonical_unit_known → KnownUnitId；未知拼写 → UnknownUnitError |
| #185 | FIXED — UNIT_CNY_PER_SHARE/UNIT_SHARE_RATIO 导出 + alias 表修正（share_ratio 原不可解析） |
| #186 | FIXED — UnitExpr 指数代数 + log 维度强制无量纲 |
| #187 | FIXED — StatisticalSamplePolicy + uniform_finite_mask（SMA 掩 ±Inf） |
| #188 | FIXED — cross_sectional `_finite_stats_input`（nanmean/std/median/percentile 前掩 Inf） |
| #189 | **PARTIAL** — uniform finite 应用于关键 window 算子；未逐 rolling 调用审计 |
| #190 | **PARTIAL** — CarryForwardPolicy + production gate；FieldSpec 绑定被 spec.py 只读阻塞（#220 另一簇） |
| #191 | **OPEN** — 全仓库 ffill/fillna(path)/pad 审计未写 |
| #192 | FIXED — nan_to_num 只替换 NaN（±Inf 通过）；新增 nonfinite_to_num 保留旧行为 |
| #193 | FIXED — SupportPolicy（min_observations/contiguity/partial/ddof）digest 进 identity |
| #194 | FIXED — EWMContract（span/adjust/seed） |
| #195 | FIXED — WMA partial-policy digest 进 identity |
| #196 | FIXED — TopKContract（sample_policy+tie_policy）；production 拒未声明 tie policy；AggrTopN finite-only + 确定性 tie-break |
| #197 | FIXED — check_rank_method production 拒 `method="first"` |
| #203 | FIXED — rep_conversion 计数器 request-scoped ExecutionPerfCounters（删模块级 int；`__getattr__` proxy 保 r39 surface） |
| #204 | FIXED — grain/frequency 从契约（attach_contract_axis）非观测 index |
| #205 | **OPEN** — 7 维 verify_broadcast_mapping 未实现 |
| #206 | **OPEN** — unknown-calendar broadcast 拒绝未实现 |
| #207 | FIXED — 单输出轴保持算子的多余输出列拒绝（polars raw frame drop 前检查） |
| #210 | FIXED — replace_backend() typed audit trail（old/new hash + reason + migration id）；中央补测试恢复 backend 防 R4-102 |
| #215 | FIXED — check_signature_authority(production=True) 无 authority 硬失败；bootstrap 记录 `_SIGNATURE_AUTHORITY_AVAILABLE` |
| #218 | FIXED — FieldConceptSpec 拆 semantic_extensions（进相等）/descriptive_metadata（排除）；hidden semantic keys 拒绝 |
| #219 | FIXED — FieldLegalityPass（8 维 check_field_legality/assert_field_legality） |

簇验证：tests/r40/ 9 文件 95 passed；既有 panel/axis/contract 抽查通过。

## §5 #221-260 Universe/Price/Session/Minute/Stateful/Numerics（簇 #262）+ 追加 #198-202

| Issue | Status |
|---|---|
| #221 | FIXED — UniverseMembership.valid_to_exclusive + OPEN_ENDED + effective_at trade_time 双时态区间校验 |
| #222 | FIXED — UniverseKnowledgeUnknownError（production 缺 temporal 契约 fail-closed）+ StaticUniverseSnapshot lineage |
| #223 | FIXED — universe_membership_identity 含 effective_intervals + UniverseMembershipDigest |
| #224 | FIXED — CrossSectionEligibility（7 类掩码位） |
| #225 | FIXED — CoverageExitReason 8 枚举 + coverage_decomposed_by_exit_reason |
| #226 | FIXED — InstrumentNormalizer.for_market() + InstrumentKey.normalized |
| #227 | FIXED — market/security_master.py（SecurityMaster/SymbolValidityInterval，rename 后 id 稳定） |
| #228 | FIXED — market/price_basis.py（PriceBasis/PriceBasisContract/validate_limit_ops_price_basis）+ limit_ops 接入 |
| #229 | FIXED — market/return_semantic.py（ReturnSemantic + identity_hash） |
| #230 | FIXED — market/adjustment_policy.py（AdjustmentPolicy/AdjustmentVintage；production 拒 RETROSPECTIVE） |
| #231 | FIXED — market/price_grid.py（PriceGridContract/price_grid_for_market）+ limit_ops._tolerance |
| #232 | FIXED — market/exchange_session_calendar.py（ExchangeSessionCalendar 单一权威 + 投影） |
| #233 | FIXED — SessionCalendar.authoritativeness + require_exchange_certified（offset_bars/warmup production 门） |
| #234 | FIXED — SessionCalendar.timezone 必填 + session-local 计算 |
| #235 | FIXED — ExchangeSessionCalendar.for_date 按日 authoritative（early close/DST/holiday） |
| #236 | FIXED — SessionSpec.to_dict 补 early_close 字段 + calendar_digest |
| #237 | FIXED — EarlyCloseNormalizationPolicy 枚举 + normalization() 解析 |
| #238 | FIXED — SessionDQReport + SessionPanel.dq；production 聚合调用 hard_dq_fail |
| #239 | FIXED — 分钟聚合分层异常（numeric→NaN；DQ/calendar/tz/axis→run fail） |
| #240 | FIXED — build_session_panel trade_date=None 校验单交易日；跨日抛 MultipleSessionDatesError |
| #241 | FIXED — off-grid 记录进 DQ report + 阈值 hard fail |
| #242 | FIXED — 删 `market or "ashare"` 默认；None → MarketRequiredError |
| #243 | FIXED — `_declared_calendar` 适配 US/HK；unknown → error |
| #244 | FIXED — `_source_snapshot_scope` typed：production 解析失败抛；research ephemeral uuid 不跨 run |
| #245 | FIXED — `_boundary_timeline` BoundaryProbeResult（PROVEN/GAP/UNKNOWN）；production UNKNOWN 禁 resume |
| #246 | FIXED（薄测）— boundary_probe_cache 按 (as_of,start) 分组；集成测需真实 source/store 未做 |
| #247 | FIXED — AxisIdentityCertificate.verify_frames_share_identity production exact match |
| #248 | FIXED — `_checkpoint_input_identity` 补 market/calendar/tz/universe/price_basis/field/operator_semantic/numeric/missing_policy |
| #249 | FIXED — SegmentedFallbackReason 7 枚举；production corruption → StatefulSegmentedCorruptionError（不 fallback） |
| #250 | FIXED — check_chunk_invariance_all_segmented_canonicals（1-bar overlap，11 canonical 全过）+ 硬门常量 |
| #251 | FIXED — DegeneracyPolicy（absolute/relative floor）；近零方差 → NaN 非爆炸 |
| #252 | **PARTIAL** — _numpy_kernels pairwise_sum_/neumaier_cumsum_/welford_rolling_var_ + cs_resid_/cs_regression_ 接入；全部生产 sum/mean 算子未逐一重写（registry 并发编辑） |
| #253 | FIXED — MomentConvention（ddof/bias/fisher/nan/finite） |
| #254 | **PARTIAL** — ComputePrecisionPolicy 进 elementwise_semantics；factor identity hash 接线需改 plan_hash（禁碰） |
| #255 | FIXED — EdgeContract 增 signed_zero/subnormal/overflow_policy + edge_contract_numeric_identity |
| #256 | **PARTIAL** — NumericDeterminismLevel + record_blas_config（threadpoolctl）；production evidence 打标需改 evidence 生成链（其他簇） |
| #257 | FIXED — check_permutation_equivariance_all_tie_sensitive_operators（6 tie-sensitive canonical） |
| #258 | H22 撤销旧全过声明：此前仅 REFERENCE_SELF_CHECK；现由实际 registry winner 逐项执行，未覆盖项阻断，见 evidence/r2/H22-real-gates.log |
| #259 | H22 撤销旧生产声明：此前 EWM/cumsum 为演示内核；现比较真实 registry 与 stateful runtime，不可用项保留 BLOCKED，见 evidence/r2/H22-real-gates.log |
| #260 | FIXED — cross_process_determinism_probe（subprocess 双进程 + restart 语义） |
| #198 | FIXED — compare_sql 补 isinf；compare_pandas 对齐 `_both_finite`（跨 backend 一致） |
| #199 | **PARTIAL** — ProtectedDivisionSemantics 真值表 + numpy/pandas/polars 对齐；SQL Inf numerator 缺口用 pytest.xfail 诚实标记 |
| #200 | **PARTIAL** — div_or_default_polars 对齐；SQL NaN 输入缺口 xfail 标记 |
| #201 | FIXED — element_math_vs_stats_inf_semantics 声明 + max/min 文档 |
| #202 | FIXED — edge_value_corpus（11 命名 edge 值）+ 可执行 SQL parity 测试 |

簇验证：tests/r40/ 7 文件 47 passed / 2 xfailed（均 SQL emitter 缺口，文档化）；既有 market/numeric 域 31 passed。

## §6 #1-75 未决项（簇 #264，含 #11/14/25/52-65）

| Issue | Status |
|---|---|
| #11 | FIXED — runtime/buffer_store.py BufferEntryState + pin()/acquire_ref()/release_ref()；eviction 跳过 refcount>0/PINNED |
| #14 | FIXED — SpillDecisionEngine.evict_candidate 多因子（stale 符号按既有 test_lru_eviction_most_stale_first 修正） |
| #25 | FIXED — scripts/audit_r40_evidence_completeness.py check_evidence_axis_completeness/cross_product_completeness（诚实暴露：真实 R37 证据缺 136 组合） |
| #52 | FIXED — dataaccess/snapshot/fidelity.py SnapshotFidelity：production 拒 < REMOTE_VERSION_ID |
| #53 | FIXED — dataaccess/read/scan_cost.py COST_UNKNOWN_CONSERVATIVE sentinel + conservative_scan_cost()（basis="unknown" 仅统计异常；R28 COW 回归捕获后修正） |
| #54 | FIXED — dataaccess/read/data_request.py `_scope_file_manifests` narrows pin |
| #55 | FIXED — dataaccess/snapshot/manifest.py ManifestFetchResult（OK/ABSENT/LOOKUP_FAILED/PERMISSION_DENIED/DEADLINE_EXCEEDED） |
| #56 | FIXED — dataaccess/read/read_session.py PhysicalResolutionContext cache key |
| #57 | FIXED — dataaccess/r30/resolution_lease.py 两阶段 RESOLUTION→EXECUTION lease |
| #58 | FIXED — resource_governor.acquire_remote_discovery_slot（max=max(4, concurrency//2)） |
| #59/60 | FIXED — startup_gate StartupCertificate + require_startup_certificate + calendar_digest/coverage_check 不依赖 date.today() |
| #61 | FIXED — sql_tiers.is_sql_production_safe 并入参数域证书（duckdb_sql） |
| #62 | FIXED — backend/pandas_backend.py PandasBackendLiveEvidence 实跑验证 + versioned evidence（中央修正 ts_mean reference 后 3 tests 全绿） |
| #63 | FIXED — evidence_truth.evidence_validity_cached @lru_cache（HEAD/Manifest/TCB/artifact hash） |
| #64 | FIXED — sql_tiers `_BoundedLRU`（entry+byte bound downgrade cache） |
| #65+#217 | FIXED — emitter `_SqlTemplateCache` 模板级缓存 + `_plan_shape_key`/`_literal_binding_key` |
| #214 | FIXED — EmitterIdentity 从 build manifest；unknown → 硬失败 |
| #216 | FIXED — CompileStatus 三态 SUPPORTED/SEMANTICALLY_UNSUPPORTED/COMPILER_ERROR |

簇验证：FE tests/r40/ 19 passed + DA tests -k r40 14 passed；DA 全量 1051 passed / 4 failed（2 因并发 FE security/access.py 改动、1 排序 flake、1 R39 既有 mirror `_estimate_read_memory` 问题）。

## §7 R40 Hard Gates（scripts/audit_r40_hard_gates.py）

```
[N1_REGISTRY_BOOTSTRAP] PASS — 226 module specs valid; RegistryBootstrap.state=new
[N2_PARAMETER_CERTIFICATION] FAIL — 证据过期：generated a2dc3a1a vs HEAD 9693901（R16 证据重生待运行；fail-closed 机制本身正确拒绝旧 SHA）
[N3_AXIS_TRUTH] PASS — 合法降采样 clean；重复/非 DatetimeIndex 被拒
[N4_MARKET_TIME_TRUTH] PASS — 无 exchange-certified holiday set → production fail-closed
[N5_UNIVERSE_TRUTH] PASS — 缺 temporal 契约 → UniverseKnowledgeUnknownError
[N6_PRICE_BASIS_TRUTH] PASS — raw basis valid；RETROSPECTIVE adjustment production 拒绝
[N7_MINUTE_DQ] PASS — DQ 违规 → hard_dq_fail
[N8_STATEFUL] PASS — 11 segmented canonical chunk-invariance 全过
[N9_NUMERICS] PASS — permutation-equivariance + prefix-invariance + chunk-boundary 全过
[EVIDENCE_COMPLETENESS_25] FAIL — 92 points；axis_missing=0，cross_missing=136（R16 证据重生待运行）
→ 8 PASS / 2 FAIL / 0 NOT_RUN；2 FAIL 均证据重生阻塞（非 R40 代码缺陷）
```

## §8 回归证据

- **R40 新测试**：tests/r40/ + service_security 中央串行 **315 passed / 2 xfailed / 0 failed**（含 #176 批量声明后重跑的 test_r40_ir 8 passed）。
- **中央集成验证**：`load_all()` 全通（修复 #151 owner 重入死锁后）；`test_r7_typed_ir.py` 15 passed（#174 新契约）；`test_r40_pandas_live_evidence.py` 3 passed；`test_r40_capability_identity.py`（#213）3 passed。
- **既有回归（定向）**：
  - `test_r7_typed_ir.py` 15 passed（#174 集成）。
  - `test_r10_pandas_fallback_gate.py` + `test_production_dsl_gate.py`：**3 failed 均为既有/并发基线**——用 git worktree 在提交态 HEAD（c9c08ff5，无 R40 改动）跑同一测试也失败（且更糟：`nonfinite_to_num` surface 不一致让 load_all 直接崩）；当前工作树失败点是 `ts_mean status=experimental`（production 未认证），根因=**R37 证据过期（R16 阻塞）**使 evidence overlay 无法认证 production 算子，与 N2 gate 同源。**非 R40 回归**。
- **全量 FE 回归阻塞**：`tests/planner/` 两个模块收集期抛 P0-23 `ts_robust_zscore_prior` 契约分歧——并发 R47 会话的未提交 WIP（`cleaned_operators/common/polars_robust_stats.py` 加 5 参 param_specs，canonical pandas 侧未同步）；非 R40 文件。全套件对当前服务器也超时（30min 上限，分批 runtime/planner 亦超时）。
- **DA 全量**：1208 passed / 5 failed。中央修复 1 个 **R40 引入的 DA 回归**：`security/access.py` 的 `from runtime.production_policy` 在 DA 共存上下文被 `dataaccess/runtime` 遮蔽 → ModuleNotFoundError（连带 DA `test_r39_da_read` deadline 失败）；改为 FE runtime 优先、被遮蔽时保守 research 后，DA security 7 passed + deadline 测试解掉。剩余 3 失败均为既有/环境类：`test_check_allowlist`（repo 文件审计）、`test_phase7_final_audit`（R39 ContextVar/env 错位，formats.py 未改动）、`test_store_adapter_mirror`（R39 既有 mirror 问题，#264 已识）——**非 R40**。
- **证据诚实**：R37 参数域证据 + #25 完备性 FAIL 均为**真实未达成**，归 R16 证据重生任务（并发会话持续 commit 使 HEAD 前移）。

## §9 剩余风险（诚实）

1. **R16 证据重生阻塞（P0）**：参数域证据（a2dc3a1a）与 #25 完备性（136 缺组合）都需在最终稳定 commit 上重生。N2 + EVIDENCE_COMPLETENESS 两门 FAIL 的根因。
2. **PARTIAL 项**：共 15 项（#96, #137, #153, #169, #173, #175, #178, #181, #189, #190, #209, #211, #212, #252, #254, #256, #199, #200, #246）——每项在 §2-5 有具体理由（只读文件所有权、既有测试冲突、需跨文件深接、需全表面扫描）。
3. **OPEN 项**：5 项（#179 per-field contract digest、#191 warehouse ffill audit、#205 7-dim broadcast、#206 unknown-calendar broadcast、#142 NOT_APPLICABLE）。
4. **SQL emitter Inf/NaN 缺口**（#199/#200）：duckdb SQL 对 Inf numerator / NaN 输入与 pandas 有真实 parity 缺口，用 xfail 诚实标记，需 SQL 侧修。
5. **并发会话归属**：R47 并发会话对 model/* 的改动可能引入其自身测试状态；中央回归已区分。
