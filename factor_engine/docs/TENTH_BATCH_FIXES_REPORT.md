# FactorEngine Review #10 — 整改完成报告

审计基线：`main@a2dc80cf52a9473a518983d4a38e3d291bc041b9`（最新 main；本轮执行期间并发会话持续提交，HEAD 已前移至 ecffd56，本报告所有改动基于最新工作树验证）。

本轮原则：已修问题不回滚；production 一律 fail-closed；不留 fail-open 路径；全部永久 regression tests；真实报告既有失败。

---

## A. 逐项状态总表

### 一、Production / PIT / Execution Contract

| 项 | 状态 | 修复 |
|---|---|---|
| #1 DataSourceBuildContext 进 factory | **fixed** | `storage/factory.py` 新增 `DataSourceBuildContext`（run_mode/market/calendar_id/timezone/pit_enforce/enforce_mining_gate/snapshot_policy/coverage_policy），`build_data_source(config, build_context=…)` 递归注入 composite/long_table/intraday 子源；`FactorEngine.from_loaded_config` + `config_runtime.config_data_scope_key` 从 RunConfig 构造 context。production config 自动开启 strict field/mining/PIT gate，不再靠 YAML 记得。`_ensure_no_extra_options` 放行 per-source run_mode/production/strict_unknown_fields/enforce_mining_gate/snapshot_now_only/mining_coverage_threshold。7 tests |
| #2 PIT 四层三态 fail-closed | **fixed（并发会话+本会话）** | dataset/table 层 UNKNOWN(None) 不再当 True；`assert_four_layer_pit` production 下 UNKNOWN==reject，research 告警降级；`_NO_KNOWLEDGE_TIME_FUNDAMENTALS` 数据集 production 禁 PIT。IR semantic lattice `pit_safe` 三态（见 #6）。tests/runtime/test_r10_pit_tristate.py 5 tests |
| #3 strict_unknown_fields=False 不能降级 production | **fixed** | `DataAccessSource.__init__` 重写：production 是地板，显式 False 无法关掉 fail-closed；research 显式 True 可以更严。工厂只转发 run_mode + mining gate，production/strict 由构造器从 run_mode 推导。test_r10_build_context |
| #4 execution_contract exception→stateless | **fixed** | `runtime/execution_contract.py`：新增 `ExecutionContractResolutionError`；`_resolve/_metadata/_checkpoint_spec` 加 `strict=`；production 下 registry/checkpoint lookup 异常 raise；research 显式标记 `resolution_error="UNKNOWN_EXECUTION_CONTRACT"` + `chunking="unknown"`，`requires_full_history` 把 unknown 当 full-history（绝不静默 stateless）。6 tests |
| #5 typed broadcast MultiIndex/Polars long 强验证 | **fixed（PanelIdentity 轴验证）** | `_polars_bridge.PanelIdentity.from_frame`：MultiIndex long panel 现在提取真实 time-axis hash（不再 None）+ instrument level hash（非 time 层+value 列），日期轴移动/标的池变化 loud-fail。5 tests |
| #6 semantic lattice pit_safe 缺失默认 | **fixed** | `ir/types.py` lattice `pit_safe` 三态 AND（任何 proven-unsafe→False；有 None→None=UNKNOWN，绝不当 safe）；`ir/analyzer.py` 无声明节点默认 `pit_safe=None`。production 四层门在数据层拒绝 UNKNOWN。5 tests |
| #7 production pandas fallback hard gate 真调用 | **fixed** | `runtime/engine.py run()` 在 backend execute + `assert_production_fastpath_runtime` 之后、接受结果之前调用 `assert_no_production_pandas_fallbacks`（原已存在但只 log）；`FactorEngine.__init__` 新增 `production_fallback_policy` 参数透传 ExecutionContext（error 默认 / warn 显式放行）。后端 `polars_long_backend` 也在 post-execute 调用。5 tests |
| #8 next_trading_day production 无市场日历必须拒 | **fixed** | `pit_contract.pit_asof_join` / `select_visible_row_bundles` 新增 `production=`；production + next_trading_day + 无 market_calendar → raise（不再 fallback 决策网格）；修复 `_market_visible_shift` tz 检测 bug（`cal.tz.zone` 对 UTC-aware 是 None → 误走 `tz_localize` 抛 "Already tz-aware"，改 `cal.tz is not None`）。tests/storage/test_r10_pit_closure.py |

### 二、YAML / Runtime Config

| 项 | 状态 | 修复 |
|---|---|---|
| #9 unknown key fail-fast | **fixed** | `runtime/config.py` 新增 `_forbid_unknown`（extra="forbid"）；top-level + factor/backend/engine/run/dq/pit/pipeline/materialization/incremental 全 section 校验；`mdoe: production` typo → 立即失败。兼容既有 profile 的 `data_access`/`label` section |
| #10 bool 严格解析 | **fixed** | `_strict_bool`：`"false"`→False；拒绝 Python truthiness；只接受 bool/0/1/规范字符串 |
| #11 int 不截断 | **fixed** | `_strict_int`：5.9 → raise（不再 int(5.9)=5）；接受整数/整值 float/整值字符串 |
| #12 MarketExecutionSpec | **fixed** | `RunConfig.market_execution` property 产出 `MarketExecutionSpec(market/calendar_id/timezone/decision_time/bar_timestamp_role)`；RunConfig 增 timezone/decision_time/bar_timestamp_role 独立字段。9 tests |

### 三、Registry / Semantic Identity

| 项 | 状态 | 修复 |
|---|---|---|
| #13 implementation hash 排序碰撞 | **fixed（agent）** | `registry._code_payload` 改按 disassembled bytecode 的 `(opcode, resolved_operand)` 序列计算（不排序 co_consts/co_names）；`2*x+3` vs `3*x+2` 不碰撞；跨进程稳定；nested code 保序递归。tests/operators/test_r10_registry_identity.py |
| #14 `_freeze_value` nested repr | **fixed（agent）** | object ndarray/MultiIndex/custom 元素 canonical recursive freeze；不可冻结 raise TypeError 而非 repr 依赖 digest。同上 |
| #15 overlay/rename 保留完整 contract | **fixed（agent）** | `register_catalog_only(merge_existing=True)` / `rename_canonical` 保留 param_specs/panel/scalar/units/aliases/grain/availability/semantic_version/history；合并前验证 rich contract 等价，不等价 `ValueError`。同上 |
| #16 FactorSemanticIdentity | **fixed（agent）** | `runtime/factor_identity.py`：14 字段 frozen dataclass + `identity_digest()`（SHA-256，任何字段变化→新 digest）+ `compute_factor_identity(plan, ctx)`（复用 compute_ir_hash / operator/field catalog hash / source_dependency_hash）+ `FactorIdentityMismatch`。tests/runtime/test_r10_factor_identity.py |
| #17 production 禁 `__no_hash_provided__` | **fixed（agent）** | `materializer.materialize` 增 `production=`；production + `NO_FACTOR_IDENTITY` → raise；research 保持 legacy sentinel。同上 |

### 四、参数 / 搜索空间

| 项 | 状态 | 修复 |
|---|---|---|
| #18 active_when 无 controller 用 default | **fixed（agent）** | `parameter_canonicalizer`：`_resolve_controller_default()` 读 controller ParamSpec.default；inactive 参数显式非默认值 → reject。tests/operators/test_r10_paramspec.py（22 tests） |
| #19 alias 单一逻辑绑定 | **fixed（agent）** | canonical+alias 或两 alias 同绑 → `ValueError: duplicate logical binding`；alias 映射到同一 canonical key |
| #20 COMPOSITIONAL vs TERMINAL_RANK sensitivity | **fixed（agent）** | `sensitivity_mode` 区分；compositional 绑定 raw-value behavior（f→2f/f+1 不判 insensitive）；terminal-rank 保留 scale/shift-invariant |
| #28 显式 panel metadata > name heuristic | **fixed（agent）** | `classify_panel_scalar_params`：显式 panel_params/input_fields/scalar_params 无条件胜出；numeric-control-name heuristic 仅 legacy fallback |
| #29 contract 构造失败 fail-closed | **fixed（agent）** | `build_operator_contract()` 失败 → `CONTRACT_ERROR`，绝不 name-guess fallback；`operator_contract_searchable()` 排除 |
| #30 relational 异常 = CONTRACT_ERROR | **fixed（agent）** | predicate raise → `ParameterContractError`/`CONTRACT_ERROR`，不再当 NOT_APPLICABLE |

### 五、搜索去重 / 多样性

| 项 | 状态 | 修复 |
|---|---|---|
| #21 multi-regime per-day dedup | **fixed（agent）** | 12 regimes（gaussian/heavy_tail/trend/mean_revert/ties/gaps 含缺行/positive_only/event_bool/event_signed/group/ohlc/ashare）；per-day Spearman_t/top/bottom-decile/position overlap → median/p10/p90/duplicate-day ratio；仅多数 regime+日期都等价才删。tests/operators/test_r10_dedup_regimes.py（13 tests） |
| #22 FactorKind 强制 | **fixed（agent）** | `dedup_bucket` 前缀 factor_kind（ALPHA/CONDITION/EVENT/GLOBAL_STATE 永不同组）；per-kind 签名（CS rank / TS rank / bool state / sign state） |
| #23 transition/state-label signature | **fixed（agent）** | `transition_signature`/`state_label_signature`；CONDITION/EVENT 用 state-label 一致性门控 |

### 六、成本 / 资源模型

| 项 | 状态 | 修复 |
|---|---|---|
| #31 declared cost contract 非免死金牌 | **fixed（agent）** | `search_budget_gate`：valid contract ∧ est runtime≤budget ∧ est memory≤budget 才 searchable；declared 只保证估算准，不授超预算权 |
| #32 shape 进入 cost formula | **fixed（agent）** | `CostShape(T,N,window,group_size,n_features,k,bins,backend)` + per-category 复杂度（TS rolling O(TNW)、CS sort O(TN log N)、KNN/graph O(TN²)、kernel Gram O(W²) mem、matrix O(p³)、transport、group breadth） |
| #33 kernel Gram memory O(W²) | **fixed（agent）** | 不再 O(W) |
| #35 cost UNKNOWN → unsearchable | **fixed（agent）** | `estimate_runtime` raise `CostUnknownError`；production search 不回落便宜默认 |
| #34 research-only 去 prefix | **fixed（agent）** | tier 判定走 AuthoringTier/surface，不再 name prefix。tests/operators/test_r10_cost_model.py |

### 七、Audit Framework

| 项 | 状态 | 修复 |
|---|---|---|
| #36 ColumnPermutation 逆排列 | **fixed（agent）** | `out.reindex(columns=original_columns)` / positional inverse；`['ZZZ','AAA','MNO']` 不再 false-fail |
| #37 GroupVintage 读 membership semantics | **fixed（agent）** | `_membership_semantics(op)`：historical_membership vs current_members_retrospective 分别审计 |
| #38 TemporalExclusion 读 MissingPolicy | **fixed（agent）** | carry-state：缺失输入 carry，不误报 NaN-required |
| #39 EffectiveSample 检查 DOF/effective_n | **fixed（agent）** | required_n/estimator_dof 门控，短面板不再仅靠 warmup 通过 |
| #40 ScaleAudit 读 MetamorphicContract | **fixed（agent）** | scale_invariant/translation_invariant/sign_equivariant/unit_covariant/none；undeclared 保持 legacy 断言（R9 测试不破），declared none 不强制 scale 不变 |
| #41 Complexity 拆 model + benchmark | **fixed（agent）** | `audit_complexity_vs_param`（确定性 cost-model）+ `audit_performance_benchmark`（wall-clock 仅显式 max_seconds 才 flag） |
| #42 Golden 0 overlap 不 PASS | **fixed（agent）** | `inconclusive` 状态。tests/operators/test_r10_audits_closure.py（16 tests） |

### 八、Composite / Multi-source

| 项 | 状态 | 修复 |
|---|---|---|
| #43 exact join key_matched 统计 | **fixed（agent）** | `key_matched_rows = len(anchor_index ∩ series.index)`，源无该 key 不再虚报 |
| #44 Composite cache 随子 snapshot 失效 | **fixed（agent）** | `_child_snapshot_manifest()`/`_invalidate_if_snapshot_changed()`：任一子源 snapshot_id 变化 → 清 `_column_cache`+`_anchor_index_cache` |
| #45 pit_asof_join 同 key tie-break | **fixed（本会话）** | 确定性全序 `[available_at, instrument, period_end, revision_id]`；merge_asof 取最新 period/revision，绝不靠原行序。tests/storage/test_r10_pit_closure.py |
| #46 same_day 需 TIMESTAMP 精度 | **fixed（本会话）** | `_same_day_precision_check`：date-only(midnight) PubDate production raise、research 告警保 legacy |

### 九、Materializer / Incremental

| 项 | 状态 | 修复 |
|---|---|---|
| #47 partition resume 绑 identity | **fixed（agent）** | `.identity.json` sidecar：identity_digest/source_snapshot/dependency_hash/partition_input_fingerprint/run_generation；任一不符或缺失 → production 重算，research legacy skip |
| #48 all-NaN 输出清旧值 | **fixed（agent）** | production+incremental 自动写 tombstone（不靠 caller null_overwrite）；真空帧仍 rows_written=0；research 保持历史 skip |
| #49 generation-atomic | **deferred** | 见 D（需 catalog/staging 重构，与 #51 的 per-factor engine 重建一并做） |
| #50 FactorDependencyEdge | **fixed（agent）** | `factor_dependency_edge` 表（factor_id/source_dataset/field_id/physical_field/transform/snapshot_semantics/lookback）；一因子多 edge |
| #51 DataEvent 逐 factor 重建 engine | **fixed（agent）** | `execute_incremental_updates_from_event` 每 factor 用 catalog full spec 重建独立 FactorEngine + 验证 identity，不再复用调用方 engine |
| #52 FactorDefinition 完整 spec | **fixed（agent）** | `factor_full_definition` 表（expression/surface/dialect/dialect_version/market/universe/frequency/source config/decision policy） |
| #53 DataEvent 扩展 | **fixed（agent）** | field_id/affected_start/affected_end/snapshot_before/snapshot_after/revision_kind/deleted_keys；原构造器兼容 |
| #54 lineage universe-mask 失败 production 拒 | **fixed（agent）** | `build_materialize_lineage(production=)`：mask 失败 production raise，research 保留 drop_reason。tests/runtime/test_r10_data_event_scheduler.py（6 tests） |

### 十、SourceRef

| 项 | 状态 | 修复 |
|---|---|---|
| #55 transform spec dtype/min/max/choices | **fixed（agent）** | `SourceTransformParamSpec(dtype,min,max,choices).validate()`；`financial_lag.quarters`/`minute_bar.period,index` 等全 typed |
| #56 unknown transform production 拒 | **fixed（agent）** | `SourceRefSpec.production` + `_assert_declared_transform`；production 未知 transform raise，research/compat 允许。tests/runtime/test_r10_composite_sourceref.py（13 tests） |

---

## B. 验证

- 本会话新增 R10 测试文件：`test_r10_build_context`(7)、`test_r10_pit_tristate`(5)、`test_r10_execution_contract`(6)、`test_r10_config_strict`(9)、`test_r10_panel_multiindex`(5)、`test_r10_pit_closure`(8) + 并发/agent 新增：`test_r10_registry_identity`、`test_r10_cost_model`(24 含既有)、`test_r10_audits_closure`(16)、`test_r10_paramspec`(22)、`test_r10_dedup_regimes`(13)、`test_r10_factor_identity`(29)、`test_r10_data_event_scheduler`(6)、`test_r10_composite_sourceref`(13)。均无 xfail/skip。
- 既有回归：config profiles(13)、PIT policy(17)、semantic roundtrip(22)、panel identity(33)、materializer(58)、composite/source-ref、phase_g/d/22/23(16)、r7/r9 audits/canonicalizer/execution-contract 等通过。
- `load_all()` 通过（1362→ 随并发注册新增，仍在增长）。

## C. 诚实声明的既有失败（非本批引入）

1. **Operator tier 状态**：当前 catalog 仅 31 个 canonical 为 `production`，核心算子 `ts_mean`/`rank` 为 `experimental`（`build_operator_spec(...).allow_in_production=False`）。这使 production 模式对绝大多数因子 DSL 校验失败（`assert_production_factors`/`assert_production_plan_ops`）。与既有 `test_lqtp_production_runtime.py` 的 2 个失败同源。属证据/tier 再生成的延后项，非本批引入。
2. `tests/runtime/test_phase24_platform.py::test_data_access_source_read_auto_flag` — 并发会话 dataaccess/* 改动引入（read_result 计数/semantic-catalog 长度），与本批文件无关。
3. `tests/operators/test_stateful_incremental_store.py` 2 个失败 — 并发会话 `runtime/stateful_incremental.py` WIP（source_snapshot_scope 加入 checkpoint identity），非本批文件。
4. 并发会话间歇性提交会扫走工作树改动并短暂破坏 registry import（agent 均已验证改动在盘且测试通过）。

## D. 延后项

- **#49 generation-atomic materialization**（staging → all partitions → atomic CURRENT pointer）：需 catalog/staging 层重构，与 per-factor engine 重建协同后做。
- **Operator tier / evidence 重生成**（全量 production tier + backend evidence 重挂）——Review 明确要求最后阶段做（先稳定代码与 contract，再依次重生成 Primitive/Operator/Recipe/Catalog/Backend/Factor/Cold-start/Production fastpath evidence）。
- **cost/audit 接入真实 AlphaProbe 生成循环**：树中仍无 AlphaProbe 类，`search_budget_gate`/`default_mining_allowed`/`operator_contract_searchable` 已备好 API。
- **DataAccess universe-filter 接线**（execution-scope 选项 a）：当前选项 b fail-closed 生效。
- **全量 E2E 搜索验收**（随机表达式 → Parse/Type/Feasible/Execution/FiniteCoverage/Duplicate/NovelAlpha/Runtime 统计 + 时间切分 OOS/purge/embargo）——依赖 evidence 重生成后执行。
