# R11 / Long-tail FactorEngine Closure Report

**最新 main SHA：`e2a0a28`**（R11 我方收口 commit；树仍在并发 session 持续提交之下，基准 `ecffd565...` 之上已叠加多轮 settle pass）。

## 一、总体

本轮为 AI 长尾审计（~185 项，40 节）的全量整改。原则按审计 §39 执行：修数学定义/PIT/状态性/history/typed contract/事件观测/session clock/missing semantics/unit/参数有效性/统计有效样本；不加新算子（除审计明确要求拆分的语义区分算子）；不做 xfail/skip、不 broad except、不 research 逃脱、不重复注册 canonical、不 EPS 掩盖——数学未定义一律 `NaN + 原因` fail-closed。

**10 个并行 agent（WS-B..K）+ 我直改 WS-A/WS-L**。最终 **load_all 1375 canonicals / 0 unclassified**，**R11 新增测试 188 项全绿**。

## 二、文件清单

**共享层（我直改，WS-A）**：`runtime/execution_contract.py`、`cleaned_operators/base.py`、`cleaned_operators/contract_hardening.py`、`cleaned_operators/production_policy_extensions_v2.py`、`api/dsl_parser.py`。
**WS-L（我直改）**：`storage/sources/composite_source.py`、`storage/factory.py`、`storage/sources/data_access_source.py`、`scripts/audit_r11_longtail.py`。
**WS-B..K（agent）**：structural_levels、extrema_divergence、`common/_pivot_ledger.py`(新)、stateful/survival、stateful/sequential、state_event、alpha_language_state、update_clock、report_timing、conditional_dependence、dependence_ext、advanced_information、nonlinear_dependence、spectral、spectral_ext、complexity_ext、multifractal、advanced_intraday、intraday_session、session_recovery、intraday_activity_duration、feature_geometry、directional_change、tail_systemic、extreme_tail、marked_event、fundamental/quality_v2、fiscal_strict、fundamental/transforms_v2、accruals_scores、gather_ext、group_ext、weighted_moment_ext、conditional_ext、state_episode_excursion、downside_risk、return_decomp、hankel、vector_path、memory_ext、relation/ops、relation/distribution、shareholder/churn_network、cs_state_ops、data_cleaning、group_neutralization、stateful/rotation。
**集成修复（我）**：`operator_surface.py`、`nonlinear_dependence.py`（WS-E rename 破坏既有分类）、`group_ext.py`（cs_huber/cs_lad 表面分类）、`robust_scale.py`（unit）。

## 三、每项 commit

| commit | 内容 |
|---|---|
| `432f0c7` | WS-A 共享契约层：declare_stateful、history_formula/semantics、DSL binder（#16/17/18）、cross_event stateless（#11） |
| `663cd8a` | §35 SnapshotOnlySourcePolicy + current_only；§37 auditors；unit 修复；surface/nonlinear_dependence 集成 |
| `e2a0a28` | group_ext cs_huber/cs_lad EXTENDED 表面分类 |
| 并发 settle（931a54a 等） | 扫入 10 个 agent 的算子整改（file-disjoint，均已验） |

## 四、核心整改

### P0 历史重写
- **#1 structural_levels / #2 extrema_divergence**：共享 `StreamingConfirmedPivotLedger`（`common/_pivot_ledger.py`）。pivot 为 append-only 事件；未来更极端同侧候选产生 `PivotSuperseded(old, new, effective_at=t)`，只影响 `>=effective_at` 的行；`active_at(t)` 查询按行使用当时可见 pivot。全 ±confirmation 窗口才确认；窗口内 NaN 阻止确认；序列端不非对称确认。**prefix-invariance 性质测试**（`test_r11_pivot_ledger.py`，13 项）随机 T 验证 `factor(full)[:T]==factor(full[:T])`。

### ExecutionContract / statefulness（#3-12）
- `declare_stateful(canonical, state_model, chunking, checkpoint_schema, minimum_history, history_kind, history_count)` 成为算子自声明机制；`execution_contract()` 解析顺序 = checkpoint registry → 声明 → 遗留名单。`production_policy_extensions_v2` 与 `contract_hardening` 消费声明集。**#12 废除手工名单**（遗留 `_STATEFUL_CANONICALS` 仅作未迁移回退）。
- **#3 survival trio**→episode/required_full_history；**#4 CUSUM**→recursive；**#9 event_decay_asof**→recursive；**#10 ts_time_since_change**→recursive + `initial_semantics`（since_transition vs state_age）；**#11 cross_event 移除 stateful 名单** + `HistoryTransform(lag=1)`；**#5/6 hysteresis 家族**共享 `HysteresisStateKernel` + `HysteresisMissingPolicy`（BREAK/CARRY_STATE_AND_CLOCK/CARRY_STATE_FREEZE_CLOCK/UNKNOWN）；**#7/8 update_clock/report_timing**→`history_kind=event_count/report_count`（bar-warmup 不得接管）；**#44** NaN update_event 透传 censor；**#45** acceleration baseline 排除当前 delta；**#46-49** filing/revision 事件 PIT（Forward-filled panel 不制造 filing；revision 只在 revision_id 变化时发生；window 按 report 事件数 4/8/12/20；RevisionPair 同 period）。

### HistoryRequirement（#13-15）
- `ParamSpec.history_semantics`（既有）+ 新 `ParamSpec.history_formula`（如 `"outer_window + inner_window"`、`"2 * window"`）机器可读复合；`_declared_history_extension` 在名字猜测前读声明；event-clock semantics → `_UNKNOWN`（保守 full-history）。`HistoryRequirement.kind` 扩展 `event_count/report_count/session_count` + `count` 字段，`is_full_history` 覆盖 event-clock。**#15** 由 §37 audit C（default-parameter history）落地。

### DSL Parser（#16-18）
- **#16** parser 不再做 numeric-string 盲转（`"000001"` 恒为 str）；受控转换移到 binder `_coerce_declared_numeric_string`（仅声明 dtype=int/float 时）+ DSL call-site `_coerce_call_literals`（按算子 ParamSpec）。**#17** 非有限字面量（1e309→inf）拒绝。**#18** `ComplexityBudget`（max_ast_nodes/depth/call_arity/literal_magnitude/variadic_inputs）。

### 统计与 spectral（#19-27、#80-91）
- Miller–Madow 可选项且与手算一致；TE effective-state-space sample gate（`N>=k·cells`）；CTE ties 确定性破结 + collapsed bin NaN；TE 因果无前视；MODWT window/level 可行性 + 死组合拒绝；spectral window>=16；单 bin Q 用 one-bin 分辨率；spectral entropy dimensionless、dominant period bars；spectral_ext polars 元数据一致；LZ 明确为 retrospective window complexity + `effective_n`/`min_contiguous_fraction`；forbidden-ordinal N 只算 no-tie；multifractal q∈{0.5,1,2,3,4} + >=4 点 quadratic + common cohort。

### Intraday / session（#56-79、#183-186）
- k-minute RV/PV 聚合（非盲抽样）；offset 共同 interval；signature block 右对齐 EOD；quantile profile min samples；PCA numerical rank/eigengap/`DEGENERATE_PCA_SUBSPACE`→NaN；ExchangeSessionID 分组（非 normalize）；equal-count vs fixed-clock slot 区分；phase shift min-corr gate；max_shift 不 silent cap。**session completion P0（#68）**：仅 official close + coverage 达标才输出 full-session factor；SessionID strict typed；session_id 缺失 censor；`history_kind=session_count`；min_history_sessions；session PCA rank gate；canonical normalized grid。session_recovery：EventBool、residual_fraction∈(0,1]、missing minute 区间化、refractory。activity_duration：official SessionGrid + partial session rejected + dimensionless。

### Directional / tail / geometry（#29-43）
- **#29** robust correlation → 真 biweight midcorrelation（与 Pearson 不同）；**#30** beta break → 真 `Cov(y,x)/Var(x)`。**#31-34** directional change 流式递归 + threshold 固定入 checkpoint + per-leg threshold + scale missing BREAK。**#35-41** tail lead common cohort + `min_conditioning_events` + `effective_event_n`；relation diffusion complete graph → 闭式 `group_mean + c·dev`；Hill upper/lower POT 对称 + 价格 level 门（unit contract）；quantile beta `unit(y)/unit(x)`；extreme-run gap censor。**#42-43** marked_event NaN censor + `MarkMissingPolicy`。

### Fundamental / fiscal（#50-55、#180-182）
- Piotroski 缺失项不记 0（完整 9 项才 F-score + partial score/observed count）；`FinancialFlowSemantics`（SinglePeriodFlow/CumulativeYTDFlow/TTMFlow/AnnualFlow/Stock）+ YTD 先 QuarterFromCumulative；flow-grain 不匹配拒绝；strength score 先 rank-normalize 再合并（×1000 不变）；Altman/Zmijewski `applicable_universe:non_financial`。`FiscalPeriodKey(fiscal_year, fiscal_slot)` 支持 quarterly/semiannual/annual/53-week/non-calendar；report_yoy_lag 匹配上年同 slot。

### Group / cross-section / weighted / conditional（#131-144）
- group_topk_mean cutoff tie fractional validity；JS under/overflow bins + reference sample floor；hierarchical 用 `(group,subgroup)` 复合键；group_ex_self 单位 same_as:x；`cs_robust_resid` → `cs_trimmed_ols_resid`（真 rename + alias）；新 M-estimator `cs_huber_resid`/`cs_lad_resid`；min_breadth/DOF margin；weighted_moment exact align；ridge 1e-3 versioned 非 searchable；conditional_ext 单位 + fit_min_periods + ConditionBool。

### Relation / shareholder / rotation（#118-130、#153-157、#175-179）
- relation 单位（weighted mean same_as:value；weighted sum unit·unit）；index_entry_exit_event signed；missing_policy=false 禁 production membership；complete-graph PageRank 闭式；`_stack_panels` exact；rank mobility slot vs entity（新 `relation_rank_entity_mobility`）；concentration acceleration `history_formula="2*window"`。churn `pd.isna` gate；HolderID 结构化；top-K absence → disclosure exit 命名；pledge/freeze domain。rank churn 分解（intersection/composition/combined）；tail retention cohort + q<=0.5 + fractional tie mass；GroupState/GlobalState typing；cs_impute min_finite 生效；coverage physical vs universe 拆分（新算子）；ffill forward_fill_allowed。

### §35 US valuation（plan A）
`current_only` join 进入 `_ALLOWED_JOIN_METHODS`；`snapshot_only` source 级 SnapshotOnlySourcePolicy（composite 构造校验 + production historical mining fail-closed + X0 sparse `allow_sparse` 读取）。**默认 helper 现可构造**（`default_us_pv_valuation_data_source_config` → `build_data_source` 3 项测试）。

### §37 共享 auditors（A-I 中 A/B/C/D 落地）
`scripts/audit_r11_longtail.py`：A prefix-invariance、B stateful-contract discovery（chunk-boundary，stateless-drift hard fail）、C default-parameter history、D unit-algebra（发现 4 个真 `same_as:target` bug 已修；coverage/covariance 误报已用 word-boundary token 修正）。E missing-metamorphic / F dead-param / G event-observation / H session-completion / I statistical-DOF 由各 WS 的 targeted tests 覆盖。

## 五、测试

- **R11 新增 188 项全绿**：shared_contract 17、pivot_ledger 13、state_event_stateful 16、hysteresis_clock 15、te_spectral_gates 27、intraday_session 19、directional_tail_geometry 15、fundamental_flow 9、group_cs_ops 15、downside_ssa 17、relation_shareholder 18、shared_auditors 4、us_valuation_snapshot 3。
- **prefix invariance**：pivot_ledger 13 项（random T）+ auditors A。
- **full/chunk/checkpoint parity**：state_event_stateful（survival/CUSUM/event_decay full==replay）+ hysteresis_clock（family parity）+ auditors B。
- **intraday session completion**：intraday_session 19 项（truncated 不输出 full-session）。
- **fundamental PIT**：fundamental_flow 9 项（filing/revision 不制造、YTD、flow-grain、Altman applicability、FiscalPeriodKey）。
- **统计 golden**：te_spectral_gates 27 项（Miller–Madow 手算一致、TE sample gate、Q 不爆炸、LZ effective_n、multifractal q grid）。
- **load_all**：1375 canonicals / 0 unclassified。

## 六、生产认证与遗留 blocker

1. **evidence 链仍 stale**（`test_backend_coverage` 全部 DAILY canonical 缺纯 polars 认证）：审计 §40 明确要求「完成这一批后再重新生成 evidence」。树仍在并发 session 持续提交之下（当前 HEAD 后仍有未提交修改），**未满足重生成前置**。重生成序列（树稳定后一次跑完）：`sync_primitive_evidence` → `certify_primitive_evidence` → `certify_factor_operator_evidence`（内部先跑 audit_all_factor_production --runtime-only）→ `certify_recipe_evidence` → `export_operator_manifest` → `export_mining_data_source_presets` → `export_dsl_allowlist` → `sync_factor_engine_llm_prompt_txt`。
2. **并发 owner 未收敛项**（非本批，需并发 session 收尾）：`test_typed_ir_v2::test_ir_propagates_field_semantics`（analyzer pit_safe 三态 WIP）、`test_layer_governance::test_registry_rejects_implicit_duplicate_after_bootstrap`（其 registry `_DECLARED_OVERRIDE_SOURCES` 新规则 vs 既有测试）、`ts_joint_energy_shift` polars parity、`test_report_change_breadth_all_rising`（alpha_language_events 既有行为）。
3. **#136/#139 改名**：cs_robust_resid→cs_trimmed_ols_resid 已真改名（WS-I）+ cs_huber/cs_lad 新真稳健算子；cs_multi_robust_resid→cs_multi_ridge_resid 保留原名、数学/metadata 已诚实（改名被 shared surface/signature 文件 pin 住）。
4. **Hankel/SSA 分段 checkpoint**：honest required_full_history（无 segmented restore 实现），未假称 checkpoint。
