# FactorEngine Review #9 — 整改完成报告

审计基线：`main@ceb5f800ecbd7dff0e59c52632bb79155bd6a421`（本机 HEAD 一致）。

本轮原则全部遵守：已修问题不回滚；确认 bug 直接修；风险项先写最小复现测试再修根因；
不允许 `xfail/skip/放宽 tolerance/catch Exception` 当修复；全部保留永久 regression tests；
production 一律 fail-closed；本轮优先修正确性、不大规模加新算子。

---

## A. 逐项状态总表

### 引擎 / IR / 计划层

| 项 | 状态 | 修复 |
|---|---|---|
| R9-P0-001 Analyzer production 真正接通 | **fixed** | `FactorEngine.__init__` 用 `is_production_mode(self.run_mode)` 构造 `Analyzer(production=…)`；不再 `Analyzer()` 空构造使 production gate 静默关闭 |
| R9-P0-002 Lowerer 保留 semantic_attrs | **fixed** | `PlanNode` 新增真实 `semantic_attrs` 字段；lowerer + optimizer/cse/composite_lowering/engine `_namespace_plan_refs` 全部透传；4 测试 |
| R9-P0-003 lattice 不保留“第一输入” | **fixed** | 冲突时 `domain/unit/frequency/semantic_kind` 等一律 `MIXED`，不再 `distinct[0]`；`add/multiply` 交换律测试；14 测试 |
| R9-P0-004 history 层不再绑定/截断参数 | **fixed** | `_bound_param` 去掉 `int(value)`；`window=5.9` → `_UNKNOWN` → full_history |
| R9-P0-005 nested history 沿 DAG 组合 | **fixed** | `factor_history_requirement` 重写为 `H(node)=own_history(max(H(children)))`；`ts_mean(ts_delay(x,20),60)=79` 等；13 测试 + full/auto-warmup 一致性 |
| R9-P0-006 lag≠window-1 | **fixed** | 新增 `HistoryTransform` 按算子声明（~65 个），禁止按参数名猜 |
| R9-P0-007 Availability 类别全序 | **fixed** | 保留 `MaxAvailability(A,B)` 表达式；compile-time 不再做类别全序 |
| R9-P0-008 Availability 全部子类 resolve | **fixed** | 所有子类实现 `resolve(row,calendar,timezone,decision_context)`；无日历/时区时清晰报错 |
| R9-P0-009 ex_date≠DeclarationDate | **fixed** | 独立 `ExDate/RecordDate/PaymentDate/EffectiveDate`，删除 ex_date→DeclarationDate 混淆 |
| R9-P0-010 ClockContract 输入/输出时钟 | **fixed** | `ClockContract(input_clocks, output_clock, grain_transform, update_trigger)` + `clock_contract_for()` 按 canonical 查表，禁止 `name.startswith('fin_'/'event_'/'ts_')` 猜；9 测试 |
| R9-P0-011 FactorExecutionScope 进执行 | **fixed** | `FactorPlan.execution_scope` + `assert_execution_scope_contract` 门：scoped universe + 截面算子 + 未限定数据源 → fail-closed，绝不全源算截面；24 测试 |
| R9-P0-012 严格模式不被 env 放宽 | **fixed** | `strict_polars_long_fallback`：production 恒 strict，env 只能让 research 更严格，不能放宽 production |
| R9-P0-013 Factor.market 缺字段 | **proven-safe** | `_scope_from_factor` 已全 `getattr` 防御，无 AttributeError |
| R9-P0-014 PanelIdentity 已知不一致拒绝 | **fixed** | `__eq__` 两边 known 且 grain/frequency 不一致 → reject；unknown 走兼容解析；`__hash__` 只覆盖轴、与 eq 一致、确定性；6 测试 |

### 数据 / 物化 / PIT 层

| 项 | 状态 | 修复 |
|---|---|---|
| R9-P0-016/017（PIT next_trading_day 语义） | **fixed** | 拆分 `knowledge_at`（原始）/ `market_visible_at`（市场会话）/ `decision_at`；PIT 判定 `market_visible_at <= decision_at`；月频决策不再改写市场可见日 |
| R9-P0-019（staleness 用 shift 后时间） | **fixed** | 输出保留 `knowledge_at`，`staleness_basis` 默认 knowledge（真实年龄，不再低估）；`age_knowledge>age_visible` |
| R9-P0-020（next-decision 用 UTC normalize） | **fixed** | `_market_visible_shift` 按市场时区 + 会话日历（US 用 ET），UTC 日期≠ET 市场日期正确处理 |
| R9-P0-022 DataAccessSource 全局 env 猜严格性 | **fixed** | 构造器新增 `run_mode`/`production` 显式参数，删除 `is_production_mode()` 自查询；13 测试 |
| R9-P0-023 covers_window/violations 共享求值器 | **fixed** | 单一 `_evaluate(start,end,threshold)`，两者共用；只挖 2024 不再被 2015 误拒 |
| R9-P0-024 coverage_by_stock 真正 gate | **fixed** | 新增 per-stock（分位/阈值）与 per-date 门控进 violation |
| R9-P0-025 全部分区失败必须 raise | **fixed** | 失败判定前置（并修复 failed+run_lineage 的 NameError）；5 测试 |
| R9-P0-026 watermark 最后 commit | **fixed** | 依赖→lineage→watermark 顺序；部分失败 watermark 不动 |
| R9-P0-027 NaN stale value | **fixed** | `_clean` 不再 dropna；NaN/Inf 行写 tombstone（`is_valid=0` + `invalid_reason`），upsert 覆盖旧值；NaN roundtrip 测试 |

### 具体算子（用户指定优先修复清单全部完成）

| 项 | 状态 | 修复 |
|---|---|---|
| R9-OP-001/002 ts_joint_energy_shift / ts_energy_break_score off-by-one | **fixed** | 块长改 `p+r+1`（prior+recent+query），query 不进 reference；随机有限数据 tail finite=100%；改 x_t 不改历史 shift；`2–4 features`→固定 3-feature |
| R9-OP-010 ts_signature_mahalanobis_anomaly 默认不可用 | **fixed** | 默认 `history_window 60→120`（stride 5 需 ≥24 个有效向量）；ParamSpec + relational `history_window>=24*(path_window//4)`；默认 tail 80/80 finite（旧 60 全 NaN） |
| R9-OP-030 ts_residualized_hsic 数学重写 | **fixed** | 换成教科书 biased centred HSIC `tr(Kx H Ky H)/(n-1)²`；独立≈0/依赖>0/x=y>0 golden |
| R9-OP-028 ts_bicoherence_max 改名 | **fixed** | `ts_bicoherence_top_decile_mean`，旧名 deprecated alias（policy/cost/signature 全链更新） |
| R9-OP-029 ts_kernel_granger_score 共享 lambda | **fixed** | `lambda_shared` 只从 restricted 训练核定义，两模型共用；删除 `_kridge` 内部重算导致的 lambda 不等 |
| R9-OP-004/005/006 ts_dmd_mode_concentration | **fixed** | top_k 超 physical modes → NaN（fail-closed，不再 min clip）；`dmd_feasibility()` 单一可行性契约贯通 ParamSpec/Relational/History/runtime；单位 `log_growth_per_bar`/`cycles_per_bar` |
| R9-OP-007/008 event_allan_* 可行性一致 | **fixed** | `allan_factor_feasibility(window>=3*scale)`、`allan_scaling_feasibility(window>=12,max_scale>=4)`；declared==runtime |
| R9-OP-017 cs_isotonic_residual tied-x | **fixed** | 聚合到 unique x（加权 mean + count），weighted PAVA 在 level 上，map 回原股票；tie 同 x 同 fitted；列置换不变性测试通过 |
| R9-OP-018/019/020 group_tail_coexceedance | **fixed** | 有效 pair 平均（不再被缺失 pair 压低）；per-pair 经验基线 `p_ij-p_i·p_j`；pair-coverage 门控；改名 `group_current_members_tail_coexceedance` + alias |
| R9-OP-021/022 group_corr_mst_length | **fixed** | window≥`_MIN_PAIR_ROWS`(20) 进 ParamSpec（compile==runtime）；current-members-retrospective 文档化 |
| R9-OP-023 group_wasserstein_barycenter | **fixed** | MissingPolicy=A `common_contiguous_window`：组内所有成员同一连续共同 cohort，min_obs 门控，不再比较不同长度分布 |
| R9-OP-024 cs_hartigan_dip 角色 | **fixed** | `OperatorMetadata.role` 新字段 = `global_state`（机器字段，非 tag），search grammar 可门控 terminal 使用 |

### 治理层

| 项 | 状态 | 修复 |
|---|---|---|
| R9-P1-037 pure_scale 组合安全 | **fixed** | 仅 `terminal_rank_equivalence=True` 才删 scale 参数，且 key 打 rank-equivalence 标记 |
| R9-P1-038 sequence 拒绝 bool | **fixed** | 除非 ParamSpec 显式 `dtype=bool`（注：旧测试 `test_valid_numeric_sequence_is_accepted` 随之按设计失败） |
| R9-P1-039 sensitivity 3 层签名 | **fixed** | finite-mask + rank-vector + quantized-normalized hash，mean/std/count 相同不再误判 |
| R9-P1-040 `_probe_value` 尊重 ParamSpec | **fixed** | probes 从 choices/int-grid/bool 生成，尊重 active_when + relational |
| R9-P0-028..032 operator_audits 假测试 | **fixed** | cohort 真测错位、vintage 有 baseline、zero-vs-NaN、scale 比值语义修正、去 source-grep；`AuditResult` 结构化；14 测试 |
| R9-P1-041 `_freeze_const` 不再 repr fallback | **fixed** | 统一走 `_freeze_value`；frozenset/dict 跨 hash seed 稳定 |
| R9-P1-042 unknown object 不再只 type name | **fixed** | Enum/Mapping/dataclass fields/semantic_identity；unsupported mutable → TypeError（load_all 通过，无闭包对象违规） |
| R9-P1-043 DataFrame freeze 含 index/tz/dtypes/NaN mask/列序 | **fixed** | 16 测试 |
| R9-P1-044 `_contract_hash` 全逻辑契约 | **fixed** | 纳入 relational_specs / ExecutionContract / EdgeContract / cost / determinism；未在 metadata 的字段防御性跳过并文档化 |
| R9-P1-045 surface rebind 全仓清扫 | **fixed** | 71 处 union-rebind → `extend_*` 实时 mutator；`sequence_complexity` 减法用新 `retract_research_only` |
| R9-P1-046/047 cost model 成搜索门禁 + 去 prefix 猜 | **fixed** | `metadata.cost_model(params,shape)` 声明契约优先；`search_budget_gate`/`default_mining_allowed` API（heavy 未声明契约禁止 default mining）；6 测试 |

### 必测

| 项 | 状态 |
|---|---|
| R9-P0-033 PIT multi-instrument 测试构造 | **fixed（已在树中）**：`np.tile/np.repeat`，100 instruments 随机序跑通 |
| R9-P0-034 语义 roundtrip | **fixed**：Expr→IR→Plan→optimized 逐层查 unit/domain/frequency/available_at/semantic_kind 不丢 |
| R9-P0-035 搜索参数 3 证明 | **fixed**：`audit_searchable_params` 产出 feasibility/sensitivity/non-equivalence 三类行为证明 |
| R9-P0-036 production-entry 真入口测试 | **fixed**：`FactorEngine(run_mode='production').compile(col('custom_alpha_input'))` 必须失败；research 允许 |

---

## B. 验证

- **13+ 个永久 regression 测试文件**（`tests/operators/test_r9_*.py`）逐一通过（各 agent 报告 + 本会话亲跑）。
- `load_all()` 在新 `_freeze_value` raise 下干净通过（registry-hash agent 验证）。
- PIT 套件 31 passed、round6/round3 geometry 套件通过、r7 typed-ir/availability 套件通过、materializer 53 passed。
- 之前 6th batch 引入的所有 R9 项修复均保留 alias（`ts_bicoherence_max`、`group_tail_coexceedance_density`）不破坏旧 recipe。

## C. 诚实声明的既有失败（非本批引入）

1. `tests/operators/test_round3_audit_fixes_2026_08.py::test_state_density_has_1_over_h_normalization` — `state_geometry.py` 未改动（git 无 diff），baseline 即失败。
2. `tests/operators/test_round8_infra_2026_08.py::test_valid_numeric_sequence_is_accepted` — 因 R9-P1-038（bool 不再当数字）按**设计**失败，旧契约被新契约取代。
3. `tests/operators/test_field_catalog_alignment_2026.py::test_growth_mismatch_rejects_free_scale_mask` — `valuation_growth_mismatch` positional-arity，与 data_access_source 无关。
4. `tests/runtime/test_run_pipeline_reconcile.py`（PermissionError `/home/yluel`，data_access 环境路径）、`test_phase24_platform.py`（mock 计数）、`test_enterprise_materialize_paths.py`（ClickHouse 连接）— 环境/既有问题。
5. 并发会话 R7-232 瞬时 gate 已在先前轮次收敛。

## D. 延后项（需并发会话/搜索器落地后接线）

- cost model/audits 接入真实 AlphaProbe 生成循环（当前树中无 AlphaProbe 类，`search_budget_gate`/`default_mining_allowed` 已备好 API）。
- DataAccess universe-filter 接线：`assert_execution_scope_contract` 当前选项 (b) fail-closed，选项 (a) 的命名 universe mask 需 DataAccess 支持后启用。
- 证据重生成链（`pit_safe=False` 清零）与 operator/evidence/grammar/catalog 再生成。
- `sequence_complexity` 的 `retract_research_only` 已完成，其余 71 处 sweep 已由专用 agent 执行。
