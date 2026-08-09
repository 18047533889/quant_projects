# FactorEngine — 算子语义审计整改报告（2026-08-09）

审计基线：工作树 `main@ecffd56+`（执行期间并发会话持续提交）。本轮针对**算子实现本身**
的数学/语义 bug（外部 AI 深度 review：P0-1…P0-12 / P1-1…P1-8 / P1-18、P1-19 / P2-1、
P2-2 / downside 类型 / 15 条机器审计规则）。11 个并行 agent 严格文件不相交分包 + 本会话
直改（P0-10 官方场内网格、机器审计框架、pandas twin 补洞）。

本轮原则：已修问题不回滚；数值语义错误直接改数学；cohort/state 语义一律
native-date 定义 → 再 cohort 限制；production 侧 fail-closed；全部永久 regression tests；
诚实报告既有失败与延后项。

---

## 一、P0 修复（数学/语义错误，全部永久测试）

| 项 | 状态 | 修复 |
|---|---|---|
| **P0-1 `ts_hill_tail_index` 双重取尾** | **fixed** | 原来 POT 取一次尾、再对 exceedance 乘一次 `tail_fraction`（120/0.2/10 默认 → k=4<10 → 长期 NaN）。改为**单次取尾**：阈值 `u=Q(1-frac)`（上）/`u=Q(frac)`（下），全部 exceedance 作为 Hill 样本，`ξ=(1/k)Σlog(mag_i/u)`（u>0 门控）。Pareto 合成 golden（shape∈{0.3,0.5} 恢复在 0.25 内）、默认参数 finite-rate、上下尾 mirror 测试。测试 `tests/operators/test_r11_extreme_tail.py` |
| **P0-2 `ts_mean_excess_slope` 下尾符号** | **fixed** | 下尾原来阈值 `u` 后仍对原始 `u` 回归 → 方向与上尾相反，文档「正=厚尾」反号。改为下尾 `y=-x` 完全复用上尾 mean-excess，满足 `ME_lower(x)=ME_upper(-x)`，slope 解释一致。mirror 测试。同上 |
| **P0-3 `fin_fundamental_strength_score` cash_yield 无量纲错误** | **fixed** | `OCF/|ROA|`（金额/比例=金额，混入 size）→ **CashROA=`OCF/AverageAssets`**；新增 `avg_assets` 入参（`inventory_turnover` 与 `revenue_growth` 之间）。semantic change 已注释。测试 `tests/operators/test_r11_fundamental_scores.py` |
| **P0-4 strength 缺分量比较偏差** | **fixed** | 原来对 8 分量直接求和（3/8 与 8/8 直接可比）。改为**观测分量均值** + `MIN_FUNDAMENTAL_COMPONENTS=6`（<6 → NaN）；新增 `fin_fundamental_strength_coverage`（0..8）。同上 |
| **P0-5 `piotroski_f_score` 完整性** | **fixed** | 缺失分量不计 →「4/4 全过=4」「9/9 过 4=4」不可区分。拆三算子：`piotroski_f_score`（**必须 9 项全观测**否则 NaN）、`piotroski_partial_score`（passed/observed）、`piotroski_observed_count`（0..9）。同上 |
| **P0-6 Altman/Zmijewski 适用域 fail-closed** | **fixed** | 声明 `applicable_universe=non_financial` 但 `mask=None` 仍出值。改为 **`mask=None` → ValueError**（硬语义，不能靠调用方记得传）；有 mask 保留 NaN-outside-universe。同上 |
| **P0-7 `cs_rank_composition_churn` 组迁移漏算** | **fixed** | 分组分支先取 `(g_t==lab)&(g_{t-L}==lab)` 交集 → 真正 A→B 迁移被消失。改为**native-date 成员集**：`A_cur`/`A_prev` 各日期全池集合，churn=`|Δ|/|∪|`（含仅昨日成员）。测试 `tests/operators/test_r11_rotation_cohort.py` |
| **P0-8 `cs_tail_retention` 顺序反** | **fixed** | 分组分支先交集再定义 tail → cohort 变动反写过去 tail 成员。改为 **native-date tail 定义 → cohort 限制 → overlap**：tail 权在各日期完整 group cohort 上算，再按 intersection/historical/current cohort 求留存。同上 |
| **P0-9 `hierarchical_group_neutralize` 未知组→0** | **fixed** | 未知 group/subgroup 单独成键 → mean=自己 → 残差 0。改为未知标签**保持 NaN**（永不入键）；polars 与 pandas twin（group_ext）都修。测试 `tests/operators/test_r11_group_honesty.py`（+ pandas twin 补测） |
| **P0-10 intraday 官方网格自证** | **fixed（本会话）** | `_session_expected_grid` 用观测数据众数定义官方宽度/收盘 →「整天缺最后一分钟」自证 completed。彻底删除 modal 推断：算子新增 `calendar`（`runtime.session_calendar.SessionCalendar`）参数，官方宽度/收盘**只来自日历**；无日历 → 全部 NaN + 一次 RuntimeWarning（fail-closed，绝不 infer）。`intraday_session_shape_novelty`/`intraday_profile_pca_residual`/`intraday_activity_duration_curvature` 三算子同改。测试 `tests/operators/test_r11_official_grid.py`（6 tests，含「239 栏天天缺尾 → 0 输出」核心反自证）；并发 test_r11_intraday_session.py 19 tests 已改传日历并全绿 |
| **P0-11 旧 conditional 算子 ConditionBool** | **fixed** | `ts_count_if/sum_if/mean_if/std_if/last_if/days_since/true_streak` 原来 `ne(0)/bool(value)` 隐式把任意 NumericSeries 当 ConditionBool。统一镜像 `conditional_ext._assert_condition_bool`（值∈{0,1,NaN}，其余 raise）。测试 `tests/operators/test_r11_conditional_semantics.py`（12 tests） |
| **P0-12 `ts_days_since`/`ts_true_streak` NaN 状态** | **fixed** | 未知事件状态被当已知（NaN 时仍输出距离 / streak=0）。改为 **NaN@t → 输出 NaN + 状态确定性丢失**（last_true/streak 重置，直到显式 true 重建）；false=0 确认。同上 |

## 二、P1 修复（统计/数值定义）

| 项 | 状态 | 修复 |
|---|---|---|
| **P1-1 `feature_geometry` 相关矩阵不 PSD** | **fixed** | 两两 bicor 拼矩阵不保证 PSD，原来 `max(eig,0)` 截断改变 trace/eigen 质量，且 dominant direction 用原始矩阵。新增 **Higham 最近-PSD 相关投影** `_nearest_psd_correlation`（交替投影→单位对角），`_canonical_corr` 单一投影矩阵喂给**所有** eigen 消费者（eigenvalues/mode_share/effective_rank/dominant_direction/subspace_rotation 同源）。测试 `tests/operators/test_r11_feature_geometry_psd.py`（6 tests，含 0.9/0.9/-0.9 非 PSD 对抗） |
| **P1-2 `ts_feature_effective_rank` unit** | **fixed** | `unit="rank"`（误导为序数 rank）→ `unit="dimensionless"` + `semantic_kind:effective_dimension` tag。同上 |
| **P1-3 tail dependence 命名** | **fixed** | 固定 q 条件概率 ≠ 渐近 tail-dependence 系数。新 canonical `ts_upper_tail_coexceedance_probability`/`ts_lower_tail_coexceedance_probability`，旧名 deprecated alias。测试 `tests/operators/test_r11_tail_dependence_ties.py` |
| **P1-4 tail dependence 上下 tie 不对称** | **fixed** | 上尾 `>` 下尾 `<=`，tie 堆（0 收益/涨跌停）上下尾有效质量不同。新增**分数边界 tie**：`_fractional_tail_membership` 让有效尾质量两侧都等于 q·N；上/下概率对称。同上 |
| **P1-5 `ts_extremal_index` cluster 语义** | **fixed** | 原 runs estimator run-length=1 + NaN `continue` 不重置 → 「Extreme,NaN×3,Extreme」算同一簇。改为 **NaN 断簇** + 新增 `run_length` 参数（exceedance 间隔 < run_length 才续簇）。测试 `test_r11_extreme_tail.py` |
| **P1-6 `_reject_price_level` 数值启发降级** | **fixed** | 全正值+sd(level)/sd(diff)>5 → raise 是数值启发当语义权威。改为 **DQ warning（一次/进程）**，语义权威在 typed FieldSpec（`input_units=signed_return_...` 已声明）；纯价格仍可算但告警。同上 |
| **P1-7 `relation_topk_concentration` rank-slot** | **fixed** | 原来按值排序取 top-k（s1 缺失 → s6 补位）。改为 **rank-slot 语义**：前 k 个 slot 任一缺失 → 该 cell NaN（fail-closed，未知 top1 不可由 top6 顶替）；分母=全部提供 slot 之和。测试 `tests/operators/test_r11_relation_rank_slots.py` |
| **P1-8 relation unit contract 无效引用** | **fixed** | `same_as:value`/`valuemount` 无效单位 → `same_as:share`（与 relation_share_mobility 一致）+ docstring 说明。同上 |
| **P1-18 TE peak 共享分箱** | **fixed** | 每个 lag 在 `_te_from_transitions` 内重新 quantile 分箱 → peak_lag 混入 discretization/有效样本变化。改为**窗口内一套参考 bins**，所有 lag 复用。测试 `tests/operators/test_r11_te_peak_binning.py` |
| **P1-19 TE peak max-selection 偏差** | **fixed** | 独立源 `max_L TE_L` 天然正偏。新增 **`ts_transfer_entropy_peak_excess`** = `max_TE_real − E[max_TE_surrogate]`（n_surrogates=20，seed 固定可复现）；独立源 excess≈0，x→y 依赖 excess>0。同上 |

## 三、P2 修复（canonical 诚实性 + 类型合同）

| 项 | 状态 | 修复 |
|---|---|---|
| **P2-1 `cs_multi_robust_resid` → `cs_multi_ridge_resid`** | **fixed** | 真实实现是 standardized ridge（固定 ridge=1e-3）。新 canonical `cs_multi_ridge_resid`（诚实描述，ridge=版本化实现常量非可搜参数），旧名 deprecated alias（`OperatorRegistry.register_alias`）。测试 `tests/operators/test_r11_multi_ridge_rename.py`（6 tests）；test_r11_group_cs_ops 断言更新 |
| **P2-2 `cs_robust_resid` → `cs_trimmed_ols_resid`** | **fixed** | 真实实现是 trimmed-OLS。新 canonical `cs_trimmed_ols_resid`（polars + pandas twin），旧名 alias；`classify_canonical("cs_robust_resid")=="daily"` 保持。`cs_huber_resid`/`cs_lad_resid` 新算子延后（表面分区风险）。测试 `tests/operators/test_r11_group_honesty.py`（7 tests）；test_polars_group/test_r11_group_cs_ops 62 passed |
| **downside/upside 类型合同** | **fixed** | `input_units={"x":"price"}` + 默认 target=0 → 正价格死因子。改 `x: return_or_same_unit_numeric`、`target: same_unit_as:x`、`output same_as:x`；描述说明 MAR/目标收益语义，正价格 target=0 是 contract violation（typed 层拒绝），不再是静默 ~0。测试 `tests/operators/test_r11_downside_contract.py`（6 tests）；test_r11_downside_ssa 断言更新 |

## 四、机器算子语义审计框架（§8，15 条规则）

新增 `cleaned_operators/semantic_audit.py` + `tests/operators/test_r11_semantic_audit.py`：

1. `default_output_finite` — 默认参数 finite 比例 <5% → error（立刻能抓 `ts_hill_tail_index` 类死因子）。
2. `parameter_injectivity` — int 参数 5.1 静默截成 5 → error（走中央 `_normalise_integer`）。
3. `relational_constraints` — `min_periods>window`/`lag>=window` 静默接受 → error。
4. `mirror_symmetry` — 上/下尾 `lower(x)≈upper(-x)`。
5. `column_permutation` — CS 算子打乱股票列 undo 后必须一致（抓 tie/列序 bug）。
6. `missing_state` — 事件/条件 NaN（未知）不能被当「确认」。
7. `semantic_type_gate` — Price/Return 不可互换；拒绝 price 的合同不得静默收 price（raise 或 DQ warning）。
8. `finite_range` — 声明 probability/correlation 的必须在界内。
9. `scale_shift_invariance` — dimensionless 算子对合法缩放不变。
10. `group_migration` — A→B 迁移必须真实反映到 churn/retention。
11. `native_cohort` — retention/overlap 先 native-date 定义再 cohort 限制。
12. `psd_geometry` — spectral 算子 eigen 指标必须同源同一 PSD 矩阵（由 test_r11_feature_geometry_psd 钉住）。
13. `golden_reference` — Hill 在 Pareto DGP 恢复已知 shape（|Δ|<0.25）。
14. `canonical_honesty` — robust/hill/allan/granger/tail_dependence/extremal/entropy 名必须绑定义/公式。
15. `default_searchability` — 默认参数不得 all-NaN/常数/恒零。

框架：`AuditFinding/AuditReport`，每条规则 `check(op, ctx)`，`run_audit(sample_names, rule_names)`；每规则报告 ran 计数与**诚实 skip 原因**（不静默截断覆盖率）。`test_r11_semantic_audit.py` **10 passed**：7 框架/负检测 + 3 正收益（Pareto golden 用 `U**(-ξ)`——`(1-U)**(-1/ξ)` 是 tail index=1 的陷阱）。

## 五、验证

- 新增测试文件：`test_r11_extreme_tail`（A）、`test_r11_fundamental_scores`（B，8）、
  `test_r11_rotation_cohort`（C）、`test_r11_group_honesty`（D，7+pandas twin）、
  `test_r11_conditional_semantics`（E，12）、`test_r11_feature_geometry_psd`（F，6）、
  `test_r11_tail_dependence_ties`（G）、`test_r11_relation_rank_slots`（H，5）、
  `test_r11_te_peak_binning`（I）、`test_r11_downside_contract`（J，6）、
  `test_r11_multi_ridge_rename`（K，6）、`test_r11_official_grid`（本会话，6）、
  `test_r11_semantic_audit`（本会话）。均无 xfail/skip。
- 既有回归：`test_r11_intraday_session`（19，改传日历）、`test_r11_downside_ssa`+`test_deepening`（21）、
  `test_r11_relation_shareholder`（18）、`test_polars_group`+`test_r11_group_cs_ops`+`test_new_atomic_ops`（62）、
  `test_financial_next_stage`/`test_r11_fundamental_flow`/`test_round3_audit_fixes`/`test_field_catalog_alignment` 断言已更新。
- `load_all()` 通过。

## 六、诚实声明的既有失败 / 延后

1. **并发会话 R4-100 `ts_extremal_index/polars` arity 瞬时不一致**：执行期间多次出现（Agent A 加 `run_length` 到 pandas 侧、polars 侧滞后），随并发会话同步修复后消失；不是本批引入，已在 memory 登记。
2. `test_round3_audit_fixes_2026_08.py::test_state_density_has_1_over_h_normalization`（state_geometry KDE，非本批文件）。
3. **延后**：`cs_huber_resid`/`cs_lad_resid` 新算子；BENFORD/relation `report_cross_item_benford_divergence` 事件时钟重设计（review 文本截断，细节未齐）；`_day_curvature` 近常数带噪数据返回 NaN 的既有行为（非本 review 项）；`test_daily_panel_ops.py` 整文件 skip（registry 解析到 overhaul/daily.py 旧实现，不归 daily_panel）。
