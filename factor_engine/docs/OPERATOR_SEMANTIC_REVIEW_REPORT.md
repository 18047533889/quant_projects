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

---

# 第二轮（round-2）：算子 correctness / 搜索空间 / 命名信任 全量整改（2026-08-09）

第二轮外部 review（算子本身优先，不再管仓库/提交/CI/物化外围）。范围 = 10 项算子 correctness P0
（§一）+ 技术指标假参数搜索空间（§二）+ 回归模型数学（§三）+ MI/TE（§四）+ 序数/复杂度（§五）
+ Markov/状态（§六）+ Directional Change（§七-十 可见部分）+ state_since/KNN unit（§十一-十二）
+ 多输入轴对齐（§十三 P0）+ piotroski/mask（§十四-十五）+ relation 定义（§十六-十七）
+ best_lag/spectral（§十八-十九）+ **全库 Operator Closure Audit（§二十，18 类机器审计）**。
14 个并行 agent 严格文件不相交 + 本会话直改（closure-audit 引擎、polars realized_beta twin、
composite KAMA knob 移除、2 个过期 KAMA 测试、surface retract 调和、轴对齐验证）。

## 七、算子 correctness（§一，P0）

| 项 | 状态 | 修复 | 测试 |
|---|---|---|---|
| **§1 `ts_current_drawdown_duration` NaN 穿越** | **fixed** | 回溯 streak 循环 `continue`→`break`：NaN 是 episode 硬边界，duration 永不跨 gap 累计。pandas+polars 双修 | test_r11_round2_drawdown_bestlag |
| **§2 realized `_market_model` ddof** | **fixed** | `np.cov`(÷n−1)/`np.var`(÷n) → 统一 population moments `cov=mean((r−r̄)(m−m̄))`, `beta=cov/var`；下游 idio var/skew/kurt + market R² 一并修正 | test_r11_round2_realized_beta（7） |
| **§3 日内 realized 混入隔夜收益** | **fixed** | 每交易日第一根分钟 return=NaN（日历日边界硬断），intra beta/corr/semibeta/idio/R² 全清洗。**polars twin（polars_intraday_full `_beta_frame`）由本会话同步** | 同上 |
| **§4 ex-self 权重重新引入坏市值** | **fixed** | 单次构建 ValidatedMarketWeightPanel（finite && >0），全 market 与 ex-self 共用；polars twin 同修（`w>0` 过滤） | 同上 |
| **§5 KAMA 停牌后未重新 warm-up** | **fixed** | `break + rewarm` 单一政策：gap 使状态失效，须重新积够 `er_window` 连续 finite 才 re-seed；冻结直线消失（indicators_v2 + polars_signal + **composite_fastpath 由本会话同步**） | test_r11_round2_technical_state（54） |
| **§6 Supertrend 只查 close** | **fixed** | high/low/close + 派生 ATR/basic upper/lower 全部 finite 才 state-valid，否则 break state | 同上 |
| **§7 KAMA/Supertrend/PSAR 隐藏参数** | **fixed** | 删除 `missing_policy`/`max_gap` hidden knobs（Option A：单 production 政策），DSL 只见正式签名 | 同上 + test_audit_deep_dive 2 个过期测试更新 |
| **§8 Supertrend(multiplier=0)** | **fixed** | `multiplier>0`：ParamSpec min=1e-9 + 运行时 ValueError | 同上 |
| **Markov prior-only 转移概率** | **fixed** | 无观测出边状态 → 转移行全 NaN（fail-closed），单状态窗口熵/持久性 NaN 非 0，min_count 绑定期拒绝 | test_r11_round2_markov_degenerate |
| **ConditionBool state/event/CTA 族** | **fixed** | `episode` 四新 canonical + `state_since_trend_tstat` 的 `reset_condition`、`state_event` 5 个算子的 condition slot 统一 `_assert_condition_bool`（∈{0,1,NaN}） | test_r11_round2_state_condition（61） |
| **Directional Change initial extrema + PositivePrice** | **fixed** | 首事件极值 = pre-confirmation running high/low（非固定首价）；`price>0` typed contract + 运行时 fail-closed；threshold≤0 拒绝 | test_r11_round2_directional_change（21） |
| **§42 `state_since_reduce` 动态 unit** | **fixed** | 拆 `state_since_sum/mean/count/last`（unit same_as:x / same_as:x / count / same_as:x），旧名 alias → sum；`state_since_trend_tstat` → dimensionless | test_r11_round2_state_condition |
| **§43/44 KNN unit bug** | **fixed** | `cs_knn_peer_mean_ex_self` → `same_as:target`；`cs_knn_graph_dirichlet_energy` → `unit(target)*unit(target)` | test_r11_round2_knn_units（10） |
| **§十三 多输入轴对齐** | **verified** | 中央 `validate_operator_call` + `align_panel_inputs(strict_axes=True)` 已强制 index/columns 一致（PanelAxisMismatch）；closure-audit `axis_integrity` 全库验证 6 类错位（列置换/日期位移/缺股/多股/重复时间戳/重复标的）必须拒绝 | test_r11_round2_closure_audit |

## 八、搜索空间清理（§二 + §十七，ParamSpec sweep）

- **技术指标全参数 ParamSpec**（Agent C）：DMI±/DX/NATR、PPO/PVO 系、CMO、Vortex、Keltner 系、
  TSI/TSI_signal、DEMA/TEMA、Ichimoku 全系、KAMA、Supertrend、PSAR —— 每个参数正式
  `ParamSpec`（window 严格 int、multiplier finite>0、PSAR acceleration<=maximum），`20.5` 拒绝不截断；
  `RelationalParamSpec`：fast<slow（PPO/PVO/KAMA）、TSI long>short（双 EMA 交换=重复搜索节点）、
  UltimateOscillator short<medium<long。test_r11_round2_technical_state。
- **relation 整参不截断**（Agent I）：hhi_change/entropy_change/rank_mobility/share_mobility +
  topk k + event 六算子 window/lag 全部 strict-int ParamSpec + `event_effective_lag<window` 关系约束。
  test_r11_round2_relation_defs（17）。
- **ordinal 可行性**（Agent M）：`window ≥ (order−1)·delay + min_embeddings`（min_embeddings≥8）
  绑定期拒绝，不再 run-then-all-NaN。test_r11_round2_ordinal（11）。

## 九、命名/数学信任（§三/四/五/六/十一/十六/十八/十九）

| 项 | 修复 |
|---|---|
| **§10/11 variance ratio** | `ts_variance_ratio` → `ts_variance_ratio_proxy`（alias 保留），输入语义=PriceLevel/LogPriceLevel（typed 拒绝 Return）；新增 **`ts_lo_mackinlay_vr` + `ts_lo_mackinlay_z`**（真 Lo–MacKinlay 重叠估计 + 异方差稳健 z） |
| **§12 level/vol shift 压缩 NaN 时间** | `drop-finite compress` → trailing-contiguous 处理，结构变化位置在物理时间计算 |
| **§13 CUSUM 名过宽** | `ts_cusum_break_score` → `ts_cumulative_deviation_score`（alias 保留），docstring 声明是 heuristic 非 CUSUM 检验 |
| **§9 huber/ridge 样本内残差** | 拆 `_in_sample_resid` / `_predictive_resid`（[t−W,t−1] fit→score t，无 look-ahead），自动挖因子优先 predictive；旧名 alias |
| **§14-18 MI/distance** | `window`=aligned pairs（raw=window+lag，不同 lag 可比）；`lag` strict-int + `<window`；`min_periods` ParamSpec(min=10) 去 clamp；`normalized` 严格 bool；`ts_distance_cov` unit=`sqrt(unit(x)*unit(y))` |
| **§19 Effective TE surrogate** | 压缩后 circular shift → **原始时间轴** block-shift 后重建 triples 再 re-mask，surrogate 与真实共享缺失拓扑 |
| **§20-22 forbidden ordinal** | 旧算子→`ts_forbidden_ordinal_pattern_excess`（max-0 保留）；新增 **`_ratio`**（raw F_obs）+ **`_signed_excess`**（F_obs−F_null 不 clip，保留"比随机少 pattern"信息） |
| **§23 Markov 退化状态** | 见七（prior-only + 单状态熵） |
| **§24 Directional Change** | 见七 |
| **§26 best_lag_corr** | 拆 `ts_best_lag_corr_raw`（干净 max|corr|）+ `ts_best_lag_corr_excess`（circular block permutation 估计 null，独立≈0）；Fisher-z heuristic 名不保留 |
| **§27 spectral** | 拆 `ts_return_spectral_entropy` / `ts_detrended_level_spectral_entropy`（typed input：return / (log)price level，互拒 fail-closed）；`ts_spectral_entropy` 保持 live canonical = return 方向（IR `_SPECTRAL_FAMILY` 按字面名 gate） |
| **§16 relation kurtosis** | `relation_distribution_kurtosis`（Pearson≈3）→ **`relation_distribution_pearson_kurtosis`** + 新增 **`relation_distribution_excess_kurtosis`**（=Pearson−3） |
| **§14/15 piotroski** | `piotroski_f_score` **严格**（issuance cap_g≤0，无股本增长）+ 新增 **`piotroski_f_score_tolerant`**（cap_g≤0.05）；9 分量完整性保留 |
| **§十五 `_masked()` ApplicabilityBool** | `mask.notna()&ne(0)` → **ApplicabilityBool {0,1,NaN}**：有限非{0,1} raise，0/NaN 不适用；喂 piotroski/altman/zmijewski/strength |

## 十、Operator Closure Audit 引擎（§二十，全库自动排雷）

新增 `cleaned_operators/closure_audit.py` + `tests/operators/test_r11_round2_closure_audit.py`。
遍历最终注册表全部 canonical，逐类机器审计（与 round-1 semantic_audit 的 15 规则互补）：

1. `param_type_fuzzing` — 每 scalar 参数试 5.1/True/NaN/Inf/None；**静默接受 = 假参数维度**
   （`int()` 截断 / truthiness bool）。数字字符串 "5" 不探测（round-11 契约合法强转，避免假阳性）。
2. `parameter_grid_dead_region` — 合法参数域采样；恒 all-NaN 区域 = ParamSpec 缺口。
3. `axis_integrity` — 多输入算子 6 类错位必须拒绝（中央门验证）。
4. `missing_time_topology` — 窗口算子不得跨内部 NaN 块桥接（drop-finite 重连）。
5. `recursive_rewarmup` — 状态算子 gap 后不得立刻输出自信值。
6. `condition_bool_contract` — condition/event/trigger panel 槽拒绝非{0,1,NaN}。
7. `unit_algebra` — tstat→dimensionless、energy→unit²、peer-mean→same_as（vs metadata）。
8. `boundary_behavior` — 最小参数值干净拒绝，不崩（IndexError/KeyError）。
9. `semantic_duplicates` — canonical 名 stem 聚类。

诚实记账：NOT_APPLICABLE 带原因（永不静默截断覆盖率）；RULE 自身 raise = AUDIT_ERROR（release 门禁）。

## 十一、验证

- round-2 新增测试文件（agent 各自验证）：drawdown_bestlag、realized_beta（7）、technical_state（54）、
  markov_degenerate、state_condition（61）、directional_change（21）、knn_units（10）、
  piotroski_applicability（8）、relation_defs（17）、mi_defs（18）、effte_surrogate、ordinal（11）、
  spectral_split（6）、closure_audit（8）；均为独立 pass，无 xfail/skip。
- 既有回归：intraday_next_stage 93、dynamics_pack、relation 系、stateful/event 系、te_spectral_gates、
  geometry_math、fundamental 系 —— agent 各自复跑全绿（load_all 慢但可过）。
- **最终合并验证**：round-2 全 15 个 `test_r11_round2_*.py` **269 passed**；受连锁影响的既有回归
  （audit_deep_dive / market_language / directional_tail_geometry / advanced_ops / regression_models /
  audit_fixes / field_catalog / final_pack）调和后 **254 passed**；final_pack 单文件 65 passed。
  唯一剩余失败 = `test_strict_unknown_fields_fail_closed_in_production`（`storage/sources`
  data_access 字段解析，与算子语义无关，Agent J/本会话多次复现确认既有）。
- 连锁调和（本会话）：DC PositivePrice → 旧黄金测试重推（2/9→3/9、4/9→5/9、3/6/1/6→4/6/3/6，
  prefix_causality +200 平移）+ KAMA 过期测试重写 + kramers_moyal 绑定拒绝改期望 +
  regression 旧名→新 canonical 的三测试文件/两 manifest（production_hardening、semantic_certification）
  + operator_policy 补 2 kurtosis canonical + final_pack surface/scope 表项。
- 本会话直改：composite_fastpath KAMA knob 移除、polars_intraday_full realized twin 三修、
  test_audit_deep_dive 2 个过期 KAMA 测试重写、surface retract 调和（state_since_reduce）。
- 诚实声明（load_all 慢/瞬时断）：`semantic_certification` SQL-capability 全库检查随 canonical 数量
  变慢（~2-4 min，非死锁）；并发会话在改 surface/governance，执行期 load_all 多次瞬时失败，
  全部为共享文件竞态，非本批引入。延后：`common/polars_state_event.py` polars 侧 ConditionBool
  校验（无效输入侧 pandas/polars 行为分歧，仅 invalid 输入）；`composite_fastpath` 与 indicators_v2
  的 KAMA 数值对齐复核。

---

## §十二 第三轮专项审计 — 整改计划与执行

外部 AI 第三轮「全算子专项审计（按 shared kernel/operator family）」：~147 条新整改项
（§一 P0-A..P0-P / P1-A..P1-N 算子 correctness + §二 ParamRole 参数角色 + §三 12 类机器审计）。
本会话先逐项核对现状再改：已由前两轮完成的条目标记 ALREADY，需改的按文件不相交 agent 分包
（与并发会话共享的中央文件只由本会话改：registry/operator_surface/operator_policy/
production_hardening/semantic_certification/contract_hardening/registration_audit/rolling_pack/
layer_governance/catalog docs）。

### 第一批 确定算错或语义错误（Wave-1 agent）
- stat（common/statistics+polars_statistics）：1 Mad→ts_mean/median_absolute_deviation 拆分；
  2 ACF(lag=0) 空切片；3 ACF 不可估→NaN；4 Mode max_frequency≥2；5 Residual 族定义核验。
- group_cs（common/group+group_polars+time_series+polars_ops+gtja_compat+daily_panel+
  polars_batch_mirror）：6-8 group_decay_linear window dead param/ties 平均 rank/改名；
  9-10 aggr_top_n CS routing 分类 + 非法 aggr_func raise；11 ts_argmax/argmin d/window 双参数。
- candle（price_volume/candle_geometry_v2+polars_candle+candle_pattern_engine*）：15 OHLC invariant
  统一门；16 candle_gap_atr 用 ATR_{t-1}。
- liquidity（price_volume/liquidity_v2+polars_liquidity_v2）：17 volume_autocorr window+lag；
  18 volume pct-change 0-base；19 NonNegativeVolume；20 ADL→rolling_adl_flow 拆分；21 volume_to_range；
  26 impulse_strength polars 侧 denominator 用 vol_{t-1}。
- structure（price_volume/structure_patterns_*+polars_structure+technical_extensions+
  technical_structure_repairs+technical/polars_misc_v2）：12 ts_days_since_high/low tie 取最近；
  13 resistance/support log-slope 拆分；22-23 swing 共享 SwingSegment kernel；24 fit R² points≥3；
  25 line_parallelism normalized slope；26 impulse_strength pandas 侧。
- level_profile（structural_levels+candle_state_space+rqa_ext+recurrence_analysis）：14 pivot
  staleness；63 Matrix Profile motif age off-by-(L-1)；64 constant subsequence→NaN；65-66 Mahalanobis
  center/covariance 一致 + N≥5p；67-68 RQA effective-length/fixed-RR。
- fundamental（fundamental/*+common/_pivot_ledger+shareholder/churn_network）：27 ledger 修订期支持；
  28 holder coverage；29 consecutive-period 语义；30 fundamental score threshold-relative 限制。
- tsmodel（ts_model/dynamic_regression+state_space+volatility）：31 dynamic regression 系数 slot；
  32 Kalman scale/gap；33 GARCH raw price 拒绝；34 HAR variance/vol 命名。
- complexity（complexity_ext+ts_model/complexity+sequence_complexity+threshold_cycle）：44 ordinal
  ties→drop embedding；45 sample entropy A=0→NaN；46 MSE 右对齐；47 multiscale entropy slope=log-scale；
  48 two-state regime→真 Markov switching（π=PTπ）。
- spectral_mem（ts_model/wavelet_spectral+spectral_ext+memory_ext+research_spectral）：55-57 wavelet
  NaN/固定 window/单 band H=0；58 spectral 禁 partial warmup；59 peak concentration N-dependence；
  60-61 ACF-time/IACT 标准 estimator + Geyer IPS；62 fractional diff cutoff 跟 d + discarded mass gate。

### 第二批 实盘统计稳定性（Wave-2/3 agent）
- dependence（nonlinear_dependence+dependence_ext+conditional_dependence+advanced_information）：
  49-51 conditional TE 去 jitter/effective bins/可行默认；52 pdCor 真 Székely 或 proxy 改名；
  53 HSIC bandwidth median(positive)；54 CMI effective_bins 归一化。
- geometry（polars_geometry_math+cross_section_ext+cross_section_local+cross_section+dynamic_knn）：
  91 knn_local_moran 改名或标准 Local Moran；92 isotonic 方向去 in-sample；93 KNN DOF k≥5(d+1)；
  94 local_gradient_norm unit=same_as:target；95 tangent PCA rank gate；+49-54 polars twins。
- glr_spread（glr_change+ohlc_spread）：69 GLR perfect-split 非 EPS；70 scan-selection bias；
  71 EDGE valid-pair gate；72 Abdi negative→NaN；73 PS amount 单位。
- composition_clock（composition+activity_clock+volume_clock+update_clock）：74-77 CompositionSchema
  同单位/正/同一经济整体/PartId；78 activity_clock._align raise；79 include_current 拆分；
  80 activity=0 价格一致性；81 Q=0 session_open prepend；82 buckets 固定 grid。
- intraday（intraday_session+advanced_intraday+intraday/time_structure_v2+session_recovery+
  intraday_activity_duration）：83 slot_set==official_set；84 bar 分辨率来自 contract；85 history_days
  completed sessions；86 RMSE-normalize；87 时区显式；88 min_events；89 residual_fraction 一致；
  90 Wasserstein 命名。
- tail_turnover（weighted_tail+polars_chip_tail+alpha_language_distribution+distribution_break+
  turnover_survival）：113-117 weighted semivariance/下偏差拆分+单位+ECDF inverse+tie fractional；
  139 energy distance U/V-stat 声明；140 joint_energy asof_previous；141 copula null calibration；
  142 负 turnover fail-closed；143 price missing+turnover→NaN 全分布；144 old_mass 诊断+严阈值；
  145 _MAX_OLD_MASS versioned；146-147 weighted cost ECDF + vol-scaled entropy。
- prospect_robust（prospect_theory+robust_stats+common/polars_robust_stats）：118 CPT coverage≥0.8；
  119 behavioral_score 单位；120-121 quantile_range/trimmed_mean unit=same_as:x；122 robust_zscore
  拆 inclusive/prior；123 min_periods≥max(5,0.5·window)。
- concentration_path（direction_concentration+alpha_language_shape）：124 threshold-relative 输入
  semantic 限制；125 ts_abs_entropy nats/normalized 拆分；126 normalized excess HHI 优先；
  127-131 turning_rate 分母/常量路径声明/EPS→NaN/endpoint 不 drop-finite。
- grpspec_dmd（group_spectrum+dmd）：132 group_state 标记；133 breadth key 带 schema version；
  134 重复 feature 表达式禁止；135 eigen-gap 进 definition metadata；136 DMD log-energy；137 DMD
  level/return 拆分；138 dominant oscillatory frequency。
- multiscale（multiscale_trend+crossing+envelope+interval_geometry）：96+ crossing/envelope 批次
  （review 文本截断，按 priority 句落实）。

### 第三批 清搜索空间（中央 + agent 配合）
- §二 ParamRole：base.py 已有 ParamRole/param_search_grade；本会话补 STATE_THRESHOLD 角色并把
  param_search_grade 接入实际搜索 grammar（operator_cost_model/search 枚举），
  NUMERICAL/POLICY/ESTIMATOR_RESOLUTION 默认不进 AlphaProbe；各 agent 对 bins/grid/n_segments/
  ridge/eigen-gap/surrogate shifts 等参数声明 role 并 searchable 收敛。
- §三 12 类审计：closure_audit 已有 param_type_fuzzing(1)/axis_integrity(2)/missing_time_topology(3)/
  recursive_rewarmup(4)/unit_algebra(5)/boundary_behavior(8)/semantic_duplicates(11)；
  operator_audits 有 golden_reference(6)/effective_sample_size(8)/column_permutation(9)。
  本会话补：null_calibration(7)、practical_usability(12)、semantic_input_type(10)。
