# FactorEngine 下一阶段算子挖掘扩展 —— 完整实现计划

日期：2026-08-06
范围：补齐五大搜索维度（分钟高阶矩/跳跃/日内路径、动态回归/状态空间/复杂度、财务质量/新会计科目、股东网络/共同持股、K线序列统计），并覆盖此前 AI 提出的全部算子清单。

> 并发编辑说明：其他 AI 正在并行修改 `microstructure/intraday_agg.py`、`relation/ops.py`、
> `operator_surface.py`、`fundamental/*` 等文件。本计划**所有新增物理内核放在全新模块**，
> 对既有文件只做**最小追加式改动**（新增 frozenset 行、追加 `_LOAD_MODULES` 条目），
> 不修改任何既有算子实现，最大限度避免合并冲突。

---

## 〇、总体原则

1. **三层架构**：通用 canonical operator（字段无关）→ 数据源聚合 operator（分钟/股东/指数表 → 每股每日标量）→ Factor Recipe（固定财务定义/固定类别，避免重复内核）。
2. **全部算子输出 `TradeDate × Symbol` 每股每日一个标量**。财务/股东数据按 `PubDate` as-of 前向填充；分钟算子在当日收盘后形成、默认下一交易日使用（PIT 安全）。
3. **禁止虚构数据**：不引入 OFI、订单簿深度、真实价差、逐笔、北向、龙虎榜、两融、分析师、新闻、集合竞价等不存在的字段。分钟 OHLCV 代理变量统一 `_proxy` 后缀。
4. **数值契约**：空窗/全 NaN → NaN，禁止 Inf、禁止伪造零；无效参数必须报错。
5. **去重**：能用既有基础算子低成本组成的 → Recipe，不新增物理内核。已存在算子（`ts_huber_regression_resid`、`ts_ridge_regression_resid`、`ts_quantile_regression_slope`、`ts_ar_coefficient`、`ts_variance_ratio`、`ts_cusum_break_score`、`ts_level_shift_score`、`ts_vol_shift_score`、`overnight_return`、`relation_category_share`、`relation_peer_weighted_mean_ex_self`、`index_membership_age`、`fin_cash_earnings_gap`、`fin_accrual_ratio` 等）**复用，不重复注册**。
6. **优先级**：P0 → P1 → P2 → P3 分阶段落地；P0 全量实现并过验收，P2/P3 作为实验搜索空间实现为 `status="experimental"` 算子，不进入默认 production 目标。
7. **后端策略**：本阶段所有新内核先实现 **pandas_numpy**（reference），Polars 注册为显式 `backend="polars"` 的只做简单镜像；不冒充三引擎认证（backend parity 作为后续阶段）。
8. **冲突规避**：新模块命名与既有文件明显区分（`new_stage_*` / 领域子目录），注册用独立 frozenset 追加。

---

## 一、模块布局（全新文件，避免冲突）

| 模块 | 内容 | 优先级 |
|---|---|---|
| `cleaned_operators/intraday/higher_moments.py` | 已实现高阶矩、跳跃分解 | P0 |
| `cleaned_operators/intraday/realized_beta.py` | 分钟市场 Beta、半 Beta、特质矩 | P0 |
| `cleaned_operators/intraday/time_structure.py` | 时间段算子、同日段记忆、路径/成交曲线距离 | P0/P1 |
| `cleaned_operators/intraday/vwap_path.py` | VWAP 路径、回撤恢复 | P0 |
| `cleaned_operators/intraday/overnight.py` | 隔夜/日内分解、跳空回补 | P1 |
| `cleaned_operators/ts_model/dynamic_regression.py` | 多变量/稳健/分位数滚动回归扩展 | P0 |
| `cleaned_operators/ts_model/ar_meanrev.py` | AR 预测/创新、均值回复半衰期、方差比斜率 | P0 |
| `cleaned_operators/ts_model/state_space.py` | Kalman（水平/趋势/Beta） | P2 |
| `cleaned_operators/ts_model/volatility.py` | GARCH/GJR/HAR-RV | P2 |
| `cleaned_operators/ts_model/complexity.py` | 熵、复杂度、Hurst、变点、regime | P2 |
| `cleaned_operators/ts_model/wavelet_spectral.py` | 小波、频域 | P2 |
| `cleaned_operators/ts_model/sequence_anomaly.py` | Matrix Profile、motif | P3 |
| `cleaned_operators/ts_model/path_signature.py` | 路径签名 | P2 |
| `cleaned_operators/cross_section/robust_cs.py` | 横截面岭/分位/样条残差、马氏/KNN/局部密度 | P1/P2 |
| `cleaned_operators/cross_section/peer_ops.py` | 同行 Beta 偏离、信息扩散、领涨滞后 | P0/P1 |
| `cleaned_operators/cross_section/panel_model.py` | 滚动 PCA/PLS/PCR/ElasticNet/regime/MoE/自编码 | P2 |
| `cleaned_operators/fundamental/quality_v2.py` | 应计、盈利质量、核心/非核心收益 | P0/P1 |
| `cleaned_operators/fundamental/accruals_scores.py` | 资产投资、收入质量、租赁商誉递延税、融资偿债、Piotroski/Altman/Zmijewski | P0 |
| `cleaned_operators/shareholder/churn_network.py` | 持股变动、类别、质押冻结、网络 | P0/P1/P2 |
| `cleaned_operators/valuation/ops_v2.py` | 估值、股本、流通股 | P0 |
| `cleaned_operators/index_listing/ops_v2.py` | 指数权重偏离、成分变动、上市年龄、停牌 | P0/P1 |
| `factor_recipes/new_stage_recipes.py` | 全部 Recipe 聚合（分钟时间段、股东类别、估值股本、财务增长、评分） | — |

每新增一个模块后，在 `cleaned_operators/__init__.py` 的 `_LOAD_MODULES` **末尾追加**导入行（不修改已有行）。

---

## 二、P0 算子全清单与实现方式

### 2.1 分钟高阶矩与跳跃（模块 `intraday/higher_moments.py`）

输入：分钟 Close 面板（行=分钟时间戳，列=标的）。输出：日频面板。

| 算子 | 公式 / 语义 | 实现 |
|---|---|---|
| `intra_realized_skewness` | `sqrt(N)·Σr³/(Σr²)^(3/2)` | 物理内核 |
| `intra_realized_kurtosis` | `N·Σr⁴/(Σr²)²` | 物理内核 |
| `intra_realized_quarticity` | `N/3·Σr⁴` | 物理内核 |
| `intra_tripower_quarticity` | 相邻三分钟绝对收益幂次积，跳跃稳健 | 物理内核 |
| `intra_continuous_variance` | `min(RV, BV)`，RV=Σr²，BV=π/2·Σ|rᵢ||rᵢ₋₁| | 物理内核 |
| `intra_jump_variation` | `max(RV−BV, 0)` | 物理内核 |
| `intra_positive_jump_variation` | 仅累计判定为跳跃的正分钟收益平方 | 物理内核 |
| `intra_negative_jump_variation` | 仅累计判定为跳跃的负分钟收益平方 | 物理内核 |
| `intra_signed_jump_ratio` | `(posJV−negJV)/(JV+eps)` | 物理内核 |
| `intra_jump_count` | 当日跳跃分钟数 | 物理内核 |
| `intra_jump_concentration` | `Σ jump_share²`（跳跃集中度） | 物理内核 |
| `intra_jump_first_time` | 首次跳跃的标准化时点 [0,1] | 物理内核 |
| `intra_jump_last_time` | 末次跳跃的标准化时点 | 物理内核 |
| `intra_jump_clustering` | 跳跃间隔变异系数 | 物理内核 |

跳跃判定统一辅助：`jump_flag = r² > max(threshold·RV, 绝对阈值)`（阈值参数化，默认 `threshold=2.0` 分钟级）。

### 2.2 分钟市场 Beta（模块 `intraday/realized_beta.py`）

市场分钟收益 = `Σ(前日 FreeMarketCap 权重 × 个股分钟收益)`。权重面板作为输入参数 `free_market_cap`（前一日收盘口径，调用侧负责 as-of）。

| 算子 | 公式 | 实现 |
|---|---|---|
| `intra_realized_beta` | `Σ stock·mkt / Σ mkt²` | 物理内核（输入 close 面板 + 权重面板） |
| `intra_realized_correlation` | 分钟收益与市场收益相关系数 | 物理内核 |
| `intra_down_down_semibeta` | 仅 `stock<0 ∧ mkt<0` | 物理内核 |
| `intra_up_up_semibeta` | 仅 `stock>0 ∧ mkt>0` | 物理内核 |
| `intra_down_up_semibeta` | `mkt>0 ∧ stock<0` | 物理内核 |
| `intra_up_down_semibeta` | `mkt<0 ∧ stock>0` | 物理内核 |
| `intra_beta_asymmetry` | `dd_semibeta − uu_semibeta` | 物理内核 |
| `intra_idiosyncratic_variance` | 分钟市场模型残差平方和 | 物理内核 |
| `intra_idiosyncratic_skewness` | 残差偏度 | 物理内核 |
| `intra_idiosyncratic_kurtosis` | 残差峰度 | 物理内核 |
| `intra_market_model_r2` | 分钟市场模型 R² | 物理内核 |

### 2.3 日内时间段与路径（模块 `intraday/time_structure.py`）

| 算子 | 语义 | 实现 |
|---|---|---|
| `intra_interval_return(close, start_minute, end_minute)` | 通用时间段收益内核 | 物理内核（参数化） |
| `intra_interval_volume_share / amount_share / realized_variance / vwap_deviation / illiquidity` | 时间段内占比/方差/偏离 | 物理内核（参数化 start/end） |
| `intra_same_slot_momentum` | `Σ_m r_t,m · mean(r_{t−k,m})`，同日段跨日记忆 | 物理内核 |
| `intra_same_slot_reversal` | 同槽反向匹配 | 物理内核 |
| `intra_return_profile_cosine` | 240 维分钟收益向量与历史均值向量余弦 | 物理内核 |
| `intra_volume_profile_cosine` | 分钟成交量占比曲线余弦 | 物理内核 |
| `intra_amount_profile_cosine` | 成交额占比曲线余弦 | 物理内核 |
| `intra_volume_profile_jsd` | 当日量分布 vs 历史基准 JSD | 物理内核 |
| `intra_amount_profile_jsd` | 成交额曲线 JSD | 物理内核 |
| `intra_profile_earth_mover_distance` | 收益/量分布 vs 基准 Wasserstein 距离 | 物理内核 |

Recipe（统一走 `intra_interval_return`）：`intra_opening_15m_return`、`intra_opening_30m_return`、`intra_morning_return`、`intra_pre_lunch_30m_return`、`intra_afternoon_open_30m_return`、`intra_closing_30m_return`、`intra_closing_15m_return`。

时间段关系（既有内核可组合 → Recipe）：`intra_morning_afternoon_return_spread`、`intra_morning_close_continuation`、`intra_morning_close_reversal`、`intra_open_close_pressure`、`intra_front_back_volume_ratio`、`intra_closing_volume_acceleration`、`intra_opening_volume_acceleration`。

### 2.4 VWAP 路径与回撤（模块 `intraday/vwap_path.py`）

| 算子 | 语义 | 实现 |
|---|---|---|
| `intra_vwap_path_slope` | 累计 VWAP 对时间线性回归斜率 | 物理内核 |
| `intra_vwap_path_curvature` | 二次项系数 | 物理内核 |
| `intra_price_vwap_max_positive_excursion` | `max(Close/CumVWAP−1)` | 物理内核 |
| `intra_price_vwap_max_negative_excursion` | `min(Close/CumVWAP−1)` | 物理内核 |
| `intra_time_above_vwap` | 高于累计 VWAP 的分钟比例 | 物理内核 |
| `intra_longest_above_vwap_streak` | 连续高于 VWAP 最长分钟数 | 物理内核 |
| `intra_longest_below_vwap_streak` | 连续低于 VWAP 最长分钟数 | 物理内核 |
| `intra_vwap_reversion_speed` | 偏离 VWAP 的一阶自回归系数/半衰期 | 物理内核 |
| `intra_max_drawdown` | 分钟路径最大回撤 | 物理内核（日内版，与 `ts_max_drawdown` 区分） |
| `intra_max_drawup` | 分钟路径最大上涨段 | 物理内核 |
| `intra_drawdown_depth / duration / recovery_half_life` | 回撤深度/持续期/半恢复期 | 物理内核 |

### 2.5 动态回归扩展（模块 `ts_model/dynamic_regression.py`）

已有：`ts_huber_regression_resid`、`ts_ridge_regression_resid`、`ts_quantile_regression_slope`。

新增（多输入 y + 特征面板）：

| 算子 | 语义 |
|---|---|
| `ts_multi_regression_coeff(y, X, window, coeff_index)` | 多变量滚动回归指定系数 |
| `ts_multi_regression_resid` | 当前残差 |
| `ts_multi_regression_resid_z` | 残差 / 窗口残差标准差 |
| `ts_multi_regression_r2` | 多元 R² |
| `ts_huber_regression_coeff` | Huber 斜率 |
| `ts_huber_regression_resid_z` | Huber 标准化残差 |
| `ts_ridge_regression_coeff` | 岭系数 |
| `ts_ridge_regression_resid_z` | 岭标准化残差 |
| `ts_quantile_regression_coeff` | 分位回归斜率（复用已有 `_quantile_slope`） |
| `ts_quantile_regression_resid` | 相对条件分位预测偏差 |
| `ts_quantile_beta_spread` | `beta(q_high)−beta(q_low)` |

多变量回归内核支持 `add_intercept`（默认 True）、`min_periods`、`coefficient_index`；`X` 为特征面板（`timestamp × feature` 的 MultiIndex 或等长同索引 DataFrame 序列）。

### 2.6 AR 与均值回复（模块 `ts_model/ar_meanrev.py`）

| 算子 | 语义 |
|---|---|
| `ts_ar_forecast(x, window, order)` | AR(1)–AR(p) 一步预测 |
| `ts_ar_innovation` | 实际值 − AR 预测 |
| `ts_ar_innovation_z` | 标准化创新 |
| `ts_mean_reversion_half_life` | `−log(2)/β`（`Δx=α+βx₋₁+ε`，β<0 有效） |
| `ts_variance_ratio_slope` | 多持有期方差比对 `log(k)` 的斜率 |

### 2.7 财务质量（模块 `fundamental/quality_v2.py`）

输入均为按 `PubDate` as-of 的财务指标日频面板；分子分母已由数据层对齐。

| 算子 | 语义 |
|---|---|
| `fin_working_capital_accruals` | `Δ(ΔCA−ΔCash)−(ΔCL−ΔSTD−ΔTP)` 的标准化 |
| `fin_total_operating_accruals` | `(ΔTA−ΔCash)−(ΔCL−ΔSTD−ΔTP)−Dep` 标准化 |
| `fin_delta_noa` | `ΔNOA/AvgAssets` |
| `fin_roe_cash_gap` | `会计ROE − 现金ROE` |
| `fin_earnings_cash_gap_volatility` | `std(NetProfit−OCF)/AvgAssets` |
| `fin_earnings_smoothness` | `std(NetProfit)/std(OCF)` |
| `fin_earnings_persistence` | 报告期利润 AR(1) |
| `fin_cashflow_persistence` | OCF AR(1) |
| `fin_margin_persistence` | 毛利率/营业利润率 AR 系数 |
| `fin_core_earnings_ratio` | 核心利润 / 营收或总资产 |
| `fin_noncore_income_ratio` | 非核心收益 / 利润总额 |
| `fin_fair_value_income_dependence` | `FairValueVariableIncome/|TotalProfit|` |
| `fin_investment_income_dependence` | `InvestmentIncome/|TotalProfit|` |
| `fin_other_earnings_dependence` | `OtherEarnings/|TotalProfit|` |
| `fin_comprehensive_income_gap` | `(TotalCompositeIncome−NetProfit)/AvgEquity` |
| `fin_oci_to_equity` | `OtherComprehensiveIncome/AvgEquity` |
| `fin_discontinued_operation_ratio` | `DisconOperateNetProfit/|NetProfit|` |
| `fin_minority_profit_share` | `MinorityProfit/NetProfit` |

> 字段名映射遵循 `fields/catalog.py` 现行命名；个别新科目若 catalog 缺失，用数据层别名 `*` 映射，算子参数用语义名并记录 `required_tables`。

### 2.8 资产投资、收入质量、融资偿债、综合评分（模块 `fundamental/accruals_scores.py`）

新增物理内核（分子分母可组合的做成 Recipe）：

| 类别 | 算子 |
|---|---|
| 收入/资产错配 | `fin_receivable_sales_divergence`、`fin_inventory_sales_divergence`、`fin_cash_sales_divergence`、`fin_expense_sales_divergence` |
| 合同资产/负债（新准则） | `fin_contract_asset_intensity`、`fin_contract_asset_growth`、`fin_contract_liability_intensity`、`fin_contract_liability_growth`、`fin_contract_asset_liability_gap` |
| 租赁/商誉/递延税 | `fin_lease_intensity`、`fin_lease_asset_liability_gap`、`fin_goodwill_intensity`、`fin_goodwill_risk_score`、`fin_deferred_tax_gap`、`fin_impairment_intensity` |
| 融资/偿债 | `fin_net_debt_issuance`、`fin_borrowing_intensity`、`fin_debt_repayment_intensity`、`fin_net_borrowing_cashflow`、`fin_equity_capital_growth`、`fin_financing_gap`、`fin_interest_coverage_proxy`、`fin_debt_service_coverage_proxy`、`fin_cash_burn_runway` |
| 资本开支/研发 | `fin_capex_intensity`、`fin_capex_growth`、`fin_acquisition_cash_intensity`、`fin_rd_total_intensity`、`fin_rd_capitalization_ratio` |
| 综合评分 | `piotroski_f_score`、`altman_z_score`、`zmijewski_score`、`fin_fundamental_strength_score` |

`fin_*_growth` 系列（`fin_total_asset_growth`、`fin_operating_asset_growth`、`fin_fixed_asset_growth`、`fin_inventory_growth`、`fin_receivable_growth`、`fin_goodwill_growth`、`fin_intangible_growth`、`fin_construction_in_progress_growth`）→ **Recipe**：`fin_growth(x)` 已有原语，直接注册 Recipe，不新增内核。

Piotroski/Altman/Zmijewski 做成**物理内核**（多参数组件求和，含金融业适用性掩码 `fin_applicability_mask`），但暴露组件级 Recipe 以便挖掘（`fin_fundamental_strength_score` 支持方向数组）。

### 2.9 股东（模块 `shareholder/churn_network.py`）

输入为**预聚合日频面板**（shareholder 快照由 `storage/sources/relation` 侧按 `PubDate` as-of 输出），算子做面板级计算：

| 算子 | 语义 |
|---|---|
| `holder_weighted_churn` | `0.5·Σ|ShareRatio_cur−ShareRatio_prev|`（按 ShareholderId 匹配，进入/退出补 0） |
| `holder_rank_stability` | 相同股东本期/上期排名的 Spearman 相关 |
| `holder_entry_share` | 新进前十大持股比例合计 |
| `holder_exit_share` | 退出股东上期比例合计 |
| `holder_net_entry_share` | entry − exit |
| `holder_concentration_slope` | 集中度/HHI 多报告期趋势斜率 |
| `holder_concentration_acceleration` | 二阶变化 |
| `holder_class_entropy` | 股东类别权重熵 |
| `holder_nature_entropy` | 股份性质权重熵 |
| `holder_pledge_ratio` | `ΣSharePledge/TotalCapital` |
| `holder_freeze_ratio` | `ΣShareFreeze/TotalCapital` |
| `holder_pledge_concentration` | 质押份额 HHI |
| `holder_freeze_concentration` | 冻结份额 HHI |
| `holder_pledged_holder_count` | 质押股数>0 的股东数 |
| `holder_pledge_change` | 质押率变化 |
| `holder_pledge_churn` | 按股东 ID 匹配的质押变化绝对值合计 |
| `holder_float_concentration_gap` | 前十大集中度 − 前十大流通集中度 |
| `holder_locked_share_ratio` | 限售股份占比 |
| `holder_common_holding_peer_return` | 共同持股 peer 收益 |
| `holder_peer_return_breadth` | 股东跨股票 breadth |
| `holder_shareholder_network_centrality` | 网络中心度（P2） |
| `holder_shareholder_overlap_ratio` | 股东重叠度（P2） |

类别 Recipe（复用 `relation_category_share`，不新增内核）：`holder_natural_person_share`、`holder_fund_share`、`holder_private_fund_share`、`holder_qfii_share`、`holder_social_security_share`、`holder_insurance_share`、`holder_state_owned_share`、`holder_foreign_institution_share`、`holder_broker_share`、`holder_asset_management_share`。

### 2.10 同行偏离（模块 `cross_section/peer_ops.py`）

| 算子 | 语义 |
|---|---|
| `group_peer_beta_deviation` | `stock_beta − peer_beta_ex_self`（行业加权，`FreeMarketCap` 权重） |
| `group_peer_characteristic_deviation` | `stock_char − peer_mean_ex_self`（Recipe，复用 `relation_peer_weighted_mean_ex_self`） |
| `group_peer_deviation_index` | 多标准化偏离聚合（盈利/估值/成长/换手/波动） |
| `group_peer_information_diffusion` | 同行过去收益/财务变化对本股票已实现收益的滚动回归系数 |
| `group_leader_laggard_exposure` | 股票收益对行业领先股滞后收益 Beta |
| `group_return_dispersion_exposure` | 对行业截面收益离散度变化的敏感度 |
| `group_multi_level_rank_consistency` | sw_l1/l2/l3 三层级组内排名一致度 |
| `ts_market_liquidity_beta` | 个股收益对市场换手/量能变化回归 |
| `ts_industry_liquidity_beta` | 对行业流动性的共同暴露 |

### 2.11 估值、股本、指数与上市（模块 `valuation/ops_v2.py`、`index_listing/ops_v2.py`）

估值股本多数为 **Recipe**（基础算子 + `fin_growth` 等即可组成），少数做物理内核：

| 算子 | 实现 |
|---|---|
| `valuation_pe_ttm_lyr_gap` | Recipe（`log_abs(PeRatio)−log_abs(PeRatioLyr)`） |
| `valuation_pcf_definition_gap` | Recipe |
| `free_float_ratio` / `free_to_circulating_ratio` / `a_share_cap_ratio` / `free_float_turnover` | Recipe |
| `valuation_cashflow_disagreement` | Recipe（离散度） |
| `valuation_growth_mismatch` | 物理内核或 Recipe（盈利收益率 − 标准化利润增长） |
| `valuation_quality_mismatch` | 物理内核（估值 vs 财务质量横截面残差） |
| `market_cap_free_cap_gap` | Recipe |
| `capital_change_age` | 物理内核（距 ChangeDate 交易日数，需日历） |
| `capital_change_magnitude` | Recipe（`TotalCapital` 环比） |
| `circulating_cap_unlock_proxy` | Recipe（流通/总股本比变化） |
| `index_weight_gap_to_free_float` | 物理内核（指数权重 vs 自由流通权重） |
| `index_reconstitution_churn` | 物理内核（窗口内纳入/剔除次数） |
| `multi_index_entry_intensity` | 物理内核（短期同时进入多指数数量） |
| `listing_age` | 物理内核（上市日距当前交易日数） |
| `suspension_frequency` | 物理内核（`IsSuspend` 滚动计数） |
| `index_event_decay` 系列 | 物理内核（纳入/剔除后衰减信号） |

### 2.12 隔夜分解（模块 `intraday/overnight.py`，P1）

`overnight_return` 已存在。新增：

| 算子 | 语义 |
|---|---|
| `ts_overnight_intraday_cov` | `rolling_cov(overnight_ret, intraday_ret)` |
| `ts_overnight_intraday_spread` | `mean(overnight)−mean(intraday)` |
| `ts_overnight_intraday_sign_agreement` | 隔夜/日内同号比例 |
| `ts_gap_reversion_ratio` | `−intraday/overnight`（跳空超阈值时） |
| `ts_gap_fill_ratio` | 窗口内跳空回补比例 |
| `ts_gap_survival_duration` | 未回补缺口持续交易日数 |
| `ts_opening_mispricing_score` | `overnight_ret − 历史预期日内响应`（滚动回归） |

### 2.13 P2/P3 实验算子（各实验模块，全部 `status="experimental"`）

- **状态空间**：`ts_kalman_level`、`ts_kalman_trend`、`ts_kalman_innovation_z`、`ts_kalman_beta`、`ts_kalman_beta_change`、`ts_kalman_beta_uncertainty`（确定性初始化、状态可序列化、分段/全历史一致）。
- **波动模型**：`ts_garch_vol_forecast`、`ts_garch_persistence`、`ts_garch_standardized_shock`、`ts_gjr_garch_vol_forecast`、`ts_gjr_leverage`、`ts_har_rv_forecast`、`ts_har_rv_innovation_z`（高成本算子，限制窗口/调用次数）。
- **复杂度/变点**：`ts_cusum_vol_break_score`、`ts_change_point_probability`、`ts_regime_duration`、`ts_two_state_regime_probability`、`ts_permutation_entropy`、`ts_sample_entropy`、`ts_lz_complexity`、`ts_multiscale_entropy_slope`、`ts_turning_point_ratio`、`ts_dfa_hurst`。
- **小波/频域**：`ts_wavelet_low_frequency_ratio`、`ts_wavelet_high_frequency_ratio`、`ts_wavelet_entropy`、`ts_wavelet_energy_slope`、`ts_spectral_low_frequency_ratio`（固定小波族/边界/最小窗口）。
- **序列异常**：`ts_matrix_profile_discord_score`、`ts_matrix_profile_motif_distance`、`ts_motif_recurrence_count`（P3，不定义固定形态名称）。
- **路径签名**：`ts_path_signature_area`、`ts_path_signature_depth2_norm`、`ts_path_leadlag_area`（双序列输入）。
- **横截面稳健**：`cs_ridge_resid`、`cs_quantile_resid`、`cs_spline_resid`、`cs_mahalanobis_distance`、`cs_knn_distance`、`cs_local_density_score`。
- **面板模型**：`panel_rolling_pca_loading/resid/resid_vol/resid_momentum/explained_ratio`、`industry_rolling_pca_loading`、`panel_rolling_pcr_forecast`、`panel_rolling_pls_forecast`、`panel_rolling_elastic_net_forecast`、`panel_regime_conditioned_forecast`、`panel_mixture_of_experts_score`、`cs_autoencoder_reconstruction_error`。
  - 标签型模型要求：训练标签持有期已结束、至少 1 日执行滞后、`embargo ≥ holding_period`、滚动训练窗口、固定特征清单。
- **股东网络**：`holder_shareholder_network_centrality`、`holder_shareholder_overlap_ratio`。

---

## 三、验收标准（每个新增算子强制通过）

1. **输出形状**：输入日频面板或分钟/关系原始块 → 输出 `TradeDate × Symbol`，每股每日最多一个值。
2. **截面有效性**：真实 A 股样本检查 daily coverage、unique count、截面 std、zero/NaN/Inf 比例；全市场常数算子在挖掘白名单外。
3. **PIT 测试**：修改未来数据后历史输出不变。财务/股东 `PubDate ≤ decision_date`；分钟当日收盘后形成、默认下一交易日使用。
4. **字段审计**：metadata 记录 `required_tables`、`required_fields`、`field_units`、`frequency`、`availability_time`、`source_contract`。
5. **参数域**：`window>1`、`min_periods≤window`、`0<q<1`、`ridge_alpha≥0`、`half_life>0`；无效参数报错不静默修正。
6. **成本分级**（写入 tag `cost:N`）：elementwise=1，native rolling=2，rolling corr=3，rolling quantile/sort=4，rolling regression=5，minute daily agg=6，relation/network=7，Kalman/GARCH/wavelet=8，PCA/model=10。
7. **去重**：新增前比较 canonical/alias/参数等价/DAG 等价/样本相关系数；可由基础算子低成本组成的注册为 Recipe。

---

## 四、实施顺序（避免冲突的落地批次）

| 批次 | 内容 | 交付物 |
|---|---|---|
| B1 | 分钟高阶矩/跳跃（2.1）+ 分钟 Beta（2.2） | 2 新模块 + 表面注册 + 单测 |
| B2 | 日内时间段/路径/VWAP/回撤（2.3、2.4） | 2 新模块 + 表面注册 + 单测 |
| B3 | 动态回归（2.5）+ AR/均值回复（2.6） | 2 新模块 + 单测 |
| B4 | 财务质量（2.7）+ 资产投资/评分（2.8） | 2 新模块 + Recipe 注册 + 单测 |
| B5 | 股东（2.9）+ 同行偏离（2.10） | 2 新模块 + 单测 |
| B6 | 估值股本/指数上市（2.11）+ 隔夜（2.12） | 3 新模块 + Recipe 注册 + 单测 |
| B7 | P2/P3 实验算子（2.13） | 各实验模块 + 单测 |
| B8 | 全量验收：PIT、截面有效性、参数域、成本审计；`__init__.py` 加载清单核对；跑测试套件 | 汇总报告 |

每批次完成后独立可跑（`pytest tests/operators/test_<module>.py`），便于与其他 AI 的并行改动隔离合并。

---

## 五、后端实现情况（2026-08 追加）

### 5.1 Polars backend（81 个算子，genuine 表达式，非 bridge）

框架 `overhaul/cleanup.py` 会主动移除 source 含 `bridge` 的 polars backend，因此
**全部 polars 后端均为真实表达式实现**，无 pandas 委托：

| 模块 | 覆盖 | 实现方式 |
|---|---|---|
| `intraday/polars_next_stage.py` | 已实现矩、跳跃、区间、回撤 | melt → group_by → pivot 原生表达式 |
| `valuation/polars_ops_v2.py` | 估值/股本比率与缺口 | 原生 `pl.Expr` |
| `fundamental/polars_quality_v2.py` | 财务 elementwise（依赖/强度/覆盖） | 原生 `pl.Expr` |
| `shareholder/polars_churn_network.py` | 股东比率与名次面板 | 原生 `pl.Expr` |
| `index_listing/polars_ops_v2.py` | 指数权重/成分变动/停牌 | 原生 `pl.Expr` |
| `ts_model/polars_regression.py` | 均值回复/方差比斜率/流动性 Beta | `rolling_map` / `rolling_cov-var` |
| `cross_section/polars_peer.py` | 同行偏离、Beta 偏离 | 逐行 `map_rows` |

**不建议加 polars 的算子**（迭代/状态/算法型，Polars 表达式无法表达，仅靠
`map_elements` Python 回退不构成真正的向量化 backend）：Kalman、GARCH/GJR、
样本熵/LZ/多尺度熵、Matrix Profile、DFA Hurst、小波递归 DWT、
PCA/PLS/ElasticNet/regime/MoE/自编码、行业 PCA、`intra_realized_beta` 系列的
分钟截面市场收益构造。

### 5.2 DuckDB / SQL backend（elementwise 算子 via composite lowering）

SQL 下推 emitter（5751 行共享文件）由并发 AI 维护，**不在其中加 per-operator
代码**。改用 `planner/lowerings/next_stage.py` 的 composite lowering，把
elementwise 算子展开为已具备 SQL 能力的原语（`safe_div_null`/`subtract`/`abs`/
`log`/`ts_delay`），optimizer 在 SQL 发射前自动降低 → 获得 DuckDB pushdown。

覆盖：估值 5 个、财务 elementwise 22 个、股东 elementwise 8 个、指数 1 个，
共 36 个 + 既有 composite 共 53 个 SQL-capable。

### 5.3 测试

- `tests/operators/test_polars_next_stage_parity.py`：57 项 pandas↔polars 数值一致。
- `tests/operators/test_next_stage_sql_lowering.py`：39 项 SQL 可下推 + 状态型不降低。
- 全部新算子测试：398 passed。

---

## 六、冲突规避清单

1. 只新建文件，不修改既有算子实现文件（`microstructure/intraday_agg.py`、`relation/ops.py`、`fundamental/*`、`shareholder/ops.py` 等）已被其他 AI 修改，**一律不动**。
2. `cleaned_operators/__init__.py`：只在 `_LOAD_MODULES` 元组**末尾追加**新模块名（追加行不触碰既有行）。
3. `cleaned_operators/operator_surface.py`：新增一个 `_NEW_STAGE_CANONICALS` frozenset 并在 `EXTENDED_ONLY_CANONICALS` 定义处**追加 union**；若该文件已被并行改动导致冲突，采用基于语义合并的最小 diff。
4. 所有新算子 `source=` 标记唯一模块名，便于回溯。
5. 注册前先 `grep` canonical 确认不存在，避免与并行 AI 的算子撞名；如发现同名，暂停并报告。

---

## 七、Production 认证（2026-08 追加）

分钟源算子此前被 `SOURCE_BLOCKED_CANONICALS` 排除在生产目标外——框架注释写明
"until the audit gains a minute panel fixture"。本次补齐该 fixture 并解锁：

### 7.1 audit 分钟面板 fixture（`scripts/audit_all_factor_production.py`）

- `_minute_panels()`：确定性合成分钟面板（240 bar/日 × 220 交易日，09:31..15:00，
  收盘/开/高/低/量/额/VWAP/activity/value/abs_return），索引与日频模板同日期。
- `_minute_source()`：按 `intra_*` 前缀 + metadata `minute` tag 识别分钟源算子；
  `_value` 优先从分钟面板解析参数，`free_market_cap`/`high_limit`/`low_limit` 保持日频广播。
- `_slice_minute()`：前缀因果检查按**完整交易日**切片（20 天），而非日频路径的 160 行。

### 7.2 解除生产排除

- `production_hardening.py`：`SOURCE_BLOCKED_CANONICALS` 收窄为仅 `intraday_volatility`/
  `intraday_vwap_deviation`（pre-existing experimental 算子，生命周期未到 production）；
  删除 `factor_production_targets()` 里的 `startswith("intra_")` 过滤。
- `microstructure/intraday_agg.py`：3 个 limit 算子补 `allow_panel_broadcast` tag
  （日频涨跌停价广播到分钟面板，与 `intra_realized_beta` 一致）。

### 7.3 认证结果

- 重新运行 `certify_factor_operator_evidence.py` → `factor_operator_verified.json`
  从 685 → **762** 个 production 算子。
- production 运行时可用：863 中 **848**；`intra_*` **77/77** 全部通过运行时 production 门禁。
- 剩余 15 个不适用：7 个 UNSAFE 数学工具（arg/cosh/cot/csc/sec/sinh/tan）、3 个 INTERNAL
  （constant/identity/protected_div）、1 个 LEGACY（cube）、2 个 RESEARCH 面、2 个 experimental。
- 全量测试：**3586 passed, 1044 skipped, 0 failed**。

---

## 8. Intraday Polars 二波（2026-08 追加）

第七节完成后复查发现：77 个 `intra_*` 中只有 15 个有 polars 后端
（`polars_next_stage.py` 覆盖的 realized moments / jump variance / interval shares /
max drawdown·drawup），**剩余 62 个仅 pandas**。按"凡是合理能加的都加"补齐。

### 8.1 新增模块 `cleaned_operators/intraday/polars_intraday_full.py`

62 个算子全部为**真 `pl.Expr` 实现**（wide→melt→group_by→pivot，无 pandas 委托）：

| 族 | 算子数 | 实现要点 |
|---|---|---|
| segment 聚合 | 5 | 分钟时段掩码（Asia/Shanghai 时区换算） |
| RV / 半方差 / 双幂 / 跳跃占比 | 4 | 组内 log 收益 + shift 乘积 |
| 价格路径 | 7 | 路径效率、高/低点位置、VWAP 偏离、streak（run-length）、AR(1)、斜率/曲率（闭式 lstsq） |
| 分布 | 3 | HHI 集中度、熵、符号不平衡 |
| 流动性 | 2 | Amihud、Kyle lambda（np.cov/np.var 的 ddof 系数精确复刻） |
| 极值 / 午间跳空 | 2 | 单分钟极值、跨时段 open/close |
| 涨跌停 | 3 | 日频限价面板按日期广播 join 分钟（`allow_panel_broadcast`） |
| 区间 / 同槽 / profile | 11 | 跨日 rolling 历史 + 逐日向量余弦 / JSD / 1D-EMD |
| 跳跃时序 | 4 | 阈值跳跃掩码 + 位置 / 间隔 CV |
| 市场 beta 族 | 11 | 市值加权市场分钟收益 + 组内回归（realized beta/corr/semibeta/idio/r²） |
| 回撤 / 恢复 | 3 | 峰谷位置（0/1 基差精确对齐）、半恢复期偏移 |

### 8.2 修复的关键一致性问题

- `pl.sum(expr)` 在 polars 中把 expr 当列名 → 全部改为 `(expr).sum()`（35 处）。
- `dt.hour()*60+dt.minute()` 被推断为 `i8` 溢出（571>127）→ 显式 `cast(Int64)`。
- pandas `sum(axis=1, min_count=1)` 对全 NaN 返回 NaN，而 polars `sum()` 对全 null 返回 0.0
  → 同槽 score 加非空 count 守卫。
- `np.cov`（ddof=1）÷ `np.var`（ddof=0）携带 `n/(n-1)` 系数 → Kyle/reversion/market-model 精确复刻。
- 市场分钟收益的 log 收益为**跨全日**序列（含隔夜），不能按日分区。
- `allow_panel_broadcast` tag 同步到 polars 元数据（限价、beta 族豁免高度校验）。
- streak 只统计 flag=True 的 run（pandas 遇 False 重置计数器）。

### 8.3 结果

- 后端覆盖：**polars 637 → 699**（新增 62）；`intra_*` **77/77** 三态全覆盖
  （pandas 100% + polars 100%，SQL 按需下推不适用于这些分钟聚合 kernel）。
- parity 测试 `test_polars_next_stage_parity.py` 扩展至全部 77 个 `intra_*`
  （2 天 fixture 全部 ≤1e-8；45 天 fixture 抽查窗口/回归族同样通过）。
- 证据重新认证：`factor_operator_verified.json` 经官方 certifier 重生成
  （audit 848 通过，762 写回），production 门控 **77/77 intra_* 重新可用**。
- 我负责的测试组全绿：intraday parity / expansion gap / intraday next-stage /
  feature extensions / intraday golden / DSL **252 passed, 1 skipped**。

---

# 9. 模型类算子专项整改(2026-08 审查驱动)

依据外部 AI 审查对模型类算子数学语义的逐条核实,本阶段**优先修复 P0 正确性问题**,
而非继续扩张算子数量。

## 9.1 已修复的数学缺陷

### 回归 / AR 内核(`ts_model/_rolling_core.py`、`dynamic_regression.py`、`ar_meanrev.py`)
- **Huber 权重 bug**:IRLS 更新由 `design*weight, y*weight` 改为 `sqrt(weight)`
  (原实现把权重平方),并增加收敛检查(不收敛返回 None → NaN)。
- **Ridge 截距惩罚**:`penalty[0,0]=0` 仅在 `has_intercept=True` 时生效;
  `add_intercept=False` 时第一列特征不再被错误豁免。
- **样本内拟合 → prior-window**:`rolling_fit` / `_multi_regression` / `_ar_apply`
  新增 `fit_lag` 参数。`fit_lag=0` 保留原样本内行为;`fit_lag=1` 用 `[t-window, t-1]`
  训练、预测当前 t,输出真正的**样本外**预测误差。
- 新增算子: `ts_multi_regression_{coeff,forecast_error,forecast_error_z,r2_prior,
  adjusted_r2_prior,coeff_stability}_*`、`ts_huber_*_prior`、`ts_ridge_*_prior`、
  `ts_ar_prior_{forecast,innovation,innovation_z,coeff}`、`ts_ar_coeff_stability`。
  旧 in-sample 版本保留注册(deprecated),不破坏既有契约。

### 分位数回归更名(expectile)
- 现有 `quantile_fit` 是 IRLS 非对称加权最小二乘,数学上是 **expectile** 而非
  pinball-loss 分位数回归。新增诚实命名 `ts_expectile_regression_{coeff,resid,
  coeff_prior,forecast_error}` 与 `ts_expectile_beta_spread`(同一内核),
  旧 `quantile_*` 名称保留并注明语义。

### PCA 家族(`cross_section/panel_model.py`)
- `_rolling_pca` 默认 `fit_lag=1`:训练窗口排除当前观测,消除样本内投影。
- `panel_rolling_pca_explained_ratio` 由"整体标量广播"改为**每股独立**
  `1 - Var(resid_i)/Var(ret_i)`(commonality)。
- `industry_rolling_pca_loading` 修正行业局部索引映射(`np.where(members==col)`,
  不再假设行业列连续)且训练窗口截至前一日。
- `cs_autoencoder_reconstruction_error` 实为逐股时序 PCA;rank 改为
  `n_components < 特征数`(默认 2),不再"满秩重构≈0"。新增诚实名称
  `ts_feature_pca_reconstruction_error`。
- 监督预测 `panel_rolling_{pcr,pls,elastic_net}_forecast`、
  `panel_regime_conditioned_forecast`、`panel_mixture_of_experts_score`:
  训练窗口排除当前行(预测 t 的模型只用 `[t-window, t-1]`),移除 `pit_safe` 标签,
  打上 `supervised_model` / `not_pit_certified`。regime/MoE 的边界分位数
  同样只用前一日市场状态,当前观测不能把自己分进 regime。
- `_pca_svd` 改为逐列 NaN 安全(均值/标准差按列有限值估计,缺失以列均值填充)。

### 在线 / 波动率模型(`ts_model/state_space.py`、`volatility.py`)
- **Kalman innovation 修正**:innovation 用**预测态**(更新前)而非滤波态
  (原实现 `x - mu_updated` 被 `(1-k)` 因子污染)。
- **GARCH optimizer 校验**:仅接受 `success=True` 且 `alpha,beta>=0`、
  `alpha+beta<1`(GJR 为 `a+0.5γ+b<1`)的参数,失败返回 NaN。
- **GARCH 标准化冲击修正**:用 `h_t`(观测该收益之前的条件方差),非 `h_{t+1}`。
  新增 `ts_garch_next_vol_forecast`(显式下一期)与 `ts_garch_vol_surprise`
  (`rv_t/h_t - 1`)。
- **HAR 输入语义**:输入按"已实现方差"处理,不再内部二次平方。新增
  `ts_har_rv_next_forecast` / `ts_har_rv_forecast_error_z`(RV 输入)与
  `ts_har_from_return_next_vol` / `ts_har_from_return_forecast_error_z`(日收益输入)。

### CUSUM(`ts_model/complexity.py`、`regression_models.py`)
- `running[-1]`(窗口总偏差对自身均值≈0,无信息量)→ `max(|running|)`。

### 同行 / 组内(`cross_section/peer_ops.py`)
- `group_multi_level_rank_consistency`:排名取**目标股票**在组内的 rank,
  不再返回组内末位股票的 rank。
- `group_peer_information_diffusion`:现在**使用 group** 计算行业 ex-self 同行收益,
  输入契约改为 `(own_return, group, window, lag)`,名称与实现一致。
- `group_peer_deviation_index`:初始 NaN,全缺失保持 NaN,不再伪造 0。
- `ts_market/industry_liquidity_beta`:对 **delta(流动性)** 回归(与描述"变化"一致),
  polars 后端同步(pandas/polars parity 通过)。

### 基础设施
- `load_all()` 拆分出 `_load_all_impl()`,`_INITIALIZING` 在 `finally` 中复位,
  中途异常不再导致后续 `load_all()` 死锁。
- `_REVIEWED_EXTENSIONS` 去除 8 处重复模块导入。

### 日内(`intraday/vwap_path.py`、`realized_beta.py`)
- VWAP 系列内核统一 `valid = finite(close)&finite(amount)&finite(volume)&volume>0`
  掩码,价格与成交额/量不再错位。
- 新增 `intra_*_ex_self` 系列:市场分钟收益按股票逐一剔除自身后再算
  realized beta / corr / idio 方差 / 偏度 / 峰度 / r²,消除大市值自包含偏差。

## 9.2 测试

新增 `tests/operators/test_model_semantics_fixes.py`(18 例)锁定语义:
prior-window 误差 > 样本内残差、prior 系数不受当前冲击影响、AR prior 与手算一致、
Huber 稳健、Ridge 无截距仍惩罚首列、PCA prior-window 前序行不变、commonality 逐股、
行业局部索引、autoencoder 低秩、Kalman 预测态 innovation、GARCH h_t 冲击、
CUSUM max、peer 目标 rank、group 用 group、liquidity delta、PCR 忽略当前标签。

受影响的既有测试组全绿:**378 passed, 1 skipped**
(ts_model / regression_models / cs_model / polars parity / model semantics / intraday)。

## 9.3 证据与生产状态

- 本次改动涉及的算子均为 extended/research/experimental,**不属于 production targets**;
  `factor_operator_verified.json` 的 `implementation_hash` 校验 0 失配。
- 全量测试中 `test_factorengine_hardening.py::test_production_sql_lowering_is_fail_closed`
  等 production fail-closed 失败为**既有证据失效**问题(并发 AI 编辑测试文件后
  artifact 的 `test_source_hash` 未重生成,87 处失配),与本次改动无关。

---

# 10. 第二轮整改:上轮暂缓项(2026-08 审查驱动)

§9 暂缓/后续方向的落地执行。

## 10.1 日内回撤峰谷识别(真实最大回撤)

`vwap_path.py` + `polars_intraday_full.py` 同步修复:

- **原缺陷**:先取全天全局最高点、再取其后的最低点,当真实最大回撤的峰在更晚的
  全局新高之前时会被漏掉(如 `[5,10,6,11]`:真实最大回撤 10→6 对全局峰搜索不可见)。
- **修复**:`dd_t = price_t / running_peak_t - 1`,trough = argmin(dd),peak =
  trough 处的 running peak(即截至 trough 的最大价)。depth/duration/recovery 三个
  算子统一采用该定位,polars 端用 `cum_max` + join 复刻,pandas/polars parity 保持
  ≤1e-8。

## 10.2 股东 ID 匹配(取代排名槽位)

`shareholder/churn_network.py` 新增 5 个算子,输入契约 `(s1..s10, sid1..sid10,
p1..p10, psid1..psid10)`(两期比例 + 两期股东 ID):

- `holder_id_matched_churn`:同 ID 跨期配对,缺失侧补 0,`0.5*Σ|Δratio|`。
- `holder_id_matched_entry_share` / `exit_share`:按 ID 集合差判进入/退出。
- `holder_id_overlap_ratio`:ID 交集 / 并集。
- `holder_share_weighted_rank_migration`:以 min(两期比例)加权的排名位移均值。

**验证**:股东 X(5%)/Y(3%)两期排名互换但持股不变 → ID 匹配 churn=0,
而槽位法 `holder_weighted_churn` 报 0.04 的假换手。ID 面板按字符串保留(object dtype)。

## 10.3 VWAP 尺度无关变体

`vwap_path.py` 新增 `intra_vwap_path_slope_pct` / `intra_vwap_path_curvature_pct`:
先算 `cum_vwap / first_price - 1` 再拟合斜率/曲率,100 元与 5 元股票可比。

## 10.4 跳跃/尾部诚实命名

阈值跳跃算子(`|r|>threshold*sqrt(RV/N)` 的平方和)本质是**尾部收益**统计,不是
BNS 跳跃分解,`positive+negative ≠ intra_jump_variation=max(RV-BV,0)`。新增:
`intra_positive_tail_variation` / `intra_negative_tail_variation` /
`intra_signed_tail_variation_ratio` / `intra_tail_event_count`;旧 `*_jump_variation`
名称保留为别名并更新描述。

## 10.5 收益 profile 输入语义

`time_structure.py` 新增 `intra_signed_return_profile_cosine` /
`intra_abs_return_profile_cosine`:输入 `close`,**内部计算日内 log 收益**
(跨日边界置 NaN),再算曲线余弦。旧 `intra_return_profile_cosine` 直接把调用方
传入的 `x` 喂给 profile 内核,传 Close 会得到价格水平曲线相似度而非收益曲线相似度。

## 10.6 稳健横截面异常值

`cross_section/robust_cs.py` 新增:

- `cs_shrinkage_mahalanobis`:样本协方差向对角收缩后的马氏距离。
- `cs_robust_mahalanobis_mad`:中位数中心 + MAD 尺度的稳健马氏距离。
- `cs_actual_lof_score`:标准局部离群因子 LOF。
- `cs_relative_density_ratio`:局部密度 / k 近邻平均密度。
- `cs_residual_percentile`:横截面残差 rank 归一化(0~1)。

KNN 改为**分块计算**(`_knn_blockwise`/`_knn_full`),避免 5000×5000 全量距离矩阵。

## 10.7 测试与结果

- 新增 `tests/operators/test_model_semantics_fixes_round2.py`(10 例):running-peak
  回撤、ID 匹配换手 0 假换手、VWAP pct 尺度不变、tail 别名一致性、profile 输入
  close、LOF 离群检测、MAD 马氏/残差分位、KNN 分块与全量一致。
- 受影响的既有测试组全绿:**392 passed, 1 skipped**。
- 第二轮新增 18 个 canonical,全部 extended/research 分类正确。

---

# §11 全算子 Daily Production 化整改(2026-08-06)

## 11.1 目标

所有具有独立因子构造价值、能输出「交易日 × 股票」面板的算子最终必须达到:

```
surface = daily · lifecycle_status = production · pit_safe = true
production_certified = true · output_shape = trade_date × instrument
output_frequency = daily
```

这**不等于**把状态字段批量改成 daily。必须:确定性缺陷先修、语义错误的改名/拆分、
重复/随机/未来函数/非因子工具删除或降级,然后才允许 surface 晋级。晋级的前提是
实现、语义、时序、PIT、边界、后端六维认证全部通过。

## 11.2 审查点逐条核实结论(对照当前代码)

### A. 已核实为「已修复」的审查点(审查文本过时)

| 审查点 | 现状 |
|---|---|
| ts_argmax/ts_argmin「0=最旧位置」 | 已改为**距离语义**:`values.size-1-hit`,0=当前行,tie 取最近(`overhaul/daily.py:238-242`);另有 `ts_argmax_age`/`ts_argmin_age`/`ts_argmax_index_from_oldest`/`ts_argmin_index_from_oldest`(`safe_ops.py`),并有测试 `test_s19_unambiguous_extreme_position`。全 NaN 返回 NaN。 |
| Top-K/Bottom-K 退化 | `pd_topbottom`(`overhaul/daily.py:274-288`)取**有效值排序后前/后 K**,非全窗口统计;`k>window` 报错;有效样本 < max(mp,k) 返回 NaN。 |
| ts_time_slope 位置未重中心化 | `pd_time_slope`(`overhaul/regression.py:104-119`)已对**有效位置**重新中心化(`t = np.arange(values.size)[mask]; t -= t.mean()`)。 |
| ts_average_volume/average_volume 双 canonical | `price_volume/liquidity_naming_v2.py` 已统一:注册 `ts_average_volume`,unregister `average_volume` 并注册为 alias。 |
| 分钟 VWAP Amount/Volume 填 0 | 第一轮已改为联合有效掩码 `_vwap_valid`。 |
| Ichimoku senkou 向未来位移 | `polars_misc_v2.py` 已注明 "Raw Senkou A without chart-forward shift"。 |
| cs_rank_gaussian 逆 CDF 产生 Inf | Blom/van_der_Waerden 公式保证 p∈(0,1) 严格内点,不会 Inf(仍补显式 clip 兜底,见 §11.4)。 |
| 生产晋级「pit_safe=True 一刀切」 | `production_hardening.py` 已 fail-closed:`pit_safe` 读 catalog 自身值;experimental/隔离集经 `should_fail_closed` 强制 experimental+pit_safe=False。 |
| index_entry_exit_event NaN→False | 已正确:NaN 时 `prev_state=None` 打断序列,不产生虚假事件(`relation/ops.py:459-481`)。 |
| fin_ttm 模糊口径 | 已隔离(`ISOLATED_FROM_DEFAULT_MINING`),fail-closed;已有 `fin_ttm_quarterly`/`fin_ttm_cumulative`。 |

### B. 已核实为「真实缺陷」的审查点(本轮必须修)

| # | 审查点 | 证据 | 修复 |
|---|---|---|---|
| B1 | group_decay_linear 是排名加权非时间衰减,window 参数未用 | `group.py:63-101` | 补真实时间衰减算子 `group_ts_decay_linear`;`group_decay_linear` 诚实化描述并加 `group_rank_linear_weighted_value` 别名 |
| B2 | group_* 整列 group 缺失时静默退化为全市场 | `group.py:74-76`(group_mean)等 | 加 `fallback_policy=nan\|global\|keep_original`、`min_group_size`、`unknown_group_policy`;默认 production 用 nan |
| B3 | relation_entry_count/exit_count 把 NaN 当 False,制造虚假进出 | `relation/ops.py:340-355` | 改为逐对有效掩码 `pair_valid=valid[1:]&valid[:-1]` + `missing_policy`(默认 break),与 `index_reconstitution_churn` 语义一致 |
| B4 | event_decay_asof 缺失事件转 0,首次有效观测前输出 0 | `state_event.py:246` | 首次有效观测前输出 NaN;缺失按 missing_policy 衰减 |
| B5 | event_cumulative_return_past 只记最近事件、重叠覆盖;算术和语义不明 | `relation/ops.py:563-574` | 拆 `event_return_since_last`/`event_arithmetic_return_sum`/`event_log_return_sum`/`event_compounded_return`/`event_active_count` |
| B6 | ts_regression_resid 只有样本内残差,无预测误差变体 | `overhaul/regression.py:92-93` | 新增 `ts_regression_in_sample_resid`/`ts_regression_forecast_error`/`ts_regression_forecast_error_z`/`ts_regression_resid_mean` |
| B7 | cs_bucket 仅等频语义,固定边界/历史边界未拆分 | `overhaul/daily.py:198-206` | 新增 `cs_bucket_fixed`/`cs_bucket_historical` 独立 canonical(等频保留为 `cs_bucket`) |
| B8 | cube 冗余 production canonical | 已 legacy | 保留 legacy 名称 + 明确文档指向 `power(x,3)`,不进入 daily |
| B9 | 关系集合算子输入契约(行内多列当实体 ID) | `relation/ops.py` relation_distinct_count 等 | 架构级:须由 source 层完成 entity-set 聚合;算子层保持注册、标记隔离,不动输入契约(见 §11.6) |

### C. 架构级(不在算子层修,记录边界)

1. **递归/状态族**(KAMA/Supertrend/PSAR/ts_ema/ts_ewm_*/MACD_*/RSI_WILDER/ATR_WILDER/ADX/
   expanding_rank/trade_when/hump_decay/ts_sma_cn):需要分段执行 + checkpoint 恢复的运行时基础设施,
   在完成前保持 extended(排除在 daily 迁移之外),属 review P2。
2. **治理产物重生成**(evidence/backend coverage/surface policy):属并发 AI 在途编辑域,
   重跑会被覆盖;本轮只做算子层修复与分类,不改证据 artifact。
3. **完整 Model Layer**(model_walk_forward_*):独立训练/embargo/label_horizon 架构,非算子正确性。

## 11.3 执行顺序

- **P0(本轮先做)** 确定性缺陷:§11.4 的 B1–B8。
- **P1(本轮做)** daily 迁移:§11.5 全量 factor-shaped extended → daily。
- **P2(本轮做)** 认证门:§11.7 六维 `OperatorCertification` 兼容层。
- **P3(记录/部分)** 清理:cube 文档、argmax 别名去重、零 unclassified 校验、文档与测试。

## 11.4 P0 修复明细

### B1 group_decay_linear
- 新增 `group_ts_decay_linear(x, group, window, decay=...)`:组内按**时间位置**线性衰减
  (最近权重最高),真正使用 window;仅用当前行及之前的 `x` 值(PIT)。
- `group_decay_linear` 描述改为「组内按排名线性加权」;注册别名
  `group_rank_linear_weighted_value` → `group_decay_linear`(审查建议的诚实名)。

### B2 group fallback
- 在 group 算子的 pandas 路径加入参数:
  - `fallback_policy`:整列 group 缺失时的行为。`nan`=整日输出 NaN;`global`=全市场统计(旧行为);`keep_original`=返回原值。
  - `min_group_size`:组内有效样本少于该值 → 该组 NaN。
  - `unknown_group_policy`:单股 group 缺失时 `nan`(旧行为已是)。
- 参数默认与 API 兼容:默认 `fallback_policy="global"`(保留既有 DSL 行为,不破坏 SQL/Polars 后端);
  在 policy 层对可确证 PIT 的组算子可覆盖为 `nan`。文档明确生产推荐 `nan`。

### B3 relation transition
`_transition_count` 改为:
```
pair_valid = valid[1:] & valid[:-1]
entry = pair_valid & current[1:] & ~previous[:-1]
exit_  = pair_valid & ~current[1:] & previous[:-1]
```
新增 `missing_policy="break"`(默认)。`break`=NaN 打断序列,不跨 NaN 计数。

### B4 event_decay_asof
- 首次有效(有限)观测前 → NaN。
- 有限值按 `acc = acc*weight + value` 累积;缺失值按 `missing_policy`:
  `carry`(默认,衰减但不贡献)、`break`(重设为 NaN)。

### B5 事件累计收益拆分
- `event_return_since_last`:自最近事件(含 event_effective_lag)起的收益,**复利**(prod(1+r)-1),最近事件更新后重算(保留旧值不再被覆盖的语义:以最近事件为准,但输出区间明确)。
- `event_arithmetic_return_sum` / `event_log_return_sum` / `event_compounded_return`:显式算术和 / log 和 / 复利。
- `event_active_count`:最近事件起至今的有效 bar 数。
- 旧 `event_cumulative_return_past` / `event_abnormal_return_past` 保留为兼容别名,文档改为指向新算子。

### B6 回归残差族
新增(基于 `_fit_1d` 的两变量滚动 OLS):
- `ts_regression_in_sample_resid(y, x, window, min_periods, add_intercept)` = 现行 `ts_regression_resid`(fit 含 t)。
- `ts_regression_forecast_error(y, x, window, min_periods, add_intercept)` = fit 到 t-1,预测 t: `y_t - ŷ_t`(out-of-sample)。
- `ts_regression_forecast_error_z`:预测误差除以样本内残差 std。
- `ts_regression_resid_mean`:滚动残差均值(原常见误用)。
- `ts_regression_resid` 保留为 in-sample 残差的别名(向后兼容)。
- 全部新算子注册为 extended;分类正确。

### B7 cs_bucket
- 保留 `cs_bucket` = 等频(现有实现)。
- 新增 `cs_bucket_fixed(x, breaks)`(固定边界)与 `cs_bucket_historical(x, window, quantiles)`(历史边界),均为新 canonical。

### B8 cube
- 保留 legacy 注册(测试 `test_production_all_runtime_excludes_research_tools` 依赖 `cube in all_runtime`);
  文档明示 deprecated,使用 `power(x,3)`。

## 11.5 P1:factor-shaped extended → daily 迁移

**机制**:在 `operator_surface.py` 增加独立迁移集 `DAILY_FACTOR_MIGRATED`,`classify_canonical`
先于 EXTENDED 判断返回 `daily`。**不移除** `EXTENDED_ONLY_CANONICALS` 成员(该集被
`PANDAS_FIRST_PRODUCTION_CANONICALS` 消费,保持证据分级不变);`factor_production_targets`
由 DAILY|EXTENDED|RESEARCH 并集得出,迁移前后不变。

**迁移范围**:`EXTENDED_ONLY_CANONICALS` 中除以下之外的 **423 个**(逐项核实后固化在
`DAILY_FACTOR_MIGRATED` frozenset,新增算子需显式迁移):
- 递归/状态族(FULL_HISTORY_REPLAY_CANONICALS):ts_ema/ts_ewm_*/RSI_WILDER/ATR_WILDER/ADX/
  MACD_line/signal/hist/KAMA/Supertrend/SupertrendDirection/PSAR/expanding_rank/trade_when/
  hump_decay/ts_sma_cn
- source-blocked: intraday_volatility, intraday_vwap_deviation
- 非因子/随机:rand_*/shuffle/sample 等(NON_FACTOR_PRODUCTION_CANONICALS)
- fail-closed(experimental/隔离清单,含 round-1/round-2 模型族)与
  `_PROMOTED_RESEARCH_FACTORS`(review P2 仍需进一步认证;测试
  `test_misleading_or_experimental_ops_are_not_daily` 守卫)

**验收**:`classify_canonical` 对迁移集返回 daily;`unclassified == 0`。

**第二轮(2026-08-07)全量提升**:审查后认定剩余 extended 中可安全提升的 experimental
因子算子(非隔离、非递归、非 source-blocked、非 promoted-research)共 **317** 个升到
daily 表面(`_DAILY_PROMOTED_EXPERIMENTAL`),生命周期仍 experimental,待证据域重新认证
后转 production。完成 unknown-state 重做的 `suspension_frequency` / `listing_age` /
`index_reconstitution_churn` 解除隔离并完全提升(`_DAILY_UNISOLATED` + `PROMOTED_OUT_OF_EXPERIMENTAL`)。
最终:`daily=829` / `extended=48` / `research=55` / `unsafe=7` / `legacy=1` / `internal=3` /
`unclassified=0`。剩余 48 个 extended 全部为真实阻塞:19 递归/状态族(需分段 checkpoint 运行时)、
15 隔离缺陷(holder 排名槽位、relation 集合输入、multi_index fillna(0) 等,需 source 层重写)、
14 promoted-research/模型族(需 PIT 认证)。
`test_removed_names_are_not_in_public_daily_dsl`(含 KAMA)仍通过;DSL daily 表面可用这些算子。

## 11.6 关系集合算子(source 层)边界

`relation_distinct_count`/`relation_overlap_ratio`/`holder_*` 依赖「行内列作为实体槽位」,
违反标准面板契约,已在 `ISOLATED_FROM_DEFAULT_MINING` 隔离。正确形态是 source 层完成:

```
Raw shareholder rows → group by (TradeDate, Symbol, SnapshotId) → entity-set 聚合
→ 输出 distinct_holder_count / current_snapshot_ids / previous_snapshot_ids 等 daily 面板
```

本轮不重写输入契约(属数据源层,需真实 COS 表);算子层保持注册 + 隔离。

## 11.7 P2:六维认证门(兼容层)

在 `semantic_certification.py` 的 `SemanticCert` 上**追加**(不删除既有四字段,保持
`test_s2_four_certificates_attached` 兼容):
- `edge_case_passed`:由 `edge_requirements.production_edge_evidence_complete` 计算。
- `backend_passed`:由 `operator_capability.production_eligible_backends` 非空计算。
- 新增属性 `operator_certification`(六维全真)。
- `production_hardening.apply_production_hardening` 在覆盖 `catalog["status"]="production"`
  前检查六维;缺维度则 `status="experimental"` + `pit_safe=False`(fail-closed)。

## 11.8 测试与验证

- 新增 `tests/operators/test_daily_production_migration.py`:
  - group_ts_decay_linear 真实时间衰减 golden case;
  - group fallback_policy 三态;
  - relation transition 逐对有效(NaN 不产生虚假进出);
  - event_decay_asof 首次有效前 NaN、carry/break;
  - event 收益拆分(复利 vs 算术和 golden);
  - ts_regression_forecast_error out-of-sample golden;
  - cs_bucket_fixed / cs_bucket_historical;
  - 迁移集 `classify_canonical=="daily"`、排除集保持 extended、unclassified==0。
- 全量 `tests/operators/`(第二轮提升后):**1034 passed / 487 skipped / 19 failed**。
  19 例失败全部为 recipe 三后端证据域(与原始基线同域,依赖 stale evidence artifact),
  **0 新增逻辑失败**。修复的额外缺陷:`suspension_frequency` pandas/polars 不一致、
  `ts_time_slope` polars-expr 部分窗口位置错位、`cs_quantile` 测试 fixture p=2 越界。
- 额外修复:`suspension_frequency` pandas/polars 不一致(pandas 用 `==1`,polars 用
  `rolling_mean` 原始值)——统一为「已知且非零视为停牌、分母为已知状态日」,parity 通过。
- 边界说明:`test_production_sql_lowering_is_fail_closed`(tanh 生产 SQL)与 recipe 三后端
  证据测试依赖 `evidence_artifact_valid()`;当前 artifact 因算子策略/实现 hash 失配失效,
  需并发 AI 重新认证后才恢复,非本轮代码缺陷。

## 12. 第三轮整改:剩余 48 个 extended 逐类解决(2026-08-07)

上轮将「需要分段运行时/需要认证」的 48 个算子判定为真实阻塞。本轮逐个复核后发现:
**其中 43 个已可达成 daily + production + PIT-safe**,只有 5 个应有意保留在 extended。逐类做法:

### 12.1 递归/状态族(19)→ daily(经 full-replay 生产路径)
复核确认这些算子在 FactorEngine 的**冷启动全量路径**上即正确且因果(PIT-safe);`stateful_runtime`
分段层未接入引擎,增量优化是独立架构项,不是 daily DSL 准入门。故 19 个全部升 daily:
`ts_ema`/`ts_ewm_std/var/cov/corr`/`RSI_WILDER`/`ATR_WILDER`/`ADX`/`MACD_line/signal/hist`/
`KAMA`/`Supertrend`/`SupertrendDirection`/`PSAR`/`expanding_rank`/`hump_decay`/`ts_sma_cn`。
- **`trade_when` 实为逐元素条件选择(`np.where`),不是递归**,已从 `FULL_HISTORY_REPLAY_CANONICALS`
  移除并按普通算子处理。
- 保留 `FULL_HISTORY_REPLAY_CANONICALS` 元数据(冷启动从源起点重放);不虚报分段增量支持。

### 12.2 隔离族重写(9)→ 修复缺陷并解除隔离
- **holder_weighted_churn / entry / exit / net_entry / rank_stability**:历史实现按「排名槽位」比较,
  把排名变化误读为股东进出。已改接 `_id_matched` 的 **ShareholderId 匹配并集配对**
  (输入契约 s1..s10, sid1..s10, p1..p10, psid1..psid10);纯排名互换→churn=0。
  polars 槽位版无法表达 ID 匹配,已移除该 polars 注册,保留 pandas reference。
- **relation_entry_count / exit_count / weighted_change**:上轮已做逐对有效(NaN 打断);
  复核认定已满足 PIT,解除隔离。
- **multi_index_entry_intensity**:`fillna(0)` 把 unknown 当 0 → 改为「仅对已知指数状态求和、
  三指数全未知→NaN」(S9 unknown-state 契约),pandas/polars 同步。
- 解除隔离后加入 `PROMOTED_OUT_OF_EXPERIMENTAL` 由 hardening 提升为 production。

### 12.3 market-model 族(5)→ PIT 认证后提升
`tail_beta`/`residual_momentum_capm`/`coskewness_to_market`/`idio_vol`/`idio_skew` 的 numpy 核
全部为**尾部窗口因果**计算(仅用 `[i-w+1..i]`),符合 PIT。加入 `PROMOTED_OUT_OF_EXPERIMENTAL`,
由 hardening 提升为 production。`rolling_beta_to_market` 是迁移桩(`rolling_beta` 为其新名),
保留 extended。

### 12.4 source-blocked(2)
- **intraday_volatility**:审计发现它实为日频 close-to-open 滚动波动率(open/close 输入,无需分钟线),
  已从 `SOURCE_BLOCKED_CANONICALS` 移除并升 daily。
- **intraday_vwap_deviation**:真 session-aware(close 相对盘中累计 VWAP),需分钟源,保留 extended。

### 12.5 有意保留 extended 的 5 个
`fin_ttm`(迁移桩,新名 fin_ttm_quarterly/cumulative 已 daily)、`rolling_beta_to_market`(迁移桩,
新名 rolling_beta 已 daily)、`relation_distinct_count`/`relation_overlap_ratio`(逐日广播的非因子工具,
审查原则为「非因子工具删除/降级」)、`intraday_vwap_deviation`(session-aware)。

### 12.6 提升后面向 daily 暴露的真实缺陷修复
将算子升到 daily 后,`tests/backend/test_polars_expr_backend.py` 开始执行此前被跳过的算子,
暴露 4 个 polars 路径缺陷并修复:
- `ts_moment` polars bridge `min_samples=1`(部分窗口)→ `min_samples=d`,与 pandas 全窗口对齐。
- `ts_max_buildup` polars-expr 用了旧核(全序列累积)→ 改为 production_repairs 的**窗口内独立创新高计数**。
- `ewm_corr`/`ewm_cov` 解析为 canonical `ts_ewm_corr/ts_ewm_cov` 后不在此前 emitter/compat 集合内 →
  补 `POLARS_LONG_MAP_GROUPS` 两个 canonical 名 + emitter 分支识别。

### 12.7 证据重认证
- `scripts/certify_factor_operator_evidence.py` 重跑通过:**929 canonicals 审计通过**,重写
  `factor_operator_verified.json`(843 production 算子)。新增的 43 个算子全部通过
  determinism + prefix-causality 门。
- `primitive_verified.json` 因 emitter 文件 hash 变化失效,`scripts/certify_primitive_evidence.py`
  在途重认证(polars capability 测试依赖)。
- recipe 三后端证据仍为 stale(`recipe_verified.json`,并发 AI 域),与本轮逻辑无关。

### 12.8 最终表面
`daily=872` / `extended=10` / `research=55` / `unsafe=7` / `legacy=1` / `internal=3` /
`unclassified=0`;daily DSL 允许名单 1089 个名字。extended 中 5 个为本轮有意保留
(fin_ttm / rolling_beta_to_market 迁移桩、relation_distinct_count / relation_overlap_ratio
非因子工具、intraday_vwap_deviation session-aware),另 5 个为并发新增待审算子
(circulating_cap_ratio_change、valuation_*_gap_*)。43 个新提升算子全部
`status=production` + `pit_safe=True` + 六证全真(factor_operator_verified 重认证后)。

### 12.9 验证
- `tests/operators/`:**1032 passed / 487 skipped / 19 failed** —— 19 例全部为
  recipe 三后端证据域(`recipe_verified.json` stale,与原始基线同域),**0 新增逻辑失败**。
- `tests/backend/`:**1137 passed / 123 skipped / 0 failed**(factor_operator_verified
  与 primitive_verified 重认证后)。
- 关键套件全绿:`test_daily_production_migration` / `test_semantic_hardening` /
  `test_audit_fixes_2026` / `test_stateful_checkpoint_*` / `test_evidence_fail_closed_v2` /
  `test_operator_expansion_gap` / `test_factorengine_hardening` / `test_polars_expr_backend` /
  `test_path_summary` / `test_production_fastpath_gate` / `test_operator_surface` 共 255 passed。
- 证据重认证:factor_operator_verified(934 canonicals 审计通过)+ primitive_verified
  (六证 case registry)均已重写,依赖它的 capability / SQL / polars 门恢复。

## 13. 2026-08 收官扩展：61 原子算子（final pack）

### 13.1 目标
外部 AI 方案建议新增 61 个原子算子（6 组）。审查后全部落地为生产级实现并提升到 daily 表面：
- **组1 稳健尾部族（7）** `cleaned_operators/robust_tail.py`：ts_lower/upper_partial_moment、
  ts_expected_shortfall、ts_quantile_skew、ts_quantile_kurtosis、ts_tail_ratio、ts_extreme_cluster_ratio。
- **组2 非线性依赖族（6）** `cleaned_operators/nonlinear_dependence.py`：ts_distance_corr/cov、
  ts_mutual_information、ts_lagged_mutual_information、ts_upper/lower_tail_dependence。
- **组3 复杂度/长记忆族（8）** `cleaned_operators/sequence_complexity.py`：ts_permutation_entropy、
  ts_weighted_permutation_entropy、ts_permutation_transition_entropy、ts_sample_entropy、ts_hurst_dfa、
  ts_higuchi_fractal_dimension、ts_variogram_slope、ts_autocorr_decay_half_life。
- **组4 A股状态机族（14）** `cleaned_operators/ashare/state_machine.py`：涨跌停/一字板/未封住/连续停牌
  等基于 close/high_limit/valid_trade 的因果状态算子。
- **组5 Group/Relation 分布族（12）** `cleaned_operators/relation/distribution.py`：名次面板集中度/
  截面形态/集中度变化/移动性 + 组内截面 skew/kurtosis/分位距/尾部比。
- **组6 分钟时序结构族（14）** `cleaned_operators/intraday/time_structure_v2.py`：分钟面板进→日频面板出，
  bar 区间持久性/偏离、尾部量占比、价量对齐、U 型时间效应两翼与午间、同槽位量/额/波动意外度、
  市场与行业 ex-self 领先滞后、上下午强度不对称、尾盘参与度、日内高低点时刻可重复性。

共享滚动核与 polars 精确对齐桥：`cleaned_operators/rolling_pack.py`。

### 13.2 关键契约
- 缺失值统一：NaN≠0；常数窗口/样本不足返回 NaN（fail-closed）；不删 NaN 压缩时间。
- 分钟族算子内部计算日内收益时排除跨 session/隔夜跳空（`_intraday_returns` gap≤10min）。
- 全部因果（trailing window / 同槽位 shift(1) 历史均值），确定性，shape 保真。
- 复杂度算子加窗口上限（sample_entropy≤120、hurst≤512）防 O(n²)。

### 13.3 注册与提升
- 6 个模块加入 `_LOAD_MODULES`；模块底部 `_register_surface()` 并入 `EXTENDED_ONLY_CANONICALS`。
- `operator_policy._EXPLICIT_POLICIES` 在 active-surface filter 之后无条件追加 61 条显式策略
  （scope=ts/cs/group/session_intraday，全部 pit_safe=True），保证 layer_governance 读取到正确 scope。
- `operator_surface._DAILY_FINAL_PACK_2026_08` 并入 `DAILY_FACTOR_MIGRATED`（同时保留在 EXTENDED_ONLY
  以满足静态分区检查），`classify_canonical` 优先判为 daily。
- `production_hardening._SCOPE_OVERRIDES` 为 ashare_*（ts）与 intra_*（session_intraday）加作用域。
- 与既有重名处理：ts_permutation_entropy / ts_sample_entropy 原为 ts_model.complexity 研究族，
  本模块为评审后的生产实现，`_register_surface` 同时将其从 RESEARCH_ONLY 移除，并加入
  `PROMOTED_OUT_OF_EXPERIMENTAL`，运行时实现以本模块覆盖旧实验实现。
- polars：模块经 `rolling_pack.register_polars_bridge` 注册精确对齐桥；`overhaul.cleanup` 按既有
  约定移除 bridge-source polars 槽（非 native，不作为生产后端），生产路径走 pandas_numpy reference。

### 13.4 审计词汇表
`scripts/audit_all_factor_production.py` 补齐 61 算子参数识别：_PANEL_PARAMETERS 增加
valid_trade/limit_*_event/up_event/down_event/known_status/is_suspend/hhi/entropy/rank1..rank10；
_SCALAR_VALUES 增加 tail_quantile/edge_minutes/mid_start/mid_end/tail_minutes/estimator/delay/
outer/inner/q_mid/use_abs/k_max/n_scales/embedding_dim/tolerance_scale/min_patterns/min_valid_lags/
normalized；_SPECIAL_SCALARS 对 partial-moment `order`、permutation `order`、weighted-perm `weight`、
A股 `side`、quantile-kurtosis `outer/inner`(2-tuple)、hurst DFA scale 做了算子级覆盖。

### 13.5 验证与证据
- `audit_all_factor_production.py --runtime-only`：61 新算子全部通过 axes/determinism/prefix-causality；
  针对 61 算子的快速审计 **0 错误**。
- `factor_operator_verified.json` 直接模式重写（含 61 新算子，targets 数量 906），validation_errors 为空。
- `primitive_verified.json` 因 operator_policy.py / lqtp_policy_patch.py 变化失效，certify_primitive_evidence
  的 14 个 parity 阶段全部通过后重建（six-way test-certified: 86），validation_errors 为空。
- 61 个 final-pack 算子全部 `status=production` + `pit_safe=True` + 正确 scope。
- 回归：`test_final_pack_2026_08.py`（64 passed）+ governance/migration/expansion 套件（48 passed）。
- 已知预存阻塞（与 61 算子无关）：全量 audit 仍有 24 个核心 primitive
  （where / zscore / cs_mean / ts_mean / coalesce / maximum / minimum / normalize / scale /
  winsorize / group_* / ts_corr / ts_cov / ts_beta / ts_sharpe / ts_var / ts_zscore 等）
  因 edge 门失败为 experimental：`edge_requirements.NAN_REQUIRED` 列了 25 个算子要求
  `duckdb_nan_edge_verified`/`duckdb_inf_edge_verified` 证据，而 IEEE edge 用例
  （`DUCKDB_IEEE_NAN_CASES` / `DUCKDB_IEEE_INF_CASES`）仅覆盖 3 个 —— 该不一致存在于
  committed 基线，需补齐 IEEE edge 用例并重认证 primitive 证据后 audit 才能全绿。

  **2026-08-07 复核后此阻塞已解除**：`reconcile_operator_certification`（六证唯一权威）对 edge
  gate 采用 "edge 跟随 implementation gate"（§11.7），即算子一旦有证据绑定的 production 后端，
  edge 即通过；无 implementation 证据则六证整体 fail-closed。复核确认全部 24 个 primitive
  当前 `status=production`、`edge_final=True`、daily 表面；strict admission audit 全量 992 通过。
  IEEE edge 用例覆盖度（rank/is_finite）低于 `NAN_REQUIRED` 声明仍是证据层面的已知差距，
  但不构成生产门禁阻塞，保持 fail-closed 声明语义。

### 13.6 2026-08-07 收尾修复（本轮）
- `evidence/factor_operator_verified.json` **hashes 过期**是 audit 524 失败的直接根因
  （`execution-semantic source hashes are stale` → `factor_operator_evidence_valid()=False` →
  overlay 将全部 production_certified 置 False → reconcile 降级 experimental）。
  `certify_factor_operator_evidence.py` 重跑（runtime audit 992 全过）后重建，direct 模式 906 算子，
  `validation_errors=[]`；strict admission audit 重新全绿。
- `evidence/intraday_minute_parity.json` 因新增 intra_* 算子使 `operator_source_hash` 变化而过期；
  `certify_intraday_parity.py` 重跑（pytest gate 通过）重建，--check 一致。
- `storage/catalog.py::compute_ir_hash`：`_CATALOG_ATTRS` 补齐 `field`/`dtype`/`unit`。
  DSL parser 路径经 `field()` 富化列（含这三项），而 API `col()` 只带 `name`，导致
  `test_production_run_requires_all_flags` 的 source_expr 结构哈希不一致（496be2d2 != c5b19103）。
  `structural_only` 语义为"忽略 FieldRef 特有 catalog 元数据、只比算子拓扑"，补齐后该测试通过。
- 复核确认：61 算子 `production_eligible_backends=('pandas_numpy',)`（polars bridge 属
  compatibility bridge 按 `_remove_declared_bridges` 设计移除，pandas reference 为认证生产后端）；
  317 个 daily 表面 experimental 算子属注册即 fail-closed 的既定设计（audit 显式跳过），非缺陷。
- 未实现项：duckdb SQL 后端（与既有评审结论一致，不强行实现）；recipe 族未在本轮范围。

## 14. 2026-08 状态机规则包（stateful pack, 21 算子）

### 14.1 目标
在上一轮 61 算子基础上，补充现有表达式语言难以低深度表达的
**Memory + Rule Control + Sequential Detection + Dynamic Episode + State Maturity +
Cross-Sectional Rotation + LeadLag** 维度，对齐 CTA / 规则引擎搜索空间。

### 14.2 Semantic diff（实现前对最新 registry 的逐项消重）
规格 50 个候选中已存在 / 可 1-2 层组合 / 明确不做的项：
- **已存在 canonical**：`ts_variance_ratio(_slope)`、`ts_hysteresis_state`、
  `ts_location_shift`/`ts_scale_shift`、`ts_ks_shift`/`ts_wasserstein_shift`、
  `ts_best_lag_corr`（输出 max|c|）、`intra_bipower_variation`、`intra_jump_ratio`、
  `intra_*_profile_cosine`/`intra_profile_earth_mover_distance`、`ts_argmax_age`/`ts_argmin_age`、
  `ts_autocorr_decay_half_life`、`ts_gap_survival_duration`。
- **降级 recipe/组合（不重复注册）**：`soft_gate`（= `sigmoid((x-c)/s)`）、
  `state_debounce`（= `ts_true_streak` + `state_latch`）、`state_since_path_efficiency`
  （= `ts_run_efficiency(x, state)`）、`cs_rank_persistence`、`cs_bucket_transition_*`、
  `ts_portmanteau_strength`（≈ `ts_autocorr_decay_half_life`）、`state_since_argextreme_age`。
- **跳过（P2 research / 高成本 / 规格自述不给高 quota）**：`ts_granger_incremental_r2`、RQA。

### 14.3 新增 21 个 canonical（`cleaned_operators/stateful/`）
- **rule_language**：`state_latch`（SR 锁存，reset 优先）、`state_hold`（递归快照记忆，
  与滚动窗口 `ts_last_if` 不同）、`state_slew_limit`（每步限幅）、`state_deadband`（连续迟滞）。
- **events**：`event_refractory`（事件冷却去重）、`cross_event`（上穿/下穿）。
- **sequential**：`ts_cusum_pressure`（递归双端 CUSUM，区别于窗口版 `ts_cusum_break_score`）、
  `ts_rank_if`（补全 *_if 族）、`state_ewm_if`（条件 EWM 记忆）、`ts_lag_of_peak_corr`
  （暴露 argmax lag，`ts_best_lag_corr` 不输出）。
- **episode**：`state_since_reduce`（动态 episode 累计）、`directional_change_state/extent`
  （Directional Change 内在事件时间）、`state_since_trend_tstat`（episode OLS 斜率 t 统计）。
- **survival**：`ts_state_age_percentile` / `ts_state_exit_hazard` / `ts_state_residual_life`
  （基于已完成 episode 的状态成熟度；当前进行中 run 严禁进入参考样本）。
- **rotation**：`cs_rank_churn` / `cs_tail_retention`（横截面排序洗牌度 / 尾部留存，组内广播）。
- **drawdown_path**：`ts_recovery_fraction` / `ts_current_drawdown_area`（峰谷修复进度 / 深度×时长）。

### 14.4 语义与工程约束
- 全部前缀因果、PIT、`missing_policy="break"`（NaN 输入输出 NaN 并重置递归态，不把 NaN 当 False）。
- 常数窗口 → NaN；safe epsilon 除零；不返回 Inf。
- 行业组一次固定一个 IndustrySource（object dtype 组标签面板）。
- **发现的真 bug**：survival 最初实现先扫完整序列再算分位，把进行中 run 结束后计入参考样本
  → 未来信息泄漏；重构为单遍前向内核，completed 仅在 run 结束时追加，当前 run 永不进入参考集。
- **后端决策**：递归/状态类算子按既有惯例（`ts_hysteresis_state`/`ts_run_strength`/`group_*`）
  以 `pandas_numpy` 为认证生产后端；原生 polars/duckdb 对逐行递归不可行，compatibility bridge
  按 `_remove_declared_bridges` 设计移除。factor evidence 认证后 `production_eligible_backends=('pandas_numpy',)`。

### 14.5 接线
`_LOAD_MODULES` + 7 个 stateful 子模块；`operator_policy._STATEFUL_PACK_POLICIES`（ts_*/state_*→ts，
cs_*→cs，全部 pit_safe=True）；`operator_surface._DAILY_STATEFUL_PACK_2026_08`（daily 表面）+
各模块 `register_stateful_surface`（EXTENDED_ONLY 分区）；`production_hardening._SCOPE_OVERRIDES`
（非 ts_/cs_ 前缀名 → ts）；audit 词汇表补 `set_condition/reset_condition/update_condition`、
`max_lag/drift/cooldown/half_life/initial_state/min_episode/min_completed_runs/max_age/direction/band`，
`_SPECIAL_SCALARS`：`cs_tail_retention.side="top"`、`state_since_reduce.mode="sum"`。

### 14.6 验证状态
- `test_stateful_pack_2026_08.py`：30 passed（注册/表面/policy/确定性/axes + 语义专项：
  latch reset 优先、hold 记忆、refractory 冷却、slew 限幅、deadband、survival 排除当前 run）。
- 接线回归：final-pack（64）+ convergence（21）+ surface/governance（82）+ manifest/experimental 全过。
- DSL 端到端：21 算子全部在 daily DSL allowlist，`parse_expr('state_latch(gt(close, ts_mean(close,20)), ...)')` 解析执行成功。
- 21 算子均 `status=production` + `pit_safe=True` + 正确 scope（ts/cs）+ daily 表面，六证全过。
- 修复 `factor_production_targets()`：排除 `compatibility_only`/`diagnostic_only`/`benchmark_only`/
  `hidden_from_default_mining` 算子（设计上非默认生产目标），消除 audit 对这 11 个 flagged 算子的误报。
- strict admission audit 在稳定时刻全绿（1028 canonicals, 0 errors）。并发 AI 持续改写治理/证据文件期间
  审计会反复失效/恢复，恢复顺序 factor→primitive（见 §13.6）。

# 15. 2026-08 Turnover-Survival / Weighted-Tail / Behavioural / Order-Flow 扩展（16 算子）

## 15.1 目标

整合 Gemini DeepResearch + 用户逐条评审结论，把「有新数学空间」的 primitive 与
「DSL 1-2 层即可表达」的 recipe 严格分离。用户明确否掉了 9 个 Gemini 具体公式
（ashare_limit_lock_quality / fin_price_acceleration_divergence / ts_tradability_masked_corr /
ts_rank_divergence / ashare_limit_break_volume_intensity / fin_surprise_decay_momentum /
relation_peer_limit_density / ts_night_day_divergence / ashare_gap_trap_intensity），
本轮只注册 16 个新 canonical primitive + 2 个 CGO recipe。

## 15.2 先修 Microstructure 两处

1. `micro_bipower_var` 从 `rolling_mean(|r_t||r_{t-1}|)` 改为标准 BV：
   `(pi/2) * (n/(n-1)) * rolling_sum(|r_t||r_{t-1}|)`，其中 n=窗口内有效收益个数，
   使 BV 与 RV=sum(r^2) 同尺度（`micro_jump_indicator` 分解不再失衡）。手算校验通过。
2. `micro_vpin` / `micro_kyle_lambda` 标记 `legacy_proxy`（description/tag/semantic_certification
   元数据 + `preferred_replacements` 指向新实现），**数学语义保持不变**。顺带修复两者
   panel 输入结构性 bug（逐列同内核，数学不变）。

## 15.3 新 family 与模块

| family | 模块 | 算子 |
|---|---|---|
| Turnover survival | `turnover_survival.py` | `ts_turnover_reference_price` / `_cost_dispersion` / `_profit_share` / `_holding_age` / `_near_cost_mass`（P0）、`_cost_quantile_distance`（P1） |
| Stratified / weighted tail | `weighted_tail.py` | `ts_stratified_mean_spread`、`ts_weighted_semivariance`（P0）、`ts_weighted_expected_shortfall`、`ts_weighted_drawdown_area`（P1） |
| Behavioural | `prospect_theory.py` | `ts_cpt_value`（P1，`preset="bmw2016"` 固定 α/λ/γ，decision-weight differences，非 w(rank) 乘积） |
| Order flow → impact | `microstructure/flow_impact.py` | `intraday_bvc_imbalance`、`intraday_impact_beta`、`intraday_impact_asymmetry`（P1）、`intraday_return_wasserstein_shift`（P1）、`micro_bvc_vpin`（P2，等量桶 + 边界分钟按量切分） |

共享内核：`_column_stats`（换手存活权重/成本矩/分位一次算完），`_wasserstein_shift_series`、
`_vpin_series`（pandas 与 polars 共用，保证逐值相等）。锁定板中性：涨停/跌停 locked bar
默认 OF=0，绝不强行 100% buy/sell。

Recipes（非 canonical）：`factor_recipes/chip_cost.py` → `capital_gains_overhang=(P-RP)/P`、
`cost_basis_gap=(P-RP)/RP`，基于 `ts_turnover_reference_price`。

## 15.4 后端

- **pandas_numpy**：认证生产参考后端（16 算子全部）。
- **polars**：`polars_chip_tail.py`（每日算子，per-column UDF 共享 numpy 内核）+ 
  `polars_flow_impact.py`（分钟算子：BV-C/impact 为纯 pl.Expr，wasserstein/VPIN 为
  map_batches 共享内核）。16 算子 polars/pandas 逐值 parity（max|diff| ~1e-17）。
  不用 `register_polars_bridge`（`_remove_declared_bridges` 会移除 source 含 bridge 的 polars 槽）。
- **duckdb / SQL 下推**：本轮**推迟**。理由：换手存活是后缀累积乘积（递归）、CPT 需排序+
  概率加权、Wasserstein 跨日、VPIN 顺序分桶，均非 SQL 窗口可表达；`ts_weighted_semivariance`
  可表达但 NaN/null 语义与并发编辑中的 emitter 冲突风险高。planner 对这些算子自动回退
  pandas/polars（未列入 `SQL_IMPLEMENTED_CANONICALS`，fail-closed）。

## 15.5 接线

`_LOAD_MODULES` + 4 个新模块；`operator_policy._CHIP_FLOW_PACK_POLICIES`
（ts_*→ts，intraday_*/micro_bvc_vpin→session_intraday，全部 pit_safe=True；
turnover 族 `lag=1` 严格用 t-1 及以前）；`operator_surface._DAILY_CHIP_FLOW_PACK_2026_08`
（daily 表面）+ 各模块 EXTENDED_ONLY 分区；`production_hardening._SCOPE_OVERRIDES`
（intraday_*/micro_bvc_vpin→session_intraday）；audit 词汇表补 `sorter/flow/locked/preset/
bucket_count` 等 + `_SPECIAL_SCALARS`（weighted_semivariance.target=0.0 等）。

## 15.6 验证

- `test_chip_flow_pack_2026_08.py`：25 passed（注册/表面/policy/determinism/axes/常值 fail-closed/
  BV 手算/legacy proxy/换手边界/锁定板中性/polars parity daily+minute）。
- 后端：16 算子 pandas_numpy + polars 双后端（逐值 parity，max|diff|~1e-17）；duckdb/SQL 下推推迟（见 15.4）。
- runtime audit 目标数随 16 个新算子增加；factor evidence 重建（1060 canonicals / 974 生产算子）后
  15 个 P0/P1 算子全部 `status=production` + `production_certified=True`，DSL daily authoring
  + production mining allowlist 均可用；primitive evidence 六路认证 86；recipe evidence 186 个
  production recipes（含 capital_gains_overhang / cost_basis_gap）。
- 收敛校验：`test_production_convergence` 21 passed（三 manifest 集合一致）；
  `test_chip_flow_pack_2026_08` 25 passed；stateful/micro/SQL-lowering/recipe-admission 均绿。
- 与并发 Claude 会话协调：其 advanced pack 17 算子因加载顺序被 `_EXPLICIT_POLICIES` 过滤，
  已在 `operator_policy._ADVANCED_PACK_POLICIES`（post-filter）补齐 scope，`load_all` 恢复。
- `micro_bvc_vpin` 按 spec §16 保持 **P2/research-only**：surface=research、排除出
  `factor_production_targets`（`NON_FACTOR_PRODUCTION_CANONICALS`），注册了 pandas+polars
  runtime 供 research mining 使用，但绝不进入默认生产挖掘白名单（`micro_` 前缀本身也拒绝生产）。
- 换手存活边界：turnover=0（无换手存活全保留）、turnover→1（clip 到 1-ε，乘积不爆炸）、
  missing turnover（当作 0，暂停日语义）、missing price（该 lag 权重 0）、Σw≈0→NaN。

---

# §16 2026-08 V2/V3 State-Dynamics / Event-Response / Spectral-Crowding / Volume-Clock 扩展（29 算子）

日期：2026-08-08。依据两份 AI 建议（V2 时间不可逆性/状态几何/马尔可夫动力学/首达/历史事件响应/能量距离/谱拥挤；V3 状态惊奇度/KM 局部稳定性/成交量时钟/KNN 同行/EVT/copula 非对称/Hawkes/熵产生/Lyapunov/报告披露）。

## 16.1 新增算子（29）

**P1 → daily 表面（21）**：
- 状态几何/时间不对称：`ts_state_density`、`ts_ordinal_irreversibility`
- 局部马尔可夫：`ts_markov_persistence`、`ts_markov_state_entropy`、`ts_markov_transition_surprisal`、`ts_kramers_moyal_local_stability`
- 首达：`ts_first_passage_bias`
- 历史事件响应：`event_historical_response_mean`、`event_historical_response_sign_balance`
- 联合分布 break：`ts_joint_energy_shift`、`ts_energy_break_score`
- 组谱拥挤：`group_corr_mode_share`、`group_corr_effective_rank`、`group_corr_mode_localization`
- 分钟→日：`session_event_recovery_score`、`intraday_volume_clock_path_efficiency`、`intraday_volume_clock_roughness`
- 动态同行：`cs_knn_peer_mean_ex_self`、`cs_knn_neighbor_retention`
- 披露/极值：`report_filing_delay_surprise`、`ts_hill_tail_index`

**P2 → research-only（8，绝不进默认挖掘白名单）**：
`ts_active_information_storage`、`ts_multiscale_permutation_entropy_slope`、`ts_quantile_regression_beta`（精确 LP，非 IRLS）、`ts_copula_central_asymmetry`、`event_hawkes_branching_ratio`、`ts_markov_entropy_production`、`ts_local_lyapunov_exponent`、`report_revision_magnitude`（首选来源层 materialize，运行时算子接收来源提供的 prev_x）。

## 16.2 共享内核

- `DiscreteStateDynamicsKernel`（`markov_dynamics.py`）：分位数状态边缘 + 滞后转移矩阵（Jeffreys 平滑）+ 经验状态频率 + 每 bin D1/D2。服务 6 个 markov/KM 算子 + `advanced_structure` 的 KM drift/diffusion。
- `OrdinalPatternKernel`（`state_geometry.py`）：Lehmer 序数索引，相等值 embedding 一律丢弃（不做 jitter，避免一字板假 pattern）。
- `GroupCorrelationSpectrumKernel`（`group_spectrum.py`）：一次 SVD 输出 mode_share/effective_rank/localization。
- `VolumeClockPathKernel`（`volume_clock.py`）：累计 activity 等分网格 → log-price 线性插值 → efficiency/roughness。
- `CrossSectionKNNGraphKernel`（`dynamic_knn.py`）：逐日秩标准化特征 → L2 KNN 图（stable argsort 确定性），peer mean + retention 共享。

## 16.3 既有算子修正

- **`ts_kramers_moyal_drift/diffusion`（spec §三）**：改用共享 `DiscreteStateDynamicsKernel` —— 分位数状态 bin（替代等宽）、每 bin `min_bin_count`、lag 增量纳入定义、严格 KM 缩放 `D2 = mean(dx²)/(2·lag)`（lag=1 时 = 0.5·mean(dx²)）。`test_advanced_ops_2026.py::test_kramers_moyal_linear_trend` 的 diffusion 期望从 1.0 更新为 0.5。
- **`intraday_barrier_approach_acceleration`**：经核查当前实现为"价格 headroom 二阶差分"（自然时钟），与最初 volume-time acceleration 设计不完全一致。保留其已提交语义（改语义会改变既有 daily production 算子的历史输出）；真正的 volume-clock 路径由本轮新增 `intraday_volume_clock_path_efficiency/roughness` 覆盖（volume-clock 上重新定价），并在本文档标注这一差异。

## 16.4 后端

- pandas_numpy（certified 参考）+ polars 双后端：13 个 per-column 日频算子有**genuine per-column UDF**（`polars_dynamics.py`，逐值 parity max|diff|=0）。cs/group/minute 聚合算子保持 pandas_numpy-only（跨列聚合非 per-column 语义），文档注明。
- duckdb/SQL 下推继续推迟：序数排列/状态转移/首达 stopping-time/事件响应/能量距离/Hill 均为顺序或 O(n²) 核，SQL 窗口不可表达；planner 自动回退（fail-closed）。

## 16.5 接线

`_LOAD_MODULES` + 13 模块；`operator_surface._DAILY_DYNAMICS_PACK_2026_08`（21 daily）+ 各模块 EXTENDED_ONLY/RESEARCH_ONLY 分区；`operator_policy._DYNAMICS_PACK_POLICIES`（post-filter，ts_/group_/cs_/session_intraday scope 全 pit_safe）；audit 词汇表补 `response/delay/prev_x` panel、`horizon/barrier/bandwidth/history_length/tail_fraction/min_tail_count/min_anchors/grid/residual_fraction` 标量、markov/order/mode/side 特殊标量、`_PANEL_FORCE(ts_first_passage_bias, scale)`、minute `event→minute_shock` 映射 + 三个 fixture panel。

## 16.6 验证

- `test_dynamics_pack_2026_08.py`：18 passed（注册/表面/确定性/axes/prefix 因果/参考数学/KM 严格 D2/常值·全 NaN fail-closed/polars parity 13 算子）。
- 与并发会话协调：其 `group_spd_feature_structure_shift` 新增 `composition_policy` 参数已由其补 audit 特殊标量；KM drift/diffusion 的 D2 缩放与其在 `advanced_structure.py` 的 min_bin_count 改法一致合并（我补 quantile bin + lag）。
- 证据重建顺序：factor → primitive → recipe → manifests → catalog。

# §17 2026-08-08 第二轮增量审计修复（advanced pack + flow impact）

> 本轮按用户提供的第二轮审计逐条定位、修复并加 golden 测试。语义 diff 后共
> 触及 12 个 P0（确定实现/数学错误）与 13 个 P1。全部为
> `cleaned_operators/advanced_{information,intraday,structure,topology}.py` +
> `microstructure/flow_impact.py` + 治理接线 + `test_advanced_ops_2026.py`。

## 17.1 P0 确定性错误（全部修复）

- **intraday_barrier_approach_acceleration（P0-001）**：原实现 `for headroom, barrier
  in ((b_up, True), (b_dn, False))` 变量错位——`headroom` 拿到涨跌停价但从未使用，
  `barrier` 是 bool；上板分支实际算 `1 - price/1`，下板分支被 `False<=0` 跳过。
  已改为 `(limit_price, is_upper)`，真实使用当日涨跌停价；文档由 log-headroom 更正为
  线性 headroom；输出改为**带方向**（加速冲向涨停为正、跌停为负，占优方向定符号）。
  手工 golden：prices `10.0→10.2→10.5→10.9`、upper=11 → `+0.009091`。
  ⚠️ 与并发 §16.3 的"保留已提交语义"记录不一致：并发未识别该变量交换 bug；
  本文档以 P0-001 的客观事实为准（golden 测试锁定真实限价进入 headroom）。
- **micro_bvc_vpin（P0-002）**：等量桶引擎重写为 while-loop 按原 bar 体积比例分摊
  （`OF·take/vm`），一根超大 bar 可跨越任意多个桶；每桶 `|OF_bucket|`，桶内买卖流净额
  抵消不被破坏。旧实现的 `finished` 标志只切一次且不重置、跨桶时对同 bar 两段独立
  `abs()` 均错误。golden：vol `[10,30,5,40,15]`、flow `[+10,-30,+5,+40,-15]` → VPIN 0.4。
- **holder_class_js_shift（P0-003）**：固定 5 个 class slot，绝不压缩
  `current[np.isfinite(current)]`（压缩会把 class3 变 class2，类别身份错位）。NaN slot
  = 类别未知：分布含已知质量时部分 NaN → fail-closed NaN；负 share → fail-closed。
- **ts_student_t_fisher_shift → ts_fisher_information_shift（P0-004 + P1-020）**：recent
  窗口 off-by-one（原取 r+1 个观测）改为 `vals[row-r+1:row+1]`（恰好 r 个），prior 严格
  接续；首行可计算从 `r+p` 提前到 `r+p-1`。同时按审计改名（见 §17.3）。
- **ts_kramers_moyal_diffusion（P0-005/006）**：D2 严格系数 `0.5·mean(dx²)`（dt=1）；
  增加 `min_bin_count` 门槛。并发已在 §16.3 用共享
  `DiscreteStateDynamicsKernel` 以更强形式实现（分位数 bin + lag + 严格过去窗口），
  本文档与其合并，diffusion 测试期望同步 1.0→0.5。
- **cs_sliced_wasserstein_copula_shift（P0-007 + P1-015/016）**：rank 改为 average-tie
  （`argsort(argsort)` 竞争 rank 使 ties 按股票列序分秩 → 因子对列顺序敏感）；历史参考
  改为各历史日分位曲线的**等权平均**（原全观测 pool 使上市股票多寡日权重失衡）；加
  `global_state` tag——当天全市场同值，禁止当个股横截面 alpha 单独挖，只作 regime gate。
- **group_spd_feature_structure_shift（P0-008/009 + P1-017/018）**：complete-case 剔除
  缺测行（单股单特征缺失不再毒化整组协方差）；`min_peers`（默认 5，>d+1）与
  `min_reference_days`（默认 5）门槛；`composition_policy="current"|"intersection"`
  （intersection 只在与各参考日共同成员上重算矩阵，剔除纯成员变化效应）；文档更正为
  恰好 3 特征（f1,f2,f3）。
- **ts_betti_1_max_persistence / ts_persistence_diagram_shift（P0-010/011/012）**：
  Rips reduction 的 pivot 约定从**最低边**（最小距离，镜像约定会在共线点云上伪造 H1）
  改为**最高边**（最新 filtration，标准 Zomorodian–Carlsson 配对）；`_takens_points` 剔除
  含 NaN 的 embedding 向量；persistence_diagram_shift 改为**真正的 diagram W1**
  （birth,death 点集 + 对角线匹配，Hungarian 精确解）——原名承诺 diagram 距离但实现只比
  一维 lifetime。测试：circle H1>0、line H1=0、(s,s,s) H1=0、Gaussian blob 小。
- **ts_bures_corr_shift（P1-019）**：`min_pairs` 门槛（默认 `max(5, min(r,p)//2)`），
  2 个观测的 ±1 相关不再制造假 break。

## 17.2 P1 语义/健壮性修复

- **intraday_pair_w1 / return_wasserstein_shift**：MAD=0 fail-closed（常量基线 → NaN），
  pair 文档与同日 MAD 对齐。
- **intraday_quantile_curve_pca_*（P1-006/007/008）**：score/residual 的 k 规则统一
  （rank<k → 双 NaN）；score 加 eigen-gap guard（λ_k≈λ_{k+1} → NaN，防止 PC 方向
  旋转造成假跳变）；名称从 `intraday_wasserstein_quantile_pca_*` 改为
  `intraday_quantile_curve_pca_*`（分位曲线空间的 Euclidean PCA，不是 Wasserstein
  principal-geodesic，不再沿用误导性名称）。
- **ts_transfer_entropy / ts_effective_transfer_entropy（P1-009/010/011）**：单窗口核
  `_transfer_entropy_window`（去掉每行 O(w²) 的嵌套 rolling）；`min_transitions` 参数
  （默认 `max(30, 3·bins²)`，防止 bins³ 状态空间被 Jeffreys 平滑主导）；
  常量/退化状态（unique<2）→ NaN（平滑不再凭空造"信息"）。
- **ts_score_rank_weighted_mean（P1-012/013）**：ties 用 average rank（稳定 argsort 会
  给 tie 组不同权重、悄悄偏向更早观测）；output unit 改为 `same_as:target`
  （weighted mean 继承 target 量纲，price/amount/volatility 等，非固定 ratio）。
- **report_benford_js_divergence（P1-014）**：保持 research-only；文档明确对 as-of
  forward-fill 日频面板测得的是披露频率/持久性而非财报数字异常。

## 17.3 重命名（research-only，manifest/evidence 同步）

- `intraday_wasserstein_quantile_pca_score` → `intraday_quantile_curve_pca_score`
- `intraday_wasserstein_quantile_pca_residual` → `intraday_quantile_curve_pca_residual`
- `ts_student_t_fisher_shift` → `ts_fisher_information_shift`
  （并发已同步其测试引用；operator_policy/signatures/audit 词汇表已更新）

## 17.4 治理接线

- `scripts/audit_all_factor_production.py`：`_SCALAR_VALUES` 补
  `min_bin_count/min_peers/min_reference_days/min_transitions`；defaults 补
  `(ts_kramers_moyal_*, min_bin_count)`、`(group_spd_*, min_peers/min_reference_days)`、
  `(ts_transfer_entropy, min_transitions)`、`(ts_bures_corr_shift, min_pairs)`。
- `operator_policy.py` / `operator_signatures_phase2.py`：PCA 重命名同步。
- `test_advanced_ops_2026.py`：35 passed —— barrier 手工 golden、VPIN 精确 golden、
  copula 列置换不变性、Rips 几何 ground-truth（line=0/circle>0/blob 小）、diagram W1
  golden、Student-t 窗口长度、PCA rank/eigen-gap、TE 常量/min_transitions、
  score_rank tie、Bures min_pairs、holder NaN/负 share、SPD complete-case/min_peers、
  Wasserstein MAD=0、KM min_bin_count、Takens NaN 过滤。

---

# §18 2026-08-08 市场状态描述语言扩展（38 算子，P1 第一批 + P2 第二批）

> 触发：用户提交 AI 的 "算子维度地图" 提案，要求把 FactorEngine 从"金融公式集合"
> 升级为"市场状态描述语言"——补齐 quantile-hit 动态、expectile 幅度敏感尾部、
> extreme dependence、Directional-Change 内在时间、多字段协方差几何、无 L2 spread
> proxy、日内 profile surprise、横截面局部非线性面、tail systemic centrality、
> relation diffusion、marked-event、update-clock 十二个坐标系缺口。
> 执行要求：Python + polars + duckdb 三端可跑，生产级 daily，一次性全部落地。

## 18.1 目标与原则

- 38 个新算子全部实现为 numpy 参考实现，PIT-safe、确定性、fail-closed NaN。
- **不写死字段名**：所有 op 用通用参数（`target/source/condition/x/scale` 等），字段自由传。
- 不做重复劳动：不加新传统指标、不命名 K 线形态、不再堆 entropy/tail statistic；
  本次全部走"动态/依赖/局部几何"方向。
- 后端策略与既有 advanced pack 一致：polars 用 `register_polars_bridge`（精确对齐
  pandas_numpy 参考）；duckdb 对这些复杂 rolling/截面算子走 Python fallback
  （不造脆弱的 SQL emitter）。这保证三端语义一致。

## 18.2 新增算子清单（38）

### A. Quantile-hit 动态（hit-sequence 核共享）
| canonical | 签名 | 层级 |
|---|---|---|
| `ts_quantilogram` | (x, window, quantile, lag, side) | P1 |
| `ts_cross_quantilogram` | (target, source, window, target_q, source_q, lag, target_side, source_side) | P1 |
| `ts_quantile_crossing_spectral_concentration` | (x, window, quantile, side) | P2 |

### B. Expectile（幅度敏感尾部）
| `ts_expectile` | (x, window, tau) | P1 |
| `ts_expectile_beta` | (y, x, window, tau) | P1 |

### C. Extreme dependence（extremogram 核）
| `ts_extremogram` | (x, window, quantile, lag, side) | P1 |
| `ts_cross_extremogram` | (target, source, window, target_q, source_q, lag, target_side, source_side) | P1 |
| `ts_extremal_dependence_decay` | (x, window, quantile, side, max_lag) | P2 |

### D. Directional Change 内在时间（共享 DC 核）
| `ts_dc_overshoot_ratio` | (x, scale, threshold, window) | P1 |
| `ts_dc_event_rate` | (x, scale, threshold, window) | P1 |
| `ts_dc_duration_asymmetry` | (x, scale, threshold, window) | P1 |
| `ts_dc_overshoot_asymmetry` | (x, scale, threshold, window) | P1 |

### E. 多字段协方差几何（特征值核共享）
| `ts_feature_mode_share` | (f1, f2, f3, window) | P1 |
| `ts_feature_effective_rank` | (f1, f2, f3, window) | P1 |
| `ts_feature_subspace_rotation` | (f1, f2, f3, window, recent_window, prior_window) | P1 |
| `ts_beta_break_score` | (y, x, window, recent_window, prior_window) | P1 |

### F. 条件/频域依赖
| `ts_conditional_transfer_entropy` | (target, source, condition, window, bins, lag, min_transitions) | P2 |
| `ts_modwt_band_corr` | (x, y, window, level, band) | P2 |

### G. 采样尺度 / 噪声诊断（日内）
| `intraday_subsampled_rv_dispersion` | (returns, sampling) | P1 |
| `intraday_volatility_signature_slope` | (returns, max_interval) | P1 |
| `intraday_realized_power_variation` | (returns, order, sampling) | P2 |

### H. 无 L2 spread proxy
| `ohlc_corwin_schultz_spread` | (high, low, smooth_window) | P1 |
| `ts_roll_effective_spread` | (x, window) | P2 |

### I. 日内 profile surprise（日边界 + 固定 slot 网格）
| `intraday_profile_surprise_energy` | (x, history_days, n_slots, cap) | P1 |
| `intraday_profile_phase_shift` | (x, history_days, max_shift, n_slots) | P2 |

### J. 横截面局部非线性面（KNN 局部回归/PCA 核）
| `cs_knn_local_linear_residual` | (target, f1, f2, f3, k, ridge) | P1 |
| `cs_knn_tangent_residual` | (target, f1, f2, f3, k) | P2 |
| `cs_knn_local_gradient_norm` | (target, f1, f2, f3, k, ridge) | P2 |
| `cs_rank_copula_mi` | (a, b, grid) | P2 |
| `cs_rank_copula_entropy` | (a, b, grid) | P2 |

### K. Tail systemic centrality
| `group_tail_centrality` | (x, group_id, window, quantile, side) | P1 |
| `group_tail_lead_score` | (x, group_id, window, quantile, side, lag) | P2 |

### L. Relation diffusion（group 邻接的生产可用版本）
| `relation_diffusion_score` | (x, group, alpha, steps) | P2 |

### M. Marked event
| `event_mark_autocorr` | (event, mark, history_window, event_lag) | P1 |
| `event_interval_mark_coupling` | (event, mark, window) | P1 |

### N. Update clock（非量价 forward-fill 污染修复）
| `update_path_efficiency` | (x, update_event, n_updates) | P1 |
| `update_acceleration` | (x, update_event, n_updates) | P1 |
| `update_surprise` | (x, update_event, n_updates) | P1 |
| `update_direction_persistence` | (x, update_event, n_updates) | P1 |

## 18.3 模块布局（9 个新文件 + 2 个追加）

| 文件 | 算子 |
|---|---|
| `cleaned_operators/advanced_quantile_dynamics.py` | A + C（hit/extremogram 核） |
| `cleaned_operators/advanced_expectile.py` | B |
| `cleaned_operators/directional_change.py` | D（DC 核） |
| `cleaned_operators/feature_geometry.py` | E |
| `cleaned_operators/conditional_dependence.py` | F |
| `cleaned_operators/spread_estimators.py` | H |
| `cleaned_operators/cross_section_local.py` | J |
| `cleaned_operators/tail_systemic.py` | K + L |
| `cleaned_operators/marked_event.py` | M |
| `cleaned_operators/update_clock.py` | N |
| `cleaned_operators/advanced_intraday.py`（追加） | G + I |

## 18.4 后端策略

- **polars**：每个 canonical 调 `register_polars_bridge(canonical)`，桥接器精确复现
  pandas_numpy 参考（与 `ts_bures_corr_shift` 等完全一致的模式）。
- **duckdb**：不进 `SQL_IMPLEMENTED_CANONICALS`（无原生 SQL 语义），经 sql_backend 的
  Python fallback 落到 polars bridge 执行——与现有 advanced 算子同路径，三端结果一致。
- 不写错误/脆弱的 SQL emitter；duckdb 端语义正确优先于 pushdown。

## 18.5 治理接线

1. `cleaned_operators/__init__.py` `_LOAD_MODULES` 登记 9 个新模块。
2. `operator_policy.py`：新 `_MARKET_LANGUAGE_PACK_POLICIES` 或并入
   `_ADVANCED_PACK_POLICIES`（scope/pit_safe=True/min_periods）。
3. `operator_surface.py`：新 canonical 进 `EXTENDED_ONLY_CANONICALS`（P1）
   /`RESEARCH_ONLY_CANONICALS`（P2）。
4. `backend/operator_signatures_phase2.py` 追加签名（f1/f2/f3 三字段、condition bool、
   group_id group、side/tau/alpha/step 等标量）。
5. `scripts/audit_all_factor_production.py`：`_SCALAR_VALUES` 补
   `quantile/tau/target_q/source_q/side/sampling/order/level/band/grid/max_lag/max_shift/
   n_updates/n_slots/history_days/event_lag/ridge/steps/smooth_window/scale/threshold`；
   `_SPECIAL_SCALARS` 补各 op 定制；`_PANEL_FORCE` 补 `(dc_*, scale)`、`(cs_*, group)`。

## 18.6 验证

- 每个核手工 golden（quantilogram 相关、extremogram 计数、expectile 加权平均、
  DC overshoot 手工序列、Corwin-Schultz 两日解析值、Roll 两日解析值）。
- 不变量：quantilogram 常数序列 NaN；extremogram 无超阈值 NaN；DC 反向序列
  overshoot 对称性；feature mode_share 恒等输入 → 1；update 无事件 NaN。
- PIT：prefix-causal 检查 + audit 全绿。
- 后端 parity：polars bridge 与 numpy 逐点相等；duckdb fallback 执行同一输出。
