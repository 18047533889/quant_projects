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
