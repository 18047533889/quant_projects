# factor_engine 算子深度参考手册（OPERATORS_DEEP_REFERENCE）

> **生成方式**: 本手册由 `scripts/generate_operators_deep_reference.py` 从四个真相源机器合并生成：
> ① `cleaned_operators/docs/operators_catalog.json`（1737 canonical 元数据）
> ② `cleaned_operators/docs/算子全览.md`（人工编写公式节）
> ③ `docs/operator_core_specs.yaml`（86 条核心 spec：空值/值域/溢出/后端策略）
> ④ `cleaned_operators/docs/operator_doc_semantics.py`（115 条显式精讲 OpDoc）
> **勿手改本文件** —— 改对应真相源后重跑生成器（文档与代码同步硬性规则）。
> 生成覆盖: canonical 1737 个；其中带公式/讲解文档的 184 个，其余为元数据骨架条目（已如实标注「暂无公式」）。

## 四要素说明

每个算子四段：

1. **标签** — surface（日线/扩展/研究）、scope（时序/截面/分组/逐元素）、PIT 安全性、别名
2. **公式/构造** — LaTeX 公式（无公式时给构造式说明）
3. **可用条件** — lookback/min_periods/lag、DSL 白名单、空值/值域/溢出策略、生产准入、后端覆盖
4. **功能讲解** — 含义、计算方式、精讲、示例

## 快速导航（按族）

- **截面/排名/中性化**（79）
- **分组 group_**（51）
- **时序 ts_**（555）
- **均线/MA 族**（6）
- **技术指标 (MACD/RSI/ATR/布林等)**（51）
- **事件/条件/门控**（44）
- **量价/流动性**（110）
- **波动/风险估计**（60）
- **分钟级 intraday**（20）
- **基本面/财务/PIT**（168）
- **A 股特有**（21）
- **数学/安全运算**（25）
- **比较/逻辑/判断**（12）
- **其他**（535）

---

## 截面/排名/中性化（79 个）

### `cs_actual_lof_score`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_autoencoder_reconstruction_error`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_bucket`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\mathrm{cs_bucket}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

按交易日横截面平均排名分桶。

计算: 在 other 语义下对 panel 输入执行 `cs_bucket`；具体边界条件（min_periods、NaN）以实现代码为准。

### `cs_bucket_fixed`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_bucket_historical`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_count`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `c_count`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_coverage_ratio`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_demean`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `CS_DEMEAN`, `c_demean`

**公式/构造**:

$$\tilde{X}_{i,t} = X_{i,t} - \frac{1}{N_t}\sum_j X_{j,t}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面去均值。

计算: 每个交易日减去截面算术平均。

备注: 截面数据减去均值

示例: ``c_demean(returns)``

精讲: 每个交易日减去截面算术平均。

### `cs_empirical_bayes_shrinkage`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_factor_bucket_return`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_fill_mean`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_fill_median`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_hartigan_dip`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 100
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_huber_resid`

**标签**: `日线` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_impute_mean`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_impute_median`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_isolation`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_isolation_forest_score`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_isotonic_residual`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_isotonic_residual_lagged_direction`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_distance`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_graph_dirichlet_energy`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_local_gradient_norm`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_local_linear_residual`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_local_moran`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_neighbor_retention`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_peer_mean_ex_self`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_knn_tangent_residual`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_lad_resid`

**标签**: `日线` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_local_curvature`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_local_density`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_local_density_score`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_mad`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\mathrm{cs_mad}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面中位绝对偏差（MAD，广播到各列）。

计算: 在 other 语义下对 panel 输入执行 `cs_mad`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``cs_mad(PE)``

### `cs_mad_zscore`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\mathrm{cs_mad_zscore}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

MAD 稳健 Z-Score：(x - median) / MAD。

计算: 在 other 语义下对 panel 输入执行 `cs_mad_zscore`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``cs_mad_zscore(PE)``

### `cs_mahalanobis_distance`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_mean`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `c_mean`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_multi_resid`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\mathrm{cs_multi_resid}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

多变量横截面 OLS 残差。

计算: 在 other 语义下对 panel 输入执行 `cs_multi_resid`；具体边界条件（min_periods、NaN）以实现代码为准。

### `cs_multi_ridge_resid`

**标签**: `扩展` · `截面` · `PIT 安全` · 别名: `cs_multi_robust_resid`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_neighbor_gap`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_neutralize`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_pct_rank`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `rank_pct`

**公式/构造**:

$$\mathrm{cs_pct_rank}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面百分位排名（同 rank_pct）。

计算: 在 other 语义下对 panel 输入执行 `cs_pct_rank`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``cs_pct_rank(close)``

### `cs_physical_panel_coverage`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_predictability_mosaic_score`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_quantile`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `c_percentile`, `quantile`

**公式/构造**:

$$\mathrm{cs_quantile}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面 p 分位数值（广播到各列）。

计算: 在 other 语义下对 panel 输入执行 `cs_quantile`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``cs_quantile(PE, 0.5)``

### `cs_quantile_resid`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_rank_churn`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_rank_combined_churn`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_rank_composition_churn`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_rank_copula_entropy`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_rank_copula_mi`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_rank_gaussian`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_regression`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$Y_{i,t} = \alpha_t + \beta_t X_{i,t} + \varepsilon_{i,t}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面回归（mode 控制输出：0=残差，1=β，2=拟合值）。

计算: 每个交易日对 (X,Y) 做 OLS，按 mode 返回残差、斜率或拟合值。

备注: 截面回归: mode=0 残差, 1 beta, 2 拟合值

示例: ``cs_regression(factor, size, 0)``

精讲: 每个交易日对 (X,Y) 做 OLS，按 mode 返回残差、斜率或拟合值。

### `cs_relative_density_ratio`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_resid`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\varepsilon_{i,t} = Y_{i,t} - (\alpha_t + \beta_t X_{i,t})$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面线性回归残差。

计算: 每个交易日 OLS：Y ~ X，输出 ε = Y - (α + βX)。

备注: 截面线性回归残差 y - (α + βx)

示例: ``cs_resid(factor, size)``

精讲: 每个交易日 OLS：Y ~ X，输出 ε = Y - (α + βX)。

### `cs_residual_percentile`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_ridge_resid`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_robust_mahalanobis_mad`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_shrink_to_group_mean`

**标签**: `扩展` · `分组` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_shrinkage_mahalanobis`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_sliced_wasserstein_copula_shift`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_spline_resid`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_std`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `c_std`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_sum`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `c_sum`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_tail_breadth`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_tail_retention`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_trimmed_ols_resid`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查` · 别名: `cs_robust_resid`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_universe_coverage`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_valid_count`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_weighted_demean`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_weighted_mean`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_weighted_percentile_rank`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_weighted_zscore`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs_wls_resid`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\mathrm{cs_wls_resid}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

加权横截面回归残差。

计算: 在 other 语义下对 panel 输入执行 `cs_wls_resid`；具体边界条件（min_periods、NaN）以实现代码为准。

### `normalize`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

$$\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{\cdot,t}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面归一化（去均值）。

计算: 每个交易日 X_{i,t} 减去截面均值。

备注: 归一化到[0, 1]（按行 min-max）

示例: ``normalize(x)``

精讲: 每个交易日 X_{i,t} 减去截面均值。

### `rank`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `CS_RANK`, `RANK`, `c_rank`, `cs_rank`, `cs_rank_01`, `panel_rank`

**公式/构造**:

$$R_{i,t} = \frac{\mathrm{rank}(X_{i,t}) - 1}{N_t - 1}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面百分位排名。

计算: 在同一交易日 t，对所有标的的 X_{i,t} 做秩变换，映射到 [0,1]。

备注: 截面 0-1 排名；pandas 百分位排名见 rank_pct

示例: ``rank(close)``

精讲: 在同一交易日 t，对所有标的的 X_{i,t} 做秩变换，映射到 [0,1]。

### `rank_corr`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `RANKCORR`, `RANK_CORR`, `rankcorr`

**公式/构造**:

$$\rho_s = \mathrm{corr}\bigl(\mathrm{rank}(X),\mathrm{rank}(Y)\bigr)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

秩相关系数（Spearman 型）。

计算: d=0：截面内对 rank(X)、rank(Y) 求 Pearson 相关；d>0：窗口内分别做秩变换后求相关。

备注: 秩相关系数（d=0 为截面，d>0 为时序窗口）

示例: ``rank_corr(close, volume, 20)``

精讲: d=0：截面内对 rank(X)、rank(Y) 求 Pearson 相关；d>0：窗口内分别做秩变换后求相关。

### `winsorize`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `WINSORIZE`, `c_winsorize`

**公式/构造**:

$$\tilde{X}_{i,t} = \mathrm{clip}\bigl(X_{i,t}, Q_p, Q_{1-p}\bigr)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面缩尾。

计算: 将截面分布两端超出 [p, 1-p] 分位的值截断到分位点。

备注: 缩尾处理

示例: ``winsorize(x, 0.05, 0.95)``

精讲: 将截面分布两端超出 [p, 1-p] 分位的值截断到分位点。

### `winsorize_mean`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{mean}(\mathrm{winsorize}(X))$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截尾均值。

计算: winsorize 后求均值。

备注: winsorize_mean（Polars 桥接）

示例: ``winsorize_mean(x, 0.1)``

### `zscore`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `CS_ZSCORE`, `ZSCORE`, `c_zscore`, `cs_zscore`, `panel_standardize`, `panel_zscore`, `standardize`

**公式/构造**:

$$Z_{i,t} = \frac{X_{i,t} - \bar{X}_{\cdot,t}}{\sigma_{\cdot,t}}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面 Z 分数。

计算: 每个交易日对全市场去均值、除以截面标准差。

备注: 对截面数据进行Z-Score标准化（与c_zscore相同）

示例: ``zscore(PE)``

精讲: 每个交易日对全市场去均值、除以截面标准差。

## 分组 group_（51 个）

### `group_corr_mst_length`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_count`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\mathrm{group\_}count(X_{i,t}, g(i))$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组截面 count。

计算: 每个 (timestamp, group) 内计算 count，再映射回各标的。

备注: 组内非空计数

示例: ``group_count(close, industry_code)``

### `group_current_members_tail_coexceedance`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_tail_coexceedance_density`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_decay_linear`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_rank_linear_weighted_value`

**公式/构造**:

$$\sum_j w_j X_j,\ w_j \propto (N-j+1)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内线性衰减加权。

计算: 组内按排名线性递减权重。

示例: ``group_decay_linear(ROE, industry_code, 5)``

### `group_distribution_js_divergence`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_ex_self_mad`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_ex_self_mean`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_ex_self_quantile`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_ex_self_std`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_ex_self_weighted_mean`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_coverage_ratio`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_effective_rank`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_corr_effective_rank`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_mode_localization`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_corr_mode_localization`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_mode_share`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_corr_mode_share`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_second_mode_localization`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_corr_second_mode_localization`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_spectral_gap`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `group_corr_spectral_gap`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_feature_valid_member_count`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_impute_median`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_kurtosis`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_leader_laggard_exposure`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_max`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\mathrm{group\_}max(X_{i,t}, g(i))$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组截面 max。

计算: 每个 (timestamp, group) 内计算 max，再映射回各标的。

备注: 组内最大值

示例: ``group_max(close, industry_code)``

### `group_mean`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\bar{X}_{g,t} = \frac{1}{|G|}\sum_{j\in G} X_{j,t}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内均值（广播回原行）。

计算: 每个 (t, group) 内算术平均，赋给组内每个标的。

备注: 组内均值

示例: ``group_mean(ROE, industry_code)``

精讲: 每个 (t, group) 内算术平均，赋给组内每个标的。

### `group_min`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\mathrm{group\_}min(X_{i,t}, g(i))$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组截面 min。

计算: 每个 (timestamp, group) 内计算 min，再映射回各标的。

备注: 组内最小值

示例: ``group_min(close, industry_code)``

### `group_multi_level_rank_consistency`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_multi_resid`

**标签**: `扩展` · `分组` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_neutralize`

**标签**: `日线` · `分组` · `PIT 安全` · 别名: `INDUSTRY_NEUTRAL`, `INDUSTRY_NEUTRALIZE`, `IND_NEUTRALIZE`, `NEUTRALIZE`, `c_neutralize`, `group_demean`, `ind_neutralize`, `industry_neutral`, `industry_neutralize`, `neutralize`, `panel_neutralize`

**公式/构造**:

$$\tilde{X}_{i,t} = X_{i,t} - \bar{X}_{g(i),t}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内去均值（行业/分组中性化）。

计算: 每个 (t, group) 内 X - 组内均值；别名 group_demean、neutralize。

备注: 在指定分组内进行去均值处理 (x - group_mean)

示例: ``group_demean(ROE, industry_code)``

精讲: 每个 (t, group) 内 X - 组内均值；别名 group_demean、neutralize。

### `group_normalize`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\tilde{X} = \frac{X - X_{\min}}{X_{\max}-X_{\min}}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内 [0,1] 归一化。

计算: 组内 min-max 缩放。

示例: ``group_normalize(ROE, industry_code)``

### `group_peer_beta_deviation`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_peer_deviation_index`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_peer_information_diffusion`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_percentile`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\mathbf{1}\{X_{i,t} \ge Q_p(X_{g,t})\}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内分位指示。

计算: 判断值是否高于组内分位阈值。

备注: 组内 top/bottom 分位掩码；与 pandas 审计实现语义一致。

示例: ``group_percentile(ROE, industry, 0.2)``

### `group_quantile_spread`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_rank`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$R_{i,t} = \mathrm{rank}_{g(i)}(X_{i,t})$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内截面排名。

计算: 每个 (t, group) 内对 X 做 [0,1] 秩分位。

备注: 组内排名 pct

示例: ``group_rank(ROE, industry_code)``

精讲: 每个 (t, group) 内对 X 做 [0,1] 秩分位。

### `group_rank_weighted_value`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_return_dispersion_exposure`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_signal_attraction_share`

**标签**: `扩展` · `分组` · `⚠️ PIT 需核查` · 别名: `relation_pagerank_centrality`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_skewness`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_spd_feature_structure_shift`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_std`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\sigma_{g,t} = \mathrm{std}_{j\in G}(X_{j,t})$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内标准差（广播）。

计算: 每个 (t, group) 内样本标准差。

备注: 组内标准差

示例: ``group_std(ROE, industry_code)``

精讲: 每个 (t, group) 内样本标准差。

### `group_sum`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\mathrm{group\_}sum(X_{i,t}, g(i))$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组截面 sum。

计算: 每个 (timestamp, group) 内计算 sum，再映射回各标的。

备注: 组内求和

示例: ``group_sum(volume, industry_code)``

### `group_tail_centrality`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_tail_lead_score`

**标签**: `扩展` · `分组` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_tail_ratio`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_topk_mean`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_ts_decay_linear`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_valid_count`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_wasserstein_barycenter_distance`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_weighted_mean`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_weighted_zscore`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `group_winsorize`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$\tilde{X} = \mathrm{clip}(X, Q_a, Q_{1-a})$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内缩尾。

计算: 组内两端超出分位截断。

示例: ``group_winsorize(ROE, industry_code, 0.05)``

### `group_zscore`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

$$Z_{i,t} = \frac{X_{i,t} - \mu_{g(i),t}}{\sigma_{g(i),t}}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

组内 Z 分数。

计算: 每个 (t, group) 内标准化。

备注: 组内 zscore

示例: ``group_zscore(ROE, industry_code)``

精讲: 每个 (t, group) 内标准化。

## 时序 ts_（555 个）

### `ts_abdi_ranaldo_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_abs_concentration`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_abs_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_abs_entropy_nats`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_abs_entropy_normalized`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_active_information_storage`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_activity_clock_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_activity_clock_lagged_value`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_activity_clock_lagged_value_prior`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `activity_clock_lagged_value_prior`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_adaptive_noise_kalman`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_alpha_beta_filter`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_coeff_stability`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_coefficient`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_fitted_value`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_forecast`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_in_sample_resid`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_innovation`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_innovation_z`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_prior_coeff`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_prior_forecast`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_prior_innovation`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ar_prior_innovation_z`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_argmax`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `m_argmax`, `ts_arg_max`

**公式/构造**:

$$\arg\max_{0\le k < d} X_{i,t-k}\;\;\text{（返回偏移量，0=最新）}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

窗口内最大值距当前 bar 的偏移（0 表示最新 bar 为最大值）。

计算: 在窗口 {t-d+1,…,t} 内找 argmax，返回 t - t^*（t^* 为最大值所在时刻）。

备注: 窗口最大值距当前 bar 的距离（0=当前，tie 取最近）

示例: ``ts_argmax(high, 20)``

精讲: 在窗口 {t-d+1,…,t} 内找 argmax，返回 t - t^*（t^* 为最大值所在时刻）。

### `ts_argmax_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_argmax_index_from_oldest`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_argmin`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `m_argmin`, `ts_arg_min`

**公式/构造**:

$$\arg\min_{0\le k < d} X_{i,t-k}\;\;\text{（返回偏移量，0=最新）}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

窗口内最小值距当前 bar 的偏移（0 表示最新 bar 为最小值）。

计算: 在窗口 {t-d+1,…,t} 内找 argmin，返回 t - t^*。

备注: 窗口最小值距当前 bar 的距离（0=当前，tie 取最近）

示例: ``ts_argmin(low, 20)``

精讲: 在窗口 {t-d+1,…,t} 内找 argmin，返回 t - t^*。

### `ts_argmin_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_argmin_index_from_oldest`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_autocorr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_autocorr}(X)$$

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动自相关 corr(x_t, x_{t-lag}) within window。

计算: 在 other 语义下对 panel 输入执行 `ts_autocorr`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``ts_autocorr(returns, 20, 1)``

### `ts_autocorr_decay_half_life`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_autocorrelation_time`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_integrated_autocorrelation_time`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_autocorrelation_time_initial_positive_sequence`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_average_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bds_statistic`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 30
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bessel_lowpass_causal`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_best_lag_corr_excess`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_best_lag_corr_raw`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_best_lag_corr`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_beta`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `beta`, `m_beta`, `rolling_beta`

**公式/构造**:

$$\beta = \mathrm{Cov}(X,Y)/\mathrm{Var}(Y)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的Beta。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算Beta；min_periods 规则以实现为准。

备注: 滚动 Beta

示例: ``m_beta(returns, market, 20)``

### `ts_beta_break_score`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_beta_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_betti_1_max_persistence`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bicoherence_top_decile_excess`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bicoherence_top_decile_mean`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_bicoherence_max`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 16
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_binned_response_curvature`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_binned_response_monotonicity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bottomk_mean`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_bottom_n_avg`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bottomk_std`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bottomk_sum`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_bottom_n_sum`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_breakdown_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_breakout_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_bures_corr_shift`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_butterworth_lowpass_causal`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_causal_local_linear_smoother`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_causal_savgol_endpoint`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_change_point_probability`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_channel_position`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_channel_width`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_channel_width_atr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_channel_width_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_channel_width_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_chatterjee_xi`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_chord_excursion_area`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_conditional_mutual_information`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_conditional_transfer_entropy`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_confirmed_pivot_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_confirmed_pivot_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_consolidation_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_consolidation_volume_decay`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_consolidation_width`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_copula_central_asymmetry`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_corr`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Corr`, `TS_CORR`, `corr`, `correlation`, `m_cor`, `ts_correlation`

**公式/构造**:

$$\rho_{XY}^{(d)} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动 Pearson 相关系数。

计算: 窗口内 corr(X, Y)。

备注: 滚动相关系数 (ts_correlation的别名)

示例: ``ts_corr(close, volume, 20)``

精讲: 窗口内 corr(X, Y)。

### `ts_corr_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_count_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_count_if}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动统计条件为真的次数。

计算: 在 other 语义下对 panel 输入执行 `ts_count_if`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_cov`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Cov`, `Covariance`, `TS_COV`, `cov`, `m_cov`, `ts_covariance`

**公式/构造**:

$$\mathrm{Cov}^{(d)}(X,Y) = \frac{1}{d-1}\sum(X-\bar{X})(Y-\bar{Y})$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动协方差。

计算: 窗口内 Cov(X, Y)。

备注: 滚动协方差 (与m_cov相同)

示例: ``ts_cov(returns, market, 20)``

精讲: 窗口内 Cov(X, Y)。

### `ts_cov_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_coverage_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cpt_value`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cross_extremogram`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cross_quantilogram`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cross_spectral_coherence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cross_spectral_phase`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_crossing_acceleration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_crossing_speed`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cumulative_deviation_score`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_cusum_break_score`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_current_drawdown_area`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_current_drawdown_duration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cusum_pressure`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_cusum_vol_break_score`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_days_since`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `event_age`

**公式/构造**:

$$\mathrm{ts_days_since}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

距最近一次条件成立的交易行数。

计算: 在 other 语义下对 panel 输入执行 `ts_days_since`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_days_since_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_days_since_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dc_duration_asymmetry`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dc_event_rate`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dc_overshoot_asymmetry`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dc_overshoot_ratio`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_decay_exp_window`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\sum_{k=0}^{d-1} e^{-\lambda k} X_{t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

指数衰减滚动窗口。

计算: 窗口内指数权重加权。

备注: 指数加权滚动

示例: ``ts_decay_exp_window(volume, 10, 0.5)``

### `ts_decay_linear`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `DECAY_LINEAR`, `TS_DECAY_LINEAR`, `decay_linear`, `ts_decay`

**公式/构造**:

$$\sum_{k=0}^{d-1}\frac{d-k}{\sum_{j=1}^{d}j} X_{t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

线性衰减加权求和。

计算: 窗口内按 1,2,…,d 线性权重加权；别名 decay_linear、ts_decay。

备注: 线性加权滚动平均

示例: ``ts_decay_linear(close, 20)``

精讲: 窗口内按 1,2,…,d 线性权重加权；别名 decay_linear、ts_decay。

### `ts_delay`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `DELAY`, `Delay`, `Ref`, `delay`, `m_delay`, `prev`, `shift`

**公式/构造**:

$$X_{i,t-d}$$

**可用条件**:

- 最少样本 min_periods = 1
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滞后 d 期。

计算: X_{t-d}。

备注: n期滞后 (与Ref相同)

示例: ``ts_delay(close, 5)``

精讲: X_{t-d}。

### `ts_delay_intrinsic_dimension`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_delta`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Delta`, `Diff`, `TS_DELTA`, `delta`, `deltas`

**公式/构造**:

$$\Delta_d X_t = X_{i,t} - X_{i,t-d}$$

**可用条件**:

- 最少样本 min_periods = 1
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

d 期差分。

计算: X_t - X_{t-d}。

备注: n期差分 (与Delta相同)

示例: ``ts_delta(close, 1)``

精讲: X_t - X_{t-d}。

### `ts_detrended_level_spectral_entropy`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dfa_hurst`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_corr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_correlation_partial_proxy`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_partial_distance_correlation`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_cov`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_to_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_to_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_to_resistance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_distance_to_support`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_dominant_frequency`

**标签**: `研究专用` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 12
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_dominant_growth_rate`

**标签**: `研究专用` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 12
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_level_dominant_frequency`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_level_dominant_growth_rate`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_level_mode_concentration`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_mode_concentration`

**标签**: `研究专用` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 12
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_return_dominant_frequency`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_return_dominant_growth_rate`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dmd_return_mode_concentration`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_dominant_cycle_period`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 16
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_downside_deviation`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_edge_effective_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_effective_transfer_entropy`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_effective_turning_rate`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ema`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `EMA`, `ema`, `ewm`, `ewm_mean`

**公式/构造**:

$$\mathrm{EMA}_t = \alpha X_t + (1-\alpha)\mathrm{EMA}_{t-1}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

指数移动平均（EMA）。

计算: 对序列做指数加权平滑，近期权重更大；span 参数以实现为准。

备注: 指数移动平均

示例: ``EMA(close, 12)``

精讲: 对序列做指数加权平滑，近期权重更大；span 参数以实现为准。

### `ts_endpoint_deviation`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_energy_break_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_envelope_boundary_dwell`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_envelope_compression`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_envelope_pressure`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_event_spacing_cv`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `event_interval_cv`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_event_spacing_mean`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `event_interval_mean`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_evt_threshold_stability`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ewm_corr`

**标签**: `日线` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `polars`

**功能讲解**:

算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ewm_cov`

**标签**: `日线` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `polars`

**功能讲解**:

算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ewm_std`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ewm_std`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ewm_var`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ewm_var`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expected_shortfall`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expected_shortfall_asymmetry`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 6
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile_beta`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile_beta_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile_regression_coeff`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile_regression_coeff_prior`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile_regression_forecast_error`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_expectile_regression_resid`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_extrema_confirmation_rate`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_extrema_divergence_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_extremal_dependence_decay`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_extremal_index`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_extreme_cluster_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_extremogram`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_feature_effective_rank`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_feature_mode_share`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_feature_pca_reconstruction_error`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_feature_subspace_rotation`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ffill_limited`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_fir_lowpass_causal`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_first_passage_bias`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_first_passage_conditional_time`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_first_passage_hit_probability`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_fisher_information_shift`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_forbidden_ordinal_pattern_excess`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_forbidden_ordinal_pattern_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_forbidden_ordinal_pattern_signed_excess`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_fractional_difference`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_fractional_difference_discarded_weight_mass`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_gap_fill_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_gap_reversion_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_gap_survival_duration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_garch_next_vol_forecast`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_garch_vol_forecast`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_garch_persistence`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_garch_standardized_shock`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_garch_vol_surprise`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_generalized_hurst_exponent`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_generalized_hurst_spread_q1_q4`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_multifractal_width`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_gjr_garch_vol_forecast`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_gjr_leverage`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_glr_mean_shift_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_glr_variance_shift_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_gpd_shape_pwm`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_h_infinity_level_filter`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hampel_filter_causal`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hankel_effective_rank`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hankel_singular_gap`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_har_from_return_forecast_error_z`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_har_from_return_next_vol`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_har_rv_forecast_error_z`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_har_rv_innovation_z`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_har_rv_next_var_forecast`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_har_rv_next_vol_forecast`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_har_rv_forecast`, `ts_har_rv_next_forecast`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hartigan_dip`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_higuchi_fractal_dimension`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hill_tail_index`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hodges_lehmann_location`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hsic`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_coeff`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_coeff_prior`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_forecast_error`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_forecast_error_z`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_in_sample_resid`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_huber_regression_resid`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_predictive_resid`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_huber_regression_resid_z`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hurst_dfa`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hvg_assortativity`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hvg_clustering_coefficient`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hvg_degree_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hvg_forward_backward_asymmetry`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hvg_motif_entropy`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hysteresis_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_hysteresis_state`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_impulse_return`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_impulse_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_impulse_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_industry_liquidity_beta`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_interval_exploration_efficiency`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_interval_nesting_depth`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_interval_occupancy_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_interval_occupancy_mode_distance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_interval_overlap_connected_component_ratio`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_interval_overlap_component_ratio`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_interval_union_coverage`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_joint_energy_shift`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_jump_bipower_proxy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kalman_beta`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kalman_beta_change`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kalman_beta_uncertainty`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kalman_innovation_z`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kalman_level`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kalman_trend`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kama`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kernel_granger_score`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 30
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_km_diffusion_gradient`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_km_equilibrium_distance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_km_quasipotential_depth`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kramers_moyal_diffusion`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kramers_moyal_drift`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kramers_moyal_local_stability`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ks_shift`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 6
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_kurt`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Kurt`, `TS_KURT`, `kurt`, `m_kurt`, `ts_kurtosis`

**公式/构造**:

$$\mathrm{kurt}^{(d)}(X) = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3$$

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的峰度。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算峰度；min_periods 规则以实现为准。

备注: 滚动峰度 (与m_kurt相同)

示例: ``m_kurt(returns, 20)``

### `ts_l1_trend_filter_trailing`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_l_kurtosis`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_l_skewness`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lag_of_peak_corr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 6
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lagged_mutual_information`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_last_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_last_if}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

窗口内最近一次条件成立时的值。

计算: 在 other 语义下对 panel 输入执行 `ts_last_if`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_last_pivot_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_last_pivot_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lempel_ziv_complexity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_level_shift_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_leverage_effect`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_line_convergence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_line_parallelism`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lo_mackinlay_vr`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lo_mackinlay_z`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_local_lyapunov_exponent`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_location_shift`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 6
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_log_return`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `log_returns`

**公式/构造**:

$$\mathrm{ts_log_return}(X)$$

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

d 期对数收益率 ln(x_t / x_{t-d})。

计算: 在 other 语义下对 panel 输入执行 `ts_log_return`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``ts_log_return(close, 1)``

### `ts_lower_partial_moment`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lower_tail_coexceedance_probability`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_lower_tail_dependence`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_lz_complexity`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mad`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `m_mad`

**公式/构造**:

$$\mathrm{MAD} = \mathrm{median}(|X-\mathrm{median}(X)|)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的MAD。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算MAD；min_periods 规则以实现为准。

备注: 滚动中位绝对偏差；与 pandas 审计实现语义一致。

示例: ``ts_mad(returns, 20)``

### `ts_market_liquidity_beta`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_committor`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_entropy_production`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_mean_first_passage_time`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_persistence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_spectral_gap`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_state_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_stationary_surprisal`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_markov_transition_surprisal`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mass_concentration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_matrix_profile_discord_score`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_matrix_profile_motif_distance`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_matrix_profile_motif_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_matrix_profile_motif_frequency`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_matrix_profile_neighbor_dispersion`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_matrix_profile_novelty`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_multivariate_matrix_profile_novelty`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_max`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Max`, `TS_MAX`, `m_max`, `window_max`

**公式/构造**:

$$M_{i,t} = \max_{0\le k<d} X_{i,t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的最大值。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算最大值；min_periods 规则以实现为准。

备注: 滚动最大值 (与m_max相同)

示例: ``ts_max(high, 20)``

### `ts_max_buildup`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\sum_{s=t-d+1}^{t} \mathbf{1}\{X_{i,s} = \max_{u\le s} X_{i,u}\}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

窗口内持续创新高的次数。

计算: 统计最近 d 期内 X 创历史新高的次数。

备注: ts_max_buildup（Polars colwise 核）

精讲: 统计最近 d 期内 X 创历史新高的次数。

### `ts_max_chord_excursion`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_max_drawdown`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_max_drawdown}(X)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口最大回撤。

计算: 在 other 语义下对 panel 输入执行 `ts_max_drawdown`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_max_drawdown_activity_cost`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_max_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mean`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Mean`, `SMA`, `TS_MEAN`, `m_avg`, `ma`, `mean`, `move`, `running_mean`, `window_mean`

**公式/构造**:

$$\bar{X}_{i,t} = \frac{1}{d}\sum_{k=0}^{d-1} X_{i,t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的算术平均。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算算术平均；min_periods 规则以实现为准。

备注: 滚动均值 (与m_avg相同)

示例: ``ts_mean(close, 20)``

### `ts_mean_abs_deviation`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `Mad`, `mad`, `ts_mean_absolute_deviation`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mean_excess_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mean_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_mean_if}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动条件均值。

计算: 在 other 语义下对 panel 输入执行 `ts_mean_if`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_mean_reversion_half_life`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mean_reversion_ou_approx_half_life`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mean_reversion_ou_approx_half_life_prior`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_median`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Median`, `m_median`, `median`

**公式/构造**:

$$\mathrm{median}(X_{i,t-d+1:t})$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的中位数。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算中位数；min_periods 规则以实现为准。

备注: 滚动中位数

示例: ``m_median(close, 20)``

### `ts_median3_causal`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_median_abs_deviation`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_median_absolute_deviation`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_min`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Min`, `TS_MIN`, `m_min`, `window_min`

**公式/构造**:

$$m_{i,t} = \min_{0\le k<d} X_{i,t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的最小值。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算最小值；min_periods 规则以实现为准。

备注: 滚动最小值 (与m_min相同)

示例: ``ts_min(low, 20)``

### `ts_min_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mmd_rbf_shift`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_modwt_band_corr`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_moment`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$M_k = \frac{1}{d}\sum_{j=0}^{d-1}\bigl(X_{i,t-j}-\mu\bigr)^k$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动 k 阶中心矩。

计算: 在窗口内先算均值 μ，再算 E[(X-μ)^k]。

备注: 窗口 k 阶中心矩（pandas 核 parity）

示例: ``ts_moment(close, 20, 3)``

精讲: 在窗口内先算均值 μ，再算 E[(X-μ)^k]。

### `ts_monotonicity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_motif_recurrence_count`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_adjusted_r2_prior`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_coeff`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_coeff_prior`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_coeff_stability`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_forecast_error`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_forecast_error_z`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_r2`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_r2_prior`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_resid`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multi_regression_resid_z`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multifractal_asymmetry`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multifractal_curvature`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multifractal_spectrum_width`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multiscale_entropy_slope`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multiscale_permutation_entropy_slope`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multiscale_trend_consensus`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multiscale_trend_curvature`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_multiscale_trend_dispersion`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_mutual_information`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_nearest_structural_level_distance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_negative_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_new_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_new_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_nth_pivot_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_nth_pivot_high_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_nth_pivot_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_nth_pivot_low_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_nth_value`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_nth_value}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口第 N 大或第 N 小有效值。

计算: 在 other 语义下对 panel 输入执行 `ts_nth_value`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_opening_mispricing_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ordinal_irreversibility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_overnight_intraday_cov`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_overnight_intraday_sign_agreement`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_overnight_intraday_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_partial_corr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_partial_corr}(X)$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

控制单一变量后的滚动偏相关。

计算: 在 other 语义下对 panel 输入执行 `ts_partial_corr`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_pastor_stambaugh_liquidity_gamma`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_path_efficiency`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_path_leadlag_area`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_path_signature_area`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_path_signature_depth2_norm`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pattern_symmetry`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pct`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `TS_PCT`, `m_pct_change`, `pct_change`, `returns`, `ts_return`

**公式/构造**:

$$x_{i,t} = \frac{X_{i,t}}{X_{i,t-d}} - 1$$

**可用条件**:

- 最少样本 min_periods = 1
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

相对 d 期前的收益率（百分比变化）。

计算: 对每个标的独立计算：当前值除以 d 期前的值再减 1；d=1 时等价于 returns/ pct_change。

备注: d 期变化率

示例: ``ts_pct(close, 1)``

精讲: 对每个标的独立计算：当前值除以 d 期前的值再减 1；d=1 时等价于 returns/ pct_change。

### `ts_permutation_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_permutation_transition_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_persistence_birth_dispersion`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_betti_crocker_bifurcation_score`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_persistence_diagram_shift`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_persistence_entropy_h0`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_persistence_entropy_h1`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pettitt_change_score`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pickands_tail_index`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pivot_high_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pivot_high_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pivot_high_spacing`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pivot_low_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pivot_low_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_pivot_low_spacing`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_poly2_coeff`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$X_{i,t-j} \approx a + b\tau + c\tau^2,\quad \tau=0,\ldots,d-1$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 在窗口内拟合 X ≈ a + bt + ct²，返回 c。

### `ts_poly2_forecast_error`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_poly2_forecast_error_z`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_poly2_prior_coeff`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_poly2_resid`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\varepsilon_{i,t} = Y_{i,t} - \hat{Y}_{i,t},\quad \hat{Y}=a+bX+cX^2$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 拟合 Y ≈ a + bX + cX²，输出最新观测的拟合残差。

### `ts_positive_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_prev_high`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_prev_low`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_price_delay`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_product`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\prod_{k=0}^{d-1} X_{i,t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的连乘。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算连乘；min_periods 规则以实现为准。

备注: 滚动乘积；与 pandas 审计实现语义一致。

示例: ``ts_product(1 + returns, 20)``

### `ts_pseudocount_sample_entropy`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_qn_scale`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Percentile`, `TS_QUANTILE`, `m_percentile`, `percentile`

**公式/构造**:

$$Q_q\bigl(\{X_{i,t-k}\}_{k=0}^{d-1}\bigr)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的分位数。

计算: 在长度为 d 的窗口内，对 {X_{t-d+1},…,X_t} 取 q 分位点（线性插值，与 pandas quantile 一致）。

备注: 滚动窗口分位数

示例: ``ts_quantile(returns, 20, 0.75)``

精讲: 在长度为 d 的窗口内，对 {X_{t-d+1},…,X_t} 取 q 分位点（线性插值，与 pandas quantile 一致）。

### `ts_quantile_beta_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_beta_spread_prior`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_crossing_spectral_concentration`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_kurtosis`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_range`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_regression_beta`

**标签**: `研究专用` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_regression_coeff`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_regression_coeff_prior`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_regression_resid`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_regression_slope`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_skew`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_transport_curvature`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantile_transport_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_quantilogram`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_range_expansion`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_rank`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `TS_RANK`, `m_rank`

**公式/构造**:

$$\frac{\#\{k: X_{i,t-k} \le X_{i,t}\} - 1}{d - 1}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

窗口内当前值的百分位排名。

计算: 在长度为 d 的窗口内，将 X_t 与窗口内所有值比较，输出 [0,1] 的秩分位。

备注: 滚动排名 (与m_rank相同)

示例: ``ts_rank(volume, 10)``

精讲: 在长度为 d 的窗口内，将 X_t 与窗口内所有值比较，输出 [0,1] 的秩分位。

### `ts_rank_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ratio`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ratios`

**公式/构造**:

$$R_{i,t} = \frac{P_{i,t}}{P_{i,t-1}}$$

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

价格比率（相邻 bar）。

计算: P_t / P_{t-1}，即 1 + 简单收益率。

备注: ts_ratio（Polars 桥接）

示例: ``ratios(close)``

精讲: P_t / P_{t-1}，即 1 + 简单收益率。

### `ts_realized_quarticity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recovery_fraction`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_determinism`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_diagonal_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_divergence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_laminarity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_longest_vertical_length`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_mean_diagonal_length`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_rate`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_recurrence_trapping_time`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regime_duration`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_forecast_error`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_forecast_error_z`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_in_sample_resid`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_intercept`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `intercept`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_r2`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_resid`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `rolling_residual`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_resid_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_resid_mean`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_regression_slope`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `TS_REGRESSION_SLOPE`, `ts_regression`

**公式/构造**:

$$\mathrm{ts_regression_slope}(X)$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动回归。

计算: 在 other 语义下对 panel 输入执行 `ts_regression_slope`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``ts_regression(y, x, 20, 0, 'slope')``

### `ts_regression_tstat`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_regression_tstat}(X)$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动回归斜率 t 统计量。

计算: 在 other 语义下对 panel 输入执行 `ts_regression_tstat`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_residualized_hsic`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 24
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_resistance_break`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_resistance_fit_r2`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_resistance_level`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_resistance_log_slope`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_resistance_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_response_slope_asymmetry`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_return_spectral_entropy`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_coeff`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_coeff_prior`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_forecast_error`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_forecast_error_z`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_in_sample_resid`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_ridge_regression_resid`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_predictive_resid`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ridge_regression_resid_z`

**标签**: `日线` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_robust_ema`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_robust_zscore_inclusive`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `ts_robust_zscore`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_robust_zscore_prior`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_roll_effective_spread`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_rolling_median_causal`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_rolling_sr_gaussian_mean_shift_score`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_sr_gaussian_mean_shift_score`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 12
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_roughness`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_rqa_determinism_fixed_rr`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_rqa_laminarity_fixed_rr`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_run_concentration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_run_efficiency`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_run_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_sample_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_scale_shift`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 6
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_score_rank_weighted_mean`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_semivariance_balance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_sharpe`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_sharpe}(X)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动夏普：mean/std * sqrt(ann_factor)。

计算: 在 other 语义下对 panel 输入执行 `ts_sharpe`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``ts_sharpe(returns, 60)``

### `ts_sign_cluster_index`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_sign_persistence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_signature_mahalanobis_anomaly`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_skew`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Skew`, `TS_SKEW`, `m_skew`, `skew`, `ts_skewness`

**公式/构造**:

$$\mathrm{skew}^{(d)}(X) = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的偏度。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算偏度；min_periods 规则以实现为准。

备注: 滚动偏度 (与m_skew相同)

示例: ``m_skew(returns, 20)``

### `ts_sma_cn`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_centroid`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 16
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_flatness`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_low_frequency_ratio`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_lowpass_trailing`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_peak_concentration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_spectral_quality_factor`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ssa_denoise_trailing`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ssa_prior_reconstruction_error`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_ssa_reconstruction_residual`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_staleness`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_state_age_percentile`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_state_density`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_state_entry_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_state_exit_hazard`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_state_integral`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_state_residual_life`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_std`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Std`, `TS_STD`, `m_std`, `running_std`, `std`, `std_n`, `ts_std_dev`, `ts_stddev`, `window_std`

**公式/构造**:

$$\sigma_{i,t}^{(d)} = \sqrt{\frac{1}{d-1}\sum_{k=0}^{d-1}(X_{i,t-k}-\bar{X})^2}$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的样本标准差。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算样本标准差；min_periods 规则以实现为准。

备注: 滚动标准差 (ts_std_dev的别名)

示例: ``ts_std(returns, 20)``

### `ts_std_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_std_if}(X)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动条件样本标准差。

计算: 在 other 语义下对 panel 输入执行 `ts_std_if`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_stratified_mean_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_structural_level_density`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_structural_level_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_student_t_kalman_filter`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_sum`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Sum`, `TS_SUM`, `m_sum`, `running_sum`, `sum`, `sum_n`, `window_sum`

**公式/构造**:

$$S_{i,t} = \sum_{k=0}^{d-1} X_{i,t-k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的求和。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算求和；min_periods 规则以实现为准。

备注: 滚动求和 (与m_sum相同)

示例: ``ts_sum(volume, 20)``

### `ts_sum_decay`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_sum_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_sum_if}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动条件求和。

计算: 在 other 语义下对 panel 输入执行 `ts_sum_if`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_super_smoother`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_support_break`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_support_fit_r2`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_support_level`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_support_log_slope`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_support_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_swing_amplitude`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_swing_amplitude_atr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_swing_amplitude_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_swing_duration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_swing_velocity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_tail_imbalance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_tail_mean`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_tail_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_threshold_cycle_asymmetry`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_threshold_cycle_period`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_time_since_change`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_time_slope`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Slope`, `TS_TIME_SLOPE`, `slope`

**公式/构造**:

$$\mathrm{ts_time_slope}(X)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

滚动时间斜率。

计算: 在 other 语义下对 panel 输入执行 `ts_time_slope`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``ts_time_slope(close, 20)``

### `ts_time_under_water`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_topk_mean`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `tm_top_n_avg`, `ts_top_n_avg`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_topk_std`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_top_n_std`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_topk_sum`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `TS_TOPK_SUM`, `m_top_n_sum`, `tm_top_n_sum`

**公式/构造**:

$$\sum_{j=1}^{k} X_{i,t}^{(j)},\quad X_{i,t}^{(j)}\text{ 为窗口内第 }j\text{ 大值}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内最大的 k 个值的和。

计算: 对每个窗口，将 d 个观测降序排列，取前 k 个求和。

备注: 滚动窗口内 Top-K 求和

示例: ``ts_topk_sum(volume, 20, 5)``

精讲: 对每个窗口，将 d 个观测降序排列，取前 k 个求和。

### `ts_total_variation_filter_trailing`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_transfer_entropy`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_transfer_entropy_peak_excess`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_transfer_entropy_peak_lag`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_transfer_entropy_peak_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_transition_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_transition_intensity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_trend_break_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_trend_tstat`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_trend_tstat}(X)$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动时间趋势斜率 t 统计量。

计算: 在 other 语义下对 panel 输入执行 `ts_trend_tstat`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_trimmed_mean`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_true_streak`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{ts_true_streak}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截至当前连续条件成立长度。

计算: 在 other 语义下对 panel 输入执行 `ts_true_streak`；具体边界条件（min_periods、NaN）以实现代码为准。

### `ts_turning_intensity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turning_point_ratio`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turning_rate`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_age_dispersion`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_cost_dispersion`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_cost_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_cost_entropy_vol_scaled`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_cost_mode_distance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_cost_quantile_distance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_cost_skew`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_holding_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_near_cost_mass`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_old_mass`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_profit_share`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_turnover_reference_price`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_two_state_regime_probability`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_upper_partial_moment`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_upper_tail_coexceedance_probability`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_upper_tail_dependence`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_upside_deviation`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_valid_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_value_at_argextreme`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_var`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `Var`, `m_var`, `var`

**公式/构造**:

$$s_{i,t}^{2(d)} = \frac{1}{d-1}\sum_{k=0}^{d-1}(X_{i,t-k}-\bar{X})^2$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动窗口内的样本方差。

计算: 对每个标的，在长度为 d 的窗口 {t-d+1,…,t} 上计算样本方差；min_periods 规则以实现为准。

备注: 滚动方差

示例: ``m_var(returns, 20)``

### `ts_variance_ratio_proxy`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_variance_ratio`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_variance_ratio_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_variogram_slope`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vector_path_curvature`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vector_path_efficiency`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vector_self_intersection_rate`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vector_state_local_density`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vector_state_mahalanobis`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vector_turning_coherence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_acceleration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_clustering`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_of_vol`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_pvariation_roughness`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_pvariation_scaling_exponent`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_scaling_break`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_shift_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_vol_term_structure`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wasserstein_shift`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 6
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wavelet_energy_slope`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wavelet_entropy`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wavelet_high_frequency_ratio`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wavelet_low_frequency_ratio`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wavelet_lowpass_reconstruct`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_wavelet_shrinkage_trailing`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_downside_deviation`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_weighted_semivariance_sqrt`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_drawdown_area`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_expected_shortfall`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_permutation_entropy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_semivariance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_standardized_moment`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_weighted_time_centroid`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_zero_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ts_zscore`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `m_zscore`

**公式/构造**:

$$Z_{i,t} = \frac{X_{i,t} - \mu_{i,t}^{(d)}}{\sigma_{i,t}^{(d)}}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

滚动 Z 分数（窗口内标准化）。

计算: 在长度为 d 的窗口内计算均值 μ 与标准差 σ，再对当前值标准化。

备注: 滚动Z-Score (与m_zscore相同)

示例: ``ts_zscore(close, 20)``

精讲: 在长度为 d 的窗口内计算均值 μ 与标准差 σ，再对当前值标准化。

## 均线/MA 族（6 个）

### `ALMA`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `DEMA`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `HMA`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `KAMA`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{KAMA}_t = \mathrm{KAMA}_{t-1} + SC\cdot(P_t - \mathrm{KAMA}_{t-1})$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

考夫曼自适应均线 KAMA。

计算: 按效率比 ER 自适应平滑系数。

备注: 考夫曼自适应移动平均

示例: ``KAMA(close, 10)``

### `TEMA`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `WMA`

**标签**: `扩展` · `时序` · `PIT 安全` · 别名: `ts_wma`, `wma`

**公式/构造**:

$$\mathrm{WMA}_t = \frac{\sum_{k=0}^{d-1}(d-k)P_{t-k}}{\sum_{k=1}^{d}k}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

加权移动平均。

计算: 线性递增权重，近期权重更大。

示例: ``WMA(close, 10)``

精讲: 线性递增权重，近期权重更大。

## 技术指标 (MACD/RSI/ATR/布林等)（51 个）

### `ADX`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_adx`

**公式/构造**:

$$\mathrm{ADX} = \mathrm{EMA}(\mathrm{DX}),\quad \mathrm{DX}=\frac{|\mathrm{DI}^+-\mathrm{DI}^-|}{\mathrm{DI}^++\mathrm{DI}^-}\times 100$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

平均趋向指数 ADX。

计算: 基于 +DI/-DI 与 DX 的平滑（TA-Lib 标准定义）。

备注: 平均趋向指数

示例: ``ADX(high, low, close, 14)``

精讲: 基于 +DI/-DI 与 DX 的平滑（TA-Lib 标准定义）。

### `atr_acceleration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `atr_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `atr_percentile`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `atr_short_long_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ATR_WILDER`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_atr_wilder`

**公式/构造**:

$$ATR_t = EWM_\alpha(TR),\quad \alpha = 1/w$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

Wilder 平均真实波幅。

计算: TR 用 alpha=1/window 的 EWM 平滑。

备注: Wilder 平滑 ATR

示例: ``ATR_WILDER(high, low, close, 14)``

精讲: TR 用 alpha=1/window 的 EWM 平滑。

### `atr_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `bollinger_pct_b`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `bollinger_width`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `CMF`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `CMO`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `DMI_minus`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `DMI_plus`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_breakout_down`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_breakout_up`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_channel_position`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_lower`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_mid`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_position`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_upper`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `donchian_width_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ForceIndex`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `keltner_breakout_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `keltner_compression`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `keltner_width_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `KeltnerLower`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `KeltnerMid`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `KeltnerPosition`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `KeltnerUpper`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `MACD_hist`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{HIST} = \mathrm{DIF} - \mathrm{DEA}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

MACD 柱状图。

计算: DIF - DEA。

备注: MACD 柱

示例: ``MACD_hist(close, 12, 26, 9)``

精讲: DIF - DEA。

### `MACD_line`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `MACD`

**公式/构造**:

$$\mathrm{DIF} = \mathrm{EMA}_{12}(P) - \mathrm{EMA}_{26}(P)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

MACD DIF 线。

计算: 快 EMA 减慢 EMA。

备注: MACD 线

示例: ``MACD_line(close, 12, 26)``

精讲: 快 EMA 减慢 EMA。

### `MACD_signal`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{DEA} = \mathrm{EMA}_9(\mathrm{DIF})$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

MACD 信号线 DEA。

计算: 对 DIF 做 EMA。

备注: MACD 信号线

示例: ``MACD_signal(close, 12, 26, 9)``

精讲: 对 DIF 做 EMA。

### `MFI`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `PPO`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `PPO_hist`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `PPO_signal`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `PSAR`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `psar_days_since_flip`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `psar_direction`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `psar_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `psar_flip`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `RSI_WILDER`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ts_rsi_wilder`

**公式/构造**:

$$RSI = 100 - \frac{100}{1 + \frac{EMA(gain)}{EMA(loss)}}$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

Wilder 相对强弱指数。

计算: gain/loss 用 alpha=1/window 的 EWM 平滑，min_periods=window。

备注: Wilder 平滑 RSI

示例: ``RSI_WILDER(close, 14)``

精讲: gain/loss 用 alpha=1/window 的 EWM 平滑，min_periods=window。

### `Supertrend`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `supertrend_days_since_flip`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `supertrend_direction`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `supertrend_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `supertrend_flip`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `SupertrendDirection`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `TSI`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `TSI_signal`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `UltimateOscillator`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

## 事件/条件/门控（44 个）

### `and_`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a \land b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：逻辑与。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: 逻辑与

### `event_abnormal_return_past`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_active_count`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_allan_factor`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_allan_log_mean`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_allan_scaling_slope`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_arithmetic_return_sum`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_cluster_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_cluster_duration`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_cluster_mean_size`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_cluster_score`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_cumulative_return_past`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_decay_asof`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `event_decay`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_decay_window`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_direction_imbalance`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_direction_persistence`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_fano_excess`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_fano_factor`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_flip_density`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_frequency`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_hawkes_branching_ratio_proxy`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `event_hawkes_branching_ratio`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- 内含滞后 lag = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_historical_response_mean`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_historical_response_sign_balance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_interval_mark_coupling`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_interval_memory`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_level_survival_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_local_variation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_log_return_sum`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_mark_autocorr`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_rate_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_recency_z`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_refractory`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_response_decay_rate`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_response_dispersion`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_response_effective_events`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_response_overlap_ratio`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_response_peak_lag`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_response_reversal_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_return_since_last`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `event_compounded_return`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `event_streak`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `not_`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\lnot a$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：逻辑非。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: 逻辑非

### `or_`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a \lor b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：逻辑或。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: 逻辑或

### `trade_when`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\alpha_t^{\mathrm{out}} = \begin{cases}\alpha_t & \mathrm{trigger}_t\\ \mathrm{exit}_t & \text{否则}\end{cases}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: trigger 为真时输出 alpha，否则输出 exit_（或 hold 逻辑以实现为准）。

### `where`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `IIF`, `WHERE`, `if`, `if_else`, `iif`

**公式/构造**:

$$\begin{cases}a & c\\ b & \neg c\end{cases}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

条件选择。

计算: condition 为真取 a，否则 b（同 if_else）。

备注: 条件选择（与if_else相同）

示例: ``where(volume > 1000, 1, 0)``

## 量价/流动性（110 个）

### `abnormal_turnover`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `abnormal_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `abs_return_volume_corr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `adv`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `amihud_illiquidity`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_down_volume_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_up_volume_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `average_turnover`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `dollar_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `dollar_volume_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `down_volume_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_turnover`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `float_share_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `free_float_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `free_float_share_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `free_float_turnover`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_float_concentration_gap`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_weight_gap_to_free_float`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_amihud`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_interval_illiquidity`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_interval_volume_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_interval_vwap_deviation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_liquidity_resilience_curve_fit`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_longest_above_vwap_streak`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_longest_below_vwap_streak`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_price_vwap_max_negative_excursion`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_price_vwap_max_positive_excursion`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_segment_volume_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_segment_vwap_deviation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_slot_volume_surprise`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_smart_money_vwap_ratio`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_vwap`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_tail_volume_share`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_time_above_vwap`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_at_price_profile`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_imbalance`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_price_alignment`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_profile_cosine`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_profile_jsd`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_profile_peak_geometry`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_profile_supply_structure`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_volume_profile_value_area`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_above_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_cross_count`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_path_curvature`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_path_curvature_pct`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_path_slope`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_path_slope_pct`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_vwap_reversion_speed`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volume_clock_path_efficiency`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volume_clock_roughness`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_vwap_deviation`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_amihud_own`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_herfindahl_volume`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_liquidity_decay_base`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_market_share_turnover`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_mean_variance_turnover`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_own_volume_share`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_ret_volume_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_turnover_distribution_skew`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_volume_amihud_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_volume_amplification`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lf1_volume_concentration`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_volume_adjusted_momentum`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_volume_flow_regime`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_volume_imbalance`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `open_to_vwap_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `price_turnover_divergence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `price_volume_divergence`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `real_turnover_rate`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{TO}_t = \frac{\mathrm{volume}_t}{\mathrm{float}_t}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

真实流通盘换手率。

计算: 成交量除以有效流通股本。

备注: 真实换手率 volume/float

精讲: 成交量除以有效流通股本。

### `relative_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `return_per_turnover`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `return_turnover_beta`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `return_volume_beta`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `return_volume_corr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `rolling_vwap`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `signed_dollar_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `signed_volume`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `signed_volume_imbalance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_l1_turnover_prox`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sv_signed_volume_volatility`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `true_turnover_rate`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_acceleration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_adjusted_volatility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_autocorr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_chip_age_cost_surface`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_chip_overhang_surface`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_momentum`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_shock`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_volatility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `turnover_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `up_down_volume_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `up_volume_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vax_liquidity_adjusted_return`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vax_liquidity_penalty_exposure`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vax_ret_per_liquidity_unit`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_acceleration`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_autocorr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_momentum`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_price_range_density`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `volume_to_range`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_shock`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_volatility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_weighted_momentum`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_weighted_return`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `volume_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vwap_deviation`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vwap_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vwap_premium_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vwap_slope_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vwap_to_close_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

## 波动/风险估计（60 个）

### `aq1_cash_flow_volatility`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `coskewness_to_market`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\frac{E\bigl[(r_i-\mu_i)(r_m-\mu_m)^2\bigr]}{\sigma_i\,\sigma_m^2}$$

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

精讲: E[(r_i-μ_i)(r_m-μ_m)²] / (σ_i σ_m²)。

### `fin_earnings_cash_gap_volatility`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_growth_volatility`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `garman_klass_vol`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `hfl_variance_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `idio_skew`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{skew}(\hat\varepsilon)$$

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 窗口内 CAPM 残差的三阶标准化矩。

### `idio_vol`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\sigma_\varepsilon = \mathrm{std}(\hat\varepsilon)$$

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 窗口内 CAPM 回归残差的标准差。

### `intra_bipower_variation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_continuous_variance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_drawdown_depth`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_drawdown_duration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_drawdown_recovery_half_life`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_idiosyncratic_kurtosis`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_idiosyncratic_kurtosis_ex_self`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_idiosyncratic_skewness`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_idiosyncratic_skewness_ex_self`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_idiosyncratic_variance`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_idiosyncratic_variance_ex_self`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_interval_realized_variance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_variation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_max_drawdown`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_negative_jump_variation`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_negative_tail_variation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_positive_jump_variation`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_positive_tail_variation`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_kurtosis`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_semivariance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_skewness`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_variance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_segment_realized_vol`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_signed_tail_variation_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_slot_volatility_surprise`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_medrv`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_realized_power_variation`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_realized_semivariance_balance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volatility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volatility_concentration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volatility_entropy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volatility_signature_slope`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_volatility_time_centroid`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lqtp_historical_cvar`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `overnight_volatility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pca_resid_vol`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `parkinson_vol`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `range_volatility`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_distribution_excess_kurtosis`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_distribution_pearson_kurtosis`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `relation_distribution_kurtosis`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_distribution_skew`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `rogers_satchell_vol`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vr1_ewma_range_vol`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vr1_garman_klass_ext`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vr1_parkinson_close_scale`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_dispersion_vol`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_downside_vol_share`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_long_short_vol_beta`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_vol_acceleration`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_vol_level_score`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_vol_of_vol`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `yang_zhang_vol`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

## 分钟级 intraday（20 个）

### `intraday_activity_duration_curvature`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_barrier_approach_acceleration`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_bvc_imbalance`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_impact_asymmetry`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_impact_beta`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_impact_decay_rate`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_jump_test_stat`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_minrv`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_profile_pca_residual`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_profile_phase_shift`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_profile_surprise_energy`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_quantile_curve_pca_residual`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_quantile_curve_pca_score`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_return_wasserstein_shift`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 30
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_rv_signature_curvature`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_rv_signature_slope`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_session_shape_novelty`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_subsampled_rv_dispersion`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intraday_wasserstein_pair_distance`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `session_event_recovery_score`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

## 基本面/财务/PIT（168 个）

### `altman_z_score`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `book_to_price`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `earnings_yield`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_accrual_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_acquisition_cash_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_actual_expectation_divergence`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_announcement_lag`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_applicability_mask`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_average_balance`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_beat_streak`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_borrowing_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cagr`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_capex_growth`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_capex_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cash_burn_runway`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cash_conversion`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cash_earnings_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cash_sales_divergence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cashflow_persistence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_common_size`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_component_score`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_comprehensive_income_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_contract_asset_growth`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_contract_asset_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_contract_asset_liability_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_contract_liability_growth`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_contract_liability_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_core_earnings_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_cv`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_days_since_expectation_revision`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_days_since_update`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_debt_repayment_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_debt_service_coverage_proxy`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_deferred_tax_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_delta_noa`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_diff`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_discontinued_operation_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_divergence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_earnings_persistence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_earnings_smoothness`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_equity_capital_growth`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expectation_dispersion`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expectation_revision`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expectation_revision_count`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expectation_revision_magnitude`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expectation_revision_pct`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expectation_revision_speed`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_expense_sales_divergence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_fair_value_income_dependence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_financing_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_fundamental_strength_coverage`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_fundamental_strength_score`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_goodwill_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_goodwill_risk_score`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_growth`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_growth_acceleration`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_growth_change`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_growth_persistence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_growth_stability`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_impairment_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_interest_coverage_proxy`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_inventory_sales_divergence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_investment_income_dependence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_lag`

**标签**: `日线` · `fundamental_period` · `PIT 安全` · 别名: `report_lag`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_lease_asset_liability_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_lease_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_log_change`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_margin_persistence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_mean_abs_deviation`

**标签**: `扩展` · `fundamental_period` · `PIT 安全` · 别名: `fin_mad`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_median_abs_deviation`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_minority_profit_share`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_miss_streak`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_monotonicity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_negative_streak`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_net_borrowing_cashflow`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_net_debt_issuance`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_noncore_income_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_oci_to_equity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_other_earnings_dependence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_pct_change`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_percentile_history`

**标签**: `日线` · `fundamental_period` · `PIT 安全` · 别名: `report_rank`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_percentile_vs_prior_history`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_period_restated`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_period_revision_age`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_period_revision_count`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_positive_streak`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_qoq`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_quarter_from_cumulative`

**标签**: `日线` · `fundamental_period` · `PIT 安全` · 别名: `report_single_quarter`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_range`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_rd_capitalization_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_rd_total_intensity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_receivable_sales_divergence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_restated_flag`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_revision_count`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_revision_delta`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_revision_direction`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_revision_magnitude`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_revision_pct`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_roe_cash_gap`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_seasonal_percentile`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_seasonal_zscore`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_sign_change_count`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_stability`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_staleness`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查` · 别名: `report_age`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_std`

**标签**: `日线` · `fundamental_period` · `PIT 安全` · 别名: `report_rolling_std`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_surprise`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_surprise_event_percentile`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_surprise_event_zscore`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_surprise_zscore`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查` · 别名: `report_surprise_to_trend`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_total_operating_accruals`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_trend_acceleration`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_trend_r2`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_trend_slope`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_trend_tstat`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_ttm`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_ttm_cumulative`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_ttm_quarterly`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_working_capital_accruals`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_working_capital_change`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_yoy`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_zscore_history`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fin_zscore_vs_prior_history`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fundamental_staleness`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_class_entropy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_class_js_shift`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_common_holding_peer_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_company_ownership_hhi`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_concentration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_concentration_acceleration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_concentration_change`

**标签**: `研究专用` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_concentration_slope`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_count_change_rate`

**标签**: `研究专用` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_disclosure_count`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_disclosure_coverage`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_entry_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_exit_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_freeze_concentration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_freeze_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_id_matched_churn`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_id_matched_entry_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_id_matched_exit_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_id_overlap_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_locked_share_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_nature_entropy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_net_entry_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_observed_topk_hhi`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_peer_return_breadth`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_pledge_change`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_pledge_churn`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_pledge_concentration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_pledge_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_pledged_holder_count`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_rank_stability`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_share_weighted_rank_migration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_shareholder_network_centrality`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_shareholder_overlap_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_topk_share_sum`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `holder_weighted_churn`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `period_average`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `period_cagr`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `period_change`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `period_lag`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

$$\mathrm{period_lag}(X)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

按已披露报告期进行滞后。

计算: 在 other 语义下对 panel 输入执行 `period_lag`；具体边界条件（min_periods、NaN）以实现代码为准。

### `period_stability`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `quarter_from_cumulative`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ttm_from_cumulative`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ttm_from_quarterly`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `yoy_by_period`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

## A 股特有（21 个）

### `ashare_days_since_limit_down`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_days_since_limit_up`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_failed_limit_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_asymmetry`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_distance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_down_streak`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_down_touch`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_event_density`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_failed`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_one_price`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_open_down_streak`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_open_failed`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_open_up_streak`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_touch_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_up_streak`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_limit_up_touch`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_one_price_limit_streak`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_open_at_upper_limit`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ashare_suspension_episode_length`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `limit_down_close`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `limit_down_state`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `limit_up_close`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `limit_up_state`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

## 数学/安全运算（25 个）

### `abs`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `ABS`

**公式/构造**:

$$|x|$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素绝对值。

计算: 对 panel 每个元素应用绝对值。

备注: 绝对值

示例: ``abs(Return)``

### `add`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a + b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素加法。

计算: 两个同形序列按位置加法。

### `cbrt`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\sqrt[3]{x}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素立方根。

计算: 对 panel 每个元素应用立方根。

备注: 立方根

示例: ``cbrt(volume)``

### `ceil`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\lceil x\rceil$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素向上取整。

计算: 对 panel 每个元素应用向上取整。

备注: 向上取整

示例: ``ceil(price)``

### `clip`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `CLIP`, `cap`, `clamp`

**公式/构造**:

$$\tilde{x} = \min\bigl(\max(x, lo), hi\bigr)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截断到区间 [lo, hi]；别名 cap、clamp。

计算: 逐元素 clip，超出上下界的值截断。

备注: 裁剪到 [lo, hi]

示例: ``cap(x, -3, 3)``

精讲: 逐元素 clip，超出上下界的值截断。

### `divide`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a / b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素除法。

计算: 两个同形序列按位置除法。

### `exp`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$e^x$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素自然指数。

计算: 对 panel 每个元素应用自然指数。

备注: 指数

示例: ``exp(Return)``

### `floor`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\lfloor x\rfloor$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素向下取整。

计算: 对 panel 每个元素应用向下取整。

备注: 向下取整

示例: ``floor(price)``

### `inverse`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `inv`, `reciprocal`

**公式/构造**:

$$1/x$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素倒数。

计算: 对 panel 每个元素应用倒数。

备注: 倒数

### `log`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `LOG`, `ln`

**公式/构造**:

$$\ln x$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素自然对数。

计算: 对 panel 每个元素应用自然对数。

备注: 自然对数

示例: ``log(close)``

### `log_abs`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\ln|x|$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素绝对值对数。

计算: 对 panel 每个元素应用绝对值对数。

备注: Polars log_abs

示例: ``log_abs(x)``

### `maximum`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `fmax`

**公式/构造**:

$$\mathrm{maximum}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素 max(x,y)。

计算: 在 other 语义下对 panel 输入执行 `maximum`；具体边界条件（min_periods、NaN）以实现代码为准。

### `minimum`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `fmin`

**公式/构造**:

$$\mathrm{minimum}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素 min(x,y)。

计算: 在 other 语义下对 panel 输入执行 `minimum`；具体边界条件（min_periods、NaN）以实现代码为准。

### `multiply`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a \cdot b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素乘法。

计算: 两个同形序列按位置乘法。

### `neg`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `negate`, `reverse`

**公式/构造**:

$$-x$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素取负。

计算: 对 panel 每个元素应用取负。

备注: 取负

示例: ``neg(close)``

### `power`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `POWER`, `pow`

**公式/构造**:

$$a^b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素幂。

计算: 两个同形序列按位置幂。

备注: 幂

示例: ``pow(close, 2)``

### `protected_div`

**标签**: `内部` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

$$\begin{cases}a/b & |b|>\epsilon\\ \mathrm{NaN} & \text{否则}\end{cases}$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

安全除法。

计算: 分母接近 0 或 NaN 时返回 NaN 或默认值，避免 inf。

备注: protected_div（Polars 桥接）

示例: ``x / 20``

精讲: 分母接近 0 或 NaN 时返回 NaN 或默认值，避免 inf。

### `round`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `ROUND`

**公式/构造**:

$$\mathrm{round}(x)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素四舍五入。

计算: 对 panel 每个元素应用四舍五入。

备注: 四舍五入

示例: ``round(price, 2)``

### `safe_div_null`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `div_or_null`, `safe_div`

**公式/构造**:

$$\mathrm{safe_div_null}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

safe_div_null（Polars 桥接）。

计算: 在 other 语义下对 panel 输入执行 `safe_div_null`；具体边界条件（min_periods、NaN）以实现代码为准。

### `sign`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `SIGN`

**公式/构造**:

$$\mathrm{sgn}(x)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素符号函数。

计算: 对 panel 每个元素应用符号函数。

备注: 符号函数

示例: ``sign(Return)``

### `signed_log`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{sgn}(x)\ln|x|$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

保号对数。

计算: sgn(x)·ln|x|。

备注: 符号对数

示例: ``signed_log(returns)``

### `signed_sqrt`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{sgn}(x)\sqrt{|x|}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

保号开方。

计算: sign(x)·√|x|。

备注: signed_sqrt（Polars 桥接）

精讲: sign(x)·√|x|。

### `sqrt`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `SQRT`

**公式/构造**:

$$\sqrt{x}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素平方根。

计算: 对 panel 每个元素应用平方根。

备注: 平方根

示例: ``sqrt(close)``

### `subtract`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a - b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素减法。

计算: 两个同形序列按位置减法。

### `tanh`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\tanh x$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素双曲正切。

计算: 对 panel 每个元素应用双曲正切。

备注: tanh

示例: ``tanh(x)``

## 比较/逻辑/判断（12 个）

### `coalesce`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `COALESCE`

**公式/构造**:

$$\mathrm{coalesce}(a,b,\ldots) = \min\{v: v \text{ 非 NaN}\}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

返回第一个非 NaN 值。

计算: 从左到右扫描参数，取首个有限值。

备注: Polars coalesce

精讲: 从左到右扫描参数，取首个有限值。

### `eq`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a = b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：等于。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: Polars eq

### `fillna_const`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\tilde{x} = \begin{cases}x & \text{有效}\\ c & \text{NaN}\end{cases}$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

常量填充缺失。

计算: NaN 替换为指定常数。

备注: 常量填充

示例: ``fillna_const(close, 0)``

### `ge`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a \ge b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：大于等于。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: Polars ge

### `gt`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a > b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：大于。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: Polars gt

### `is_finite`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathbf{1}\{|x|<\infty\}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

是否有限数。

计算: 非 NaN 且非 inf。

备注: 是否有限

示例: ``is_finite(value)``

### `is_infinite`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `IS_INFINITE`, `is_inf`

**公式/构造**:

$$\mathrm{is_infinite}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

是否为 ±Inf。

计算: 在 other 语义下对 panel 输入执行 `is_infinite`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``is_infinite(value)``

### `is_not_null`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `IS_NOT_NULL`

**公式/构造**:

$$\mathrm{is_not_null}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

是否为非 NULL。

计算: 在 other 语义下对 panel 输入执行 `is_not_null`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``is_not_null(close)``

### `is_null`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `IS_NULL`

**公式/构造**:

$$\mathrm{is_null}(X)$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

是否为 NULL。

计算: 在 other 语义下对 panel 输入执行 `is_null`；具体边界条件（min_periods、NaN）以实现代码为准。

示例: ``is_null(close)``

### `le`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a \le b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：小于等于。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: Polars le

### `lt`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a < b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：小于。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: Polars lt

### `ne`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$a \ne b$$

**可用条件**:

- DSL 白名单: **在**
- 空值策略: propagate ｜ 值域: all_numeric ｜ 溢出: ieee_propagate
- 状态: production ｜ 生产准入: allowed
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素比较：不等于。

计算: 两个序列按位置比较，输出布尔或 0/1。

备注: Polars ne

## 其他（535 个）

### `a_share_cap_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `accounting_comparability_score`

**标签**: `扩展` · `fundamental_period` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 12
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `acos`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\arccos x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素反余弦。

计算: 对 panel 每个元素应用反余弦。

备注: 反余弦

示例: ``acos(value)``

### `acos_bounded`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `aq1_accrual_ratio_dispersion`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `aq1_accrual_stability`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `aq1_cash_conversion_strength`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `aq1_working_capital_accrual`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `arg`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\arg(z)$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`

**功能讲解**:

复数辐角（弧度）。

计算: angle(z)。

备注: 相位

示例: ``arg(x)``

### `asin`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\arcsin x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素反正弦。

计算: 对 panel 每个元素应用反正弦。

备注: 反正弦

示例: ``asin(value)``

### `asin_bounded`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `atan`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\arctan x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素反正切。

计算: 对 panel 每个元素应用反正切。

备注: 反正切

示例: ``atan(value)``

### `atan2`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\theta = \mathrm{atan2}(y,x)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

双参数反正切。

计算: atan2(y,x)，象限正确。

备注: atan2

示例: ``atan2(y, x)``

### `baseline_scaled_wasserstein_distance`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `benchmark_excess_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `benchmark_relative_price`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `beta_divergence_pct`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `beta_residual_z`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `bounded_nvi`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `bounded_pvi`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `bvc_imbalance_ma`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `bvc_sign_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `calendar_day_diff`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_abs_body`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_body`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_body_percentile`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_body_position`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_body_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_body_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_body_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_close_location`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_close_strength`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_direction`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_gap`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_gap_atr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_gap_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_inside_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_lower_shadow`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_lower_shadow_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_lower_shadow_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_overlap_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_pattern_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_range`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_range_atr`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_range_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_range_percentile`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_range_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_rejection_lower`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_rejection_upper`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_upper_shadow`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_upper_shadow_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_upper_shadow_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candle_wick_balance`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `candlestick_pattern`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `capital_change_age`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `capital_change_magnitude`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cash_flow_lifecycle_stage`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `category_age`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `category_age_lower_bound`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `category_frequency`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `category_transition_rate`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `category_transition_surprise`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_dark_cloud_cover`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_doji`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_dragonfly_doji`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_engulfing`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_evening_star`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- 内含滞后 lag = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_gravestone_doji`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_hammer`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_hanging_man`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_harami`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_harami_cross`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_inside_bar`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_inverted_hammer`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_marubozu`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_morning_star`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- 内含滞后 lag = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_outside_bar`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_piercing`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_shooting_star`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_spinning_top`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_three_black_crows`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- 内含滞后 lag = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_three_white_soldiers`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- 内含滞后 lag = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_tweezer_bottom`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cdl_tweezer_top`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ChaikinOscillator`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `chikou_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `choppiness_index`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `circulating_cap_ratio_change`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `circulating_cap_unlock_proxy`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `composition_aitchison_distance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `composition_clr_component`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `composition_entropy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `composition_ilr_balance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `composition_js_divergence`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `composition_normalized_entropy`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `consolidation_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `consolidation_range_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `constant`

**标签**: `内部` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `CoppockCurve`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `corwin_schultz_spread`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cos`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\cos x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素余弦。

计算: 对 panel 每个元素应用余弦。

备注: 余弦

示例: ``cos(angle)``

### `cos_phase`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cosh`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\cosh x$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素双曲余弦。

计算: 对 panel 每个元素应用双曲余弦。

备注: Polars cosh

示例: ``cosh(x)``

### `cot`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\cot x = 1/\tan x$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

余切。

计算: 1/tan(x)。

备注: Polars cot

示例: ``cot(angle)``

### `cross_event`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_monthly_lag_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_seasonal_relative_rank`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_seasonal_residual_smoothness`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_week_cycle_phase`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_weekday_anomaly`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_weekday_effect_strength`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_weekday_lag_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_weekday_lag_zscore`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `cs1_weekly_harmonic_power`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `csc`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\csc x = 1/\sin x$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

余割。

计算: 1/sin(x)。

备注: Polars csc

示例: ``csc(angle)``

### `cube`

**标签**: `遗留` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `date_diff_days`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `dema_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `digital_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$C_t = \begin{cases}C_{t-1}+1 & \left|\frac{X_t}{X_{t-1}}-1\right|\le\theta\\ 0 & \text{否则}\end{cases}$$

**可用条件**:

- 最少样本 min_periods = 1
- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

精讲: 若 |X_t/X_{t-1}-1| ≤ threshold 则计数 +1，否则归零；仅保留 ≥ run 的片段。

### `directional_change_extent`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `directional_change_state`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `DX`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `EaseOfMovement`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `efficiency_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ElderRay`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ema_crossover`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ema_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ep1_earnings_autocorr`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ep1_earnings_consistency`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ep1_earnings_surprise_decay`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ep1_roa_stability`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_burst_duration`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_cross_events_spacing`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_event_rate_decay_slope`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_event_response_amplitude`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_event_window_return_gradient`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_half_life_decay_count`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_marked_event_decay`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_post_event_hazard`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_recency_decay`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `erd_sign_consistent_decay`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `es1_earnings_cv`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `es1_earnings_mean_reversion_speed`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `es1_earnings_smoothness`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `es1_negative_earnings_streak`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `es1_revenue_earnings_divergence`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ewm_corr`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

$$\rho_t^{\mathrm{ewm}} = \mathrm{EWM\_corr}(X,Y)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

指数加权相关。

计算: span/decay 参数的 EWM corr。

备注: EWM 相关

示例: ``ewm_corr(close, volume, 20)``

### `ewm_cov`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

$$\mathrm{EWM\_cov}_t(X,Y)$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

指数加权协方差。

计算: span/decay 参数的 EWM cov。

备注: EWM 协方差

示例: ``ewm_cov(close, volume, 20)``

### `ex_self_mad_z`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ex_self_mean_gap`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ex_self_rank_pct`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ex_self_zscore`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `exp_neg`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$e^{-x}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素负指数衰减。

计算: 对 panel 每个元素应用负指数衰减。

备注: Polars exp_neg

示例: ``exp_neg(x)``

### `expanding_rank`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `cum_rank`

**公式/构造**:

$$R_t = \frac{\#\{s\le t: X_s\le X_t\}-1}{t-1}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

扩展秩分位。

计算: 从样本起点到 t 的 expanding rank。

备注: 扩展窗口百分位排名

示例: ``expanding_rank(close)``

### `ffill_limit`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_acceleration`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_accrual_quality`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_ar_resid_std`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_asymmetric_elasticity`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_asymmetric_timeliness`

**标签**: `扩展` · `fundamental_period` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_autocorr`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_change_direction_agreement`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_direction_consistency`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_pair_direction_agreement`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_pct_change`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_perpetual_inventory`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 8
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_regression_resid_std`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_reversal_ratio`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_rolling_std`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_sign_agreement`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_sign_consistency`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_standardized_surprise`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fiscal_true_streak`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `FisherTransform`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `fix`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{trunc}(x)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素向零取整。

计算: 对 panel 每个元素应用向零取整。

备注: Polars fix

示例: ``fix(-3.7)``

### `flex_max`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `max`

**公式/构造**:

$$\max(a,b)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素逐元素 max。

计算: 两个同形序列按位置逐元素 max。

备注: max(x,y) 或 rolling max

示例: ``max(abs(a), abs(b))``

### `flex_min`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `min`

**公式/构造**:

$$\min(a,b)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素逐元素 min。

计算: 两个同形序列按位置逐元素 min。

备注: min(x,y) 或 rolling min

示例: ``min(low, delay(close, 1))``

### `free_to_circulating_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `hfl_hurst_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `hierarchical_group_neutralize`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `high_low_spread_proxy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `hump_decay`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\Delta x_t \leftarrow \begin{cases}0 & |\Delta|<h\\ \Delta & \text{否则}\end{cases}$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

阈值衰减（抑制微小变化）。

计算: 变化幅度小于 hump 时不更新，否则按规则衰减（见实现）。

备注: 阈值衰减

示例: ``hump_decay(close, 0.05)``

精讲: 变化幅度小于 hump 时不更新，否则按规则衰减（见实现）。

### `ichimoku_cloud_position`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ichimoku_cloud_width`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ichimoku_kijun`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ichimoku_senkou_a`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ichimoku_senkou_b`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ichimoku_tenkan`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `identity`

**标签**: `内部` · `逐元素` · `PIT 安全`

**公式/构造**:

$$Y = X$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

恒等映射。

计算: 返回输入不变。

备注: Polars identity

示例: ``identity(x)``

### `index_entry_exit_event`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_event_decay`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_member`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_membership_age`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_reconstitution_churn`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_weight`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `index_weight_change`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `industry_fiscal_resid`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `industry_rolling_pca_loading`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `industry_size_neutralize`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `size_industry_neutralize`

**公式/构造**:

$$\tilde y = y-\bar y_{\mathrm{ind}},\ \tilde s=\ln(\mathrm{cap})-\overline{\ln(\mathrm{cap})}_{\mathrm{ind}},\ \varepsilon=\tilde y-\hat\alpha-\hat\beta\tilde s$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

精讲: y 与 ln(合法市值) 均先行业 demean，再对 demean 后的 size 取截面残差；等价行业 dummy + ln(mcap) 联合回归。合法市值要求 mcap>0 且有限。

### `intra_abs_return_profile_cosine`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_amount_profile_cosine`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_amount_profile_jsd`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_bar_range_deviation`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_bar_range_persistence`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_beta_asymmetry`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_close_participation`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_common_trading_intensity`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_concentration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_consolidation_quality`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_down_down_semibeta`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_down_up_semibeta`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_dynamic_stock_graph_features`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_entropy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_eod_reversal_decomposition`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_event_pre_post_contrast`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_event_window_reduce`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_extreme_bar_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_high_low_affinity`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_high_time`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_impulse_event_detector`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_industry_lead_lag_ex_self`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_interval_amount_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_interval_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_clustering`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_concentration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_count`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_first_time`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_last_time`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_jump_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_kyle_lambda_proxy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_limit_duration`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_limit_first_hit_time`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_limit_pre_hit_pressure_profile`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_limit_reopen_count`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_local_conditional_entropy`

**标签**: `扩展` · `session_intraday` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 10
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_low_time`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_lunch_gap_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_market_lead_lag_ex_self`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_market_model_r2`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_market_model_r2_ex_self`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_max_drawup`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_multiresolution_resample_reduce`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_neighbor_event_class`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_path_efficiency`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_post_impulse_response`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_price_delay`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_probe_outcome_score`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_profile_earth_mover_distance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_range_gap_flag`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_beta`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_beta_ex_self`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_correlation`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_correlation_ex_self`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_realized_quarticity`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_response_curve_features`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_return_activity_corr`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_return_profile_cosine`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_round_price_barrier_response`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_round_price_clustering_share`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_same_slot_momentum`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_same_slot_reversal`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_same_slot_zscore`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_segment_amount_share`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_segment_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_session_boundary_jump`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_session_mean_reversion`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_session_return_asymmetry`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_signed_imbalance_proxy`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_signed_jump_ratio`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查` · 别名: `intraday_signed_jump_balance`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_signed_return_profile_cosine`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_slice_mask_pair_reduce`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_slice_mask_reduce`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_slot_amount_surprise`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_count`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_dwell_stats`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_follow_beta`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_follow_corr`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_follow_ratio`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_interval_moment`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_pair_same_slot_corr`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_sum`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_state_transition_entropy`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_supply_absorption_score`

**标签**: `扩展` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_tail_event_count`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_tripower_quarticity`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_up_down_semibeta`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_up_up_semibeta`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_ute_high`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `intra_ute_low`

**标签**: `日线` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `is_nan`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `IS_NAN`

**公式/构造**:

$$\mathbf{1}\{\mathrm{isNaN}(x)\}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

是否为 NaN。

计算: 逐元素 NaN 检测。

备注: 是否为 IEEE NaN

示例: ``is_nan(close)``

### `kama_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `lerp`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$L = a + f(b-a),\ f\in[0,1]$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

线性插值。

计算: a + f(b-a)。

示例: ``lerp(a, b, 0.5)``

### `listing_age`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `log10`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\log_{10} x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素常用对数。

计算: 对 panel 每个元素应用常用对数。

备注: log10

示例: ``log10(price)``

### `log2`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\log_2 x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素以 2 为底对数。

计算: 对 panel 每个元素应用以 2 为底对数。

备注: Polars log2

示例: ``log2(volume)``

### `log_positive_or_nan`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_corr_momentum`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_cs_momentum_dispersion`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_momentum_regime`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_momentum_speed_change`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_momentum_stability`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_momentum_strength`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_rank_momentum_gap`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `m1_ranked_momentum`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ma_slope_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `market_cap_free_cap_gap`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `micro_bvc_vpin`

**标签**: `研究专用` · `session_intraday` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 20
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

session_intraday算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `multi_index_entry_intensity`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `NATR`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `negative_event_age`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `negative_event_rate`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `nonfinite_to_num`

**标签**: `研究专用` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_abs_imbalance_trend`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_dominant_direction`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_imbalance_agreement`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_imbalance_cv`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_imbalance_persistence`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_reversal_rate`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ofi_zero_flow_balance`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `ohlc_corwin_schultz_spread`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `open_close_return`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `intraday_return`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `overnight_return`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 内含滞后 lag = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_async_beta_ex_self`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_factor_pocket_strength`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_mixture_of_experts_score`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_peer_graph_aggregate`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_predictability_mosaic_score`

**标签**: `扩展` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_regime_conditioned_forecast`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_elastic_net_forecast`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pca_explained_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pca_loading`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pca_resid`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pca_resid_momentum`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pcr_forecast`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `panel_rolling_pls_forecast`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_123_bear`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_123_bull`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_ascending_triangle`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_bear_flag`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_bear_pennant`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_breakdown_retest`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_breakout_retest`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_broadening`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_bull_flag`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_bull_pennant`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_cup`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_cup_handle`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_descending_triangle`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_double_bottom`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_double_top`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_falling_channel`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_falling_wedge`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_head_shoulders`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_inverse_head_shoulders`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_rectangle`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_rising_channel`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_rising_wedge`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_rounding_bottom`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_rounding_top`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_sym_triangle`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_triple_bottom`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `pattern_triple_top`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `piotroski_f_score`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `piotroski_f_score_tolerant`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `piotroski_observed_count`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `piotroski_partial_score`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `positive_event_age`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `positive_event_rate`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `price_impact`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `price_spread_deviation`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\frac{X_{i,t}}{\bar{X}_{i,t}^{(d)}} - 1$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

相对窗口均值的偏离率。

计算: 当前值除以 d 期滚动均值再减 1。

备注: price_spread_deviation（Polars colwise 核）

示例: ``price_spread_deviation(close, 20)``

精讲: 当前值除以 d 期滚动均值再减 1。

### `PVO`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `PVO_hist`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `PVO_signal`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `QQE`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `reg_forecast_error_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `reg_r2_trailing`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `reg_residual_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `reg_slope_tstat`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_category_share`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_category_signed_contribution`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_concentration_acceleration`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_diffusion_score`

**标签**: `扩展` · `分组` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_distinct_count`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_entropy`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_entropy_change`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_entry_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_exit_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_hhi`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_hhi_change`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_jaccard`

**标签**: `扩展` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_overlap_ratio`

**标签**: `扩展` · `截面` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_peer_weighted_mean_ex_self`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_rank_entity_mobility`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_rank_mobility`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_rank_weighted_sum`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_share_mobility`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_topk_concentration`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_topk_sum`

**标签**: `日线` · `截面` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

截面算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_weighted_change`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relation_weighted_std_ex_self`

**标签**: `日线` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `relative_strength_group_pct`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_benford_js_divergence`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_change_breadth`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_change_coherence`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_filing_delay_surprise`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_revision_magnitude`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_rolling_mean`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `report_yoy_lag`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `residual_momentum_capm`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\sum_{s=t-d+1}^{t}\hat\varepsilon_{i,s},\quad r_i=\alpha+\beta r_m+\varepsilon$$

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 窗口内 r_i = α + β r_m + ε，输出 Σε。

### `revision_delta`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `roll_spread_proxy`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `rolling_adl_flow`

**标签**: `日线` · `时序` · `PIT 安全` · 别名: `ADL`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `rolling_beta_to_market`

**标签**: `扩展` · `时序` · `⚠️ PIT 需核查` · 别名: `FP_BETA`, `ROLLING_BETA_TO_MARKET`, `fp_beta`

**公式/构造**:

$$\beta_{i,t} = \frac{\mathrm{Cov}(r_i,r_m)}{\mathrm{Var}(r_m)}$$

**可用条件**:

- 最少样本 min_periods = 2
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 窗口内 Cov(r_i, r_m) / Var(r_m)；请优先使用 production 算子 rolling_beta。

### `rolling_obv`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `rolling_pvt`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `row_sum_skipna`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `RSX`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `same_clock_lag`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `saturate`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\min(\max(x,0),1)$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

饱和到 [0,1]。

计算: clip(x, 0, 1)。

备注: 限制到 [0,1]

示例: ``saturate(Return)``

### `scale`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `SCALE`, `c_scale`

**公式/构造**:

$$\tilde{X}_{i,t} = \frac{X_{i,t}}{\sum_j |X_{j,t}|}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截面缩放至单位 L1 范数。

计算: X / sum(|X|)（截面内）。

备注: 缩放数据使sum(abs(x))=指定值（与c_scale相同）

示例: ``scale(Return, 1)``

精讲: X / sum(|X|)（截面内）。

### `sec`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\sec x = 1/\cos x$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

正割。

计算: 1/cos(x)。

备注: Polars sec

示例: ``sec(angle)``

### `senkou_span_causal_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sigmoid`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

Sigmoid 映射。

计算: 1 / (1 + e^{-x})。

备注: Sigmoid

精讲: 1 / (1 + e^{-x})。

### `signed_event_decay`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `signed_event_rate`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `signed_power`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{sgn}(a)\cdot|a|^b$$

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素保号幂。

计算: 两个同形序列按位置保号幂。

备注: 符号幂

示例: ``signed_power(zscore, 2)``

### `sin`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\sin x$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素正弦。

计算: 对 panel 每个元素应用正弦。

备注: 正弦

示例: ``sin(angle)``

### `sin_phase`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sinh`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\sinh x$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素双曲正弦。

计算: 对 panel 每个元素应用双曲正弦。

备注: Polars sinh

示例: ``sinh(x)``

### `size_neutralize`

**标签**: `日线` · `截面` · `PIT 安全` · 别名: `CAP_NEUTRALIZE`, `MARKET_CAP_NEUTRALIZE`, `SIZE_NEUTRALIZE`, `cap_neutralize`, `market_cap_neutralize`

**公式/构造**:

$$Y_{i,t} = \alpha_t + \beta_t \ln(\max(\mathrm{cap}_{i,t}, 1)) + \varepsilon_{i,t}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

市值中性化（对 log(cap) 截面回归残差）。

计算: 每个交易日 Y 对 log(市值) 回归，取残差。

备注: size_neutralize（Polars 桥接）

示例: ``size_neutralize(factor, market_cap)``

精讲: 每个交易日 Y 对 ln(市值) 回归取残差；仅 mcap>0 且有限参与（R19-024）。

### `sma_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `spectral_energy_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `spectral_trend_share`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sqrt_abs`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\sqrt{|x|}$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素绝对值开方。

计算: 对 panel 每个元素应用绝对值开方。

备注: sqrt_abs（Polars 桥接）

示例: ``sqrt_abs(x)``

### `square`

**标签**: `日线` · `逐元素` · `PIT 安全` · 别名: `sqr`

**公式/构造**:

$$x^2$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素平方。

计算: 对 panel 每个元素应用平方。

备注: 平方

示例: ``square(x)``

### `sr_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sr_touch_count`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_adaptive_deadband`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_adaptive_slew_limit`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_age`

**标签**: `扩展` · `逐元素` · `PIT 安全` · 别名: `state_episode_duration`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_confidence_weighted_ema`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_cost_aware_deadband`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_cost_aware_slew`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_deadband`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_dwell_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_age`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_age_capped`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_age_lower_bound`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_censored_flag`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_efficiency`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_excursion_balance`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_mae`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_mfe`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_episode_retrace_ratio`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_ewm_if`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_flip_age`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_flip_density`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_hold`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_l2_partial_adjustment`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_latch`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_persistence`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_quantile_hysteresis`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_rank_deadband`

**标签**: `扩展` · `分组` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

分组算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_since_count`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_since_last`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_since_mean`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_since_sum`

**标签**: `扩展` · `逐元素` · `PIT 安全` · 别名: `state_since_reduce`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_since_trend_tstat`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_slew_limit`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_transition_count`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_transition_rate`

**标签**: `扩展` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_transition_surprise`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `state_uncertainty_deadband`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `suspension_frequency`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `suspension_status_coverage`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sv_net_flow_direction`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sv_own_flow_fraction`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sv_self_relative_change`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sv_signed_beta_market`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `sv_signed_shock_persistence`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tail_beta`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

$$\beta_{\mathrm{tail}} = \frac{\mathrm{Cov}(r_i,r_m\mid r_m\le Q_q)}{\mathrm{Var}(r_m\mid r_m\le Q_q)}$$

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

精讲: 仅在 r_m 处于窗口内最低 q 分位（如 5%）的样本上估计 Beta。

### `tan`

**标签**: `不安全` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\tan x$$

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

逐元素正切。

计算: 对 panel 每个元素应用正切。

备注: 正切

示例: ``tan(angle)``

### `tema_distance_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tenkan_kijun_cross`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tod_close_midday_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tod_edge_activity`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tod_intraday_range_position`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tod_open_midday_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tod_overnight_activity_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `tradable_state`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `trading_day_diff`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `true_range`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `true_range_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `true_range_surprise`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `true_range_zscore`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `truncate`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\mathrm{trunc}(X,\ k)$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

截断小数位。

计算: 保留 k 位小数（截断而非四舍五入）。

备注: Polars truncate

示例: ``truncate(price, 2)``

### `ulcer_index`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `unitize`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

$$\tilde{X} = 2R - 1$$

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

归一化到 [-1,1]。

计算: 2·rank - 1 或等价缩放。

备注: unitize（Polars 桥接）

示例: ``unitize(x)``

### `update_acceleration`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 4
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `update_direction_persistence`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `update_path_efficiency`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 3
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `update_surprise`

**标签**: `扩展` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 5
- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_discount_regime_share`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_earnings_yield_ma_diff`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_earnings_yield_persistence`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_earnings_yield_slope`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_equity_yield_dispersion`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_fcf_yield_growth`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_pe_beta_to_market`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_qmj_quality_rank`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_relative_valuation_gap`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_valuation_percentile_own`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_valuation_stat_spread`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_valuation_z_own`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `val1_valuations_lag_component`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_cashflow_disagreement`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_growth_mismatch`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_pcf_definition_gap`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_pcf_gap_positive`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_pcf_gap_signed_log`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_pe_gap_positive`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_pe_gap_signed_log`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_pe_ttm_lyr_gap`

**标签**: `日线` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `valuation_quality_mismatch`

**标签**: `日线` · `逐元素` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `VortexMinus`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `VortexPlus`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vpin_pct`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vr1_range_everage`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vr1_range_to_close_eff`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vr1_rogers_satchell`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_fractional_share`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `vv1_regime_change_ratio`

**标签**: `扩展` · `逐元素` · `⚠️ PIT 需核查`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- DSL 白名单: ⚠️ **不在**（默认 DSL 不可写，需白名单/别名路径）
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

逐元素算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `wavelet_detail_energy_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `zero_return_ratio`

**标签**: `日线` · `时序` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`, `sql`

**功能讲解**:

时序算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。

### `zmijewski_score`

**标签**: `日线` · `fundamental_period` · `PIT 安全`

**公式/构造**:

（暂无公式 —— 见讲解中的构造说明）

**可用条件**:

- 最少样本 min_periods = 1
- DSL 白名单: **在**
- 后端: `pandas_numpy`, `polars`

**功能讲解**:

fundamental_period算子（基础骨架句，暂无深挖文档 —— 公式与语义待补；参数与标签以下方元数据为准）。
