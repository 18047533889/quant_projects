# FactorAnalyzer 与 Exposures 输出总览

本文档汇总 `factor_layer/alphapurify/FactorAnalyzer.py` 与 `factor_layer/alphapurify/Exposures.py` 的主要输出内容，供测试产物核对使用。

## 1. FactorAnalyzer 输出

### 1.1 运行后实例属性输出（`run()`）

- `retruns_dict`：按调仓周期保存收益时序表（含分位收益、滚动收益、累计净值、换手等列）。
- `indus_returns_dict`：按调仓周期保存行业收益贡献（启用分组时）。
- `ls_turnovers_dict`：按调仓周期保存多空组合平均换手。
- `ics_dict`：按收益预测周期保存 IC / Rank IC 时序。
- `ls_stats_panel`：多空统计面板。
- `l_stats_panel`：多头统计面板。
- `s_stats_panel`：空头统计面板。
- `ic_stats_panel`：IC 统计面板。
- `ls_monthly_panel`：多空月度聚合面板。
- `l_monthly_panel`：多头月度聚合面板。
- `s_monthly_panel`：空头月度聚合面板。
- `ic_monthly_panel`：IC 月度聚合面板。
- `mean_returns_dict`：按调仓周期的均值收益。
- `mean_ics_dict`：按收益预测周期的均值 IC。
- `mean_ic_autocorrs_dict`：按收益预测周期的 IC 自相关均值。
- `ic_indus_contribs_dict`：按收益预测周期的行业 IC 贡献。

### 1.2 图形输出（Plotly）

- `create_single_fac_ic_sheet()`
- `create_long_short_return_sheet()`
- `create_long_return_sheet()`
- `create_short_return_sheet()`
- `create_single_fac_full_sheet()`（内部顺序调用多张图）

以上函数均会绘图（`fig.show()`）；传 `return_fig=True` 时返回 `plotly.graph_objects.Figure` 对象，可由运行脚本落盘。

## 2. Exposures 输出

`Exposures.py` 内包含两个核心分析类：`PortfolioExposures` 与 `PureExposures`。

### 2.1 PortfolioExposures 输出

#### 运行后数据输出（`run()`）

- `result_df`（Polars）主要列包括：
  - `*_expo`：暴露值
  - `intercept`：截距项
  - `*_attr`：归因收益
  - `*_cum_ret`：归因累计收益
  - `intercept_cum_ret`：截距累计收益
  - `portfolio_ret`：组合收益
  - `portfolio_cum_ret`：组合累计收益

#### 图形输出

- `plot_portfolio_exposures()`
- `plot_portfolio_returns()`
- `plot_portfolio_exposures_and_returns()`

### 2.2 PureExposures 输出

#### 运行后数据输出（`run()`）

- `result_df`（Polars）：纯暴露与归因主结果。
- `corr_df`（Pandas）：因子与暴露时序相关。
- `corr_matrix`（Pandas）：相关矩阵。

#### 图形输出

- `plot_pure_exposures()`
- `plot_pure_returns()`
- `plot_pure_exposures_and_returns()`
- `plot_correlations()`

## 3. 图片输出说明

当前配置化运行链路会统一保存 `PNG` 图片，便于在 IDE 或文件管理器中直接预览。
