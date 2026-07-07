# Exposures / FactorAnalyzer 新旧版差异报告

## 对比范围
- 旧版：
  - `factor_layer/alphapurify/Exposures.py`
  - `factor_layer/alphapurify/FactorAnalyzer.py`
- 新版（简化版）：
  - `factor_layer/alphapurify/Exposures_abstract.py`
  - `factor_layer/alphapurify/FactorAnalyzer_abstract.py`

## 总体结论
- 新版不是重写算法，而是 **“抽象外观层（Facade）+ 旧引擎委托”**。
- 数值计算仍由旧版类执行（`PortfolioExposures`、`PureExposures`、`FactorAnalyzer`）。
- 新版主要改动是把流程拆成更清晰的阶段方法，减少大文件直接暴露的复杂度。

## Exposures：改了什么

### 结构层面
- 新增 `ExposureRunResult` 数据类，集中承载输出：
  - `result_df`
  - `corr_df`（仅 Pure 暴露）
  - `corr_matrix`（仅 Pure 暴露）
- 新增两个抽象入口类：
  - `PortfolioExposuresAbstract`
  - `PureExposuresAbstract`

### 执行流程层面
- 把 `run()` 组织成三步：
  - `_prepare_inputs()`
  - `_compute_outputs()`
  - `_publish_outputs()`
- 其中 `_compute_outputs()` 内部调用旧版引擎的 `run()`，保证行为兼容。

### 对外能力层面
- 绘图方法仍可用，抽象类内部直接转发到旧版引擎方法：
  - `plot_portfolio_exposures`
  - `plot_portfolio_returns`
  - `plot_pure_exposures` 等

### 实际影响
- 代码更短、更易读；
- 计算逻辑与结果预期保持一致（因为核心计算未迁移）。

## FactorAnalyzer：改了什么

### 结构层面
- 新增 `AnalyzerRunResult` 数据类，统一承载关键输出：
  - `ls_stats_panel`
  - `l_stats_panel`
  - `s_stats_panel`
  - `ic_stats_panel`
- 新增 `FactorAnalyzerAbstract` 作为简化入口。

### 执行流程层面
- `run()` 被拆成：
  - `_prepare_inputs()`
  - `_execute_core_analysis()`
  - `_publish_outputs()`
- `_execute_core_analysis()` 内部调用旧版 `FactorAnalyzer.run()`，然后提取关键面板结果。

### 对外能力层面
- 原有图表/报表方法继续可用，通过代理转发：
  - `create_single_fac_full_sheet`
  - `create_long_return_sheet`
  - `create_long_short_return_sheet`
  - `create_short_return_sheet`
  - `create_single_fac_ic_sheet`

### 实际影响
- 对调用方更友好（入口更干净）；
- 与旧版兼容性高（底层仍是同一计算引擎）。

## 测试验证结果
- 测试脚本（历史草稿）：`0431-测试文件/test_database_to_exposures_factoranalyzer_abstract_compare.py`
- 验证结论：
  - `Database -> Exposures -> FactorAnalyzer` 全链路可正常输出；
  - `PortfolioExposures` 与 `PortfolioExposuresAbstract` 输出一致；
  - `PureExposures` 与 `PureExposuresAbstract` 输出一致；
  - `FactorAnalyzer` 与 `FactorAnalyzerAbstract` 关键统计面板输出一致。

## 备注
- 由于新版抽象类当前是“委托旧版引擎”的设计，理论上只要代理关系不变，就应保持结果一致。
- 若后续把算法逐步迁移到抽象类内部，建议继续沿用本次一致性测试作为回归基线。
