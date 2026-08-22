# 组合暴露分析

## 1. 功能边界

本模块消费 Barra-lite 的因子暴露 B 和因子收益 f，支持：

- 实际、目标和执行目标权重口径；
- 组合风格与申万一级行业暴露；
- 指数基准暴露及组合主动暴露；
- 风险模型覆盖率和未知权重；
- 实验性事后 B×f 因子归因。

本模块不读取 F/D，不提供绝对波动率、VaR、最小方差或正式风险贡献。

## 2. 权重口径

默认 `realized_close`：

```python
weight = pf.asset_value(group_by=False) / pf.value()
```

该口径会反映 accurate 模式中的停牌、涨跌停、整手、资金和成交延迟。
`scheduled_target` 与 `execution_target` 用于比较目标和真实风险暴露；
后者目前只由 fast 模式提供。

## 3. 暴露口径

组合暴露：

```text
x[p,t] = sum_i(weight[p,i,t] * B[i,t])
```

主动暴露：

```text
x[active,t] = x[portfolio,t] - x[benchmark,t]
```

输出同时包含 NAV 权重暴露和已投资资金归一化暴露。现金在 NAV 口径下
暴露为零，因此会自然稀释股票风险暴露。

## 4. 缺失处理

`missing_policy`：

- `error`：任意非零权重缺少 B 时失败；
- `report_unknown`：默认，保留未知权重并报告覆盖率；
- `renormalize`：只在已覆盖资产内重新归一暴露。

禁止将未知股票的风格或行业暴露静默填为零。

## 5. 实验性归因

当前 f 由当日 B 回归当日股票收益得到。为了与该回归口径一致，归因使用：

```text
contribution[k,t] = sum_i(weight[i,t-1] * B[i,k,t]) * f[k,t]
```

它属于事后归因，不是事前预测。组合实际收益与因子贡献之差写入
`unexplained_return`，其中包括未输出的市场截距、特异收益、开盘调仓
时点差异、费用和滑点。主动归因通常比绝对归因更有解释力。

## 6. 配置

```yaml
analysis:
  exposure:
    enabled: true
    risk_model_root: ../v2_sbi_mvl_fullA
    benchmark_data_root: ../lqtp_data
    benchmark_index: 000300.SH
    weight_source: realized_close
    missing_policy: report_unknown
    min_portfolio_coverage: 0.98
    min_benchmark_coverage: 0.995
    include_attribution: true
    plot: true
    png_dpi: 150
    industry_top_n: 10
```

风险模型与回测日期自动取交集。分析目录默认输出 parquet、覆盖率 CSV、
元数据 JSON、`exposure_dashboard.html`，以及以下风格暴露时序图：

- `portfolio_style_exposure.png`
- `benchmark_style_exposure.png`（配置基准时）
- `active_style_exposure.png`（配置基准时）

每张 PNG 按风格因子分成独立纵向子图并共享时间轴，零暴露使用虚线标记。

行业暴露默认输出：

- `active_industry_heatmap.png`
- `latest_active_industry_exposure.png`
- `portfolio_vs_benchmark_industry.png`
- `portfolio_industry_allocation.png`

热力图使用以零为中心的红蓝色阶；最新主动暴露图展示全部申万一级行业；
组合/基准对比选择主动权重绝对值最大的 `industry_top_n` 个行业；配置图
展示全期平均权重最高的 TopN，并将其余行业、未知权重和现金分别汇总。
交互式 HTML 同时包含行业热力图和最新行业截面。
