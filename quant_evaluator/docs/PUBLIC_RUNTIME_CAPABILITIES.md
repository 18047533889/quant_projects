# 公开运行能力与输入/输出契约

本页补充 `METRIC_REFERENCE.md` 的调用层说明。指标是否可请求以当前注册表为准；
`METRIC_REGISTRY_COVERAGE.csv` 是迁移追踪表，不能把其中历史遗留的 `NOT_IMPLEMENTED` 当作当前运行结论。

## 三个统计指标

设某因子的有效日 IC 序列为 $`x_1,\ldots,x_n`$，均值为 $`\bar x`$。公开 `evaluate`
会从 `FactorBatch` 与 `LabelBundle` 计算所需的日 IC 制品，调用方不需要自行构造内部中间量。

### `hac_tstat`

默认 `max_lag=5`、`kernel="bartlett"`、`min_periods=30`。仅裁掉首尾缺失值；内部缺失或无穷值
不会被压缩成连续日历，而会产生缺失证据。实现以 $`n`$ 为分母计算

```math
\gamma_\ell={1\over n}\sum_{t=\ell+1}^{n}(x_{t-\ell}-\bar x)(x_t-\bar x),
\quad w_\ell=1-{\ell\over L+1},
```

```math
\widehat{\mathrm{Var}}(\bar x)={1\over n}
\left(\gamma_0+2\sum_{\ell=1}^{L}w_\ell\gamma_\ell\right),
\quad t_{HAC}={\bar x\over\sqrt{\widehat{\mathrm{Var}}(\bar x)}}.
```

不足门限或方差非正时结果为缺失值。注册适配器返回每个因子的 $`t_{HAC}`$。

### `block_bootstrap_ci`

默认块长 $`b=10`$、重复数 $`B=1000`$、置信度 95%、固定随机种子 0、`min_periods=60`。
每次重复从原始时间轴均匀抽取 $`\lceil n/b\rceil`$ 个连续块，拼接后截取前 $`n`$ 个观测并求均值。
令这些均值的 2.5% 和 97.5% 分位数为 $`q_{.025},q_{.975}`$。底层函数返回完整区间，公开注册指标返回

```math
h={q_{.975}-q_{.025}\over2}.
```

同一次批量计算的因子共享抽样起点；含缺口或无穷值的列不会压缩日历后重抽样，而是返回缺失证据。

### `benjamini_hochberg_correction`

对 $`m`$ 个有限 p 值升序排列为 $`p_{(1)}\le\cdots\le p_{(m)}`$。在给定 FDR 水平
$`\alpha`$ 下，拒绝截至最大的 $`k=\max\{i:p_{(i)}\le i\alpha/m\}`$，校正 p 值为

```math
q_{(i)}=\min\left(1,\min_{j\ge i}{m p_{(j)}\over j}\right).
```

结果映射回原因子顺序。公开 `evaluate` 入口先从 Spearman 日 IC 计算默认 HAC 双侧 p 值
（`min_periods=30`、`max_lag=5`、Bartlett kernel），再返回每个因子的 BH 校正 p 值；拒绝掩码和发现数
属于底层策略诊断，不是该标量指标的公开值。

```python
request = EvaluationRequest(
    factor_batch,
    labels,
    metric_ids=("hac_tstat", "block_bootstrap_ci", "benjamini_hochberg_correction"),
    tier="research",
)
bundle = evaluate(request)
for metric_id in request.metric_ids:
    values = bundle.artifacts[metric_id].values  # shape: (F,)
    one_value = bundle.get_metric(metric_id, factor_batch.factor_ids[0])
```

## 结果读取：标量与序列分开

`get_metric(metric_id, factor_id)` 用于逐因子标量。序列、向量和矩阵不放进 `metric_values`；
统一从类型化制品读取：

```python
bundle = evaluate(factor_batch, labels, metrics=("rank_ic_series",))
daily_rank_ic = bundle.artifacts["rank_ic_series"].values  # shape: (T, F)
```

## 轴坐标与数值数组边界

常规时间和资产轴首选一维 `int64` 数组，但这不是普遍强制要求。`AxisRef.size`、
`LabelBundle.horizon` 与 `execution_delay` 接受 Python/NumPy 整数，并明确拒绝 `bool`、浮点数和字符串。
数值坐标必须有限，datetime 坐标不能含 `NaT`；时间坐标必须严格递增，所有坐标必须唯一。

`AxisRef` 也支持普通字符串、`datetime64`、`timedelta64`、布尔坐标，以及由不可变标量组成的 object 数组，
例如字符串、整数、日期时间、`Decimal` 和 NumPy 标量。object 坐标的公开读取返回只读的分离副本；
列表、字典、自定义可变对象，以及含嵌套 object/结构字段的 NumPy 标量会被拒绝。

```python
time_axis = AxisRef("time", "int64", T, np.arange(T, dtype=np.int64))
asset_axis = AxisRef(
    "asset", "object", 3,
    np.asarray(["000001.SZ", "600000.SH", "700.HK"], dtype=object),
)
```

轴坐标的 dtype 自由度不延伸到计算载荷：因子值、标签值、价格和收益面板仍须遵守各自的实数数组合同，
不能用 object 坐标的许可绕过载荷校验。

## 组合收益与回撤：两阶段类型化调用

推荐的 cohort 组合路径必须同时显式提供独立的 `HoldingReturnPanel` 和冻结的 `PortfolioSpec`。
前向标签不能冒充持有期组合收益。

```python
holding_returns = HoldingReturnPanel.from_prices(
    prices,
    time_axis=factor_batch.time_axis,
    asset_axis=factor_batch.asset_axis,
    source_ref="vendor:close-prices:v1",
    price_basis="close_to_close",
)
portfolio_spec = PortfolioSpec(holding=2, n_quantiles=5, per_side_cost=0.0005)
stage_one = evaluate(
    factor_batch, labels, metrics=("long_short_returns",),
    holding_returns=holding_returns, portfolio_spec=portfolio_spec,
)
long_short = stage_one.artifacts["long_short_returns"]
```

这里的显式 `PortfolioSpec` 约束 cohort 构造路径。兼容路径
`evaluate(factor_batch, labels, metrics=("long_short_returns",))` 可不传 spec，直接使用标签收益；底层 legacy
`compute_long_short_returns(factor_values, forward_returns, ...)` kernel 同样不接收 `spec` 参数。不要把兼容路径
描述成 cohort 输入，也不要把两套接口混写。

回撤指标必须在第二次调用中接收类型化的组合制品：

```python
portfolio = ProbePortfolioArtifact(
    long_short.values,
    time_index=long_short.time_axis.time_index,
    factor_ids=factor_batch.factor_ids,
    provenance={"source_metric": "long_short_returns"},
)
drawdown = evaluate(
    factor_batch, labels, metrics=("max_drawdown",), portfolio_returns=portfolio,
)
max_drawdown_by_factor = drawdown.artifacts["max_drawdown"].values
```

省略 `ProbePortfolioArtifact` 会关闭式失败；运行层不会把 `LabelBundle` 的前向收益静默当作已构造的组合轨迹。
