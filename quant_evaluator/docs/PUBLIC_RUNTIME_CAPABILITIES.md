# 公开运行能力与输入/输出契约

> 最后更新：2026-09-24（Asia/Hong_Kong）。文档维护规则：代码行为变更必须同一轮同步本文档并更新此时间戳。

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

底层 BH、Bonferroni、Holm 和 Šidák 的 p 值输入必须是实概率：复数输入直接拒绝，
不能通过丢弃虚部参与检验。对实数数组先以原精度检查有限值是否落在 [0, 1]，
再转为 float64 计算；例如 longdouble 略大于 1 的值不会因舍入成 1 而被接受。
NaN 和正负无穷仍按原有规则作为缺失检验排除，不改变有效检验数的既有口径。

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


## 默认后端与 `backend="auto"` 的保守选择范围

公开 `evaluate(...)`（省略 `backend` 或传入 `None`）和显式
`evaluate(..., backend="auto")` 按整次请求选择 CPU 或 CUDA，并在返回的
`bundle.metadata` 中记录选择。策略版本为 `ashare_public_routes_20260927_v9`。
以下 701 日规则继续适用于短历史配置。
它依据已注册 A 股日行情构造的 701 个交易日、5314 只股票、2 个因子输入，
逐指标比较完整公开入口的 CPU/CUDA 结果。冷调用和交错顺序的热调用都更快、
且数值、有效掩码、计数和输入证据对拍通过的 12 项才列入 CUDA 候选：

`daily_quantile_monotonicity_rate`、`daily_quantile_monotonicity_series`、
`ic_ir`、`ic_median`、`ic_std`、`quantile_monotonicity`、
`quantile_returns_daily`、`quantile_returns_full`、`quantile_spread`、
`rank_ic`、`rank_ic_series`、`turnover`。

另有三组完整公开入口的多指标批次，在相同真实行情规模与 NVIDIA L20 上
两轮冷/热 A/B 都显示 CUDA 较快，且全部 8 次完整输出对拍通过。只有下列
**完全相同的规范指标集合**可整批转 CUDA；顺序和指标别名不影响集合匹配：

- `rank_chain`：`rank_ic`、`rank_ic_series`、`ic_std`、`ic_ir`。
- `quantile_chain`：`quantile_returns_full`、`quantile_returns_daily`、
  `quantile_spread`、`quantile_monotonicity`、
  `daily_quantile_monotonicity_rate`。
- `mixed_core`：`rank_ic`、`rank_ic_series`、`ic_ir`、
  `quantile_spread`、`turnover`、`factor_turnover_rate`。

短历史三组之外的组合均使用 CPU。另有真实 COS 全量 2586 日 × 5461 股 × 2 因子的精确三指标组合 `rank_ic`、`quantile_spread`、`factor_turnover_rate` 已完成整批 CPU/CUDA A/B 与完整产物对拍，在该精确形状、默认参数、L20 和足额显存下整批使用 CUDA；相邻规模及其子集不据此开放。相同的规范三指标集合在独立 F8 COS 面板的精确 2586 日 × 5461 股 × 8 因子形状也完成六轮 CPU/CUDA A/B 和完整输出对拍，自动路由使用独立 profile 与至少 14 GiB 有效空闲显存门槛，三项统一走 CUDA，回执原因是 `certified_batch_real_cos_f8_mixed_three`；相邻形状和其它 F8 指标集合仍为 CPU。F8 CPU 热运行约 34.35–34.50 秒，整批 CUDA 约 9.06–9.12 秒，实测 GPU 峰值 9,332,757,504 字节。已对拍的 `portfolio_core` 包含类型化组合轨迹，
目前也未开放自动 CUDA，因为它超出本策略的输入合同。所有自动 CUDA 路径都要求设备型号恰为 `NVIDIA L20`。单指标要求
空闲显存与按 `GPUExecutionPolicy.max_vram_fraction` 计算的有效预算均至少
8 GiB；三组批次的这两个门槛均为 12 GiB。例如比例上限为 40% 时，
批次至少需 30 GiB 实际空闲显存。型号或预算不满足时自动使用 CPU，
并记录具体原因。8 GiB 高于单指标实测约 4.64 GiB 的峰值，
12 GiB 高于混合批次实测约 9.00 GB 的峰值，分别留有余量。

短历史自动 CUDA 路径还要求：因子和标签载荷均为 float64；时间长度为
600–800、资产数为 5000–5500、因子数恰为 2；使用默认指标参数，
且没有自定义 context、分桶构造参数、组合轨迹、持有收益、可交易面板、
日历快照、暴露面板、泛化证据或自定义 Evaluator。指标别名按注册表解析，
例如 `ic.rank.mean` 采用 `rank_ic` 的选择规则。设备不可用时自动走 CPU。
多年路径同样要求 float64、默认指标参数及无上述特殊输入。

多年大盘路径对单指标 `rank_ic` 和 `quantile_spread` 开放；
`factor_turnover_rate` 单独请求保持 CPU。上述精确三指标组合是独立的整批认证例外，
不适用单指标区域外推。公开入口在真实 COS 因子面板的六种形状
（1000–2586 日、5000–5461 股、1 或 2 因子）完成 12/12 数值、掩码、
计数和证据对拍。F2 从 1000 日起、F1 从 2000 日起可进入路由；
F1 的 1000 日边界上 `rank_ic` 冷启动 CUDA 慢于 CPU，因此保持 CPU。
已测核心范围上限为 2600 日、5500 股。为容纳后续行情更新，
2601–3200 日或 5501–6000 股作为有界外推区，选路原因标记
`bounded_extrapolation_real_cos_headroom`；外推区尚无直接速度对拍。
超出这些边界使用 CPU，CUDA 执行时出错仍直接报错。

多年路径沿用 L20 限制和空闲显存/策略有效预算双门槛。`quantile_spread`
至少要求 8 GiB；`rank_ic` 要求不低于 8 GiB，且 F2 按
`768 × 日数 × 股票数 × 因子数` 字节、F1 按 `1024 × 日数 × 股票数`
字节估算最低有效显存。两系数约为六种实测峰值每单元的 1.5 倍向上取整。
不足时自动退回 CPU，并记录 `insufficient_cuda_memory`。精确三指标组合沿用
F2 `rank_ic` 的大面板显存预算公式（此形状约 21.7 GB 有效空闲显存）；
实测整批 CUDA 峰值约 11.23 GB。该请求 CPU 约 8.5–8.9 秒、CUDA
约 4.5–5.0 秒，逐指标 CUDA/CUDA/CPU 拆分计时约 5.6–5.7 秒；
数值与逐字段证据对拍通过。

另有真实 COS 独立 8 因子面板在精确 2586 日 × 5461 股 × 8 因子形状上，
对单指标 `rank_ic`、`rank_ic_series`、`ic_ir`、`quantile_spread`、
`factor_turnover_rate`、`quantile_returns_daily`、`quantile_returns_full` 完成公开入口
CPU/CUDA A/B 与完整输出对拍。这七个单指标只在该精确形状、float64、默认参数、
无特殊输入且 NVIDIA L20 下自动使用 CUDA；
形状、因子数或请求指标不同均不沿用此认证。rank 家族按实测
9,332,757,504 字节峰值的 1.5 倍设置有效空闲显存下限
（13,999,136,256 字节）；分位收益、分位差与换手继续要求至少 8 GiB。所有路径同时
检查实际空闲显存和乘以 `GPUExecutionPolicy.max_vram_fraction` 后的有效预算。
相同规范三指标集合的 F8 整批路由至少要求 14 GiB 有效预算，另完成六轮 CPU/CUDA
A/B（CPU 热 34.35–34.50 秒、CUDA 热 9.06–9.12 秒，峰值 9,332,757,504 字节），
完整输出逐字段对拍通过。整批证据见
`docs/benchmarks/real_cos_f8_mixed_20260927.json`。单指标证据见
`docs/benchmarks/real_cos_f8_rank_20260927.json`、
`docs/benchmarks/real_cos_f8_quantile_20260927.json`、
`docs/benchmarks/real_cos_f8_turnover_20260927.json`。
`rank_ic_series` 六轮交错 A/B 的 CPU/CUDA 热运行约 20.61–20.74/5.90–5.99 秒；
`ic_ir` 约 20.56–20.58/5.96–6.01 秒。两者的六轮完整产物、掩码、计数、
provenance、MetricValue 与配置哈希均对拍通过，GPU 峰值均为 9,332,757,504 字节。
证据见 `docs/benchmarks/real_cos_f8_rank_ic_series_20260927.json` 和
`docs/benchmarks/real_cos_f8_ic_ir_20260927.json`；未据此开放 F12 或多指标组合。
`quantile_returns_daily` 六轮 CPU/CUDA 热运行约 9.50–9.85/5.43–5.89 秒；
`quantile_returns_full` 约 9.47–9.53/5.40–5.41 秒。日序列 CUDA 曾缺失标签身份、
参数和有效期计数等 provenance；补齐后六轮的数值、轴、掩码、计数、完整 provenance
与配置哈希全部对拍通过。两项 GPU 峰值均为 1,436,371,456 字节。证据见
`docs/benchmarks/real_cos_f8_quantile_returns_daily_20260927.json` 与
`docs/benchmarks/real_cos_f8_quantile_returns_full_20260927.json`；F12 和组合请求仍保持 CPU。

独立 F12 COS 面板在精确 2586 日 × 5461 股 × 12 因子形状上，单指标
`rank_ic`、`quantile_spread`、`factor_turnover_rate` 均完成公开入口 CPU/CUDA
A/B 和完整输出对拍。rank CPU/CUDA 热运行分别为 31.607/9.449 秒，GPU 峰值
9,332,757,504 字节；quantile 为 21.612/8.787 秒，峰值 1,436,371,456 字节；
turnover 为 19.510/14.297 秒，峰值 1,020,942,336 字节。这三个指标仅在该精确
形状自动使用 CUDA，并要求 float64、默认参数、无特殊输入、NVIDIA L20；rank
至少需 14 GiB 有效空闲显存，quantile 与 turnover 至少需 8 GiB。F12 多指标
请求保持 CPU；三项单指标的回执原因均为 `certified_single_metric_real_cos_f12`。证据见 `docs/benchmarks/real_cos_f12_rank_20260927.json`、
`docs/benchmarks/real_cos_f12_quantile_20260927.json` 和
`docs/benchmarks/real_cos_f12_turnover_20260927.json`。

其他规模和特殊输入使用 CPU，表示尚无足够的公开入口性能证据；
不代表 CUDA 无法执行这些指标。

另有 `coverage`、`pearson_ic`、`pearson_ic_series`、
`pearson_ic_std` 在本次真实行情的热运行中 CUDA 更快，但冷调用 CPU 更快，
因此未纳入首版自动白名单。已显式指定 `backend="cuda"` 或
`backend="gpu"` 的调用仍要求 CUDA；设备或指标不支持时直接报错。
自动路由不会在 CUDA 执行失败后悄悄重算 CPU。

公开 `evaluate()` 的后端值仅接受 `None`（默认自动选路）、`cpu`、`auto`、
`cuda`、`cuda_strict`、`gpu`，以及 `.value` 为这些值的枚举。
未知值会抛出 `InvalidContractError`，包括误传 `numba`、`polars`、
`cpu_fast` 或拼错的 CUDA 名称。`numba`、`polars` 目前仅在底层
`compute_daily_ic()` 的后端参数中可用，不能据此推断公开入口已使用它们。

默认调用的 `bundle.metadata["backend_requested"]` 为 `"default"`，显式
`backend="auto"` 为 `"auto"`；显式 `backend="cpu"` 固定参考 CPU 路径。
`backend_strategy` 区分 `"auto"` 和 `"explicit"`，
`backend_used` 为 `"cpu"` 或 `"cuda"`，
`auto_backend_policy` 为策略版本，
`auto_backend_reason` 记录选路原因，
`auto_backend_profile` 区分 701 日、COS 已测核心、有界外推区和精确 F8/F12 区，
`metric_backends` 给出原请求指标名到实际后端的映射。
`execution_receipt` 保存这些路由字段、语义 `config_hash` 及独立的
`receipt_hash`，便于区分默认、显式 CPU 与显式 CUDA 的执行记录。
该记录不参与现有 `config_hash` 和指标制品内容哈希；同一输入和参数的
CPU/CUDA 结果仍需按指标容差核验。基准脚本与逐项证据见
`docs/benchmarks/public_backend_routes_20260926_interleaved.json` 与
`docs/benchmarks/public_batch_routes_20260926.md`、
`docs/benchmarks/real_cos_route_region_20260927.json`。

MetricInstance 聚合结果另以 `instance_execution_receipts` 按实例 ID 保存
实际 `backend_used`、实例 `config_hash` 及来源分组的执行回执。
每个子结果的 `instance_execution_receipt` 与聚合映射对应；聚合
`execution_receipt` 绑定所有实例回执哈希和聚合 `config_hash`。
聚合层不提供单一 `backend_used`，因为不同实例可能采用不同后端。
