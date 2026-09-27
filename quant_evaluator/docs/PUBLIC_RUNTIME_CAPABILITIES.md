# 公开运行能力与输入/输出契约

> 最后更新：2026-09-27（Asia/Hong_Kong）。文档维护规则：代码行为变更必须同一轮同步本文档并更新此时间戳。

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


## 多期限 CUDA 因子 tile 复用

`evaluate_many` 和 `evaluate_horizons` 在每个期限的公开合约、样本掩码、
sealed split 与后端选择均通过预校验后，对合格请求使用同一个
`DeviceEvaluationSession`。执行器按因子 tile 上传一次，再逐个期限上传标签并计算；
Spearman/Pearson 的秩与相关中间量每个期限独立生成，绝不跨不同 pairwise-finite
掩码复用。`evaluate_horizons` 保持 IC 家族；`evaluate_many` 还可复用
自含的 quantile 链、`coverage`、`turnover` 和 `factor_turnover_rate`。
注册别名先按规范指标 ID 判断共享资格，公开返回仍保留调用方的别名键；
例如 `ic.rank.daily` 与 `rank_ic_series` 都可触发同一条共享路径。
每个期限仍必须独立通过原有 CUDA 准入；扩展共享上传不扩大 `auto` 的指标、形状或参数认证范围。
需要组合/持有期收益或暴露面板等额外类型化输入的指标仍沿用原有路径。
显式 CUDA 失败仍报错，OOM 只重切当前因子 tile。

```python
from quant_evaluator import evaluate_many

metrics = ("quantile_returns_full", "quantile_returns_daily", "quantile_spread",
           "quantile_monotonicity", "daily_quantile_monotonicity_rate")
by_label = evaluate_many(batch, (h1_labels, h2_labels), metrics=metrics)
# 省略 backend 等同 auto：仅在每个期限均获认证时选择共享 CUDA。
forced = evaluate_many(batch, (h1_labels, h2_labels), metrics=metrics,
                       backend="cuda_strict")
custom = evaluate_many(batch, (h1_labels, h2_labels),
                       metrics=("quantile_returns_daily",),
                       metric_parameters={"quantile_returns_daily": {"n_quantiles": 1}},
                       backend="cuda_strict")
```

`metric_parameters` 逐期限原样转交公开 `evaluate`；自定义参数不继承默认
`auto` 的 CUDA 性能认证，需显式选择并自行核验设备预算。

每个期限返回的 `h2d_bytes`、`d2h_bytes` 等设备会话计数在复用路径中属于
**所有期限共享的会话总量**，由
`device_session_counter_scope="shared_session_total"` 标识，并提供
`shared_session_label_count` 与 `shared_session_factor_tiles_processed`。
这些计数不能逐期限相加。结果制品、配置哈希及逐期限标签溯源仍各自独立。

真实 COS 同一绑定 manifest 的 2586 日 × 5461 股 × 2 因子、两个 AdjVwap
前向期限在 NVIDIA L20 上做了顺序 CUDA / 共享 CUDA 的 ABBA。第二期限由连续
真实交易区间的单期收益复合，未知末期标为无效。两个期限每轮的配置哈希、
完整 `SeriesMetricArtifact` 每个字段、逐因子 MetricValue 与语义溯源均对拍通过。
顺序调用与复用调用的中位耗时分别为 3.418057 秒和 3.121428 秒
（约 1.095 倍）；因子上传 2 次降为 1 次，H2D 从 677,863,008 降至
451,908,672 字节，D2H 均为 82,784 字节，GPU pool 峰值均为
11,234,754,560 字节。该 A/B 只验证此规模、F2、双期限
`rank_ic_series` 请求，不扩展单次请求的自动路由认证。
可复跑脚本为 `quant_evaluator/scripts/benchmark_real_cos_multi_horizon_f2.py`，
小型源绑定、逐字段布尔对拍与性能证据见
`quant_evaluator/docs/benchmarks/real_cos_multi_horizon_f2_20260927.json`。

同一 COS manifest 的 F24 实测进一步覆盖 2586 日 × 5461 股 × 24 因子、
两个 AdjVwap 期限与 `quantile_returns_full`、`quantile_returns_daily`、
`quantile_spread`、`quantile_monotonicity`、
`daily_quantile_monotonicity_rate` 五项组合。顺序 CUDA / 共享 CUDA 按
ABBA 交错执行，两个共享轮次的制品逐字段、配置哈希与语义溯源均与首个顺序
轮次一致；随后 `auto` 对两个期限都选择 CUDA，共享上传一次且同样对拍通过。
顺序与共享路径的中位耗时为 35.753644 秒和 33.983173 秒，约快 5.2%；
因子上传从 2 次、5,422,904,064 字节降为 1 次、2,711,452,032 字节，
H2D 从 5,648,858,400 降为 2,937,406,368 字节。单轮 GPU 峰值约
3.25 GB，进程 RSS 高水位约 25.0 GiB；数据载入另耗时 96.38 秒，不含于
上述计算耗时。证据见
`quant_evaluator/docs/benchmarks/real_cos_multi_horizon_f24_quantile_chain_20260927.json`；
脚本可用 `--profile f24-quantile-chain` 复跑。这个结果只支持该 F24 双期限、
默认参数、五项组合；不能推断 F32/F64、任意参数或全部指标都以 GPU 最快。

## 默认后端与 `backend="auto"` 的保守选择范围

公开 `evaluate(...)`（省略 `backend` 或传入 `None`）和显式
`evaluate(..., backend="auto")` 按整次请求选择 CPU 或 CUDA，并在返回的
`bundle.metadata` 中记录选择。策略版本为 `ashare_public_routes_20260927_v17`。
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

短历史三组之外的组合均使用 CPU。另有真实 COS 全量 2586 日 × 5461 股 × 2 因子的精确三指标组合 `rank_ic`、`quantile_spread`、`factor_turnover_rate` 已完成整批 CPU/CUDA A/B 与完整产物对拍，在该精确形状、默认参数、L20 和足额显存下整批使用 CUDA；相邻规模及其子集不据此开放。相同的规范三指标集合在独立 F8 COS 面板的精确 2586 日 × 5461 股 × 8 因子形状也完成六轮 CPU/CUDA A/B 和完整输出对拍，自动路由使用独立 profile 与至少 14 GiB 有效空闲显存门槛，三项统一走 CUDA，回执原因是 `certified_batch_real_cos_f8_mixed_three`；相邻形状及未认证的 F8 指标集合仍为 CPU。F8 CPU 热运行约 34.35–34.50 秒，整批 CUDA 约 9.06–9.12 秒，实测 GPU 峰值 9,332,757,504 字节。已对拍的 `portfolio_core` 包含类型化组合轨迹，
目前也未开放自动 CUDA，因为它超出本策略的输入合同。所有自动 CUDA 路径都要求设备型号恰为 `NVIDIA L20`。单指标要求
空闲显存与按 `GPUExecutionPolicy.max_vram_fraction` 计算的有效预算均至少
8 GiB；短历史三组批次的这两个门槛均为 12 GiB。例如比例上限为 40% 时，
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
`docs/benchmarks/real_cos_f8_ic_ir_20260927.json`；F12 单指标仍不开放，多指标组合须有独立整批证据。
`quantile_returns_daily` 六轮 CPU/CUDA 热运行约 9.50–9.85/5.43–5.89 秒；
`quantile_returns_full` 约 9.47–9.53/5.40–5.41 秒。日序列 CUDA 曾缺失标签身份、
参数和有效期计数等 provenance；补齐后六轮的数值、轴、掩码、计数、完整 provenance
与配置哈希全部对拍通过。两项 GPU 峰值均为 1,436,371,456 字节。证据见
`docs/benchmarks/real_cos_f8_quantile_returns_daily_20260927.json` 与
`docs/benchmarks/real_cos_f8_quantile_returns_full_20260927.json`；F12 单指标和未认证的组合请求仍保持 CPU。

同一 F8 COS 面板的两组完整多指标请求另有六轮交错 CPU/CUDA A/B：
`rank_chain`（`rank_ic`、`rank_ic_series`、`ic_std`、`ic_ir`）的 CPU
热运行约 20.48–20.89 秒，CUDA 约 6.00–6.03 秒；`quantile_chain`
（`quantile_returns_full`、`quantile_returns_daily`、`quantile_spread`、
`quantile_monotonicity`、`daily_quantile_monotonicity_rate`）的 CPU 热运行约
28.32–28.64 秒，CUDA 约 5.44–5.62 秒。每组六轮的完整制品、有效掩码、
计数、provenance、标量结果与配置哈希逐字段对拍通过。自动路由仅对精确
2586 日 × 5461 股 × 8 因子、默认参数及 L20 放行这两个完整集合，
顺序不限；rank 至少要求 14 GiB、quantile 至少要求 8 GiB 有效空闲显存。
子集、含重复项、其它组合和相邻形状继续使用 CPU；F12 需另看下文的独立证据。证据：
`docs/benchmarks/real_cos_f8_rank_chain_20260927.json` 与
`docs/benchmarks/real_cos_f8_quantile_chain_20260927.json`。
独立 F12 COS 面板在精确 2586 日 × 5461 股 × 12 因子形状上，单指标
`rank_ic`、`quantile_spread`、`factor_turnover_rate` 均完成公开入口 CPU/CUDA
A/B 和完整输出对拍。rank CPU/CUDA 热运行分别为 31.607/9.449 秒，GPU 峰值
9,332,757,504 字节；quantile 为 21.612/8.787 秒，峰值 1,436,371,456 字节；
turnover 为 19.510/14.297 秒，峰值 1,020,942,336 字节。这三个指标仅在该精确
形状自动使用 CUDA，并要求 float64、默认参数、无特殊输入、NVIDIA L20；rank
至少需 14 GiB 有效空闲显存，quantile 与 turnover 至少需 8 GiB。未认证的 F12 多指标
请求保持 CPU；三项单指标的回执原因均为 `certified_single_metric_real_cos_f12`。证据见 `docs/benchmarks/real_cos_f12_rank_20260927.json`、
`docs/benchmarks/real_cos_f12_quantile_20260927.json` 和
`docs/benchmarks/real_cos_f12_turnover_20260927.json`。

F12 的完整 `rank_chain` 和 `quantile_chain` 分别完成独立六轮交错公开入口
CPU/CUDA/auto A/B，所有指标的数值、制品类型、掩码、计数、provenance、
MetricValue 和配置哈希逐字段对拍通过。rank 链 CPU 热运行约 30.79–30.89 秒，
CUDA 约 9.66–9.67 秒，峰值显存 9,332,757,504 字节；quantile 链 CPU 热运行
约 47.73–48.06 秒，CUDA 约 8.74–8.75 秒，峰值显存 1,436,371,456 字节。
仅在精确 2586 日 × 5461 股 × 12 因子、float64、默认参数、无特殊输入和
NVIDIA L20 上，对这两个完整指标集合自动选择 CUDA；rank 至少要求 14 GiB、
quantile 至少要求 8 GiB 有效空闲显存，且实际空闲显存也不得低于门槛。
子集、重复项、其它未认证 F12 组合和相邻形状仍走 CPU。证据见
`docs/benchmarks/real_cos_f12_rank_chain_20260927.json` 与
`docs/benchmarks/real_cos_f12_quantile_chain_20260927.json`。

独立 F13 COS 面板在精确 2586 日 × 5461 股 × 13 因子形状上，对完整
`rank_chain` 和 `quantile_chain` 各完成六轮交错公开入口 CPU/CUDA/auto A/B。
每组的数值、制品类型、掩码、计数、provenance、MetricValue 和配置哈希
逐字段对拍通过。rank 链 CPU 热运行 34.12–35.16 秒、CUDA 10.31–10.97 秒，
GPU 峰值 9,332,757,504 字节；quantile 链 CPU 53.23–53.28 秒、CUDA
9.49–9.52 秒，峰值 1,436,371,456 字节。基准前的 `auto` 尚未放行 F13，
因此六轮中的两次 `auto` 均回退 CPU；路由更新后的复测见
`docs/benchmarks/real_cos_f13_rank_chain_auto_20260927.json` 与
`docs/benchmarks/real_cos_f13_quantile_chain_auto_20260927.json`。
现在仅在此精确形状、float64、默认参数、无特殊输入和 NVIDIA L20 下，
对上述两个完整集合自动选 CUDA；rank 要求至少 14 GiB、quantile 至少
8 GiB 有效及实际空闲显存。单指标、子集、重复项、其他组合和相邻形状
不沿用 F13 批次认证。原始证据见
`docs/benchmarks/real_cos_f13_rank_chain_20260927.json` 与
`docs/benchmarks/real_cos_f13_quantile_chain_20260927.json`。

F13 的单项 `factor_turnover_rate` 在 GPU 内部改为有界时间块计算，
消除了逐交易日 Python→GPU 同步。真实 COS 全量 2586 日 × 5461 股 ×
13 因子的六轮交错 CPU/CUDA/auto A/B，完整数值、制品、掩码、计数、
MetricValue、来源和配置哈希对拍通过。CPU 热运行 21.34–21.63 秒，
CUDA 9.37–9.46 秒，GPU pool 峰值 1,129,165,824 字节；基准前两次
`auto` 因未认证而走 CPU。现在仅该精确形状、float64、默认参数、
无特殊输入、NVIDIA L20 和至少 8 GiB 有效及实际空闲显存下，
此单项自动走 CUDA。证据见
`docs/benchmarks/real_cos_f13_factor_turnover_rate_20260927.json`；
改后路由实测见
`docs/benchmarks/real_cos_f13_factor_turnover_rate_auto_20260927.json`。
该指标是分位成员变化率，不是 rank-weight `turnover`，两者不共享认证。

同一绑定 COS manifest 的精确 2586 日 × 5461 股 × 2 因子面板上，
`rank_ic_series` 单指标完成六轮交错 CPU/CUDA/auto 公开入口 A/B。
每轮完整序列制品、有效掩码、计数、逐因子 MetricValue、来源与配置哈希
对拍通过。CPU 热运行 6.16–6.23 秒，CUDA 1.56–1.57 秒；CUDA
GPU pool 峰值 11,234,754,560 字节。基准前两次 `auto` 均因该指标
尚未认证而走 CPU；更新后的公开入口复测见
`docs/benchmarks/real_cos_f2_rank_ic_series_auto_20260927.json`。
现在仅对该精确形状、float64、默认参数、无特殊输入和 NVIDIA L20
的单项请求选择 CUDA，并要求有效与实际空闲显存均至少 14 GiB。
相邻形状、其他 F2 指标或组合不沿用此认证。原始证据见
`docs/benchmarks/real_cos_f2_rank_ic_series_20260927.json`。

F12 另有精确三指标混合请求 `rank_ic`、`quantile_spread`、
`factor_turnover_rate` 的独立六轮公开入口 A/B：CPU 热运行约
53.65–53.95 秒，CUDA 约 16.03–16.23 秒，GPU 峰值
9,332,757,504 字节；三项的数值、制品、掩码、计数、provenance、
MetricValue 和配置哈希均与 CPU 对拍通过。只在相同精确 F12 面板、
float64、默认参数、无特殊输入及 NVIDIA L20 下，对此完整集合自动选 CUDA；
有效空闲显存和实际空闲显存均至少 14 GiB。指标顺序不限，但子集、重复项
或其它组合不沿用此认证。证据见
`docs/benchmarks/real_cos_f12_mixed_three_20260927.json`。

同一绑定 COS manifest 的精确 2586 日 × 5461 股 × 24 因子面板上，
完整 `quantile_returns_full`、`quantile_returns_daily`、`quantile_spread`、
`quantile_monotonicity`、`daily_quantile_monotonicity_rate` 集合通过六轮交错
CPU/CUDA/auto 公开入口 A/B。五项的完整制品、有效掩码、计数、来源、
逐因子 MetricValue 及配置哈希全部对拍通过。CPU 热运行约 112–113 秒，
CUDA 约 18.30–18.31 秒，约快 6.1 倍；CUDA GPU pool 峰值
1,436,371,456 字节。原始证据见
`docs/benchmarks/real_cos_f24_quantile_chain_20260927.json`。
仅对此精确形状、float64、默认参数、无特殊输入、NVIDIA L20，且有效与
实际空闲显存均至少 8 GiB 的完整集合自动选择 CUDA。单项、子集、
混合请求、相邻形状及其他硬件不沿用此认证；改后自动路由实测见
`docs/benchmarks/real_cos_f24_quantile_chain_auto_20260927.json`。
随后只对默认五项完整 quantile 链启用专用显存估算；自定义指标参数和混合请求仍用
原保守估算。独立六轮 A/B 将 F24 因子 tile 从 8 扩至 24、GPU tile 次数从 3
降至 1，CUDA 热运行 17.33–17.36 秒、`auto` 17.43–17.79 秒；CPU
两端约 112 秒，完整指标对拍通过。GPU pool 峰值约 3.25 GB，仍低于会话预算。
该经验估算不是对任意形状的显存保证：单行工作区另做 admission，CuPy OOM 仍按
策略重切 tile。证据见 `docs/benchmarks/real_cos_f24_quantile_tile_ab_20260927.json`。

同一精确 F24 面板的完整 `rank_chain`（`rank_ic`、`rank_ic_series`、
`ic_std`、`ic_ir`）另完成 CPU/CUDA/auto/auto/CUDA/CPU 六轮交错 A/B，
每轮各含冷、热调用。CPU 热调用为 57.97–58.22 秒，CUDA 为
19.76–20.02 秒；CUDA 峰值 9,450,477,056 字节，按 8 因子 tile
运行 3 次。六轮的配置哈希、数值、有效掩码、计数、来源和逐因子
MetricValue 均对拍通过。改路由前两轮 `auto` 都留在 CPU；现在仅对
精确 2586 日 × 5461 股 × 24 因子、float64、默认参数、无特殊输入、
单 NVIDIA L20 且实际及有效空闲显存至少 14 GiB 的完整集合选 CUDA。
子集、其它组合、F32、相邻形状或其它硬件均不继承此认证。改路由后另做完整
六轮交错复测，两轮 `auto` 都实际选择 CUDA，热调用 19.73–20.18 秒，
与强制 CUDA 相近，所有制品对拍通过。实测证据见
`docs/benchmarks/real_cos_f24_rank_chain_20260927.json` 和
`docs/benchmarks/real_cos_f24_rank_chain_auto_20260927.json`。

F32 的完整 rank 链、完整分位链、默认三指标整批及三个单指标请求已分别认证自动 CUDA。只读预检命令：

```bash
python -m quant_evaluator.scripts.benchmark_real_cos_factor_batch --factor-count-profile 32 --preflight-only
```

它只检查绑定 manifest、对象字节数及 RAM/VRAM 余量，不会载入因子数组或执行评估。
`--max-working-gib` 默认 50；提高该上限不能绕过可用内存与预留余量检查。
只有显式加 `--run`，且预检
通过后，才会执行整批多指标 CPU/CUDA/auto 六轮 A/B；可选
`--factor-profile-mode single` 改为逐指标诊断。当前绑定 manifest 有 61 个
符合筛选条件的对象，最小 32 个合计 1,946,203,073 字节。精确内容哈希的
F32 样本按已测最大父/worker RSS 合计加 20% 后上取整为 50 GiB，
另需至少 8 GiB 可用内存余量；其它 manifest 仍用旧保守模型。
现有加载器最多支持 32 个因子，F64（64 因子）
暂不能用该脚本验证。

CPU 参考后端的 Spearman 日 IC 已改为每次最多处理 128 个
`(交易日, 因子)` 行，避免整段历史的哨兵值与秩数组同时驻留；精确路径还直接
把 validity 并入 pairwise mask，不再复制一整份 float64 因子面板。
这不改变统计口径。中等规模固定种子 A/B（600 日 × 2000 股 × 12 因子）
两轮交错测得进程 RSS 约从 1.66 GiB 降至 0.52 GiB，计算耗时约从
1.58–1.60 秒降至 1.33–1.36 秒，输出数组与计数字节哈希相同；
跨 tile、ties、缺失和有效性掩码也与历史逐行参考实现逐位一致。
脚本与原始数字见 `scripts/benchmark_cpu_ic_row_tiles.py` 和
`docs/benchmarks/cpu_ic_row_tiles_20260928.md`。这只是合成证据；
F24/F32 真实全市场复测见下文。

新版加载器在同一绑定 manifest 上单独载入 F32 全历史批次（不运行评估）：
2586 日 × 5461 股 × 32 因子，32 个已校验 COS 对象，日期跨度
2016-01-04 至 2026-08-25，耗时 124.916 秒，父进程 RSS 峰值
13,838,164 KiB。此测量只覆盖数据装载，不单独证明评估阶段的
内存安全、口径或性能；后续分阶段实测见下文。预检已经依据
完整双调用的最大 RSS 校准，资源不足时仍会拒绝运行。

随后在同一 F32 绑定面板上对完整 `rank_chain`（`rank_ic`、
`rank_ic_series`、`ic_std`、`ic_ir`）做 CPU 与 `cuda_strict`
各两次公开入口调用。CPU 冷/热为 81.849/78.405 秒，CUDA 为
38.160/33.895 秒；CUDA 热调用在这次并发环境中约快 2.31 倍。
CPU/CUDA 的配置哈希相同，四项完整制品（数值、有限与有效掩码、
计数、来源及逐因子 MetricValue）均按现有严格比较器通过。
CPU/GPU worker RSS 峰值分别为 23,761,400/25,434,024 KiB，
GPU pool 峰值 9,451,395,584 字节。这次双调用仅为先行探针；
后续交错和 `auto` 复测结果见下文。

随后按 CPU/CUDA/CUDA/CPU 四轮交错、每轮冷/热各一次复测，
CPU 热运行 79.105/78.837 秒，CUDA 34.181/34.349 秒，
约 2.3 倍加速；配置哈希相同，四轮四项完整制品全部对拍通过。
在加入精确路由后，又用一轮 CPU 参考及两轮 `auto` 冷/热调用复测：
两轮 `auto` 均实际走 CUDA，热运行 34.485/34.310 秒，
完整制品均与 CPU 对拍通过。仅对这个精确形状、float64、默认参数、
完整 rank 链、无特殊输入及单 NVIDIA L20 且有效空闲显存至少 14 GiB
启用自动 CUDA；子集、单指标及混合请求不继承本链认证，须各自独立验证。

同一 F32 面板的完整分位链（`quantile_returns_full`、
`quantile_returns_daily`、`quantile_spread`、`quantile_monotonicity`、
`daily_quantile_monotonicity_rate`）另做 CPU/CUDA/CUDA/CPU 四轮交错。
CPU 冷调用 168.197/167.598 秒，CUDA 34.749/34.886 秒，
约快 4.8 倍；五项完整制品逐轮对拍通过。加入精确路由后，两轮
`auto` 均实际选 CUDA（34.682/34.729 秒）且与 CPU 对拍通过。
仅对相同精确形状、完整五项、float64、默认参数、无特殊输入、
单 NVIDIA L20 且有效空闲显存至少 8 GiB 自动选 CUDA。
两条 F32 链的详细请求、耗时、内存峰值及排除范围见
`docs/benchmarks/real_cos_f32_chain_ab_20260928.md`。

同一 F32 面板的默认三指标整批请求（`rank_ic`、`quantile_spread`、
`factor_turnover_rate`）完成 CPU/CUDA/auto/auto/CUDA/CPU 六轮交错，
每轮各含冷、热调用。CPU 热运行 160.171/160.457 秒，强制 CUDA
36.414/36.219 秒，两轮 `auto` 实际走 CUDA、热运行
35.907/35.992 秒，约为 CPU 的 4.45 倍。六轮配置哈希一致，
三个完整制品逐轮对拍通过；GPU pool 峰值 9,451,395,584 字节。
仅对精确 F32 形状、这三个默认指标的完整集合、float64、默认参数、
无特殊输入、单 NVIDIA L20 且有效空闲显存至少 14 GiB 自动选 CUDA。
原始报告见 `docs/benchmarks/real_cos_f32_default_batch_20260928.json`。

同一 F32 面板的 `rank_ic`、`quantile_spread`、`factor_turnover_rate`
又分别完成单指标 CPU/CUDA/CUDA/CPU 四轮交错和两轮真实 `auto` 复测。
三项 CPU 两端分别约 85/80/69 秒，CUDA 分别约 39/36/36 秒；
每项四轮配置一致、完整制品对拍通过，两轮 `auto` 均实际走 CUDA
且与 CPU 参考对拍通过。单指标 `rank_ic` 要求至少 14 GiB 有效及
实际空闲显存，另两项至少 8 GiB；float64、默认参数、无特殊输入及
单 NVIDIA L20 的精确 F32 形状之外仍用 CPU。详见
`docs/benchmarks/real_cos_f32_single_routes_20260928.md`。

新版 CPU 行块与加载器组合在同一 F24 rank 链上另完成六轮真实 COS
公开入口复测，完整制品对拍通过；父进程 RSS 峰值 10,850,236 KiB，
CPU worker RSS 峰值 19,223,060–19,248,388 KiB。两个 `auto` 均走 CUDA。
CPU 热调用约 60.8 秒、CUDA/auto 约 25.2–25.5 秒；与旧轮次速度差异
可能受同机并发 GPU 任务影响，不能据此作独立性能改进结论。
原始报告为 `docs/benchmarks/real_cos_f24_rank_chain_row_tiles_20260928.json`。

自定义分位参数：`n_quantiles=1` 是合法的单桶请求，有限因子值全部进入第 0 桶；
形状单调性此时没有相邻桶，结果为缺失。CPU/CUDA 现在都支持这一口径。
`min_assets` 必须是正整数，零、布尔值、浮点数和字符串均在计算前拒绝，
避免空桶除零及 CPU/CUDA 接受域不一致。

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

调用者可以在同一公开函数中选择执行方式，并检查实际回执：

```python
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy
from quant_evaluator.runtime.evaluator import evaluate

metrics = ("rank_ic_series",)
fastest_certified = evaluate(batch, labels, metrics=metrics)  # 等同 backend="auto"
reference = evaluate(batch, labels, metrics=metrics, backend="cpu")
forced_gpu = evaluate(
    batch, labels, metrics=metrics, backend="cuda_strict",
    gpu_policy=GPUExecutionPolicy(
        device_ids=(0,), max_vram_fraction=0.75,
        oom_retile=True, max_host_result_bytes=256 * 1024 * 1024,
    ),
)
print(fastest_certified.metadata["backend_used"])
print(fastest_certified.metadata["auto_backend_reason"])
```

`auto` 是经过特定硬件、形状、指标集合和默认参数 A/B 认证的确定性选择，
不是对所有 164 项指标的实时穷举测速；未认证请求会回到 CPU。
`cuda_strict` 用于主动验证未认证的 GPU 路径，设备、指标或预算不满足时
直接报错，不静默重跑 CPU。`GPUExecutionPolicy` 的 `max_vram_fraction`
限制会话显存预算，`oom_retile` 控制显存不足时是否缩小因子 tile，
`max_host_result_bytes` 限制一次工作器物化结果的主机内存；增大预算
不代表该形状获得 `auto` 性能认证。当前 `pinned_host_memory`、
`async_transfer`、`double_buffer` 仅是请求字段，实际传输仍为同步、
单缓冲；`bundle.metadata` 的 `effective_transfer`、`effective_pinned`
和 `effective_buffers` 是生效状态。把这些能力列为
`required_capabilities` 会关闭式报错，不会假装启用。

默认调用的 `bundle.metadata["backend_requested"]` 为 `"default"`，显式
`backend="auto"` 为 `"auto"`；显式 `backend="cpu"` 固定参考 CPU 路径。
`backend_strategy` 区分 `"auto"` 和 `"explicit"`，
`backend_used` 为 `"cpu"` 或 `"cuda"`，
`auto_backend_policy` 为策略版本，
`auto_backend_reason` 记录选路原因，
`auto_backend_profile` 区分 701 日、COS 已测核心、有界外推区和精确 F8/F12/F13/F24 区，
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
