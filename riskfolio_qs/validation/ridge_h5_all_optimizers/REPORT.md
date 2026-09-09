# Ridge_h5 全优化器验收报告

## 结论

本次共测试 `riskfolio_qs 0.2.3` 注册的 9 个优化器，结果为：

- 9/9 完成；
- 9/9 健康检查通过；
- 所有凸优化结果均为 `optimal`；
- 所有规则型结果均为 `rule`；
- 无 fallback；
- 无 NaN/Inf 权重；
- 无预算、净/总敞口、个股上下限或已启用硬约束失败；
- 最大求解后约束误差为 `2.77e-6`，低于系统的 `1e-5` 验收容差。

没有发现新的求解器逻辑 Bug。不同优化器之间的主要差异来自风险模型、约束集合和目标函数，而不是异常输出。

生产指数增强的优先选择仍然是
`meanvar_enhance_barra_precomputed`。它直接使用 v1 B/F/D、风险口径最明确，
本次5日总运行时间约1.61秒，并且 v1 Barra 复核的年化 TE 严格落在3%上限内。

## 测试目的与口径

本次是优化器功能验收，不是策略收益回测。测试窗口只选连续5日，以便验证：

- 多日持仓串联；
- 换手硬约束；
- TE 硬约束；
- 个股、行业、风格和市值约束；
- 不同风险模型的一致性与差异；
- 所有优化器在同一输入上的行为是否符合定义。

测试日期：

- 2025-12-18
- 2025-12-19
- 2025-12-22
- 2025-12-23
- 2025-12-24

测试数据：

- alpha：`outputs/Ridge_h5_pred.parquet` 的原始 `pred`；
- alpha 类型：5日预期收益；
- benchmark：真实 `000905.SH` 中证500成分权重；
- Barra：`v1_sbi_fullA`；
- 行情：`lqtp_data/StockDailyBar`；
- 市值：`lqtp_data/StockValuationDaily.MarketCap`；
- 初始持仓：首日 benchmark；
- 可交易状态：5日内始终非停牌且成交量、成交额为正。

为避免指数换仓和停牌干扰优化器检测，使用5日内始终有 Ridge、Barra、行情且可交易的固定股票池：

- 资产数：488；
- 原中证500权重保留率：97.1503%；
- benchmark 在保留股票池内重新归一化到100%。

`y_true` 没有进入任何优化过程，只用于求解完成后的独立观察。

## 原始 Barra 路径的数据质量

`meanvar_enhance_index` 和 `minvar_enhance_index` 需要历史特异收益，而 v1
交付的是预计算特异方差。因此本次使用真实数据重建：

\[
\epsilon_{i,t}=r_{i,t}-B_{i,t}f_t
\]

其中个股收益来自 `StockDailyBar.Return / 10000`，暴露和因子收益来自 v1。

重建结果：

- 每只股票最少有效特异收益观测：242；
- 中位有效观测：252；
- 因子收益全部有限；
- 测试日行情覆盖完整；
- 样本特异方差与 v1 特异方差 Pearson 相关：0.8983；
- Spearman 相关：0.9380；
- 样本方差/v1 EWMA 方差中位数：1.6711。

这说明重建数据能够用于接口和优化器验收，但它不等于 v1 的 EWMA 特异风险。
因此原始 Barra 路径和 precomputed 路径产生差异是合理结果。

## 健康检查结果

| 优化器 | 状态 | fallback | 最大约束误差 | 结果 |
|---|---:|---:|---:|---|
| minvar_enhance_index | optimal | 否 | 3.08e-15 | 通过 |
| meanvar_enhance_index | optimal | 否 | 1.61e-6 | 通过 |
| minvar_enhance_barra_precomputed | optimal | 否 | 1.06e-14 | 通过 |
| meanvar_enhance_barra_precomputed | optimal | 否 | 3.10e-7 | 通过 |
| minvar_enhance_hist | optimal | 否 | 1.11e-16 | 通过 |
| meanvar_enhance_hist | optimal | 否 | 1.26e-7 | 通过 |
| meanvar_absolute_return | optimal | 否 | 2.77e-6 | 通过 |
| topn_long_only_equal_weight | rule | 否 | 0 | 通过 |
| topn_long_short_equal_weight | rule | 否 | 0 | 通过 |

规则型优化器按其自身定义验收：

- TopN long-only：净敞口100%、无空头、单只2%；
- TopN long-short：净敞口0%、总敞口100%、10多10空、单只±5%。

## 核心结果对比

下表的 TE 使用统一的 v1 precomputed Barra 独立复核，因此可以跨优化器比较。
“模型内 TE”则是各凸优化器自身风险模型报告的口径。

| 优化器 | 耗时/5日 | 平均换手 | 平均主动份额 | 有效持仓数 | 预测主动收益5日 | 实现主动收益5日 | v1 TE均值/最大 | 模型内TE均值/最大 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| minvar 原始Barra | 10.39s | 0.00% | 0.00% | 330.0 | 0.000% | 0.000% | 0.000% / 0.000% | 0.000% / 0.000% |
| meanvar 原始Barra | 19.27s | 20.00% | 43.90% | 128.5 | 0.417% | -0.537% | 2.382% / 2.464% | 2.886% / 3.000% |
| minvar precomputed | 1.19s | 0.00% | 0.00% | 330.0 | 0.000% | 0.000% | 0.000% / 0.000% | 0.000% / 0.000% |
| meanvar precomputed | 1.61s | 20.00% | 48.79% | 111.3 | 0.459% | -0.458% | 2.767% / 3.000% | 2.767% / 3.000% |
| minvar 历史协方差 | 9.72s | 0.00% | 0.00% | 330.0 | 0.000% | 0.000% | 0.000% / 0.000% | 0.000% / 0.000% |
| meanvar 历史协方差 | 19.77s | 20.00% | 50.35% | 88.3 | 0.481% | -0.320% | 3.530% / 3.923% | 3.000% / 3.000% |
| meanvar 绝对收益 | 5.86s | 25.00% | 62.95% | 37.5 | 0.670% | -1.430% | 8.616% / 10.137% | 10.073% / 12.368% |
| TopN long-only | 0.20s | 59.11% | 91.43% | 50.0 | 0.745% | -1.031% | 8.735% / 11.272% | 不适用 |
| TopN long-short | 0.30s | 72.65% | 98.37%* | 20.0 | 1.302%* | -4.482%* | 23.060% / 23.990%* | 不适用 |

`*`：long-short 是净敞口0的绝对多空组合，与100%长仓 benchmark
并非同一投资问题；相对 benchmark 的主动份额、主动收益和 TE 仅作风险尺度观察，
不能与长仓指数增强直接排名。

## 各优化器行为解释

### 三个 minvar 优化器

三个 minvar 结果几乎完全相同，最后一日相互 L1 距离小于 `2.1e-7`。

原因不是优化器失效，而是：

- 初始持仓就是 benchmark；
- benchmark 满足全部个股和暴露约束；
- minvar 没有 alpha 收益项；
- \(w=b\) 的主动风险、交易成本和换手同时为零。

因此返回 benchmark 是精确的数学最优解。minvar 适合作为风险基线或可行组合修复器，
不应被视为主动增强策略。

### meanvar_enhance_barra_precomputed

这是本次最符合生产指增定义的结果：

- 5日耗时1.61秒；
- 平均换手20%，命中换手上限；
- 平均主动份额48.79%；
- v1 Barra TE 平均2.767%，最大3.00003%；
- 行业主动暴露接近0；
- 风格主动暴露最大0.10，正确命中风格 band；
- 最大主动个股权重2%，正确命中个股主动上限；
- 预测主动收益均值0.459%/5日。

它直接使用用于独立复核的同一套 B/F/D，所以模型内风险和外部复核一致。

### meanvar_enhance_index

原始 Barra 路径同样正确：

- 换手、个股主动权重、行业和风格约束均有效；
- 模型内 TE 最大约3%，没有硬约束失败；
- 外部 v1 TE 平均2.382%，低于模型内2.886%；
- 最后一日与 precomputed 结果的权重 Rank 相关为0.910；
- 两者 L1 距离为0.477。

差异主要来自特异风险估计：重建的样本特异方差中位数是 v1 EWMA 方差的1.67倍，
原始路径在本次样本上更保守。

### meanvar_enhance_hist

历史协方差路径满足了自身全部硬约束：

- 自身模型报告 TE 为3%；
- 求解状态和数值误差合格；
- 换手20%，主动个股权重2%。

但用 v1 Barra 复核后：

- 外部 TE 平均3.53%；
- 最大3.92%；
- 行业主动偏离平均4.51%；
- 风格主动偏离平均0.133。

这是因为历史路径默认不启用行业、风格中性，而且3%硬约束使用的是历史协方差，
不是 Barra。它适合作为 Barra 缺失时的 fallback，不能被解释为“满足 Barra 3% TE”。

性能也是明显边界：488只股票、5日耗时19.77秒，约为 precomputed
路径的12倍。主要成本来自稠密历史协方差和带二次 TE 约束的锥规划。

### meanvar_absolute_return

结果符合绝对收益定义，但不属于指数增强：

- 个股最大权重5%；
- 换手25%，命中自身上限；
- 有效持仓约37只；
- v1 Barra 相对指数 TE 平均8.62%；
- 行业主动偏离平均10.80%；
- 风格主动偏离平均0.709。

它没有主动权重、行业中性、风格中性或 TE 硬约束，因此高偏离不是 Bug。

### TopN long-only

规则正确执行为50只等权、每只2%，但没有风险和换手约束：

- 平均换手59.11%，最大91.56%；
- v1 TE 平均8.74%，最大11.27%；
- 行业主动偏离平均12.59%；
- 风格主动偏离平均0.733。

适合做纯信号基线，不适合作为受控指数增强组合。

### TopN long-short

规则正确执行为10多10空、每只±5%、净敞口0%、总敞口100%。

它没有 Barra 风险、换手、行业或风格约束。相对长仓中证500计算得到的高 TE
主要来自投资问题本身不同，不能据此判定实现异常。

## 权重相似度

最后一个测试日：

- 三个 minvar 几乎完全一致；
- 原始 Barra meanvar 与 precomputed meanvar Rank 相关：0.910；
- precomputed meanvar 与历史 meanvar Rank 相关：0.762；
- 原始 Barra meanvar 与历史 meanvar Rank 相关：0.713；
- precomputed 与历史 meanvar 的 L1 距离：0.891；
- precomputed 与原始 Barra meanvar 的 L1 距离：0.477。

说明 alpha 方向总体一致，但风险模型会显著改变仓位幅度和具体持仓结构。

## 关于这5日的实现收益

测试期 Ridge 信号的日截面 Rank IC 均值为 `-0.0872`，Pearson IC
均值为 `-0.1601`。因此所有主动增强结果在这5个日期的后验实现主动收益均为负。

这是该时间窗口的 alpha 表现，不是优化器健康度问题：

- 优化器对 `y_true` 不可见；
- 所有组合都提高了模型预测收益；
- 收益方向与事后标签不一致，来自信号误判；
- 5个相互重叠的5日标签不能用于策略绩效结论。

不能为了得到好看的实现收益而事后挑选正 IC 日期。本报告只将实现收益作为
“权重确实响应 alpha”的辅助证据。

## 最终判断

### 通过且推荐

- `meanvar_enhance_barra_precomputed`：当前 Ridge_h5 指数增强首选。
- `minvar_enhance_barra_precomputed`：风险基线/benchmark 修复器。

### 通过但有明确用途边界

- `meanvar_enhance_index`：验证通过，但需要真实 F_spec；已有 B/F/D 时没有必要重估。
- `minvar_enhance_index`：正确，但通常退化为 benchmark。
- `meanvar_enhance_hist`：可作为缺 Barra fallback，性能较慢且不保证 Barra TE。
- `minvar_enhance_hist`：正确，但 benchmark 可行时仍返回 benchmark。
- `meanvar_absolute_return`：绝对收益组合，不是指增。
- 两个 TopN：信号对照组，不是风险受控组合。

本轮没有发现需要修改优化器数学逻辑的 P0/P1 Bug。检测暴露出的主要工程边界是
历史协方差 meanvar 的速度，以及不同风险模型之间不可互换的 TE 语义。

## 复现与产物

复现命令：

```powershell
cd D:\quantsociety\whh_local_workspace\riskfolio_qs
python scripts\validate_all_optimizers_ridge_h5.py
```

关键产物：

- `manifest.yaml`：输入、日期、哈希和测试口径；
- `health_checks.csv`：逐优化器健康检查；
- `optimizer_comparison.csv`：聚合对比；
- `independent_metrics.parquet`：逐日独立 v1 Barra 风险与收益观察；
- `last_date_weight_rank_correlation.csv`：权重秩相关；
- `last_date_weight_l1_distance.csv`：权重 L1 距离；
- 每个优化器目录中的 target positions、trades、summary 和 metadata。
