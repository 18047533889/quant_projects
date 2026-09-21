# RankIC 系数路径性能核验

本调整不改变 Spearman 的数学定义。先保留因子与标签同时有限的资产，
再分别计算平均并列秩，最终计算两个秩向量的 Pearson 相关系数：

```math
IC_t=\mathrm{Corr}\left(R(x_{t,I_t}),R(y_{t,I_t})\right),\qquad
I_t=\{i:x_{t,i},y_{t,i}\ \mathrm{are\ finite}\}.
```

最低有效资产数和常数截面返回 NaN 的规则不变。有限极大值不先做差来
判断常数，避免不必要的溢出。二元非恒定信号仍有定义，不随意判无效。

旧路径调用完整 scipy.stats.spearmanr，最后丢弃 p 值。
新路径直接使用 SciPy average rankdata 与同布局的 NumPy corrcoef，
保留 SciPy 的 [1, 0] 提取位置和舍入顺序，只省去未使用的 p 值检验，
以及排名前为判断常数所做的重复排序。

显著性检验、bootstrap、选择阈值、有效掩码、缺失规则、训练／验证／测试
隔离均未改变。本函数仍只提供系数；不得把没有 p 值误解为显著。

专项对照涵盖 2、3、7、64、256 个资产，float32、float64、int64，
随机值、并列值、常数、成对缺失和极端有限值。与 SciPy 系数使用完全
相等断言，而不是放宽误差容忍度掩盖变化。

## 真实端到端 A/B（2026-09-22）

通过 DataAccess 读取 weekly_db5616cd85a477b3；500 日 × 256 股票，
94 候选，99 次 bootstrap。两种实现各预热一次，三组交替运行。
中位耗时 16.6194 → 13.6843 秒，减少约 17.7%；优化输出、有效掩码、
计划身份及全部因子／候选记录完全一致。不是仅测一行相关系数的局部速度。
不含数据读取、首次 FE 加载，也没有 TEST 评分或删减候选。
[计时和等值核验摘要](rank_ic_coefficient_real_ab_20260922.json)。

QE 全套 1286 passed、2 skipped、10 warnings（103.39 秒）；两项跳过为
test_metamorphic.py 中 build/lib 未实现的 zscore/standardize 和 dropna/row-drop
文档契约测试，不将其计为通过。
优化库＋预处理库 1800 passed、1 xfailed、16 warnings（116.38 秒）；
HP 非因果滤波仍为 OFFLINE_ONLY。
平台根测试仍有 14 项既有收集错误（56.99 秒），
[模块清单](https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer/blob/main/docs/METHOD_AUDIT_20260922.md)。
