# TRAIN 正负方向候选的 RankIC 复用

你调用 `optimize_factor_batch(..., allow_research=True)` 时，默认配置允许方向与平滑组合。搜索按相邻顺序评估同一平滑计划的正、负方向。库在正方向计算后缓存负方向的 RankIC，再按原规则评分两个候选。

## 计算口径

在 RAW、候选和标签共同有效的同一截面上，令平均并列排名为 $R(x)$，有效资产数为 $n$。反向候选满足：

$$R(-x)=n+1-R(x),\qquad \operatorname{RankIC}(-x,y)=-\operatorname{RankIC}(x,y).$$

常量或观测不足的截面仍保留 NaN。候选的正负方向共享单元覆盖率、有效 IC 日期和 RAW_FULL 日期基准；它们各自保留原先的分数、候选身份和失败记录。

有限 IC 中出现零时，库保留原计算路径。直接计算和对结果取负可能产生不同的 IEEE 正负零；库为保持 canonical 证据的逐位一致性，跳过该候选的反向缓存。

## 复用边界

当前入口仅为 `CAUSAL_SMOOTHING` 和 `DECAY_REFINEMENT` 的相邻方向提案启用复用。库沿用内容哈希检查候选值、共同掩码、标签内容及时间坐标和 `minimum_assets`，在负方向请求中仍计算键并核验。改变输入会产生缓存未命中。

`PairICCache` 在一次批量调用中至多保存两条 IC 序列，并复制读写数组。库不缓存整个候选面板网格。临时掩码和键计算仍需要与 TRAIN 面板大小成比例的内存。

成本后收益、换手和联合评分继续各自计算。验证阶段只评估训练选出的冻结赢家；库不复用 TEST 标签，也不从验证失败后的第二名重新选择。

## 配置

你可以通过 `BatchOptimizationConfig(compose_smoothing_sign=False)` 禁用方向组合。显式 `families` 不包含 `SIGN_ORIENTATION` 时，库也不追加方向组合。开启组合后，正负提案都计入 `maximum_candidates`；超出预算时，库整因子保留 RAW。

## 验证范围

回归测试比较直接计算与缓存计算的 IC、NaN 掩码、计数和内容键，并覆盖并列值、有效性掩码及扩展精度标签。运行时仍使用 QE 的 canonical `exact` RankIC 口径，未将其他浮点后端的证据混入同一身份。

局部性能测试仅衡量配对 IC 计算。完整优化还包括变换落值、诊断和组合评分；你应按自己的候选数和股票规模测量端到端耗时。
