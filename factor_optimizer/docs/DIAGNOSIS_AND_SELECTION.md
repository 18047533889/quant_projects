# 训练诊断与自动候选的衔接

## 两套指标不可混用

`diagnose_training_batch` 保留原来的二十分层描述：
`long_short_sharpe` / `long_short_max_drawdown` 来自最外两层的
0.5 × (顶层平均收益 − 底层平均收益)，不扣成本；用于检查层间结构，
不是最终选优组合。该字段的已有含义未改。

新增 `selection_portfolio` 使用与自动选优完全相同的五分位、股票等绝对
权重、100% 总名义敞口、成本、建仓及空腿策略，固定按 252 日年化。
包含 RankIC、ICIR、成本后 Sharpe、回撤、平均全名义换手和最差分块 Sharpe。
数值由 QE 及研究选优公共实现计算，不另写一套指标公式。
原有已算出的 IC 直接复用，收益与换手仍按当前输入计算。
公式见 [联合选择](JOINT_SELECTION.md)。

缺失、比率不可定义、多期或重叠标签等情况明确输出 status=unavailable
及 reason，不补零，不悄悄用二十分层描述代替可执行口径。

## 换手与空仓规则保持一致

分层衰减中的 full_notional_turnover 只由信号决定，不由未来收益的可用性决定。
现在使用配置中的 research_empty_leg_policy，与选优保持一致。
在 signal_cash 下，由平值导致空腿的已知空仓日仍计入完整时间轴，
退出和重新建仓的换手均计入；全常数或不足样本的信号仍不可用。
unavailable 仍复现旧的空腿拒绝规则。

turnover_policy 与 turnover_unavailable_reason 显式记录策略和原因。
高换手的固定描述阈值仍为平均全名义换手 > 0.5，没有为真实样本调整。
缺少足够二十分层样本，并不意味着独立的五分位换手也不可计算。

## 诊断如何进入候选记录

自动流程在接受基础预处理后诊断其输出，标记 input_stage=accepted_baseline；
否则标为 raw。所有诊断仅使用 TRAIN。每个候选记录 diagnosed_issues，
关联对应的方法家族：

| TRAIN 问题 | 关联研究候选 | 边界 |
|---|---|---|
| high_turnover | CAUSAL_SMOOTHING、DECAY_REFINEMENT | 并非平滑必然改善收益 |
| negative_rank_ic | SIGN_ORIENTATION，以及包含反向的组合候选 | 负均值本身不是统计准入 |
| nonmonotonic_twenty_layer_profile | 已诊断的 U 或倒 U 修复 | 仍要求足够层内样本和时间块一致性 |
| unstable_ic_direction | 不直接指定修复 | 记录风险，不凭现象猜因果 |
| negative_worst_block | 不直接指定修复 | 用既有稳定性门槛评估 |
| joint_metrics_unavailable | 不直接指定修复 | 先检查数据和指标可用性 |

issues 中保留数值、阈值或原因。没有关联标记的候选属于其他预设探索，
不是声称每次尝试都已找到明确病因。本轮不改变候选预算、选择效用、
退化底线或验证重试规则；只把既有诊断与实际选优口径和候选记录接通。
不能把“被某种诊断关联”误写成“已经修复了这个问题”。

VALIDATION 仍只确认 TRAIN 的唯一赢家；TEST 不进入诊断或候选选择。

## 真实样本与验证边界（2026-09-22）

最终优化库＋预处理库回归：1785 passed、1 xfailed、16 warnings，137.88 秒。
预期失败是已隔离为 OFFLINE_ONLY 的非因果 HP 滤波前缀不变性测试。
API 文档已重新生成，源码一致性检查通过。
真实回放结构化证据见 [JSON](costed_diagnosis_real_20260922.json)。

通过 DataAccess 清单绑定读取 `weekly_db5616cd85a477b3`，500 日 × 256 股票。
接受基础预处理后的 TRAIN 平均全名义换手为 0.029435861423220973；
诊断与联合选优入口计算的六项指标逐项相等。TRAIN 的负 RankIC 与
分块方向不稳定均明确记录，不能只看较高的 TRAIN Sharpe 就判为优质。
这份样本的训练赢家未获验证确认，最终保留 RAW，没有查看 TEST 分数。

回归包含手算建仓、退出、重新建仓的累计换手 3，未来标签污染不改变
TRAIN 诊断，以及两个随机种子的“反向＋平滑”组合问题标记。
逐方法历史回放及不适用案例见 [方法审查](METHOD_AUDIT_20260922.md)。
缺少时点可信的暴露等必要输入的方法仍不可宣称已经真实验收。

平台根测试本轮仍有 14 个收集错误，模块清单见上述方法审查的
“全仓库验收中的现有问题”；这不等于优化库回归失败，也不能描述为全平台通过。
