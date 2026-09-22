# 联合指标不可用与分段稳定性检查

## 已确认的两个问题

1. 正常的统计证据不足与参数/实现错误共用 ValueError，导致验证指标不可用时
   被笼统归为 error_raw_retained，自动诊断无法区分原因。
2. worst_block_sharpe 使用 Python min。若第一个分段夏普有限、后续分段为 NaN，
   min 可以忽略后面的 NaN，错误给出完整的最差分段指标。
   合成面板中依次让三个时间段的组合收益为零：原实现第一段正确拒绝，
   第二、第三段均错误通过；对应回归 2 failed、1 passed。

## 修复合同

summarize 新增 JointMetricsUnavailable（仍为 ValueError 子类，兼容旧调用）：

- insufficient_observations：少于 20 行；
- missing_observations：存在不可填补的非有限观测；
- undefined_ratios：计算后的指标非有限，metrics 明确列出指标名。

错误的数组形状、非法资本尺度/负换手仍是普通 ValueError，不伪装成证据不足。
三个时间段均须具有可用夏普，任何一段不可用则 worst_block_sharpe 不可用；
不忽略、填零或删除该段，训练准入、验证和重采样统一使用该规则。

验证只捕获 JointMetricsUnavailable，记录 METRICS_UNAVAILABLE、
unavailable_code、unavailable_metrics、unavailable_role（raw/candidate/bootstrap）和原因。
保持 raw_retained，不重选、不产生收益通过声明。其他异常仍为 error_raw_retained。
重采样出现坏样本时终止该比较，不能丢掉坏样本以美化置信区间。
TRAIN 赢家身份与训练分数继续保留以便审计，TEST 不参与选择。

## 测试

类型测试最初 5 failed；加入类型后，验证行为仍有 1 failed、4 passed，
复现分类问题。重采样分支另有 1 failed、1 passed，修复后均纳入。
完整针对性测试 10 passed，覆盖三个分段位置、三种证据不足、
普通实现错误不被吞掉、只执行一次冻结验证方案、原值保持。

旧下跌冲击复验脚本中的“排名基础方案必须接受”断言在新规则下失败。
这是需要据实更正的旧证据，不应修改指标门槛让它重新通过。
随后用不预设改善结论的三因子逐方法回放与默认优化重新核验；
完整回归：1871 passed、1 xfailed、16 warnings，101.31 秒；HP 非因果滤波仍为
生产禁用的预期失败。三只真实因子合计 195 方法案例：171 executed、
24 requires_additional_inputs_or_control、0 failed。

真实结果见 [JSON](joint_availability_real_20260922.json)：
- weekly_4cbd7ca6dccc61dc 倒 U 型修复仍通过验证，验证 Sharpe -0.1774 -> 0.8771，
  最大回撤 9.55% -> 7.72%；最差分段仍为负，不能说净值稳定单调上涨。
- weekly_db5616cd85a477b3 仍未通过验证退化门槛，保留 RAW。
- downside_shock_accumulation_asym 的 RAW 训练联合指标本身含不可用分段比率，
  排名基础方案现被正确拒绝，未选择验证赢家，不是宣称排名算子损坏。
  WINSOR_BASELINE_FALLBACK.md 的旧“排名训练接受”证据基于会忽略后续 NaN 的
  指标实现，不能继续作为当前准入证明；回退机制仍保留并有合成回归验证。

所有真实回放均不评分 TEST，不发布生产因子。本次不宣称提速或全局无 bug。
