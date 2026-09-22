# 低覆盖因子隔离（研究批量入口）

## 行为
COS 示例 load_cos_sample 与命令行默认 coverage_policy="isolate"：
在既定时间分割的 TRAIN 日期上，为每只因子计算每资产有限值覆盖率，
统计覆盖率至少 90% 的资产数。如果数量小于请求的 n_assets，
将该因子记录为 insufficient_training_coverage，不送入本轮优化；
保留覆盖足够的因子，按原来共同资产规则继续。

不会降低覆盖门槛，不会依据收益隔离因子，也不会从清单中另挑替补。
不改变训练/验证/测试区间来挽救因子，未评分 TEST。
隔离不等于删除因子、质量封禁或判定它没有预测力，只表示它不能满足当前样本要求。

## 报告
inputs.sources 保留所有已读取并绑定来源的因子；
inputs.retained_factor_ids 列出实际进入 FactorBatch 的因子；
inputs.quarantined_factors 记录因子名、eligible_assets、required_assets 和原因；
inputs.train_eligible_assets 给出所有因子的训练覆盖资产数。
执行用的 lineages 和 FactorBatch.factor_ids 同步过滤，避免错配。
automatic 只包含进入优化的因子，不能把隔离记录统计为优化成功。

## 兼容与失败边界
使用 --coverage-policy strict 或 coverage_policy="strict" 可恢复原来的整批拒绝行为。
所有因子均不足时明确报错，并附每只因子的覆盖数，不能返回空的“成功”结果。
剩余因子各自足够但共同资产不足时仍报错，不通过贪心丢弃其中某只来凑样本。
本轮隔离覆盖不足，不吞掉 DataAccess 来源校验错误、非法数据或重复日期错误。
日期对齐仍使用请求样本共同日期；缺少共同日期的因子尚未实现单独分组执行。
因此这不是任意异构因子的完整分组调度。

## 回归反例
修改前，含一只全缺失因子的批次使正常因子也失败。
针对性测试修复前 3 failed、22 passed；修复后 25 passed：
- 合格因子保留、缺失因子隔离，来源记录不丢失；
- VALIDATION/TEST 段缺失不影响训练覆盖判定；
- 全部缺失明确报错；
- strict 保持整批拒绝；
- 资产选择仍使用 warmup/purge 后的同一 TRAIN 分割。

## 真实 COS 复验
仍请求新清单的三因子、256 资产、500 日，默认 bootstrap 499 次。
weekly_b1258507388baa03 和 weekly_da68888feb0a6a1a 各仅 66 个资产达到训练覆盖门槛，
被隔离并保留来源/原因；volume_state_persistence 按 256 资产继续。
该因子 103 个候选，验证触发 DEGRADATION_FLOOR_FAILED，最终 RAW retained。
没有修改覆盖率门槛或按验证收益重选资产，TEST 未评分。
见 [真实证据](cos_coverage_isolation_20260922.json)。
这证明批量入口可以隔离已识别的低覆盖案例，不代表被隔离因子已完成优化，
也不证明任意异常都可自动恢复。本次不是受控性能 A/B，不声明加速倍数。

完整优化库与预处理库回归：1840 passed、1 xfailed、16 warnings，105.56 秒。
唯一预期失败仍为 HP 非因果滤波，限制 OFFLINE_ONLY。
全仓库此前 14 项收集错误本轮未修复，不宣称全平台通过。
