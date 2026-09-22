# COS 资产筛选与 TRAIN 边界（2026-09-22）

## 问题与修复
示例 load_cos_sample 原先按前 300 个日期筛选资产覆盖率，而默认优化器
从前 60% 时间段中剔除 30 日预热及标签窗口跨界/embargo 日期，
500 日样本最终只有 267 个 TRAIN 日期。
没有使用 TEST 覆盖或收益，但“仅按同一 TRAIN 筛选”并不严格成立。

现在先用 decision_time、label_end_time 元数据调用 automatic_time_split，
然后只用返回的 train_indices 计算覆盖率并选择资产。此时无需标签收益值。
provenance 记录 asset_selection_split 与 asset_selection_train_days，
与随后优化器的划分身份核对。本示例使用默认分割；自定义分割的调用方
必须在资产筛选和优化阶段使用一致的规则，不能复用已选宇宙冒充重新筛选。

覆盖筛选仍要求每因子在 TRAIN 上至少 90% 有限值，
按最小覆盖率、平均覆盖率及资产名确定顺序，不参考验证收益。

## 反例与回归
集成测试在真实示例入口构造 a、b 两只资产：a 仅预热段缺失，
b 在实际 TRAIN 缺失。旧实现错选 b；修复后选 a，
并验证报告中的 split 身份与真正优化器一致。
只替换外部 COS/行情边界，日期对齐、资产筛选、FactorBatch/LabelBundle 实际执行。
修复前 1 failed、10 passed（资产结果断言失败）；修复后 11 passed。
完整 FO+FP 回归：1826 passed、1 xfailed、16 warnings，103.57 秒。
预期失败为非因果 HP 滤波，仍限制 OFFLINE_ONLY。未声称全仓库通过；
上一轮全仓库收集仍有 14 个错误，本轮未修复它们。

## 真实 DataAccess 回放
同一已绑定清单的两只因子、500 日 × 256 资产、2024-08-02..2026-08-25。
此次真实样本旧/新宇宙相同（增减均为 0），不能把收益变化归功于此次边界修复。
两只因子各 103 个候选；TRAIN-only 搜索；仅验证冻结赢家；TEST 未评分。
99 次 bootstrap，没有根据验证表现调整参数。
资产筛选 split 与优化 split 完全一致，TRAIN 为 267 日。
完整来源 SHA256 和指标见 cos_train_universe_20260922.json。

| 因子 | 验证结果 | RAW → 候选多空 Sharpe | RAW → 候选最大回撤 |
|---|---|---:|---:|
| weekly_4cbd7ca6dccc61dc | 接受倒 U 修复 | -0.1774 → 0.8771 | 9.55% → 7.72% |
| weekly_db5616cd85a477b3 | 退化门槛失败，保留 RAW | 0.8389 → 0.5372 | 9.62% → 9.39% |

第一只验证 RankIC -0.02155 → 0.01376，RankICIR -0.33569 → 0.21219；
最差分段 Sharpe 仍为 -1.16995，不能宣称曲线始终向上或未来收益确定。
成本率 0.001，缺失中性化暴露明确记录，尚未进行最终 TEST 认证。

## 耗时边界
本次 DataAccess 加载 7.13 秒；两因子自动优化共 61.58 秒。
这是单次共享工作树观测，不是控制变量性能 A/B，不支持新的加速倍数结论。
本修复目标为隔离正确性；已有速度 A/B 参见 CLASS_SOURCE_CACHE_PERFORMANCE.md。
