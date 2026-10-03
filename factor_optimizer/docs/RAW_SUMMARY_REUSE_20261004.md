# TRAIN RAW 指标摘要复用（2026-10-04）

## 行为与范围

`optimize_factor_batch` 在每次调用内创建一个 `RawMetricSummaryCache`。它只用于
TRAIN 联合选择时 RAW 序列的 `research_fitness.summarize` 结果：RAW baseline 分支
和候选评估分支共享该缓存。候选序列仍直接调用 `summarize`；VALIDATION 上的
RAW、候选序列及 bootstrap 逻辑也保持直接计算，不经过此缓存。

缓存键精确绑定序列的 shape、dtype、连续化后的内容字节，以及
`periods_per_year` 的类型和值。相同内容、shape、dtype 和年化参数才能命中；
同一对象被修改后会形成不同键。对象 dtype 不缓存。返回的指标映射是副本，调用方
修改返回字典不会修改缓存中的摘要。缓存只保存成功摘要，异常会原样传播且不会
被缓存。

缓存最多保留一个条目。面板大于 128 KiB 时跳过键构造和缓存，直接执行原摘要函数；
这避免为大面板保存或复制内容字节。128 KiB 限制只约束可缓存面板的内容字节，
不是输入数组、临时查询/计算工作区或进程 RSS 的上限。缓存随本次优化调用结束，
不会跨 batch、split 或 optimizer 调用共享。

## 验证与限制

截至 2026-10-04，缓存单元用例 10 passed，优化器级缓存/未缓存对照用例 3 passed；
专项两文件合计 13 passed。优化器级覆盖包括：六个 ROBUST_SCALE 候选重复 RAW
摘要的缓存/未缓存一致性；同形状、但 TRAIN 缺失位置不同的两个因子，验证缓存不会
把一个因子的 RAW 指标摘要错用于另一个因子；以及 TRAIN 标签自然缺失一整日时，真实
`JointMetricsUnavailable` 在六个候选的 RAW 摘要尝试中均保持未缓存，候选账本、RAW
保留状态、诊断和输出与未缓存路径一致。
相关 `research_fitness` / `research_batch` 回归用例 43 passed，联合指标可用性、
baseline、边界和 paired 路径用例另有 63 passed。缓存/未缓存对照检查选择状态、计划
身份、收益、VALIDATION 下界、候选账本、联合诊断和优化后数值保持一致。一个合成
重复 RAW 摘要场景的底层摘要调用数由 13 次降到 8 次；这是调用次数证据，不是墙钟
加速结果。后续真实 F16/T500/N256 的固定 ABBA 对照已完成，输出与候选账本一致：
每次 RAW 摘要计算由 1777 次降到 33 次（1744 次命中）。墙钟结果混合：一对降低
9.7799%，另一对反而增加 3.7316%，不能据此宣称稳定加速或最快。
详见 [真实对照结果](benchmarks/RAW_SUMMARY_ABBA_RESULT_20261004.md)，其中明确区分
实际测量源码与后续资格检查脚本，不把旧测量归属于新脚本。
后续 FullFO 会话 3009 实际运行 1927 项：1920 passed、7 failed、16 warnings。
七项失败集中在旧 loader 测试替身和过期 API 文档；修正后的相关专项 39 passed。
修正及内存准入实现稳定后，会话 40300 完整回归退出 0：1935 passed、16 warnings，
174.91 秒。loader、cohort、research_batch 与 summary_cache 的前后源码指纹一致。
这证明该次完整回归通过，不代表任意输入无 bug 或已达到最快后端。

实现与测试：`factor_optimizer/research_summary_cache.py`、
`factor_optimizer/research_batch.py`、`factor_optimizer/tests/test_raw_metric_summary_cache_oct04.py`
和 `factor_optimizer/tests/test_research_raw_summary_reuse_oct04.py`。
