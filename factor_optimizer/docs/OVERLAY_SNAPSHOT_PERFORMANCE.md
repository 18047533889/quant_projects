# 初始化证据快照复用（2026-09-22）

## 改动范围
优化器调用预处理执行器时会触发 FactorEngine 冷启动。真实因子 profile 中，
production_certification_overlay 对同一份 factor_operator_verified.json
反复读取/解析；1749 次读取调用累计约 6.25 秒（profile 插桩下，不能当作无插桩耗时）。

现在每次 apply_evidence_certification_overlay 只读取一次 per-gate 证据，
将同一份局部 payload 传给该次算子循环。下次 overlay 重新读取。
没有全局证据缓存，不缓存生产准入结论，不取消物理实现和各独立证据门槛；
未给 _per_gate_verified 传 payload 的单次调用保持原来的重新读取行为。
读取失败或坏 JSON 仍得到空证据，缺少独立证据不得提升认证等级。

每次调用使用一个一致的文件快照，不在循环中观察不同版本的 gate 字段。
这是启动重复工作的消除，不改变平滑公式、因子值、评分或搜索网格。

## 回归验证
新测试先复现两算子读取两次（expected 1, actual 2），修复后通过。
它通过真实 JSON 文件验证跨次更新、损坏 JSON、不通过底层认证时的拒绝行为，
并确认 runtime 执行证据不会自动升级独立 semantic/source 证据。
测试中底层认证授权边界受控；生产认证函数未改。

优化库与预处理库完整回归：1841 passed、1 xfailed、16 warnings，99.42 秒。
HP 非因果滤波为预期失败，仍限制 OFFLINE_ONLY。
FactorEngine 新用例及 r25 evidence independence 组合：6 passed、1 failed。
失败为 test_injectivity_is_not_hardcoded_false；单独运行也失败，
在同一当前工作树里切回旧 overlay 实现路径后同样失败。
这不是干净历史 checkout 的验证，也未修复该参数有效性问题。
全仓库此前 14 项收集错误本轮未解决，不宣称全平台通过。

## A/B 设计
A 从 df24b9ebe 读取旧 overlay 模块代码到内存执行；B 使用当前实现。
未复制仓库、未创建分支，其他代码和 DataAccess 数据读取保持一致。
顺序 A/B/B/A，每次独立进程冷启动。
因子 weekly_4cbd7ca6dccc61dc，经 DataAccess 从绑定 COS 清单读取；
500 日 × 256 资产，267 TRAIN 日，默认 499 次 bootstrap，103 候选。
计时覆盖 optimize_factor_batch，不包含事先加载数据的耗时。
不评估 TEST；不按验证收益调参。

验证候选记录、计划身份、训练及验证指标、最终输出数组摘要，
以及 1823 个 catalog 条目的独立证据/顶层认证标记摘要。
不将摘要检查夸大为所有运行环境的形式化等价证明。
共享工作树包含其他 AI 未提交的 FactorEngine 工作，计时也受服务器负载影响；
不宣称全局最优性能、每次固定提速或干净发布快照的全引擎认证。

## 实测结果

| 顺序 | 旧实现 A / 秒 | 新实现 B / 秒 |
|---|---:|---:|
| A/B | 46.83385 | 39.73852 |
| B/A | 46.30491 | 40.01855 |
| 中位数 | 46.56938 | 39.87854 |

本次计时范围耗时减少约 14.37%。四次候选记录、输出数组、上述认证标记摘要均一致。
新快照测试单独复验 1 passed；不把已记录的参数有效性失败改成跳过。
原始数据及摘要见 [JSON](overlay_snapshot_real_ab_20260922.json)。
