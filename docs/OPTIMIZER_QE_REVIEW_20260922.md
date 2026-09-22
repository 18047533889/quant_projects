# Optimizer / QE 本轮检查记录（2026-09-22）

正式修改目录：server-c 的 `/home/sunhaiwei/quant_projects`，直接修改 main 工作树，
没有创建分支、worktree 或代码副本。保留其他 AI 的未提交内容。

## 已复现并修复

- 逐方法审计将所有 ValueError 当成“经济指标不可用”，可能把计算错误报告为执行成功。
  现在只豁免 JointMetricsUnavailable；其他异常记为 failed，审计命令据此失败退出。
  非单期和重叠标签明确标记 unsupported_label_accounting，不冒充已有组合记账能力。
- LabelBundle 原来接受布尔、非整数及 NaN 持有期/执行延迟；现在要求整数。
- HoldingReturnPanel、ProbePortfolioArtifact 及 from_prices 的收益/价格输入现在明确只接受
  有符号整数、无符号整数和实数浮点数组；拒绝布尔、复数、对象与时间差数组。
  尤其 from_prices 不再通过先转浮点数而丢失复数虚部或掩盖非法输入。
- KAMA 的路径波动和方向计算改用 NumPy 视图及切片，递归状态与缺失值规则保持不变。
  真实 TRAIN 面板逐元素一致，内核中位耗时 0.509380 → 0.163070 秒；
  这是指定样本上的内核提速，不是全优化流程的倍数保证。

## 已取得的验证证据

- 修复后的 FO + FP 全量：1891 passed、1 xfailed、16 warnings，99.68 秒。
  唯一预期失败是非因果 HP 滤波，仍禁止生产使用。
- QE 修改前完整基线：1310 passed、2 skipped，103.86 秒。
  这是基线，不作为修改后测试已通过的证明。
- QE 输入合同最新定向回归：29 passed；from_prices 绕过和 timedelta 类型绕过
  均分别先复现 3 个失败案例，再验证修复。
- 两只真实 COS 因子，各 500 日 × 256 资产。130 个逐方法案例：
  114 executed、16 requires_additional_inputs_or_control、0 failed。
  只用 TRAIN 选择资产和搜索，用 VALIDATION 确认赢家，TEST 未评分。
  一只通过倒 U 修复，另一只未通过验证退化门槛而保留 RAW，没有验证重试。
  原始来源哈希与逐类计数见
  [结构化结果](../factor_optimizer/docs/runtime_review_real_20260922.json)。

## 文档与边界

- 三个统计指标是否实现，以公开运行入口及当前注册表为准，不能依据旧覆盖表。
- 序列结果从 bundle.artifacts[id].values 读取。
- int64 是数值坐标的安全示例，不是全局唯一允许类型；不可变标量对象轴仍按原合同支持。
- 显式持仓批次组合路径需要 PortfolioSpec 与 HoldingReturnPanel；
  legacy 单期 long_short_returns 路径与之不同，不能混称所有组合请求都需要同样输入。
- 回撤等组合指标要求 typed ProbePortfolioArtifact；两阶段请求需沿用返回序列的轴。
- 本次不承诺“没有任何 bug”或未来收益，不发布生产因子；历史 PIT 暴露、外部控制输入、
  因子生产准入和全平台其他模块仍有独立验证边界。
- 永久禁止 AI 使用任何重置卡：已写入 Mac 及 server-c AGENTS.md，只能用户手动使用。

## 最终核验与统计内核提速

- QE 修改后独立完整回归：1340 passed、2 skipped、10 warnings，101.97 秒。
  两个跳过项为 build/lib 尚无对应算子的契约占位，不是假装已测通过。
- 164 个注册指标的公式手册已重新生成并通过一致性检查。
- bootstrap 采用有界批量索引，保留随机抽样及浮点归约顺序。主代理用提交
  5767f2e41 的旧函数在内存中复核：500 × 8、1000 次抽样、块长 13，
  old/new/new/old 三轮中位 0.495446 → 0.015760 秒，区间逐位一致。
  当时有 QE 测试并行运行；该 31.44 倍是内核基准，不是端到端速度保证。
- 尝试一次默认 pytest 同时收集三个库时，已有同名 test_review_regressions
  模块产生 import-file mismatch；随后按库范围运行，取得上述完整通过结果。
  未删除缓存来掩盖问题，也不宣称默认整平台测试已全部通过。
- 独立只读复核未发现新增可操作问题。GPU、PIT 上游和生产实盘一致性不由
  本轮 CPU 合同/内核回归代替。

发布必须使用 push_both.sh 的预检和正式入口，仅提交本轮明确列出的文件。
个人总仓库接收已提交快照，HKUST 各库只接收同名子目录；其他 AI 的未提交
内容不上传。上传成功须另以 logs/push_both_result.json 的根提交与 13 个
sourceTree / verifiedRemoteTree 一致性为证，本报告本身不是上传成功回执。
