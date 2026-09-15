# 持久化预算预留金额核对

2026-09-13：直接修改 server-c 正式 main 工作树；未 commit/push，未操作生产数据。磁盘剩余约 775 GiB。

## 缺陷与修复

DurableBudgetTracker 的 commit_evaluation / release_evaluation 忽略 reserved_cost，错误金额、bool、NaN、None 仍可结算或释放有效预留。新增 8 个反例，修复前全部失败。

Facade 现先校验金额，再传给 SQLite store；金额匹配与状态变化同在 BEGIN IMMEDIATE 事务中检查，错误输入不会更新账本。保留直接 store 调用不传 expected_reserved_cost 的兼容接口，保留 STARTED 才能结算、RESERVED 才能释放的规则。

## 验证

- reserved_amount_before：8 failed；前后源码摘要一致。日志 SHA256：36cf53785069f43d3383e6d953a0cc724370df57768832f131a731e8aa559291。
- reserved_amount_final：预算和搜索限定回归 855 passed，12 warnings，无跳过；运行期间源码无变化。
- 源码摘要：eed63c13f75e52d1f09b117db8c78e53e663bb738867ca245ba9c6d00658a3c8。
- 日志 SHA256：19319da3c2cede4bc85c4bd667f8d8712dcba327b7506f7aea6dcdd916f4b844。
- git diff --check 通过。HEAD 为 e94ac507d670fd1c16b1d6a63fc5d6286daa5970；工作树已有未提交改动，均保留。

仅使用 pytest 小型临时 SQLite。未验证生产负载，也不宣称所有预算边界或整个项目已闭合。后续继续检查预留重复 attempt_id 的参数一致性、lease 数值边界，以及 OPEN_FINDINGS.md 中未完成的身份编码迁移。
