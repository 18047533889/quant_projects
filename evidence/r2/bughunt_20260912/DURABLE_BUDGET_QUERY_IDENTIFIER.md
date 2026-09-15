# 预算查询标识混淆（2026-09-14 02:07 heartbeat）

server-c main正式原树，剩774 GiB。保留已有修改，无分支/commit/push/部署或生产数据操作。开始时FE/DA任务idle；测试期间观察到FE报告文件并发修改，未触碰其文件。

budget_state及DurableBudgetTracker.has_reservation接受int/bool标识，经SQLite隐式转换误命中字符串"1"。现在查询前校验campaign_id及attempt_id；has_reservation保留None/空字符串返回False的兼容行为，其他非字符串或纯空白标识拒绝。合法状态查询行为不变。

7项小型临时SQLite测试：6项类型别名反例，以及合法缺省/缺失/预留/启动/结算查询生命周期。

- query_identifier_before：6 failed / 1 passed，源码稳定；日志SHA256 7f70d92966f074d1d416326c663e03a9f8d9c7d9699724dafb0d741df894f2f2。
- query_identifier_final：910 passed但SOURCE_CHANGED_DURING_RUN，不作为稳定PASS；并发文件factor_engine/docs/reports/2026-08-23/report_workflow.py。日志SHA256 481d234105939edbd5b08e147b362d136d26382f3dbd24e89464b4ab0421f0db。
- query_identifier_stable：910 passed / 12 warnings，无skip，源码前后稳定。
- 稳定源码摘要：5cb3235519b30f96eedeb98eabdc9260ee912bce026d403b1a0ca0f657363167。
- 稳定日志SHA256：815229679444d547b71590d74b471d70a610da607395a4ddad6eab7395f01b3b。
- git diff --check通过；HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

仅预算/搜索限定回归，非生产全项目认证。监督参数与假设账本等其他接口身份隔离、哈希codec迁移仍待完成。
