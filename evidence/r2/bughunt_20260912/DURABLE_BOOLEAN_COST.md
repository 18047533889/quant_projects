# 持久化预算布尔成本（2026-09-13 19:58 heartbeat）

server-c main原树，剩775 GiB，FE/DA无活动。保留现有改动，未commit/push/分支/复制或写生产数据库。
SQLiteCampaignStore.reserve/settle仍接受bool写作0/1；现事务开始前拒绝estimated_cost/actual_cost布尔值，保持数值与会话状态规则不变。
4项反例使用pytest小临时SQLite并重新打开验证budget_state不变；无正式数据参与。
durable_boolean_before：4 failed，保留证据。
durable_boolean_final：预算与搜索847 passed / 12 warnings，无skip，源码before=after。
源码digest：6d333986f8ab45de44f6d0078d942898d87f09f28655a9bbe4561a06e1fee7d0。
日志SHA256：46f0f115795876d3f106146d66153f9fd914cce9a2bd1aaab9968caf7985245c。
git diff --check（factor_optimizer与本轮证据）通过。
仍需审查DurableBudgetTracker.commit/release忽略reserved_cost的兼容与校验差异，未宣称所有持久化预算边界闭合。其他开放项不变。
