# 预算操作布尔成本（2026-09-13 18:56 heartbeat）

server-c main正式原树，剩775 GiB，FE/DA无活动，保留已有修改。
未commit/push/建分支/复制或改写生产记录。
BudgetTracker直接API仍将bool成本当0/1，可预留、结算及释放；现写接口在任何账目修改前拒绝bool，can_spend返回False。
合法数值与原有限/非负限制保持不变，不更改费用公式。
新增10项：budget_boolean_ops_before全部失败；反例覆盖四种写入参数及只读检查，并验证拒绝后to_dict完全不变。
budget_boolean_ops_final：预算与搜索843 passed / 12 warnings，无skip，源码before=after。
源码digest：a79d85fd5396ccf154c6940fb12798effec82e09f1854af864e28101c4c4d8b8。
日志SHA256：41d8bd517ec63cdf08ee1bc0949277c946296a55710bd8321689cd944ba81bb0。
git diff --check（factor_optimizer与本轮证据）通过。
本轮只覆盖内存BudgetTracker入口，不宣称持久化后端/第三方数值类型已全部覆盖；其他未闭合问题不变。
