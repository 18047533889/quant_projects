# 底层预算上限类型（2026-09-13 17:56 heartbeat）

server-c正式main原树，剩776 GiB，FE/DA无活动。保留原改动，未commit/push/建分支/复制/改写生产记录。
SearchBudget次数限制仅做数值比较，可接受浮点、NaN/Inf和bool；max_cost_units也接受True当1。
现max_trials/max_evaluations/max_llm_calls要求int排除bool，max_llm_calls继续允许None；成本拒绝bool，原有限正数检查保留。
合法max_llm_calls=0/None及数值成本往返不变，错误旧类型拒绝不自动改写。

新增23项：budget_limit_types_before为20 failed / 3 passed，前置证据保留。
budget_limit_types_final：预算与搜索范围833 passed / 12 warnings，无skip，源码before=after。
源码digest：98d63b716dae566540840d02ba972e2e8b93eaf5394c68545a110199f20fe7f5。
日志SHA256：4ffb3e35ab0519e6554833d4ad0488d860c880adcb6cc51c9a73855b1aaec8e8。
git diff --check（factor_optimizer与本轮证据）通过。
本轮仅预算上限构造/恢复入口；BudgetTracker直接成本API等仍需审查。公共hash迁移与并发隔离不因此关单。
