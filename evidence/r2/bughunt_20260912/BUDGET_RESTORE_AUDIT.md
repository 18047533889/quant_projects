# 预算恢复衔接复核（2026-09-13 16:56 heartbeat）

server-c正式main原树，剩776 GiB，FE/DA无活动；未改业务代码、提交、推送或生产数据。
复查SearchBudget/BudgetTracker预留、结算和恢复边界，并运行预算、恢复、durable campaign及配置数值回归。
budget_restore_audit_20260913：81 passed，无skip，源码before=after。
源码digest：a2199840c0cc4828593bc150aaa91de43221e855ba2e4846e25a058dc69db4ea。
日志SHA256：ddf35e8bd25f131df09578801f48decb550ec59f1a2bb4e0d65a23aafcc50c6c。
本轮限定回归未发现新失败，不做无关改动。底层直接预算API的bool/非整数输入可作为后续反例方向，尚未完成复现，不视为已修。
公共hash迁移、并发隔离等待办保持开放；本记录不代表生产或全平台验收。
