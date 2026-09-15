# 布尔评估成本（2026-09-13 15:54 heartbeat）

server-c正式main原树，剩774 GiB，FE/DA无活动；保留原有修改。
未commit/push/建分支/复制/改写生产记录。
evaluation_cost_units原接受bool并按0/1处理，错误类型可静默改变预留成本；现明确拒绝bool，原数值有限/非负校验不变。
保留None自动预算、0和0.0零成本、1.5数值往返兼容，不改变实际费用结算。

新增6项：boolean_cost_before为2 failed / 43 passed，保留前置日志。
boolean_cost_final：搜索目录794 passed / 12 warnings，无skip，源码before=after。
源码digest：4162410917eae93791973fb47359b3fbde87c98855b5cafcbcb95a9bfb922bea。
日志SHA256：6e500f558308b69a25cccb659faacfebf38a1cc139a83d26e840912ed68f7122。
git diff --check（factor_optimizer与本轮证据）通过。
未进行生产认证；哈希迁移、并发隔离等其他待办保持开放。
