# 搜索开关类型（2026-09-13 13:54 heartbeat）

server-c正式main原树，剩776 GiB，FE/DA无活动。保留旧修改，未commit/push/建分支/复制/改写生产记录。
enable_multifidelity和require_evaluation_protocol缺少bool校验，字符串false被当真，空容器/None被当假，配置意图静默改变。
现在与已有screening_only统一明确bool校验；保留显式True/False往返，不改变生产模式必须使用协议的独立门禁。

新增16项：search_switches_before为14 failed / 18 passed，保留日志。
search_switches_final：搜索目录781 passed / 12 warnings，无skip，源码before=after。
源码digest：400cf9d88bf96cb4954bc7e3e22c9f9001473b239b467ec3547c3bb8da552746。
日志SHA256：799a8c09566547f639e7def53b01f23657922fa72d803b11aeb21c12ca58d50c。
git diff --check（factor_optimizer与本轮证据）通过。
历史错误类型将拒绝而非自动转换。配置并发隔离和公共哈希迁移等仍开放，未宣称生产通过。
