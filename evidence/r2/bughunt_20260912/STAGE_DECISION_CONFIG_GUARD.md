# 阶段决策配置漂移（2026-09-13 09:50 heartbeat）

server-c正式main，剩776 GiB，FE/DA任务idle；保留已有修改，不触FE/DA。
未commit/push/建分支/复制或改写生产记录。

阶段request factory或provider修改配置后，原代码仍调用scheduler.advance，可能晋级并执行下一昂贵阶段。
现两次回调返回后均检查执行计划，再进行原receipt绑定验证与推进；不降级决策门禁。
反例通过现有完整廉价阶段fixture，确认advance零调用、只评估fidelity=0，请求漂移时provider也不调用。

stage_decision_drift_before：2 failed / 9 passed，保留前置证据。
stage_decision_drift_search_final：搜索目录747 passed / 12 warnings，无skip，源码before=after。
源码digest：a095105444f9e0c3eb3c13f49bbaabb4c2e8e212d9e7d5e9f1ead09f7651a25a。
日志SHA256：e534b40e1101449afb1ed9b33e38b89ce81964b581a46675f19fa735c87f7a4b。
git diff --check（factor_optimizer与本轮证据）通过。

运行期配置隔离仍非线程安全快照；其他可执行对象/plateau detector、并发瞬时修改及回调内部副作用仍需审查。公共哈希v2迁移等待办不变。
