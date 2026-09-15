# 最终决策配置漂移（2026-09-13 08:49 heartbeat）

server-c main正式原树，776 GiB可用，FE/DA任务idle，未触其文件；保留既有修改。
未commit/push/建分支/复制或改写生产记录。

_finish入口、request factory或provider回调改变SearchConfig时，原代码仍写决策并结束会话。
现分别在入口、请求生成后、provider返回后校验原执行计划；漂移时不验证/采纳返回结果、不写决策字段、不结束会话。
保留原require_bound_decision_receipt真实性绑定，不降低门禁；回调内部已发生的副作用无法自动撤销。

final_decision_drift_before：3 failed / 6 passed，覆盖入口/请求/provider三种漂移。
final_decision_drift_search_final：搜索目录745 passed / 12 warnings，无skip，源码before=after。
源码digest：241e13b35fbc3fdcb1db21947b6f33a0aec545fde639723aa14c6663582ddb2b。
日志SHA256：d1eeac1bd3bf54d71835d7fc7a971e9d3f9473a8f1a2fa725fd2fb86ea12d02d。
git diff --check（factor_optimizer及本轮证据）通过。

运行期配置隔离仍PARTIAL：阶段决策、其他回调、并发瞬时修改尚需检查，未宣称生产或线程安全认证。
