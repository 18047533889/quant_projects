# 停滞检测配置漂移（2026-09-13 10:52 heartbeat）

server-c正式main原树，剩776 GiB，FE/DA任务idle。保留旧修改，未触FE/DA。
未commit/push/建分支/复制或改写生产记录。

自定义plateau_detector改配置并返回False时，之前会再调用一次proposal才发现漂移。
现在先取得停滞结果，再校验原执行计划，最后才决定结束或继续；漂移时无额外proposal。
测试覆盖False/True两个分支。plateau_drift_before为1 failed / 12 passed；前置日志保留。
plateau_drift_search_final：搜索目录749 passed / 12 warnings，无skip，源码before=after。
源码digest：9d78fe2942d4a022f06b2236394eceec94c4097a097daee3b4a6640ed775fd02。
日志SHA256：35294b35ad483487c5c5d6454033fca8fdd5e79690e33d3e0f28ee3cc5c2318f。
git diff --check（factor_optimizer与本轮证据）通过。

边界：回调内部副作用、可执行对象替换、并发瞬时修改不属于线程安全保护；全局哈希迁移等仍开放。
