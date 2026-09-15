# 搜索运行期配置漂移首批保护（2026-09-13 05:48 heartbeat）

server-c main正式原树，磁盘776 GiB；FE/DA任务idle，未触其文件。
保留已有checkpoint/resume修复，不commit/push/复制/建分支或改写生产记录。

复现：SearchRunner与SearchSession共享可变SearchConfig，在运行前或proposal回调内修改plateau_threshold后仍继续运行。
新增执行计划校验，比较runner和session两份配置与session已绑定hash；每轮开始、proposal成功返回后检查，漂移明确ValueError终止。
不会刷新hash来认可变更，也不把漂移伪装成正常候选失败。

live_config_before：2 failed，保留前置日志。
live_config_search_final：搜索目录738 passed / 12 warnings，无skip，源码before=after。
源码digest：6ae4c2a09802ef72726949d259adf309bdf3a28b10d555f1168f6f929887c759。
日志SHA256：8be80a10dfe67a608096b653dc31977c1d708f07394cce72549d77bb83ac2b6d。
git diff --check（factor_optimizer和本轮证据）通过。

限制：本轮覆盖迭代/提案边界，不是线程安全快照。评估、验证、最终决策回调内漂移及变更后恢复原值的场景仍需审查，活跃配置隔离总体保持PARTIAL。未跑全平台或生产认证。
