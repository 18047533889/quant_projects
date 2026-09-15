# 验证回调配置漂移（2026-09-13 06:49 heartbeat）

server-c正式main原树，剩776 GiB；FE/DA任务idle，未触其文件。
保留前轮改动，未commit/push/建分支/复制或改写生产记录。

验证回调修改SearchConfig后，原代码会先采纳验证结果，合法分支甚至预留/启动评估，下一轮才发现漂移。
在_validate_trial返回后立即校验已绑定执行计划，覆盖通过、拒绝、内部捕获异常；未采纳结果且未进入评估就抛出明确错误。
不回滚已发生的提案/验证动作，不刷新配置hash，试验留在VALIDATING供诊断。

validator_drift_before：3 failed / 2 passed；前置证据保留。
validator_drift_search_final：搜索目录741 passed / 12 warnings，无skip，源码before=after。
源码digest：60120161c0edacdf5972792681e9c06ac425e486a3da24768f4d6825b6842bed。
日志SHA256：d844d66752d73b468e2bfaee036921b7925698199efd0537d6647e0f3085829e。
git diff --check（factor_optimizer与本轮证据）通过。

运行期配置隔离仍PARTIAL：评估、最终决策及其他回调边界、并发瞬时修改尚未全面覆盖。不是生产或线程安全认证。
