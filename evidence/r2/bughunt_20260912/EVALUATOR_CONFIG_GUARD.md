# 评估输出采纳前配置校验（2026-09-13 07:49 heartbeat）

server-c正式main，磁盘776 GiB，FE/DA任务idle；保留旧改动，未触FE/DA。
未commit/push/复制/建分支或改写生产数据。

评估回调修改配置后，原代码仍调用normalize_evaluation_result并读取可能已变化的objective配置。
现在回调返回后立即检查执行计划，漂移输出不进入解析；失败路径结清已执行预留、标记FAILED并记录，再传播配置错误。
不会退款已执行评估，不刷新计划hash或继续下一试验。

evaluator_drift_before：1 failed / 5 passed，反例确认发生不应有的normalization。
evaluator_drift_search_final：搜索目录742 passed / 12 warnings，无skip，源码before=after。
新反例还确认evaluations_used=1且预留已释放，试验FAILED。
源码digest：22d36c52fe1b5627809a43f513dfca8f857c1cf7df101effaca144fee45634fb。
日志SHA256：6a1e382a01b85e8a2730b324f935bef6c68cde8402591d76e1739650e84c28c4。
git diff --check（factor_optimizer与本轮证据）通过。

运行期隔离仍PARTIAL：最终决策/阶段决策等其他回调、并发修改后恢复原值及回调内部副作用不受本检查完整保护。
