# 默认调用与失败证据

本例以已经批准并配置的 `FACTOR_ENGINE_V2_PROFILE` 为前提。配置决定数据权限、证券范围、日期和输出目录；不在示例中猜测这些业务信息。

```python
from factor_engine import get_engine

# factors 为已经构造好的 Factor 对象或其迭代器，可一次提交。
def materialize_all(factors):
    with get_engine() as engine:
        receipt = engine.run_many(factors)
        return receipt, engine.policy.work_item_max_attempts

# 独立脚本必须有 main guard；不要在模块导入时启动 spawn 工作进程。
if __name__ == "__main__":
    receipt, max_attempts = materialize_all(factors)
```

不需要逐次指定后端、并行数或内存比例。默认策略为区域后端 `auto`、有效剩余内存的 80% 统一池。策略与接口回归通过不等于真实十万因子/GPU吞吐认证，也不承诺所有算子都有原生后端。

## 查看已结束任务的模型失败证据

```python
from factor_engine.runtime.persistent_run_state import PersistentRunState

reference = receipt["fit_failure_evidence"]
if reference["availability"] == "indexed":
    state = PersistentRunState(receipt["state_path"], max_attempts=max_attempts)
    try:
        after_seq = 0
        while True:
            page = state.fit_failure_evidence_page(after_seq=after_seq, limit=50)
            if not page:
                break
            for record in page:
                if record["availability"] == "UNAVAILABLE":
                    print(record["evidence_id"], "未取得证据，不能当作零失败")
                else:
                    snapshot = record["snapshot"]
                    print(record["evidence_id"], snapshot["groups"])
                    # details 含有界的原因/归属/窗口样本；注意 truncated。
            after_seq = page[-1]["seq"]
    finally:
        state.close()
else:
    print("旧任务未采集该证据，不能当作零失败")
```

请只打开该任务实际返回的状态路径，不用此构造器探测任意路径：它不是纯只读文件查看器。上例用于已经结束的任务，不绕过活动任务的协调与恢复流程。

每页同时限制条数及约 1 MiB 的载荷计费，单条快照至多 256 KiB。这是线格式边界，不是 Python 解码对象的精确 RSS 上限。

每条记录的 `scope` 为 `wave_sample`，`coverage` 为 `instrumented_fit_producers_only`：只覆盖已接入采集器的模型核，且明细会截断；不是所有算子的每次拟合全量账本。重试批次也不能直接相加当作不同因子数量。`UNAVAILABLE`、`legacy_unavailable`、截断明细和空快照都不能冒充全算子零失败证明。

## 恢复与完整性边界

默认入口接受 `engine.run_many((), resume_run_id=run_id)`，从原清单恢复，不重新提交因子定义。已完成结果先复核，不重复落值。待执行部分仅开放从未派发的 `ACCEPTED` 项：attempts=0、无 artifact generation、commit_state=NOT_STARTED，且没有已派发证据关联。

此子集必须同时通过原任务身份、完整 SQLite 工作进程归属与持久退出证明、v1 证据索引、协调锁和原始任务截止时间核验。截止时间不会在恢复时刷新。主 admission、预取和补位均使用同一资格过滤，混合已完成/未派发因子不会重算已完成项。仍为 RUNNING、有 INTENT/UNKNOWN、需要重试或对账、缺少身份/退出证明的任务继续明确拒绝；这不是任意中断的自动恢复认证。示例不启动生产重算。

单个当前执行槽在已证明退出、身份与 generation 上下文精确一致，且没有其他活动/预取/后续槽时，丢回执或超时仍可走既有有界对账路径。跨槽、过期响应和协议身份不明不会借此通道恢复。

新任务在最终回执出现前硬崩溃，可依据持久化标记和证据表完成证据身份校验；这不解除上面的待执行任务安全门禁。已有记录会校验长度、摘要、因子序号关联和结构。回执存在时，还核对批次数和序号，拒绝丢失记录。

如果尚无回执，且证据主表与关联表的全部记录被协调删除，就无法区分它与“尚未发出任何批次”。因此不能声称该窗口具有外部见证的完整记录计数，更不能声称任意篡改都可检测。
