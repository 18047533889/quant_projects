# FactorEngine / DataAccess session — paste into Claude

```text
FE/DA 持续 multi-subagent loop，直到我说停或 loop/HALT 存在。

读 loop/care.md、loop/orchestration.md、loop/status.md、loop/queue.md。你是 Coordinator，禁止 bulk 改代码。

硬规则：任何一个 subagent 返回 ≠ 结束。立刻 spawn 下一个角色。Reviewer 结束后立刻下一轮 Finder。不要总结完停住。用户沉默也要继续改。

每轮：Finder 找 FE/DA 缺陷发票 → Dispatcher 发牌给 ≤2 writer-fe（文件不重叠）→ Tester → Reviewer(diff) → loop/status.md 一行 → 马上下一轮。

Agent：.claude/agents/ finder dispatcher writer-fe tester reviewer coordinator。

硬约束：working tree 事实源；禁 git checkout/restore/stash/clean/reset；≤15GiB；BLAS/OMP/MKL/Polars=1；未跑=NOT_RUN。不要 plan。不要加载 archives。停：我说停，或 touch loop/HALT。
```
