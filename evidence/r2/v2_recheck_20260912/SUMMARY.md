# V2 整改交接（2026-09-12）

修改已直接落在 server-c 的 /home/sunhaiwei/quant_projects，仍为 main；没有创建分支或 worktree，没有 commit、push、部署、生产发布或删除正式数据。子代理均使用 GPT-5.6-sol。保留了其他任务的 FactorEngine 改动。

## 本轮修复

1. 平台：坏输入隔离、不同评估上下文去重、多上下文持久化主键碰撞、重启语义冲突恢复、持久化 job 临时失败重试、未成熟标签等待、重复方法覆盖。
2. FA：健康政策哈希绑定完整评分与准入内容；外部可变列表深冻结，不能篡改已创建政策。
3. FP：封闭未验证原生执行旁路，研究 reference 要显式 opt-in；混合证券键类型在透视前明确拒绝。
4. QE：可执行成本轨迹使用独立 ExecutablePortfolioArtifact，研究 probe 保持不同身份；修复与 FA 新生产合同对接的集成 fixture，并补错 recipe 拒绝反例。
5. 跨库 shadow replay 明确研究资格，不松生产门。FO 已有正确实现保留，research_only 能力未伪造升级。

## 实际测试

| 范围 | 结果 | 证据限制 |
|---|---|---|
| FP 全量 | 408 passed，1 xfailed | fp_stable_retry.json：稳定 PASS |
| FA + FO 全量 | 2393 passed | fa_fo_final.json：运行期间其他 FE 源码变化 |
| QE 主测试 | 1250 passed，2 skipped | qe_stable.json：运行期间平台源码变化 |
| QE adapter 集成 | 2 passed | qe_adapters_final.json：稳定 PASS |
| 平台全量（排除 live PG） | 479 passed，1 skipped | platform_final.json：运行期间 FE 源码变化，且在最后 maturity 补丁之前 |
| 平台最终差额及相邻回归 | 57 passed | platform_pending_final.json：稳定 PASS，包含最后 maturity 补丁 |
| 唯一实现/包碰撞检查 | 1 passed | canonical_sources.json：稳定 PASS |
| QE 有界真实 CUDA 专项 | 52 passed，0 skipped | qe.md；非全部指标大规模认证 |

上述范围有重叠，不能简单相加为唯一用例总数。未运行/skip/xfail 不记为执行通过。所有初始失败和并发变更收据均保留，没有改日志冒充全绿。

新进程运行身份检查显示导入来自正式 server-c 树，import_errors 为空；已有常驻 worker 的内存代码未验明。
生成 capability_matrix.csv 共 163 行；仅本轮实跑的 20×20 合成 rank_ic 公共 CPU 与 strict-CUDA 标 DONE，其他未跑层/指标仍为 NOT_RUN。这不是全指标 CPU/GPU 验收。

## 未完整闭合

[LEDGER.md](LEDGER.md) 与 [ledger.json](ledger.json) 完整覆盖原任务书 97 个编号，逐项列出源码、测试及剩余限制；没有把局部回归等同 97 项生产验收。

还需要：稳定后的 FE→FP→QE→FO→FA→平台→模型联合验收；经批准的真实数据快照/样本与业务 profile；可信 evidence/context resolver；真实规模和全指标硬件基准；历史政策哈希对应的影响清单与重评迁移；实际存储上的授权 shadow/GC/回滚验证。没有权限或证据的生产资格继续 fail closed。

另一个 FE 任务确认目前不便冻结源码，其后续用途/落值/目录修复仍在进行；本次交接不等待或覆盖它的工作，也不声称最终全栈已冻结验收。
