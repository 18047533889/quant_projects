# R5 默认 auto 批量落值：实现与验证（2026-09-13）

## 工作范围与结论

直接修改 server-c 正式工作树 /home/sunhaiwei/quant_projects，保留原有未提交改动。
分支仍为 main，HEAD 为 e94ac507d670fd1c16b1d6a63fc5d6286daa5970。
没有创建分支、worktree、整库/数据副本，没有 commit、push、部署或发布生产因子。
本文仅列本轮可归因变更；不能把整个现有 dirty diff 归为本轮修改。

默认持久化入口已实际接通 DAG/CSE、native fusion、adaptive scheduler、
auto operator backend、自动 worker，以及原有有效剩余内存 80% 统一资源政策。
这不是全硬件绝对最快、永不 OOM、10 万真实因子全量完成或全算子 GPU 认证。

## 默认调用

```python
from factor_engine import get_engine

# 先使用已有、获批准的 FACTOR_ENGINE_V2_PROFILE 业务配置。
with get_engine() as engine:
    receipt = engine.run_many(all_factors)
```

调用方无需选择性能参数。日期、universe、快照和授权存储目标仍须由业务配置确定。
receipt 提供逐因子结果索引和终态，不把全部面板返回到内存。
旧的低层 FactorEngine API 保持兼容，不等价于此默认持久化入口。
使用说明：factor_engine/docs/DEFAULT_AUTO_RUN_MANY_R5.md。

## 真实代码变更

1. runtime/default_engine.py 显式传递优化参数，防止旧环境开关关闭默认 CSE/融合。
2. runtime/auto_dag_admission.py 与 bounded_pipeline.py 流式估计全任务图元数据；
   预算准入后实际扩展计算批次为整任务。图元数据租约在所有子进程关闭后释放，
   隔离未退出进程时不虚假释放。回执记录准入范围和原因。
3. 整图及单槽默认任务使用完整 CPU/IO 权限；未分区代理仍继承正确的原生线程上限。
   资源不足保持原有有界分批、读算写预算与有限终态。
4. cache/session.py、buffer_store.py、resource_broker_ipc.py 将 CSE 容量接到
   实际父 broker 动态预算。零余量不增加假预算，压力收缩不提前释放活跃值。
5. batch_service.py 修复 shared-of-shared 引用计数和回收：
   根可达闭包、悬空/孤儿/cycle 检查、迭代深链，以及共享任务完成后释放上游引用。
6. engine.py、multibackend/batch_global_optimizer.py、physical_execution_index.py
   修复多根物理执行、缺省 node_id 绑定、真实跨后端转换、根可达区域执行。
   同一物理计划复用不可变元数据索引，避免每根全图扫描；多线程计划替换身份检查通过。
7. 共享物化与根使用同一全局物理后端选择；共享节点执行、governed put、
   成本估计、上游引用释放使用 optimizer bound graph。
   CSE_SHARED 任务纳入所选物理 workspace/transfer 预算，不绕过已有缓存治理。
8. execution_ledger.py 从真实 batch output 抽取有界账本，直接落值出口原子写入
   execution-<evidence_id>.json。单份上限 64 KiB（按实际缩进 JSON 编码），
   不包含面板、因子定义或全图。诊断失败为 best-effort，不触发已落值因子的重试。
9. run_dag_catalog.py 支持独立流式逻辑元数据登记，但未接默认执行：
   当前执行器尚不消费其预编译结果，挂载只会增加登记成本。

## 最终代码上的联合回归

每组完整命令、耗时、返回码在同名 *-watchdog.json，测试明细在 *.log。
执行 watchdog 使用 0.1 秒采样的 2 GiB 测试进程族 RSS 阈值，非内核硬内存限制。
下列组存在交叉覆盖，不能相加解释成不同测试数量。

| 证据前缀（evidence/r5/） | 环境 | 结果 | 采样峰值 bytes |
| --- | --- | --- | --- |
| integrated-final-r5 | 项目 pandas 2.3.3 | 132 passed | 748134400 |
| default-final-r5 | 项目 pandas 2.3.3 | 34 passed | 1054457856 |
| cse-budget-final-r5 | 项目 pandas 2.3.3 | 23 passed | 783831040 |
| integrated-final-pandas3-r5 | 系统 pandas 3.0.5 | 149 passed | 736514048 |
| pipeline-final-r5 | 项目 pandas 2.3.3 | 52 passed | 1796313088 |

- 真实 CSE 组包括 5 根与 513 根；默认测试使用 native_fusion=True、
  max_workers=None，不传 n_jobs。每例验证 ts_min 只执行 1 次、源批读 1 次、
  共享值读取次数等于根数、全部数值等于独立非 CSE 参考、CSE 缓存与租约释放。
  这是 research adaptive 真实算子执行，不是 canonical production facade E2E。
- 物理共享测试使用真实全局 optimizer、真实 Pandas neg 结果，令 Hybrid.execute
  抛错以证明共享值没有重新走 Hybrid 路由；另测预计算值保持 governed put。
- 持久化流水线测试包含实际 spawn、账本文件、原生线程限制和隔离租约生命周期，
  使用测试引擎，不冒充生产数据链路验收。
- 10 万轻量目录测试：run-dag-catalog-final.log，5 passed / 107.10 秒，
  采样峰值 349401088 bytes。它不是 10 万真实因子的吞吐量测试。
- git diff --check（FactorEngine 及本轮顶层物理测试）通过。
- 旧的 pipeline-regression.log 曾有 2 个 FakeBroker 合约失败；fixture 已修复，
  fixture-two.log 两项通过。全套最终重跑另见 pipeline-final-r5.log/json。

## 未实现或未验证的边界

- 10 万因子可一次提交，不代表必然进入单张执行 DAG。图元数据估计为启发式，
  最多使用当时 execution budget 的四分之一；直接落值回执还受 256 MiB 队列预算。
  保守估计通常使 10 万根自动回退有界分批。
- 跨分批没有持久化受治理共享值缓存，跨批次仍可能重算。没有承诺
  “10 万因子所有公共表达式全程一次计算”。解决此项仍需流式执行/跨批值生命周期接线。
- auto 只在当前可用、符合语义与能力要求的路径中作成本选择，未做全部硬件实测最优证明。
  未重新开启 R4 已禁用的不稳定快速统计核；没有 GPU 全量性能或数值验收。
- production 默认端到端缺少本轮批准的真实业务源/输出范围。严格 canonical source
  禁止本地 read_root 覆盖，现有诊断缺 clean-cos-ro；非 canonical source 缺正式 FieldSpec。
  保留安全门，不猜 COS 凭据或修改源权限。default-source-safety-gate 测试绿色；
  default-physical-cse.log 是该受阻诊断证据，不是成功验收。
- 80% 是有效剩余内存的治理目标，不是物理内存占用保证。实时压力可能导致等待、
  有限资源拒绝或失败；Python/原生分配不能由启发式估计保证永不 OOM。
- 共享任务专用 thread_budget 与已有原生线程控制仍需更广的并发实测，
  不因本轮子进程启动限额修正就宣称所有嵌套线程超订阅问题已解决。

## 磁盘与协作

最终检查约 776 GiB 可用；本轮 evidence/r5 约 260 KiB（报告写入前）。
本轮已用完的小补丁在 Mac /private/tmp 和 server-c /tmp 按精确文件名清理；
保留正式代码与回归证据，不删除正式数据。子代理为 GPT-5.6-sol，按文件直接协作。
