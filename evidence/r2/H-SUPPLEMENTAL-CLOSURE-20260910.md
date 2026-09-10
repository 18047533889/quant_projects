# H01–H36：主干补充整改与有限验证记录

任务书：FactorEngine_DataAccess_主干合并补充审计_新增任务.md。
正式工作树：server-c `/home/sunhaiwei/quant_projects`，分支 `main`。
开始基线与本轮结束前核对的 HEAD：`c5e880db91f672d101676f92f7901361ea48522d`。
本轮直接修改正式工作树，未创建分支/worktree/源码副本，未 commit、push、部署、发布生产因子或安装依赖。子代理均为 GPT-5.6-sol。

本记录区分“已实施代码和指定回归”与“完整生产认证”。它不是 1756 算子、全部参数和全部后端的零缺陷证书。早期失败日志保留；早期进度文件不作为最终状态。

## 清单对应的代码整改

| 任务 | 已实施内容 | 可核对证据 |
|---|---|---|
| H01–H03 | 解码字节码名称操作数；保留默认参数、命名闭包、真实实例 `_fn` 和 helper 身份；Pandas/整数/轴/缺失身份不做有损强转 | `evidence/h01_h05_identity_supplement_20260910.json` |
| H04–H05 | 可逆类型保持的冻结/解冻；执行变体绑定实际代码，Polars delegate 不冒充 native | 同上 |
| H06–H08 | Polars rank 保留 null 位置；组件调用统一完整绑定；拒绝 NaN/Inf 权重 | `evidence/h06_h08_rank_component_supplement_20260910.json` |
| H09 | SQL ddof=0/1/2 与有效样本门槛 | `evidence/h_sql_audit_20260910.json` |
| H10 | 更新真实使用 API 对应的 Pandas/Polars 最低版本声明，现有团队锁定版本不降级 | `evidence/h10_h25_h26_supplement_20260910.json`；未运行最低版本环境 |
| H11 | 有限正数校准、NumPy telemetry、128 keys/256 samples、median/MAD/p95、4 MiB 原子持久化、完整形状键 | `data_access/tests/unit/test_h11_calibration_bounds.py` |
| H12–H13 | 删除错误性质与重复声明；验证实际声明；前缀/分块检查全部列、轴、dtype、attrs 和复数虚部 | `tests/operators/test_h_validation_lazy_regressions.py` |
| H14–H17 | SQL 模板绑定完整上下文；批/单计划准入一致；财务会计期间及 revision 策略；UTF-8 字节上限 | `evidence/h_sql_audit_20260910.json` |
| H18–H21 | 当前预测 finite 与失败原因；K 次成熟拟合且截止点复用；recurrence 真计数；matrix-profile 静态可行域 | `evidence/r2/H18-H33-final-evidence.md` |
| H22 | 真 registry winner 与 stateful runtime 执行；参考演示单独标识；ADX 各阶段成熟与参数化 2*window，旧 checkpoint 拒绝 | `evidence/r2/H22-real-gates-final.log`、`evidence/h22_adx_contract_review_20260910.json` |
| H23 | 独立输入与参考快照；检测输入修改、第二列污染；空集、无有效覆盖及不支持元数据拒绝 | `tests/operators/test_h_validation_lazy_regressions.py` |
| H24 | 批 SQL 共享非叶子 DAG 编译，真实 DuckDB 执行与 EXPLAIN | `evidence/h_sql_audit_20260910.json` |
| H25–H26 | 逐项裁决并删除 4 个 tracked `.orig/.rej`；路径门禁精确区分历史声明与运行期错误引用 | `evidence/h10_h25_h26_supplement_20260910.json`、`evidence/r2/H-path-gate-final.log` |
| H27–H30 | 日内完整内容身份；finite/empty；线性大小网格、按需统计和物理 owner 租约；物理相邻收益与共同支持 | `evidence/r2/H18-H33-final-evidence.md` |
| H31–H33 | HAR 可行下限 55/58；R² 成对分母；市场收益 finite/ex-self/空日期/轴一致 | 同上 |
| H34–H35 | SQL 时间排名使用历史窗口；七个候选遵守 min_periods，仍不授予生产安全资格 | `evidence/h_sql_audit_20260910.json` |
| H36 | 物理列缓存、返回完整性、alias/context；命中也执行缩减；现有 broker 准入及物理 owner 延迟释放；Source 单层缓存避免重复计费 | `evidence/h36_resource_followup_20260910.json` |

## 扩展复核修复与保留边界

- 修复实际 SQL 滚动峰度门槛；补 ALMA/CoppockCurve 的真实 DuckDB 实现及独立 NumPy 对照。
- 77 个 SQL 最小计划测试缺真实输入/参数，已按契约补齐；15 个确实缺少精确 SQL 实现的条目已撤销错误 implemented/parity 广告，并记录 `SQL_EXPLICIT_UNSUPPORTED_REASONS`。这 15 个仍没有 SQL 实现，不能说全后端已补齐；现有非 SQL 路径保留。生产安全集合未提升。
- 修复 AR 窗口把 W 个源行误作 W 对样本、Huber 样本不足原因分类；Huber 精确共识边界另有严格多数、秩、原坐标误差和零尺度最优性条件验证，具体测试见模型记录。
- GARCH 和非对称 pair/barrier 旧测试必须按当前真实输入及 strict-prior 契约运行，不通过改生产默认参数让错误调用成功。
- 旧 `.v8_finite_stage`、`.v8_ingest_ownership`、`.v8_inspection_stage` 合计约 420 KiB，逐文件确认已吸收/被正式实现替代后删除；哈希及裁决见 `evidence/v8_staging_residue_review_20260910.json`。它们原本未受 Git 跟踪，不能用 Git 恢复；4 个 tracked 合并残留可从本轮基线恢复。
- approved content digest 仍走现有严格 eager 核验。H36 不代表 D02 受管理原生 lazy 身份闭环已完成。

## 根代理独立测试

- `.venv/bin/python -m pytest -q data_access/tests/unit --disable-warnings --maxfail=3`：1716 passed，11 warnings；`evidence/r2/H-dataaccess-unit-final.log`。
- `.venv/bin/python -m pytest -q tests/backend_sql`：741 passed，23 skipped，2 warnings；`evidence/r2/H-sql-full-final.log`。跳过项为既有外部数据要求、旧 alias 矩阵和空 production-safe 参数集，不计作通过。
- 身份、H06–H08、ADX、路径负控、H22/H36、续算、R40 数学、Source lazy/并发/snapshot cache 联跑：112 passed，3 warnings；`evidence/r2/H-identity-math-source-union-final.log`。
- 真实 H22 gate：13 个声明前缀夹具与 11 个声明 stateful 夹具有限通过；包含 NaN/Inf 和非默认 ADX 单独测试。不外推成所有因果算子或全参数域认证。EWM cov/corr 的 Polars 入口诚实记录为 Pandas delegate。
- 模型/日内最终独立组合的完整命令及结果补充在本记录末尾；模型责任证据见 `evidence/r2/H18-H33-final-evidence.md`。
- 路径门禁与 staged/unstaged `git diff --check` 通过。警告包含测试用未知 Polars 实现分类，未放松 production gate。

各组可能覆盖相同模块/反例，不把测试数相加当作唯一算子数。

## 性能、转换与环境账本

- 合成 H19 截止点复用为 30 次拟合而非重复 170 次；不是全任务吞吐倍率。
- H29 的 4/8/16 天 × 3 bar × 2 列网格 buffer 为 192/384/768 bytes；无按交易日数平方复制。无缓存准入时不再复制整份统计包，准入后按最后物理 owner 生命周期计费，分配失败释放租约。
- H24：8 个根节点的共享 mean 子图编译 1 次，SQL UTF-8 长度小于 8 KiB，实际 DuckDB EXPLAIN 已执行。不据此承诺生产数据上的固定提速。
- H36：alias 只在返回边界 rename(copy=False)，不复制值 buffer；Source 路径禁用 bundle 二级驻留，Source 是缓存单一计费方。未知/零预算不兜底申请另一份内存池。
- 环境：Python 3.12.3 / NumPy 2.2.6 / Pandas 2.3.3 / Polars 1.42.1 / DuckDB 1.5.4 / PyArrow 25.0.0。没有安装并运行最低依赖版本组合。
- 未运行真实 A 股/COS 生产落值、十万因子或 GPU 压力认证；没有修改既有 auto/80% 默认策略，也没有要求调用者新增性能参数。
- 磁盘复核约 787 GB 可用；本轮未复制仓库、数据、虚拟环境或 `.git`。自己创建的小补丁文件已清理，必要测试/哈希证据保留。

## 默认调用与统一预算补充回归

独立默认双因子回归暴露了真实资源问题：controller 使用旧 reserve 公式得到零目标，而 broker 的统一 execution pool 仍有正额度。现 controller envelope 使用既有 execution budget；未知/真零预算保持零。进一步将计算 peak 承诺与物理内存租约合并原子核算，保留写出额度。Job child、writer/IPC、缓存物理生命周期及串行入口的后续实施和验证见下节。

- `evidence/r2/H-model-intraday-union-final.log`：269 passed，2 warnings，39.42 秒。命令为 `.venv/bin/python -m pytest -q tests/operators/test_model_semantics_fixes.py tests/operators/test_model_semantics_fixes_round2.py tests/operators/test_model_final_certification_r2.py tests/operators/test_intraday_next_stage.py tests/operators/test_advanced_intraday.py tests/operators/test_regression_models.py factor_engine/tests/intraday/test_h27_h33_intraday_regressions.py factor_engine/tests/operators/test_intraday_sufficient_stats.py factor_engine/tests/ts_model/test_h18_h31_model_regressions.py factor_engine/tests/operators/test_v9_m01_shared_huber.py factor_engine/tests/operators/test_v9_m01_cs_huber_shared.py --maxfail=5`。包含不伪造预算的 default-broker 双因子调用，以及显式有界数学夹具；旧失败记录保留在 `H-model-intraday-union-before-resource-fix.log`。
- 资源修改后重跑 `.venv/bin/python -m pytest -q data_access/tests/unit --disable-warnings --maxfail=3`：1716 passed，11 warnings，44.08 秒；`evidence/r2/H-dataaccess-unit-post-resource.log`。
- 上述结果是有限合成回归，不是生产落值或无 OOM 保证。Job 接线完成后另跑最终资源/默认入口回归。

补充独立回归（2026-09-10，结果有交集，不累加作唯一测试数）：

- `H-crosspackage-post-resource.log`：112 passed，3 warnings，40.01 秒，范围同前述身份/数学/Source 组合。
- `H-resource-scheduler-final.log`：42 passed，4.37 秒；命令 `.venv/bin/python -m pytest -q tests/r38/test_fixed_cadence_autopilot.py tests/r38/test_dynamic_sink_and_service_lease.py tests/runtime/test_execution_resources.py tests/runtime/test_r21_work_conserving_scheduler.py tests/runtime/test_r21_resource_preemption.py tests/runtime/test_parallel_region_scheduler_failure.py tests/multibackend/test_r45_scheduler_closure.py --maxfail=5`。
- `H-resource-fixture-mirrors-final.log`：15 passed，1.49 秒；包内 execution_resources/r45 两镜像。旧资源测试混用固定 4/8/16 GiB 虚构容量与服务器实际 cgroup 占用，现改为自洽 snapshot 或明确 synthetic execution pool；32 GiB 测试机假设删除，保留与真实探测值相等的严格防放大断言。CPU 非内存约束用例的 worker peak 改为确实足够小的 128 MiB，不放宽生产预算。
- `H-worker-ingest-resource-post-fixture.log`：38 passed，2 warnings，33.30 秒；命令 `.venv/bin/python -m pytest -q factor_engine/tests/runtime/test_v6_worker_resource_authority.py factor_engine/tests/runtime/test_v8_bounded_ingestion_lease.py factor_engine/tests/backend/test_v9_m36_resource_admission.py --maxfail=3`。
- `H-broker-host-final.log`：38 passed，1.88 秒；命令 `.venv/bin/python -m pytest -q tests/r27/test_resource_broker.py tests/r38/test_host_lease_tree.py factor_engine/tests/runtime/test_h36_unified_execution_envelope_20260910.py factor_engine/tests/runtime/test_v8_resource_wait_fairness.py --maxfail=3`。含同步两线程争抢同一 execution pool 的反例和释放后归零。

## Job、写出与 CSE 的最终接线

- `job_scoped_lease.py` 配对申请 job child 与 broker；host 拒绝/异常回滚 child；suppression 避免显式配对又被 raw hook 重复收费；真实 JobLease 与 broker identity 不符时，双方账本均不变。
- `ResourceBroker` 的 memory/task/protected-egress 原生入口均消费活跃 job；回调在物理 broker 账释放后归还 child。protected egress 两个物理 lease 共用一个总量 child，半释放不提前归还整块额度。并发竞争和异常回滚有专门负控。
- Job 关闭会拒绝新 child，已有 child 排空前保留 root；终结 child ID 从 parent 列表移除，避免长任务无界积累。调度器显式捕获 caller JobLease 并在执行线程绑定，不依赖 ContextVar 自动跨线程传播。
- DA query workspace 按 memory+scan 同时进入 job 和 host 两层账本。明确的 `HostLeaseAdmissionDenied` 在 production/interactive_research 均不可回退到另一套本地预算；普通非准入异常仍保留原有模式语义。
- bounded pipeline 显式将 parent JobLease 交给 IPC controller；writer/egress、compute/compile proxy 的 raw 请求由父 broker 同一 scope 处理。真实 IPC 测试验证申请、拒绝、半释放、完全释放和 job 关闭。active-job 缺少 engine_factory 的旧 fork 副本路线在任何输入遍历、目录/worker/sink 动作前以配置错误拒绝；正式 DurableFactorEngine.run_many 已提供受管理 factory，不是新增 fork 支持。
- 并行及默认串行 scheduler 均做 task 准入。CSE 完成时，broker 原子将运行峰值转换为实际 entry 大小的纯内存 lease，并立刻归还 CPU/IO；没有先释放再申请的未计费窗口。Job child 仍保守持有原峰值承诺至物理 owner 消失，不宣称精确 RSS 计量或最优并行度。
- `GovernedBufferStore` 使用稳定 generation snapshot，跟踪 NumPy view/base、Pandas blocks 与 axes/MultiIndex owners。逻辑清理、替换、spill、evict 和 store GC 只删除引用，不提前归还外部数组/索引仍使用的内存。未知 native/extension owner 明确失败关闭；绕过预检查或 finalizer 注册失败时保守持有，不能冒充完整原生 lazy 生命周期支持。
- 已真正 spill/recompute/lazy 或零字节的 CSE 与“未知但仍驻留”分开处理；synthetic fusion key 不再被直接当作 DAG task 索引。
- read/sink/automatic queue 的软目标按活跃 job 收缩；fresh 但来自较大额度的 autopilot snapshot 同样只缩不放大，不重置控制周期或放宽 pressure。

新增独立结果（其余命令同上述对应组合）：

- `H-dataaccess-unit-job-hook.log`：1718 passed，11 warnings，44.34 秒。
- `H-unified-resource-post-profile.log`：105 passed，7.34 秒。14 文件覆盖 r27 broker、r38 host/autopilot/sink、新统一池/active-job/pipeline scope、V8 等待、job helper、execution_resources、preemption、并行失败和 r45；早期失败日志保留。资源计划测试显式隔离 CPU 与内存约束；探测硬 CPU 上限与合法用户较小 CPU 配额分开断言。
- `H-ipc-crosssuite-final.log`：37 passed，6.27 秒；`factor_engine/tests/r27/test_resource_broker_ipc.py`、`factor_engine/tests/runtime/test_v8_broker_epoch_protocol.py`、`factor_engine/tests/runtime/test_v8_broker_ipc_quotas.py`、`factor_engine/tests/r27/test_v6_broker_summary_rpc.py`。
- `H-model-intraday-post-cse.log`：269 passed，2 warnings，40.00 秒，含不伪造预算的默认双因子调用。
- `H-cse-physical-full-final.log`：69 passed，9 warnings，40.41 秒；`.venv/bin/python -m pytest -q factor_engine/tests/runtime/test_cse_memory_lease.py tests/runtime/test_cse_scheduler_lease.py tests/planner/test_cse_run_many.py tests/backend/test_polars_long_cse_sql_partial.py tests/runtime/test_r21_cse_cache_identity.py factor_engine/tests/runtime/test_cse_cache.py --maxfail=4`。
- `H-r39-batch-final.log`：24 passed，0.59 秒，两个 r39 batch-service 镜像；fake Context 补实际使用的 task_id/factor_id。
- Polars collect 反例按阶段验证为 source=1、shared=0、root=2；不把有界 source-wave 一次物化误算成共享计算提前 collect。根及包内镜像均保留严格阶段断言；诊断栈见 `H-cse-collect-trace.log`（该临时探针替换 PYTHONPATH 导致已有 sitecustomize 的 alphaprobe 提示，不用于环境认证；常规回归不用该探针）。

默认小规模兼容调用为 `engine.run_many(factors)`；可运行完整样例见 `factor_engine/tests/operators/test_v9_m01_cs_huber_shared.py::test_default_pandas_batch_bridge_preserves_translation[default-broker]`，不指定并行、预算或后端性能参数。该内存返回式小样例不等同于十万因子持久化认证；批量生产落值仍须走已有受管理 durable/DataAccess 入口，本轮未发布生产因子。

## 最后验收核对

最终资源/CSE/串行/并行/微批/失败组合：159 passed，13.99 秒，`H-resource-cse-acceptance-final.log`。确切命令：

```sh
.venv/bin/python -m pytest -q tests/r27/test_resource_broker.py tests/r38/test_host_lease_tree.py factor_engine/tests/runtime/test_h36_unified_execution_envelope_20260910.py factor_engine/tests/runtime/test_h36_active_job_broker_hook_20260910.py factor_engine/tests/runtime/test_v8_resource_wait_fairness.py factor_engine/tests/runtime/test_v9_bounded_pipeline_job_scope.py tests/runtime/test_job_scoped_lease.py tests/r38/test_fixed_cadence_autopilot.py tests/r38/test_dynamic_sink_and_service_lease.py tests/runtime/test_execution_resources.py tests/runtime/test_r21_work_conserving_scheduler.py tests/runtime/test_r21_resource_preemption.py tests/runtime/test_parallel_region_scheduler_failure.py tests/multibackend/test_r45_scheduler_closure.py factor_engine/tests/runtime/test_cse_memory_lease.py tests/runtime/test_cse_scheduler_lease.py tests/r27/test_dag_scheduler_shard_fusion.py factor_engine/tests/runtime/test_run_many_sink_backpressure.py --maxfail=5
```

旧微批/共享调度完整文件 35 passed；对应 r27 包内镜像单独 6 passed（同名镜像不可在默认 pytest import 模式下混合收集，不以删除 pycache 掩盖模块名冲突），见 `H-microbatch-mirrors-rerun.log`。全体修改/新增 Python 编译检查、staged/unstaged diff 检查、路径门禁均通过。当前工作树改动与新增 Python/TOML/shell 共 100 个文件的哈希在 `H-worktree-source-sha256-20260910.txt`；它覆盖现有工作树，不把用户或其他 AI 的既有改动据为本轮独有。

生产持久化的既有默认入口如下（仅使用管理员已配置并授权的 `FACTOR_ENGINE_V2_PROFILE`；本轮未执行此生产示例，不猜数据、COS 或审批身份）：

```python
from factor_engine import get_engine
with get_engine() as engine:
    receipt = engine.run_many(factors)
```

最后核对仍为 `main` / `c5e880db91f672d101676f92f7901361ea48522d`；未 commit/push/部署。磁盘约 786 GB 可用，pytest 临时目录合计约 11 MiB；本轮自己的补丁、探针及过时小草稿已清理，没有复制正式库、数据或环境。本记录前述 15 个未实现 SQL、D02、最低版本环境和生产/GPU/十万因子认证边界仍然有效，不能据这些有限回归宣称“全算子全后端零问题”。
