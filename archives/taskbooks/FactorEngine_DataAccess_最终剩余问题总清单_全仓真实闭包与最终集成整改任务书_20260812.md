# FactorEngine × DataAccess 最终剩余问题总清单
## —— 全仓真实闭包、最终集成、生产可达性、证据完整性与深架构剩余整改任务书

> **用途：可直接整份发送给 Coding AI / Claude Code / Codex / Cursor。**
>
> 仓库：`https://github.com/18047533889/quant_projects`
>
> 本轮核验时 GitHub 可见最新 `main`：
>
> ```text
> ea02121edf2a9dd19b067dfe88f20d483cc9c0fb
> feat: R41/R42/R32 FE-DA closure, filter layer, and taskbook prompts.
> ```
>
> 当前 GitHub combined status：**无可见 status checks**。
>
> **执行本文件前必须重新 `git fetch` 并记录真实执行起点 HEAD。**
>
> 本文不是新的“再扩 300 条”任务书，而是把此前 R39 / R41 / R42 / R43、Model Operators、DataAccess R32、以及最新并行 Claude 审计暴露出的**所有剩余问题统一收口**。已经有明确代码+测试+current-SHA 证据证明修好的，不重复改；本地 agent 自报修好但尚未进入最终 main 的，必须先集成再验证。
>
> 最终目标只有一个：
>
> ```text
> 不再存在：
> - ledger 说 IMPLEMENTED、生产路径却不可达
> - evidence 写 PASS、实际行为 gate 没跑
> - 本地 working tree 修了、final main 没有
> - source tree 能 import、wheel 不能 import
> - 测试很多但不是 current SHA / clean tree / final wheel
> - 模块各自 exactly-once，系统整体却可能重复/泄漏/失真
> ```

---

# 0. 状态定义

```text
CONFIRMED_OPEN
    当前 main 或当前真实证据可直接证明仍未闭环。

ORPHANED
    基础设施/类/测试存在，但生产主路径没有调用。

NOT_RUN_EVIDENCE
    结构存在，但行为证据未执行，不得宣称 production-ready。

LOCAL_FIX_WAITING_INTEGRATION
    agent 报告已修并有局部测试，但当前可见 final main 尚不能证明。

DEEP_ARCH_NOT_CLOSED
    明确承认未完成的深架构/性能项。

VERIFY_REQUIRED
    agent/ledger 与 current code 结论冲突，需要重新实测。

P1_BACKLOG
    不阻塞当前 P0 correctness，但应纳入下一阶段性能/运维闭环。
```

最终只允许：

```text
CLOSED_WITH_CURRENT_HEAD_PROOF
NOT_APPLICABLE_WITH_PROOF
BLOCKED_BY_EXTERNAL_DEPENDENCY
```

对于 P0，不允许：

```text
PARTIAL
NOT_RUN
LIKELY_CLOSED
AGENT_REPORTED_DONE
```

---

# 1. Final Integration / Evidence Integrity

## REM-001 [P0][CONFIRMED_OPEN] 建立唯一 Final Integration Branch / Worktree
所有 agent 独立 worktree/branch 工作，禁止多个 Claude 直接修改同一个 dirty main。

## REM-002 [P0][CONFIRMED_OPEN] 只有中央 reconciler 可以写 CLOSED
普通 agent 最多写 `READY_FOR_INTEGRATION`。只有 merge 到 final integration HEAD、clean tree、全量测试、wheel、current-SHA evidence 后才可 CLOSED。

## REM-003 [P0][CONFIRMED_OPEN] 重写 R42 closure ledger
原先的 `101 IMPLEMENTED / 551+ focused tests` 已被后续审计证明不可靠。必须逐项重建真实状态。

## REM-004 [P0][CONFIRMED_OPEN] 建立 Reachability Audit
任何“实现完成”必须证明从 `FactorEngine.run/run_many/materialize/HTTP/mining/DataAccess public read` 能 reach 到目标代码。类存在不等于生产可达。

## REM-005 [P0][CONFIRMED_OPEN] current-SHA Evidence
所有最终 evidence 必须绑定：final commit SHA、wheel/container digest、dependency lock digest、test suite version、runtime capability profile。

---

# 2. R42 Compiler / Operator Contract 剩余

## REM-006 [P0][CONFIRMED_OPEN] OutputShapeContract 真正贯穿生产
必须贯穿 `metadata → build_operator_spec → OperatorSpec.output_shape → to_dict → manifest → semantic digest → analyzer/compiler → backend admission → production certificate`。

## REM-007 [P0][CONFIRMED_OPEN] shape-changing operator 禁止 loose catalog string admission
`shape_preserving=False` 时必须有 typed OutputShapeContract。

## REM-008 [P1][CONFIRMED_OPEN] Determinism 禁止 name/tag heuristic
建立显式 DeterminismContract。

## REM-009 [P1][CONFIRMED_OPEN] NumericalStability 禁止 scope/name heuristic
建立显式 NumericStabilityContract。

## REM-010 [P1][CONFIRMED_OPEN] panel/scalar/context params 禁止签名黑名单作为 production authority
production operator 显式声明参数角色。

## REM-011 [P0][ORPHANED] CompilerPassManager 未接入 Optimizer.optimize()
当前最大 orphaned 基础设施之一。必须让 production optimizer 真正消费 CompilerPassManager/PassContract。

## REM-012 [P0][ORPHANED] PassContract / NumericPolicy / CBO 可达性重审
所有依赖 PassManager 的 R42 项都要重新 trace；不可达即 ORPHANED。

## REM-013 [P1][VERIFY_REQUIRED] R42 300 项逐项重新分类
分类仅允许 CLOSED / ORPHANED / NEEDS_VERIFY / FALSE_STUB。

## REM-014 [P1][P1_BACKLOG] SemanticLattice 与 SemanticIdentity 维度统一
compiler type system最终覆盖 unit、price basis、flow、period duration、availability、knowledge time、revision、universe、missing policy、market、frequency。

## REM-015 [P0][CONFIRMED_OPEN] CNY/share 与 CNY 维度真正分开
`unit_normalization(CNY/share → CNY)` 不得视作同维度 1:1 conversion。

---

# 3. Model Operators：结构完成，行为认证未完成

## REM-016 [P0][NOT_RUN_EVIDENCE] 88 个 direct-use model operators Parameter Domain
当前关键证据：0/88 direct-use model operators 有 parameter-domain certification。

## REM-017 [P0][NOT_RUN_EVIDENCE] 88 个 direct-use model operators Oracle
逐 canonical reference/golden/edge corpus。

## REM-018 [P0][NOT_RUN_EVIDENCE] Causality/no-future
修改未来数据，过去输出必须不变。

## REM-019 [P0][NOT_RUN_EVIDENCE] Missing/Gap behavior
覆盖 NaN/Inf/gaps/sparse/state gaps。

## REM-020 [P0][NOT_RUN_EVIDENCE] Unit behavior
不只结构声明，必须运行行为证据。

## REM-021 [P0][NOT_RUN_EVIDENCE] Optimized vs Reference parity
所有 production fast/native path 对 reference parity。

## REM-022 [P0][NOT_RUN_EVIDENCE] Negative controls
未来污染、错单位、错 market/calendar、参数越界、wrong state、invalid chunking 必须真的 FAIL。

## REM-023 [P0][NOT_RUN_EVIDENCE] Stateful batch/chunk/restart parity
one-shot、chunked、checkpoint/restart 一致。

## REM-024 [P0][CONFIRMED_OPEN] `final_direct_use_ready_count` 必须达到 required count
为 0 时不得声称 model operators production-ready。

## REM-025 [P0][CONFIRMED_OPEN] Model evidence 重新绑定最终 SHA
旧 `6fd71595...` evidence 在 final integration HEAD 改变后全部 stale。

---

# 4. R41 Modeling 主体剩余的最终收口

## REM-026 [P0][LOCAL_FIX_WAITING_INTEGRATION] 279/279 modeling tests 在 final merged clean tree 重跑
本地 agent 全绿不等于最终 main 全绿。

## REM-027 [P0][CONFIRMED_OPEN] R41 evidence 在最终 clean commit 后重生
model hard gates、artifact、trainer、evaluation、security 全部重新绑定 final SHA。

## REM-028 [P0][VERIFY_REQUIRED] `load_all()` 在 clean final tree 明确证明无 hang
要求 clean venv + timeout + wall time。

---

# 5. R39 明确未完成的深架构

## REM-029 [P1][DEEP_ARCH_NOT_CLOSED] PERF-017 native source-level OutputSlice
backend 从源头只生成/读取必要 slice。

## REM-030 [P1][DEEP_ARCH_NOT_CLOSED] PERF-020 shared-memory / process worker lane
PlanId + BufferRef → long-lived worker，避免大 DataFrame pickle。

## REM-031 [P1][DEEP_ARCH_NOT_CLOSED] PERF-073 DataAccess ScopedConnectionPool
连接复用、状态 reset、deadline interrupt、lease ownership、no temp leakage。

## REM-032 [P1][DEEP_ARCH_NOT_CLOSED] PERF-078 NativePipelineRegion
连续 backend-compatible subgraph 合成 native region。

## REM-033 [P1][DEEP_ARCH_NOT_CLOSED] PERF-079 RollingStateBlock
同 input/window 多 rolling outputs 共享 traversal/state。

## REM-034 [P1][DEEP_ARCH_NOT_CLOSED] PERF-080 PrimitiveBlock
跨因子共享 return/finite mask/rolling primitive/rank input/neutralization matrix 等。

---

# 6. StreamingResultSink / Writer

## REM-035 [P0][CONFIRMED_OPEN] accepted 只在 enqueue 成功后增加
失败 put 不得污染 durable accounting。

## REM-036 [P0][CONFIRMED_OPEN] queue_bytes 必须是 sink 总预算
`sum(worker_queue_capacity) <= configured_total_queue_bytes`。

## REM-037 [P0][CONFIRMED_OPEN] Oversized ResultItem policy
item > queue budget 时必须 spool/direct/borrow/shard 之一，不得只等 timeout。

## REM-038 [P0][CONFIRMED_OPEN] 未知 writer exception 不得默认 transient
UNKNOWN 默认 permanent/unknown，只有 typed transient 可 retry。

## REM-039 [P1][P1_BACKLOG] writer backpressure 改为 throughput/drain-rate 驱动

---

# 7. FE × DA 入口一致性：本地已报修，等待 final integration

## REM-040 [P0][LOCAL_FIX_WAITING_INTEGRATION] calendar_id 传递
最终验证 pipeline/batch/config/materialize/event-driven 全路径。

## REM-041 [P0][LOCAL_FIX_WAITING_INTEGRATION] validate_spec backend 类型冲突
最终保证 ComputeRequest.backend / validate-spec / execution factory 接受集合一致。

## REM-042 [P0][VERIFY_REQUIRED] QueryBudget.tighten 类型安全实测
最终代码必须拒绝：
`max_rows=True`、`max_elapsed_ms=True`、`require_columns="false"`、`max_rows=1.2`。


# 8. Version / Packaging Governance

## REM-043 [P0][CONFIRMED_OPEN] DataAccess 唯一版本权威
当前曾同时出现：
`root distribution 0.10.2`、`r30.__version__ 0.10.3`、agent wheel `0.11.0.dev0+untagged`。
必须只留一个 distribution version authority，子模块使用 API generation 而不是私有 package version。

## REM-044 [P0][CONFIRMED_OPEN] `data_access.r30` 必须真实进入 wheel
源码目录存在不等于正式 wheel 包含。clean venv、源码目录外真实 import 验证。

## REM-045 [P0][VERIFY_REQUIRED] FactorEngine wheel 必须确认包含 `modeling`
不得用源码 checkout import 代替 clean wheel proof。

## REM-046 [P0][CONFIRMED_OPEN] FactorEngine 对 DataAccess dependency range 收紧
不能继续允许极旧 API 版本满足安装。按真实 compatibility CI 设 lower/upper bound。

## REM-047 [P0][P1_BACKLOG] FE↔DA Capability Handshake
startup 比对 API/read/snapshot/resource/semantic contract generations。

## REM-048 [P1][P1_BACKLOG] FE↔DA Compatibility Matrix
维护 Python、DuckDB、Polars、FE、DA 真实兼容矩阵。

## REM-049 [P1][VERIFY_REQUIRED] Build version / source SHA / wheel digest 一致
最终 `/health`、BuildManifest、package metadata、evidence 不得报告不同版本。

---

# 9. DataAccess R32 自报完成后的最终真实性审计

## REM-050 [P0][VERIFY_REQUIRED] R32 128 P0 项做 final-main Reachability Audit
“代码存在 + 新测试存在”不够，必须证明 public path 实际消费。

## REM-051 [P0][VERIFY_REQUIRED] R32 wheel 版本与最终 metadata 对齐
如果 final build 不产生报告所写的版本，completion report 必须修正。

## REM-052 [P0][VERIFY_REQUIRED] HTTP QueryBudget 字段完整保留
重点验证：
`max_scan_objects/max_scan_bytes/max_remote_list_objects/max_remote_requests/max_estimated_memory`。

## REM-053 [P0][VERIFY_REQUIRED] StartupCertificate 真正 frozen、mode-bound
research certificate 不得进入 production；certificate 绑定 build/config/registry/calendar/policy/source capability。

## REM-054 [P0][VERIFY_REQUIRED] Symlink/TOCTOU 做真实 race/fault injection
不只 unit path normalization。

## REM-055 [P0][VERIFY_REQUIRED] Generation atomicity 做 crash/power-loss test
必须覆盖 temp write、fsync、rename、directory fsync、catalog commit。

## REM-056 [P0][VERIFY_REQUIRED] HTTP stream exactly-once slot release
disconnect、exception、normal finish 三条路径都 exactly once。

## REM-057 [P0][VERIFY_REQUIRED] DQ fail-closed 真实异常注入
checker 自身异常、schema mismatch、coverage unknown 均不能默认为 PASS。

## REM-058 [P0][VERIFY_REQUIRED] Price-basis unknown fail-closed 全入口一致
Python/HTTP/batch/config/mining 都不能回落 RAW。

## REM-059 [P0][VERIFY_REQUIRED] DataAccess R32 public API surface 与 wheel inventory 对齐
文档 37 public symbols 必须全部真实 import。

---

# 10. JobStore / Queue / Idempotency

## REM-060 [P0][CONFIRMED_OPEN] SQLite idempotency durable winner
`ON CONFLICT DO NOTHING` 后必须读取 durable winner并返回同一个 run_id。

## REM-061 [P0][CONFIRMED_OPEN] JobStore create durable-first / rollback
持久化失败不得留下 memory ghost job。

## REM-062 [P0][CONFIRMED_OPEN] JobStore update/CAS durable-first / rollback
memory 与 SQLite/manifest 不能分叉。

## REM-063 [P0][CONFIRMED_OPEN] JobRecord 不得作为 store-owned mutable object 泄露
`get/list` 返回 immutable snapshot/copy，更新只能走 store API。

## REM-064 [P0][CONFIRMED_OPEN] STORE.create → QUEUE.submit transactional
建议 SUBMITTING + transactional outbox + dispatcher + QUEUED。

## REM-065 [P0][CONFIRMED_OPEN] queue admission check+reserve+enqueue 原子
global/per-principal limits 不得并发超卖。

## REM-066 [P0][CONFIRMED_OPEN] submit/drain 同一 barrier
drain barrier 后不得再进新 job。

## REM-067 [P0][CONFIRMED_OPEN] stop/drain terminalize pending queue jobs
不能留下 durable QUEUED + stopped worker + leaked quota。

## REM-068 [P0][CONFIRMED_OPEN] queued cancel 不应再申请 JobLease/执行
使用 queue tombstone/cancelled-pending set。

## REM-069 [P0][CONFIRMED_OPEN] active job retry 必须拒绝
QUEUED/RUNNING/CANCELLING 不可创建并发 retry。

## REM-070 [P0][CONFIRMED_OPEN] retry attempt lineage
必须有 root_operation_id / parent_run_id / attempt_id / retry_reason。

## REM-071 [P0][CONFIRMED_OPEN] side-effect commit receipt
materialize retry前先 reconcile generation commit outcome，禁止 blind retry。

## REM-072 [P0][CONFIRMED_OPEN] worker ownership / heartbeat fencing
worker_id + epoch + lease_token，stale worker不能写终态/commit。

## REM-073 [P1][P1_BACKLOG] heartbeat monitor 用 indexed query
不要每周期扫描全部历史 jobs。

## REM-074 [P1][P1_BACKLOG] queue/drain 改事件驱动
减少 sleep polling。

---

# 11. HostResourceCoordinator / DataAccess Governor

## REM-075 [P0][CONFIRMED_OPEN] release 成功后才能本地标 released
HostLease/DA reservation release异常不得先把 handle 标完成。

## REM-076 [P0][CONFIRMED_OPEN] release failure 必须进入 reconciliation
禁止 `except: pass` 永久泄漏 accounting。

## REM-077 [P0][CONFIRMED_OPEN] DA parent JobLease rejection 不得 fallback local governor
明确三态：NO_PARENT / GRANTED / PARENT_REJECTED。

## REM-078 [P0][CONFIRMED_OPEN] host-backed duplicate query_id 禁止覆盖旧 reservation
先 duplicate check，再注册 active reservation。

## REM-079 [P0][CONFIRMED_OPEN] host-backed fast path 不能绕过 per-principal / active-query governance
host 是 memory authority，不等于其它治理全部跳过。

## REM-080 [P1][P1_BACKLOG] JobLease child accounting O(1)
维护 active_child_memory/cpu/io counters，不每次扫描 children。

## REM-081 [P1][P1_BACKLOG] IO token 与 CPU token 分离
local read、remote read、writer、metadata 分开。

## REM-082 [P1][P1_BACKLOG] resource predicted-vs-actual calibration
每 job 回写 reserved vs peak RSS/scan/spill，更新 P95。

---

# 12. DataAccess ReadHandle / ScanHandle

## REM-083 [P0][CONFIRMED_OPEN] ReadHandle CLOSED 后不得通过 cached pandas/polars 再读
所有 public read 先检查 state。

## REM-084 [P0][CONFIRMED_OPEN] ReadHandle terminal stats exactly once
materialize→wait→close 不得重复累计 elapsed。

## REM-085 [P0][CONFIRMED_OPEN] ReadHandle reservation release 异常可重试/可 reconciliation
release callback 成功前不要丢 ownership。

## REM-086 [P0][CONFIRMED_OPEN] ScanHandle sibling reservation ownership
派生 filter/select handle 不能共享一个“任何 sibling close 就释放”的裸 reservation。

## REM-087 [P0][CONFIRMED_OPEN] ScanHandle close 后 collect 必须 fail
禁止 ungoverned execution。

## REM-088 [P0][CONFIRMED_OPEN] ScanHandle repeated collect 必须定义
one-shot / cache-result 二选一，不能第二次无 lease 再 scan。

## REM-089 [P0][CONFIRMED_OPEN] research snapshot revalidation lineage 必须描述实际读取 object set
不能 snapshot 去掉 changed files，但 LazyFrame 仍读旧路径。

## REM-090 [P0][CONFIRMED_OPEN] ScanHandle derived transform 更新 lineage
filter/select/rename 后 lineage 不得仍等于原 plan。

## REM-091 [P1][P1_BACKLOG] 拆 `ScanPlan` 与 `ScanResultHandle`
lazy composition plan 不应长期持 live execution reservation。

## REM-092 [P1][P1_BACKLOG] PreparedRead 两阶段 lease
resolution lease 与 execute lease 分开，避免 prepare 后长期占内存预算。

---

# 13. Startup / Readiness / Release Evidence

## REM-093 [P0][CONFIRMED_OPEN] FactorEngine readiness 不得依赖固定 UNKNOWN placeholders
S20/S23-S30 类 closure evidence 与 runtime ready 分离。

## REM-094 [P0][CONFIRMED_OPEN] release blocker 禁止 `_check_done()` 无条件 PASS
必须有行为 evidence/negative control。

## REM-095 [P0][CONFIRMED_OPEN] idempotency blocker 做并发行为测试
100 concurrent same key → exactly one durable winner。

## REM-096 [P0][CONFIRMED_OPEN] StartupCertificate evidence hash 绑定真实依赖
build、registry、calendar、policy、credential/source capability。

## REM-097 [P0][CONFIRMED_OPEN] StartupCertificate mode binding
research certificate 不能复用 production。

## REM-098 [P0][CONFIRMED_OPEN] StartupCertificate dependency invalidation
registry/calendar/policy变更立即 stale，不只24h TTL。

## REM-099 [P1][P1_BACKLOG] Liveness / Readiness / Degraded 分离
ready 只判断当前实例能否安全接单。

## REM-100 [P1][P1_BACKLOG] disk readiness 使用真实 watermark
不是 `disk_free_mb >= 0`。


# 14. HTTP / Config / Request Identity

## REM-101 [P0][LOCAL_FIX_WAITING_INTEGRATION] validate-spec 与 typed request schema parity
本地已报修，final main 必须验证。

## REM-102 [P0][CONFIRMED_OPEN] config-path job 冻结配置内容
提交时读取/解析/canonicalize/content digest，worker 不得只保存 path 后执行期重新读可变文件。

## REM-103 [P0][CONFIRMED_OPEN] config-path policy snapshot
job 绑定 immutable policy generation/digest，执行期不得读 reload 后的新 global policy。

## REM-104 [P0][CONFIRMED_OPEN] request digest 绑定所有 observable execution semantics
至少重新核：name、timeout、factor_id、expression、config content digest、market/calendar/universe/source/backend/precision。

## REM-105 [P0][CONFIRMED_OPEN] materialize idempotency identity 补全
不能只绑定 config_path/write_target/policy。

## REM-106 [P0][CONFIRMED_OPEN] catalog generations 必须是真 generation
market/calendar/backend evidence/compiler build 不能只是字符串名称 hash。

## REM-107 [P0][CONFIRMED_OPEN] source profile hash 禁止 `default=str`
使用 typed canonical serializer。

## REM-108 [P1][P1_BACKLOG] request budget snapshot 真正 per-request 一次
不能 validator 各自重新读 env。

## REM-109 [P1][P1_BACKLOG] request body byte cap 下沉到 ASGI/proxy receive
不能先 `await request.body()` 完整入内存后才检查。

## REM-110 [P1][P1_BACKLOG] DataAccess URI API validators 与普通 ReadRequest 对齐
columns/time/instruments/filter complexity/limit 不得绕过普通 endpoint 限制。

---

# 15. Security / Classification / Observability

## REM-111 [P0][CONFIRMED_OPEN] preview classification failure 默认 REDACT
classification infra 异常不得 raw preview fail-open。

## REM-112 [P0][CONFIRMED_OPEN] config-path result classification
没有 inline source_cfg 时，也必须从 frozen config/source lineage 推导 classification。

## REM-113 [P0][CONFIRMED_OPEN] traceback 不得直接作为普通 job artifact
public_error 与 admin/debug trace 分离；统一 sink-level redaction。

## REM-114 [P0][P1_BACKLOG] derived factor/model classification 自动 lineage meet/max
public + premium/restricted source → derived artifact继承最高限制。

## REM-115 [P1][CONFIRMED_OPEN] async request context 用 ContextVar
FastAPI 不使用 threading.local 作为 principal/request identity carrier。

## REM-116 [P1][CONFIRMED_OPEN] log context 必须 scope/reset
请求结束清除 run_id/execution_id/principal。

## REM-117 [P1][CONFIRMED_OPEN] histogram memory bounded
禁止永久 append raw observation list。

## REM-118 [P1][CONFIRMED_OPEN] metrics label cardinality 强制 schema
禁止 factor_id/request_id/artifact_id 等无界 label。

## REM-119 [P0][CONFIRMED_OPEN] structured logging 最后一层统一 secret sanitizer
`default=str` 不能直接输出 credential/config/URI/exception repr。

## REM-120 [P1][P1_BACKLOG] 泄漏/状态分裂 observability
监控 active leases/handles/streams/connections/orphan generations，以及 queue_without_store、running_without_lease 等 divergence。

---

# 16. Generation / Durable Write / Mutation Lock

## REM-121 [P0][CONFIRMED_OPEN] generation commit 真正 durability barrier
data/temp manifest fsync → atomic rename → parent dir fsync → catalog commit。

## REM-122 [P0][CONFIRMED_OPEN] explicit empty generation 同样 durable
delete-all 不是特殊例外。

## REM-123 [P0][CONFIRMED_OPEN] duplicate upsert keys 拒绝
new_table 内部 duplicate keys 必须 fail 或显式 conflict policy。

## REM-124 [P0][CONFIRMED_OPEN] SQL identifiers/literals/path 全部安全 quoting/binding
partition columns/path 不直接字符串插 SQL。

## REM-125 [P0][CONFIRMED_OPEN] partition path canonical escaping
partition value 含 `/`, `=`, `..`, Unicode 特殊值时安全 round-trip。

## REM-126 [P0][CONFIRMED_OPEN] generation validation 提升
校验 schema一致、expected partitions、key uniqueness、identity、checksum/content digest。

## REM-127 [P1][P1_BACKLOG] generation write 避免 O(P×N) partition filtering
一次 partition writer / DuckDB COPY PARTITION_BY / Arrow dataset write。

## REM-128 [P1][P1_BACKLOG] generation writer 统一 WriterPolicy
compression、row_group_size、stats、dictionary、sorting metadata 不随调用点漂移。

## REM-129 [P0][CONFIRMED_OPEN] writer fencing token
stale writer失去 lock/lease 后不能 flip manifest。

## REM-130 [P1][P1_BACKLOG] old generation GC 考虑 in-flight reader
manifest flip 后不能立即删除旧 generation。

## REM-131 [P0][P1_BACKLOG] DurableCommitReceipt
包含 generation_id/manifest/schema/rows/partitions/content_digest/fsync/catalog状态，供 retry/recovery。

## REM-132 [P1][P1_BACKLOG] mutation lock corrupt payload recovery
torn lock JSON不能永久砖化 dataset。

---

# 17. Universe / Calendar / Data Semantic Accuracy

## REM-133 [P0][CONFIRMED_OPEN] named universe 对所有因子应用 output domain
TS-only factor声明 CSI300 时也不能输出全 A。

## REM-134 [P0][CONFIRMED_OPEN] CS operator区分 computation universe 与 output universe
CS 需要完整计算域；最终输出域单独明确。

## REM-135 [P0][CONFIRMED_OPEN] instrument_filter 不得因为“非空”就视为 universe 已匹配
必须比较 ResolvedUniverseSnapshot / FilterIdentity digest。

## REM-136 [P0][CONFIRMED_OPEN] cross-sectional capability 来自 typed OperatorAxisContract
registry lookup异常不得 name-prefix fallback fail-open。

## REM-137 [P0][P1_BACKLOG] Universe/industry membership 双时点
validity + knowledge intervals。

## REM-138 [P0][P1_BACKLOG] SessionOrdinal 成为 FE/DA 公共时间底层类型
自然日/业务日/observed row index 不混用。

## REM-139 [P0][P1_BACKLOG] minute bar/session schedule 完全来自 MarketCalendarSnapshot
不能固定 240 bars/09:30等 fallback 进入 production。

## REM-140 [P0][P1_BACKLOG] financial FiscalPeriodId / duration continuity
YTD→single period 需 fiscal calendar/period continuity。

---

# 18. DataAccess 读取与 Deadline

## REM-141 [P0][VERIFY_REQUIRED] Arrow/stream/scoped 四条 deadline 语义完全一致
absolute deadline 从入口生成一次，pool wait/execute/batch consume都消费同一剩余时间。

## REM-142 [P1][DEEP_ARCH_NOT_CLOSED] per-query watchdog thread 收敛到 DeadlineManager
高并发 deadline 线程数 O(1)。

## REM-143 [P1][DEEP_ARCH_NOT_CLOSED] ScopedConnectionPool 接真实 scoped SQL path
每查询新 duckdb.connect 不应长期存在。

## REM-144 [P1][P1_BACKLOG] actual remote request metering
LIST/HEAD/GET/range/retry 真正扣 remote budget，而不是只估对象数。

## REM-145 [P1][P1_BACKLOG] cold/warm ScanCost 分位数 calibration
dataset 一个 EMA 不足，按 ScanShape/host/storage/cache/concurrency 建 P50/P95。

## REM-146 [P1][P1_BACKLOG] downstream-aware read strategy
比较端到端 TTDC，而不是固定 rows/bytes threshold 选择 DuckDB/Polars/stream。

---

# 19. Columnar Runtime / Native Reachability

## REM-147 [P1][DEEP_ARCH_NOT_CLOSED] DataAccess → Arrow/Relation first
减少 DA→Pandas→Arrow→Polars。

## REM-148 [P1][DEEP_ARCH_NOT_CLOSED] BufferRef 真实物理 representation
不得只贴 preferred representation 标签；source 返回 actual handle/representation/bytes/snapshot。

## REM-149 [P1][P1_BACKLOG] AxisIdentityCertificate 热路径 O(1)/向量化验证
避免每次 `.to_list()` Python boxing 比较整轴。

## REM-150 [P1][DEEP_ARCH_NOT_CLOSED] UnifiedQueryGraph
FE 与 DA 共享 SourceResolve/Scan/PITJoin/Panelize/Primitive/Materialize physical graph。

## REM-151 [P1][P1_BACKLOG] long↔wide 转换最小化
pivot 延迟，只在真正需要 wide TS kernel 时做。

## REM-152 [P1][P1_BACKLOG] internal dense SecurityMasterId / SessionOrdinal
避免 Python ticker/Timestamp 进入 hot kernel。

---

# 20. FactorMatrix / Delta / Incremental

## REM-153 [P1][P1_BACKLOG] Delta storage 生产默认切换前做真实 shadow validation
read amplification、compaction debt、recovery 全通过后再变默认。

## REM-154 [P1][P1_BACKLOG] DeltaIndex
读取只选相关 fragments，不 base+全部 deltas 全扫。

## REM-155 [P1][P1_BACKLOG] RowDelta 与 ColumnDelta 分离
新日期与新增因子列采用不同物理策略。

## REM-156 [P1][P1_BACKLOG] Matrix 新增列不得重写旧列
factor block layout + manifest。

## REM-157 [P1][P1_BACKLOG] legacy Pandas outer merge 热路径继续去除
Arrow/DuckDB keyed overlay。

## REM-158 [P1][P1_BACKLOG] ChangeImpact 精确重算范围
source revision → affected instruments/dates/factors/history transfer，不全历史重算。

## REM-159 [P1][P1_BACKLOG] source-aware dependency watermark
不同 source 独立 watermark/generation。

---

# 21. Mining / AlphaProbe / CogAlpha 吞吐剩余

## REM-160 [P1][P1_BACKLOG] semantic duplicate 分层
Text/AST/CanonicalIR/semantic-equivalent/high-correlation 分开，只有严格等价可跳执行。

## REM-161 [P1][P1_BACKLOG] Negative Compile Cache
确定性无效 candidate O(1) 拒绝。

## REM-162 [P1][P1_BACKLOG] dependency-signature clustering
按 source/columns/history/backend/primitive family 聚类，减少扫描/转换。

## REM-163 [P1][P1_BACKLOG] MiningCampaignSession pinned snapshot
一个 campaign 内 source/universe/compiler snapshot 不漂移。

## REM-164 [P1][P1_BACKLOG] multi-fidelity evaluation
L0 legality → L1 small DQ → L2 recent → L3 full historical → L4 robust OOS。

## REM-165 [P0][P1_BACKLOG] pre-screen False-Negative Budget
便宜预筛只能减少成本，不能大量误杀最终好因子。

## REM-166 [P1][P1_BACKLOG] FactorBlock streaming evaluator
候选先 evaluate，只有通过者 durable materialize。

## REM-167 [P1][P1_BACKLOG] factor evaluation cache
factor semantic identity + snapshot + universe + metric convention。

---

# 22. Numeric / Cross-backend Accuracy

## REM-168 [P0][VERIFY_REQUIRED] NumericPolicy 是否真实进入生产 path
类存在不够，必须 trace 到 backend selection/compute/materialization identity。

## REM-169 [P0][VERIFY_REQUIRED] QuantizationCertificate 生产 float32 hard gate
未认证 float32 不得 production publish。

## REM-170 [P0][P1_BACKLOG] backend parity truth table
NaN/Inf/±0/tiny/huge/constant/ties/gaps 全 backend。

## REM-171 [P0][P1_BACKLOG] moment convention 单一 authority
ddof/bias/fisher/EWM adjust/ignore_nulls/quantile interpolation。

## REM-172 [P1][P1_BACKLOG] stable reductions
large-window sum/mean/cov/variance 用 compensated/Welford/QR/SVD 等稳定算法。

## REM-173 [P0][P1_BACKLOG] ApproximationCertificate
randomized SVD/approx quantile/sketch 未来进入 production 前必须误差证明。

---

# 23. 性能证据：当前 QUICK benchmark 不足

## REM-174 [P1][CONFIRMED_OPEN] R39 只证明 QUICK workload
300股×252日×100因子、约26% TTDC 改善不能代表目标 workload。

## REM-175 [P1][CONFIRMED_OPEN] 跑 1000 / 5000 / 10000 factor 全 A benchmark
至少5年日频，真实生产 source。

## REM-176 [P1][CONFIRMED_OPEN] 跑 1-day incremental benchmark
目标：historical rewrite bytes=0、full history rescan=0。

## REM-177 [P1][CONFIRMED_OPEN] 跑 minute→daily 真实大规模 benchmark
100/500 聚合，验证 source scan count ≈ unique source scopes。

## REM-178 [P1][CONFIRMED_OPEN] 跑 cold / warm / steady-state
避免复用已填 lake/cache 产生假加速。

## REM-179 [P1][CONFIRMED_OPEN] 跑 contention benchmark
factor batch + model training、2 mining campaigns、compaction、remote scan。

## REM-180 [P1][CONFIRMED_OPEN] 性能 evidence 绑定机器负载
记录 CPU/IO/RSS/其它并发 pytest，不再把高噪声数字当稳定结论。


# 24. CI / Build / Evidence 最终闭环

## REM-181 [P0][CONFIRMED_OPEN] 当前 final main 必须有真实 CI checks
不能再出现 GitHub commit 无 status checks 仍声称 production closed。

## REM-182 [P0][CONFIRMED_OPEN] clean-tree 全量测试
所有并发 agent 停止后，在同一个 final integration HEAD 上一次性跑完整测试。

## REM-183 [P0][CONFIRMED_OPEN] source tree tests 与 wheel tests 分开
源码全绿不代表 wheel 可用。

## REM-184 [P0][CONFIRMED_OPEN] FE clean wheel smoke
源码目录外 import modeling/runtime/service/storage 并跑最小 factor/model workflow。

## REM-185 [P0][CONFIRMED_OPEN] DA clean wheel smoke
源码目录外 import r30/read/write/service，并跑实际 read/snapshot/governed path。

## REM-186 [P0][CONFIRMED_OPEN] FE+DA 两 wheel 联合 clean-install integration
不能分别 wheel green、组合后 ABI crash。

## REM-187 [P0][CONFIRMED_OPEN] final build evidence 只从实际执行生成
禁止手工改 JSON SHA / PASS 字段作为“证据重生”。

## REM-188 [P0][CONFIRMED_OPEN] NOT_RUN 在 final production gate 中等于 FAIL
结构 gate 可 NOT_APPLICABLE，但 required behavior gate 不允许 NOT_RUN。

## REM-189 [P0][CONFIRMED_OPEN] baseline/pre-existing 失败重新冻结
最终 baseline 必须是 clean known commit；“pre-existing”必须在该 commit 真实复现。

## REM-190 [P0][CONFIRMED_OPEN] 测试进程清场
全量 acceptance 前停止无关 pytest/agent，记录 host load，避免 20–100 分钟并发进程污染结论。

## REM-191 [P1][P1_BACKLOG] 测试 shard / marker 重构
长 suite 按 operators/modeling/dataaccess/integration/perf 拆分，减少 agent 重复跑全量。

## REM-192 [P1][P1_BACKLOG] flaky test registry
禁止“偶尔过”直接计 green；flaky 必须有 owner/root cause/status。

---

# 25. R32 明确剩余 150 P1 的处理规则

## REM-193 [P1][CONFIRMED_OPEN] 导入 R32 P1 backlog 作为最终附表
DataAccess agent 已明确报告仍有约150项 P1 未完成。Coding AI 必须读取真实 R32 P1 ledger/taskbook，逐项映射，不允许用“非阻塞”把它们全部忽略。

## REM-194 [P1][VERIFY_REQUIRED] R32 P1 分四类
重新分类：
`PERFORMANCE` / `OBSERVABILITY` / `OPERABILITY` / `MAINTAINABILITY`。

## REM-195 [P1][VERIFY_REQUIRED] P1 中凡会造成 OOM/资源泄漏/错误恢复/无界内存的升级为 P0
严重度按实际影响，不按旧文档标签机械继承。

## REM-196 [P1][P1_BACKLOG] P1 性能项必须有 before/after benchmark
无 benchmark 的“优化”只能 CLOSED_FUNCTIONAL，不得 CLOSED_PERFORMANCE。

---

# 26. Final Integration 必须运行的负向测试

## REM-197 [P0] R42 orphan infrastructure reachability test
对 CompilerPassManager/PassContract/NumericPolicy 记录真实 production call counter，必须 >0。

## REM-198 [P0] 88 model canonical parameter-domain completeness
required=88，certified=88，missing=0。

## REM-199 [P0] 88 model canonical behavior evidence
oracle/causality/unit/missing/parity/negative required gates 全部 executed。

## REM-200 [P0] Queue concurrent idempotency
多线程/多进程 same key → exactly one durable winner。

## REM-201 [P0] JobStore persistence failure injection
create/update/CAS 各阶段失败 → memory durable state一致。

## REM-202 [P0] submit-vs-drain race
barrier 后 0 new enqueue。

## REM-203 [P0] retry-after-commit
data已commit但job finalization失败 → retry finalize/no-op，不重复副作用。

## REM-204 [P0] stale worker fencing
旧worker恢复后不能 publish/update终态。

## REM-205 [P0] ReadHandle close-state
close后所有 public read reject。

## REM-206 [P0] ScanHandle sibling ownership
sibling close不能让另一个无 lease执行。

## REM-207 [P0] parent lease rejection
DA不得 fallback local governor。

## REM-208 [P0] startup research→production certificate
research cert不可直接复用。

## REM-209 [P0] readiness live behavior
坏 idempotency/坏queue/坏resource/坏source时 readiness真实降级，不靠源码 inspect。

## REM-210 [P0] config-path TOCTOU
提交后修改文件 → frozen config执行或明确拒绝。

## REM-211 [P0] preview classification failure
classification异常 → redact，不泄露。

## REM-212 [P0] generation crash matrix
写文件、fsync、rename、dir fsync、catalog、job finalization逐点 kill。

## REM-213 [P0] universe poison
CSI300 factor给错误10只instrument_filter → hard reject。

## REM-214 [P0] CNY/share unit negative
CNY/share→CNY直接 normalization必须失败。

## REM-215 [P0] current-SHA evidence tamper
改任一 evidence SHA/依赖 digest → freshness gate fail。

---

# 27. Final Integration 的执行顺序

## Phase 0 — Freeze
```text
停止所有并发写同一 working tree 的 agent
创建 final-integration branch/worktree
记录 base HEAD
```

## Phase 1 — Merge 已经本地报修的成果
重点：
```text
calendar_id
validate_spec backend
Version/Packaging agent
R41 final fixes
R32 final fixes
```

每个独立 commit，逐个 focused test。

## Phase 2 — Truth Audit
```text
重写 R42 ledger
Reachability Audit
模型算子 evidence inventory
R32 P1 inventory
```

## Phase 3 — P0 code closure
按优先级：
```text
CompilerPassManager production reachability
88 model canonical certification
version/wheel
writer accounting
JobStore/Queue/idempotency
lease/handle ownership
startup/readiness
generation durability
universe/unit
```

## Phase 4 — Deep architecture
```text
R39 6项
ScopedConnectionPool
NativePipelineRegion
RollingStateBlock
PrimitiveBlock
shared-memory process lane
source-level OutputSlice
```

## Phase 5 — Clean acceptance
```text
clean tree
full FE tests
full DA tests
FE×DA integration
wheel build
clean install
negative tests
fault injection
```

## Phase 6 — Performance
```text
1000/5000/10000 factors
1-day incremental
minute→daily
cold/warm
contention
```

## Phase 7 — Evidence Freeze
```text
freeze final SHA
regenerate all evidence from execution
verify evidence SHA/wheel digest
commit evidence
CI on final SHA
```

---

# 28. 最终 Release Hard Gates

## HG-FINAL-01 Truth
```text
FALSE_IMPLEMENTED_LEDGER_COUNT == 0
ORPHANED_P0_COUNT == 0
```

## HG-FINAL-02 Model Operators
```text
DIRECT_USE_REQUIRED == DIRECT_USE_READY
PARAM_DOMAIN_MISSING == 0
BEHAVIOR_NOT_RUN == 0
```

## HG-FINAL-03 Compiler
```text
PRODUCTION_OPTIMIZER_USES_PASS_MANAGER == true
PASS_CONTRACT_REACHABLE == true
```

## HG-FINAL-04 Packaging
```text
ONE_DA_VERSION_AUTHORITY
FE_WHEEL_MODELING_IMPORT_PASS
DA_WHEEL_R30_IMPORT_PASS
FE_DA_CLEAN_INSTALL_PASS
```

## HG-FINAL-05 Job State
```text
IDEMPOTENCY_SPLIT_BRAIN == 0
GHOST_JOB == 0
ORPHAN_QUEUE_JOB == 0
BLIND_RETRY_SIDE_EFFECT == 0
```

## HG-FINAL-06 Resource
```text
UNRELEASED_LEASE_WITHOUT_RECONCILIATION == 0
PARENT_REJECT_LOCAL_FALLBACK == 0
SCAN_AFTER_RELEASE == 0
```

## HG-FINAL-07 Data Semantics
```text
WRONG_UNIVERSE_EXECUTION == 0
UNKNOWN_PRICE_BASIS == 0
CNY_PER_SHARE_DIMENSION_COLLISION == 0
```

## HG-FINAL-08 Durability
```text
GENERATION_COMMIT_WITHOUT_FSYNC_BARRIER == 0
STALE_WRITER_COMMIT == 0
DUPLICATE_UPSERT_KEY == 0
```

## HG-FINAL-09 Evidence
```text
CURRENT_SHA_MISMATCH == 0
FINAL_REQUIRED_NOT_RUN == 0
CI_STATUS_REQUIRED_AND_PRESENT == true
```

## HG-FINAL-10 Performance
```text
NO_CORRECTNESS_REGRESSION
1000_FACTOR_TTDC_NO_REGRESSION
1DAY_INCREMENTAL_NO_HISTORY_REWRITE
```

---

# 29. Coding AI 最终交付格式

最终必须输出一份：
`docs/FINAL_INTEGRATION_CLOSURE_REPORT.md`

至少包含：

```text
1. execution-start SHA
2. final SHA
3. clean/dirty state
4. merged agent commits
5. REM-001..REM-215 status
6. R42 300项真实重分类统计
7. 88 model canonical certification matrix
8. R32 P1 remaining matrix
9. R39 6深架构状态
10. FE full tests
11. DA full tests
12. FE×DA integration tests
13. wheel clean-install
14. negative/fault-injection results
15. current-SHA evidence digest
16. benchmark before/after
17. exact remaining blockers
```

禁止只写：

```text
“全部完成”
“xx个测试通过”
“agent全部交付”
```

而不提供：

```text
命令
SHA
测试文件
pass/fail/not-run
evidence path
reachability proof
```

---

# 30. 最终收官标准

只有以下全部成立，才能说整个 FactorEngine × DataAccess 基础设施“生产闭包”：

```text
A. current main 就是所有整改实际代码，不存在未集成本地修复
B. 所有 P0 基础设施都在真实生产入口可达
C. 88 direct-use model operators 行为证据完整
D. R42 ledger 与代码一一对应，无虚假 IMPLEMENTED
E. FE/DA wheel 与源码、版本、依赖完全一致
F. Job/Queue/Lease/Handle/Write 的跨模块状态机可做 crash/retry/cancel 证明
G. PIT/universe/calendar/unit/price-basis 机器语义统一
H. final SHA 有 clean CI、clean wheel、current-SHA evidence
I. 目标规模性能不回归，且增量链路真正少扫/少算/少写
```

在此之前，最准确的状态是：

```text
CORE FUNCTIONALITY LARGELY IMPLEMENTED
FINAL INTEGRATION NOT CLOSED
BEHAVIOR EVIDENCE INCOMPLETE
DEEP PERFORMANCE ARCHITECTURE PARTIAL
```

这比“98.9% 完成”更接近真实工程状态。
