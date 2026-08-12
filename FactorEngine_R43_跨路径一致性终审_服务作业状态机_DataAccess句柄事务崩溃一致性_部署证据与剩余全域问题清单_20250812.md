# FactorEngine × DataAccess R43 跨路径一致性终审
## —— 服务作业状态机、句柄所有权、事务/崩溃一致性、Startup/Readiness 证据、部署与剩余全域问题整改任务书

> **用途：整份直接发送给 Coding AI / Codex / Cursor / Claude Code。**
>
> 仓库：`https://github.com/18047533889/quant_projects`
>
> 本轮审计时 GitHub connector 可见 `main`：
>
> ```text
> 4b1577fce14c097fba48619885b38d6c596242eb
> ```
>
> **执行本文件前必须重新 `git fetch`，以执行时最新 `main` 为真实基线。**
>
> 本文件明确 **不重复 R42-001~R42-300**。R42 重点是统一编译执行、列式 IO、CSE/native region、数值精度、增量物化和 mining throughput；R43 专门继续审计：
>
> ```text
> HTTP / Python API
> → Validation
> → JobStore
> → Queue
> → Cancellation / Retry
> → Host Lease
> → DataAccess Reservation
> → ReadHandle / ScanHandle
> → QueryBudget
> → Generation Write
> → Startup Certificate
> → Readiness / Preflight / Evidence
> → Observability
> → Packaging / CI / Recovery
> ```
>
> 标记：
>
> ```text
> [VERIFIED] 当前审计 HEAD 可直接由代码证明的问题
> [ARCH]     下一阶段应补的架构能力，实施前需映射最新 HEAD
> [P0]       可能造成错误结果、权限/资源绕过、重复执行、状态分裂或生产不可用
> [P1]       明显稳定性/性能/可维护性问题
> [P2]       中长期平台化能力
> ```

---

# 0. R43 的核心判断

R42 之后，下一类最危险的问题不再是“某个 kernel 慢一点”，而是：

```text
局部模块都自称 fail-closed / exactly-once / immutable / durable，
但跨模块组合后并没有形成同一个事务和同一个所有权模型。
```

本轮重点检查五类一致性：

```text
1. Admission Consistency
   请求是否只准入一次、资源是否只租一次、幂等是否真的不重复执行。

2. State Consistency
   JobStore 内存态、SQLite/manifest、Queue、worker、cancel token 是否一致。

3. Ownership Consistency
   ReadHandle / ScanHandle / reservation / stream / sibling handle 谁拥有谁、谁释放谁。

4. Durability Consistency
   写完、rename、manifest pointer、fsync、catalog commit 到底在哪一点才叫 durable。

5. Evidence Consistency
   startup/readiness/CI/evidence 是否证明“当前这份代码和配置真的可生产”，
   还是只检查“某个函数存在 / 某个字符串出现 / 返回固定 PASS”。
```

---

# 1. 本轮直接确认的最高优先级问题


## R43-001 [VERIFIED][P0] Queue admission 的“检查 + 占位”不是原子事务

文件：

```text
factor_engine/service/queue.py
```

当前 `submit()` 先在锁内读取：

```text
running_now
queued_now
per_principal
```

释放锁后判断限制，随后再次加锁才增加 per-principal 计数并 `put_nowait()`。

两个并发 submitter 可以同时看到：

```text
per_principal = N-1
```

然后都通过并各自 +1，超过限制。

### 修复

把：

```text
state check
global/per-principal limit check
per-principal reservation
queue insertion
```

放到同一个 admission transaction；失败必须回滚 reservation。

---

## R43-002 [VERIFIED][P0] `submit()` 与 `drain()` 的 service state 存在 TOCTOU

`submit()` 对：

```text
_state == "accepting"
```

的检查不与最终 queue insertion 共用同一个锁。

可能：

```text
submit: 看见 accepting
drain:  切成 draining
submit:  继续 enqueue
```

### 修复

`accepting → draining` 与 admission 使用同一状态锁/condition/CAS。

Hard Gate：

```text
JOB_ENQUEUED_AFTER_DRAIN_BARRIER == 0
```

---

## R43-003 [VERIFIED][P0] `drain()` 超时后只处理中途 running jobs，pending queue 会成为 orphan

当前 drain force 分支主要：

```text
running → interrupted/cancel
```

但未把仍在 queue 里的 queued/pending jobs 全部：

```text
durably CANCELLED/INTERRUPTED
decrement principal counters
remove/tombstone queue item
release associated token
```

之后 service state 进入 stopped，worker 不再正常消费这些 item。

### 后果

可能留下：

```text
JobStore: QUEUED/SUBMITTED
Queue: 内存残留
worker: 已停止
principal quota: 被占用
```

---

## R43-004 [VERIFIED][P0] `stop()` 同样缺 pending-job terminalization

直接 stop 不能只停线程。

必须有：

```text
StopAdmission
→ DrainOrCancelPending
→ FenceRunning
→ PersistTerminalStates
→ ReleaseQuotas
→ StopWorkers
```

---

## R43-005 [VERIFIED][P1] queued job cancel 不会真正从 pending queue 移除

`cancel()` 修改状态/事件，但 queue entry 仍存在。

worker 后续仍会 dequeue，甚至走到 JobLease admission。

应增加：

```text
queue tombstone / cancelled-pending set
```

在资源申请前 O(1) 跳过。

---

## R43-006 [VERIFIED][P1] `cancel()` 持有 queue lock 做 JobStore I/O

持 queue `_lock` 期间：

```text
STORE.get
STORE.update
```

如果 SQLite/fsync 变慢，会阻塞：

```text
submit
worker cleanup
snapshot
其它 cancel
```

### 修复

锁内只改队列内状态；持久化通过短事务/outbox，不要把磁盘 I/O 包在 scheduler lock 内。

---

## R43-007 [VERIFIED][P0] heartbeat monitor 没有 worker ownership fencing

monitor 遍历 durable store 中的 RUNNING jobs，并按 heartbeat 过期标 interrupted。

未来一旦使用共享 SQLite/多 service instance：

```text
instance A
```

可能看到：

```text
instance B 正在跑的 job
```

然后错误 interrupt。

### 修复

JobRecord 必须有：

```text
worker_instance_id
worker_epoch
lease_token
heartbeat_version
```

monitor 只能 CAS 自己 lease 的 job。

---

## R43-008 [VERIFIED][P1] heartbeat monitor 每周期扫描全部历史 jobs

当前是：

```text
STORE.list_jobs()
→ 过滤 RUNNING
```

job 历史越大，monitor 越慢。

需要持久层索引：

```sql
WHERE status='running'
AND heartbeat_at < ?
AND worker_instance_id=?
```

---

## R43-009 [VERIFIED][P1] 未估算 job 默认拿 safe memory 的固定 25%

这会导致资源准入过于粗糙：

```text
4 个未知 job ≈ 100% safe memory
```

但真实 shape 可能完全不同。

必须要求 service admission cost model 产出：

```text
P50/P95 peak
expected output
scan bytes
```

未知成本 production 走 conservative class，而不是一个固定比例。

---

## R43-010 [VERIFIED][P1] queue shutdown/drain 使用 sleep polling

不是 correctness bug，但长期服务应改：

```text
Condition/Event driven
```

而不是每 200ms 醒来检查。

---

# 2. HostResourceCoordinator / JobLease

## R43-011 [VERIFIED][P0] Host lease handle 在真正 release 前先把自己标记 `released=True`

文件：

```text
factor_engine/runtime/host_resource_coordinator.py
```

当前 wrapper release 的形态是：

```text
self.released = True
try coordinator.release(...)
except: pass
```

如果 coordinator release 失败：

```text
host accounting 没释放
handle 却认为已释放
以后无法 retry
```

这是典型资源账实永久分裂。

### 修复

```text
coordinator release 成功
→ 才切 local state RELEASED
```

失败必须 telemetry + recovery queue。

---

## R43-012 [VERIFIED][P1] JobLease.release 丢弃 coordinator release 的成功/失败信息

需要返回/记录：

```text
ReleaseReceipt
released_bytes
released_tokens
children_remaining
generation
```

而不是 fire-and-forget。

---

## R43-013 [VERIFIED][P0] ResourceDecision 仍可能基于 legacy `_job_lease_bytes`，不是当前 active JobLease

service 实际用：

```text
request_job_lease()
```

但部分 decision/summary 仍依赖 legacy global scalar。

可能出现：

```text
controller 建议 2GiB wave
当前 job root lease 只剩 512MiB
→ admission 一直被 child lease 拒绝
```

### 修复

decision 必须是：

```text
HostEnvelope
∩ current JobLease remaining
∩ job QoS
```

---

## R43-014 [VERIFIED][P1] coordinator rejection history 无界增长

成功 terminal lease 有 bounded history，但 rejected strings/list 应同样：

```text
ring buffer + counters
```

否则长运行服务每次资源拒绝都积内存。

---

## R43-015 [VERIFIED][P1] coordinator summary 在主锁内执行较重的 probe/decision/reconcile

可能阻塞真实 lease admission。

### 修复

锁内 snapshot accounting；
锁外做：

```text
OS probes
PSI
disk
decision formatting
```

---

## R43-016 [VERIFIED][P1] coordinator 读取 broker private `_usable_spill()`

这形成跨模块私有 API 耦合。

改成正式：

```text
broker.spill_capacity()
```

并版本化 contract。

---

## R43-017 [VERIFIED][P1] IO token ceiling 仍以 CPU slot 数做代理

磁盘/NVMe/COS bandwidth 与 CPU 核数不是同一个资源。

至少区分：

```text
local_read_tokens
remote_read_tokens
writer_tokens
metadata_tokens
```

---

## R43-018 [VERIFIED][P1] child lease admission 每次扫描 children 求和

高 task 数 job 下形成 control-plane 放大。

JobLease 应维护：

```text
active_child_memory
active_child_cpu
active_child_io
```

O(1) 增减。

---

# 3. JobStore：内存态、manifest、SQLite 的真实事务问题

## R43-019 [VERIFIED][P0] SQLite idempotency conflict 时会出现“内存新 job / SQLite 老 job” split-brain

文件：

```text
factor_engine/service/jobstore.py
```

当前 create 路径：

```text
内存先写新 job
SQLite:
INSERT ... ON CONFLICT(...) DO NOTHING
```

但 DO NOTHING 后没有：

```text
检查 rowcount
SELECT durable winner
用 winner 修正内存
```

### 并发场景

进程 A 已插入 idempotency tuple；
进程 B：

```text
内存记 run_B
SQLite 保留 run_A
create() 却仍可能返回 run_B
```

P0。

---

## R43-020 [VERIFIED][P0] `create()` 先改内存索引再持久化，持久化失败会留下 ghost job

应改：

```text
durable transaction / manifest prepare
→ commit
→ publish in-memory snapshot
```

或发生异常完整 rollback：

```text
_jobs
_idempotency_index
```

---

## R43-021 [VERIFIED][P0] `update()` 同样先改内存再写持久层

SQLite/fsync 失败：

```text
GET API 看到新状态
重启后恢复旧状态
```

必须明确哪个才是 authority。

推荐：

```text
durable-first + immutable returned snapshot
```

---

## R43-022 [VERIFIED][P0] CAS transition 只处理 rowcount conflict，没完整处理 commit exception rollback

任何：

```text
disk I/O
SQLite locked
fsync
serialization
```

异常都必须把 memory 恢复旧版本。

---

## R43-023 [VERIFIED][P0] manifest migration 不是 version-by-version migration，而是直接改 schema_version

旧 payload：

```text
v1 → 直接标成 current
```

不等于真正迁移。

必须：

```python
MIGRATIONS = {
  1: migrate_1_to_2,
  2: migrate_2_to_3,
}
```

每步可测试、可拒绝。

---

## R43-024 [VERIFIED][P1] manifest schema version `int(True)` 会接受 bool

schema version 要 exact integer：

```text
bool reject
float reject
string 是否接受要明确
```

---

## R43-025 [VERIFIED][P1] SQLite corrupt row 被 quarantine 后仍留在 DB

每次重启：

```text
再次读坏 row
再次 quarantine
```

会形成重复告警/磁盘增长。

需要：

```text
corrupt_rows table
或
mark quarantined
或
transactional delete after evidence copy
```

---

## R43-026 [VERIFIED][P0] `get()` / `list_jobs()` 返回 store-owned mutable `JobRecord`

调用方可以：

```python
job = STORE.get(id)
job.status = ...
```

在没有 update/CAS 的情况下直接改内存 authority。

### 修复

返回：

```text
deep immutable snapshot / copy
```

更新只能：

```text
STORE.transition / STORE.update_with_version
```

---

## R43-027 [VERIFIED][P1] `manifest_path` 在 manifest 写完后才塞进 `job.artifacts`

因此：

```text
当前进程内 job.artifacts 有 manifest_path
重启 restore 后没有
```

同一 JobRecord 跨 restart 结构不一致。

---

## R43-028 [VERIFIED][P1] `deadline_monotonic` 被持久化

monotonic 时钟只对当前进程 boot/clock domain 有意义。

durable payload 应保存：

```text
deadline_wallclock UTC
timeout_duration
```

monotonic deadline 只能 ephemeral。

---

## R43-029 [VERIFIED][P1] manifest backend 没有真正跨进程 CAS

即使 production 当前想限制单 worker，domain contract 仍应写清：

```text
ManifestJobStore = single-process only
SQLiteJobStore = multi-process with lease fencing
```

而不是靠调用方猜。

---

# 4. Service Request Models / Validation

## R43-030 [VERIFIED][P0] `validate_spec()` 与 `ComputeRequest.backend` 类型契约互相冲突

文件：

```text
factor_engine/service/models.py
factor_engine/service/app.py
```

`ComputeRequest.backend`：

```python
backend: Optional[str]
```

合法：

```text
"polars"
"duckdb_sql"
"auto"
```

但 `validate_spec()` 当前循环：

```python
for key in ("data_source", "backend", "engine"):
    if key exists and not isinstance(value, dict):
        error
```

所以：

```json
{"backend":"polars"}
```

可能先被 production route `validate_spec()` 判错，根本到不了 typed model。

### Hard Gate

```text
VALIDATE_SPEC_AND_EXECUTION_MODEL_ACCEPTANCE_DIFF == 0
```

---

## R43-031 [VERIFIED][P0] `/validate-spec` 没使用声明好的 `ValidateRequest`

因此：

```text
ValidateRequest.extra="forbid"
```

没有成为真实 HTTP contract。

validate endpoint 直接接受 raw dict。

---

## R43-032 [VERIFIED][P1] `validate_spec()` 接受 `engine` object，但 ComputeRequest 没有 `engine`

出现：

```text
validate 成功
execution 422 extra forbidden
```

必须只存在一个 request schema。

---

## R43-033 [VERIFIED][P1] 所谓“per-request immutable size budget”实际上一个请求内会多次重新读 env

例如：

```text
Pydantic field validator → size_budget()
_validate_and_build_request → size_budget()
```

env 在请求中途变化可得到不同限制。

### 修复

middleware/request entry 生成唯一：

```text
RequestBudgetSnapshot
```

通过 validation context 传给所有 validator。

---

## R43-034 [VERIFIED][P1] `size_budget()` 用裸 `int(env)`，没有完整范围校验

环境可以配置：

```text
0
-1
极大整数
非法字符串
```

导致 request-time 异常或失效。

预算配置应 startup parse + strict bounds。

---

## R43-035 [VERIFIED][P1] `_RequestModel.enforce_size()` 是抽象壳，具体 request 并未统一实现

当前 size enforcement 散在：

```text
Pydantic validators
validate_spec
_submit_job
```

需要一个真正不可绕过的：

```text
RequestValidatorPipeline
```

---

## R43-036 [VERIFIED][P0] `ValidatedFactorRequest` 虽 frozen，但 `extra: dict` 仍可变且不进 digest

如果以后有人把语义字段放进 `extra`：

```text
validation digest 不变
execution semantic 却变
```

要么 deep-freeze；
要么删除；
要么明确 diagnostics-only 并禁止 execution 读取。

---

## R43-037 [VERIFIED][P1] service pre-admission cost 是 regex/string heuristic，却作为硬拒绝依据

当前 cost estimator：

```text
正则找 function call
固定 costly operator set
固定 minute multiplier
```

它可以：

```text
高估 → 错杀合法请求
低估 → 放进超大请求
```

服务硬准入至少消费 compiler cheap-pass 的 typed cost，而不是另一套字符串模型。

---

## R43-038 [VERIFIED][P1] `_read_json_body` 先 `await request.body()`，再检查 1MB cap

这只能限制“解析后处理”，不能限制 ASGI 已经分配进内存的 request body。

需要：

```text
server/proxy content-length limit
streamed receive cap
```

---

# 5. Validated Request Identity / Idempotency

## R43-039 [VERIFIED][P0] `_catalog_generations()` 多个“generation”实际上只是名称 hash

当前：

```text
market_gen = hash(market 字符串)
calendar_gen = hash(calendar 名称)
backend_evidence_gen = hash(backend 名称)
compiler_build_gen = hash(package version)
```

并没有绑定真实：

```text
MarketRegistry generation
CalendarSnapshot content
BackendEvidence artifact
Compiler source/build SHA
```

### 后果

真实语义变了，request digest 可能不变。

---

## R43-040 [VERIFIED][P0] source profile binding 使用 `json.dumps(... default=str)` / `str(config)`

semantic hash 不应依赖任意对象字符串化。

必须用 strict typed canonical serializer。

---

## R43-041 [VERIFIED][P0] HTTP `ExecutionSemanticIdentityV2` 里多项 contract hash 仍是标签代理

当前可见：

```text
field_contract_hash ≈ hash(market string)
source_dependency_hash ≈ hash(calendar string)
universe_membership_hash ≈ hash(requested symbol strings)
```

不是实际：

```text
resolved FieldContracts
resolved source dependency set
resolved dynamic universe membership snapshot
```

这会让“类型叫 semantic identity”比实际绑定强度更强。

---

## R43-042 [VERIFIED][P0] materialize request digest 严重欠绑定

当前 materialize digest 核心只绑定：

```text
config_path
write_target
policy
```

却漏：

```text
factor_id
author
frequency
description
expression
config content hash
config schema version
source snapshot
materialization semantic identity
```

### 直接后果

同一 idempotency key + 同一路径，但不同 factor_id/expression：

```text
可能被错误视为同请求
```

---

## R43-043 [VERIFIED][P0] cancellation token 在 `STORE.create()` 前创建，幂等 race 会泄漏 token 并可能重复 submit winner job

流程：

```text
run_B token 创建
STORE.create(job_B)
```

若 durable idempotency winner 实际是 `job_A`，create 返回已有 job：

```text
token_B 仍在 _JOB_TOKENS
随后代码可能继续 QUEUE.submit(job_A)
```

这是幂等性与 queue submission 没成为一个事务。

---

## R43-044 [VERIFIED][P0] `STORE.create()` 成功后 `QUEUE.submit()` 失败会留下 ghost durable job

例如：

```text
queue full
state draining
admission failure
```

此时 durable store 已有 job，但 job 并未真正入队。

需要：

```text
SUBMITTING
→ queue admission receipt
→ QUEUED
```

的 transactional outbox/state machine。

---

## R43-045 [VERIFIED][P0] queue 把 JobRecord 改成 QUEUED 后没有同步 durable update

崩溃窗口：

```text
durable: SUBMITTED
memory:  QUEUED
queue:   已入队
```

恢复后状态与实际历史不一致。

---

## R43-046 [VERIFIED][P1] 同一个 job deadline 建了两个独立 `monotonic()+timeout`

```text
job.deadline_monotonic
CancellationToken.deadline_monotonic
```

存在微小漂移且双 authority。

应该生成一次：

```text
DeadlineIdentity
```

所有层引用。

---

# 6. Job Execution / Retry / Policy Snapshot

## R43-047 [VERIFIED][P1] service 创建了 `ThreadPoolExecutor`，但 BoundedJobQueue 自己也有 worker threads

在当前 app 主路径：

```text
QUEUE worker thread 直接 run_fn
```

`EXECUTOR` 没看到承担 job execution。

如果确实无消费者：

```text
多建一套线程池
shutdown 又多等一套
```

应删除，或明确唯一 worker executor。

---

## R43-048 [VERIFIED][P0] retry endpoint 可以对 active job 重试

当前仅禁止：

```text
succeeded
cancelled
timed_out
interrupted
```

因此：

```text
queued
running
cancelling
```

并未明确被禁止。

这会产生同一业务 work 的并发副本。

---

## R43-049 [VERIFIED][P0] retry 注释说“new attempt”，但 `_submit_job` 总是 `attempt=1`

没有：

```text
parent_run_id
root_run_id
attempt=N+1
retry_reason
```

真正 attempt lineage。

---

## R43-050 [VERIFIED][P0] materialization retry 缺 durable side-effect fencing

写操作失败可能发生在：

```text
data committed
catalog 未更新
manifest flipped
response 未返回
```

不能只根据 JobStatus FAILED 就重跑。

需要：

```text
MaterializationCommitReceipt
generation_id
commit_state
```

决定：

```text
resume / finalize / no-op / rollback
```

---

## R43-051 [VERIFIED][P1] retry deterministic-error 分类仍是手写少量 error code

应该由统一 ErrorTaxonomy：

```text
VALIDATION
SEMANTIC
PIT
RESOURCE
TRANSIENT_IO
DURABILITY_UNKNOWN
SIDE_EFFECT_COMMITTED
```

驱动。

---

## R43-052 [VERIFIED][P0] policy snapshot 记录了 digest/version，但 config-path execution 仍读取当前全局 SOURCE_POLICY

提交时说：

```text
in-flight job 保留旧 policy snapshot
```

但 `_validate_config_path_sources()` execution 时消费全局 `SOURCE_POLICY`。

reload 后：

```text
提交时允许
执行时可能被新 policy 拒绝
```

或反向。

要把真正 immutable policy object/snapshot ID 绑定到 job。

---

# 7. Result Preview / Access Classification

## R43-053 [VERIFIED][P0] result preview redaction 失败时 fail-open

当前 `_redact_preview_for_access()`：

```python
except Exception:
    return None
```

`None` 意味着：

```text
原始 preview 写入 artifacts
```

安全分类基础设施故障时反而泄露。

production 必须：

```text
redaction unknown → REDACT
```

---

## R43-054 [VERIFIED][P0] config-path 计算结果 preview 没有 source_cfg，因此无法继承 source classification

`_summarize_result()` 只从：

```text
execution["data_source"]
```

取分类。

config_path 模式主要只有：

```text
execution["config_path"]
```

结果 preview 可能直接原样存。

---

## R43-055 [VERIFIED][P1] 先把 result 转成 plaintext preview，再决定是否 redact

当前：

```text
result.head(5).to_string()
→ redaction decision
```

应该顺序反过来：

```text
classify
→ allowed 才 format
```

---

## R43-056 [ARCH][P0] artifact classification 必须成为 JobArtifact 的字段，而不是 preview 特例

每个 artifact：

```python
ArtifactRef(
    kind,
    uri,
    classification,
    derived_from,
    digest,
)
```

读取 endpoint 再根据 principal 授权。

---

# 8. FactorEngine Readiness / Preflight / Release Blockers

## R43-057 [VERIFIED][P0] 当前 `/readyz` 默认逻辑实际上会固定不 ready

文件：

```text
factor_engine/service/release_blockers.py
```

默认：

```text
S20 → UNKNOWN
S23 → UNKNOWN
S24 → UNKNOWN
S25 → UNKNOWN
S26 → UNKNOWN
S27 → UNKNOWN
S28 → UNKNOWN
S29 → UNKNOWN
S30 → UNKNOWN
```

而 `_readiness_check()`：

```text
ready = FAIL==0 AND UNKNOWN==0
```

因此按当前默认检查：

```text
/readyz → 503
```

除非外部改写 checks。

### 修复

readiness 不应依赖“代码里永远 UNKNOWN 的占位符”。

要么提供真实 live evidence resolver；
要么把 closure-only checks 与 runtime readiness 分开。

---

## R43-058 [VERIFIED][P0] 多个 blocker 只是 `_check_done()` 无条件 PASS

例如：

```text
生产 endpoint downgrade
auth bypass
source allowlist
bounded request
queue
deadline
graceful shutdown
retry side effects
observability
```

这些不是“live predicate”。

### 后果

release scorecard 可以显示 PASS，但机制已经回归坏掉也不会发现。

---

## R43-059 [VERIFIED][P0] idempotency blocker 只检查字段存在

当前：

```text
JobRecord 有 idempotency_key
JobStore class 存在
→ PASS
```

它完全不能发现本轮 R43-019 的真实 SQLite race。

---

## R43-060 [VERIFIED][P0] multi-process JobStore blocker 只 inspect source 里有没有 `"sqlite3"`

这不能证明：

```text
部署实际启用 SQLite
CAS 正确
WAL
idempotency race
worker fencing
```

---

## R43-061 [VERIFIED][P1] Observability blocker 无条件 PASS

而当前 observability 本身存在：

```text
unbounded histogram
async context bleed
label cardinality
```

blocker 应测真实 invariants。

---

## R43-062 [VERIFIED][P1] production preflight 文档宣称检查更多维度，实际实现很少

`service/preflight.py` 当前主要：

```text
ambient run mode
data_access import
evidence artifact
dependency versions
disk
```

没有真正实现文档声称的完整：

```text
resource limits
data roots/hosts
catalog schema
queue/store health
policy frozen state
```

---

## R43-063 [VERIFIED][P1] dependencies preflight 对所有依赖只记录 missing，但整个 dependencies check 仍 `ok=True`

即使某个当前启用 capability 真正需要：

```text
numba/polars/duckdb/clickhouse
```

缺失也不会直接使该 check fail。

应该根据 DeploymentCapabilityProfile 判定 required set。

---

## R43-064 [VERIFIED][P0] 当前 HEAD 没有 GitHub commit status / workflow run 证据

审计时：

```text
main = 4b1577f...
combined statuses = []
workflow runs = []
```

所以不能把“代码里有很多 PASS 注释”当作当前 HEAD CI 证据。

---

# 9. Observability

## R43-065 [VERIFIED][P1] histogram 把每个观测值永久 append 到 list

文件：

```text
factor_engine/service/observability.py
```

长运行：

```text
span latency 每次 append
→ 无界内存
```

应该：

```text
HDRHistogram / fixed buckets / DDSketch / ring
```

---

## R43-066 [VERIFIED][P1] metrics snapshot 的 counter 聚合近似 O(K²)

对每个 `(name,label)`：

```text
又扫描全部 counters 求同名 total
```

label 变多后 metrics endpoint 本身变慢。

---

## R43-067 [VERIFIED][P1] `counter_value()` 无锁读取共享 dict

应与所有读写统一 lock/snapshot。

---

## R43-068 [VERIFIED][P0] HTTP async context 却使用 `threading.local()` 保存 request/run/principal

FastAPI 同一个 event-loop thread 可交错多个 request coroutine。

`threading.local` 不是 request context。

必须：

```text
contextvars.ContextVar
```

否则日志 principal/request_id 有串扰风险。

---

## R43-069 [VERIFIED][P1] log context 没有 scope/reset

`bind_log_context()` 只覆盖传入字段，不会在请求/作业结束清空旧：

```text
run_id
execution_id
principal
```

后续日志可能继承上一个 context。

---

## R43-070 [VERIFIED][P1] `trace_span(run_id=...)` 参数实际上没绑定 run_id

函数签名有 run_id，但内部：

```text
info(span.start...)
```

未使用该参数设置 context/field。

调用者以为 trace 有 run_id，实际可能没有。

---

## R43-071 [VERIFIED][P1] MetricsRegistry 没有实际 label cardinality allowlist

文档说 bounded labels，但代码：

```text
任意 labels dict
→ 直接变成 key
```

应：

```text
per metric allowed labels
allowed values bucket
overflow label
```

---

## R43-072 [VERIFIED][P0] `json_log(... default=str)` 没有统一 secret sanitizer

任何调用者把：

```text
config
URI
credential object
exception
```

塞进 fields，都可能 string 化到日志。

要统一：

```text
RedactedStructuredLogger
```

---

# 10. DataAccess ReadHandle

## R43-073 [VERIFIED][P0] CLOSED handle 仍可通过已缓存 pandas/polars 结果读取

文件：

```text
dataaccess/read/read_handle.py
```

`to_pandas()`：

```text
if _pandas_df is not None:
    return cached
```

没有先 `_ensure_open()`。

`to_polars()` 同理。

因此：

```text
handle.to_pandas()
handle.close()
handle.to_pandas()  # 仍返回
```

违背“CLOSED hard reject all reads”。

---

## R43-074 [VERIFIED][P1] terminal stats 可能重复 finalize，并把 close 前 idle 时间重复算入

materialize 后：

```text
_state=MATERIALIZED
_consume_started 仍保留
```

第一次 `_terminal_finalize()` 已加 elapsed。

以后 `close()`：

```text
又 _finalize_stats()
→ 从旧 consume_started 再加一遍
```

### 修复

增加：

```text
_stats_finalized
consume_finished_at
```

exactly once。

---

## R43-075 [VERIFIED][P0] reservation release function 在调用前就被置空

如果 release callback 抛异常：

```text
handle 已无法 retry release
reservation 仍可能活着
```

和 HostLease release 是同一类 bug。

---

## R43-076 [VERIFIED][P1] cleanup callbacks / source.close 失败全部静默吞掉

至少 production telemetry 必须记录：

```text
cleanup_failure
resource_kind
execution_id
```

必要时进入 leak reconciler。

---

## R43-077 [VERIFIED][P1] 已物化 table 的 `stream()` 不检查 deadline

stream source/lazy 每 batch检查，但 table path直接 yield batches。

如果服务语义是：

```text
deadline 覆盖 response consumption
```

这里会绕过。

---

## R43-078 [VERIFIED][P1] ReadHandle 状态机没有并发 owner/lock contract

同一个 handle 被两个线程同时：

```text
to_arrow()
stream()
close()
```

会 race。

要么：

```text
明确 ThreadUnsafe + owner thread assertion
```

要么加 lock/CAS。

---

# 11. DataAccess ScanHandle：当前最严重的 ownership 问题之一

## R43-079 [VERIFIED][P0] 派生 ScanHandle siblings 共用一个 reservation，任意 sibling close 都会提前释放

文件：

```text
dataaccess/read/scan_handle.py
```

`__getattr__` 对：

```text
lf.filter()
lf.select()
...
```

返回新 ScanHandle，但：

```text
_prepared=self._prepared
```

即共享同一个 `resource_reservation`。

场景：

```text
h1 = scan(...)
h2 = h1.filter(...)
h1.close()       # release reservation
h2.collect()     # 仍执行真实 scan
```

注释说“released 标记保证 exactly once”，但：

```text
exactly once release
!=
正确 ownership lifetime
```

必须 refcount lease 或 single-owner derived plan。

---

## R43-080 [VERIFIED][P0] `ScanHandle.close()` 后还能 `collect()`

ScanHandle 没有：

```text
OPEN/CLOSED/FAILED
```

状态。

`close()` 只释放 reservation。

之后 collect 仍执行查询，形成：

```text
unguarded execution
```

---

## R43-081 [VERIFIED][P0] 同一个 ScanHandle 可以重复 `collect()`

第一次：

```text
execute
release reservation
```

第二次：

```text
再次执行 LazyFrame
reservation 已 released
```

production 应：

```text
one-shot
或
materialize/cache result
```

---

## R43-082 [VERIFIED][P0] research 模式 snapshot revalidation 发生变化时，会把“发生变化的文件”从新 snapshot 中删掉，但 LazyFrame 仍然读取它们

`_revalidate_snapshot()`：

```text
changed → 加 changed list
未 changed → restat append
research → rebuild_snapshot_files(snapshot, restat)
```

因此新 lineage snapshot 描述的是：

```text
未变化文件集合
```

但 LazyFrame 实际仍基于原 plan 读包括 changed object 的路径。

这是严重 lineage 错配。

research 也不能“撒谎”。

---

## R43-083 [VERIFIED][P0] collect audit 使用旧 `self.snapshot`，不是 revalidation 返回的 `snapshot`

即使上一步研究模式重建了 snapshot：

```text
ReadResult.snapshot = new snapshot
audit.extra.snapshot_id = old self.snapshot.snapshot_id
```

同一次读有两套 snapshot identity。

---

## R43-084 [VERIFIED][P0] 派生 `.filter/.select/.rename` handle 继续复用原始 ReadLineage

实际查询列/过滤已经变化，但 lineage 没更新。

需要：

```text
LogicalReadTransform lineage IR
```

派生 handle 每个 transformation 更新。

---

## R43-085 [VERIFIED][P1] `collect()` 直接访问 `_store._pipeline`，但 `_store` 类型上允许 None

组合/独立构造 ScanHandle 时可能直接 AttributeError。

应该：

```text
store required for governed handle
```

在构造时验证。

---

## R43-086 [VERIFIED][P1] pipeline execute counter 被 ScanHandle 直接手工 `+=1`

这破坏：

```text
ReadPipeline 自己作为阶段计数 authority
```

而且当前 counters 还是共享对象。

应该由 pipeline 的 execute scope 自己计。

---

## R43-087 [ARCH][P0] ScanHandle 需要 LeaseRefCount

推荐：

```text
PreparedReadLease
├─ refcount
├─ state
├─ execute_once token
└─ terminal release
```

derived handles：

```text
clone logical plan
retain lease
```

直到所有 siblings terminal。

---

# 12. QueryBudget

## R43-088 [VERIFIED][P0] `QueryBudget.tighten()` 会绕过自身 strict typing

文件：

```text
dataaccess/read/query_budget.py
```

当前：

```python
int(val)
float(val)
bool(val)
```

先 coercion，再 `replace()`。

于是：

```python
tighten(max_rows=True)        → 1
tighten(max_elapsed_ms=True)  → 1.0
tighten(require_columns="false") → True
```

构造函数本来明确拒绝这些值，却被 helper 绕过。

### 修复

tighten 必须复用：

```text
_positive_int
_positive_finite_float
_strict_bool
```

---

## R43-089 [VERIFIED][P1] strict mode 错误消息仍提示可用 `require_columns=False` 放宽

但 `resolve_query_budget()` 会再与 production floor 合并，实际放不宽。

运维提示必须与真实策略一致。

---

## R43-090 [VERIFIED][P1] production floor 只设置 max_rows + require_columns

当前默认没有统一：

```text
max_scan_bytes
max_scan_objects
max_result_bytes
max_elapsed_ms
max_estimated_memory
```

如果 dataset policy 也没配，仍可做超重扫描。

需要 Deployment Budget Profile。

---

## R43-091 [VERIFIED][P0] governed Polars collect 在 `estimated_memory=None` 时仍会把所有 Arrow batches 存进 list，最后 concat 成完整 Table

它虽然逐 chunk 检查 result bytes，但：

```text
没有 estimated memory
+
production floor 没 max memory
```

仍可能 OOM。

### 修复

要么：

```text
stream terminal
```

要么从 ScanCost/plan 强制提供 peak estimate；
不能“估计可选”。

---

## R43-092 [VERIFIED][P1] Polars chunk size 固定 100,000

应消费：

```text
live resource decision
row width
result budget
```

动态 chunk。

---

## R43-093 [VERIFIED][P1] deadline 只在 chunk 之间检查，单个 slow chunk 不能主动 interrupt

如果一个 Polars batch 执行 60 秒：

```text
1s deadline
```

也要等 batch 返回。

需要 engine-native cancellation 或 process/task cancellation lane。

---

## R43-094 [VERIFIED][P1] remote request budget 仍更多是在估“对象数”，不是实际请求次数

真实远程成本包括：

```text
LIST pagination
HEAD
range GET
retry
redirect
```

应由 credential/storage client runtime counter 直接扣预算。

---

# 13. GlobalResourceGovernor 与 FE Host Lease 桥

## R43-095 [VERIFIED][P0] Host child lease request 失败会回退到本地 governor，从而可能绕过当前 JobLease

文件：

```text
dataaccess/runtime/resource_governor.py
```

逻辑：

```text
try host child lease
host_lease is None
→ local governor admission
```

但 `None` 同时可能表示：

```text
没有 active FE JobLease
或
有 JobLease，但 parent 剩余预算不足
```

第二种绝对不能 fallback。

### 修复

返回 typed：

```text
NO_PARENT
GRANTED
PARENT_REJECTED
```

仅 NO_PARENT 的 standalone DA 才走 local governor。

---

## R43-096 [VERIFIED][P0] Host-backed fast path 在 duplicate query-id check 之前直接写 `_active`

如果 host lease 成功：

```python
_active[query_id] = reservation
return
```

所以 duplicate query id 可以覆盖旧 reservation：

```text
旧 host child lease 泄漏
active map 指向新 reservation
```

---

## R43-097 [VERIFIED][P0] Host-backed fast path也绕过 max_active_queries / per_principal 等本地非内存治理

“host 是 memory authority”不等于：

```text
active-query count
principal fairness
```

也该跳过。

应把资源维度拆清楚，不能一个 if branch 绕过整个 local admission policy。

---

## R43-098 [VERIFIED][P0] release host child lease 异常被静默吞掉

再次产生 host accounting leak。

需要 reconciliation。

---

## R43-099 [VERIFIED][P1] local governor admission 每次 sum 全部 active reservations

```text
used memory
scan bytes
per principal
```

都遍历 active dict。

大并发下维护 O(1) counters。

---

## R43-100 [ARCH][P1] Host/DA governor 需要 typed delegation contract

推荐：

```python
HostLeaseBridgeResult(
    status=NO_PARENT|GRANTED|REJECTED,
    lease_ref,
    reason,
)
```

禁止用 `None` 表示两个完全不同的状态。


# 14. DataAccess HTTP Service

## R43-101 [VERIFIED][P0] `_api_budget()` 手工重建 QueryBudget，直接丢失多个新预算字段

文件：

```text
dataaccess/service/app.py
```

当前只复制：

```text
max_rows
max_result_bytes
max_elapsed_ms
max_scan_files
require_columns
require_time_range
```

漏掉：

```text
max_scan_objects
max_scan_bytes
max_remote_list_objects
max_remote_requests
max_estimated_memory
```

这恰好违反 `QueryBudget` 自己已经写明的：

```text
不要手写复制 budget
```

### 修复

只用：

```python
base.tighten(...)
base.with_overrides(...)
```

并修好 R43-088 的 strict coercion。

---

## R43-102 [VERIFIED][P0] `/v1/read_uri` 创建 handle 时在 execution scope 内，但真正 `to_arrow()` 在 scope 外

当前：

```text
with execution_scope(ctx):
    handle = store.read_uri(...)

table = handle.to_arrow()
```

如果 handle 是 lazy/stream：

```text
真正 verify/execute/nested read
```

发生在 request principal context 退出之后。

这正是 arrow-stream endpoint 已经专门修过的同类 bug，却在 read_uri 又出现。

### 修复

整个：

```text
handle lifetime + terminal materialization
```

都在 execution scope 内。

---

## R43-103 [VERIFIED][P1] Arrow IPC / Parquet 普通 endpoint 先把完整输出写进 `BytesIO`

虽然返回类型叫：

```text
StreamingResponse
```

但实际上：

```text
full table
→ full BytesIO
→ response
```

内存峰值接近：

```text
input Table + serialized buffer
```

大结果应直接流式 writer → response chunks。

---

## R43-104 [VERIFIED][P1] JSON 输出 Arrow→Pandas→JSON，形成明显多份内存

路径：

```text
Arrow
→ Pandas
→ JSON string
→ parsed Python objects
→ JSONResponse 再编码
```

尤其：

```python
json.loads(df.to_json(...))
```

是非常重的中转。

JSON 只适合严格小结果；
生产大结果应拒绝或流式 NDJSON。

---

## R43-105 [VERIFIED][P1] service `query_slots` 与 DataAccess governor 是两套并发权威

HTTP 层：

```text
threading.BoundedSemaphore(settings.max_concurrency)
```

DA 内部又：

```text
GlobalResourceGovernor
DuckDB semaphore
Host bridge
```

这会：

```text
双闸
配置漂移
真实 effective concurrency 难解释
```

HTTP 可以保留 connection/request cap，但资源 admission 应由一棵 governor tree 给 decision。

---

## R43-106 [P1][ARCH] request_id header 需要长度/字符 contract

当前客户端可以直接提供 request id。

应限制：

```text
ASCII-safe
max length
no control chars
```

避免日志/trace 放大。

---

## R43-107 [P1][ARCH] JSON/Arrow/Parquet serialization 自身也应占 Writer/Memory Lease

现在查询资源治理主要覆盖 scan/compute。

response serialization 同样可能占：

```text
数 GB buffer
CPU compression
```

应该进入 request resource tree。

---

# 15. DataAccess Generation Write / Crash Durability

## R43-108 [VERIFIED][P0] generation pointer flip 没有 fsync temp manifest 和 parent directory

文件：

```text
dataaccess/write/generation.py
```

当前：

```text
tmp.write_text(...)
os.replace(tmp, manifest)
```

缺：

```text
fsync(temp file)
fsync(root directory)
```

### 风险

进程级 rename 是原子；
但掉电/OS crash durability 不等于已证明。

如果系统把：

```text
manifest flip
```

定义为 durable commit point，就必须做真正 durability barrier。

---

## R43-109 [VERIFIED][P0] explicit empty generation metadata 同样没 fsync

delete-all 的：

```text
generation.meta.json
```

是权威空集证明。

必须与普通 generation data 一样 durable。

---

## R43-110 [VERIFIED][P1] generation 分区工具函数反复新建 `duckdb.connect(":memory:")`

包括：

```text
partition_rel_dirs
read_partition_typed
filter_table_by_partition
merge multi-key
```

写路径完全绕开现有 DataAccess shared/scoped connection management。

### 后果

大量：

```text
connection startup
catalog setup
memory pools
```

---

## R43-111 [VERIFIED][P1] `write_generation_files()` 对 P 个 partition 可能形成 O(P×N) full-table filtering

逻辑：

```text
SELECT DISTINCT partition values
for each partition:
    filter whole Arrow table
```

例如：

```text
10年×12月 = 120 partitions
```

同一整表可能被反复扫描 120 次。

### 修复

一次：

```text
partition writer / DuckDB COPY PARTITION_BY / Arrow partitioning
```

---

## R43-112 [VERIFIED][P1] `rows_in_partition()` 又调用 `filter_table_by_partition()`，可能重复扫描

如果上层已经筛过同 partition：

```text
row count
```

不要再重新 filter。

---

## R43-113 [VERIFIED][P0] SQL identifier 拼接没有统一 quote

例如：

```python
SELECT DISTINCT {', '.join(pcols)}
_t.{col}
```

合法但带：

```text
空格
保留字
特殊字符
```

的字段会破 SQL。

所有 identifiers 必须通过同一 quote helper。

---

## R43-114 [VERIFIED][P0] `read_partition_typed()` 把文件路径直接插进 SQL string

当前：

```python
read_parquet('{p}', ...)
```

路径中含 `'` 即会破坏 SQL；
内部路径也不应靠“通常没有单引号”保证正确性。

使用：

```text
bound params
或安全 literal encoder
```

---

## R43-115 [VERIFIED][P0] Hive partition value 直接 `str(v)` 拼目录，没有 path-safe encoding

如果 partition value 含：

```text
/
=
..
特殊 Unicode
```

目录结构可能改变。

要采用：

```text
canonical hive escaping
```

并 reader 对称 decode。

---

## R43-116 [VERIFIED][P0] generation validation 主要只验证“有 parquet + footer 可读 + row count”

尚未证明：

```text
schema 全分区一致
partition key 唯一
expected partitions 完整
key uniqueness
factor/version identity
content checksum
```

所以“validate passed”命名强于真实证明。

---

## R43-117 [VERIFIED][P0] `merge_tables_by_keys()` 没有拒绝 `new_table` 内部重复 upsert keys

当前做：

```text
old 中删掉所有 new keys
concat new
```

如果 new 自己有重复 key：

```text
重复记录原样进入新 generation
```

必须先：

```text
assert unique(new keys)
```

或定义 deterministic conflict policy。

---

## R43-118 [VERIFIED][P1] one-key 与 multi-key merge 使用两套不同实现

```text
one key → pyarrow.compute.is_in
multi key → DuckDB NULL-safe anti join
```

需要 edge oracle：

```text
NULL
NaN
string/int casts
```

确保 key semantics 一致。

---

## R43-119 [P1][ARCH] generation write helper 需要显式 TransactionContext

不要假设调用者一定外层持 lock。

API 应是：

```python
with GenerationTransaction(...) as txn:
    txn.write(...)
    txn.validate(...)
    txn.commit()
```

`flip_generation_pointer()` 不应作为随便可直接调用的 public commit primitive。

---

## R43-120 [P1][ARCH] generation orphan GC 需要正式策略

每次 COW/失败可能留下：

```text
old generations
aborted candidates
tmp files
quarantine
```

需要：

```text
manifest reachability
reader grace period
lease/fencing
GC journal
```

---

## R43-121 [P1][ARCH] old-generation GC 必须考虑 in-flight readers

manifest 已 flip 后：

```text
旧 reader
```

可能仍拿旧 generation。

不能立即 delete。

需要：

```text
generation ref/TTL/reader epoch
```

---

## R43-122 [P1][ARCH] generation commit 应输出 `DurableCommitReceipt`

字段：

```text
generation_id
manifest_version
schema_digest
row_count
partition_count
content_digest
fsync_complete
catalog_commit
```

Job retry 根据 receipt，而不是猜。

---

# 16. Mutation Lock

## R43-123 [VERIFIED][P1] corrupt lock payload 会变成永久无法自动回收的 lock

`_read_payload(path) is None` 时：

```text
_can_break_lock → False
```

之后只能每次 timeout。

需要：

```text
corrupt-lock quarantine
inode metadata
safe stale rule
operator recovery command
```

不能一个 torn JSON 锁永远砖化 dataset。

---

## R43-124 [VERIFIED][P1] cross-host owner crash 只能依赖 lease/hard-break，恢复时间非常长

默认：

```text
lease ~ 1h
hard_break ~ 1h
```

跨 host 又不能 os.kill 检测。

实际恢复可能很慢。

共享文件系统更适合：

```text
distributed lock service
或
short lease + heartbeat + fencing token
```

---

## R43-125 [P0][ARCH] File lock 还缺“写入 fencing token 进入 generation”

即使 stale lock 被 break：

```text
旧 writer
```

可能还在后台继续写自己的 candidate。

最终 publish 必须验证：

```text
transaction fencing token 仍是 current
```

否则 stale writer 不能 commit。

---

## R43-126 [P1][ARCH] release 后保留 released lock 文件会让每个下一 writer 先走异常路径

这是安全 tradeoff，但可以把：

```text
owner-safe cleanup
```

交给独立 janitor，减少 steady-state FileExistsError/unlink 开销。

---

# 17. DataAccess StartupCertificate

## R43-127 [VERIFIED][P0] research startup gate 有 problems 仍生成 `passed=True` certificate

文件：

```text
dataaccess/runtime/startup_gate.py
```

research：

```text
收集 problems
logger.warning
return build_startup_certificate(True, problems)
```

这本身可以理解为“research 可用”，但 certificate 字段只有：

```text
passed=True
```

没有 mode。

---

## R43-128 [VERIFIED][P0] `require_startup_certificate()` 复用已有 certificate 时不绑定 production/research mode

逻辑：

```text
已有 cert
passed
未过期
→ 直接 return
```

没有：

```text
cert.run_mode == requested production
```

### 严重场景

```text
先 research require certificate
→ problems 非空但 passed=True
后来同 store 进入 production require
→ 直接复用 research cert
```

这是 production startup gate bypass。

---

## R43-129 [VERIFIED][P0] StartupCertificate `evidence_hash` 在成功时基本是常量

hash 来源：

```text
gate_name + sorted(problems)
```

全部 PASS 时：

```text
problems=[]
```

所以不同：

```text
registry
calendar
policy
credential
source snapshot provider
build
```

都可能得到同一个 evidence hash。

这不是 evidence identity。

---

## R43-130 [VERIFIED][P0] certificate 不绑定 build/config/registry/calendar/source generations

需要：

```python
StartupEvidenceIdentity(
    build_sha,
    registry_digest,
    policy_digest,
    credential_scope,
    calendar_digests,
    source_provider_generation,
    resource_profile,
)
```

---

## R43-131 [VERIFIED][P0] 24h TTL 内配置发生变化，旧 certificate 仍可复用

例如：

```text
calendar 被替换
credential scope 被改
registry reload
policy reload
```

证书应按 dependencies 自动 invalidation，不只按时间。

---

## R43-132 [VERIFIED][P1] single-worker startup check 只看特定 env 变量

真实 deployment 可能：

```text
gunicorn -w 4
k8s replicas 4
```

但 `DATA_ACCESS_WORKERS` 未设。

仅 env heuristic 无法证明 single-worker contract。

---

## R43-133 [P1][ARCH] startup certificate 应是 capability-specific

不是一个：

```text
production_startup passed
```

覆盖一切。

应分别证明：

```text
local parquet read
COS read
PIT read
factor lake read
matrix read
clickhouse
```

启用哪个 capability，要求哪个 certificate。

---

# 18. FactorEngine execution scope / universe

## R43-134 [VERIFIED][P0] named universe 的实际应用仍没有成为通用执行 contract

文件：

```text
factor_engine/runtime/engine.py
```

当前 universe gate 重点只在：

```text
plan 含 cross-sectional operator
```

时阻止 scoped factor 在 full source 上计算。

如果是纯 TS factor：

```text
Factor.universe="CSI300"
data source = 全A
```

`assert_execution_scope_contract()` 因为没有 CS op 会直接 return。

最终输出可能仍包含全 A。

### 正确语义

universe 不只是 CS rank 的计算域；
也是：

```text
factor output domain
```

所有因子都应应用 resolved membership。

---

## R43-135 [VERIFIED][P0] 任意 non-empty `instrument_filter` 被当成“source 已 scoped”，没有证明它等于 factor universe

场景：

```text
factor universe = CSI300
source instrument_filter = 随便 10 只股票
source.universe 未声明
```

当前 `_data_source_scoped()` 可以返回 True。

对 CS operator：

```text
gate 放行
```

但计算 domain 仍错。

### 修复

比较：

```text
ResolvedUniverseSnapshot.membership_digest
==
DataSourceFilter.membership_digest
```

---

## R43-136 [VERIFIED][P1] cross-sectional detector 对 registry lookup 异常 fail-open

`_plan_has_cross_sectional_ops()`：

```python
try registry...
except Exception:
    pass
```

如果新 operator 的 category 识别失败且名称又没命中前缀/静态集合：

```text
CS gate 可能漏检
```

production 未知 category 应 conservative。

---

## R43-137 [VERIFIED][P1] cross-sectional 语义仍维护静态名称集合

这和 R42 的 compiler contract方向一致，但这里是明确 execution-gate bypass surface。

应该直接消费：

```text
OperatorAxisContract
requires_full_cross_section
```

---

## R43-138 [VERIFIED][P1] `_field_catalog_version()` 是进程永久 lru_cache(maxsize=1)

如果未来支持 field registry hot reload：

```text
source scope hash 仍使用旧 field catalog version
```

需要 registry generation 驱动 invalidate。

---

# 19. Packaging / Version Compatibility

## R43-139 [VERIFIED][P0] FactorEngine 声明 `data-access>=0.2.0`，但当前代码明显依赖远晚于 0.2 的接口/治理机制

文件：

```text
factor_engine/pyproject.toml
dataaccess/pyproject.toml
```

当前 DataAccess 自身版本：

```text
0.10.2
```

FactorEngine code 使用大量：

```text
PreparedRead
ReadPipeline
ScanCost
host lease bridge
RuntimeModeIdentity
...
```

但 resolver 被允许装：

```text
data-access 0.2.x
```

这会形成“安装成功、运行时 crash”。

### 修复

根据真实 compatibility CI 设：

```text
data-access >= tested_min,<breaking_upper
```

---

## R43-140 [VERIFIED][P1] 对 pre-1.0 DataAccess 没有 upper compatibility bound

0.x 阶段 minor 通常可包含 breaking change。

应该有：

```text
supported version matrix
```

而不是无限 `>=`.

---

## R43-141 [P1][ARCH] FE / DA 应发布 CompatibilityManifest

例如：

```json
{
  "factor_engine": "0.4.x",
  "data_access": ">=0.10.2,<0.11",
  "duckdb": "...",
  "polars": "...",
  "python": [...]
}
```

startup 校验。

---

## R43-142 [P1][ARCH] optional backend 依赖要由 capability handshake 校验

安装：

```text
factor-engine
```

不等于：

```text
polars backend ready
duckdb backend ready
service ready
```

暴露：

```text
RuntimeCapabilityProfile
```

---

# 20. CI / Evidence / Release

## R43-143 [VERIFIED][P0] 当前 HEAD 没有 connector 可见的 CI workflow run/status

因此执行 Coding AI 时必须亲自跑：

```text
FE unit
DA unit
wheel install
cross-package integration
production hard gates
```

并把 raw evidence 与 final SHA 绑定。

---

## R43-144 [P0][ARCH] Release blocker 与 CI evidence 必须使用同一 EvidenceStore

不能：

```text
CI 一套 JSON
readyz 一套 inspect
docs 一套 PASS
```

统一：

```text
EvidenceArtifact(
  build_sha,
  suite,
  result,
  environment,
  expires,
  dependencies
)
```

---

## R43-145 [P1][ARCH] evidence 要区分 code evidence 与 deployment evidence

代码通过：

```text
operator parity
```

不等于机器部署通过：

```text
disk
credential
calendar
source access
resource limits
```

两个 certificate 分开。

---

## R43-146 [P1][ARCH] readiness 不应要求“历史 closure drill evidence”实时存在

例如：

```text
rollback drill
security fuzz
nightly perf
```

应决定：

```text
release gate
```

而不是每个 process runtime `/readyz` 的永久 UNKNOWN。

---

## R43-147 [P1][ARCH] readiness 应只判断“当前实例能否安全接新请求”

包括：

```text
store
queue
governor
disk watermark
policy
source critical health
worker state
```

---

## R43-148 [P1][ARCH] liveness/readiness/degraded 分开

```text
live = process/event loop活
ready = 可接请求
degraded = 可接但部分 capability unavailable
```

不要只有 ready/not-ready。

---

# 21. 还有一批必须继续处理的跨模块架构问题

## R43-149 [ARCH][P0] Job submission 需要 Transactional Outbox

推荐流程：

```text
BEGIN durable transaction
  create job SUBMITTING
  create queue-outbox record
COMMIT

dispatcher:
  claim outbox
  enqueue
  mark QUEUED
```

即使进程在任意点 crash，也能恢复。

---

## R43-150 [ARCH][P0] Queue 中 item 应只有 `run_id + attempt_id`，不要携带 live mutable JobRecord

worker dequeue 后：

```text
从 JobStore CAS claim
```

拿 immutable execution snapshot。

这样不会 queue object 与 store object 分裂。

---

## R43-151 [ARCH][P0] Job 状态机必须 machine-enforced

例如：

```text
SUBMITTED
→ QUEUED
→ CLAIMED
→ RUNNING
→ FINALIZING
→ SUCCEEDED

任何阶段
→ CANCELLING
→ CANCELLED
```

不允许直接：

```text
arbitrary job.status = ...
```

---

## R43-152 [ARCH][P0] worker claim 使用 lease/fencing token

```text
claim_epoch
lease_until
worker_id
```

旧 worker 失去 lease 后：

```text
禁止写最终状态
禁止 publish side effect
```

---

## R43-153 [ARCH][P0] materialize publish 与 job completion 需要 commit handshake

不能：

```text
writer commit
job store update
```

两个互不相干。

至少记录：

```text
generation commit receipt
→ job FINALIZING
→ job SUCCEEDED
```

恢复时可重放 finalize。

---

## R43-154 [ARCH][P0] Cancellation 也需要 durable intent

当前 token 主要在内存。

持久化：

```text
cancel_requested_at
cancel_generation
```

worker 每 stage/fence 读取或接受 push signal。

---

## R43-155 [ARCH][P1] Retry 应以 root operation 为中心

```text
operation_id
attempt_id
```

不是每 retry 生成完全无关系的新 job。

---

## R43-156 [ARCH][P1] ErrorTaxonomy 单一事实源

Queue、scheduler、writer、service retry 不应分别：

```text
字符串 contains
手写 tuple
默认 transient
```

统一 typed exception categories。

---

## R43-157 [ARCH][P1] ResourceLease 也应有状态机

```text
REQUESTED
GRANTED
RELEASING
RELEASED
LEAK_SUSPECTED
```

release failure 进入 reconciler，而不是 `except: pass`。

---

## R43-158 [ARCH][P1] 所有资源 owner 提供 `close receipt`

包括：

```text
ReadHandle
ScanHandle
JobLease
DAScanLease
DuckDB connection
Writer
Spill
```

长跑 soak 可核对：

```text
created == released
```

---

## R43-159 [ARCH][P1] Handle API 区分 `PlanHandle` 与 `ResultHandle`

当前 ScanHandle 同时：

```text
可派生 lazy plan
又拥有 execution reservation
```

导致 sibling ownership 混乱。

更合理：

```text
ScanPlan     # immutable, no live execution lease
execute()
→ ScanResultHandle  # owns lease/stream
```

---

## R43-160 [ARCH][P1] PreparedRead 也不要在 prepare 阶段长期持 execution memory reservation

如果用户：

```text
prepare
几分钟后才 execute
```

长期占 lease。

可以两阶段：

```text
ResolutionLease
→ executable immutable plan
→ ExecuteLease at terminal execution
```

---

# 22. 恢复/崩溃点审计必须补齐

## R43-161 [ARCH][P0] JobStore crash matrix

逐点 kill：

```text
before create
after memory insert
after manifest temp write
after fsync
after rename
after SQLite insert
before enqueue
after enqueue
before QUEUED durable update
```

重启后不得 ghost/duplicate。

---

## R43-162 [ARCH][P0] Generation crash matrix

逐点 kill：

```text
candidate mkdir
partial parquet
footer complete
validation complete
manifest temp
manifest fsync
rename
directory fsync
catalog update
job finalization
```

每点恢复定义清楚。

---

## R43-163 [ARCH][P0] Retry-after-commit test

强制：

```text
generation 已 durable
HTTP/job final status 写失败
```

retry 必须：

```text
识别已有 receipt
finalize/no-op
```

绝不能重复生成逻辑副作用。

---

## R43-164 [ARCH][P0] Cancel-during-commit test

取消发生在：

```text
manifest flip 前
manifest flip 后
catalog commit 中
```

最终状态必须解释：

```text
CANCELLED_BEFORE_COMMIT
SUCCEEDED_COMMIT_WON
COMMITTED_BUT_FINALIZATION_FAILED
```

不能一律 CANCELLED。

---

## R43-165 [ARCH][P0] stale worker fencing test

worker A 停 heartbeat；
worker B takeover；
A 恢复继续运行。

A 的：

```text
job update
artifact publish
lease release
```

必须全部被 fencing token 拒绝。

---

# 23. API Contract Consistency

## R43-166 [ARCH][P0] 建一个“同请求跨入口一致性矩阵”

同 factor：

```text
Python API
YAML
CLI
research HTTP
production HTTP
batch
materialize
incremental
```

对比：

```text
canonical formula
market
calendar
universe
source
PIT
backend
precision
identity
result
```

---

## R43-167 [ARCH][P0] validate endpoint 与 execute endpoint 必须复用同一 Pydantic/domain validator

不再有手写：

```text
validate_spec(dict)
```

另一套。

---

## R43-168 [ARCH][P1] OpenAPI schema 要自动 contract-test

随机合法 payload：

```text
OpenAPI accepts
→ endpoint accepts
```

随机非法：

```text
OpenAPI rejects
→ endpoint rejects
```

---

## R43-169 [ARCH][P1] Python domain object validation 与 HTTP validation 对齐

例如：

```text
Factor.name
backend
market
frequency
universe
```

不能 service 拒绝、Python API放行，或反之。

---

# 24. Observability / Operations 下一步

## R43-170 [ARCH][P1] 用 bounded histogram，不保存 raw samples

每个指标：

```text
count
sum
buckets/quantiles sketch
```

---

## R43-171 [ARCH][P1] 建 request-scoped TraceContext

```text
request_id
run_id
attempt_id
execution_id
principal_id
tenant
```

ContextVar 自动 scope/reset。

---

## R43-172 [ARCH][P0] Log redaction 应在 logger sink 最后一层再做一次

即使上游忘了 sanitize：

```text
password/token/signed URI
```

也不能写出去。

---

## R43-173 [ARCH][P1] 资源泄漏 metrics

至少：

```text
active_job_leases
active_child_leases
active_da_reservations
open_read_handles
open_scan_handles
open_streams
open_duckdb_connections
spill_bytes
orphan_generations
```

---

## R43-174 [ARCH][P1] 状态 divergence metrics

例如：

```text
queue_without_store
store_queued_not_in_queue
token_without_job
running_without_lease
lease_without_running_job
```

这些比普通 latency 更能发现系统性 bug。

---

# 25. Deployment / Build / Compatibility

## R43-175 [ARCH][P0] wheel smoke 必须从“源码目录外”运行

否则 source tree import 会掩盖：

```text
缺包
缺 package-data
错误 import layout
```

测试：

```text
build wheel
new venv
cd /tmp
pip install wheel
import
run minimal FE+DA
```

---

## R43-176 [ARCH][P0] FE+DA integration wheel matrix

至少：

```text
Python supported minors
minimum supported DA
current DA
Polars min/current
DuckDB min/current
```

---

## R43-177 [ARCH][P1] pre-1.0 dependency upgrade 做 compatibility gate

DataAccess：

```text
0.10 → 0.11
```

升级前自动跑 FactorEngine integration corpus。

---

## R43-178 [ARCH][P1] Build SHA 必须来自构建系统，不依赖运行目录 Git 可见性

wheel/container：

```text
package metadata
```

内置：

```text
build sha
build time
dirty flag
dependency lock hash
```

---

## R43-179 [ARCH][P1] production evidence 绑定 built wheel hash

不是只绑定：

```text
source commit
```

最终部署 artifact 应有：

```text
wheel/container digest
```

---

## R43-180 [ARCH][P1] 配置文件也需要 content digest

Job request 如果用：

```text
config_path
```

不能只记路径。

提交时：

```text
读取/授权/解析
→ canonical config digest
→ job bind
```

执行时要么使用 frozen content，要么验证 unchanged。


# 26. Config-path jobs：提交时和执行时不是同一个输入

## R43-181 [VERIFIED][P0] config-path job 没有冻结 config 内容，存在 validate→execute TOCTOU

service job 主要持久化：

```text
config_path
```

worker 真正执行时才：

```text
load_config(config_path)
```

提交到执行之间文件可以被修改。

### 后果

```text
request digest 绑定的是提交时参数
实际运行的是执行时文件
```

尤其 materialize digest 还只绑定路径。

### 修复

提交时：

```text
authorize path
read bytes
parse
canonicalize
content digest
store frozen config artifact
```

worker 执行 frozen content 或验证 path content hash 不变。

---

## R43-182 [VERIFIED][P0] config-path source policy 同样存在 policy/file 双 TOCTOU

执行期再：

```text
load_config
SOURCE_POLICY.validate_source
```

因此：

```text
file content
policy content
```

都可能和提交时不同。

Job 必须绑定：

```text
config_digest
policy_digest
source_contract_digest
```

---

## R43-183 [VERIFIED][P0] raw traceback 被直接写入 job artifacts

`_job_wrapper`：

```python
job.artifacts["traceback"] = traceback.format_exc(limit=15)
```

这没有经过外部 error sanitizer。

traceback/exception message 可能含：

```text
local paths
dataset roots
URI
SQL
config values
```

如果 artifact endpoint 对普通 job owner 可见，就会泄内部细节。

### 修复

区分：

```text
public_error
operator_debug_trace
```

后者仅 ADMIN/internal observability，且 sink 级 redaction。

---

## R43-184 [VERIFIED][P0] side effect 已完成但最终 JobStore update 失败时，没有 durable “finalization uncertain” 状态

例如：

```text
factor materialization 已 commit
job.status = SUCCEEDED in memory
STORE.update() 因磁盘失败
```

worker finally 仍会 drop cancellation token。

重启后 durable job 可能仍是 RUNNING/旧状态。

需要：

```text
COMMIT_RECEIPT + FINALIZATION_PENDING
```

恢复器负责补 final status。

---

## R43-185 [VERIFIED][P1] final persistence failure 不进入统一 ServiceError/JobStatus taxonomy

最后 `STORE.update(job)` 位于主要 execution exception handling 之后。

需要：

```text
ExecutionFailed
CommitFailed
FinalizationPersistenceFailed
```

分别记录。

---

## R43-186 [ARCH][P0] ConfigArtifact 应是一等 job input

```python
ConfigArtifact(
    canonical_payload,
    content_digest,
    source_path,
    schema_version,
    policy_digest,
)
```

job 执行只消费它。

---

# 27. Readiness 本身还需要重新定义

## R43-187 [VERIFIED][P1] FactorEngine readiness 只判断 `disk_free_mb >= 0`

也就是：

```text
只要 disk_usage 调用成功
即使只剩 1MB
也不因为 disk watermark 失败
```

production 应使用：

```text
absolute free
percentage free
writer reserve
spill reserve
```

---

## R43-188 [ARCH][P1] Queue health 应进入 readiness

例如：

```text
all workers dead
queue state stopped
pending age P99 exploding
fatal writer
```

process 还活着也不应 ready。

---

## R43-189 [ARCH][P1] HostResourceCoordinator reconciliation 偏差进入 readiness/degraded

若：

```text
lease accounting drift
```

达到阈值，应停止接新生产 job。

---

## R43-190 [ARCH][P1] DataAccess readiness 也应验证 actual API budget profile

不能启动时正常，但：

```text
HTTP _api_budget 丢了字段
```

仍 ready。

可运行一个 synthetic request contract probe。

---

# 28. DataAccess Startup Gate 还有语义设计问题

## R43-191 [VERIFIED][P1] `_source_snapshot_provider_available()` 只 smoke 固定名字

当前主要：

```text
factor_matrix
factor_lake
```

真正 production job 可能依赖其它 critical datasets。

应由：

```text
deployment required-dataset set
```

驱动。

---

## R43-192 [ARCH][P1] Calendar certificate 绑定 requested coverage，不应只看“最近120天”

历史回测/研究生产任务真正需要：

```text
2016–2025
```

最近 calendar 健康不证明历史 coverage 完整。

Job compile 应给：

```text
required_calendar_interval
```

再验证。

---

## R43-193 [ARCH][P1] Source snapshot provider startup probe 不应替代 request-level exact snapshot proof

startup 只证明：

```text
provider 可达
```

每个 request 仍必须证明：

```text
自己 exact object set
```

二者 evidence 类型分开。

---

# 29. DataAccess service 输出预算仍不覆盖 serialization

## R43-194 [VERIFIED][P1] `max_result_bytes` 约束 Arrow table，不等于 JSON serialized bytes

JSON 可能比 Arrow 大很多。

应有：

```text
max_wire_bytes
```

或 JSON 更小独立上限。

---

## R43-195 [VERIFIED][P1] Parquet/Arrow IPC compression/serialization 内存没有纳入 query slot/resource lease

需要 WriterLease/SerializationLease。

---

## R43-196 [ARCH][P1] Client disconnect 应主动取消底层 serialization/read

stream endpoint 已 close generator，但普通先物化 endpoint：

```text
client disconnect
```

发生在完整读取/序列化之后时，仍浪费资源。

异步 cancellation token 应贯穿。

---

# 30. Security / authorization 与 derived data

## R43-197 [ARCH][P0] derived factor classification 不应只看 source config tag

实际 factor 可能来自：

```text
source A public
source B premium
durable factor C restricted
model score D
```

需要 compiler lineage 自动：

```text
max/meet classification
```

写入 FactorSemanticIdentity/ArtifactRef。

---

## R43-198 [ARCH][P0] classification unknown 必须是 fail-closed 状态，不等于 public

统一：

```text
PUBLIC
BASIC
FUNDAMENTAL
PREMIUM
RESTRICTED
UNKNOWN
```

UNKNOWN 不能 rank=0。

---

## R43-199 [ARCH][P1] `request_metadata.roles` 不应成为 preview 授权 authority

角色/principal 应来自：

```text
authenticated immutable PrincipalSnapshot
```

而不是普通 request metadata 字典。

---

## R43-200 [ARCH][P1] JobRecord 应持久化 principal policy snapshot ID，而不是只存角色文本

便于 artifact read/replay audit。

---

# 31. Generation 数据布局进一步需要处理的正确性细节

## R43-201 [VERIFIED][P1] generation helper 的 parquet writer 没有统一 WriterPolicy

当前直接：

```python
pq.write_table(...)
```

没有统一：

```text
compression
row_group_size
statistics
dictionary
page index
sorting metadata
```

DataAccess 的读取成本优化会被 writer layout 随调用点漂移抵消。

---

## R43-202 [VERIFIED][P1] COW hardlink fallback 到 `copy2` 时可能突然复制整段历史

跨 filesystem/hardlink 不支持时：

```text
增量写
→ 大量历史完整复制
```

必须 preflight：

```text
filesystem capability
```

并让 cost model 知道 COW 是：

```text
LINK
REFLINK
COPY
```

---

## R43-203 [ARCH][P1] 优先尝试 reflink/content-addressed object ref

比普通硬链接更适合部分 filesystem/container。

---

## R43-204 [ARCH][P1] Generation inventory 应独立 manifest 化

不要每次 `rglob()` 整个 generation 才知道 objects。

记录：

```text
partition
path
rows
bytes
schema hash
checksum
```

---

## R43-205 [ARCH][P0] Generation reader 应验证 manifest 指针和 inventory 是同一 generation transaction

不是：

```text
pointer 指 gid
然后自己 glob 任意文件
```

每代有 immutable generation manifest。

---

# 32. Resource accounting 还缺真实值校准

## R43-206 [ARCH][P1] JobLease 需要 predicted-vs-actual peak calibration

当前多数 lease 是 estimate。

每 job 完成：

```text
reserved memory
actual peak family RSS
scan bytes
spill bytes
```

回写 calibration。

---

## R43-207 [ARCH][P1] Reservation 不能只在“正常 terminal”统计 release

release exception 也要：

```text
LEAK_SUSPECTED
```

进入后台 reconciliation。

---

## R43-208 [ARCH][P1] OOM 后要回写 admission underprediction，不只缩 shape

同 dependency signature 下以后直接使用更保守 P95。

---

## R43-209 [ARCH][P1] resource limits 需要 tenant/principal fairness

不仅：

```text
max active per principal
```

还要长期吞吐公平：

```text
DRF / weighted fair share
```

避免一个大 mining campaign长期占满。

---

# 33. JobStore / SQLite 还需要真正的数据库事务设计

## R43-210 [ARCH][P0] SQLite 开启 WAL + busy_timeout + explicit transaction

不能依赖默认 connection 行为。

测试：

```text
concurrent create/update/CAS
process crash
database locked
```

---

## R43-211 [ARCH][P0] Job row 增加 monotonic `version`

每个更新：

```sql
UPDATE ... WHERE run_id=? AND version=?
```

成功：

```text
version += 1
```

---

## R43-212 [ARCH][P0] idempotency winner 在数据库层返回

使用：

```text
INSERT ... ON CONFLICT ...
RETURNING
```

或事务内 select，唯一决定 durable winner。

---

## R43-213 [ARCH][P1] manifests 只作为审计/灾备镜像，不与 SQLite 同时都当 authority

否则：

```text
SQLite committed
manifest fail
```

谁是准？

明确：

```text
SQLite authority
manifest derived audit
```

或者反过来。

---

## R43-214 [ARCH][P1] JobStore startup 要做 SQLite↔manifest reconciliation

输出：

```text
missing manifest
manifest newer
SQLite newer
checksum mismatch
orphan manifest
```

不能静默选其中一个。

---

# 34. Direct API / Batch API consistency

## R43-215 [ARCH][P0] `run`, `run_many`, HTTP inline, config path 必须构造同一个 `ExecutionRequest`

目前每个入口自己：

```text
parse options
source build
market/calendar
DQ/PIT defaults
```

R42 提到 unified execution API，这里要求做成 hard gate：

```text
ENTRYPOINT_SEMANTIC_REQUEST_DIFF == 0
```

---

## R43-216 [ARCH][P0] universe resolution 在 compile 前完成

不要只把 `"CSI300"` 当字符串 hint。

必须 resolve：

```text
UniverseSnapshot
membership digest
knowledge policy
```

进入 source filtering 和 identity。

---

## R43-217 [ARCH][P0] TS-only factor 也必须执行 output-universe projection

CS 因子同时要求：

```text
computation universe
```

TS 因子至少要求：

```text
output universe
```

两者显式区分。

---

## R43-218 [ARCH][P1] 数据源声明 `instrument_filter` 时构造 FilterIdentity

包含：

```text
sorted security IDs / bitmap digest
source
knowledge date
```

不能只用“非空”做安全判断。

---

# 35. Failure taxonomy / retry 统一化

## R43-219 [ARCH][P0] 禁止通过 exception message substring 作为 production 主要分类器

字符串仅 legacy fallback。

正式异常：

```python
ErrorCategory
Retryability
SideEffectState
ResourceImpact
```

---

## R43-220 [ARCH][P0] Unknown 默认不 retry

production：

```text
UNKNOWN → fail + evidence
```

研究模式可诊断性 retry 一次，但不能混成 production policy。

---

## R43-221 [ARCH][P0] ENOSPC / EACCES / EROFS 等 OSError 明确 permanent

Timeout/connection reset 才 transient。

---

## R43-222 [ARCH][P0] SideEffectUnknown 单独类别

如果异常发生在 commit boundary：

```text
不知道写成功没有
```

不能按 transient 自动重跑。

先 reconcile commit receipt。

---

# 36. Evidence 需要防“自证”

## R43-223 [ARCH][P0] blocker 不能用“代码里存在某类/字段”证明行为正确

例如：

```text
JobRecord 有 idempotency_key
```

不能证明 idempotency。

必须执行行为测试。

---

## R43-224 [ARCH][P0] readiness probe 不直接 `inspect.getsource()` 证明机制

源码 inspection 只能辅助诊断，不是 runtime behavior evidence。

---

## R43-225 [ARCH][P1] CI evidence 要有负向测试

例如 idempotency：

```text
100 concurrent same key
→ exactly one durable run_id
```

而不只 happy path。

---

## R43-226 [ARCH][P1] release evidence 绑定 test binary/wheel

防：

```text
源码测试 A
部署 wheel B
```

---

# 37. Soak / Chaos 场景

## R43-227 [ARCH][P1] 24h service soak

持续：

```text
submit/cancel/retry/read/materialize
```

检查：

```text
RSS slope
thread count
FD count
histogram memory
lease count
jobstore consistency
```

---

## R43-228 [ARCH][P1] Queue saturation chaos

同时：

```text
queue full
drain
cancel
retry
shutdown
```

验证无 orphan。

---

## R43-229 [ARCH][P1] Disk pressure chaos

模拟：

```text
ENOSPC during parquet
ENOSPC during manifest
ENOSPC during JobStore
```

不同 commit state 正确恢复。

---

## R43-230 [ARCH][P1] SQLite lock/IO chaos

注入：

```text
database locked
commit failure
fsync failure
```

确保 memory 与 durable 不分裂。


# 38. DataAccess HTTP Request Model 还有直接可绕过的限制

## R43-231 [VERIFIED][P1] `MAX_TIME_RANGE_YEARS=50` 定义了但 `ReadRequest` 没真正执行该限制

文件：

```text
dataaccess/service/models.py
```

当前 time_range validator 只检查：

```text
长度=2
不是 [null,null]
```

没有：

```text
ISO 日期是否合法
start <= end
跨度 <= 50 年
```

常量存在不等于能力存在。

---

## R43-232 [VERIFIED][P1] `ReadRequest.instrument_filter` 只限制列表数量，不校验元素

缺：

```text
非空
唯一
最大代码长度
合法 instrument id grammar
```

重复 5000 次同 symbol 仍通过。

---

## R43-233 [VERIFIED][P1] `params` 没有 key 数/嵌套深度/value 大小 contract

ParametricDataset 最终会验证已知参数，但 HTTP 层仍可能先接收/解析很大的 nested payload。

需要：

```text
max keys
max depth
max string bytes
```

---

## R43-234 [VERIFIED][P0] `ReadURIRequest` 没复用 ReadRequest 的 columns/instrument/time_range validators

它重新声明：

```text
columns
time_range
instrument_filter
```

但没有对应 validator。

所以 privileged URI API 绕过：

```text
MAX_COLUMN_COUNT
MAX_COLUMN_LEN
MAX_INSTRUMENT_FILTER_CARDINALITY
time_range shape/semantic checks
```

---

## R43-235 [VERIFIED][P1] `ReadURIRequest.limit` 没有 ge=1 / 上限

应：

```text
1 <= limit <= service max rows
```

并与 QueryBudget tighten。

---

## R43-236 [VERIFIED][P1] `ReadURIRequest.format` 是任意 str，不是 Literal/enum

虽然 runtime 可能再验证，但 HTTP contract 应与 capability list一致。

---

## R43-237 [VERIFIED][P1] `ReadURIRequest.filters` 是任意 dict，无 cardinality/depth/field count上限

这是 privileged endpoint，也更需要 bounded predicate complexity。

---

## R43-238 [VERIFIED][P1] `FactorReadRequest.factor_ids` 不拒绝重复 IDs

```text
500 个位置
```

可以全是同一 factor。

会放大 SQL/merge 工作或产生重复输出语义。

---

## R43-239 [VERIFIED][P1] `FactorReadRequest` 的单个 factor_id 没统一走 FactorId domain validator

应该复用：

```text
security.factor_id
```

而不是只判 non-empty list/count。

---

## R43-240 [VERIFIED][P1] `FactorReadRequest.limit` 没有正数/上限约束

同样应归一到 QueryBudget。

---

## R43-241 [VERIFIED][P1] DataAccess service 没看到统一 request-body byte cap middleware

FactorEngine service 有自己的 `_read_json_body`，DataAccess FastAPI endpoints主要直接由 Pydantic 解析。

需要在：

```text
reverse proxy / ASGI middleware
```

先限制 body bytes。

---

## R43-242 [VERIFIED][P1] `/v1/metrics` 只要求任意 API key，没有独立 observability permission

这会把全局：

```text
query counts
slow counts
elapsed
```

暴露给普通数据读取 principal。

如果多 tenant，应该：

```text
metrics:read
```

或仅内网管理面。

---

# 39. FactorEngine request digest 还没有绑定整个 HTTP request

## R43-243 [VERIFIED][P0] compute request digest 没绑定 `name`

`ValidatedFactorRequest` 绑定 formula/context，但没有 factor display/ID name。

同 idempotency key：

```text
request A name=alpha_A
request B name=alpha_B
```

可能被视为同 validated digest。

即使数值相同，job/artifact identity 不同，也应判 request mismatch。

---

## R43-244 [VERIFIED][P0] compute request digest 没绑定 `timeout_seconds`

同 key 重试：

```text
timeout=10s
timeout=3600s
```

当前可能仍被看成同 request。

幂等 digest 应绑定所有会改变 observable execution semantics 的字段。

---

## R43-245 [VERIFIED][P1] `max_cost_per_job` 等 admission policy request 字段也应明确是否进入 idempotency identity

原则：

```text
如果字段会改变请求是否被接受/如何执行
→ 要么进 digest
→ 要么明确声明 non-semantic transport hint
```

不能隐式。

---

## R43-246 [ARCH][P0] 建统一 `CanonicalRequestIdentity`

来自 typed request model 的：

```text
全部 semantic/execution fields
```

自动 canonical serialize。

不再手工挑一部分塞进 ValidatedFactorRequest。

---

# 40. Job result / artifact 还要限制大小

## R43-247 [VERIFIED][P1] result preview 虽只有 head(5)，但超宽 DataFrame 仍可能产生巨大字符串

例如：

```text
5 rows × 20,000 columns
```

`to_string()` 很大。

限制：

```text
max preview rows
max preview cols
max preview bytes
```

---

## R43-248 [VERIFIED][P1] traceback artifact 只 limit stack frames，不限制总字符串字节

异常 message 本身可以非常大。

artifact store 应有 per-artifact byte cap。

---

## R43-249 [ARCH][P1] JobStore artifacts 不存大型 inline payload

只存：

```text
ArtifactRef
```

真正 bytes 放受控 artifact store。

---

# 41. StartupCertificate 对象本身还不够 immutable

## R43-250 [VERIFIED][P1] `StartupCertificate` 是 mutable dataclass，`problems` 也是 mutable list

certificate 作为 production admission evidence 应：

```text
frozen
tuple
typed dependency digests
```

防止运行期被普通代码改写。

---

## R43-251 [ARCH][P1] certificate 有签发者/版本

字段：

```text
certificate_schema_version
issuer_build
issued_for_mode
issued_for_capabilities
```

---

# 42. DataAccess remote / query budgeting 需要闭环真实 request 数

## R43-252 [ARCH][P1] RemoteRequestMeter 统一 LIST/HEAD/GET/range/retry

每个 HTTP/COS 调用：

```text
before request → consume token
after request → latency/status
```

不靠对象数近似。

---

## R43-253 [ARCH][P1] Remote retry 也消耗 request budget

避免网络抖动时：

```text
一次逻辑请求
→ 无限/大量物理 retry
```

突破成本限制。

---

## R43-254 [ARCH][P1] metadata discovery 与 data GET 分别计 budget

因为成本/并发模型不同。

---

# 43. Config / policy reload 要有 generation fencing

## R43-255 [ARCH][P0] 每次 reload 产生 monotonic policy generation

job bind：

```text
policy_generation
policy_digest
```

execution 使用对应 snapshot。

---

## R43-256 [ARCH][P0] Registry reload 同理

```text
operator_registry_generation
field_registry_generation
source_registry_generation
calendar_generation
```

进入 ExecutionRequest。

---

## R43-257 [ARCH][P1] hot reload 不能原地 mutate global singleton

构造新 immutable snapshot：

```text
validate
→ atomic pointer swap
```

旧 job 继续持旧 snapshot ref。

---

# 44. 多进程/多实例服务必须有明确支持级别

## R43-258 [ARCH][P0] FactorEngine service 当前应明确 `SINGLE_PROCESS_ONLY`，直到 durable queue/worker lease 完成

不要只在 env 配错时 warning。

startup 输出：

```text
deployment_mode
supported_workers
```

---

## R43-259 [ARCH][P0] DataAccess service 同理

进程级：

```text
query_slots
GlobalResourceGovernor
DuckDB engine
cache
```

多 worker 会按 worker 倍增。

---

## R43-260 [ARCH][P1] 多实例真正支持后使用 shared limiter + durable job queue

例如：

```text
Postgres/Redis
```

但本地 host resources 仍由每 host coordinator 决定。

---

# 45. API 与资源治理之间的 transport backpressure

## R43-261 [ARCH][P1] HTTP request queue 不应无限依赖 server threadpool backlog

当 query slots 满：

```text
快速 429/503 + Retry-After
```

并暴露 queue wait。

---

## R43-262 [ARCH][P1] 大流式响应要有 slow-client backpressure budget

客户端读取很慢时：

```text
数据库/Arrow stream/lease
```

会长期占住。

需要：

```text
max stream lifetime
max idle write interval
```

---

## R43-263 [ARCH][P1] client disconnect 统一转 CancellationToken

FE 与 DA 两个 service 都要贯穿底层执行。

---

# 46. 数据源/结果空集语义

## R43-264 [ARCH][P0] 合法空集与 source resolution failure 在所有入口统一 typed

DataAccess 已有：

```text
EmptyPhysicalScope
```

FE/result writer/HTTP 也要区分：

```text
EMPTY_VALID
NO_COVERAGE
SOURCE_FAILURE
FILTERED_ALL
```

不能都变：

```text
empty DataFrame
```

---

## R43-265 [ARCH][P0] empty factor result 不自动等价于“成功计算”

尤其：

```text
universe resolution 失败
source columns missing
```

production 要解释 EmptyReason。

---

# 47. 时间/超时 authority 再收一层

## R43-266 [ARCH][P0] Service JobDeadline、DA DeadlineContext、DuckDB deadline 使用同一个 DeadlineIdentity

字段：

```text
wallclock_deadline
monotonic_deadline
origin
request_id
```

从入口生成一次。

---

## R43-267 [ARCH][P1] 持久化只保存 wallclock + duration，恢复时重新构造 monotonic

避免 R43-028。

---

## R43-268 [ARCH][P1] queue wait 也必须消耗 job deadline

不要：

```text
排队 1h
开始执行后再给完整 1h
```

---

# 48. 数据/作业 lineage 需要融合

## R43-269 [ARCH][P1] Job lineage 直接引用 FactorExecutionIdentity

不要 JobStore 再手写一套：

```text
request_digest
policy_digest
```

统一 ID。

---

## R43-270 [ARCH][P1] materialization lineage 引用 DurableCommitReceipt

可以从 job：

```text
→ execution
→ source snapshots
→ factor semantic identity
→ generation
```

完整追溯。

# 49. R43 必须新增的测试 / Hard Negative Cases

## R43-271 [P0] Queue concurrent admission test

```text
100 threads
same principal
limit=5
```

任何时刻：

```text
admitted <= 5
```

---

## R43-272 [P0] Submit-vs-drain race test

barrier 精确卡在：

```text
state check
final enqueue
```

drain barrier 后不得出现新 queue item。

---

## R43-273 [P0] Drain pending-job recovery test

queue 中有 100 pending；
force timeout；
shutdown/restart。

必须：

```text
0 orphan QUEUED jobs
0 leaked principal counts
```

---

## R43-274 [P0] SQLite idempotency multi-process test

至少两个独立进程同时：

```text
same idempotency key
same principal
same job type
same digest
```

断言：

```text
exactly one durable run_id
所有 caller 返回同一个 winner
```

---

## R43-275 [P0] SQLite idempotency mismatch test

同 key，不同 digest：

```text
所有 loser 必须 409
```

不能其中一个写内存 ghost。

---

## R43-276 [P0] JobStore persistence failure rollback test

在：

```text
create
update
CAS
```

分别注入异常。

断言：

```text
memory state == durable state
```

---

## R43-277 [P0] Retry active-job rejection test

```text
QUEUED
RUNNING
CANCELLING
```

全部不可创建并发 retry attempt。

---

## R43-278 [P0] Retry-after-durable-side-effect test

见 R43-163。

---

## R43-279 [P0] ReadHandle close-state test

```text
to_pandas
close
to_pandas
```

必须拒绝（若 CLOSED contract 保持）。

同理 Polars。

---

## R43-280 [P0] ReadHandle stats exactly-once finalize

物化→等待→close：

```text
elapsed 不得重复增长
```

---

## R43-281 [P0] ScanHandle sibling lease test

```text
h1
h2=h1.filter(...)
h1.close()
h2.collect()
```

在正确 refcount 设计下仍受 lease；
旧设计应测试失败。

---

## R43-282 [P0] ScanHandle collect-after-close test

必须 fail。

---

## R43-283 [P0] ScanHandle repeated collect test

必须：

```text
reuse materialized result
或 fail
```

绝不无 lease 第二次 scan。

---

## R43-284 [P0] Research snapshot mutation test

scan 后替换文件，再 collect：

返回 lineage 必须描述**实际读到的 object set**。

---

## R43-285 [P0] QueryBudget.tighten strict typing test

必须拒绝：

```python
max_rows=True
max_elapsed_ms=True
require_columns="false"
```

---

## R43-286 [P0] Host lease rejection no-fallback test

active JobLease 剩余不足：

```text
DA child lease REJECTED
```

必须拒绝 query，不能 local governor 接管。

---

## R43-287 [P0] Host-backed duplicate query id test

不能覆盖旧 reservation。

---

## R43-288 [P0] Startup certificate mode test

```text
research certificate with problems
→ switch/request production
```

必须重新 gate 并失败。

---

## R43-289 [P0] Startup certificate dependency mutation test

签发后修改：

```text
calendar
registry
policy
```

证书立即 invalid。

---

## R43-290 [P0] FactorEngine readyz behavioral test

不允许：

```text
永久 UNKNOWN 占位导致永远不 ready
```

也不允许：

```text
_check_done 让坏机制 PASS
```

---

## R43-291 [P0] Service backend schema parity test

对每个 backend：

```text
validate-spec accepts
typed model accepts
execution factory accepts
```

三者集合完全一致。

---

## R43-292 [P0] Config-path TOCTOU test

提交 job 后修改 YAML。

系统必须：

```text
执行 frozen config
或
拒绝 digest changed
```

---

## R43-293 [P0] Result preview classification failure test

人为让 classification module raise。

production artifact 必须：

```text
redacted
```

---

## R43-294 [P0] Generation power-loss durability test

使用 filesystem/fault injection 验证：

```text
data fsync
manifest fsync
directory fsync
```

commit point。

---

## R43-295 [P0] Generation duplicate-key test

new batch 有重复 upsert key：

```text
hard fail
```

除非显式 conflict policy。

---

## R43-296 [P0] URI HTTP limits parity

ReadURIRequest：

```text
columns/instruments/time range/filter complexity
```

不能比普通 ReadRequest 更宽，除非显式 privileged budget。

---

## R43-297 [P1] Observability 24h cardinality/heap test

确认：

```text
histogram memory bounded
labels bounded
ContextVar no bleed
```

---

## R43-298 [P1] Wheel compatibility matrix

见 R43-175/176。

---

## R43-299 [P1] Multi-entry semantic consistency matrix

见 R43-166。

---

## R43-300 [P0] Current-SHA release evidence gate

最终：

```text
final HEAD
wheel digest
test evidence
hard gate report
benchmark
```

必须是同一 build generation。

---

# 50. R43 Hard Gates

## HG-R43-01 Job Admission

```text
ADMISSION_LIMIT_OVERSHOOT == 0
ENQUEUE_AFTER_DRAIN_BARRIER == 0
ORPHAN_PENDING_AFTER_STOP == 0
```

## HG-R43-02 JobStore

```text
IDEMPOTENCY_DURABLE_WINNER_COUNT == 1
MEMORY_DURABLE_STATE_DIVERGENCE == 0
MUTABLE_RECORD_OUTSIDE_STORE_UPDATE == 0
```

## HG-R43-03 Retry

```text
ACTIVE_JOB_DUPLICATE_RETRY == 0
ATTEMPT_LINEAGE_MISSING == 0
UNKNOWN_SIDE_EFFECT_BLIND_RETRY == 0
```

## HG-R43-04 Resource Lease

```text
PARENT_LEASE_REJECTED_LOCAL_FALLBACK == 0
DUPLICATE_QUERY_RESERVATION_OVERWRITE == 0
RELEASE_FAILURE_SILENT == 0
```

## HG-R43-05 ReadHandle / ScanHandle

```text
COLLECT_AFTER_RELEASE == 0
SIBLING_PREMATURE_LEASE_RELEASE == 0
REPEATED_UNGOVERNED_COLLECT == 0
SNAPSHOT_LINEAGE_ACTUAL_READ_MISMATCH == 0
```

## HG-R43-06 QueryBudget

```text
TIGHTEN_TYPED_BYPASS == 0
HTTP_BUDGET_FIELD_LOSS == 0
PRODUCTION_UNBOUNDED_MEMORY_COLLECT == 0
```

## HG-R43-07 Generation

```text
GENERATION_COMMIT_WITHOUT_DURABILITY_BARRIER == 0
DUPLICATE_UPSERT_KEYS_ACCEPTED == 0
UNSAFE_PARTITION_PATH_ENCODING == 0
```

## HG-R43-08 Startup / Ready

```text
RESEARCH_CERT_REUSED_AS_PRODUCTION == 0
SUCCESS_CERT_WITHOUT_DEPENDENCY_DIGEST == 0
FIXED_UNKNOWN_READINESS == 0
UNCONDITIONAL_PASS_BLOCKER == 0
```

## HG-R43-09 Service Security

```text
REDACTION_FAILURE_RAW_PREVIEW == 0
CONFIG_PATH_RESULT_UNCLASSIFIED == 0
RAW_TRACEBACK_EXPOSED_TO_NORMAL_USER == 0
```

## HG-R43-10 Universe

```text
NAMED_UNIVERSE_OUTPUT_NOT_APPLIED == 0
INSTRUMENT_FILTER_UNIVERSE_UNPROVEN == 0
CS_CATEGORY_LOOKUP_FAIL_OPEN == 0
```

## HG-R43-11 API Contract

```text
VALIDATE_EXECUTE_SCHEMA_DIFF == 0
URI_REQUEST_LIMIT_BYPASS == 0
IDEMPOTENCY_DIGEST_OBSERVABLE_FIELD_OMISSION == 0
```

## HG-R43-12 Evidence

```text
FINAL_HEAD_WITHOUT_CI_EVIDENCE == 0
SOURCE_TEST_ARTIFACT_BUILD_MISMATCH == 0
```

---

# 51. 推荐执行优先级

## Phase A — 先修真正 P0 状态分裂

优先顺序：

```text
1.  R43-019  SQLite idempotency split-brain
2.  R43-020~022 JobStore memory/durable rollback
3.  R43-043~045 create→queue 非事务 + QUEUED 不持久
4.  R43-001~004 queue admission/drain
5.  R43-048~050 retry active/side-effect fencing
```

---

## Phase B — 修资源 lease 与 handle ownership

```text
R43-011~018
R43-073~087
R43-095~100
```

这是长期 OOM/资源泄漏/unguarded read 的关键。

---

## Phase C — 修 Startup/Readiness

```text
R43-057~064
R43-127~133
R43-187~193
```

---

## Phase D — 修 HTTP / Identity / Policy

```text
R43-030~046
R43-052~056
R43-101~107
R43-181~200
R43-231~249
```

---

## Phase E — 修 Generation durable transaction

```text
R43-108~126
R43-201~205
```

---

## Phase F — 观测、打包、CI、Chaos

```text
R43-065~072
R43-139~148
R43-170~180
R43-206~230
R43-271~300
```

---

# 52. Coding AI 必须生成的 R43 Closure Ledger

文件：

```text
factor_engine/docs/R43_ISSUE_CLOSURE_LEDGER.md
```

以及 DataAccess 对应：

```text
dataaccess/docs/R43_DATAACCESS_CLOSURE_LEDGER.md
```

每项：

```text
ID
baseline HEAD
current HEAD
classification
root cause
changed files
behavior before
behavior after
tests
negative test
fault injection
evidence
status
```

状态只允许：

```text
FIXED
FIXED_ALREADY_WITH_CURRENT_HEAD_PROOF
NOT_APPLICABLE_WITH_PROOF
ARCHITECTURE_BACKLOG_WITH_JUSTIFICATION
BLOCKED_BY_EXTERNAL_DEPENDENCY
```

`VERIFIED + P0` 不允许留下：

```text
PARTIAL
NOT_RUN
OPEN
```

---

# 53. R43 执行完成后必须再次搜索的模式

```text
except Exception: pass
except Exception:
return None
ON CONFLICT DO NOTHING
job.status =
_store.update
queue.put
threading.local
threading.Thread
time.sleep(
monotonic()
default=str
traceback.format_exc
to_string()
BytesIO()
duckdb.connect(":memory:")
os.replace
write_text
pq.write_table
str(v)
.join(pcols)
_prepared=self._prepared
release_reservation
collect()
collect_batches
```

对每个命中判断：

```text
是不是 authority bypass
是不是 release failure
是不是 durability gap
是不是 state split
```

新增问题编号：

```text
R43-NEW-301+
```

---

# 54. 最终 Definition of Done

R43 完成不是“测试绿了”这么简单。

必须能证明：

## Job

```text
一个请求只产生一个 durable winner
queue/store/token/lease 状态一致
cancel/retry/crash 可恢复
```

## DataAccess

```text
任何真实 scan 都有有效 reservation
任何 reservation 都有唯一 owner
handle 派生不会提前 release
snapshot 永远描述实际读到的 objects
```

## Write

```text
generation commit point 真正 durable
retry 能识别 commit outcome
没有 mixed generation / lost update
```

## Startup

```text
production certificate 只能由 production checks 签发
certificate 绑定实际 config/build/calendar/policy/source generations
```

## Readiness

```text
不靠固定 PASS
不靠永久 UNKNOWN
真实判断当前实例是否安全接单
```

## Evidence

```text
final source
built wheel
CI
hard gates
部署 capability
```

全部同一 build identity。

---

# 55. 最终执行指令

```text
A. git fetch，读取执行时真实 main HEAD
B. 读取 R42 closure，禁止重复改已闭环项
C. 逐项证明 R43 VERIFIED 问题在当前 HEAD 是否仍存在
D. 先修 JobStore/Queue/idempotency/transaction P0
E. 再修 HostLease + DA reservation + ReadHandle/ScanHandle ownership
F. 修 StartupCertificate / readyz / release blockers
G. 修 HTTP validate/execute schema、request identity、config-path freeze
H. 修 result preview / traceback classification
I. 修 QueryBudget HTTP字段丢失与 tighten typed bypass
J. 修 Generation fsync、partition path、duplicate keys、write transaction
K. 修 named universe output/computation domain
L. 建 current-SHA CI/wheel/fault-injection evidence
M. 运行 R43-271~300 negative tests
N. 再全仓搜索 R43 §53 模式
O. 新问题编号 R43-NEW-301+
P. 所有 VERIFIED P0 闭环后才能结束
```

---

# 56. R43 总结

FactorEngine 和 DataAccess 当前最值得继续改的，已经不是“再写多少代码”的问题，而是把下面这些词真正做到系统级一致：

```text
immutable
idempotent
exactly-once
fail-closed
durable
owned
cancelled
retried
validated
certified
```

现在多个模块局部上已经写了这些概念，但 R43 发现的主要问题正是：

```text
A 模块的 exactly-once
+
B 模块的 release-once
+
C 模块的 idempotency
```

组合起来并不自动等于：

```text
整个请求 exactly-once。
```

R43 的目标就是把：

```text
请求
→ job
→ queue
→ worker
→ lease
→ read
→ compute
→ write
→ durable commit
→ final status
```

真正收成一条可以做 crash/retry/cancel 证明的事务链。
