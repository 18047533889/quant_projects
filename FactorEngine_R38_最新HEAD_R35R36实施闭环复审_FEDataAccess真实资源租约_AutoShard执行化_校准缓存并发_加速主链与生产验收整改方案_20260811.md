# FactorEngine R38：最新 HEAD R35/R36 实施闭环复审——FE×DataAccess 真实资源租约、AutoShard 执行化、校准/缓存/并发修复、加速主链落地与生产验收整改方案

> **审计基线：** `main@d34cc9f50df92a10ca5c84a6d64d4682d2cd9f23`  
> **仓库：** `18047533889/quant_projects`  
> **范围：** `factor_engine/` + `dataaccess/` + 两者之间的真实运行时边界  
> **定位：** 本文是 **R35/R36 已实现内容的实施闭环复审**，不是 R37 的替代版，也不重复 R37 已负责的全量语义/PIT/参数域/证据体系。  
> **执行要求：** Coding agent 应直接修改真实主执行链；禁止只修改报告、gate、注释或测试期望值来宣布完成。

---

# 0. 本轮为什么需要单独做 R38

最新 `main` 已经加入了大量 R35/R36 的实现，包括：

- `NumbaKernelRegistry`
- `FastLinearWindowEngine`
- `PCABlock / GARCHFitBlock`
- `ResourceController / ResourceDecision`
- `HostResourceCoordinator`
- `ResourceCalibrationStore`
- `RunPeakSampler`
- `GovernedBufferStore`
- `AutoShardPlanner`
- FE scheduler 对资源控制器的消费
- DataAccess governor 的动态上限 setter
- service broker `ContextVar`
- R35/R36/R37 evidence scaffold

这些方向总体正确，但本次最新 HEAD 复审发现：**部分模块目前还是“控制面/合同层看起来完成，真实执行链并没有按这个合同执行”**。如果直接把这些模块当成已经闭环，会产生比“没有功能”更危险的问题——系统可能基于错误的资源账本启动本来不该启动的任务，或者 benchmark 看起来很快但 production 路径根本没走加速实现。

因此 R38 专门处理这一类问题：

```text
模块已经存在
≠
主链已经接入
≠
真实运行行为与文档一致
≠
生产闭环
```

---

# 1. 与 R37 的边界

## 1.1 R37 继续负责的事项

以下事项继续由 R37 作为主任务书，不在 R38 重复铺开：

- 全 production canonical 的 Per-Canonical Correctness Ledger
- 全参数域认证与 production call membership
- EvidenceTruth 全局体系
- PIT / availability / universe / calendar / corporate action / financial bitemporal
- DataKnowledgeIdentity / FactorIdentity
- FE×DA 最终 Unified QueryGraph 的完整语义设计
- OutputShapeContract
- exactly-once generation 的全局体系
- ChangeImpactDAG / 历史修订传播
- current-HEAD 全量 release evidence

R38 中如果提到这些，只是因为它们与当前 R35/R36 实现缺口直接相连。

## 1.2 R38 只收以下类型的问题

```text
R35/R36 已宣称完成
但真实主链未完成；
或者最新实现本身新增了
runtime / concurrency / resource / performance correctness 缺陷。
```

---

# 2. 本轮结论

当前 HEAD 不建议把 R35/R36 视为完全“工程闭环”。

尤其需要优先解决以下问题：

1. **AutoShard 目前并没有真的 shard 执行，只把 resource contract 的 peak_memory 改小。**
2. **真实 OOM 不会进入 shrink/replan/shard 路径，而是直接作为 permanent error 抛出。**
3. **ResourceCalibrationStore 当前收到的 `elapsed_ms` 和 `peak_mem` 并不是真实观测。**
4. **GovernedBufferStore `put()` 拒绝写入时，调用方忽略 False 并直接 return。**
5. **production 路径仍然存在 raw `shared_result_cache[sid] = value` fallback。**
6. **所谓 spill 当前只是 release，完全没有 spill/reload。**
7. **HostResourceCoordinator parent/child lease 的内存账目存在双算，CPU/IO/spill 也未真正约束。**
8. **FE 与 DataAccess 尚未真正使用同一套 child lease；当前主要是把一个 safe cap 写给 DA。**
9. **DataAccess standalone governor 默认 memory cap 仍可为 None。**
10. **Service queue 并没有真正做 HostCoordinator job lease admission。**
11. **ResourceController 被多个 scheduler 热循环直接 tick，控制速度与循环次数相关，而不是时间相关。**
12. **ResourceDecision 的很多字段并未被真实 consumer 消费。**
13. **scheduler 用 `target_cpu_tokens` 当 task concurrency 上限，而不是 `target_concurrency`。**
14. **运行中 memory pressure 变大时，已生成的 read waves 不会重新拆小。**
15. **sink queue 预算只在创建时决定，不会随压力动态缩容。**
16. **DuckDB / Polars thread token 仍大量基于 env/default 推测，而不是实际 lease。**
17. **FastLinearWindowEngine 名义上是 sliding sufficient-statistics，实际仍逐窗口 rescanning。**
18. **Numba kernels 已经 benchmark，但实际 cleaned operator 主链仍大多调用 Python/Numpy reference 实现。**
19. **model-family shared PCA state 与真实 panel PCA canonical 的语义已经出现漂移。**
20. **DataReadSession 通过修改共享 Store `_resolution_cache` 实现 session cache，存在并发覆盖竞态。**
21. **R36 的若干 hard gate 检查的是“代码里有这个东西”，不是“真实 workload 已按它执行”。**
22. **最新 HEAD 当前没有可见的 current-SHA CI/workflow 证明。**

这些属于 R38 的直接整改范围。

---

# 3. R38 Definition of Done

本轮完成后必须达到：

```text
A. AutoShard 真的把一个 task 变成 N 个可执行 shard + merge，而不是只改内存数字；
B. OOM 会自动进入 smaller-shape replan，并且相同 shape 不重试；
C. ResourceCalibrationStore 只接收真实 runtime observation；
D. FE/DA/service/writer 都通过同一个 HostResourceCoordinator lease tree；
E. parent/child lease 不 double count；
F. CPU / IO / spill 也被 lease 真正约束；
G. resource controller 按固定时间 cadence 更新，不被 scheduler loop 次数放大；
H. scheduler 只读取最新 ResourceDecision；
I. ResourceDecision 的每个字段都有唯一 consumer；
J. dynamic read-wave / sink / cache / block 在压力变化时真实调整；
K. GovernedBufferStore 无 production raw bypass；
L. buffer spill 能写出、校验、reload 或按 policy recompute；
M. DataReadSession 并发不会互相覆盖 resolution cache；
N. Numba/FastLinear/model-family shared state 真正进入 FE 主执行路径；
O. 所有 fast path 都与 authoritative reference 保持 parity；
P. R36 gate 重新从 behavior 验证，而不是 source presence；
Q. current final SHA 有真实 CI + integration + stress evidence。
```

---

# 4. P0：AutoShard 必须从“改合同”变成“真实执行”

## R38-P0-001：当前 `_auto_shard_replan()` 是伪 sharding

### 当前实现问题

当前 scheduler 在无法 admission 时调用 `AutoShardPlanner`，然后核心动作是：

```python
new_contract = replace(
    contract,
    peak_memory_bytes=plan.per_shard_peak_bytes,
    output_bytes=plan.per_shard_peak_bytes,
)
dag.tasks[tid] = rebase_task(
    task,
    resource_contract=new_contract,
)
```

这只是把**资源预测数字**变小。

真实的：

- node_ref 没切；
- input data 没切；
- time range 没切；
- instrument scope 没切；
- 没增加 warmup overlap；
- 没创建 shard child tasks；
- 没增加 merge barrier；
- backend 仍然执行完整原 task。

结果是：

```text
原 task 真实要 20GB
↓
contract 被改成 4GB
↓
ResourceBroker 认为能放下
↓
实际仍执行 20GB workload
↓
更容易 OOM
```

这是 P0。

### 必须改成

新增真实：

```python
@dataclass(frozen=True)
class ShardDescriptor:
    shard_id: str
    dimension: str
    input_slice: object
    warmup_slice: object | None
    output_slice: object
    checkpoint_ref: object | None
    preserves_full_cross_section: bool
    merge_order: int

@dataclass
class ShardExecutionPlan:
    original_task_id: str
    shards: tuple[ShardDescriptor, ...]
    merge_policy: str
```

scheduler replan 时必须：

```text
original task
→ shard child 1
→ shard child 2
→ ...
→ shard child N
→ MERGE task
```

而不是保留原 task。

### 各类 task shard 语义

#### elementwise

允许：

```text
asset shard
time shard
```

#### time-series rolling

优先：

```text
asset shard
```

若 time shard：

```text
input window = output block + lookback overlap
output只保留非 overlap 部分
```

#### cross-sectional

只允许：

```text
time shard
```

每个 date block 必须含完整当日 universe。

严禁：

```text
把5000只股票拆成5组
每组自己rank
```

#### group

默认 time shard，必须保持当日完整 group membership。

#### stateful

优先 asset shard。

time shard 只有：

```text
validated checkpoint
```

才允许。

#### PCA / panel model

```text
time shard
+
完整 cross-section
+
training history overlap
```

#### minute→daily

按：

```text
session/day
```

切。

---

## R38-P0-002：Shard merge 必须有独立 contract

merge 必须明确：

- index union
- duplicate key policy
- shard ordering
- overlap trim
- dtype
- timezone
- column order
- missing cells
- state checkpoint continuity

增加：

```text
ShardMergeContract
```

所有 shard family 做：

```text
full-run == sharded-run
```

---

## R38-P0-003：shard 结果不得在 merge 前全部常驻内存

如果：

```text
20 shards × 1GB
```

全部存到 list 再 concat，仍会 OOM。

应该：

```text
shard compute
→ optional DQ
→ streaming merge / write
→ release shard
```

对于最终需要整体 merge 的情况，使用 Arrow/Parquet spill relation 或 bounded merge buffer。

---

# 5. P0：OOM 必须驱动真实 replan

## R38-P0-004：当前 MemoryError 直接 fatal，不会 auto-shard

当前：

```text
MemoryError / "out of memory"
→ ERROR_PERMANENT
→ scheduler raise
```

`AutoShardPlanner` 只在 admission 连续无进展 200 轮后尝试。

这意味着：

```text
预测低估
→ task被成功admit
→ 真正OOM
→ 直接失败
```

并没有 R36 所要求的：

```text
OOM
→ underprediction
→ shrink
→ shard
→ retry smaller workload
```

### 必须新增错误类型

```python
class ResourceUnderpredictionError(...)
class OOMReplanRequired(...)
```

OOM 不属于普通 transient，也不属于“不可重试 permanent”。

它属于：

```text
same shape不可重试
smaller shape可以重试
```

---

## R38-P0-005：OOM recovery 流程

必须：

```text
catch MemoryError / backend OOM
→ mark shape underpredicted
→ ResourceCalibrationStore.record(oom=True)
→ reduce target_concurrency
→ reduce read wave
→ reduce block
→ evict/spill low-value buffers
→ AutoShardPlanner.replan_after_oom
→ ensure new shape < old shape
→ execute smaller shards
```

硬 invariant：

```text
new_shape_signature != failed_shape_signature
```

否则禁止 retry。

---

## R38-P0-006：DuckDB / Polars OOM 也归一到同一 taxonomy

识别：

- DuckDB OutOfMemoryException
- ArrowMemoryError
- Polars ComputeError 中明确 OOM 类
- Python MemoryError
- OS/cgroup memory event

统一转成：

```text
OOMReplanRequired
```

但不要把：

```text
semantic error
schema error
PIT error
```

误当资源错误重试。

---

# 6. P0：ResourceCalibrationStore 当前被错误数据污染

## R38-P0-007：`elapsed_ms` 当前实际上是绝对 monotonic timestamp

当前 `_record_timing()` 写：

```python
finished_at_ms = time.monotonic() * 1000
```

然后 `_record_task_calibration()` 直接把：

```python
elapsed = finished_at_ms
```

写进 CalibrationStore。

这不是 duration。

### 必须修

task observation 必须有：

```python
@dataclass
class TaskRunObservation:
    task_id: str
    started_at_monotonic: float
    finished_at_monotonic: float
    elapsed_ms: float
    baseline_family_pss: int
    peak_family_pss: int
    output_bytes_actual: int
    spill_bytes_actual: int
    read_bytes_actual: int
    write_bytes_actual: int
    backend_threads_actual: int
```

---

## R38-P0-008：`peak_mem` 当前仍然写的是预测 contract

当前：

```python
peak = contract.peak_memory_bytes
```

这会导致：

```text
预测模型
→ 用自己的预测值当“真实值”
→ 再学习自己的预测
```

完全失去 calibration 意义。

### 必须改

至少区分：

```text
observed_peak_mem
predicted_peak_mem
attribution_quality
```

可选 attribution quality：

```text
isolated
low-concurrency
concurrent-marginal
run-level-only
unattributed
```

只有足够可信的 observation 才进入 shape P99 主模型。

---

## R38-P0-009：`output_bytes` 必须来自真实 result

不能：

```python
output_bytes = contract.output_bytes
```

应该：

```python
estimate_object_bytes(result)
```

或 BufferRef/Arrow object 自带实际 byte size。

---

## R38-P0-010：CalibrationStore 必须避免被旧错误样本继续污染

由于当前 HEAD 已可能产生错误 observation，修复后：

- bump calibration schema version；
- 旧 schema 数据不进入新 P99；
- 可保留旧数据做历史诊断；
- `calibration_source_version` 写入 store；
- hardware fingerprint + runtime version + schema version 共同做 key。

---

# 7. P0：ResourceController 改成固定 cadence 的单一控制循环

## R38-P0-011：当前 controller 被 scheduler loop 次数驱动

当前每个 scheduler loop 都：

```python
broker.resource_decision()
→ _refresh(force=True)
→ controller.tick()
```

多 job 同时运行时，同一个 broker/controller 被多个 scheduler thread 高频 tick。

这会导致：

- `_stable_count` 增长速度由 job 数决定；
- `cooldown` 由 loop 次数决定；
- recovery 可能在几毫秒内完成；
- pressure down 也可能被重复乘很多次；
- controller mutable state 无统一锁；
- PSS/PSI 采样被频繁执行。

### 正确架构

新增：

```text
ResourceAutopilotService
```

一个 process-wide background control loop：

```text
every 500ms~1s
→ ResourceMonitor.sample()
→ ResourceController.tick()
→ atomic publish ResourceDecisionSnapshot
```

scheduler：

```text
只读 last_decision()
```

绝不自己 tick controller。

---

## R38-P0-012：ResourceDecision 加时间与版本

```python
ResourceDecision(
    decision_id,
    generated_at_monotonic,
    valid_until,
    signals_version,
    ...
)
```

scheduler 若 decision 太旧：

```text
conservative fallback
```

而不是自行创建第二套 decision。

---

## R38-P0-013：不要在 scheduler 热循环 `_refresh(force=True)`

PSS 读取可能触碰 `/proc/<pid>/smaps_rollup`，process family 又有递归 child。

应由 monitor 以受控 cadence 采样。

小批量 benchmark 必须测：

```text
resource monitoring overhead
```

---

# 8. P0：ResourceDecision 的预算必须从一个 SafeEnvelope 统一分配

## R38-P0-014：当前各预算独立 clamp，可能总和超过 safe memory

当前大致：

```text
wave = safe * 10% with min 256MB
block = safe * 5% with min 64MB
sink = safe * 5% with min 128MB
cache = hard * 15%
...
```

在低 headroom 时，absolute minimum 可能已经高于 safe memory。

### 必须实现

```python
MemoryBudgetAllocator.allocate(
    safe_memory,
    active_workload,
    pressure,
)
```

满足硬 invariant：

```text
sum(all simultaneously-live memory reservations)
+ emergency reserve
<= current safe envelope
```

---

## R38-P0-015：cache budget 必须跟 live safe envelope 收缩

不能长期：

```text
cache_budget = hard RAM * 15%
```

外部算法占走大量 RAM 后，cache target 也要缩。

推荐：

```text
cache_target =
min(
  static_max,
  safe_envelope * adaptive_cache_fraction,
  job_lease_remaining
)
```

压力解除后再慢慢增长。

---

## R38-P0-016：spill budget 必须来自磁盘，不是 RAM

spill budget 依据：

- spill filesystem free
- reserved free space
- IO pressure
- write workload
- disk quota
- remote spill cost

不能用：

```text
hard_memory * 15%
```

作为主 spill budget。

---

# 9. P0：HostResourceCoordinator lease tree 当前账本要重做

## R38-P0-017：parent + child 当前会 double count memory

当前 host used 是：

```python
sum(all active leases.memory_bytes)
```

若：

```text
JobLease = 16GB
child ComputeLease = 8GB
```

host账本会算：

```text
24GB
```

但 child 是 parent 内部额度，不应再次占 host。

### 统一两种合法模型，二选一

#### 方案A：root reservation accounting

只有 root lease 计入 host reserved。

children 只在 parent 内分配，不再计 host。

#### 方案B：leaf accounting

parent只是 ceiling，不计 reserved；
只有 active leaf leases 计入 host。

推荐方案A，较容易做 job QoS。

---

## R38-P0-018：CPU / IO / spill 当前只记录、不约束

`HostResourceLease` 有：

```text
cpu_tokens
io_tokens
spill_bytes
```

但 request_lease 主要只校验 memory。

必须维护：

```text
host_cpu_tokens_reserved
host_io_tokens_reserved
host_spill_reserved
```

并与 ResourceBroker/DA/DuckDB/Writer 共用。

---

## R38-P0-019：Job memory lease 不能是 coordinator 的一个全局 scalar

当前：

```python
self._job_lease_bytes
```

并发 job A/B 会共享这一字段。

必须：

```text
JobLease object
```

显式传递：

```text
Service Job
→ FactorEngine execution context
→ scheduler
→ DataAccess child scan
→ writer child lease
```

---

## R38-P0-020：release parent 必须递归释放整棵 lease tree

当前只处理 direct children。

需要：

```text
parent
├─ compute
│  └─ numba
└─ data
   └─ duckdb
```

任何 parent cancel/fail：

```text
整棵树 terminal/released
```

并支持 idempotent release。

---

## R38-P1-021：released lease 不应永久保留在主 dict

增加：

- active leases
- recent terminal ring buffer
- durable metrics

避免长运行 service 内 `_leases` 无限增长。

---

## R38-P0-022：`summary()` 必须纯读，不能顺手修改 DA governor

当前 HostCoordinator summary 里调用：

```text
apply_da_envelope()
```

属于 observability side effect。

修成：

```text
summary() -> pure snapshot
sync_da_limits() -> explicit action
```

---

# 10. P0：FE×DataAccess 真正统一资源，而不是“给 DA 写一个 cap”

## R38-P0-023：当前 `apply_da_envelope()` 不是 child lease

现在主要是：

```text
Host safe envelope
→ set DA max_total_reserved_memory
```

DA 实际 read pipeline 仍然：

```text
ResourceReservation
→ DataAccess GlobalResourceGovernor.admit()
```

HostCoordinator 并不知道这次 read 实际保留了多少。

所以这仍然是：

```text
两个账本
+
一个共享上限
```

而不是：

```text
一个账本
```

### 必须新增 DataAccess bridge

建议在 DataAccess：

```python
class HostResourceLeaseProvider(Protocol):
    def request_child_lease(...)
    def release_child_lease(...)
```

`ReadPipeline.admit()`：

```text
如果有 host provider
→ request DataAccessScanLease
→ DA local governor 只做 DA-specific细分限制
→ 不再重复全局内存 admission
```

---

## R38-P0-024：PreparedRead 保存真实 HostLeaseRef

`PreparedRead` 增加：

```text
host_lease_id
resource_scope_id
```

execute / generator close / exception：

```text
release exactly once
```

---

## R38-P0-025：DA standalone 模式必须自己安全

如果 DataAccess 独立 service 使用，没有 FE coordinator：

当前：

```text
max_total_reserved_memory=None
max_total_scan_bytes_inflight=None
```

不应在 production 等价于无限。

独立模式：

```text
detect cgroup / host / RLIMIT
→ safe envelope
→ auto bounded governor
```

若无法探测：

```text
production fail closed 或 conservative fixed safe cap
```

---

## R38-P0-026：DA governor setter 要加锁

以下动态 setter：

```text
set_max_total_reserved_memory
set_max_total_scan_bytes_inflight
```

必须与 `admit()` 同锁。

cap 被收缩到低于当前 reservations 时：

- 不 kill incumbents；
- 阻止新 admission；
- telemetry 标 `over_current_target`；
- incumbents release 后恢复正常。

---

# 11. P0：Service queue 必须真正申请 JobLease

## R38-P0-027：当前 service 仍主要靠固定 max_running

虽然 service 已拿到 HostCoordinator，但 `submit()` 的 admission 还是：

- queue count
- running count
- per-principal count

缺乏：

```text
JobResourceEstimate
→ HostCoordinator.request_job_lease
```

### 必须增加

job request 编译后或 enqueue 前有：

```python
JobResourceEstimate(
    priority,
    memory_p99,
    cpu_budget,
    io_class,
    estimated_runtime,
)
```

实际 worker 启动前申请 `JobLease`。

---

## R38-P0-028：QoS lane 真正接资源控制器

至少：

```text
CRITICAL
STANDARD
BACKGROUND
```

当 pressure 升高：

```text
BACKGROUND停止新 admission
STANDARD收缩
CRITICAL保底
```

不能只靠 FIFO。

---

# 12. P0：GovernedBufferStore 的拒绝语义现在是错误的

## R38-P0-029：`put()` False 被调用方忽略

当前：

```python
if store is not None:
    store.put(sid, value)
    return
```

但 `put()` 可能因为 budget 不够返回 `False`。

此时：

```text
value 没存
→ 函数却 return 成功
→ downstream plan_ref 读取
→ KeyError / 隐性重算失败
```

### 修改

推荐不要返回裸 bool，而是：

```python
BufferPutResult(
    status="MEMORY"|"SPILLED"|"RECOMPUTE"|"REFUSED",
    ref=...
)
```

production 必须处理每种状态。

---

## R38-P0-030：production 删除 raw dict fallback

当前最终仍有：

```python
ctx.shared_result_cache[sid] = value
```

这条 fallback 没有 run_mode guard。

改成：

```text
production:
  no governed store/cache
  => fail closed

research:
  explicit warning + telemetry
  => optional unmanaged fallback
```

更推荐最终连 research 也统一走 BufferStore。

---

## R38-P0-031：PandasBackend `plan_ref` 应读 BufferStore/BufferRef

当前 plan_ref 直接查：

```text
ctx.shared_result_cache
```

如果 BufferStore 未来支持 spill/ref，就必须改成：

```text
ctx.shared_buffers.get(sid)
```

必要时：

```text
memory ref
→ reload spill
→ recompute policy
```

---

# 13. P0：当前 spill 根本不是 spill

## R38-P0-032：`GovernedBufferStore.spill()` 现在只是 release

当前：

```python
def spill(self, key):
    self.release(key)
```

这叫 drop，不叫 spill。

### 真正需要

```python
SpillRef(
    path,
    checksum,
    bytes,
    dtype/schema,
    generation_id,
    source_identity,
)
```

`SpillStore`：

```text
spill(buffer)
reload(ref)
delete(ref)
cleanup_execution(execution_id)
```

---

## R38-P0-033：spill / recompute 要成本驱动

低成本 elementwise：

```text
drop + recompute
```

高成本：

```text
PCA state
GARCH fit
expensive temporal join
large source block
```

倾向 spill。

公式：

```text
spill if:
reload_cost + io_pressure_cost
<
recompute_cost
```

---

## R38-P0-034：spill 和 writer 共用 IO lease

避免：

```text
memory pressure
→ 大量spill
同时
→ writer大批落盘
→ IO被打爆
```

统一从 HostCoordinator 拿 IO tokens。

---

# 14. P1：BufferStore 继续修正

## R38-P1-035：`get()` 加锁

当前 backing dict 读取未统一持锁。

在多 worker root 消费、release、evict 同时发生时需保证引用生命周期。

---

## R38-P1-036：eviction 不是 LRU

当前 `_evict_for` 按 dict 顺序走，没有 hit bump。

需要：

```text
last_access
reuse_count
recompute_cost
size
```

至少 LRU，最好：

```text
reuse_probability * recompute_cost / bytes
```

---

## R38-P1-037：reconciliation sample 算法修正

当前：

```text
accounted = all keys
sampled_actual = first <=512 keys
drift = abs(accounted - sampled_actual)
```

若 keys > 512，天然产生巨大“漂移”。

应：

- keys <= sample limit：全量比；
- keys > limit：抽样估计 + extrapolation；
- 或只比较 sampled keys 对应 accounted bytes。

---

# 15. P0：ExecutionCacheSession 仍有 split-brain cache

现在同一 backing 可能同时被：

```text
ExpressionCache
GovernedBufferStore
shared_result_cache raw dict
```

三个接口管理。

这会导致：

- 两套 byte accounting；
- 两套 eviction；
- 一个接口写入另一个不知道；
- backing 被清但 accounting 未必同步。

## R38-P0-038：只保留一个 L0 CSE owner

推荐：

```text
GovernedBufferStore
```

作为 L0 权威。

`ExpressionCache` 变成 BufferStore 的 adapter，或被合并。

不能两套独立 accounting 都指向同一个 backing。

---

## R38-P0-039：`strict` 不要只看环境变量

当前：

```python
self.strict = os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production"
```

但 run mode 也可能来自 config/ExecutionContext，而 env 没设。

必须显式传：

```text
run_mode / production bool
```

从 Engine execution context 进入 CacheSession。

---

## R38-P1-040：release unregister 失败 production 也要记录

初始化注册失败已经做 fail-closed，但 release 的 unregister 异常仍直接 pass。

至少：

- telemetry
- forced reconciliation
- cleanup retry

不应完全静默。

---

# 16. P0：动态 read wave 当前不是真动态

## R38-P0-041：运行中的 pressure 变化不会拆已有 wave

`plan()` 一次构建：

```text
plan.read_waves
```

run 时虽然更新：

```text
self.wave_memory_budget = decision.read_wave_bytes
```

但 `_execute_read_waves()` 还是执行原来已经生成的 wave。

### 修法

改成 JIT wave planner：

```text
ready source tasks
→ current decision.read_wave_bytes
→ build next wave only
→ execute
→ repeat
```

或者支持：

```text
repartition_unexecuted_waves()
```

---

## R38-P0-042：source wave 也必须有 Host ScanLease

每 wave：

```text
estimated scan bytes
estimated output bytes
remote requests
```

先拿 `DataAccessScanLease`。

完成后 release。

---

# 17. P0：sink queue 必须随资源压力弹性调整

## R38-P0-043：queue budget 当前只初始化一次

创建 sink 时从 ResourceDecision 拿一次 `queue_bytes`。

后续外部算法占内存，queue 仍允许涨到旧目标。

增加：

```python
BoundedResultQueue.set_target_bytes(new_target)
```

语义：

- 缩容时不丢已有 items；
- producer 停止增长；
- 等消费降到新 target 以下；
- recovery 后缓慢放宽。

---

## R38-P0-044：多 writer queue budget 是 total，不是每 worker重复

若未来 writer_threads>1：

```text
total sink budget
```

应分配给各 worker，不能每个 worker 都拿 full budget。

---

## R38-P0-045：writer thread join timeout 必须 fail，而不是并发 drain

当前 `finish()`：

```text
join(timeout=10)
→ 不检查 thread 是否仍 alive
→ drain queue
→ main thread补写
```

如果原 writer 还没退出，可能出现并发消费/写。

必须：

```text
join
→ if any thread alive:
      fatal
      abort generation
      no manual drain write
```

---

## R38-P0-046：scheduler 必须检查 `sink.submit()` 返回值

目前很多地方：

```python
sink.submit(...)
```

不检查 False。

如果：

- queue closed
- timeout
- writer fatal

scheduler 仍可能继续计算。

改成：

```text
submit False
→ WriterFatalError
→ stop new admission
→ abort generation
```

---

# 18. P0：target_concurrency 与 target_cpu_tokens 分开

## R38-P0-047：当前 scheduler 用错字段

动态 concurrency limit 当前主要取：

```text
decision.target_cpu_tokens
```

而不是：

```text
decision.target_concurrency
```

这在 DuckDB / BLAS / Numba 多线程 task 下会错。

正确：

```text
number of running tasks <= target_concurrency

sum(active task cpu_tokens) <= target_cpu_tokens
```

两条独立约束。

---

# 19. P0：backend thread truth 统一

## R38-P0-048：`_engine_threads_for()` 仍是 env/default 猜测

当前 fallback：

```text
DuckDB = 4
Polars = 2
```

这不是实际 engine truth。

### DuckDB

由 `DuckDBLease` 明确给：

```text
threads
memory_limit
temp/spill budget
```

connection/query实际设置与 lease 一致。

### Polars

Polars pool 是 process级能力，启动后不应该假装 task级随 env 改。

启动时读取实际：

```text
polars.thread_pool_size()
```

HostCoordinator 控制：

```text
同时运行几个 Polars workloads
```

而不是伪造 task threads。

### BLAS

通过 `threadpoolctl` 在 task scope 限制。

### Numba

parallel kernel必须申请真实 token。

---

## R38-P0-049：ExecutionTraits 与 physical lowerer 不能互相矛盾

当前 ExecutionTraits 对 SQL/Polars `internal_threads=1`，而 physical lowerer 默认又可能认为 4/2。

建立唯一：

```text
BackendRuntimeProfile
```

所有地方从它读取。

---

# 20. P0：DataReadSession 存在新的并发共享 Store 竞态

## R38-P0-050：不要修改 `store._resolution_cache`

当前 session：

```python
prev = store._resolution_cache
store._resolution_cache = my_cache
...
store._resolution_cache = prev
```

两个线程：

```text
A enter → A cache
B enter → B cache
A exit  → restore None
B still running → cache被A清掉
```

这和之前 service broker module-global 竞态本质相同。

### 修法

把 resolution cache 放到 request context：

```python
@dataclass
class ReadExecutionContext:
    ...
    resolution_cache: dict
```

或：

```text
ContextVar[ResolutionCache]
```

Store prepare 时从 execution context 取，不修改 Store 全局属性。

---

## R38-P0-051：并发 session 测试

Barrier 测试：

```text
thread A session cache=A
thread B session cache=B
A/B overlap
A exit
B继续prepare
→ B仍看到B
```

再测：

- principal isolation
- credential isolation
- snapshot isolation
- cancellation isolation

---

# 21. P1：FastLinearWindowEngine 目前不是真 sliding engine

## R38-P1-052：当前 `_roll_grams()` 每个 t 都重扫窗口

虽然模块名/注释说：

```text
rolling sufficient statistics
O(T*p²)
```

但当前：

```python
for t:
    Xs = X[lo:t+1]
    XtX[t] = Xs.T @ Xs
```

仍然：

```text
O(T * window * p²)
```

而且还分配：

```text
XtX[T,p,p]
```

### 真正实现

joint-valid fast path：

```text
XtX += x_new x_new'
Xty += x_new y_new
yty += y_new²

XtX -= x_old x_old'
Xty -= x_old y_old
yty -= y_old²
```

---

## R38-P1-053：missing pattern 路径

不要因为一个 NaN 就永久退化整段。

可维护每 row contribution：

```text
valid row -> rank-one contribution
invalid row -> zero contribution
```

只要 semantic 是 pairwise/joint finite，就能滑动更新 count。

复杂 mask 才 fallback。

---

## R38-P1-054：不要保存所有 `T×p×p` Gram

按 row streaming solve 或 block solve。

如果只需要 beta：

```text
当前 Gram
→ solve
→ output
→ 下一行
```

---

## R38-P1-055：OLS 不能偷偷变 Ridge

当前：

```python
G = XtX + 1e-12 * I
solve(G, Xty)
```

近奇异情况下与 authoritative `lstsq` 语义不完全一样。

要么：

- 明确 numeric contract；
- condition number gate；
- fallback `lstsq`；
- parity。

---

## R38-P1-056：Ridge intercept 是否正则化必须跟 canonical reference一致

如果 intercept 应不正则化：

```text
diag[0] = 0
```

不要把 implementation convenience 变成语义漂移。

---

# 22. P0/P1：Numba kernels 必须真正接入 operator 主链

## R38-P0-057：standalone benchmark 不等于 FE factor 加速

当前已经有：

```text
NumbaKernelRegistry
```

但像：

- `ts_model/state_space.py`
- `ts_model/ar_meanrev.py`

仍主要直接执行 Python/Numpy循环。

所以：

```text
kernel benchmark 63x
```

不代表：

```text
FactorEngine.run(factor) 63x
```

### 必须接入

cleaned operator authoritative implementation：

```text
if certified Numba kernel available
and dtype/param domain certified
and execution variant allowed
→ numba
else
→ reference
```

---

## R38-P0-058：加速准入 key

至少：

```text
canonical
semantic_version
kernel_version
dtype
parameter domain
missing policy
state/checkpoint version
```

不能“有numba就直接用”。

---

## R38-P1-059：benchmark 改为 end-to-end

测：

```text
FactorEngine compile
→ data
→ operator
→ result
```

比较：

```text
reference route
vs accelerated route
```

记录：

- TTDC
- kernel time
- conversion time
- warm compile
- cold compile

---

# 23. P0：Model-family shared state 出现语义漂移

## R38-P0-060：PCABlock.commonality 与真实 canonical 不一致

当前 shared block 的 `commonality()` 更接近：

```text
reconstruction / current row
```

而真实 panel PCA canonical 的 commonality 是：

```text
1 - Var(resid_i) / Var(ret_i)
```

这是实质语义差异。

如果 planner未来复用 PCABlock，会改变因子。

### 修法

不要在 backend 再复制一套 PCA 数学。

提取唯一：

```python
PCAState.from_window(...)
```

真实 panel operator 和 shared-block 都调用它。

---

## R38-P0-061：PCA active coverage 必须唯一

shared `fit_pca_block(min_obs=2)` 与 panel model 的 window coverage policy 不一致。

必须统一：

```text
PCA_MIN_HISTORY
PCA_MIN_COVERAGE
rank cap
missing imputation
sign orientation
active mask
```

---

## R38-P0-062：PCA sign orientation 必须在 shared state里也一致

不能 reference canonical：

```text
stable instrument ID tie-break
```

shared block却没有。

---

## R38-P0-063：model intermediate identity 补完整 source semantics

至少包含：

```text
input expression identity
source snapshot
universe snapshot
calendar
price basis
window
params
fit cutoff
semantic version
```

不要只依赖一个模糊 input digest + universe string。

---

## R38-P1-064：GARCH 也不要复制两套 fit 实现

`model_family_kernels.py` 和 `ts_model.volatility` 不应各自长期维护 `_fit_garch` 数学。

抽成 authoritative state：

```text
GARCHState / GARCHFitResult
```

所有 outputs共享。

---

# 24. P1：Scheduler 进一步收口

## R38-P1-065：不要等200轮 hot-loop 才决定 shard

如果 ready task 的 P99 已经：

```text
> current safe envelope
```

在第一次 admission 前就应该：

```text
pre-shard
```

不是等200次拒绝。

---

## R38-P1-066：no-progress 应事件等待，不要忙循环

如果拒绝原因是：

```text
外部资源压力
```

且没有 future，应该等待：

- ResourceDecision change event
- lease released condition
- timed backoff

而不是快速200轮。

---

## R38-P1-067：fusion group resource contract 应聚合真实峰值

不能简单逐 root 各 reserve 然后执行 fused query，就默认内存峰值等于 root contract求和或独立值。

需要：

```text
FusionResourceContract
```

根据：

- shared scan
- shared intermediate
- combined output
- backend query profile

校准。

---

# 25. P1：DataAccess resource path进一步完善

## R38-P1-068：remote concurrency 也接 HostCoordinator

DA remote slot：

```text
COS LIST
HEAD
httpfs
```

应向 host IO/remote lane申请 child token。

Host ResourceDecision 的：

```text
remote_concurrency
```

必须真正控制 DA。

---

## R38-P1-069：DuckDB semaphore 与 Host CPU lease绑定

DA DuckDB slot 目前主要限制 query count。

还要保证：

```text
query threads=N
→ HostLease cpu_tokens=N
```

---

## R38-P1-070：ReadPipeline cancellation token贯穿

Host/FE job cancel：

```text
PreparedRead / Stream
```

应能尽快停止：

- remote request
- DuckDB query
- Arrow stream

并 release Host child lease。

---

# 26. R36 Hard Gates 必须重写成真实行为 gate

这部分不是替代 R37 EvidenceTruth，而是针对 R36 已经具体存在的过度声明做回归修正。

## R38-P0-071：AutoShard gate必须执行真实 workload

旧式：

```text
planner返回ShardPlan
```

不能算通过。

新 gate：

```text
whole task峰值 > envelope
→ scheduler自动切N shard
→每个shard真实执行
→merge
→result == full reference
→peak memory < envelope
```

---

## R38-P0-072：FE/DA one-authority gate必须看真实 lease tree

不能只：

```text
apply_da_envelope()返回 applied=True
```

必须：

```text
FE ComputeLease + DA ScanLease
→ 同一 HostResourceCoordinator tree
→ host reserved无double count
→ release归零
```

---

## R38-P0-073：Raw CSE bypass gate不能只 inspect source strings

必须构造 production ctx：

```text
无 buffer store
无 expression cache
```

调用 shared materialize：

```text
必须 fail closed
```

并确保 raw dict没有新增 key。

---

## R38-P0-074：Co-tenancy gate要真的模拟压力 onset + recovery

不能只：

```text
超大contract被拒
```

至少：

```text
运行一批持续task
→ 注入 external memory/cpu/io signals
→ concurrency下降
→ 无OOM
→ 解除压力
→ concurrency自动恢复
```

最好另外做真实 subprocess memory eater nightly。

---

# 27. Latest HEAD CI

## R38-P0-075：最终 SHA 必须有 current CI

当前审计基线：

```text
d34cc9f50df92a10ca5c84a6d64d4682d2cd9f23
```

当前没有可见的 commit status / workflow run 证明。

R38完成后必须：

```text
final_code_sha
==
CI sha
==
R38 evidence sha
```

R35/R36旧报告的旧 SHA 不能证明最终 HEAD。

---

# 28. R37 当前部分实现中已看到的问题——留给 R37，不计入 R38 新问题

本次最新 HEAD 已经出现一部分 R37 parameter-domain scaffold。

其中当前 evidence 显示：

- tested operators仍然很少；
- 一批 `window=500` 被 audit 标为 invalid 但 runtime接受；
- `ts_rank` oracle mismatch；
- `ts_log_return` / `ts_pct` 有 valid point no impl/error；
- 某些 certification key 的 semantic_version/evidence_hash 为空；
- `ts_delay`/`ts_delta` 里存在 lag 与 parameter_point 字段命名映射问题。

这些应该由 R37继续按全参数域/evidence truth整改。

**不要在 R38 另建第二套 ParameterDomainStore。**

---

# 29. 推荐新增/重构模块

只在没有等价现有职责时新增。

```text
factor_engine/runtime/
    resource_autopilot_service.py
    task_run_observation.py
    shard_execution_plan.py
    shard_executor.py
    shard_merger.py
    spill_store.py
    backend_runtime_profile.py
    memory_budget_allocator.py

dataaccess/runtime/
    host_resource_bridge.py
```

已有模块应直接重构：

```text
runtime/host_resource_coordinator.py
runtime/resource_autopilot.py
runtime/resource_broker.py
runtime/resource_calibration_store.py
runtime/auto_shard_planner.py
runtime/buffer_store.py
runtime/adaptive_batch_scheduler.py
runtime/streaming_result_sink.py
runtime/batch_service.py
cache/session.py
backend/context.py
backend/fast_linear_window.py
backend/model_family_kernels.py
backend/numba_kernel_registry.py
service/queue.py

dataaccess/runtime/resource_governor.py
dataaccess/runtime/read_pipeline.py
dataaccess/runtime/prepared_read.py
dataaccess/read/read_session.py
```

---

# 30. 具体执行顺序

## Phase 0：先冻结当前事实

生成：

```text
R38_BASELINE.json
R38_RUNTIME_WIRING_AUDIT.json
R38_R35_R36_CLAIM_VS_RUNTIME.csv
```

字段：

```text
claim
claimed_closed_in
actual_runtime_path
wired
behavior_test_exists
status
```

---

## Phase 1：修真实 AutoShard + OOM replan

先做：

- ShardExecutionPlan
- shard child tasks
- merge
- overlap
- state checkpoint
- OOM smaller-shape replan

这是最危险的 P0。

---

## Phase 2：修真实 calibration

- TaskRunObservation
- actual elapsed
- actual output bytes
- real/qualified memory observation
- calibration schema bump

---

## Phase 3：Host lease tree重构

- parent/child no double count
- CPU/IO/spill
- per-job lease
- recursive release
- pure summary

---

## Phase 4：FE×DA resource bridge

- DA Host child lease
- PreparedRead lease ref
- standalone DA safe cap
- setter locking
- cancellation

---

## Phase 5：ResourceAutopilotService

- fixed cadence
- one tick source
- atomic decision
- scheduler read-only

---

## Phase 6：Buffer/cache/spill

- remove raw fallback
- handle put refusal
- single L0 owner
- real SpillStore
- reload/recompute
- reconciliation

---

## Phase 7：dynamic wave / sink

- JIT read wave
- elastic queue
- fatal propagation
- writer join lifecycle

---

## Phase 8：R35 fast kernels真正接主链

- FastLinear true sliding
- Numba dispatch
- PCA/GARCH authoritative shared state
- end-to-end benchmark

---

## Phase 9：行为级 hard gates

重跑：

- resource
- co-tenancy
- auto-shard
- OOM
- DA bridge
- cache
- spill
- Numba
- model family

---

## Phase 10：current SHA CI

最终生成 R38 evidence。

---

# 31. 必须新增测试矩阵

建议：

```text
factor_engine/tests/r38/
    test_real_auto_shard_elementwise.py
    test_real_auto_shard_rolling_overlap.py
    test_real_auto_shard_cross_section.py
    test_real_auto_shard_stateful_checkpoint.py
    test_real_auto_shard_pca_full_universe.py
    test_oom_replan_smaller_shape.py
    test_no_same_shape_oom_retry.py

    test_task_calibration_actual_elapsed.py
    test_task_calibration_actual_output_bytes.py
    test_calibration_does_not_use_predicted_peak_as_actual.py

    test_host_lease_parent_child_no_double_count.py
    test_host_lease_cpu_io_spill.py
    test_host_lease_recursive_release.py
    test_multijob_independent_job_leases.py

    test_fe_da_same_host_lease_tree.py
    test_da_standalone_safe_cap.py
    test_da_dynamic_cap_locking.py

    test_resource_controller_fixed_cadence.py
    test_multi_scheduler_does_not_multitick_controller.py
    test_resource_pressure_shrink_and_time_based_recovery.py
    test_budget_sum_le_safe_envelope.py

    test_buffer_put_refusal_handled.py
    test_no_production_raw_cse_fallback.py
    test_spill_reload_roundtrip.py
    test_spill_checksum_failure.py
    test_buffer_reconciliation_large_keyset.py

    test_dynamic_read_wave_repartition.py
    test_dynamic_sink_shrink.py
    test_writer_alive_after_join_is_fatal.py
    test_sink_submit_false_aborts_scheduler.py

    test_datareadsession_concurrent_cache_isolation.py

    test_fast_linear_true_sliding_parity.py
    test_fast_linear_complexity_regression.py
    test_numba_end_to_end_operator_dispatch.py
    test_pca_shared_state_parity.py
    test_garch_shared_state_parity.py
```

---

# 32. 真实 AutoShard parity cases

## Rolling

随机：

```text
window = 5 / 20 / 120 / 252
shard boundaries随机
NaN block随机
```

比较：

```text
full
vs
asset shard
vs
time shard + overlap
```

---

## Cross-section

对每个 date：

```text
full universe rank
==
time shard rank
```

并明确测试：

```text
asset shard rank != full
```

因此 asset shard 必须被 gate 拒绝。

---

## Stateful

```text
full history
==
asset shard
==
checkpoint time shard
```

无 checkpoint：

```text
time shard fail closed
```

---

# 33. Resource Controller stress

## Case 1：空闲服务器

要求：

```text
slow up
→ 接近安全CPU额度
```

## Case 2：外部CPU开始

要求：

```text
500ms~2s内开始收缩
```

## Case 3：外部CPU退出

要求：

```text
按时间慢慢恢复
```

不能因为 scheduler loop 很快而几十毫秒恢复。

## Case 4：多 job 同时运行

controller tick count：

```text
≈ elapsed / interval
```

而不是：

```text
≈ 各job scheduler loops之和
```

---

# 34. Host lease invariant

所有时刻：

```text
root_reserved_memory <= safe_memory
leaf_cpu_tokens <= target_cpu_tokens
io_tokens <= io_capacity
spill_reserved <= usable_spill
```

如果 root accounting 模型：

```text
children之和 <= parent
但children不再重复计host memory
```

---

# 35. DataAccess 并发 session 测试

两个 `DataReadSession`：

```text
A principal A / cache A
B principal B / cache B
```

用 barrier overlap：

```text
A enter
B enter
A read
B read
A exit
B继续read
```

要求：

- B cache仍为B；
- B principal/security context不变；
- resolution cache hit不跨 session；
- Store没有被 session覆盖全局字段。

---

# 36. FastLinear benchmark

至少：

```text
T = 500 / 2500 / 10000
window = 20 / 120 / 252
p = 1 / 4 / 16
missing = 0 / 5% / block missing
```

对比：

```text
reference rescan
true sliding engine
```

必须：

```text
semantic parity先过
再报告speedup
```

测实际 complexity slope，而不是单一尺寸。

---

# 37. Numba end-to-end benchmark

不要只：

```text
kernel.call()
```

还要：

```text
FactorEngine.run(...)
```

证明 runtime path里：

```text
accelerated_kernel_used = true
```

并输出：

```text
operator
kernel
param point
dtype
rows
reference_ms
accelerated_ms
speedup
conversion_ms
compile_warmup_ms
```

---

# 38. R38 Hard Gates

建议至少：

```text
R38_REAL_AUTOSHARD_EXECUTION_PASS
R38_ROLLING_SHARD_OVERLAP_PARITY
R38_CROSS_SECTION_ZERO_ASSET_SHARD
R38_STATEFUL_SHARD_CHECKPOINT_PARITY
R38_OOM_REPLAN_TO_SMALLER_SHAPE
R38_ZERO_SAME_SHAPE_OOM_RETRY

R38_CALIBRATION_ELAPSED_IS_DURATION
R38_CALIBRATION_PEAK_IS_OBSERVED
R38_CALIBRATION_OUTPUT_BYTES_ACTUAL

R38_HOST_LEASE_NO_PARENT_CHILD_DOUBLE_COUNT
R38_HOST_LEASE_CPU_LIMIT_ENFORCED
R38_HOST_LEASE_IO_LIMIT_ENFORCED
R38_HOST_LEASE_SPILL_LIMIT_ENFORCED
R38_HOST_LEASE_RECURSIVE_RELEASE
R38_MULTI_JOB_LEASE_ISOLATION

R38_FE_DA_SAME_HOST_LEASE_TREE
R38_DA_STANDALONE_MEMORY_BOUNDED
R38_DA_DYNAMIC_LIMIT_THREAD_SAFE

R38_RESOURCE_CONTROLLER_SINGLE_FIXED_CADENCE
R38_RESOURCE_RECOVERY_TIME_BASED
R38_RESOURCE_BUDGET_SUM_WITHIN_SAFE_ENVELOPE

R38_BUFFER_PUT_REFUSAL_NOT_SILENT
R38_ZERO_PRODUCTION_RAW_CSE_FALLBACK
R38_REAL_SPILL_RELOAD_PASS
R38_BUFFER_ACCOUNTING_RECONCILES

R38_DYNAMIC_READ_WAVE_SHRINK
R38_DYNAMIC_SINK_SHRINK
R38_WRITER_LIVE_THREAD_FATAL
R38_SINK_FAILURE_STOPS_ADMISSION

R38_DATAREADSESSION_CONCURRENT_ISOLATION

R38_FAST_LINEAR_TRUE_SLIDING
R38_FAST_LINEAR_REFERENCE_PARITY
R38_NUMBA_END_TO_END_DISPATCH
R38_NUMBA_END_TO_END_PARITY
R38_PCA_SHARED_STATE_PARITY
R38_GARCH_SHARED_STATE_PARITY

R38_CURRENT_HEAD_CI_PASS
```

---

# 39. R38 Evidence Artifact

```text
factor_engine/docs/evidence/r38/
    R38_BASELINE.json
    R38_HEAD.json
    R38_R35_R36_CLAIM_VS_RUNTIME.csv
    R38_RUNTIME_WIRING_AUDIT.json

    R38_AUTOSHARD_PARITY.json
    R38_OOM_REPLAN_TRACE.json

    R38_TASK_CALIBRATION_VALIDATION.json
    R38_HOST_LEASE_TREE_STRESS.json
    R38_FE_DA_RESOURCE_BRIDGE.json
    R38_RESOURCE_CONTROLLER_TRACE.json
    R38_COTENANCY_RECOVERY.json

    R38_BUFFER_SPILL_VALIDATION.json
    R38_DATAREADSESSION_CONCURRENCY.json

    R38_FAST_LINEAR_BENCHMARK.json
    R38_NUMBA_END_TO_END_BENCHMARK.json
    R38_MODEL_FAMILY_SHARED_STATE_PARITY.json

    R38_HARD_GATES.json
    R38_FINAL_ACCEPTANCE_REPORT.md
```

---

# 40. Claim-vs-runtime ledger

R35/R36 每个已经宣称完成的条目必须补一张表：

```text
claim_id
claim_text
report
real_runtime_entrypoint
real_consumer
behavior_test
synthetic_only?
production_path?
final_status
```

例：

```text
R36 auto-shard
claimed = TRUE
runtime = scheduler._auto_shard_replan
before R38 = contract-only
after R38 = real shard children + merge
behavior test = ...
```

这张表能防止以后再出现：

```text
模块存在
→ gate TRUE
→ 报告写已完成
```

但生产主链实际上没用的问题。

---

# 41. 不允许的整改方式

## 41.1 只改 hard gate

禁止。

## 41.2 AutoShard 继续只改 peak_memory 数字

禁止。

## 41.3 OOM 原 task retry

禁止。

## 41.4 把 predicted memory 当 actual memory

禁止。

## 41.5 让 HostCoordinator 只做 envelope setter

不算 FE×DA统一资源。

## 41.6 production 保留 raw shared cache写入

禁止。

## 41.7 “spill = drop”

禁止把 drop 叫 spill。

## 41.8 只 benchmark standalone Numba函数

不能证明 FactorEngine 被加速。

## 41.9 FastLinear 每行重扫window但仍宣称 sliding O(T)

禁止。

## 41.10 复制第二套 PCA/GARCH数学

必须共享 authoritative state。

---

# 42. 最终应形成的资源执行架构

```text
                   HostResourceCoordinator
                           │
          ┌────────────────┼────────────────┐
          │                │                │
      JobLease         JobLease         JobLease
          │
    ┌─────┼───────────────┬──────────────┐
    │     │               │              │
Compute  DA Scan        Cache         Writer
Lease    Lease           Lease         Lease
    │      │               │              │
Numba   DuckDB         Buffer/Spill     Sink
BLAS    Remote
```

资源 controller：

```text
ResourceMonitor
    ↓ fixed cadence
ResourceController
    ↓
Atomic ResourceDecision
    ↓
Schedulers / DA / Writer read-only consume
```

---

# 43. 最终应形成的真实 shard 执行链

```text
Oversized Task
    ↓
AutoShardPlanner
    ↓
ShardExecutionPlan
    ↓
shard 0 ─┐
shard 1 ─┼─→ streaming merge / materialize
shard 2 ─┤
...      ┘
    ↓
parity / DQ
```

而不是：

```text
Oversized Task
    ↓
change predicted memory number
    ↓
execute same oversized task
```

---

# 44. 推荐最终主 API

用户侧不应该手调很多 knob。

```python
engine.materialize_many_auto(
    factors,
    source=...,
    priority="standard",
)
```

内部自动：

```text
compile
→ predict resource
→ host job lease
→ DA child lease
→ JIT read waves
→ native compute
→ auto shard when needed
→ governed buffers/spill
→ streaming writer
→ release all leases
```

可选 ceiling：

```python
max_cpu=...
max_memory=...
```

只是上限，不是性能调参。

---

# 45. R38 最终验收场景

## Scenario A：16GB机器上跑本来估计20GB的rolling batch

必须：

```text
pre-shard
→ 真正切数据
→ peak<safe
→ parity
→ no OOM
```

---

## Scenario B：预测错了，真实执行才OOM

必须：

```text
OOM
→ calibration tail up
→ smaller shard
→ retry
→ success
```

不能相同 task 再跑。

---

## Scenario C：两个 service jobs并发

必须：

```text
各自JobLease
→ 同HostCoordinator
→ 不互相覆盖memory ceiling
→ child lease不double count
```

---

## Scenario D：FactorEngine + DataAccess 同时占资源

必须：

```text
FE compute
+
DA scan/DuckDB
```

在同一 host lease tree里可见。

---

## Scenario E：外部算法突然吃掉一半内存

必须：

```text
controller fixed cadence发现
→ shrink
→ JIT wave变小
→ sink target变小
→ no OOM
→ 外部算法退出
→ slow recovery
```

---

## Scenario F：CSE buffer超预算

必须：

```text
put被拒
→ spill/recompute/fail-closed明确处理
```

不能 return success 后 plan_ref KeyError。

---

## Scenario G：两个 DataReadSession overlapping

必须完全隔离 resolution cache。

---

## Scenario H：PCA family multi-output

shared-state route 与每个 canonical authoritative reference 完全一致。

---

## Scenario I：Numba

实际 FactorEngine execution trace 显示：

```text
numba_kernel_used
```

并且 parity + speedup都通过。

---

# 46. 最终 coding agent 交付格式

完成后必须回答：

```text
1. Baseline SHA
2. Final SHA
3. R38 changed files
4. R38 issue closure ledger
5. 哪些R35/R36 claim此前属于partial wiring
6. real AutoShard结果
7. OOM replan结果
8. actual calibration验证
9. Host lease tree验证
10. FE×DA child lease验证
11. resource controller fixed-cadence stress
12. buffer/spill结果
13. DataReadSession concurrency结果
14. FastLinear end-to-end benchmark
15. Numba end-to-end benchmark
16. PCA/GARCH shared state parity
17. current-head CI
18. remaining issues
```

如果仍有：

```text
DEFERRED
PARTIAL
SYNTHETIC_ONLY
NOT_WIRED
```

必须明确写，不得写“全部完成”。

---

# 47. 最终优先级

```text
P0-第一组：
真实AutoShard
OOM replan
真实calibration

P0-第二组：
Host lease tree
FE×DA child lease
service JobLease
fixed-cadence controller

P0-第三组：
BufferStore refusal
raw bypass
real spill
dynamic wave/sink
DataReadSession race

P0/P1-性能：
Numba主链接入
PCA/GARCH共享语义
FastLinear真正sliding

最后：
behavior gates
current SHA CI
```

---

# 48. R38 完成后的标准

R38 不是要继续“增加模块”。

目标是把当前已经有的模块从：

```text
有类
有测试
有报告
```

推进到：

```text
真实生产主链正在使用
+
失败路径也符合合同
+
资源账本与实际资源一致
+
benchmark测到的是FactorEngine真正执行路径
+
行为级测试能够证明。
```

R38完成后，再结合 R37 的语义/PIT/参数域/证据全量整改，FactorEngine + DataAccess 才能真正进入下一轮“少做大重构、以生产稳定性和性能回归为主”的阶段。
