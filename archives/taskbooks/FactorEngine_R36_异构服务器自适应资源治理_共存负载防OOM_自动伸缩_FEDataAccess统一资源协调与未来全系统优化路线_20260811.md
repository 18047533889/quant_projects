# FactorEngine R36｜异构服务器自适应资源治理、共存负载防 OOM、自动伸缩、FE×DataAccess 统一资源协调与未来全系统优化路线

> 审计与设计基线
>
> - Repository: `18047533889/quant_projects`
> - Branch: `main`
> - Fresh HEAD: `8449d9c253c55308f3e406a15739e984387e9691`
> - 日期：2026-08-11
> - 关联整改：
>   - R33：FactorEngine × DataAccess Unified QueryGraph、批量扫描复用、DuckDB/Polars/FactorBlock/COW、TimeToDurableCommit
>   - R34：全算子真实正确性、PIT、参数域、backend variant、证据体系
>   - R35：模型算子、Numba、线程/进程、模型 family kernel、测试 obligation matrix
> - R36 定位：**把 FactorEngine 从“有资源治理模块”升级为真正能在不同服务器、不同容器配额、同机共存其他算法时自动榨取空闲资源，同时以 fail-safe 方式避免 OOM、磁盘打满、CPU/IO 争抢，并自动恢复速度的生产级 Resource Autopilot。**
>
> 本文同时列出本轮在当前 HEAD 新确认的具体实现问题，以及未来完整系统架构。凡标为“当前确认”的条目必须先修；凡标为“目标架构”的条目用于最终收口。

---

# 0. 最终目标

实际运行环境不是：

```text
固定 128GB RAM
固定 32 cores
服务器只跑 FactorEngine
```

而可能是：

```text
Server A:
    16GB RAM
    8 CPU

Server B:
    64GB RAM
    32 CPU

Server C:
    256GB RAM
    64 CPU

同一时刻：
    FactorEngine materialization
    + AlphaProbe / FactorMiner
    + 模型训练
    + 其他研究任务
    + DataAccess / DuckDB
```

因此 FactorEngine 不能要求运维人员手工调：

```text
n_jobs
DuckDB threads
Polars threads
read wave
cache size
result queue
factor block size
spill size
```

最终应该：

```text
启动时自动识别服务器硬上限
        ↓
运行时持续识别当前真实空闲资源
        ↓
预测 ready tasks 的 P95/P99 资源需求
        ↓
动态选择 concurrency / threads / wave / block / shard
        ↓
压力上升立即让路
        ↓
压力下降后逐渐自动升速
        ↓
始终保留 emergency memory reserve
        ↓
达到尽可能低的 TimeToDurableCommit
```

核心原则：

> **Fast when idle, polite when contended, fail-safe before OOM.**

---

# 1. 当前资源治理基础已经具备什么

当前 FactorEngine 已经具备不少正确基础，不能推倒重写。

## 1.1 Hard resource discovery

现有 `resource_governor.py` 已能识别：

```text
cgroup v2 memory.max / cpu.max
cgroup v1 memory limit
SLURM
RLIMIT
sched affinity
host RAM / CPU
```

并取最严格 hard limit。

这是正确的。

---

# 2. 当前 ResourceBroker 已有的正确基础

现有：

```text
ResourceSnapshot
ResourceBroker
ReservationLease
TaskResourceContract
```

能观测：

```text
host MemAvailable
cgroup memory.current
process family RSS
process family PSS
system CPU utilization
own CPU utilization
external CPU utilization
loadavg
spill free disk
disk busy
```

并做：

```text
CPU token
IO token
memory admission
spill admission
pressure stage
```

这些都应保留。

---

# 3. 当前压力档位方向正确

目前：

```text
NORMAL
PRESSURE_1
PRESSURE_2
PRESSURE_3
PRESSURE_4
CRITICAL
```

方向正确。

但当前更多是：

```text
threshold-based passive throttle
```

最终要变成：

```text
closed-loop adaptive resource controller
```

---

# 4. R36-P0-001｜`max_concurrency` 当前实际上没有真正限制 Scheduler 并发【当前确认】

`AdaptiveBatchScheduler.__init__` 接受：

```python
max_concurrency
```

并保存：

```python
self.max_concurrency = max_concurrency
```

但当前 run loop 没有把：

```text
len(futures) < max_concurrency
```

作为 admission 条件。

因此：

```text
run_many_parallel(n_jobs=4)
```

不能仅因为传入 4 就认为 scheduler 同时最多运行 4 个 task。

## 修复

每轮 admission 前：

```python
hard_target = min(
    user_max_concurrency or INF,
    broker.dynamic_target_concurrency(),
)
```

只允许：

```python
running_tasks < hard_target
```

---

# 5. R36-P0-002｜`recommended_concurrency()` 当前没有被 Scheduler 消费【当前确认】

`ResourceBroker` 已实现：

```python
recommended_concurrency()
```

但 scheduler 当前没有调用它控制实际 admission。

也就是说：

```text
broker算出了建议值
≠
scheduler实际遵守
```

## 修复

把它升级为：

```python
ResourceDecision
```

每一个 control tick 返回：

```python
ResourceDecision(
    target_concurrency,
    target_cpu_tokens,
    target_io_concurrency,
    target_read_wave_bytes,
    target_result_queue_bytes,
    target_factor_block_size,
    memory_pressure,
    cpu_pressure,
    io_pressure,
)
```

Scheduler必须消费。

---

# 6. R36-P0-003｜当前动态 CPU 只有“降速”，没有“恢复后自动升速”【当前确认】

当前 scheduler在：

```text
PRESSURE_1 / PRESSURE_2
```

调用：

```python
lower_soft_cpu_budget(factor=0.6)
```

但正常状态恢复后没有：

```text
restore / raise
```

因此可能出现：

```text
下午 14:00 另一个算法抢 CPU
→ FE soft budget 降到 60%

14:20 另一个算法结束
→ FE仍然停留在 60%
```

## 修复

必须是双向 controller：

```text
pressure ↑
→ fast multiplicative decrease

pressure ↓ 且稳定 N samples
→ slow additive increase
```

推荐：

```text
AIMD + hysteresis
```

例如：

```text
pressure:
    target *= 0.5

stable:
    target += 1
```

而不是一次性写死 0.6。

---

# 7. R36-P0-004｜`recommended_concurrency` 的 2GB/worker 固定模型过粗【当前确认】

当前：

```text
by_mem = headroom // 2GB
```

但：

```text
cheap elementwise factor
```

和：

```text
5000 stocks × 10年 × PCA/GARCH
```

峰值内存完全不同。

## 修复

改为：

```text
ready task predicted P99
```

逐 task决定能否共同 admission。

---

# 8. R36-P0-005｜ExecutionResourcePlan 默认每 worker 3GB 只能作为 Cold Start【当前确认】

当前：

```text
per_worker_peak = 3GB
```

可以作为未知 workload 首次执行的 conservative floor。

但生产长期运行后必须替换为：

```text
per operator family
per backend
per shape
per parameter domain
```

的学习值。

---

# 9. 资源预测必须从单值变成 Distribution

当前：

```text
predicted_peak_memory_bytes × uncertainty
```

最终应保存：

```text
P50
P90
P95
P99
max observed
sample_count
```

生产 admission：

```text
known workload:
    P99

cold workload:
    static estimate × high uncertainty
```

---

# 10. ResourceShapeKey

建议：

```python
ResourceShapeKey(
    canonical_family,
    backend,
    execution_variant,
    rows_bucket,
    instruments_bucket,
    window_bucket,
    input_count,
    dtype,
    representation,
    group_count_bucket,
)
```

不要按：

```text
operator name
```

一个值永久估计。

---

# 11. 真实资源结果反哺模型

每 task完成后记录：

```text
actual_elapsed_ms
actual_peak_rss_delta
actual_peak_pss_delta
actual_output_bytes
actual_spill_bytes
actual_read_bytes
actual_write_bytes
```

更新：

```text
ResourceCalibrationStore
```

---

# 12. R36-P0-006｜当前 `adapt_uncertainty()` 尚未形成完整闭环【当前确认】

ResourceBroker有：

```python
adapt_uncertainty(
    underpredict_streak,
    overpredict_streak
)
```

但当前 scheduler主路径没有建立：

```text
predicted peak
vs
actual task peak
```

的完整反馈链。

## 修复

每个 execution variant单独：

```text
prediction_error = actual / predicted
```

更新：

```text
P99 correction factor
```

---

# 13. 不能只测 Process RSS

未来使用：

```text
ProcessPool
worker processes
DuckDB child/helper
其他 FE subprocess
```

时必须监控：

```text
process family PSS
```

而不是只看主 process RSS。

---

# 14. R36-P0-007｜MemoryGovernor 的压力阶段目前主要看当前主进程 RSS【当前确认】

虽然代码已有：

```text
process_family_rss_bytes()
```

但 `MemoryGovernor._current_rss()` 当前优先：

```text
psutil.Process().rss
```

未来 process lane一旦真正启用：

```text
child workers
```

的内存会被漏掉。

## 修复

统一：

```text
process_family_pss
```

优先。

不可用才：

```text
process_family_rss
```

再不可用：

```text
self RSS
```

---

# 15. 为什么 PSS 比单纯 RSS 更好

跨进程共享：

```text
shared memory
mmap
Arrow buffers
shared libraries
```

时把每个 process RSS简单相加会重复计算共享页。

Linux可读 PSS时：

```text
PSS
```

更接近真实物理占用。

---

# 16. Hard / Soft / Emergency 三层内存 Envelope

不要只：

```text
memory_limit
```

定义：

```text
HardEnvelope
SoftEnvelope
EmergencyReserve
```

## HardEnvelope

```text
min(
    cgroup memory.max,
    SLURM,
    RLIMIT,
    explicit hard cap,
    host physical
)
```

永远不能超过。

## EmergencyReserve

为：

```text
Python allocator burst
DuckDB non-buffer-managed memory
writer burst
kernel temporary
OS
```

预留。

## SoftEnvelope

FactorEngine希望使用的动态预算。

---

# 17. Live Safe Envelope

核心公式：

```text
HostAvailable(t)
CgroupAvailable(t)
ConfiguredAvailable(t)
```

得到：

```text
LiveHeadroom(t)
```

再：

```text
SafeHeadroom(t)
=
LiveHeadroom(t)
- EmergencyReserve
- UntrackedNativeReserve
- PredictedWriterBurst
```

---

# 18. EmergencyReserve 不能只用固定百分比

最终：

```text
max(
    absolute_floor,
    fraction_of_hard_memory,
    observed_native_tail,
    observed_writer_tail,
)
```

学习服务器实际表现。

---

# 19. DuckDB 需要额外 Native Reserve

官方 DuckDB文档明确指出：

```text
memory_limit
```

只约束 buffer manager；
部分：

```text
vectors
query results
complex aggregate states
```

会在其外分配。

因此：

```text
DuckDB memory_limit
!=
process maximum memory
```

R36必须额外预留：

```text
DuckDBOutsideBufferReserve
```

官方参考：

- https://duckdb.org/docs/stable/configuration/pragmas
- https://duckdb.org/docs/current/guides/performance/oom
- https://duckdb.org/docs/lts/operations_manual/limits

---

# 20. DuckDB Memory Lease

不要：

```text
DuckDB memory_limit = 45% process
```

永远固定。

每个 DuckDB workload：

```python
DuckDBLease(
    buffer_memory_limit,
    outside_buffer_reserve,
    threads,
    temp_disk_budget,
)
```

---

# 21. DuckDB Threads 应占真实 CPU Tokens

如果：

```text
DuckDB SET threads=8
```

task不能只：

```text
cpu_tokens=1
```

必须：

```text
cpu_tokens >= actual DuckDB execution threads
```

否则资源账本失真。

---

# 22. R36-P0-008｜当前大量 stage contract 仍可能 backend_threads=1【当前确认】

TaskResourceContract支持：

```text
backend_threads
```

但现有 lowerer/contract多个路径仍默认：

```text
cpu_tokens=1
backend_threads=1
```

必须让：

```text
actual engine threads
```

和：

```text
admission tokens
```

一致。

---

# 23. Polars Thread Pool 不适合运行中随 task改

Polars官方文档说明：

```text
POLARS_MAX_THREADS
```

要在 process启动前设置；
线程池建立后不能运行中修改。

因此不能设计：

```text
task A -> Polars threads=8
task B -> Polars threads=2
```

在同一个已 import Polars 的 Python process里动态切换。

官方参考：

- https://docs.pola.rs/api/python/stable/reference/api/polars.thread_pool_size.html

---

# 24. Polars 正确的动态资源方式

有两条路线。

## 路线 A｜固定 Polars pool，调并发任务数

例如：

```text
Polars process pool size = 8
```

外部 ResourceController控制：

```text
同时允许几个 Polars query
```

## 路线 B｜Resource-Class Worker Pools

启动多个 long-lived worker：

```text
PolarsWorker_1T
PolarsWorker_2T
PolarsWorker_4T
PolarsWorker_8T
```

根据当前资源选择。

第一阶段推荐 A。

---

# 25. Polars 内存紧张时优先 Streaming

Polars官方 streaming engine：

```text
batch processing
```

可减少 memory pressure，
并支持很多 larger-than-memory workload。

官方参考：

- https://docs.pola.rs/user-guide/concepts/streaming/

---

# 26. Polars Streaming 不能无条件相信

官方 API 说明：

```text
selected engine不能处理时可能 fallback
```

因此 production memory constrained route必须：

```text
检测 physical plan / fallback
```

不能：

```text
以为 streaming
实际 full in-memory
```

然后 OOM。

---

# 27. Polars Chunk Size 也要 Resource-Aware

官方 `set_streaming_chunk_size` 明确说明：

默认 chunk size在部分数据上可能过于乐观并导致 OOM。

因此：

```text
chunk size
```

应由 Resource Autopilot基于：

```text
row width
thread count
SafeEnvelope
```

动态决定。

官方参考：

- https://docs.pola.rs/api/python/stable/reference/api/polars.Config.set_streaming_chunk_size.html

---

# 28. Linux PSI 应进入 Resource Autopilot

仅看：

```text
CPU percent
MemAvailable
```

不够。

Linux PSI提供：

```text
/proc/pressure/cpu
/proc/pressure/memory
/proc/pressure/io
```

以及 cgroup级：

```text
cpu.pressure
memory.pressure
io.pressure
```

它反映：

```text
实际 stall
```

比：

```text
CPU utilization高
```

更接近“同机是否真的争抢”。

官方参考：

- https://docs.kernel.org/accounting/psi.html

---

# 29. ResourceSnapshot 2.0

增加：

```python
cpu_psi_some_avg10
cpu_psi_full_avg10

memory_psi_some_avg10
memory_psi_full_avg10

io_psi_some_avg10
io_psi_full_avg10

memory_events_low
memory_events_high
memory_events_oom
memory_events_oom_kill

swap_current
swap_max

page_fault_rate
major_fault_rate
```

---

# 30. 为什么 PSI 很适合用户当前服务器场景

如果另一个算法：

```text
占大量 RAM但暂时没有 OOM
```

仅看 FE自身 RSS未必能立即识别。

但：

```text
MemAvailable下降
memory PSI上升
```

会非常明显。

如果另一个算法：

```text
把 NVMe打满
```

FE自身 CPU可能很低，
但：

```text
io PSI
```

会升高。

所以可以：

```text
自动让路
```

---

# 31. R36-P0-009｜当前 disk_busy 用固定 500MB/s 归一化【当前确认】

当前：

```text
500 MB/s = busy 1.0
```

但真实服务器可能：

```text
网络盘 100MB/s
SATA SSD 500MB/s
NVMe 3GB/s
RAID 8GB/s
```

固定 500MB/s 会误判。

## 修复

启动时做轻量：

```text
storage calibration
```

记录：

```text
sequential read
sequential write
random metadata latency
```

并持续用真实 latency/PSI校正。

---

# 32. 不需要每次启动跑大型磁盘 benchmark

只做：

```text
小样本安全 calibration
```

并缓存：

```text
hardware fingerprint
```

结果。

---

# 33. HardwareFingerprint

建议：

```python
HardwareFingerprint(
    hostname_hash,
    cpu_model,
    cpu_quota,
    affinity_count,
    ram_limit,
    numa_nodes,
    storage_device,
    storage_fs,
    duckdb_version,
    polars_version,
    numpy_version,
    blas_backend,
)
```

---

# 34. 同一型号服务器共享校准

ResourceCalibrationStore key：

```text
hardware fingerprint
+
software version
+
kernel shape
```

---

# 35. R36-P0-010｜4GB ReadWave 固定默认值不适合异构服务器【当前确认】

当前 scheduler：

```text
wave_memory_budget = 4GB
```

8GB/16GB机器 + 同机其他算法时过大。

256GB服务器又可能过于保守。

## 修复

```text
target_read_wave_bytes
=
min(
    calibrated_scan_optimum,
    SafeEnvelope × wave_fraction,
    current job lease,
)
```

---

# 36. ReadWave Budget 动态变化

例如：

```text
SafeEnvelope 80GB
→ wave 8GB

外部模型训练启动
SafeEnvelope 15GB
→ 后续 wave 1GB

模型结束
SafeEnvelope 70GB
→ 后续 wave逐步升到 6~8GB
```

---

# 37. ReadWave 不应在已执行 wave中途强拆

原则：

```text
running wave不 kill
future wave重新 plan
```

避免：

```text
复杂 rollback
```

---

# 38. R36-P0-011｜4GB Result Queue 固定默认同样不合理【当前确认】

`StreamingResultSink` 当前默认：

```text
queue_bytes = 4GB
```

应变成：

```text
dynamic_sink_budget
```

---

# 39. ResultQueue Auto Budget

例如：

```text
min(
    SafeEnvelope * 0.05,
    job_memory_lease * 0.10,
    configured_cap
)
```

并设置：

```text
absolute min
absolute max
```

---

# 40. R36-P0-012｜Scheduler 当前没有消费 `backpressure_ratio`【当前确认】

当前 queue自己会阻塞 producer，
但 scheduler并没有：

```text
writer backlog高
→ 减少新 compute
```

## 修复

```text
if sink.backpressure_ratio > 0.70:
    reduce compute admission

> 0.90:
    stop non-critical compute admission
```

恢复后再增速。

---

# 41. Compute / IO / Write 三阶段联合调度

最终：

```text
Read N+1
   ||
Compute N
   ||
Write N-1
```

但必须让：

```text
三者共享统一 resource authority
```

不能每层各自开满。

---

# 42. R36-P0-013｜结果队列使用 list + pop(0)【当前确认】

当前：

```python
self._items: list
pop(0)
```

大队列下是：

```text
O(n)
```

改：

```python
collections.deque
```

---

# 43. R36-P0-014｜Writer 失败当前可能无限重入队或收尾静默丢失【当前确认】

writer线程异常：

```text
重新入队
```

finish补写失败：

```text
except Exception:
    pass
```

这不符合 production durable materialization。

## 修复

Writer状态机：

```text
RUNNING
RETRYING
FAILED_FATAL
DRAINING
COMMITTED
```

超过 retry policy：

```text
batch失败
generation不得 commit
```

---

# 44. 写失败不能靠“内存里继续攒”

如果磁盘故障：

```text
writer一直失败
```

queue不能持续积累直到 OOM。

应：

```text
fatal writer state
→ stop new compute admission
→ drain/cancel safely
→ fail generation
```

---

# 45. R36-P0-015｜Parallel materialization 当前会让整批结果驻留【当前确认】

当前 config parallel batch：

```text
run_many_parallel
→ entire batch results
→ 然后逐 factor materialize
```

对于：

```text
1000 factors
```

这是大内存风险。

## 修复

parallel路径也必须：

```text
compute block
→ DQ
→ write
→ release
```

---

# 46. Production 默认 Result Policy

如果：

```text
estimated_result_bytes
```

超过：

```text
result_memory_budget
```

自动从：

```text
return
```

切：

```text
sink/materialize
```

不要要求调用者知道该选哪个。

---

# 47. 只有显式用户请求结果 DataFrame 时才允许 return-all

例如研究环境：

```text
run_many(...)
```

可以 return。

生产落值：

```text
默认 stream/sink
```

---

# 48. FactorBlock 是比单 Factor Result 更好的内存单位

内部：

```text
32 / 64 / 128 factors
```

一块。

优势：

```text
减少 Python objects
减少 writer open
减少 metadata
提高 columnar write
```

block size由资源 controller动态决定。

---

# 49. FactorBlockSize 动态公式

```text
target_block_bytes
=
min(
    writer optimum,
    SafeEnvelope * result_fraction,
    cache lifetime allowance
)
```

再：

```text
factor width estimate
→ number of factors
```

---

# 50. 当前 FE 有多套 Memory Governance，需要统一

目前至少：

```text
ResourceBroker
MemoryGovernor
ExecutionResourcePlan
service broker
```

DataAccess另有：

```text
GlobalResourceGovernor
```

这不是最终理想状态。

---

# 51. R36-P0-016｜FE 与 DataAccess 各自有“全局资源治理”【当前确认】

FE：

```text
ResourceBroker
```

DA：

```text
GlobalResourceGovernor
```

各自做：

```text
memory
concurrency
scan
DuckDB
```

可能发生：

```text
FE认为可启动
DA也认为可启动
```

但它们不知道彼此的完整资源占用。

---

# 52. 最终必须只有一个 HostResourceCoordinator

推荐：

```text
HostResourceCoordinator
│
├─ JobLease
│
├─ ComputeLease
│
├─ DataAccessScanLease
│
├─ DuckDBLease
│
├─ PolarsLease
│
├─ NumbaLease
│
├─ CacheLease
│
├─ SpillLease
└─ WriterLease
```

---

# 53. Child Lease

例如一个 factor block：

```text
JobLease
    └ ComputeLease
        ├ DA ScanLease
        ├ DuckDB CPU/Memory
        └ WriterLease
```

所有 child加总不能超过 host envelope。

---

# 54. 不允许 FE / DA Double Admission

最终：

```text
DA不是再次独立决定“我还有多少全局内存”
```

而是：

```text
DA请求 HostCoordinator child lease
```

---

# 55. R36-P0-017｜DataAccess GlobalResourceGovernor 默认 memory cap 可以是 None【当前确认】

当前 DA governor：

```text
max_total_reserved_memory=None
```

如果由 FE主控，
可以接受。

但如果独立 DataAccess service运行，
生产默认应该从：

```text
hard/live resource envelope
```

自动派生，而不是无限。

---

# 56. DataAccess ScanBudget 应成为 Host Lease 一部分

```text
scan bytes inflight
remote concurrency
DuckDB concurrency
memory
```

统一。

---

# 57. 服务层也必须共享同一个 Coordinator

`BoundedJobQueue` 当前自身有 broker。

最终：

```text
job admission
```

先问：

```text
HostResourceCoordinator
```

而不是：

```text
固定 max_running=4
```

---

# 58. R36-P0-018｜Service “shared broker context” 当前不是 ContextVar【当前确认】

当前：

```python
_service_broker_ctx = None
```

是 module-global变量。

多个 worker线程并发：

```text
A set broker
B set broker
A restore previous
```

可能导致 B仍在运行时全局 pointer被恢复。

因此所谓：

```text
跨 job共享 broker
```

存在竞态。

## 修复

最佳：

```text
真正 process-wide HostResourceCoordinator singleton
```

不需要 per-job覆盖 global pointer。

如果必须 context：

```python
contextvars.ContextVar
```

但 HostCoordinator本身仍应全局共享。

---

# 59. Job Admission 不能只看 Queue Length

提交 job前先做：

```text
cheap compile / dry plan
```

估计：

```text
minimum memory footprint
minimum source bytes
cost class
```

---

# 60. Job-level Lease

如果当前：

```text
SafeEnvelope 12GB
```

有：

```text
job A minimum 8GB
job B minimum 8GB
```

不应该同时启动。

Queue可以排 B。

---

# 61. Job Priority / QoS

推荐三档：

## CRITICAL

```text
每日生产因子落值
模拟盘/实盘必需任务
```

## STANDARD

```text
常规研究 batch
```

## BACKGROUND

```text
Factor mining
大范围冷启动探索
benchmark
```

---

# 62. 资源紧张时 Background 先让路

```text
CRITICAL
> STANDARD
> BACKGROUND
```

用户同机跑：

```text
挖因子算法
```

时，
如果每日 production落值启动：

```text
background task不再 admit新 block
```

---

# 63. 让路不等于 kill

第一阶段：

```text
不杀 running kernel
```

只：

```text
停止新 admission
```

running task结束后资源自然下降。

---

# 64. 支持 Cooperative Yield 的 Stateful Task

未来可以让可 checkpoint task：

```text
在 shard边界
```

主动：

```text
save checkpoint
yield lease
```

但这是 P2，
不要第一阶段强 kill。

---

# 65. ResourceController 状态机

建议：

```text
RAMP_UP
STEADY
THROTTLE_1
THROTTLE_2
DRAIN
EMERGENCY
RECOVERY
```

---

# 66. Fast Down / Slow Up

防 OOM更看重反应速度。

## 压力升高

立即：

```text
concurrency × 0.5
stop prefetch
reduce wave
reduce block
evict cold cache
```

## 压力解除

每：

```text
2~5 stable samples
```

只：

```text
+1 worker
或 +10% wave
```

防止 oscillation。

---

# 67. Hysteresis

例如 memory：

```text
进入 throttle:
    SafeHeadroom < 20%

退出 throttle:
    SafeHeadroom > 30% 持续 N 秒
```

不是：

```text
20%上下疯狂切换
```

---

# 68. Cooldown

每次大幅缩容后：

```text
至少等一个 control window
```

再升。

---

# 69. Memory Slope 也要看

如果：

```text
MemAvailable = 25GB
```

但过去5秒：

```text
-5GB/s
```

说明外部任务正在快速吃内存。

不要因为“现在还有25GB”继续 aggressively admit。

增加：

```text
memory_headroom_slope
```

---

# 70. Predictive Pressure

简单线性预测：

```text
time_to_reserve_boundary
=
(SafeHeadroom - minimum_reserve)
/
memory_consumption_rate
```

若：

```text
< task predicted runtime
```

停止新大 task。

---

# 71. cgroup `memory.high` 的价值

如果部署可控，
建议 FactorEngine所在 cgroup使用：

```text
memory.high
```

作为软 throttle，
再：

```text
memory.max
```

做硬上限。

这样即使 in-process controller失灵，
OS层仍有保护。

---

# 72. OS-level Resource Envelope 是第二道安全线

应用内：

```text
HostResourceCoordinator
```

OS：

```text
cgroup
```

两层共同保证。

不能只靠 Python估算。

---

# 73. CPU 共存负载

不要看到：

```text
CPU利用率 95%
```

就一定认为有问题。

如果：

```text
CPU PSI低
```

可能仍有有效并行空间。

更合理：

```text
CPU utilization
+
CPU PSI
+
run queue/loadavg
```

联合判断。

---

# 74. CPU Weight

如果用 systemd/cgroup部署：

```text
background FactorEngine
```

可以设置较低：

```text
cpu.weight
```

生产落值可以更高。

---

# 75. IO 共存负载

当前只看 throughput不够。

增加：

```text
IO PSI
read/write latency
queue depth
```

---

# 76. Remote DataAccess 也需要 Network Pressure

如 COS/S3/httpfs：

```text
remote request latency
throughput
HTTP error rate
retry rate
```

决定：

```text
remote concurrency
prefetch depth
```

---

# 77. DataAccess remote并发应自动调节

例如：

```text
latency stable
→ +1 concurrent request

429 / timeout / latency spike
→ halve
```

---

# 78. Spill 策略不能只是“内存不够就写盘”

应该比较：

```text
spill_cost
vs
recompute_cost
```

---

# 79. Cost-Based Spill

```text
if expensive shared model state:
    spill

if cheap elementwise:
    drop + recompute
```

---

# 80. Spill Priority

优先 spill：

```text
高计算成本
低重建频率
后面还有多个 consumers
```

优先 drop：

```text
便宜
单 consumer
容易重新从 source native计算
```

---

# 81. Spill 盘必须分类

```text
NVMe
SSD
network
unknown
```

网络盘 spill成本可能比重算更高。

---

# 82. Spill 与 Writer 共享 IO Budget

不能：

```text
spill全速
+
Parquet writer全速
```

一起把盘打爆。

统一：

```text
IO tokens
```

---

# 83. Spill Preemption

在：

```text
PRESSURE_2
```

就可以提前 spill高价值冷对象，
不要等：

```text
快 OOM
```

才做。

---

# 84. Spill 文件生命周期

必须：

```text
execution_id
generation_id
owner
TTL
checksum
```

异常退出后：

```text
cleanup
```

---

# 85. Disk Full

如果：

```text
spill free < reserve
```

禁止再 spill。

转：

```text
shard smaller
recompute
wait
```

---

# 86. R36-P0-019｜当前 Spill Budget 与真正统一 spill manager 尚未完全闭环【当前确认】

Task contract有：

```text
spill_bytes
```

broker也看可用 spill，

但缺少完整：

```text
BufferRef -> spill/reload/recompute
```

生命周期。

R33/R36合并完成。

---

# 87. 自动分片是防 OOM 的核心

如果一个合法 task：

```text
P99 peak > SafeEnvelope
```

不能只：

```text
reject forever
```

应：

```text
auto shard
```

---

# 88. Shardability 必须是语义定义

当前 TaskResourceContract已经有：

```text
shardable
shard_dimension
```

继续扩展。

---

# 89. Elementwise Shard

可：

```text
time
asset
```

---

# 90. 普通 TS Rolling

优先：

```text
asset shard
```

避免 window边界问题。

如果 time shard：

```text
必须带 warmup overlap
```

---

# 91. Cross-Section

不能：

```text
asset shard
```

然后每片 rank，
那会改变因子。

可：

```text
time shard
```

因为每一天仍保留完整股票截面。

---

# 92. Group Operator

是否 asset shard取决于：

```text
group完整性
```

最简单：

```text
time shard
```

---

# 93. Stateful Recursive

优先：

```text
asset shard
```

time shard只允许：

```text
有 checkpoint state contract
```

---

# 94. Full History Recursive

禁止随意：

```text
time shard
```

---

# 95. PCA / Cross-Section Model

不能随意：

```text
asset shard
```

因为 universe变化会改变 eigenvectors。

优先：

```text
time blocks
```

每个日期完整 universe，
并带历史 window overlap。

---

# 96. Minute→Daily

天然：

```text
session/day shard
```

---

# 97. AutoShardPlanner

输入：

```text
task semantic
P99 memory
SafeEnvelope
backend
history requirement
```

输出：

```text
ShardPlan
```

---

# 98. OOM Retry Policy

当前把 OOM归 permanent，
“同尺寸直接重跑”这个原则是正确的。

但未来：

```text
OOM
```

应触发：

```text
ReplanRequired
```

不是：

```text
same task retry
```

---

# 99. OOM 后动作

```text
1. mark prediction underflow
2. increase uncertainty
3. reduce concurrency
4. reduce block/wave
5. auto-shard if legal
6. replan
```

---

# 100. R36-P0-020｜Scheduler 当前 no-progress 后会报 stuck，而不是自动重规划更小 shard【当前确认】

未来：

```text
ready task始终无法 admission
```

如果：

```text
shardable
```

先：

```text
replan smaller
```

再失败。

---

# 101. Memory Liveness 比 Cache Size 更重要

DAG planner已经知道：

```text
每个 Buffer 的消费者
```

应该提前计算：

```text
birth
last_use
```

---

# 102. Buffer Liveness Plan

```text
source buffer
→ consumers
→ last consumer
→ release immediately
```

---

# 103. CSE 引用计数方向正确

现有：

```text
last consumer -> release
```

应保留。

---

# 104. R36-P0-021｜当前 CSE 可以绕过 governed ExpressionCache【当前确认】

当前 shared materialize：

```python
value = backend.execute(sub, ctx)
ctx.shared_result_cache[sid] = value
```

这是 raw dict写入。

而真正 byte-aware：

```text
ExpressionCache.set()
```

有：

```text
budget
LRU
governor
```

因此当前主 batch path可能绕过。

## 修复

ExecutionContext不再暴露可直接写 raw dict作为权威。

改：

```python
ctx.shared_buffers.put(...)
```

统一通过：

```text
GovernedBufferStore
```

---

# 105. GovernedBufferStore

```python
put(buffer, lease)
get(ref)
pin(ref)
release(ref)
spill(ref)
```

---

# 106. Raw Cache Mutation Production 禁止

任何：

```text
shared_cache[key] = value
panel_cache[key] = value
```

在 production hot path都应该 CI检测。

---

# 107. R36-P0-022｜Cache Layer 注册失败当前可被 `except: pass` 吞掉【当前确认】

`ExecutionCacheSession` 注册 governor layer当前：

```python
try:
   register_layer
except:
   pass
```

这会变成：

```text
以为有内存治理
实际没有
```

Production必须：

```text
fail closed
```

Research可以 warning。

---

# 108. R36-P0-023｜Cache Session release 要保证实际引用和账本同时释放

当前：

```text
unregister_layer
```

会移除 governor accounting。

但必须确保：

```text
backing cache refs
```

也真正：

```text
clear/release
```

否则：

```text
实际内存仍在
账面已经没了
```

---

# 109. Cache Accounting Reconciliation

定期检查：

```text
accounted bytes
vs
estimated actual store bytes
```

差异超过阈值：

```text
reconcile
+
alert
```

---

# 110. Cache 插入前必须全局 Admission

当前 L0/L1 cache主要：

```text
自身 budget
```

最终：

```text
local budget
+
HostResourceCoordinator lease
```

同时通过。

---

# 111. Cache 的真正价值应该 Cost-Aware

LRU不一定最优。

评分：

```text
reuse_probability
× recompute_cost
/
bytes
```

---

# 112. 高 reuse 小对象优先保留

低 reuse大 panel优先淘汰。

---

# 113. PanelCache Key 当前可能很贵【当前确认】

当前基于：

```text
Series全部 values
+ index
```

做 sha256。

这可以确保正确性，
但每次可能：

```text
O(N)扫描
+
contiguous copy
```

未来使用：

```text
immutable BufferRef identity
```

替代 full content hash hot path。

---

# 114. Immutable Buffer Identity

```text
source snapshot
columns
time range
universe
transform
semantic version
```

生成 stable identity。

---

# 115. Data Representation 的重复副本必须继续清

典型危险：

```text
Arrow Table
+
Pandas Series
+
Pandas wide panel
+
Polars DataFrame
```

同时存在。

目标：

```text
同一时刻尽量只有一个主 representation
```

---

# 116. Backend Boundary 只转换一次

```text
DuckDB relation
→ one Arrow/NumPy boundary
→ Numba
```

而不是：

```text
DuckDB
→ Arrow
→ Pandas
→ Polars
→ NumPy
```

---

# 117. R33 统一 QueryGraph 仍然是性能主工程

资源治理不是替代 R33。

顺序：

```text
先减少工作量
再动态管理资源
```

---

# 118. R36-P0-024｜FE BatchDataRequest 与 DA DataRequest 仍是重复 planning authority【当前确认】

R33继续执行：

```text
FE dependencies
→ DA DataRequest
→ DA PreparedRead
```

不要两套 planner。

---

# 119. R36-P0-025｜ReadWave 当前还不是完整真实执行计划【当前确认】

已知问题仍包括：

```text
SOURCE_SCAN字段映射不完整
time_range=None
physical source grouping不足
```

R33落实。

---

# 120. R36-P0-026｜Physical Stage 当前主要还是 planning view【当前确认】

当前 root最终仍：

```text
backend.execute(root)
```

Resource Autopilot要达到最优，
未来调度单位必须是真实：

```text
SourceScanStage
NativeComputeStage
ModelKernelStage
WriteStage
```

---

# 121. 为什么真实 Stage 才能精确控资源

如果整个 root一个 task：

```text
scan 5GB
compute 1GB
write 2GB
```

只能给一个粗 peak。

真实 stage后：

```text
scan lease
compute lease
write lease
```

更准。

---

# 122. R36-P0-027｜DataAccess Aggregate+Join 临时 Parquet Hot Path 要删除【当前确认】

当前：

```text
minute aggregate
→ Arrow
→ temp Parquet
→ DuckDB join
```

多了：

```text
RAM
disk
serialization
metadata
```

R33目标：

```text
DuckDB relation / registered Arrow / CTE
```

---

# 123. Temp File 与 Spill 必须区分

如果未来必须临时 materialize：

```text
planner explicit SpillRef
```

不能：

```text
偷偷 tempfile
```

资源 controller才知道磁盘用了多少。

---

# 124. ExecutionResourceScope 的 Process-Global 配置问题

当前会动态改：

```text
DUCKDB_MAX_THREADS
POLARS_MAX_THREADS
DUCKDB_MEMORY_LIMIT
```

这些属于：

```text
process/global engine settings
```

---

# 125. R36-P0-028｜ExecutionResourceScope 的 `_ENV_LOCK` 没有覆盖完整执行生命周期【当前确认】

当前：

```text
__enter__ 内 lock
→ 设置 env
→ unlock
→ workload执行
→ __exit__ 再 lock恢复
```

如果两个 scope并发：

```text
A enter sets 8
B enter sets 2
A实际执行期间可能看到 B配置
```

而恢复顺序也会互相覆盖。

## 修复

不要在并发 job中用 process-global env做 task级动态资源控制。

---

# 126. DuckDB Resource Class

DuckDB应：

```text
由 HostCoordinator分配 threads/memory
```

然后通过：

```text
connection/query-owned setting
```

在明确隔离的 execution context里应用。

若 setting实际上 DatabaseInstance全局：

```text
必须有 configuration epoch / engine instance isolation
```

---

# 127. Polars 更不能靠 scope期间改 env

官方明确：

```text
线程池不能运行中修改
```

因此删除任何“实时改变 POLARS_MAX_THREADS已经生效”的假设。

---

# 128. BLAS/OpenMP 用 threadpoolctl

NumPy/SciPy：

```text
MKL
OpenBLAS
OpenMP
```

可通过：

```text
threadpoolctl
```

在 task scope做限制。

这是更适合动态 CPU budget的库层。

---

# 129. Nested Parallelism 必须统一治理

禁止：

```text
8 factor tasks
×
8 BLAS threads
=
64 runnable
```

---

# 130. CPU Token Rule

```text
outer_workers
×
inner_threads
<= dynamic_cpu_budget
```

更准确：

```text
sum(active_task.cpu_tokens)
<= target_cpu_tokens
```

---

# 131. Small Matrix Rule

大量：

```text
5×5 / 120×5
```

rolling regression：

```text
BLAS threads=1
outer task/block parallel
```

通常更合理。

---

# 132. Large Matrix Rule

大 cross-sectional PCA：

```text
可给一个 task多个 BLAS tokens
```

由 benchmark校准。

---

# 133. Numba 与 Resource Autopilot 的结合

Numba kernel声明：

```text
nogil
parallel
threads
```

---

# 134. 第一阶段 Numba 默认 `parallel=False`

让：

```text
FactorEngine outer scheduler
```

负责并行。

---

# 135. 单个巨大 Numba block 才考虑 prange

并在 task contract写：

```text
cpu_tokens=N
```

---

# 136. Numba Memory Model

JIT kernel通常：

```text
temporary arrays少
```

但仍必须测：

```text
actual peak
```

不能假定。

---

# 137. Model Family Multi-Output 对内存也有帮助

不是只提速。

例如：

```text
PCA decomposition一次
→ 5 outputs
```

比：

```text
5份独立 PCA matrix
```

少内存。

---

# 138. GARCH Family Shared Fit

同理：

```text
FitState
→ persistence/shock/forecast
```

减少：

```text
optimizer scratch
```

---

# 139. Rolling Linear Sufficient Stats

对：

```text
OLS
Ridge
HAR
```

减少：

```text
重复 design matrix
```

同时：

```text
更快
更省内存
```

---

# 140. Resource Autopilot 控制变量

最终至少控制：

```text
target_task_concurrency
target_cpu_tokens
target_duckdb_threads
target_duckdb_memory
target_io_concurrency
target_remote_concurrency
target_read_wave_bytes
target_factor_block_bytes
target_result_queue_bytes
target_cache_budget
target_spill_budget
target_numba_block_size
```

---

# 141. 不要一次同时大幅调整全部变量

建议优先级：

```text
1. admission concurrency
2. wave/block size
3. prefetch
4. cache
5. inner threads
```

降低 controller不稳定性。

---

# 142. Controller Tick

建议：

```text
0.5~2 seconds
```

而不是每 operator/cell。

当前资源采样默认约1秒，方向合理。

---

# 143. Event-Driven Pressure

PSI可用：

```text
poll trigger
```

压力突然升高时：

```text
立即 controller tick
```

不用等下个定时周期。

---

# 144. Static Server Profile 不再需要人工配置

默认：

```text
resources.mode=auto
```

根据环境自己判断。

---

# 145. 但允许 Hard Guardrail

例如：

```yaml
resources:
  max_memory: 48GB
  max_cpu: 12
```

作为：

```text
upper bound
```

Autopilot不能超过。

---

# 146. User Limit 永远是 Ceiling，不是 Target

例如用户：

```text
max_cpu=16
```

当前外部负载高，
Autopilot可能只用：

```text
4
```

---

# 147. Server Idle 时尽量吃满

若：

```text
PSI低
MemAvailable高
writer正常
```

逐步：

```text
increase target
```

直到：

```text
性能不再提升
或
资源软目标
```

---

# 148. 需要 Throughput Feedback

不仅看资源。

如果：

```text
concurrency 8 -> 10
```

吞吐不升反降：

```text
不要继续增加
```

---

# 149. Online Hill-Climb

长期可以：

```text
小幅探索 target±1
```

观察：

```text
TTDC / throughput
```

找当前机器最优点。

---

# 150. Autotuner 不能影响 Correctness

只调整：

```text
资源/physical execution
```

不能调整：

```text
窗口
模型参数
PIT语义
```

---

# 151. QoS + Autotuner

高优先 production：

```text
更倾向占用空闲资源
```

background：

```text
更保守
```

---

# 152. Server 共存时不要争抢 Page Cache

大规模一次性 scan：

```text
不要为了“快”
```

把系统所有可用 RAM都变成自身 cache。

保留：

```text
host reserve
```

是必要的。

---

# 153. Memory Reserve 的实际目标

不是：

```text
永远留20%
```

而是：

```text
保持低 OOM probability
+
避免系统 thrashing
```

---

# 154. Swap

如果服务器启用 swap：

```text
大量 swap-in/out
```

通常已经意味着因子落值性能恶化。

ResourceSnapshot增加：

```text
swap activity
```

高 swap pressure：

```text
throttle
```

---

# 155. Major Page Fault

同样可作为 memory pressure信号。

---

# 156. jemalloc/malloc_trim 不作为主要治理

不要靠：

```text
gc.collect
malloc_trim
```

解决架构内存问题。

可观测，
但不作为主控制手段。

---

# 157. Python Object Count 仍应持续下降

FactorBlock / Arrow / ndarray有助于：

```text
减少数十万 Series/DataFrame对象
```

---

# 158. Result 生命周期

一旦：

```text
DQ通过
write durable
```

立即：

```text
release result buffer
```

---

# 159. DQ 不要强制完整复制

DQ设计：

```text
block statistics
streaming checks
```

避免：

```text
copy whole factor panel
```

---

# 160. Write DQ Pipeline

```text
FactorBlock
→ DQ stats
→ writer
```

尽量单 pass。

---

# 161. DataAccess PIT Join 也要 Streaming / Block-Aware

如果：

```text
financial PIT relation
```

可以在 DuckDB relation中完成：

```text
join
+
factor SQL
```

就不要先 collect。

---

# 162. Remote Scan 不要过度 Prefetch

Autopilot根据：

```text
memory
network
consumer rate
```

决定 prefetch depth。

---

# 163. Cache Warmup 也是 Speculative Work

压力：

```text
PRESSURE_1+
```

首先停：

```text
warmup
speculative prefetch
```

---

# 164. 当前 Pressure Stage 注释已有这些意图，但要真正执行

例如：

```text
PRESSURE_1 stop speculative prefetch
PRESSURE_2 evict cache
```

必须建立实际 action callback，
不能只返回字符串。

---

# 165. PressureActionRegistry

```python
PressureActionRegistry(
    stop_prefetch,
    shrink_read_wave,
    evict_cache,
    reduce_concurrency,
    spill_candidates,
    pause_background,
)
```

---

# 166. ResourceDecision 要可解释

每次变更记录：

```text
why
before
after
signal
```

例如：

```text
concurrency 12 -> 6
reason:
memory PSI full avg10=0.18
MemAvailable slope=-2.1GB/s
```

---

# 167. 不允许 silent resource mutation

所有：

```text
threads
memory
wave
block
```

变动都有 telemetry。

---

# 168. True Run Peak

当前 resource telemetry容易混入：

```text
lifetime ru_maxrss
```

最终必须有本次 run独立：

```text
peak_process_family_pss
peak_process_family_rss
```

---

# 169. R36-P0-029｜当前 `rss_peak_run` 不是真正独立 run peak【当前确认】

当前 finalize会把：

```text
lifetime peak
```

纳入：

```text
rss_peak_run=max(...)
```

这不能用于精确训练 task memory model。

## 修复

run start：

```text
start dedicated sampler
```

run finish：

```text
stop
```

只统计该 run时间窗口。

---

# 170. Per-Task Peak

任务开始：

```text
register task
```

定时 sampler把：

```text
current process-family PSS
```

和：

```text
active task attribution
```

记录。

---

# 171. 多 task并发时 Peak Attribution

无法精确把全进程内存逐 byte归属单 task。

可使用：

```text
isolated calibration runs
+
concurrent marginal model
```

结合。

---

# 172. Calibration Suite

Nightly：

```text
单 task isolated
```

得到 clean peak。

Production：

```text
online correction
```

---

# 173. Memory Model 不要只用实际 RSS delta

因为：

```text
allocator reuse
```

会让 delta失真。

综合：

```text
object bytes
native engine metrics
isolated peak
process family peak
```

---

# 174. DuckDB 内部 Memory Telemetry

可读取：

```text
duckdb_memory()
```

用于：

```text
buffer/intermediate component
```

校准。

---

# 175. DataAccess ScanCost 必须进入资源模型

已已有：

```text
selected_bytes
projection_bytes
estimated_rows
remote
```

继续作为输入。

---

# 176. 编译阶段先做 Memory Feasibility

计划完成后：

```text
minimum feasible peak
```

如果：

```text
即使 concurrency=1
也 > hard safe envelope
```

必须提前：

```text
auto shard
```

而不是执行后才 OOM。

---

# 177. Dry Run 输出

```text
estimated TTDC
estimated peak P99
recommended initial concurrency
recommended wave
recommended block
spill expectation
```

---

# 178. `plan_many_fast` 应升级为 Resource Autopilot Explain

现有已经输出：

```text
cpu slots
live memory
recommended concurrency
```

继续扩展。

---

# 179. Plan Explain 示例

```text
Server:
  hard RAM: 64GB
  live safe: 27GB
  CPU hard: 32
  external CPU pressure: low
  memory PSI: low

Batch:
  factors: 1000
  p99 root peak: 1.8GB
  selected source: 12GB

Initial plan:
  concurrency: 8
  DuckDB threads: 4
  read wave: 3GB
  result block: 512MB
  sink queue: 1GB
```

---

# 180. 资源变化时无需重新 compile semantic plan

只重新：

```text
physical scheduling plan
```

---

# 181. Logical Plan 与 Resource Plan 分离

```text
LogicalFactorPlan
```

不变。

```text
PhysicalResourcePlan
```

可动态重算。

---

# 182. Replan Boundary

只能在安全边界：

```text
between stages
between waves
between blocks
```

重新调整。

---

# 183. 任务完成顺序不用固定

只要：

```text
determinism contract
```

允许。

最终 factor values不能随调度顺序变化。

---

# 184. Deterministic Reduction

并行 reduction必须有明确：

```text
floating determinism level
```

R32/R34继续。

---

# 185. Resource Autopilot 不改变随机种子

若有研究随机：

```text
seed
```

与调度并发分离。

---

# 186. 当前 CSE 共享策略与内存压力需要联合

高 reuse shared node：

```text
算一次很省 CPU
```

但：

```text
如果特别巨大
```

可能占用过久。

Planner要判断：

```text
materialize
vs
recompute
vs
spill
```

---

# 187. CSE Benefit 公式加入 Memory Rent

```text
Benefit
=
SavedCompute
- MaterializeCost
- ConversionCost
- SpillCost
- MemoryRent
```

---

# 188. MemoryRent

对象越大、活得越久：

```text
MemoryRent越高
```

---

# 189. 不要因 CSE 让低内存机器更慢/更危险

CSE不是永远开启越多越好。

---

# 190. Factor Campaign Session 与资源治理

R33的：

```text
FactorCampaignSession
```

非常适合算法反复挖掘。

但 cache budget必须：

```text
elastic
```

外部资源压力高时：

```text
缩 campaign cache
```

---

# 191. Campaign Cache Admission

只保留：

```text
high reuse / high recompute cost
```

---

# 192. 同机模型训练启动时 Campaign 自动降温

不再：

```text
长期占几十GB source/PIT cache
```

---

# 193. 进程间共享 Cache 是否值得

第一阶段：

```text
不要复杂化
```

优先：

```text
单 FE service process
+
threads/native
```

---

# 194. Process Lane 真正启用后

可以考虑：

```text
mmap Arrow
shared memory
```

避免复制。

---

# 195. Process Lane 仍是 Residual Lane

R35结论继续：

```text
Numba/native优先
Process only GIL-heavy residual
```

---

# 196. Process Worker Memory 必须进入 Host Ledger

每 worker：

```text
PSS
```

被 coordinator观测。

---

# 197. Worker Crash

必须：

```text
lease自动回收
shared refs不泄漏
spill cleanup
```

---

# 198. Process Worker Preload

如果 worker预加载巨大 operator/model：

```text
memory baseline
```

要进入每 worker fixed overhead。

---

# 199. Thread Pool 不应该默认等于全部 CPU

池最大值可以大，
但：

```text
实际 admission
```

由 token控制。

---

# 200. 多服务器环境

每台服务器都运行自己的：

```text
HostResourceCoordinator
```

即可。

不要在单机未做到最优前先复杂上：

```text
cluster scheduler
```

---

# 201. Multi-Node 什么时候再做

当：

```text
单机 source scan复用
内存治理
native execution
model kernels
```

已经收口，
且单机 CPU/IO真实饱和后。

---

# 202. 分布式的未来切分单位

更适合：

```text
dataset/snapshot compatible campaign
time shard
factor family block
```

而不是：

```text
每 factor一个远程 job
```

---

# 203. GPU 暂不作为 R36 P0/P1

只有 benchmark证明：

```text
matrix-heavy workload占 TTDC大头
```

再考虑。

---

# 204. NUMA 也暂时 P2

仅多 socket大机器 + profiler显示：

```text
remote memory bottleneck
```

再做。

---

# 205. FE / DataAccess 版本兼容协议

当前：

```text
factor-engine 0.3.1
data-access 0.8.0
```

FE依赖仍：

```text
data-access>=0.2.0
```

对于现在如此深的：

```text
PreparedRead
ScanCost
PIT
relation
resource
```

耦合，仅 semver下限已经太弱。

---

# 206. R36-P1-001｜增加 FE×DA Capability Handshake【当前确认需完善】

启动：

```python
DataAccessCapabilities(
    api_version,
    scan_cost_version,
    prepared_read_version,
    pit_contract_version,
    relation_api_version,
    resource_lease_version,
)
```

FE检查。

---

# 207. 不一定要锁 exact package version

更好：

```text
capability protocol
```

兼容就允许。

---

# 208. 软件升级必须重新校准 performance

DuckDB/Polars/NumPy升级：

```text
hardware相同
```

也可能 performance变。

Calibration key包含：

```text
software version
```

---

# 209. Resource Autopilot 持久化

记录：

```text
resource_calibration.parquet
```

不要仅存内存。

---

# 210. Calibration Aging

旧数据：

```text
指数衰减权重
```

避免过去 workload永久影响。

---

# 211. Outlier

一次异常 OOM：

```text
立即提高 tail safety
```

不能被均值稀释。

---

# 212. Near-OOM 也记录

例如：

```text
SafeHeadroom < emergency threshold
```

即使没 OOM：

```text
near_oom_event
```

---

# 213. Memory Safety SLO

例如：

```text
production OOM kill = 0
near-OOM rate < threshold
```

---

# 214. Performance SLO

```text
TTDC P50/P95
```

按：

```text
server profile
batch class
```

比较。

---

# 215. 不能用最快速度牺牲系统稳定

Optimization objective：

```text
minimize TTDC
subject to:
    OOM probability < epsilon
    correctness = PASS
    resource QoS respected
```

---

# 216. Controller Objective

可以形式化：

```text
Utility =
Throughput
- λ1 * memory_pressure
- λ2 * io_pressure
- λ3 * external_interference
- λ4 * spill_cost
```

---

# 217. 不需要第一版做强化学习 Controller

第一版：

```text
AIMD + rules + calibrated costs
```

更稳定、更可解释。

---

# 218. Controller必须可回放

Resource decision log可以：

```text
replay
```

用于 debug。

---

# 219. Resource Event Log

每条：

```text
timestamp
signals
decision
running tasks
predicted memory
actual memory
```

---

# 220. 外部算法不可见时如何处理

无需知道它是谁。

只观察：

```text
MemAvailable
PSI
CPU
IO
```

即可。

---

# 221. 如果能接统一服务器 Scheduler 更好

未来若算法组有：

```text
Ray/Slurm/K8s
```

可以读：

```text
allocated resources
```

作为 hard envelope。

但 FactorEngine仍保留 live controller。

---

# 222. K8s

用：

```text
requests/limits
cgroup
```

作为 hard resource source。

---

# 223. Slurm

当前已有：

```text
SLURM memory
```

继续扩 CPU env检测。

---

# 224. Resource Priority 外部 API

FactorEngine run可传：

```python
priority="critical|standard|background"
```

---

# 225. Deadline-Aware

如果 production任务有 DDL：

```text
deadline approaching
```

且系统空闲：

```text
允许更 aggressive
```

但仍不破 emergency reserve。

---

# 226. Background Mining 可自动暂停

压力很高：

```text
停止下一批 grammar candidate
```

比让生产任务 OOM好。

---

# 227. 同一 batch 内因子 Cost Scheduling

先安排：

```text
高 reuse source
高 critical path
```

同时考虑：

```text
memory fit
```

---

# 228. Bin Packing

ready tasks按：

```text
CPU
memory
IO
```

做小型多维 bin packing。

不是简单：

```text
按 task数量
```

---

# 229. Admission Score

```text
priority
+ critical path
+ reuse
+ resource fit
+ cache locality
```

---

# 230. 当前 `remaining=set(...)` 会丢排序意图【当前确认】

scheduler把 pending转换为：

```text
set
```

admission遍历顺序不能体现完整 priority。

改：

```text
priority heap / indexed ready queue
```

---

# 231. Event-Driven Scheduler

当前：

```text
wait(..., timeout=0.05)
```

固定轮询。

未来：

```text
future completion
resource pressure event
writer pressure event
PSI event
```

驱动。

---

# 232. 50ms 对大任务不是问题，对小 batch是 overhead

small batch：

```text
direct vector/native path
```

绕开完整 scheduler。

R33继续。

---

# 233. Small Batch Bypass

例如：

```text
<= N cheap factors
```

自动：

```text
direct fused execution
```

---

# 234. Resource Autopilot 不应增加 small batch latency

Benchmark gate。

---

# 235. Input DQ 也要 Resource-Aware

大 source：

```text
DQ sample/statistics
```

尽量：

```text
pushdown
single pass
```

---

# 236. DQ 与 Data Scan 共享

不要：

```text
为了 DQ再读一次完整 source
```

---

# 237. Metadata / Schema 不要频繁扫描 Footer

DataAccess已经做了 schema epoch等优化，
继续复用。

---

# 238. Source Snapshot Pin

整个 batch：

```text
一个 query-scoped snapshot
```

资源 replan不能改变 snapshot。

---

# 239. Resource Replan 不改变数据身份

即：

```text
same batch
same semantic snapshot
```

即使：

```text
wave大小改变
backend stage改变
```

---

# 240. Auto Backend Route 仍只从 Certified Candidates 选

R34/R35继续：

```text
correctness eligibility first
```

资源紧张不能让一个：

```text
未认证 backend
```

因为省内存就进入 production。

---

# 241. Memory-Safe Backend Preference

在 eligible内部，

资源紧张时 cost函数增加：

```text
memory risk penalty
```

---

# 242. TTDC Cost 增加 Pressure Penalty

```text
Cost =
ExpectedTime
+ MemoryRisk
+ SpillCost
+ Conversion
+ ContentionPenalty
```

---

# 243. 高内存 Polars In-Memory 可以输给 DuckDB Streaming

即使平时 microbenchmark更快。

---

# 244. 低内存模式不是单独手工配置

ResourceController自动进入：

```text
memory_constrained mode
```

---

# 245. Mode 只是 explain标签

内部仍连续控制，
不是只有：

```text
fast / slow
```

两个档。

---

# 246. Resource Profiles

可以保留：

```text
latency
balanced
throughput
background
```

只影响 controller权重。

---

# 247. 默认建议 `balanced`

生产落值：

```text
throughput
```

背景挖掘：

```text
background
```

---

# 248. 容量规划报告

长期统计：

```text
batch规模
TTDC
P95 memory
CPU
spill
```

可以知道：

```text
扩 RAM还是扩 CPU更有价值
```

---

# 249. 是否需要更多 RAM

用 profiler而不是猜。

---

# 250. 是否需要更多 CPU

同理。

---

# 251. 是否需要 NVMe

若：

```text
spill/write占 TTDC大头
```

再升级。

---

# 252. R36 测试必须模拟共存负载

这轮最重要的新测试领域。

---

# 253. Resource Test R1｜8GB cgroup

模拟：

```text
hard=8GB
```

100/1000 factors。

要求：

```text
no OOM
automatic smaller wave/block/concurrency
```

---

# 254. R2｜16GB

同上。

---

# 255. R3｜64GB

要求：

```text
自动更 aggressive
```

不能仍按8GB速度跑。

---

# 256. R4｜256GB

验证：

```text
不会被固定4GB wave/4 workers限制
```

---

# 257. R5｜外部 Memory Eater 启动

运行中：

```text
external process逐步吃 RAM
```

要求：

```text
FE concurrency自动下降
no OOM
```

---

# 258. R6｜外部 Memory Eater 释放

随后释放。

要求：

```text
FE在稳定窗口后自动升速
```

这条非常重要。

---

# 259. R7｜内存突然下降

瞬时：

```text
MemAvailable -50%
```

要求：

```text
stop new admission
```

---

# 260. R8｜Memory Slope

持续快速下降。

即使尚未到 low watermark：

```text
提前 throttle
```

---

# 261. R9｜Memory PSI

注入高 pressure。

要求 controller响应。

---

# 262. R10｜外部 CPU Load

另一个进程满 CPU。

要求：

```text
FE CPU tokens下降
```

---

# 263. R11｜CPU Load 释放

要求：

```text
自动恢复
```

---

# 264. R12｜IO Saturation

外部 fio/等价测试。

要求：

```text
prefetch/writer/spill并发下降
```

---

# 265. R13｜Remote Latency Spike

COS/S3模拟延迟。

要求：

```text
remote concurrency调整
```

---

# 266. R14｜Writer Stall

writer sleep。

要求：

```text
sink queue上升
scheduler减 compute
```

---

# 267. R15｜Writer Fatal

要求：

```text
stop admission
generation fail
zero silent loss
```

---

# 268. R16｜Spill Disk Full

要求：

```text
no endless spill attempt
auto smaller shard / fail controlled
```

---

# 269. R17｜Slow Network Spill

验证：

```text
recompute beats spill
```

时 planner选择正确。

---

# 270. R18｜Giant CSE

一个 shared panel：

```text
> local cache budget
```

要求：

```text
不 raw insert
```

---

# 271. R19｜Cache Registration Failure

生产：

```text
hard fail
```

---

# 272. R20｜Cache Accounting Drift

人为制造：

```text
actual != accounted
```

必须 detect。

---

# 273. R21｜多 Service Jobs

并发：

```text
4 jobs
```

要求：

```text
共享一个 HostResourceCoordinator
```

---

# 274. R22｜Service Broker Race

重现当前：

```text
module global restore
```

竞态。

新版本必须杀掉。

---

# 275. R23｜Process Workers

如果启用 process：

```text
children PSS计入
```

---

# 276. R24｜DuckDB memory_limit 外部突发

制造：

```text
query actual RSS > configured DuckDB buffer limit
```

仍不能系统 OOM。

---

# 277. R25｜DuckDB Threads Tokens

每次：

```text
actual threads
==
lease CPU tokens
```

或严格不超过。

---

# 278. R26｜Polars Streaming

低内存环境：

```text
streaming selected
```

---

# 279. R27｜Polars Fallback

query不支持 streaming时：

```text
提前识别 / replan
```

不能 silent full-memory OOM。

---

# 280. R28｜Numba Block

自动 block大小适配 memory。

---

# 281. R29｜Run Peak

验证：

```text
run_peak
```

不是历史 lifetime peak。

---

# 282. R30｜Prediction Calibration

故意低估。

连续执行后：

```text
P99 model上调
```

---

# 283. R31｜Over-conservative

长期高估。

只能：

```text
慢慢下调
```

不能突然放松。

---

# 284. R32｜OOM Injection

强行小 hard cap。

要求：

```text
same-shape retry=0
auto replan
```

---

# 285. R33｜Small Batch

Resource controller不能让：

```text
20 cheap factors
```

比 direct path明显慢。

---

# 286. R34｜1000 Daily Factors

测：

```text
TTDC
peak memory
no OOM
```

---

# 287. R35｜Mixed Source 1000

price/fundamental/event/minute/model混合。

---

# 288. R36｜Model-heavy

测试：

```text
PCA
Kalman
HAR
GARCH
```

资源预测。

---

# 289. R37｜Incremental One-Day

应该明显少资源。

---

# 290. R38｜Historical Recompute

大窗口 correction。

---

# 291. R39｜Cancellation

running task安全结束，
new admission停止，
lease全部释放。

---

# 292. R40｜SIGTERM

generation不半提交，
spill/cache cleanup。

---

# 293. R41｜Disk Full During Commit

atomic generation仍保持旧版本可读。

---

# 294. R42｜Resource Recovery

每一种：

```text
memory
CPU
IO
```

都必须验证：

```text
压力结束后 throughput恢复
```

---

# 295. R36 Hard Gates 总表

```text
R36_MAX_CONCURRENCY_ENFORCED
R36_BROKER_RECOMMENDATION_CONSUMED
R36_RESOURCE_RECOVERY_UPSHIFT
R36_DYNAMIC_CPU_BUDGET_BIDIRECTIONAL

R36_DYNAMIC_READ_WAVE_BYTES
R36_DYNAMIC_FACTOR_BLOCK_BYTES
R36_DYNAMIC_SINK_QUEUE_BYTES

R36_SINK_BACKPRESSURE_TO_SCHEDULER
R36_WRITER_FAILURE_FATAL
R36_ZERO_FINISH_SILENT_WRITE_LOSS

R36_PROCESS_FAMILY_MEMORY_ACCOUNTED
R36_TRUE_RUN_PEAK
R36_P99_MEMORY_MODEL_ACTIVE
R36_RESOURCE_CALIBRATION_PERSISTED

R36_PSI_MEMORY_CONSUMED
R36_PSI_CPU_CONSUMED
R36_PSI_IO_CONSUMED
R36_MEMORY_SLOPE_CONSUMED

R36_ONE_HOST_RESOURCE_AUTHORITY
R36_FE_DA_DOUBLE_ADMISSION_ZERO
R36_SERVICE_BROKER_RACE_ZERO
R36_JOB_LEVEL_RESOURCE_LEASE

R36_ZERO_RAW_CSE_CACHE_BYPASS
R36_CACHE_GOVERNANCE_FAIL_CLOSED
R36_CACHE_ACCOUNTING_RECONCILES
R36_CACHE_RELEASE_REF_AND_ACCOUNTING_MATCH

R36_AUTO_SHARD_WHEN_TASK_EXCEEDS_ENVELOPE
R36_ZERO_ILLEGAL_SHARD
R36_ZERO_SAME_SHAPE_OOM_RETRY

R36_DUCKDB_CPU_TOKEN_MATCH
R36_DUCKDB_OUTSIDE_BUFFER_RESERVE
R36_DUCKDB_SPILL_QUOTA
R36_ZERO_CONCURRENT_GLOBAL_RESOURCE_SCOPE_RACE

R36_POLARS_THREAD_CONTRACT_HONEST
R36_POLARS_STREAMING_MEMORY_ROUTE
R36_POLARS_STREAMING_FALLBACK_GUARDED

R36_BLAS_NESTED_OVERSUBSCRIPTION_ZERO
R36_NUMBA_BLOCK_RESOURCE_CONTRACT

R36_CO_TENANCY_MEMORY_STRESS_NO_OOM
R36_CO_TENANCY_CPU_STRESS_NO_STARVATION
R36_CO_TENANCY_IO_STRESS_CONTROLLED
R36_RECOVERY_TO_FULL_SPEED_AFTER_EXTERNAL_LOAD

R36_SMALL_BATCH_NO_REGRESSION
R36_1000_FACTOR_TTDC_CURRENT_HEAD
R36_CURRENT_HEAD_FULL_CORRECTNESS_SUITE
```

---

# 296. P0 优先修复清单

必须先完成：

```text
P0-001 max_concurrency真正生效
P0-002 scheduler消费动态 concurrency
P0-003 压力恢复自动升速
P0-004 去固定2GB worker
P0-005 per-shape P99模型
P0-006 uncertainty反馈闭环
P0-007 process family PSS
P0-008 backend threads/token一致
P0-009 disk busy校准
P0-010 dynamic read wave
P0-011 dynamic sink queue
P0-012 sink backpressure接 scheduler
P0-013 deque
P0-014 writer fail fatal
P0-015 parallel materialize streaming
P0-016 FE/DA统一 resource authority
P0-017 DA memory cap统一
P0-018 service broker race
P0-019 spill lifecycle
P0-020 auto shard
P0-021 raw CSE cache bypass
P0-022 cache registration fail closed
P0-023 cache release/accounting
P0-024 FE/DA planner integration
P0-025 executable read wave
P0-026 executable physical stage
P0-027 temp parquet hot path删除
P0-028 global resource scope race
P0-029 true run peak
```

---

# 297. P1 完善项

```text
P1-001 FE×DA capability handshake
P1-002 PSI events
P1-003 memory slope prediction
P1-004 adaptive remote concurrency
P1-005 cost-aware spill/recompute
P1-006 cost-aware cache
P1-007 resource decision explain
P1-008 persistent calibration
P1-009 QoS lanes
P1-010 job dry admission
P1-011 ResourceShapeKey
P1-012 hardware fingerprint
P1-013 storage auto-calibration
P1-014 online throughput hill climb
P1-015 campaign cache elasticity
```

---

# 298. P2 未来项

```text
cooperative checkpoint yield
NUMA
GPU
multi-node
shared process cache
advanced online optimizer
```

只有 P0/P1收口后考虑。

---

# 299. R33 / R34 / R35 / R36 的最终关系

```text
R34:
    什么东西是正确的

R35:
    模型和 custom kernel怎么正确且快

R33:
    FE×DA如何减少扫描/转换/落盘工作

R36:
    在真实异构、共存服务器上
    如何自动把这些正确的快路径跑到尽可能快
    又不把服务器撑爆
```

---

# 300. 最终生产路径

```text
Factors
   ↓
Compile / R34 Certification
   ↓
Model Kernel / R35
   ↓
Unified FE×DA QueryGraph / R33
   ↓
Resource Autopilot / R36
   ↓
PreparedBatchRead
   ↓
ReadWave N+1
   ||
Native / Numba Compute N
   ||
DQ + Write N-1
   ↓
FactorBlock
   ↓
Atomic Generation
```

---

# 301. 最终 Resource Autopilot 伪代码

```python
while batch_not_done:

    signals = monitor.snapshot()

    envelope = envelope_model.compute(signals)

    prediction = workload_model.predict(
        ready_tasks,
        hardware=current_hardware,
    )

    decision = controller.decide(
        envelope=envelope,
        task_predictions=prediction,
        writer_backpressure=writer.backpressure,
        scan_backpressure=dataaccess.backpressure,
        priority=job.priority,
    )

    scheduler.set_targets(decision)

    for task in scheduler.best_fit_ready_tasks():
        if coordinator.can_lease(task.resource_request):
            scheduler.admit(task)

    if pressure_event:
        controller.fast_down()

    if stable_recovery:
        controller.slow_up()

    if task_cannot_fit:
        scheduler.replan_with_shards(task)

    if writer_failed:
        scheduler.stop_new_admission()
        fail_generation()
```

---

# 302. 推荐新增模块

```text
factor_engine/runtime/host_resource_coordinator.py
factor_engine/runtime/resource_monitor.py
factor_engine/runtime/resource_autopilot.py
factor_engine/runtime/resource_calibration_store.py
factor_engine/runtime/resource_shape.py
factor_engine/runtime/pressure_actions.py
factor_engine/runtime/auto_shard_planner.py
factor_engine/runtime/buffer_store.py
factor_engine/runtime/spill_store.py
factor_engine/runtime/run_peak_sampler.py

dataaccess/runtime/resource_bridge.py
```

注意：

```text
不要再建立第三/第四套独立 resource truth
```

这些模块必须围绕一个：

```text
HostResourceCoordinator
```

服务。

---

# 303. 推荐核心数据结构

```python
@dataclass(frozen=True)
class HostResourceEnvelope:
    hard_memory_bytes: int
    safe_memory_bytes: int
    emergency_reserve_bytes: int

    hard_cpu_tokens: int
    target_cpu_tokens: int

    io_capacity_score: float
    remote_capacity_score: float

    spill_free_bytes: int
```

---

# 304. ResourceSignals

```python
@dataclass(frozen=True)
class ResourceSignals:
    host_mem_available: int
    cgroup_mem_current: int | None

    family_rss: int
    family_pss: int

    memory_psi_some: float
    memory_psi_full: float

    cpu_util: float
    cpu_psi_some: float

    io_psi_some: float
    disk_latency_ms: float

    writer_backpressure: float
    scan_backpressure: float

    mem_available_slope: float
```

---

# 305. ResourceDecision

```python
@dataclass(frozen=True)
class ResourceDecision:
    target_concurrency: int
    target_cpu_tokens: int

    read_wave_bytes: int
    factor_block_bytes: int
    result_queue_bytes: int

    io_concurrency: int
    remote_concurrency: int

    cache_budget_bytes: int
    spill_budget_bytes: int

    pressure_state: str
    reasons: tuple[str, ...]
```

---

# 306. TaskResourcePrediction

```python
@dataclass(frozen=True)
class TaskResourcePrediction:
    memory_p50: int
    memory_p95: int
    memory_p99: int

    elapsed_p50_ms: float
    elapsed_p95_ms: float

    output_bytes: int
    spill_bytes: int

    cpu_tokens: int
    io_tokens: int

    confidence: float
```

---

# 307. Resource Lease

```python
@dataclass
class HostResourceLease:
    memory_bytes: int
    cpu_tokens: int
    io_tokens: int
    spill_bytes: int

    parent_lease_id: str | None
    owner: str
```

---

# 308. ResourceController 第一版不需要复杂预测模型

可以先：

```text
EWMA
rolling quantile
bucket table
```

而不是：

```text
ML model
```

---

# 309. Calibration Key 离散桶

例如：

```text
rows:
    1e4 / 1e5 / 1e6 / 1e7

assets:
    100 / 500 / 1000 / 5000

window:
    20 / 60 / 120 / 252 / full
```

---

# 310. 后续再拟合连续模型

如果 evidence足够：

```text
quantile regression
```

预测 P99。

---

# 311. Controller Safety Floor

任何 learned model不能把：

```text
安全 factor
```

降到：

```text
< configured minimum
```

---

# 312. Cold Start

未知 kernel：

```text
higher uncertainty
```

首次：

```text
低并发
```

积累数据后升。

---

# 313. 这是自动化最合理的方式

用户无需知道：

```text
这个算子大概占几GB
```

系统自己学习。

---

# 314. Release Report 应包含 Server Profile

每次 benchmark：

```text
hardware
limits
software
resource controller version
```

---

# 315. 不同服务器不要直接比较绝对时间

按：

```text
profile
```

比较。

---

# 316. 同一服务器 Controller 版本 A/B

可评估：

```text
TTDC
peak
spill
near-OOM
```

---

# 317. Shadow Autopilot

上线初期：

```text
controller只给 recommendation
旧 scheduler执行
```

比较：

```text
would-have-chosen
```

---

# 318. 再切 Active

确认稳定后：

```text
controller actual controls
```

---

# 319. Canary

先：

```text
少量 batch / server
```

---

# 320. Rollback

Resource Autopilot异常：

```text
回退 conservative static profile
```

但 production不能回退到：

```text
unbounded
```

---

# 321. Conservative Safe Mode

保底：

```text
concurrency=1
small wave
streaming
bounded cache
sink
```

---

# 322. Safe Mode 仍必须正确落值

只是慢。

---

# 323. OOM Killer 不应该成为正常控制反馈

一旦：

```text
oom_kill
```

就是严重失败。

---

# 324. cgroup memory.events

记录：

```text
high
max
oom
oom_kill
```

用于 release gate。

---

# 325. R36 Final Acceptance

最终必须同时满足：

```text
Correctness:
    R34/R35 green

Data path:
    R33 green

Resource:
    R36 green

Co-tenancy:
    no OOM
    automatic yield
    automatic recovery

Performance:
    current HEAD TTDC optimized
```

---

# 326. “最快”最终定义

不是：

```text
某个 operator microbenchmark最快
```

而是：

```text
在当前服务器、
当前同机负载、
当前数据 snapshot、
当前 batch 下，
不违反 correctness / PIT / memory safety / QoS 的前提下，
TimeToDurableCommit 最短。
```

---

# 327. 最终建议实施顺序

## Phase 1｜立即 P0 资源安全修复

```text
max_concurrency
bidirectional controller
dynamic wave/queue
backpressure
CSE cache bypass
writer fatal
service broker race
true PSS/run peak
```

## Phase 2｜统一 FE×DA Resource Authority

```text
HostResourceCoordinator
child leases
job leases
```

## Phase 3｜AutoShard / Spill / Liveness

```text
oversized task自动重规划
```

## Phase 4｜R33 Real Physical Execution

```text
真实 read/compute/write stages
```

## Phase 5｜R35 Kernel Acceleration

```text
Numba
rolling linear
model family multi-output
```

## Phase 6｜Calibration + PSI + Online Autotune

```text
持续学习服务器
```

## Phase 7｜Full Stress / Benchmark / Release

---

# 328. coding AI 不得提前结束

不能在：

```text
新增了 HostResourceCoordinator 类
```

时结束。

必须证明：

```text
不同 memory/cpu/cgroup
+
外部共存负载
+
1000 factor
+
writer stall
+
spill pressure
```

都真的通过。

---

# 329. Definition of Done Checklist

```text
[ ] max_concurrency实际生效
[ ] recommended_concurrency实际生效
[ ] 压力下降后自动恢复并发
[ ] fixed 2GB worker移除
[ ] fixed 3GB worker只做 cold fallback
[ ] P99 memory model
[ ] process family PSS
[ ] PSI memory/cpu/io
[ ] memory slope
[ ] dynamic read wave
[ ] dynamic factor block
[ ] dynamic result queue
[ ] writer backpressure进入 scheduler
[ ] writer failure fatal
[ ] parallel materialization streaming
[ ] service job resource lease
[ ] service broker race修复
[ ] FE/DA one resource authority
[ ] DuckDB token/thread/memory一致
[ ] Polars thread semantics正确
[ ] BLAS nested threads受控
[ ] CSE无 raw cache bypass
[ ] cache registration fail closed
[ ] cache accounting reconciliation
[ ] legal auto-sharding
[ ] cost-aware spill/recompute
[ ] true run peak
[ ] resource calibration persistence
[ ] hardware fingerprint
[ ] co-tenancy memory test
[ ] co-tenancy CPU test
[ ] co-tenancy IO test
[ ] recovery test
[ ] 8GB/16GB/64GB/256GB profile tests
[ ] R33 PASS
[ ] R34 PASS
[ ] R35 PASS
[ ] R36 PASS
```

---

# 330. 最终一句工程标准

> FactorEngine 不应该要求服务器“为它清场”，也不应该用固定 worker 数赌内存；它应该把自己当成一个弹性计算负载：服务器空闲时主动吃满安全余量，其他算法启动时快速收缩，压力解除后自动恢复，并且无论怎样调度都不改变因子的语义、PIT、结果和最终 generation 的一致性。


---

# 附录 A｜FactorEngine 最终总收口路线：R33～R36 之外不要遗漏的未来工程

这一附录用于回答：

```text
除了资源治理以外，
FactorEngine 后续还有什么应该继续完善？
```

结论是：

```text
有，但不应该再零散加模块。
```

后续所有工程应统一收敛到下面 12 条主链。

---

# A1. Operator Truth Layer

最终任何 canonical 都必须有唯一权威：

```text
CanonicalSpec
```

包含：

```text
name
semantic_version
role
input units
output unit
parameter domain
time semantics
missing semantics
state semantics
source requirements
backend certifications
cost class
```

避免：

```text
metadata
tags
evidence
docs
registry
```

各说各话。

---

# A2. Operator Role 不再二元 production/research

继续 R35：

```text
FAST_NATIVE_ALPHA
EXPENSIVE_CERTIFIED_ALPHA
STATE_CONDITION_EVENT
MODEL_FEATURE_SCORE
DIAGNOSTIC_RESEARCH
DELETE
```

这能解决：

```text
“有用但很贵”
```

的 operator 到底该不该保留的问题。

---

# A3. 全算子 Readiness Ledger

最终机器可读：

```text
canonical
semantic_version
parameter_domain
backend
execution_variant
status
```

例如：

```text
ts_mean
v3
window=1..252
DuckDB window-fused
DIRECT_USE_CERTIFIED
```

---

# A4. 任何 Fast Path 都从 Reference Kernel 出发

不能：

```text
为了快把 reference删掉
```

Reference负责：

```text
golden
debug
parity
fallback
```

Fast负责：

```text
production throughput
```

---

# A5. Model Truth Layer

继续 R35。

每个 model-like canonical：

```text
explicit ModelOperatorContract
```

必须明确：

```text
fit cutoff
label maturity
scaler cutoff
hyperparameter cutoff
refit policy
convergence policy
```

---

# A6. Source / PIT Truth Layer

继续 R34。

每个 source不是简单：

```text
field name
```

而是：

```text
knowledge time
event time
revision time
effective period
availability
```

---

# A7. Universe / Calendar / Session 成为一等身份

最终 factor identity必须绑定：

```text
market
universe snapshot/policy
calendar
session clock
decision time policy
```

---

# A8. Unified QueryGraph

继续 R33。

必须做到：

```text
Factor IR
+
DataAccess DataRequest
+
PIT Join
+
Minute Aggregation
+
Backend Region
+
Write
```

在一个 physical planning framework中优化。

---

# A9. SourceRef 不再是 Backend Router 的黑盒

在 route前：

```text
SourceRef
→ DataAccess relational source
```

否则：

```text
SQL eligibility
```

容易被过早禁掉。

---

# A10. Backend Region，而不是 Whole-Root Backend

一个因子：

```text
DuckDB scan/join/window
→ Numba custom state
→ Polars/cross-section
```

可以合法有多个 region。

原则：

```text
最大 native region
最少 conversion boundary
```

---

# A11. Model Intermediate CSE

继续 R35：

```text
PCAState
GARCHFitState
KalmanState
RollingLinearState
```

成为一等 reusable intermediate。

---

# A12. Rolling Sufficient Statistics Engine

优先覆盖：

```text
OLS
Ridge
HAR
beta
correlation/covariance families
```

---

# A13. Numba Kernel Layer

优先：

```text
recursive state
technical recurrence
Kalman
ElasticNet
PLS
GARCH likelihood recursion
```

---

# A14. Numba 不是 Backend Catch-All

不把：

```text
SQL
Polars
LAPACK
```

能做好的东西搬回 Numba。

---

# A15. Native Backend Certification 必须参数域化

不是：

```text
operator backend safe
```

而是：

```text
operator + parameter domain + semantic version + execution variant safe
```

---

# A16. DataAccess 最终应该提供 Batch Read Session

```text
PreparedBatchReadTemplate
```

一次解析：

```text
field
dataset
snapshot
PIT
schema
physical objects
```

多个 factor共享。

---

# A17. DataAccess 读结果保持 Native

尽量：

```text
DuckDBRelationRef
ArrowBatchStreamRef
PolarsLazyRef
NumpyBlockRef
```

不要默认：

```text
Pandas MultiIndex Series
```

---

# A18. FactorEngine Hot Path 去 MultiIndex

内部：

```text
timestamp array
instrument integer id
value block
```

到 API边界再构造 human-facing index。

---

# A19. InstrumentId / GroupId 整数化

高频：

```text
字符串比较
group labels
```

尽量预编码。

---

# A20. OrderingCertificate

DataAccess如果已经证明：

```text
timestamp/instrument sorted
```

FE不能每层再 sort。

---

# A21. Shared Masks

批量 factor共享：

```text
universe mask
ST mask
suspension mask
limit mask
industry codes
missing mask
```

---

# A22. Incremental Engine

不能只：

```text
全历史重算
```

最终：

```text
Source Change
→ ChangeImpactDAG
→ affected factors
→ affected time range
→ minimal recompute
```

---

# A23. Correction Propagation

数据源历史修订：

```text
financial revision
corporate action correction
minute correction
```

必须只重算受影响下游。

---

# A24. Stateful Checkpoint

递归算子：

```text
checkpoint
```

允许：

```text
增量从状态恢复
```

---

# A25. Checkpoint Semantic Version

operator状态结构改变：

```text
旧 checkpoint不得继续读
```

---

# A26. Exactly-Once Generation

计算成功不是最终成功。

只有：

```text
all block writes
DQ
manifest
atomic pointer
```

成功：

```text
generation COMMITTED
```

---

# A27. Parallel Materialization 必须 Streaming

R36已列。

最终：

```text
compute → write → release
```

不积整批。

---

# A28. Manifest-Level COW

继续 R33：

```text
未变化 partition
```

复用 object reference，
不物理复制。

---

# A29. Factor Lake Small File Control

大量 factor落盘不能产生：

```text
百万小 parquet
```

需要：

```text
factor block
date partition
compaction
```

---

# A30. Compaction 也要 Resource Autopilot

后台 compaction：

```text
BACKGROUND QoS
```

资源紧张先让路。

---

# A31. Writer Layout 应按下游模型读取优化

不仅追求：

```text
写最快
```

还要：

```text
训练批读快
```

---

# A32. Factor Matrix / Feature Store 接口

给模型侧：

```text
date range
universe
factor ids
```

直接返回：

```text
Arrow/NumPy block
```

避免一列列读 factor。

---

# A33. Training Zero-Copy Handoff

如果模型训练就在同机：

```text
FactorBlock
→ Arrow/Numpy
→ model
```

减少：

```text
Parquet readback
```

但只能在明确 snapshot/generation identity下共享。

---

# A34. Mining Cost-Aware Grammar

自动挖因子：

```text
不是每个 operator等概率
```

根据：

```text
cost class
resource condition
```

调 grammar。

---

# A35. Mining Three-Stage Search

```text
Cheap exploration
→ medium refinement
→ expensive advanced operators
```

---

# A36. Background Mining 自动让 Production

R36 QoS直接接 mining scheduler。

---

# A37. Cold-Start Library 也应携带 Cost Metadata

每个 seed factor：

```text
estimated compute cost
source dependencies
native coverage
```

---

# A38. Compile Many Incremental

算法每轮只新生成几十个 formula时：

```text
不要重新 compile整个 campaign
```

只编译 delta。

---

# A39. IR Intern Pool

相同 primitive/subtree：

```text
canonical intern
```

跨 factor batch复用。

---

# A40. Plan Cache 必须按真实 Semantic Identity

不能因为：

```text
source snapshot变化
```

还复用错误计划结果。

---

# A41. Plan Cache 与 Result Cache 分开

```text
compiled plan
```

和：

```text
computed values
```

生命周期完全不同。

---

# A42. Cache Admission 由价值而非“有空就放”

R36继续。

---

# A43. Long-Lived FactorCampaignSession

适合：

```text
AlphaProbe
FactorMiner
```

反复生成候选。

共享：

```text
snapshot
source relation
PIT joins
primitive intermediate
calibration
```

---

# A44. Campaign Session 必须有 Snapshot TTL

不能长时间跨：

```text
数据更新
```

仍认为同一 campaign。

---

# A45. Service API 不暴露底层 n_jobs 为主操作方式

用户默认：

```text
auto
```

高级 override才：

```text
max_cpu
max_memory
```

---

# A46. 推荐最终 API

```python
engine.materialize_many_auto(
    factors,
    priority="critical",
    resource_policy="auto",
)
```

---

# A47. Dry Plan API

```python
engine.plan_many_auto(factors)
```

输出：

```text
sources
snapshot
backends
estimated TTDC
P99 memory
initial resources
shards
```

---

# A48. Explain Resource

```python
engine.explain_resource_decisions(run_id)
```

---

# A49. Explain Correctness

```python
engine.explain_factor_certification(factor_id)
```

---

# A50. Explain Lineage

```text
formula
operator versions
source snapshots
backend variants
generation
```

---

# A51. Observability 三层

```text
Semantic
Physical
Resource
```

不能只打印 runtime日志。

---

# A52. Semantic Telemetry

```text
factor
PIT
operator evidence
```

---

# A53. Physical Telemetry

```text
backend region
scan
conversion
CSE
```

---

# A54. Resource Telemetry

R36：

```text
PSI
PSS
leases
controller
```

---

# A55. Metrics 标签不要高基数爆炸

factor id可能百万。

Prometheus只保留聚合，
详细 per-factor放：

```text
Parquet event log
```

---

# A56. Failure Taxonomy

统一：

```text
SEMANTIC
PIT
DATA
RESOURCE
BACKEND
WRITE
TRANSIENT_IO
```

---

# A57. Resource Failure 不等于 Semantic Failure

例如：

```text
task cannot fit
```

应该：

```text
replan
```

而不是永久禁 operator。

---

# A58. Backend Quarantine

某 fast variant：

```text
parity/runtime repeatedly fail
```

自动：

```text
quarantine
```

Reference仍可用。

---

# A59. Circuit Breaker

远端 source / backend连续故障：

```text
暂停新调用
```

避免雪崩。

---

# A60. Deadline / Cancellation

所有 stage应检查：

```text
CancellationToken
Deadline
```

---

# A61. Atomic Cancellation

已经写入 staging的 block：

```text
不 commit generation
```

---

# A62. Recovery

服务重启：

```text
RUNNING
```

任务变：

```text
INTERRUPTED
```

可根据 checkpoint/resume policy恢复。

---

# A63. Release Compatibility

FE与DA：

```text
capability handshake
```

R36已列。

---

# A64. Operator Schema Migration

operator：

```text
rename
delete
semantic_version bump
```

要同步：

```text
recipe
cold start
factor id
evidence
docs
```

---

# A65. Materialized Factor Migration

semantic version变化：

```text
新 factor generation
```

不要覆盖旧语义无标记。

---

# A66. Determinism Level

每个 backend variant声明：

```text
bitwise
numerically equivalent
rank-equivalent
```

---

# A67. Production 默认至少 Numerical Determinism

按 operator定义。

---

# A68. Dependency Pinning / Compatibility

FE当前 package：

```text
factor-engine 0.3.1
```

DataAccess当前：

```text
data-access 0.8.0
```

但 FE依赖下限：

```text
data-access>=0.2.0
```

随着深度耦合增加，应使用：

```text
capability protocol
```

并在 CI组合测试真实支持版本。

---

# A69. CI Matrix 不要无限版本组合

维护：

```text
minimum supported
current production
latest supported
```

三档。

---

# A70. Python / NumPy / DuckDB / Polars 版本升级

每次：

```text
correctness parity
performance recalibration
```

---

# A71. Modin 不作为主生产优化路线

如果当前没有独立真实优势 benchmark：

```text
保持 optional research
```

不要为了“并行 Pandas”引入第二套执行架构。

主线：

```text
DuckDB / Polars / NumPy / Numba
```

---

# A72. TA-Lib 同样只作为 Optional External Backend

不能成为：

```text
canonical语义唯一来源
```

Reference仍自己可证明。

---

# A73. Security / Data Governance

FactorEngine若以服务运行：

```text
用户
数据集
写 target
```

权限必须沿用 DataAccess security context。

---

# A74. Resource Priority 不能越权

高 priority只表示：

```text
资源排序
```

不表示能访问更多数据。

---

# A75. Multi-Tenant Fairness

不同用户的 background job：

```text
weighted fair queue
```

避免一个用户长期占满。

---

# A76. Per-Principal Quota

已有基础，
继续接 HostResourceCoordinator。

---

# A77. DR / Rollback

继续 R32：

```text
catalog
manifest
factor lake
```

备份/恢复演练。

---

# A78. Canary Release

新 optimizer：

```text
shadow calculate
```

与 reference比。

---

# A79. Shadow Backend

例如新 Numba kernel：

```text
一部分 run同时 reference
```

采样 parity。

---

# A80. Online Correctness Sentinel

生产低比例：

```text
fast vs reference sample
```

长期监控 drift。

---

# A81. Sampling 不要把性能拖死

例如：

```text
0.1% factor blocks
```

---

# A82. Chaos Tests

除资源外：

```text
source snapshot changes
worker crash
network error
disk full
process restart
```

---

# A83. Fuzz Tests

重点：

```text
NaN
Inf
ties
tiny sample
extreme values
parameter boundaries
```

---

# A84. Mutation Tests

继续 R30/R34：

```text
future leak
off-by-one
missing gap bridge
```

测试必须 kill。

---

# A85. Full Current-HEAD Release Suite

不能再：

```text
历史 commit测试通过
```

代替当前代码。

---

# A86. CI Evidence 和 Code SHA 强绑定

```text
evidence.code_sha
==
release_sha
```

---

# A87. Benchmark 也强绑定 SHA

以及：

```text
hardware fingerprint
```

---

# A88. Performance Regression Budget

例如：

```text
1000 daily TTDC
```

退化超过阈值：

```text
CI/release warn/fail
```

---

# A89. 不要只 Benchmark Synthetic

至少：

```text
synthetic
real A-share representative
model-heavy
minute-heavy
```

---

# A90. Real Dataset Benchmark 要固定 Snapshot

否则不可比较。

---

# A91. Capacity Benchmark

不同 server：

```text
8/16/32/64 cores
16/64/256GB
```

生成 scaling curve。

---

# A92. Scaling Efficiency

看：

```text
2x CPU
```

到底有没有：

```text
接近2x throughput
```

---

# A93. 如果没有 Scaling

先 profile：

```text
IO
serialization
global lock
memory bandwidth
```

而不是加机器。

---

# A94. Flamegraph / Profiler

每个 release performance audit：

```text
CPU profile
allocation profile
query profile
```

---

# A95. Python Allocation

重点找：

```text
DataFrame copy
column_stack
unstack
concat
dict clone
```

---

# A96. DuckDB Query Profile

找：

```text
scan
join
sort
window
spill
```

---

# A97. Polars Plan Explain

确认：

```text
projection pushdown
predicate pushdown
streaming
```

---

# A98. DataAccess Remote Profile

看：

```text
object count
HEAD/LIST
range reads
bytes
```

---

# A99. Storage Layout Feedback Loop

如果 benchmark发现：

```text
row-group pruning差
```

可能应调整：

```text
Parquet partition / row group
```

---

# A100. Factor Lake Layout 也要反馈

训练读取慢：

```text
重新调整 block layout
```

---

# A101. 不要提前分布式

再次强调：

```text
single-node elastic execution
```

先做到极致。

---

# A102. 多机后仍复用同一 Resource Autopilot

每 node自己治理。

---

# A103. Cluster Scheduler 只分配 Job/Shards

不要微管理每个 operator。

---

# A104. GPU 只在实测必要时进入

不作为“看起来高级”的功能。

---

# A105. 最终 Master Definition of Done

整个 FactorEngine 真正成熟时：

```text
1. 所有 retained operator 角色明确
2. 所有 direct-use operator证据完整
3. 所有 model timing明确
4. PIT/source/universe/calendar完整
5. FE×DA只有一个 physical data truth
6. batch共享扫描/中间计算
7. fast kernel都有 reference parity
8. 异构服务器自动资源伸缩
9. 同机共存无 OOM
10. 压力解除自动恢复吞吐
11. incremental/correction最小重算
12. materialization atomic/exactly-once
13. current-head full suite green
14. current-head benchmark达标
15. canary/shadow/rollback完善
```

---

# 附录 B｜建议 coding AI 实施总顺序

不要并行乱改所有层。

## B0｜冻结语义基线

```text
R34/R35已确认 semantic truth
```

---

## B1｜先修 R36 P0 Resource Bugs

```text
max_concurrency
upshift
backpressure
raw cache bypass
writer failure
service broker
true peak
```

---

## B2｜HostResourceCoordinator

统一：

```text
FE
DA
service
writer
```

---

## B3｜R33 Data Physical Integration

```text
one DataRequest authority
real read wave
real physical stages
```

---

## B4｜AutoShard + Buffer Liveness

避免大 task无法执行。

---

## B5｜R35 Fast Kernels

```text
Numba
rolling linear
model shared state
```

---

## B6｜Streaming Materialization

```text
FactorBlock
COW
atomic generation
```

---

## B7｜PSI + Calibration + Autotuner

在基础资源账本正确后再上。

---

## B8｜全量 Tests

```text
correctness
resource
co-tenancy
failure injection
```

---

## B9｜Benchmarks

```text
TTDC
scaling
memory
```

---

## B10｜Canary / Shadow

最后发布。

---

# 附录 C｜最终默认用户体验

理想情况下，调用方不应该写：

```python
engine.materialize_many(
    factors,
    n_jobs=12,
    wave_memory_budget=4 * GB,
    duckdb_threads=4,
)
```

而应该只写：

```python
engine.materialize_many_auto(
    factors,
    priority="critical",
)
```

系统自己：

```text
检测机器
检测共存负载
选择 backend
选择并行度
选择 wave/block
必要时 shard/spill
动态让路
动态恢复
落盘
```

如果用户确实想限制资源：

```python
engine.materialize_many_auto(
    factors,
    priority="background",
    max_memory="32GB",
    max_cpu=8,
)
```

这些只是：

```text
hard ceilings
```

而不是要求用户负责调优。

---

# 附录 D｜本轮审计的工程结论

当前 HEAD 已经不是一个“完全没有资源治理”的 FactorEngine。

已有：

```text
cgroup/SLURM/RLIMIT aware
MemAvailable
RSS/PSS
TaskResourceContract
ResourceBroker
MemoryGovernor
spill budget
bounded sink
adaptive scheduler
```

这是不错的底座。

但距离用户要求的最终状态仍有关键差距：

```text
1. scheduler动态目标没有真正闭环
2. 固定 memory budgets仍然较多
3. 只有 throttle没有完整 recovery
4. FE/DA resource truth仍分裂
5. cache存在治理旁路
6. writer pressure未反向控制compute
7. physical stages尚未真正执行
8. true per-run/per-task peak evidence不足
9. auto-shard/replan不足
10. co-tenancy pressure还未用 PSI闭环
```

所以未来最重要的系统升级不是再堆更多“资源配置参数”，而是：

> **把资源治理从配置系统升级成一个实时反馈控制系统。**
