# FactorEngine R37：最新 HEAD 剩余系统性风险终审与下一阶段全量整改方案

> **执行基线：** `main@c2309dbd4abe73945d8f1c97b402b8b4cbdc75b6`  
> **仓库：** `18047533889/quant_projects`  
> **范围：** `factor_engine/` + `dataaccess/` 及二者之间的编译、数据、执行、资源、缓存、物化、增量、证据和发布边界。  
> **性质：** 本文是下一阶段 coding agent 的直接整改任务书，不是设计建议清单。凡本文标为 P0/P1 且当前代码确实存在的问题，应修改真实实现、补测试、生成当前 HEAD 证据并完成验收。

---

# 0. 总指令

执行前先读取最新 `main`。本文审计基线是 `c2309dbd4abe73945d8f1c97b402b8b4cbdc75b6`；若执行时 HEAD 已变化，先记录新 SHA，再逐项判断本文问题是否仍成立，不得把已经修复的旧问题重复实现。

本轮目标不是继续堆算子数量，而是把 FactorEngine + DataAccess 推进到：

```text
语义正确
→ 数据 PIT 正确
→ 参数域正确
→ 优化前后等价
→ 批量执行真正复用
→ 资源治理真正闭环
→ 物化 exactly-once
→ 增量修订可追踪
→ current-HEAD 证据可信
→ 异构服务器和共存负载下长期稳定运行
```

必须遵守：

1. 先修真实实现，再写 gate。
2. 不允许 hardcode True、presence check、source-string check 伪造通过。
3. 不允许把本可修复的 production 能力简单降级成 research 作为“修复”。
4. 有价值、可修复的算子优先修复并保留；随机、未来函数、非因果、明显危险且无合理用途的算子继续遵循物理删除/安全 tombstone 原则。
5. backend 只有通过 `canonical + semantic_version + parameter_domain + source_context + execution_variant` 认证后才能进入 production route。
6. Pandas 可作为 reference/fallback/debug，但一个 unsupported node 不能把整棵 root 静默拖回 Pandas。
7. 性能最终看 **correctness-qualified Time To Durable Commit (TTDC)**，不看脱离真实链路的单算子 microbenchmark。
8. current-HEAD evidence 必须绑定 `commit_sha`、component hash、operator semantic version、source contract、backend、variant、parameter domain。
9. 最终不得只回复“测试通过”；必须给 issue closure ledger、最终 SHA、CI、evidence、benchmark、remaining exceptions。

---

# 1. 当前 HEAD 已经修掉、不得倒退的内容

## 1.1 AdaptiveBatchScheduler

当前 HEAD 已经具备：

- `max_concurrency` 被实际消费；
- scheduler 使用动态并发上限；
- sink `backpressure_ratio` 已进入动态并发限制；
- read wave 已进入主执行路径；
- SOURCE_SCAN 已可真实执行；
- virtual/barrier planning stage 不占真实 resource lease；
- fusion group 可走 multi-root/native fusion；
- ready task 有 priority；
- `FIRST_COMPLETED` 事件驱动；
- lease release 已做幂等治理。

不要重写第二套 scheduler。本轮只闭环资源自动恢复、auto-shard/replan、真实 physical-stage runtime 和统一资源权威。

## 1.2 StreamingResultSink

当前 HEAD 已经有：

- `deque`；
- writer 状态机；
- finite retry budget；
- fatal propagation；
- accepted/committed/failed durable check；
- partition-aware routing；
- backpressure signal。

不要重复“把 list 改 deque”之类整改。本轮重点是 writer 生命周期、全局 byte budget、fatal 对 scheduler 即时反压、atomic generation 和 IO lease。

## 1.3 Model timing

当前 HEAD 已经：

- 增加 `model_contract.py`；
- production 对 model-like operator 要求 explicit reviewed timing；
- name/family default timing 只能作为 research hint；
- GARCH/GJR 已改成严格 prior fit 语义；
- panel PCA coverage、stable instrument identity sign tie-break 等已有改进。

不要把这些已修问题重新当 R37 bug。R37 要做的是把最终允许 production/mining 的 model operator 证据闭环。

## 1.4 DataAccess R29

当前 DataAccess 已加强 snapshot/source resolver、security、schema epoch、read session、DuckDB semaphore、remote request/concurrency 分离、aggregate+join shared relation 主路径、frozen effective join specs、generation/COW 等。

不要重新引入第二套 join compiler、第二套 snapshot resolver 或重复 semaphore。

---

# 2. R37 总体 Definition of Done

只有同时满足以下条件，本轮才算完成：

```text
A. 所有 production canonical 都有真实 correctness evidence；
B. production call 的具体参数点处于 certified parameter domain；
C. PIT/availability/universe/calendar/price basis/revision identity 全链路可追踪；
D. optimized/fused/batched/chunked/incremental 与 reference 等价；
E. FE 与 DA 只有一个主资源权威，不存在 double admission；
F. 外部 CPU/RAM/IO 压力上升自动收缩，压力解除自动恢复；
G. 无法 fit 的大 task 自动合法 shard/replan，不同尺寸重试 OOM；
H. CSE/cache/buffer 全部经过治理，不允许 raw dict 绕过资源账本；
I. physical DAG 的关键 stage 真正执行，不只是 planning metadata；
J. writer failure/cancel/OOM/disk-full/SIGTERM 下不产生 mixed generation；
K. 数据修订通过 ChangeImpactDAG 找到最小重算范围；
L. stateful checkpoint 与 source identity 绑定，历史修订正确失效；
M. current-HEAD CI/evidence/benchmark SHA 完全一致；
N. evidence hard gates 能被负控真正打红；
O. 1000+ mixed-source factors 有当前 HEAD TTDC/peak PSS/scan reuse/write amplification 基准；
P. 1~5 个因子的小批量无明显性能回归。
```

---

# 3. Evidence Truth 与生产准入

## R37-P0-001：重做 Evidence Truth Gate 的最终逻辑

当前 `factor_engine/scripts/audit_r34_evidence.py` 虽然已比早期硬编码 gate 好，但仍可能出现 false-confidence：

- 定义了 current-bound case helper，但核心 gate 未真正完全依赖它；
- `ZERO_PRESENCE_ONLY` 类 gate 仍可能只证明脚本/字段存在，而不是“0 个 presence-only gate”；
- stale artifact 检查必须严格证明 `bound_sha == current HEAD`，不能只证明 `bound_sha` 非空；
- component hash、fixture hash、实际 executed case 都必须参与 freshness；
- source-string-only/tautological gate 需要专门负控，不应由 literal-gate scanner 代替。

建立统一 `EvidenceTruthEngine`，建议在现有 evidence 包中整合，不要重复 truth source。每个 gate 至少必须记录：

```python
GateResult(
    gate_id,
    status,
    executed_cases,
    passed_cases,
    failed_cases,
    case_ids,
    commit_sha,
    component_hashes,
    fixture_hashes,
    evidence_files,
    generated_at,
)
```

硬规则：

```text
executed_cases == 0         => NOT_RUN
failed_cases > 0            => FAIL
bound_sha != current HEAD   => STALE/FAIL
component hash mismatch     => STALE/FAIL
fixture mismatch            => STALE/FAIL
```

禁止把以下作为 correctness hard gate 的最终证明：

```python
passed = True
ok = Path(...).exists()
ok = "keyword" in source
ok = bool(bound_sha)
```

### 必须增加负控

CI 自动构造/注入：

- `ts_mean` 改成 `ts_sum` → semantic gate 红；
- `shift(1)` 改 `shift(-1)` → PIT gate 红；
- backend 输出 reorder → parity 红；
- evidence SHA 改旧 SHA → freshness 红；
- 删除真实 case execution 只保留 JSON → NOT_RUN/FAIL；
- gate 改 literal True → negative-control 红。

## R37-P0-002：建立 Per-Canonical Correctness Ledger

每个 retained canonical 一行，不允许只靠全局 test count。

建议：

```text
docs/evidence/r37/R37_OPERATOR_CORRECTNESS_LEDGER.parquet
```

字段至少：

```text
canonical
semantic_version
surface
role
status
source_module
backends
parameter_domain_status
semantic_golden_status
pit_status
source_pit_status
backend_parity_status
batch_parity_status
chunk_status
incremental_status
checkpoint_status
optimizer_diff_status
numeric_stress_status
shape_status
null_inf_status
determinism_status
current_sha
evidence_hash
final_production_ready
reason
```

`final_production_ready` 必须由真实子 gate 推导，不得人工填 True。

## R37-P0-003：所有 production canonical 必须有 independent semantic oracle

优先级：

1. 手工数学 reference；
2. 独立 slow NumPy/SciPy reference；
3. analytical fixture；
4. property/invariant；
5. 对复杂模型使用多重独立 invariant。

生产 kernel 不能自己给自己生成 expected。

## R37-P0-004：property-based testing 进入核心算子族

至少覆盖：

```text
rank: permutation equivariance、tie deterministic、monotonic transform
zscore: affine invariant、constant window
corr: symmetry、corr(x,x)=1、corr(x,-x)=-1
rolling max/min: window containment invariant
EMA/Wilder/KAMA: constant input、full/chunk equivalence
neutralize: residual orthogonality under declared metric
regression: synthetic known beta/intercept
```

## R37-P0-005：mutation testing 进入 release gate

优先覆盖 rolling、rank/neutralize、regression、financial PIT、universe/group、stateful、model、minute→daily、availability clock。关键 PIT/数学 mutation 必须被杀死。


---

# 4. 参数域认证：从“算子通过”改成“具体调用被认证”

## R37-P0-006：当前参数域认证覆盖远远不够

当前 R34 parameter-domain audit 主要覆盖少量基础 rolling/cross-section 算子。自动挖掘真正调用的是：

```text
canonical + concrete parameter values
```

而不是 canonical 默认参数。

建立 `ParameterDomainCertificationStore`，认证 key 至少：

```text
canonical
semantic_version
backend
execution_variant
source_context
parameter_point / certified_region
dtype
grain
```

对每一个 production operator：

1. 读取 `ParamSpec`；
2. 生成 boundary / near-boundary / representative / random-valid / invalid 参数；
3. 跑 independent oracle/reference/invariant；
4. 记录精确通过域；
5. production runtime 对实际 call 做 membership check。

## R37-P0-007：修正 “bool(passed)” 式过度认证

不能因为测试点中“至少一个通过”就把整个 operator 标成 certified。

区分：

```text
operator_has_any_certified_region
exact_call_is_certified
```

production 准入只能看第二个。

离散认证集如果规定 7 个点构成一个公开 certified set，必须全部通过；若只认证其中 5 个，则只能记录这 5 个精确点/区间，不能扩大声明。

## R37-P0-008：参数矩阵必须覆盖所有真正可搜索参数

至少包括：

- `window`
- `lag`
- `min_periods`
- `ddof`
- `q`
- `alpha`
- `l1_ratio`
- `n_components`
- `component`
- `label_horizon`
- `n_regimes`
- `n_experts`
- thresholds/bounds
- decay/half-life
- regression regularization
- state-machine knobs
- minute aggregation/session knobs
- group/cross-section policy knobs

有效与无效值都测：

```text
0
negative
1
small
default
large
extreme-but-valid
fractional value for int param
NaN
Inf
wrong type
bool-as-int
```

## R37-P0-009：production admission 必须消费参数域证据

不能只生成 JSON/CSV。

在 production compile/execute 前真正调用：

```python
assert_parameter_point_certified(
    canonical,
    kwargs,
    backend,
    execution_variant,
    source_context,
)
```

行为：

```text
production + uncertified point => fail closed
research + uncertified point   => allow + telemetry
```

---

# 5. Shape / Grain / Ordering Contract

## R37-P0-010：OperatorSpec.output_shape 要真正落地

当前 `OperatorSpec` 已有 `output_shape: OutputShapeContract | None` 设计，但必须确保 `build_operator_spec()` 从唯一 authoritative metadata/catalog 真正构造它，而不是字段存在但实例始终 None。

统一链路：

```text
OperatorMetadata / typed semantic contract
        ↓
OutputShapeContract
        ↓
OperatorSpec
        ↓
Analyzer
        ↓
runtime output validator
        ↓
backend parity
```

必须覆盖：

```text
daily panel -> daily panel
minute -> daily
event table -> entity-date panel
snapshot -> daily
scalar/global -> broadcast panel
group -> panel
```

production 对 shape-changing op：

```text
无显式合法 OutputShapeContract => fail closed
```

同时检查：

- index identity；
- columns identity；
- order；
- grain；
- timezone；
- duplicate key；
- broadcast axis。

---

# 6. PIT / Universe / Calendar / Corporate Action / 财务双时态

## R37-P0-011：建立统一 DataKnowledgeIdentity

建议统一：

```python
@dataclass(frozen=True)
class DataKnowledgeIdentity:
    dataset_id: str
    snapshot_id: str
    schema_epoch: str
    market: str
    calendar_id: str
    universe_snapshot_id: str
    price_basis: str
    corporate_action_vintage: str
    source_revision_id: str
    availability_policy_version: str
    timezone: str
```

这个 identity 必须进入：

- cache key；
- FactorId；
- CSE identity；
- checkpoint identity；
- materialization manifest；
- lineage；
- evidence；
- incremental invalidation。

## R37-P0-012：Universe 必须是 PIT 对象

禁止把：

```text
Universe="CSI300"
```

直接等价成今天的一张静态成员列表去回算历史。

需要：

```text
UniverseMembership(valid_time, knowledge_time)
```

测试：

- 成分加入前不能出现；
- 成分退出后不能由今天成员表重写过去；
- universe revision 能触发受影响日重算；
- same expr + different universe => 不同 FactorId/cache identity；
- cross-sectional rank/neutralize 使用当日 effective universe。

## R37-P0-013：SessionCalendar 成为统一时间权威

统一：

- open/close；
- lunch break；
- auction；
- half day；
- holiday；
- DST；
- market timezone；
- bar label；
- bar knowledge time；
- daily-close availability；
- minute→daily session boundary。

禁止业务逻辑中散落固定 `bars_per_day`、固定 09:30/15:00 等作为跨市场通用语义。

## R37-P0-014：Corporate Action / PriceBasis 进入 semantic identity

明确：

```text
raw
forward_adjusted
backward_adjusted
total_return
```

以及 corporate-action 的 vintage/knowledge time。

不同 price basis 必须：

- 不共 cache；
- 不共 CSE；
- 不共 checkpoint；
- 不共 FactorId；
- lineage 明确记录。

## R37-P0-015：财务数据实现真正双时态

统一：

```text
valid_time     = report_period
knowledge_time = announce/ingest/revision availability
```

必须处理：

- initial filing；
- revision；
- restatement；
- late filing；
- duplicated report period；
- quarter/TTM；
- fiscal year mismatch；
- cross-market accounting calendar。

任何基本面 operator 不得仅按 report_period shift 代替 PIT。

## R37-P0-016：Availability 必须是 DAG 组合属性

最终 factor availability 应来自：

```text
source availability
+ temporal transform
+ operator availability
+ model fit/label maturity
+ decision time
```

而不是一个 `pit_safe` tag。

增加 future perturbation：改变 t+1 以后数据，t 及以前 production output 必须不变。

---

# 7. FE × DataAccess：统一 QueryGraph 和真实 Physical Execution

## R37-P0-017：结束“planning view + whole-root execute”双重现实

当前 `physical_lowerer.py` 已能生成丰富 stage，但 production 仍存在 ROOT 执行完整 root plan、部分 stage 只作为 planning metadata 的设计。

风险：

- stage cost/lease 与真实执行边界不一致；
- SOURCE_SCAN 虽执行，ROOT 仍可能重新触发加载/转换；
- barrier/operator stage 的 metrics 不是实际运行时；
- optimizer 看见的是 stage，backend 真跑的是另一条路径；
- resource prediction 无法精准校准到实际 stage。

目标：

```text
UnifiedPhysicalPlan
    SourceScanStage
    TemporalJoinStage
    NormalizeStage
    NativeRegionStage
    RollingStateStage
    CrossSectionStage
    GroupStage
    ModelKernelStage
    DQStage
    WriteStage
```

每个 executable stage：

- 消费 `BufferRef/RelationRef`；
- 返回 `BufferRef/RelationRef`；
- 持有真实 resource lease；
- 记录真实 elapsed/peak/io；
- 支持 cancel/failure；
- 可作为 optimizer boundary。

ROOT 不再无条件重新执行完整 logical plan。

## R37-P0-018：FE planner 与 DA physical planner 合并成一张 QueryGraph

当前仍有：

```text
FE: BatchDataRequest / ReadWave / PhysicalLowerer
DA: DataRequest / PreparedRead / PhysicalPlan
```

不能长期维护两套 physical planning truth。

目标：

```text
Factor DSL
→ FE semantic IR
→ SourceRequirementIR
→ DataAccess prepare/resolve exact snapshot
→ DA physical source/join graph
→ FE native compute regions
→ unified executable graph
```

FE 不应自己重写 DA 的 temporal join、schema epoch、snapshot、universe source resolution、authorization、remote object resolution；DA 也不应猜 FE operator dependency。

## R37-P0-019：PreparedBatchReadSession 成为批量 source truth

1000 factors：

```text
compile all
→ union source requirements
→ one pinned snapshot session
→ clustered scan waves
→ shared relations/buffers
→ many factor compute regions
```

要求：

- snapshot一次固定；
- auth scope一次固定；
- schema epoch一次固定；
- common filters一次；
- same temporal join一次；
- common minute aggregation一次；
- projection union后再按 downstream stage裁剪。

## R37-P1-020：Source CSE 与 Compute CSE 分离

```text
SourceCSEKey
ComputeCSEKey
```

Source key 至少包含：

- dataset；
- snapshot；
- columns；
- filters；
- time range；
- instruments/universe；
- PIT join policy；
- aggregation；
- normalization；
- price basis。

避免不同 source scope 因“字段名相同”误共享。

## R37-P1-021：backend region routing 以连续 native region 为核心

顺序：

```text
hard semantic eligibility
→ candidate native regions
→ end-to-end cost
→ choose route
```

DuckDB 可以作为 DA-centric parquet/relational workload 的优先候选/tie-breaker；纯 columnar/lazy DAG 允许 Polars 胜出；递归/数值模型走 specialized kernel。

不能静态写死 `DuckDB > Polars > Pandas`，也不能一个 unsupported op 让整个 root 回退。

---

# 8. R36 资源治理真正落地

## R37-P0-022：建立唯一 HostResourceCoordinator

最终只允许一个进程级资源事实源：

```text
HostResourceCoordinator
│
├── JobLease
├── ComputeLease
├── DataAccessScanLease
├── DuckDBLease
├── PolarsLease
├── BLASLease
├── NumbaLease
├── CacheLease
├── BufferLease
├── SpillLease
└── WriterLease
```

FactorEngine ResourceBroker、MemoryGovernor、service queue 与 DataAccess GlobalResourceGovernor 应通过同一 coordinator 协调。

兼容 facade 可以保留，但不能有相互不知道的独立 memory/CPU/IO admission。

## R37-P0-023：修复 service broker 的 plain-global 竞态

当前 `service/queue.py` 的 `_service_broker_ctx` 是模块全局，多个 worker thread 手工 save/restore 会互相覆盖。

改成：

- 主资源权威：process-wide HostResourceCoordinator singleton / explicit injection；
- 当前 job：若需要，使用真正 `ContextVar[JobLease]`；
- A job 退出不得让 B job 内部 scheduler 看不到 coordinator。

并发测试：

```text
A + B 并发
A先退出
B继续内部 run_many
=> B仍看到同一 coordinator 和自己的 JobLease
```

## R37-P0-024：Fast Down, Slow Up

当前已有压力下调，但必须增加恢复路径。

建议：

```text
PRESSURE:
  task concurrency *= 0.5
  shrink wave/block
  stop speculative prefetch
  evict low-value cache

NORMAL stable N samples:
  +1 task concurrency
  +10% wave
  +10% block
  gradual cache target recovery
```

增加 hysteresis + cooldown，防振荡。

硬验收必须有“压力解除后吞吐恢复”，不能只证明不 OOM。

## R37-P0-025：废除固定 2GB/worker、3GB/worker 作为主策略

当前仍可见：

- recommended concurrency 约 2GiB/worker；
- cold-start worker peak 3GiB；
- 4GB read wave；
- 4GB result queue。

这些仅可作为 unknown workload 的保守 fallback。

建立：

```text
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
  group_count_bucket
)
→ P50/P90/P95/P99/max/sample_count
```

production admission 使用 P99。

## R37-P0-026：ResourceCalibrationStore

每个 task 结束后更新：

- actual elapsed；
- run-isolated peak family PSS/RSS；
- output bytes；
- spill bytes；
- read/write bytes；
- backend threads；
- cache reuse；
- shard shape。

按 hardware/software fingerprint 分开：

```text
CPU model
CPU quota/affinity
RAM/cgroup limit
NUMA count
storage/fs class
DuckDB/Polars/NumPy/BLAS/Numba versions
```

## R37-P0-027：真正 run-local Peak Sampler

不能拿 process lifetime `ru_maxrss` 训练每 task memory model。

采样：

```text
baseline family PSS/RSS
periodic samples
peak
task end
```

尽可能区分 task attributable delta 与共享 cache/writer/DA footprint。

## R37-P0-028：加入 Linux PSI / memory slope / cgroup events

Linux 可用时：

- `/proc/pressure/cpu`；
- `/proc/pressure/memory`；
- `/proc/pressure/io`；
- cgroup PSI；
- `memory.events`；
- swap；
- MemAvailable slope。

预测式 admission：若当前 headroom 下降速度足以在候选任务 P90 runtime 内穿过 reserve，不要启动新 task。

非 Linux 回退现有 metrics。

## R37-P1-029：disk pressure 不得固定 500MB/s

优先使用 PSI IO stall、busy_time delta、latency、小型安全 calibration、storage class；500MB/s 只能是无信息 fallback，并标记 `uncalibrated`。

## R37-P0-030：CPU tokens 必须匹配真实 backend threads

当前某些 physical context/resource contract 仍倾向 `cpu_tokens=1/backend_threads=1`。

必须满足：

```text
lease cpu_tokens >= task actual native thread budget
```

DuckDB threads 由 lease 给出；BLAS/OpenMP 用 threadpoolctl；Numba 默认 outer scheduler 并行，inner `parallel=False`，若使用 `prange` 必须申请多 token。

## R37-P0-031：Polars 不做 task级全局线程池修改

Polars pool 采用 process fixed capacity，HostResourceCoordinator 控制并发 workload 数量、block/chunk/streaming。不要在并发 task 中靠改全局 env 假装动态线程治理。

## R37-P0-032：ExecutionResourceScope 不得并发修改 process-global 设置

任何 `os.environ`、global DuckDB/Polars/native thread state 若是 process-global，不能仅锁住 enter/exit 后让多个 workload 并发运行。

改为 connection-local、worker-local、startup immutable config 或 lease-driven API；必要兼容路径可串行整个 scope lifetime，但不得产生竞态。


---

# 9. AutoShard / Replan / OOM Recovery

## R37-P0-033：无法 admission 的 task 不能只等待到 STUCK

如果：

```text
P99 task memory > current SafeEnvelope
```

运行前触发 legal sharding，而不是重复进入 ready queue等待。

按 semantic category：

```text
elementwise:
  time or asset shard

time-series rolling:
  asset first
  time shard requires warmup overlap

cross-sectional:
  time shard only
  禁止 asset shard破坏同日截面

group:
  time shard优先

stateful recursive:
  asset优先
  time shard only with checkpoint contract

PCA/panel models:
  preserve full cross-section
  time block + history overlap

minute→daily:
  session/day shard
```

AutoShardPlanner 必须读取 OutputShape/Stateful/Model contract，不能按名称猜。

## R37-P0-034：OOM 不允许原尺寸原配置重试

OOM 分类成：

```text
ResourceUnderpredictionError / OOMReplanRequired
```

处理：

1. 记录 underprediction；
2. 降低 target concurrency；
3. 缩 read wave/block；
4. legal shard；
5. 选择 spill/recompute；
6. 更新 calibration；
7. 再执行。

禁止 identical retry。

---

# 10. Buffer / Cache / CSE：消灭治理旁路

## R37-P0-035：raw shared_result_cache 写入必须退出 production hot path

当前 `_materialize_shared_subplan` 仍可直接：

```python
ctx.shared_result_cache[sid] = value
```

这会绕开 ExpressionCache/MemoryGovernor byte budget。

建立或整合 `GovernedBufferStore`：

```python
class GovernedBufferStore:
    def put(self, key, value, *, lease, identity, reuse_hint): ...
    def get(self, key): ...
    def pin(self, ref): ...
    def release(self, ref): ...
    def spill(self, ref): ...
    def reload(self, ref): ...
    def drop(self, ref): ...
```

所有大对象进入 CSE/panel/materialized buffers 都必须经过资源 accounting。

## R37-P0-036：ExecutionContext 不再暴露任意写大对象 raw dict

逐步改为 typed handles：

```text
buffer_store
relation_store
materialization_store
runtime_stats
```

兼容期可给 read-only mapping view，但 production 禁止直接赋值。

## R37-P0-037：cache governor registration 失败不得 production fail-open

当前 `cache/session.py` 对 register/unregister 仍存在 `except Exception: pass` 风格。

production：

```text
registration fail => fail closed
```

research：

```text
warning + telemetry
```

release 后增加 accounting reconciliation：

```text
declared cache bytes
actual retained refs
governor accounted bytes
```

不一致必须告警；production 超过阈值应 fail。

## R37-P1-038：CSE 加入 Memory Rent

CSE materialization 决策：

```text
Benefit =
SavedCompute
- MaterializeCost
- ConversionCost
- SpillCost
- MemoryRent
```

MemoryRent 由 bytes、lifetime、pressure、reuse distance、recompute cost 决定。

## R37-P1-039：cache key 避免 hot path 全内容 O(N) hash

对 immutable source/buffer 使用：

```text
BufferIdentity
SourceSnapshotId
column slice
time slice
universe
semantic transform hash
```

完整内容 hash 只在需要强校验/落盘 checksum 时使用，不要每次 panel lookup 都扫描全部 values/index。

---

# 11. Streaming Materialization / Writer / Exactly-Once

## R37-P0-040：writer queue byte budget 改成全 sink budget

多 writer 不能每个 queue 都拥有相同的 4GB 上限，导致总预算放大 N 倍。

定义：

```text
sink_total_budget
→ per-worker fair allocation
→ coordinator动态调整
```

并把 queue bytes 纳入 HostResourceCoordinator。

## R37-P0-041：修复 finish 的 writer-thread timeout 竞态

如果：

```python
t.join(timeout=10)
```

之后线程仍 alive，主线程不能直接 drain 同一 queue 再补写，否则可能形成并发写/重复写。

必须：

```text
close queues
→ join
→ verify every writer terminated
→ if alive: fatal/cancel/abort generation
→ only then finalize/drain
```

## R37-P0-042：sink.submit 失败必须即时停止 scheduler 新 admission

scheduler 不应忽略 submit Boolean。

```text
writer fatal / queue closed / submit timeout
→ WriterLease failed
→ no new compute admission
→ abort generation
```

## R37-P0-043：compute→DQ→write→release 真正流式

大 batch：

```text
Read N+1
Compute N
DQ N
Write N-1
Release N-2
```

不能先形成整个 `run_many_parallel` results 再物化。

production `materialize_many` 根据 predicted result footprint 自动从 return mode 切到 streaming/sink mode。

## R37-P0-044：Generation commit 原子化硬闭环

```text
staging generation
→ every block written
→ block DQ
→ manifest checksum
→ read-after-write verification
→ atomic pointer/generation publish
```

任一失败：

```text
published generation保持旧完整版本
```

## R37-P0-045：writer/generation failure injection

至少覆盖：

- disk full；
- permission denied；
- partial parquet；
- writer thread hang；
- process kill；
- SIGTERM；
- duplicate partition；
- manifest write fail；
- checksum mismatch；
- rename fail；
- remote write failure；
- cancellation mid-write。

最终不能产生 mixed generation。

---

# 12. Incremental / Revision / ChangeImpactDAG

## R37-P0-046：建立 ChangeImpactDAG

输入：

```python
SourceChange(
    dataset,
    old_snapshot,
    new_snapshot,
    affected_fields,
    affected_instruments,
    valid_time_range,
    knowledge_time,
    revision_kind,
)
```

传播：

```text
source nodes
→ transforms
→ CSE/model states
→ factors
→ checkpoints
→ materialized partitions
```

输出最小重算范围。

## R37-P0-047：支持不同 revision 类型

至少：

```text
new day append
late minute bars
corporate-action correction
financial restatement
industry classification correction
universe membership correction
schema migration
source file replacement
event correction
```

每种 revision 的传播规则必须显式。

## R37-P0-048：rolling correction impact 不能只重算被改那一天

若 t 日输入被修订，window=W：

```text
通常影响 [t, t+W-1]
```

stateful 可能影响全部后续直到合法重置；模型影响到后续 refit windows。由 operator impact contract 计算，不按统一常数猜。

## R37-P0-049：checkpoint 绑定完整 input identity

现有 stateful framework 已有 schema/version/fingerprint，应扩展 fingerprint 纳入：

- source snapshot；
- schema epoch；
- universe；
- calendar；
- price basis；
- operator semantic version；
- parameters；
- upstream IR hash。

source correction 后旧 checkpoint 必须自动失效。

## R37-P0-050：所有 stateful production op 做四路等价

硬 gate：

```text
full history
== chunked
== incremental append
== checkpoint resume
```

随机 cut points、多 instruments、NaN gap、停牌、不同 chunk size、不同 shard 都测。

不兼容 checkpoint：fail closed → full replay，不得静默继续。

---

# 13. Model Operators 最终生产收口

## R37-P0-051：ModelOperatorContract 覆盖最终 production/mining model set

当前已有 framework，但最终允许 production/default mining 的 model operator 必须全部有显式 contract：

```text
role
feature params
label param
label maturity
fit cutoff
score cutoff
scaler cutoff
hyperparameter cutoff
min train obs
missing policy
convergence policy
refit policy
stateful/checkpoint
cost class
resource shape
```

experimental/research model 可以继续非 production，但不得进入默认 production grammar。

## R37-P0-052：Model Test Obligations

每个生产 model 至少：

- fit cutoff perturbation；
- label maturity；
- scaler leakage；
- hyperparameter leakage；
- convergence failure；
- singular/rank deficient；
- missing block；
- all constant；
- reordered universe；
- deterministic rerun；
- backend/kernel parity；
- batch parity；
- future perturbation；
- incremental/chunk；
- parameter-domain；
- numeric stress。

## R37-P1-053：FastLinearWindowEngine

把可共享的 rolling sufficient statistics 统一：

```text
X'X
X'y
y'y
sum X
sum y
count
```

服务 OLS/Ridge/beta/HAR/rolling regression families，避免每个 row重复构造 matrix + `lstsq`。

必须保留 slow reference parity。

## R37-P1-054：Model-family Shared State

一次计算，多输出：

```text
PCAState:
  loading/explained/residual/commonality

GARCHFitBlock:
  alpha/beta/persistence/conditional_var/forecast/shock

RollingLinearState:
  coeff/residual/r2/diagnostics
```

CSE 应识别 family state，避免多个 canonical 各自重新 fit。

## R37-P1-055：Selective Numba

优先：

- Kalman recurrence；
- PSAR/Supertrend/KAMA/Wilder；
- state rules；
- directional-change episodes；
- AR/custom recurrence；
- ElasticNet coordinate descent；
- PLS NIPALS；
- GARCH variance recursion/likelihood；
- custom rolling callbacks。

默认：

```python
@numba.njit(cache=True, nogil=True)
```

`fastmath=False`，除非有专门数值证据。

不强行 Numba SVD/eigh/lstsq、DuckDB/Polars IO、SciPy高层 optimizer、已经高效 vectorized native op。

---

# 14. Backend / Optimizer / Batch Differential

## R37-P0-056：backend certification key 完整化

至少：

```text
canonical
semantic_version
parameter domain
source contract
backend
execution variant
dtype
grain
```

## R37-P0-057：Backend Differential

比较：

```text
reference/Pandas
vs Polars
vs DuckDB
vs specialized kernel
```

不仅比较 values，还要：

- NaN mask；
- Inf mask；
- dtype；
- index；
- columns；
- order；
- grain；
- timezone。

## R37-P0-058：Optimizer Differential

```text
no-opt reference
vs CSE
vs fusion
vs predicate pushdown
vs native region
```

优化不能改变语义。

## R37-P0-059：Batch Differential

```text
run_one × N
== run_many
== run_many_parallel
== streaming materialize
```

## R37-P0-060：Batch Permutation Invariance

将 factors 顺序随机打乱多次，结果与 FactorIdentity 必须一致；允许 timing不同，不允许数值/lineage变。

## R37-P0-061：Cache/CSE Isolation

以下变化必须 cache miss：

- snapshot；
- universe；
- calendar；
- price basis；
- source revision；
- parameter；
- semantic version；
- backend semantics version。

---

# 15. Missing / Numerical / Units

## R37-P0-062：统一 MissingValueContract

每个 operator明确：

```text
NaN input
Inf input
current NaN
partial window
all NaN
min_periods
zero denominator
constant window
```

不同 backend 必须消费同一 contract。

## R37-P0-063：统一 NumericalPolicy

覆盖：

- overflow/underflow；
- divide-by-zero；
- log/sqrt domain；
- covariance singularity；
- regression condition number；
- eigen near-tie；
- quantile ties；
- integer overflow；
- float32 downcast。

任何 silent clip/fill 都必须是显式 contract。

## R37-P1-064：轻量 Unit/Dimension metadata

至少支持：

```text
price
return
log_return
volume
currency
ratio
percentage
bp
volatility
variance
loading
score
```

不需要构造完整物理单位系统，但应阻止明显错误组合。

“看起来像 price level”的数值 heuristic 只能做 warning，不能成为 production 单一 truth；正式 unit 来自 source/field/operator contract。


---

# 16. Data Quality / Schema Evolution

## R37-P0-065：Source DQ / Factor DQ / Drift DQ 分层

### Source DQ

- duplicate key；
- missing date/session；
- instrument coverage异常；
- stale file；
- all zero；
- schema mismatch；
- timezone错位；
- impossible OHLC；
- negative volume；
- revision conflict；
- source snapshot mismatch。

### Factor DQ

- all NaN；
- low coverage；
- Inf；
- constant；
- numeric explosion；
- index mismatch；
- universe mismatch；
- grain mismatch。

### Drift DQ

- distribution shift；
- coverage shift；
- factor variance collapse；
- source field behavior change。

DQ 失败不得自动 `fillna(0)` 后继续 production。

## R37-P0-066：SchemaEpoch 进入 Factor identity

DA已有 schema epoch；FE 必须纳入 source identity、cache、CSE、checkpoint、lineage、materialization manifest。schema变更后旧结果不能静默复用。

---

# 17. FactorIdentity / Lineage / Semantic Version

## R37-P0-067：统一 FactorIdentity

建议：

```python
FactorIdentity(
    normalized_ir_hash,
    operator_semantic_versions,
    parameter_values,
    source_knowledge_identity,
    universe_identity,
    calendar_identity,
    price_basis,
    output_grain,
)
```

display name 不是 identity。

## R37-P0-068：Operator semantic version 变化触发迁移

公式语义变化时：

```text
same canonical
new semantic_version
```

旧物化不能被静默当作新语义；cache/checkpoint失效；lineage保留；根据 impact 做重算。

## R37-P1-069：安全 algebraic canonicalization / dedup

冷启动与 LLM mining 会产生：

```text
a+b vs b+a
x*1
x+0
重复 alias
参数等价表达式
```

对严格安全的等价式 canonicalize，减少编译/搜索/缓存重复；对浮点/NaN/overflow 语义不严格等价的表达式不要激进改写。

---

# 18. Service / QoS / Cancellation / Failure Taxonomy

## R37-P0-070：Job QoS 进入统一资源协调

至少三档：

```text
CRITICAL  : daily production/live
STANDARD  : normal research
BACKGROUND: mining/benchmark
```

压力时先停止 BACKGROUND 新 admission，再收缩 STANDARD，CRITICAL保留最低资源。大规模挖掘不能挤死每日生产落值。

## R37-P0-071：CancellationToken 贯穿全链路

取消传播到：

- DataAccess scan；
- remote request；
- DuckDB query；
- native compute stage；
- model loop；
- writer；
- spill；
- generation transaction。

取消后必须释放 CPU/IO/memory/cache pin/spill/writer/source session leases。

## R37-P0-072：统一 FailureTaxonomy

至少：

```text
SemanticError
PITViolation
DataQualityError
SchemaError
ParameterDomainError
ResourceAdmissionError
ResourceUnderpredictionError
OOMReplanRequired
TransientIOError
PermanentIOError
WriterFatalError
Cancellation
DeadlineExceeded
BackendBug
OptimizerMismatch
CheckpointInvalid
```

每类明确：

```text
retry?
replan?
shard?
fallback?
backend quarantine?
abort generation?
operator quarantine?
```

未知错误不要默认无限 retry。

---

# 19. FE × DA Capability Handshake

## R37-P0-073：不要只依赖 `data-access>=0.2.0`

当前 DataAccess 能力已经远超早期最低版本。随着 FE×DA 深度融合，单纯 semver lower bound 不能证明运行时具备 required capability。

新增：

```python
DataAccessCapabilities(
    protocol_version,
    prepared_read,
    batch_prepared_session,
    pinned_snapshot,
    scan_cost,
    temporal_join,
    universe_pit,
    relation_ref,
    arrow_stream,
    resource_lease_bridge,
    cancellation,
    schema_epoch,
)
```

FE startup/compile 前做 capability handshake；缺 required capability fail closed，不能执行到中间才 AttributeError。

## R37-P1-074：依赖兼容矩阵

CI 至少：

```text
minimum-supported
current-lock
latest-supported
```

覆盖：NumPy、Pandas、PyArrow、DuckDB、Polars、SciPy、Numba、DataAccess。

核心依赖升级必须重跑 semantic parity、backend parity、PIT、serialization、TTDC regression。

---

# 20. Factor Lake / 下游读取

## R37-P1-075：避免因子×天小文件爆炸

设计 FactorBlock partition，在 selective read、atomic generation、append、correction、compaction 之间平衡。

建议考虑：

```text
market
generation
date block
factor family/block
```

而不是每个 factor 每天一个独立小文件。

## R37-P1-076：Write Amplification 成为 KPI

记录：

```text
logical output bytes
physical written bytes
rewrite bytes
small file count
compaction bytes
```

增量一天不应重写巨大历史分区。

## R37-P1-077：Feature Matrix API

模型训练支持：

```text
FactorBlock / Arrow
→ select factors/dates/universe
→ Arrow/NumPy matrix
```

减少一列一列 parquet read + pandas concat。

---

# 21. Observability / Explainability

## R37-P1-078：统一 Stage Telemetry

每个真实 stage 至少记录：

```text
stage_id
factor/task ids
backend
execution_variant
rows in/out
bytes in/out
scan bytes
cache hits
CSE reuse
conversion bytes
elapsed
PSS delta/peak
CPU tokens
backend threads
spill
writer wait
source snapshot
```

## R37-P1-079：Explain APIs

至少：

```text
explain_factor
explain_batch
explain_backend_route
explain_resource_plan
explain_source_plan
explain_lineage
explain_certification
explain_incremental_impact
```

必须来自真实 planner/runtime truth，而不是注释或静态文档。

---

# 22. Current-HEAD CI / Evidence / Benchmark

## R37-P0-080：最终 HEAD 必须有可验证 CI

本轮开始时当前 HEAD 没有可直接证明最终状态的 commit workflow/status。整改完成后必须满足：

```text
final_code_sha
== ci_sha
== evidence_sha
== benchmark_sha
```

任何旧 SHA 的 “1522 passed / 973 passed / 950 passed” 都不能证明最终 HEAD。

## R37-P0-081：Test Obligation Matrix

每个 production operator：

```text
EXECUTION
SEMANTIC_GOLDEN
PARAM_DOMAIN
EDGE
TEMPORAL
FUTURE_PERTURBATION
SOURCE_PIT
BACKEND_PARITY
BATCH_PARITY
OPTIMIZER_DIFF
CHUNK
INCREMENTAL
DETERMINISM
SHAPE
NULL_INF
NUMERIC_STRESS
```

stateful 额外：

```text
CHECKPOINT
SEGMENT_PARITY
```

model 额外：

```text
FIT_CUTOFF
LABEL_MATURITY
SCALER_CUTOFF
HYPERPARAM_CUTOFF
CONVERGENCE
MODEL_ORACLE
```

## R37-P0-082：Evidence Ledger 必须回答具体调用为什么允许

最终系统应能回答：

> `canonical=ts_xxx, window=120, backend=polars, A-share daily, snapshot=X, production` 为什么被允许？

并追溯到具体 evidence cases；不能只有“operator passed”。

---

# 23. Resource / Co-tenancy 硬测试矩阵

硬件 profile：

```text
8GB
16GB
64GB
256GB
```

至少测试：

1. 外部 RAM eater启动；
2. 外部 RAM eater退出；
3. external CPU load启动/退出；
4. IO saturation启动/退出；
5. writer slowdown；
6. disk full；
7. remote latency；
8. giant CSE；
9. process child memory；
10. cache pressure；
11. DuckDB large sort/hash；
12. Polars streaming；
13. large rolling；
14. model-heavy batch；
15. mixed-source 1000 factors；
16. small 5-factor batch。

硬验收：

```text
NO OOM under supported envelope
NO permanent low-speed after pressure disappears
NO CPU oversubscription explosion
NO resource lease leak
NO mixed generation
```

特别注意：**resource recovery after pressure** 是硬 gate，不是 optional benchmark。

---

# 24. 性能基准

所有性能统计必须在 correctness hard gates 通过后执行。

## 24.1 主指标

```text
Time To Durable Commit (TTDC)
```

## 24.2 辅助指标

- compile time；
- source resolve；
- scan；
- temporal join；
- compute；
- conversion；
- DQ；
- queue wait；
- write；
- commit；
- peak family PSS；
- spill；
- physical scan bytes；
- write amplification；
- cache/CSE reuse；
- native region coverage；
- Pandas fallback count；
- factors/sec。

## 24.3 场景

```text
1 factor
5 factors
50 factors
100 factors
1000 factors
mixed-source 1000
fundamental-heavy
minute→daily-heavy
model-heavy
incremental one-day
historical correction
```

## 24.4 性能硬规则

禁止为了大 batch benchmark：

- 让小批量明显退化；
- 放宽 PIT；
- 关闭 DQ；
- 无证据降 precision；
- 静默改 NaN；
- 改模型 refit semantics；
- 缩 window；
- 改 universe；
- 使用旧 snapshot。

---

# 25. 建议新增/重构模块

文件名可按现有项目结构调整，核心是职责，不要为了对齐本文另建重复 truth。

```text
factor_engine/evidence/
    operator_evidence_store.py
    parameter_domain_store.py
    test_obligation_matrix.py
    evidence_truth.py

factor_engine/semantic/
    data_knowledge_identity.py
    factor_identity.py
    output_shape_contract.py

factor_engine/runtime/
    host_resource_coordinator.py
    resource_autopilot.py
    resource_monitor.py
    resource_calibration_store.py
    run_peak_sampler.py
    auto_shard_planner.py
    governed_buffer_store.py
    physical_stage_executor.py
    change_impact_dag.py

factor_engine/planner/
    unified_query_graph.py

dataaccess/runtime/
    resource_bridge.py

dataaccess/
    capabilities.py
```

如果现有模块已覆盖同一职责，直接扩展现有模块。

---

# 26. 执行顺序

## Phase 0：Baseline / Inventory

输出：

```text
R37_BASELINE.json
R37_CURRENT_OPERATOR_INVENTORY.parquet
R37_CURRENT_RESOURCE_ARCH.json
R37_CURRENT_FE_DA_BOUNDARY.json
```

## Phase 1：Evidence Truth + Parameter Domain

先修 evidence hard gate、parameter-domain membership、OutputShape、Per-Canonical ledger。

## Phase 2：PIT / Source Identity

完成 universe、calendar、price basis、financial bitemporal、revision、DataKnowledgeIdentity。

## Phase 3：FE×DA Unified QueryGraph

完成 source requirement lowering、PreparedBatchReadSession、真实 executable stages、BufferRef。

## Phase 4：HostResourceCoordinator

统一 FE/DA/service/writer leases；完成 fast-down slow-up、PSI、resource model、run peak calibration。

## Phase 5：GovernedBufferStore + AutoShard

关闭 raw dict bypass，完成 CSE memory rent、legal sharding、OOM replan、spill/recompute。

## Phase 6：Streaming Materialization

完成 global sink budget、writer lifecycle、fatal propagation、atomic generation、read-after-write。

## Phase 7：Incremental / Revision

完成 ChangeImpactDAG、checkpoint invalidation、minimal recompute、correction propagation。

## Phase 8：Model / Fast Kernel

在 correctness 收口后再做 full model contracts、FastLinearWindowEngine、family state、selective Numba。

## Phase 9：Full Hard-Gate Matrix

跑 semantic/PIT/parameter/backend/optimizer/batch/chunk/incremental/resource/co-tenancy/failure-injection/mutation。

## Phase 10：Current-HEAD Release

最后才生成 final evidence、current SHA CI、benchmark、release manifest、canary/shadow。


---

# 27. R37 必须新增的关键测试

建议组织：

```text
factor_engine/tests/r37/
    test_evidence_truth_negative_controls.py
    test_parameter_domain_all_production.py
    test_output_shape_contract.py
    test_universe_pit.py
    test_calendar_session_semantics.py
    test_corporate_action_price_basis.py
    test_financial_bitemporal_revision.py
    test_unified_query_graph.py
    test_source_scan_reuse.py
    test_stage_execution_no_root_recompute.py
    test_host_resource_coordinator.py
    test_resource_shrink_and_recover.py
    test_external_memory_pressure.py
    test_external_cpu_pressure.py
    test_auto_shard_semantics.py
    test_oom_replan.py
    test_governed_buffer_no_raw_bypass.py
    test_cache_accounting_reconcile.py
    test_writer_thread_timeout.py
    test_writer_fatal_stops_scheduler.py
    test_atomic_generation_failure_injection.py
    test_change_impact_dag.py
    test_state_checkpoint_revision_invalidation.py
    test_all_stateful_segment_parity.py
    test_backend_differential.py
    test_optimizer_differential.py
    test_batch_permutation.py
    test_current_sha_binding.py
    test_dataaccess_capability_handshake.py
```

DataAccess 侧补对应 integration tests，不要只在 FE mock 掉 DA 关键行为。

---

# 28. Production Hard Gates

最终至少应有：

```text
R37_CURRENT_HEAD_BOUND
R37_ZERO_FAKE_EVIDENCE_GATES
R37_ALL_PRODUCTION_CANONICALS_IN_LEDGER
R37_ALL_PRODUCTION_CALL_DOMAINS_CERTIFIED
R37_ALL_PRODUCTION_OPERATOR_GOLDENS_PASS
R37_ALL_PIT_PERTURBATIONS_PASS
R37_ALL_SOURCE_PIT_PASS
R37_UNIVERSE_PIT_PASS
R37_CALENDAR_SESSION_PASS
R37_PRICE_BASIS_IDENTITY_PASS
R37_FINANCIAL_BITEMPORAL_PASS
R37_BACKEND_DIFFERENTIAL_PASS
R37_OPTIMIZER_DIFFERENTIAL_PASS
R37_BATCH_PARITY_PASS
R37_BATCH_PERMUTATION_PASS
R37_STATEFUL_SEGMENT_PARITY_PASS
R37_INCREMENTAL_PARITY_PASS
R37_NO_RAW_BUFFER_GOVERNANCE_BYPASS
R37_ONE_HOST_RESOURCE_AUTHORITY
R37_RESOURCE_PRESSURE_SHRINK_PASS
R37_RESOURCE_RECOVERY_UPSHIFT_PASS
R37_AUTOSHARD_SEMANTICS_PASS
R37_OOM_REPLAN_PASS
R37_WRITER_FATAL_PROPAGATION_PASS
R37_ATOMIC_GENERATION_PASS
R37_CHANGE_IMPACT_PASS
R37_CURRENT_HEAD_CI_PASS
R37_CURRENT_HEAD_BENCHMARK_PASS
```

每个 gate 都必须有真实 executed cases，不能用文件存在代替行为测试。

---

# 29. 最终 Evidence Artifacts

建议：

```text
docs/evidence/r37/
    R37_BASELINE.json
    R37_HEAD.json
    R37_ISSUE_CLOSURE_LEDGER.csv
    R37_OPERATOR_CORRECTNESS_LEDGER.parquet
    R37_PARAMETER_DOMAIN_LEDGER.parquet
    R37_MODEL_CONTRACT_LEDGER.parquet
    R37_STATEFUL_PARITY_LEDGER.parquet
    R37_BACKEND_PARITY_LEDGER.parquet
    R37_OPTIMIZER_DIFF_LEDGER.parquet
    R37_SOURCE_PIT_LEDGER.parquet
    R37_RESOURCE_CALIBRATION.json
    R37_COTENANCY_STRESS.json
    R37_FAILURE_INJECTION.json
    R37_MATERIALIZATION_ATOMICITY.json
    R37_INCREMENTAL_CORRECTION.json
    R37_BENCHMARKS.json
    R37_HARD_GATES.json
    R37_FINAL_ACCEPTANCE_REPORT.md
```

所有 artifact header：

```text
commit_sha
working_tree_hash/component_hash
created_at
producer_script
fixture_hashes
protocol/schema version
```

---

# 30. Issue Closure Ledger

每一个 `R37-Px-xxx` 至少记录：

```text
issue_id
baseline_status
root_cause
files_changed
implementation_summary
tests_added
evidence
final_status
final_sha
```

不得写：

```text
fixed because code exists
fixed because test file exists
fixed because prompt要求了
```

必须引用真实执行 case。

---

# 31. 禁止的“修复方式”

## 31.1 修改审计脚本让它通过

禁止。

## 31.2 把本可 production 的算子降 research 逃避测试

除非角色审查证明它本来就不应进入 production/default mining。

## 31.3 关键路径 `except Exception: pass`

对以下路径禁止：

- resource accounting；
- cache registration；
- checkpoint validation；
- writer；
- PIT；
- snapshot；
- generation commit；
- evidence generation。

## 31.4 Production silent Pandas fallback

禁止。

## 31.5 OOM 后原尺寸 retry

禁止。

## 31.6 为省资源改变数学定义

禁止擅自：

- 改模型 refit cadence；
- 缩 window；
- 少股票；
- float64→float32；
- fill NaN；
- 改 universe；
- 改 PIT cutoff。

## 31.7 为性能跳过 DQ/PIT

禁止。

---

# 32. 最终目标架构

```text
                  ┌──────────────────────────┐
                  │      Factor DSL / IR     │
                  └────────────┬─────────────┘
                               │
                  Semantic + Certification Gate
                               │
                   Unified SourceRequirementIR
                               │
                 ┌─────────────▼──────────────┐
                 │ DataAccess Prepared Batch │
                 │ snapshot/PIT/schema/auth  │
                 └─────────────┬──────────────┘
                               │
                     Unified QueryGraph
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
      Source Scan         Temporal Join       Normalize/Aggregate
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    Native Compute Regions
              DuckDB / Polars / NumPy / Numba
                               │
                    Governed BufferRefs
                               │
               Stateful / CrossSection / Model
                               │
                           Block DQ
                               │
                     Streaming Writer
                               │
                  Generation Transaction
                               │
                       Atomic Publish
```

横向只有一个资源权威：

```text
HostResourceCoordinator
```

贯穿 service / DA scan / DuckDB / Polars / BLAS / Numba / cache / buffer / spill / writer。

纵向只有一个数据/因子身份体系：

```text
DataKnowledgeIdentity / FactorIdentity
```

贯穿 source / cache / CSE / checkpoint / factor / materialization / lineage / evidence。

---

# 33. 最终验收场景

## Scenario A：1000 普通日频量价因子

要求：

- common scan显著复用；
- 无1000次重复源读；
- 一个 unsupported node 不拖整批回 Pandas；
- peak memory受控；
- TTDC有基准；
- run_one/run_many parity。

## Scenario B：1000 mixed-source factors

量价 + 财务 + 行业 + 指数 + 事件 + minute→daily。

要求：

- 一个 pinned batch snapshot；
- PIT joins正确；
- source identity完整；
- read wave复用；
- no mixed snapshot；
- DataAccess physical work不重复规划/重复扫描。

## Scenario C：64GB服务器 + 外部算法突然占40GB

要求：

- FE发现 headroom/PSI变化；
- 停止/降低新 admission；
- 缩 concurrency/wave/block；
- 不 OOM；
- 外部负载释放后逐渐恢复。

## Scenario D：16GB服务器 + 大 stateful/model task

要求：

- 预测 task无法 fit；
- legal shard；
- checkpoint/warmup正确；
- full parity；
- 不原尺寸 OOM retry。

## Scenario E：财报历史 restatement

要求：

- DA产生 SourceChange；
- ChangeImpactDAG计算受影响 factors/dates；
- 旧 checkpoint失效；
- minimal recompute；
- 新 generation atomic publish。

## Scenario F：writer disk full

要求：

- writer fatal；
- scheduler停止新 admission；
- generation abort；
- published pointer不变；
- 后续可安全重跑。

## Scenario G：故意把 PIT kernel 改成 future shift

要求：

- future perturbation/mutation gate红；
- production release阻断。

## Scenario H：backend optimizer bug

故意把 fusion 后 rank 的 universe/order弄错，optimizer differential 必须红。

---

# 34. Coding Agent 最终交付格式

完成后必须给出：

```text
1. Final commit SHA
2. Baseline SHA
3. Changed files
4. R37 issue closure table
5. Production operator count
6. Fully certified production operator count
7. Research/experimental exceptions and reasons
8. Full pytest/current CI result
9. Negative-control result
10. Parameter-domain coverage
11. Backend differential summary
12. Optimizer differential summary
13. Stateful parity summary
14. Resource/co-tenancy summary
15. Atomic materialization failure-injection summary
16. Incremental/revision summary
17. Benchmark summary
18. Remaining known issues
```

如果还有 remaining issue，不得写“全部完成”。

---

# 35. 最后的判断标准

下一阶段不再以“代码里有没有这个类”判断完成，而要能机器回答：

### Correctness

> 这个具体 factor call 为什么是正确的？

### PIT

> 这个值在 decision time 当时真的能知道吗？

### Data Identity

> 它来自哪个 snapshot、universe、calendar、price basis、revision？

### Backend

> 为什么这个 backend/variant 对这个具体参数域被允许？

### Batch Execution

> 1000 个因子真正共享了哪些 scan、join、state 和 compute？

### Resource

> 为什么在这台机器、此刻还有其他算法运行时，这个 task 可以安全启动？

### Recovery

> 外部负载释放以后，系统是否会自己恢复速度？

### Incremental

> 一条历史数据修订影响哪些 factor/date/checkpoint？

### Materialization

> 任意阶段失败后，生产侧是否只有“旧完整 generation”或“新完整 generation”两种状态？

### Evidence

> production 结论是否由最终 HEAD 的真实行为测试得出，而不是审计脚本自己宣布？

只有这些问题都能通过 contract、ledger、test 和 runtime telemetry回答，R37 才算结束。

---

# 36. 优先级摘要

```text
P0:
Evidence Truth
Parameter-domain admission
PIT/DataKnowledgeIdentity
Unified physical execution
One HostResourceCoordinator
Bidirectional resource autotune
AutoShard/OOM replan
GovernedBufferStore
Atomic streaming materialization
ChangeImpactDAG
Stateful parity
Current-HEAD CI/evidence

P1:
FastLinearWindowEngine
Model-family CSE
Selective Numba
Factor Lake layout
Feature Matrix API
Advanced observability
Dependency compatibility matrix
Storage/write-amplification optimization

P2:
NUMA-aware placement
Distributed execution
GPU
跨机 cache
更复杂 online autotuner
```

P2 在单机语义、资源、增量、物化闭环前不要优先。

---

# 37. R37 完成后的预期状态

完成后 FactorEngine + DataAccess 应从：

```text
“功能很多、框架很强、局部验证很多”
```

进入：

```text
“每一个 production 调用都有可追溯语义和证据，
批量执行有真正统一的物理计划，
资源会按机器和共存负载自动收缩与恢复，
历史修订可以最小传播，
物化失败不会污染生产，
最终 release 与 current HEAD 严格绑定。”
```

这才是本轮下一阶段整改的生产级完成标准。
