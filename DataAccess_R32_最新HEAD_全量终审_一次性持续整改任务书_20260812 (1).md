
# DataAccess R32 — 最新 HEAD 全量终审、Canonical Runtime 收敛、PIT/快照/缓存/资源/服务/FE 联动与性能一次性持续整改任务书

> 仓库：`https://github.com/18047533889/quant_projects`  
> 审计时观察到的最新 main：`4b1577fce14c097fba48619885b38d6c596242eb`  
> 时间：2026-08-12  
> 前序：DataAccess R24–R31，以及 FactorEngine R39/R40 等整改  
> 本文件用途：**直接整体交给 Coding AI / Agent。不要拆成几十轮问答。**

---

# 0. 最高级执行规则

你不是来“逐条打勾”的，你的任务是把 DataAccess 收敛到一个能长期维护、能服务 FactorEngine、大批量因子挖掘和后续模拟盘/实盘的数据基础层。

## 0.1 开始前必须重新解析最新 main

本文件记录的审计基线是：

```text
4b1577fce14c097fba48619885b38d6c596242eb
```

但执行时仓库可能已经继续更新。

第一步必须：

```bash
git checkout main
git pull --ff-only
git rev-parse HEAD
git status --short
```

生成：

```text
R32_EXECUTION_BASELINE.md
```

记录：

```text
current_head
current_branch
dirty_state
python_version
duckdb_version
polars_version
pyarrow_version
platform
```

每一项问题必须分类：

```text
CONFIRMED_OPEN
ALREADY_FIXED
PARTIALLY_FIXED
REGRESSION
SUPERSEDED_BY_CANONICAL
NOT_APPLICABLE
DEFERRED_WITH_REASON
```

禁止看到本文件就机械再造同名模块。

---

# 1. 本轮最终原则

DataAccess 最终只允许形成下面这一条主链：

```text
FactorEngine
    ↓ typed data demands / source bindings
BatchDataRequest / ReadWavePlanner
    ↓
DataReadSession
    ↓
Canonical ReadPipeline
    ├─ Execution/Security Context
    ├─ Contract/Semantic Compiler
    ├─ PIT/Calendar/Universe
    ├─ Metadata Plane
    ├─ SourceSnapshot Resolver
    ├─ Query/Resource Admission
    ├─ PreparedRead
    ├─ SourceBlock CSE/Cache
    └─ Trace/Lineage/Audit
    ↓
Backend
    ├─ DuckDB
    ├─ Polars
    └─ PyArrow
```

不允许长期存在：

```text
canonical runtime
+
r30/r40 additive runtime
+
legacy runtime
+
FactorEngine 自己再复制一份 data semantics
```

同一个事实只能有一个 authority：

```text
一个 Calendar identity
一个 SourceSnapshot identity
一个 SemanticField authority
一个 Security context
一个 Deadline
一个 Resource envelope
一个 Batch read planner
一个 ChangeImpact contract
一个 QueryTrace
一个 MetadataPlane
```

---

# 2. 绝对不能为了性能牺牲的语义

任何“优化”不得关闭：

1. PIT；
2. no-lookahead；
3. revision/vintage；
4. calendar/session；
5. source snapshot；
6. principal/security scope；
7. contract/schema；
8. universe PIT；
9. unit/currency；
10. read/write dataset boundary；
11. fail-if-changed / pin；
12. lineage/evidence；
13. query budget；
14. execution deadline；
15. FactorEngine cross-sectional / time-series axis semantics。

---

# 3. 已观察到的当前代码事实

当前 HEAD 中至少观察到：

- `dataaccess/pyproject.toml` 仍是静态 `version = "0.10.2"`；
- 显式 package list 仍未包含 `data_access.r30`；
- 新增 `dataaccess/r30/resolution_lease.py`；
- `DataReadSession` 新增 `PhysicalResolutionContext/_ResolutionCache`；
- 新增 `SnapshotFidelity`；
- 新增 typed `ManifestFetchResult`；
- `SourceSnapshotResolver` 已增强 exact objects / fidelity / publisher manifest；
- `StartupCertificate` 已加入 startup gate；
- `GlobalResourceGovernor` 已加入 remote discovery slot；
- `ScanCost` 已加入 cost basis；
- 当前 GitHub connector 没有返回 current HEAD 的 combined status / workflow run 证据。

这些是本轮审计的事实基础，不代表其他模块没有变化。

---

# 4. P0 — 必须关闭的 correctness / security / reproducibility / production blockers



## R32-P0-001 — ResolutionLease 部分成功回滚计数错误

检查 `dataaccess/r30/resolution_lease.py::acquire_resolution()`。当前本轮申请多个 governor discovery slots 时，如果中途失败，回滚必须只释放“本次已经成功取得的 slots”，不能使用已有 `_resolution_inflight` 作为回滚数量。

修改：
- 使用 `acquired_now`；
- 本地 slot 预留与 governor slot 获取拆开；
- rollback exactly-once；
- 不得释放进入本次调用前已经持有的 slot。

测试：
- lease 已持有 1；
- 本轮请求 3；
- governor 第 2 个失败；
- 结束后原 1 个仍存在；
- 本轮成功取得的 1 个已经归还。


## R32-P0-002 — ResolutionLease 禁止持内部锁调用外部 governor

`acquire_resolution/release/release_all_resolution` 不要在持有 `_lock` 时调用 governor/HostCoordinator。

修改成：
1. 锁内检查/预留状态；
2. 锁外调用外部资源系统；
3. 锁内原子 commit；
4. 冲突则锁外 rollback。

目标：消除 lock-order inversion / callback deadlock。


## R32-P0-003 — ResolutionLease absolute_deadline 必须真正执行

当前 `absolute_deadline` 不能只是字段。

所有 discovery：
- manifest；
- LIST；
- HEAD；
- metadata lookup；
- credential refresh

都必须使用同一 `DeadlineContext.remaining()`。

进入 execution 时：
- execution lease 必须继承同一 absolute deadline；
- 已过期则禁止 transition；
- waiting slot 必须 timeout；
- cancellation 必须传播。


## R32-P0-004 — 外部 ExecutionLease 注入必须验证

`transition_to_execution(execution_lease=...)` 不得接受任意 mutable lease 对象。

至少验证：
- acquired=True；
- released=False；
- parent/job identity；
- resource envelope identity；
- absolute deadline；
- principal/security execution id。

更优：
- 接受不可伪造 `ExecutionLeaseToken`；
- 或统一由 canonical ResourceBroker 创建。


## R32-P0-005 — ExecutionLease child release 必须归还父预算

当前/旧 R30 `request_child()` 扣父 `_remaining` 后，child 自己 release 并不会自然把预算归还 parent。

重构：
- child 保存 parent ref + allocation token；
- `child.release()` exactly-once 返还；
- parent release 递归释放且不能 double return；
- 并发 request_child/release 必须 lock。

测试：
`parent=100, child=60, child.release => parent.remaining=100`。


## R32-P0-006 — ExecutionLease 全生命周期线程安全

`_acquired/_released/_children/_remaining` 当前为 mutable state。

增加：
- internal lock；
- NEW/ACTIVE/RELEASED state；
- request/release exactly-once；
- nested children concurrency；
- release 与 request_child race test。


## R32-P0-007 — Host-backed ResourceGovernor 不得绕过本地基础不变量

检查 `GlobalResourceGovernor.admit()`。

当 FE HostResourceCoordinator 成功返回 host lease 时，不能在：
- duplicate query_id；
- max_active_queries；
- per-principal fairness；
- reservation lifecycle

这些检查之前直接 return。

正确顺序：
1. 本地 identity/fairness invariant；
2. host child lease；
3. local bookkeeping；
4. 异常 rollback host lease。


## R32-P0-008 — 配置了 HostCoordinator 但调用失败时禁止静默 standalone fallback

必须区分：

```text
HOST_INTEGRATION_ABSENT
HOST_INTEGRATION_CONFIGURED
HOST_INTEGRATION_BROKEN
```

- ABSENT：可以使用 standalone policy；
- CONFIGURED+BROKEN：production fail；
- 不允许 catch Exception → None → local governor。


## R32-P0-009 — Governor release 不得持锁调用 host lease release

锁内：
- pop reservation；
- 更新本地计数。

锁外：
- release host lease。

外部 callback 失败：
- structured cleanup error；
- telemetry；
- 必要时 worker unhealthy。


## R32-P0-010 — DuckDB slot 等待必须受 deadline/cancellation

禁止无限 `Semaphore.acquire()`。

统一：
```python
acquire(timeout=deadline.remaining())
```

支持：
- cancellation token；
- queue wait metrics；
- timeout typed error；
- request deadline 到期后不能继续执行 SQL。


## R32-P0-011 — Remote/DuckDB/Discovery slot 统一成 typed ResourceSlotLease

当前 remote/discovery 多为 nonblocking bool，DuckDB 为 blocking。

统一：
```text
ResourceSlotLease
resource_kind
deadline
wait_policy
cancellation
acquired_at
released
```

避免每种资源拥有不同过载语义。


## R32-P0-012 — Standalone memory cap 应基于真实 available headroom

不要只使用 `MemTotal * 0.5`。

至少考虑：
- cgroup memory.max；
- cgroup memory.current；
- process RSS；
- RLIMIT；
- configured reserve；
- FE host envelope（若有）。

计算：
`safe_headroom = min(cgroup/rlimit/host) - current_usage - reserve`。


## R32-P0-013 — StartupCertificate 改为 immutable + subject-bound

当前证书不能只是 mutable `passed/timestamp/problems/evidence_hash`。

定义：
```python
StartupCertificate(
  status,
  subject_digest,
  gate_version,
  issued_at,
  expires_at,
  evidence_digest,
)
```

`@dataclass(frozen=True)`。

任何调用者不能直接改 `passed`。


## R32-P0-014 — StartupCertificate 不能只 hash problems

成功时 problems 为空，不能证明当前环境。

`StartupSubjectDigest` 至少绑定：
- build SHA；
- package version；
- registry digest；
- RuntimeContract digest；
- semantic schema version；
- policy digest；
- required calendar snapshot ids；
- credential provider generation/capability；
- source profile；
- deployment worker topology；
- startup gate version。


## R32-P0-015 — StartupCertificate 失效不能只靠 24h TTL

复用证书前同时校验：
```text
not expired
AND current_subject_digest == cert.subject_digest
```

关键配置变化立即 invalidate：
- policy；
- credential generation；
- registry；
- calendar；
- source endpoint；
- code build；
- worker topology。


## R32-P0-016 — Startup Gate 用 PASS/DEGRADED/FAIL，不再 problems+passed=True

research 可继续 ≠ gate passed。

定义：
```text
PASS
DEGRADED
FAIL
```

Production：
- 只接受 PASS。

Automated research：
- 可根据 policy 接受部分 DEGRADED。

Interactive：
- 显式显示 degraded evidence。


## R32-P0-017 — Startup calendar identity 必须复用 canonical CalendarSnapshot

删除/迁移 startup_gate 内另一套只 hash `trading_days` 的 calendar digest。

Canonical CalendarSnapshot 必须包含：
- market；
- exchange；
- timezone；
- sessions；
- lunch break；
- early close；
- holidays；
- source/version；
- digest。


## R32-P0-018 — Startup required markets 从 deployment contract 推导

禁止硬编码 `ashare/us` 全部必选。

根据当前 deployment registry：
- 读取可服务 datasets；
- 推导 required markets；
- 推导 required calendars/universes/providers；
- 只对本实例真正 critical dependency 做 gate。


## R32-P0-019 — Startup credential check 验证 capability/provider factory，而非要求全局 secret

支持两种合法部署：
```text
SERVER_SCOPED_CREDENTIAL
REQUEST_SCOPED_STS_FACTORY
```

startup 验证：
- provider 可用；
- scope policy；
- refresh capability；
- endpoint connectivity（按部署策略）。

不要强制 `_global_credential_provider()` 必须持有 base credential。


## R32-P0-020 — Startup source probe set 从 registry criticality 自动生成

不要只 smoke probe `factor_matrix/factor_lake`。

Registry/RuntimeContract 增加：
```text
criticality
startup_probe
required_for_production
```

自动探测：
- prices；
- financials；
- calendar；
- universe；
- factor lake；
- required minute sources。


## R32-P0-021 — Startup Gate 必须进入 ASGI lifespan

CLI 入口 gate 不是唯一入口。

直接：
```bash
uvicorn data_access.service.app:app
```
也必须 gate。

把 authoritative startup 放到 FastAPI lifespan/startup lifecycle。
CLI 只负责启动参数。


## R32-P0-022 — production 禁止 --skip-startup-gate

`--skip-startup-gate` 只能 dev/research。

strict/production：
```text
flag present → startup fail
```

不能提供一键绕过 security/PIT/snapshot gate 的生产入口。


## R32-P0-023 — /ready 必须是廉价状态检查，不重复昂贵 source discovery

不要每个 readiness probe 都重新：
- manifest；
- LIST；
- HEAD；
- source resolve。

启动/低频 deep health 负责昂贵检查；
`/ready` 只检查：
- current StartupCertificate subject；
- engine local liveness；
- resource state；
- last deep probe freshness。


## R32-P0-024 — /ready 外部错误必须 redacted

禁止 `detail=str(exc)` 直接把：
- dataset；
- local root；
- COS prefix；
- provider；
- internal contract

返回外部。

外部：
```text
readiness_error_code + request_id
```
内部日志保存 sanitized structured details。


## R32-P0-025 — ManifestFetchOutcome 必须从 provider 边界开始 typed

不要在调用者外面套 wrapper，而底层先把异常吞成 None。

定义：
```python
ManifestFetchOutcome(
 status,
 value,
 error_code,
 retryable,
 authority,
)
```

provider 本身返回 typed outcome。

生产代码逐步禁止 raw `None`。


## R32-P0-026 — Manifest absence 语义必须唯一

当 `absence_allowed=False` 时：
```text
value=None
```
绝对不能被解释成 OK。

统一状态：
- OK；
- ABSENT；
- PERMISSION_DENIED；
- DEADLINE_EXCEEDED；
- CORRUPT；
- LOOKUP_FAILED。


## R32-P0-027 — CloudErrorClassifier 适配真实 SDK 错误

支持 botocore/tencent cloud/httpfs 真错误：

- 401/403 Auth；
- 404 NotFound；
- 409/412 version/conditional；
- 429 throttling；
- 5xx；
- Timeout；
- ExpiredToken；
- InvalidToken；
- DNS/TLS。

结果：
```text
typed code
retryable
retry_after
security_sensitive
```


## R32-P0-028 — SnapshotFidelity 从一维 rank 改为多轴证据模型

不要把：
```text
publisher authority
remote/local
content identity
completeness
immutability
```
强行排成一个整数。

建议：
```python
SnapshotEvidence(
 authority,
 object_identity_strength,
 completeness,
 source_mutability,
 enumeration_strength,
)
```

Data policy 根据 dataset 类型判断 minimum evidence。


## R32-P0-029 — strict publisher manifest 必须显式 complete=true

禁止 `get('complete', True)`。

strict：
- key 必须存在；
- 类型必须严格 bool；
- 必须 True。

缺失与 false 都 fail。


## R32-P0-030 — strict publisher manifest 必须显式 objects 字段

空 generation 必须是：
```json
"objects": [],
"object_count": 0,
"complete": true
```

缺失 `objects` 不得自动当空集。


## R32-P0-031 — 空对象集也必须有 canonical digest

`content_digest_of_objects([])` 不应返回魔法空字符串。

使用：
```text
sha256(canonical_empty_object_set)
```

空/非空统一 identity 算法；
publisher declared digest 对空集也必须重算核对。


## R32-P0-032 — Manifest 0-byte object 不得被 falsy 运算吃掉

修复：
```python
entry.get("size") or entry.get("content_length")
```

0 是合法大小。

使用 key presence：
```python
size = entry["size"] if "size" in entry else ...
```


## R32-P0-033 — Manifest schema version 必须真正 compatibility gate

`manifest_version` 不能只检查非空。

增加：
- supported versions；
- min reader；
- max reader；
- migration；
- unknown future version fail-closed。


## R32-P0-034 — published_at 必须严格 timezone-aware timestamp

验证：
- ISO 8601；
- timezone；
- future skew；
- non-decreasing generation publication order；
- publisher clock policy。

非法字符串不能通过。


## R32-P0-035 — Publisher authority 与 Object identity 分离

publisher manifest 中的 local object 如果没有：
- checksum；
- immutable generation identity；
- size+mtime 等足够本地 identity

不能因为“publisher manifest”就自动变成最高证据。

authority != content proof。


## R32-P0-036 — 禁止通过字符 common prefix 推导 dataset boundary

`_common_prefix(paths)` 不得作为授权边界的最终 authority。

Boundary 必须来自：
```text
RuntimeDatasetContract / StorageContract / DatasetPathBoundary
```

glob 只负责选择对象，不负责定义权限。


## R32-P0-037 — resolve_source_snapshot helper 必须带 canonical FileSelector

不能要求每个 caller 记得注入 selector。

PreparedRead/RuntimeContract 已经知道：
- layout；
- file selector；
- dataset boundary。

Resolver 自动消费该 typed contract。


## R32-P0-038 — ResolvedSourceSnapshot digest 必须覆盖本地 mtime/checksum identity

当前 local same-path same-size replacement 不能只 hash path+size。

Object identity 至少：
- checksum（优先）；
- immutable generation；
- 或 size+mtime_ns。

本地 mtime_ns 已有字段必须进入 digest。


## R32-P0-039 — ObjectIdentity 正式建模 checksum/content hash

如果存在 `CONTENT_HASH` capability，`ResolvedObject` 必须有：
```python
ObjectIdentity(
 kind,
 algorithm,
 value,
 size,
)
```

不要 enum 有 CONTENT_HASH，但对象模型没有 checksum。


## R32-P0-040 — ResolvedSourceSnapshot.total_bytes 区分 unknown 与 0

任一 object `content_length=None`：
- total_bytes 不能静默按 0；
- 返回 None/UnknownSize；
- resource admission 使用 conservative bound。

0-byte 与 unknown 必须完全不同。


## R32-P0-041 — legacy DataSnapshot 与 ResolvedSourceSnapshot 收敛

当前 read_contract 仍有 `DataSnapshot/FileVersion`，snapshot 包又有 `ResolvedSourceSnapshot/ResolvedObject`。

最终：
```text
SourceSnapshotIdentity = data facts
ExecutionIdentity = code facts
ReadSnapshot = source + contract + request scope
```

不保留两套互相转换但算法不同的 snapshot truth。


## R32-P0-042 — DataSnapshot 禁止混入 build SHA

source/data snapshot 回答“读了什么数据”。

build SHA 属于：
```text
ExecutionIdentity
```

改代码但数据没变：
- SourceSnapshot 不应变化；
- ExecutionIdentity 应变化；
- Result/Artifact identity 根据 semantic execution policy 决定是否失效。


## R32-P0-043 — build_data_snapshot / rebuild_snapshot_files identity 算法必须完全一致

当前两条构建路径不得一个含 build_sha、一个不含。

统一通过：
```python
CanonicalSnapshotIdentityEncoder
```

所有 snapshot 创建/重建只走一个 builder。


## R32-P0-044 — read_contract remote HEAD 必须传 STS session_token

任何 boto3 client：
```python
aws_session_token=credentials.session_token
```

临时 STS 凭证缺 token 会失败或退回其他 credential chain。


## R32-P0-045 — remote metadata cache 必须 security/credential scoped

不要只用 URI 作为 key。

至少：
```text
uri
endpoint
region
credential_scope_id
credential_generation
principal/security scope when required
```

token refresh 后旧 metadata 不能继续跨 scope 使用。


## R32-P0-046 — remote metadata cache bounded + typed negative cache

当前/旧进程 dict 必须：
- max_entries；
- LRU；
- success TTL；
- NotFound TTL；
- Auth failure不跨credential generation；
- transient failure极短或不缓存。

禁止无界增长。


## R32-P0-047 — CredentialLease 真正使用 expires_at

查询开始前：
```text
expires_at >= now + expected_duration + safety_skew
```
否则刷新/失败。

stream/pinned read：
- credential lease 必须覆盖整个执行窗口；
- 中途 refresh 生成新 generation，不偷换已绑定 query 的 credential identity。


## R32-P0-048 — EnvCredentialProvider 按 credential family 原子解析

COS/AWS/S3 不能逐字段分别找第一个然后混拼。

依次尝试完整 family。
某 family 只配置一半：
- malformed；
- reject；
- 不向下一 family 偷 secret 补齐。


## R32-P0-049 — PhysicalResolutionContext 失败不得折叠成共享空字符串

registry/contract/manifest/source profile 任一步失败：

production：
```text
fail
```

research：
```text
UNKNOWN_UNSHAREABLE:<unique_generation>
```

禁止多个未知上下文共用 `""` cache identity。


## R32-P0-050 — _ResolutionCache.clear() 必须同时清 context memo

当前缓存条目清空后，`_memo[dataset]` 可能仍保存旧 mutation_generation / contract digest。

override：
```python
clear():
    super.clear()
    _memo.clear()
```

并测试 mutation 后显式 clear 能得到新 context。


## R32-P0-051 — DataReadSession 使用显式状态机 + ExitStack

状态：
```text
NEW → ACTIVE → CLOSED
```

禁止 double-enter。

`__enter__` 每设置：
- ExecutionContext；
- RuntimeMode；
- resolution cache；
- calendar freeze；
- leases

都立即注册 cleanup。

中途异常必须完整 rollback。


## R32-P0-052 — DataReadSession critical cleanup 失败不能静默

ContextVar reset / lease release / source block release / calendar cleanup 等失败：

- structured cleanup error；
- telemetry；
- production worker health downgrade；
- 不可 `except Exception: pass` 后继续服务下一请求。


## R32-P0-053 — SourceBlock cache identity 必须由 PreparedRead 自动产生

禁止 caller 手工拼。

包含：
- dataset；
- contract；
- exact snapshot；
- projection；
- filters；
- params；
- instruments；
- time range；
- PIT；
- aggregation；
- price basis；
- provider；
- security；
- representation。


## R32-P0-054 — SourceBlock cache 必须 lease/refcount exactly-once

如果有 refcount：
- acquire 返回 lease；
- close/release decrement；
- session exit release；
- pinned block 不可 evict；
- double release测试。

如果没有生命周期语义，删除假的 refcount。


## R32-P0-055 — SourceBlock/PreparedRead concurrent miss 使用 singleflight

同一 session 内 1000 factors 同时请求同一 source：
- 一个 producer；
- 其余 await；
- producer failure 对全部 waiter 同步失败；
- 失败不缓存；
- cancellation 不泄资源。


## R32-P0-056 — Query result cache 先授权，再 lookup

任何 cache hit 路径必须：
```text
authorize
→ build security-scoped cache identity
→ lookup
```

禁止缓存命中绕过 dataset/factor authorization。


## R32-P0-057 — Query cache key 必须 security/policy/source scoped

必须至少含：
- principal/security digest；
- policy digest/version；
- dataset classification；
- source access tags；
- source snapshot；
- semantic execution version。

premium/restricted data不能跨principal共享。


## R32-P0-058 — Cache hit 也必须产生 lineage/audit/provenance

缓存命中不是“没读数据”。

记录：
- cache level；
- original source snapshot；
- current security scope；
- result identity；
- hit=true；
- bytes served；
- lineage edge。


## R32-P0-059 — DataReadSession/R30 session 只保留一个 canonical session

迁移所有 r30/r40 session 能力到 `read/read_session.py`。

最终：
- `DataReadSession` 是唯一 public job read session；
- r30 alias 只兼容；
- parity后删除重复 runtime。


## R32-P0-060 — CompiledDataRequest 只接受 canonical immutable IR

`deepcopy/copy` 都失败时不能原样返回 mutable custom object。

filters/joins/predicates/aggregations：
- 编译到 frozen typed IR；
- 不支持对象直接 ValidationError；
- plan 后任何输入变异不影响执行。


## R32-P0-061 — ReadPlanIR 与 Executor 分离

不要让号称 immutable 的 plan 同时保存：
- mutable request；
- live store capability。

结构：
```text
ReadPlanIR(frozen, serializable)
ReadPlanExecutor(store)
```

`explain/hash/evidence` 只读 IR。


## R32-P0-062 — ReadPlan 所有 nested values 必须 deep frozen

`scan_costs/join_specs_effective/snapshot tokens/pinned files/derived specs` 等全部 frozen typed value。

外层 MappingProxy 不足以保证 value 不可变。


## R32-P0-063 — ScanCost discovery 全部进入 ResolutionLease

manifest miss 后 footer/stat/list/head 估算也是 discovery。

所有成本估算 I/O：
- 先获得 resolution slot；
- 受 deadline；
- 有 actual metadata-call counters。


## R32-P0-064 — Manifest 未知 rows/bytes 不得当 0

`f.rows or 0 / f.bytes or 0` 会低估。

增加：
```text
known_rows
unknown_row_objects
known_bytes
unknown_byte_objects
cost_confidence
```

unknown → conservative bound。


## R32-P0-065 — UnknownCost 不使用 magic max integer 冒充真实数据

不要用 `(1<<63)-1` 表示未知。

typed：
```python
ExactCost
EstimatedCost
ConservativeBound
UnknownCost
```

planner 根据类型处理，不把 sentinel 参与普通算术。


## R32-P0-066 — Cost calibration 改成有量纲模型

不要 `elapsed_ms / arbitrary_score`。

至少分：
- object listing latency；
- HEAD latency；
- scan MB/s；
- decompression MB/s；
- rows/s；
- serialization MB/s；
- cache hit latency。

按 query class组合预测。


## R32-P0-067 — Cost calibration scope 不能只有 dataset

key 至少：
```text
dataset/source scope
backend
remote/local
server performance profile
cold/warm level
projection bucket
object-count bucket
```

避免一次慢查询污染所有同 dataset 查询。


## R32-P0-068 — DQ checker exception 绝不能 PASS

当前 R30 DQ 多处 `except Exception -> PASS`。

统一状态：
```text
PASS/WARN/BLOCK/UNKNOWN/CHECK_FAILED
```

production certification：
- UNKNOWN != PASS；
- CHECK_FAILED != PASS。


## R32-P0-069 — DQ 列角色必须来自 SemanticField/Concept，不硬编码 physical name

OHLC/volume/report period/publish time/currency：
- 先通过 semantic catalog 找实际 physical column；
- 支持 A股/US/provider alias；
- 缺语义映射时 explicit unknown。


## R32-P0-070 — Coverage 使用 expected trading sessions / expected grain

日频 coverage 不能用自然日。

- A股/US daily：MarketCalendar sessions；
- minute：expected session bars；
- financial：expected periods/filings；
- event：event-specific expectation。


## R32-P0-071 — Coverage by_year 改为 year 内 expected coverage

禁止：
```text
year_rows / all_history_rows
```

应该：
```text
valid_expected_cells_in_year / expected_cells_in_year
```


## R32-P0-072 — Multi-day file rows 不得完整复制给区间每一天

manifest 文件覆盖 10 天、rows=10000：
不能把 10000 加到每天。

需要：
- partition/day-level statistics；
- row-group stats；
- 或明确 approximate，不能冒充 exact。


## R32-P0-073 — Field-level coverage 成为一等公民

文件存在 ≠ 某字段有效。

输出：
- field presence；
- non-null ratio；
- finite ratio；
- first/last valid；
- by_date；
- by_instrument；
- coverage authority。


## R32-P0-074 — ExperimentDataSnapshot required source identity unknown 时 fail closed

禁止 fallback：
```text
stable_digest(dataset_name)
```
冒充可复现 source。

production/automated research：
- required source unresolved → snapshot build fail。

interactive：
- `authoritative=False + unknown_sources`。


## R32-P0-075 — ExperimentDataSnapshot deep immutable

frozen dataclass 中的 `Mapping` 仍可指向 mutable dict。

用：
- FrozenMap；
- tuple entries；
- immutable typed snapshots。

构造后不能修改 dataset/source/version gate。


## R32-P0-076 — SourceSnapshot 与 ExecutionIdentity 分层

Experiment identity 中：
- data facts；
- code facts；
- security facts；
- plan facts

分开建模，再组合 ArtifactIdentity。

不要用 build SHA污染纯 source identity。


## R32-P0-077 — DataChangeSet changed_time_range 不得把 file min/max 自动称为 knowledge time

明确四条时间轴：
```text
data_time
knowledge_time
effective_time
period_time
```

文件 manifest min/max 通常是 data/period axis，不能默认叫 knowledge window。


## R32-P0-078 — Revision availability 禁止用 mtime clamp 伪造历史 vintage

如果只有当前文件：
- 只能判断当前版本何时被系统看到；
- 不能反推出 decision_time 时的历史内容。

正式 `RevisionFidelity`：
NONE / KNOWLEDGE_DATE / INGESTION_VINTAGE / TRUE_VENDOR_VINTAGE。


## R32-P0-079 — ChangeImpact rolling dependency 必须向未来 output 传播

输入 x[t] 改变：
```text
MA20 output t..t+19 受影响
```

不是向前扩“输入起点”。

FE operator trait：
```text
backward_input_horizon
forward_output_horizon
```
沿 DAG 正向传播。


## R32-P0-080 — ChangeImpact 使用 trading bars，不用 timedelta(days=N)

`window=60` 是 60 bars，不是 60 calendar days。

优先在 FE axis index传播；
日期映射使用 frozen MarketCalendar。


## R32-P0-081 — Cross-sectional operator change 扩散整个相关截面

一只股票变：
- rank；
- zscore；
- neutralize；
- regression；
- PCA；
- group rank

可能改变同日其他股票。

引入/复用：
ELEMENTWISE / TIME_FORWARD / CROSS_SECTION_ALL / GROUP_CROSS_SECTION / GLOBAL_STATE。


## R32-P0-082 — changed_instruments 不得把 min/max 当 exact set

Manifest min/max symbol 只是范围。

typed：
- ExactInstrumentSet；
- InstrumentRange；
- PartitionWide；
- UniverseWide；
- UnknownAll。

消费方必须保守传播。


## R32-P0-083 — changed_columns 空集合语义必须 typed

区分：
- ExactColumns；
- AllColumnsUnknown；
- SchemaOnly；
- NoColumnChange。

`()` 不能同时表示“不知道”和“没有变化”。


## R32-P0-084 — to_fe_input 不得只取第一个 changed column

多字段 revision 必须：
- 每个 ColumnIdentity；
- 或 typed column scope；
- 交给 FE dependency matcher。

禁止 `changed_columns[0]` 丢失其余字段。


## R32-P0-085 — Corporate action 支持 backward/bidirectional impact

拆股/复权因子改变可能改写历史 adjusted series。

ImpactDirection：
POINT_ONLY / FORWARD / BACKWARD / BIDIRECTIONAL。


## R32-P0-086 — Universe/calendar change 是一等增量事件

Universe 成分变化：
- cross-sectional factors/neutralization/portfolio exposures 受影响。

Calendar change：
- rolling bars；
- next_session；
- minute→daily；
- availability

全部应失效相关 plan/materialization。


## R32-P0-087 — FactorSourcePlan 必须保存 Concept/Column→Dataset typed binding

不能保存两个独立集合 `concepts` 与 `datasets` 再猜对应关系。

使用/复用：
```text
ColumnSourceBinding
SourceScopeId
```

包含：
market/provider/frequency/grain/timeframe/price_basis/revision。


## R32-P0-088 — Batch read 每 dataset 只投影它真正需要的字段

多源因子：
```text
price.close + financial.revenue
```
必须：
```text
daily -> close
income -> revenue
```

不能把全部 fields 复制到每个 dataset。


## R32-P0-089 — Dependency extraction production fail closed

manifest/import/parser/DSL dependency 解析失败：
- production fail；
- automated research fail；
- interactive 可 degraded。

禁止 unresolved leaf 自动 fallback default dataset。


## R32-P0-090 — FactorBatchPlan 与 ReadWavePlanner 只保留一个 execution authority

最终链：
```text
FactorSourcePlan
→ BatchDataRequest
→ ReadWavePlanner
→ PhysicalFactorDAG
```

旧 batch plan 只能 explain/compat，不再独立决定执行。


## R32-P0-091 — 真实 1000/10000 factors CSE 证据必须来自 backend actual counters

禁止：
```text
source_group_count == physical_scan_count
```
这种 planner 推导冒充事实。

实际记录：
- actual scan invocations；
- object opens；
- bytes read；
- remote requests；
- source block producers；
- consumers。


## R32-P0-092 — read_auto 不能在要求 stream 时重新全量 materialize

如果 cost planner 判定 stream：
- 返回 stream handle；
- 或要求 caller 调 streaming API。

禁止 `list(batches) -> Table` 把资源语义打回全量物化。


## R32-P0-093 — raw scan_polars/LazyFrame production surface 收口

生产/自动挖掘只给 Governed ScanHandle。

raw lazy API：
- internal/dev；
- 或明确 `_unsafe`；
- collect 前仍需 final snapshot/budget/audit。


## R32-P0-094 — stream snapshot 必须只 resolve 一次 exact object set

创建 reader、lineage、snapshot、post-verify 都必须绑定同一个 PreparedStreamRead。

禁止为了构建 snapshot 第二次 resolve，造成 reader 实际对象和报告对象不同。


## R32-P0-095 — stream/iterator 断连时必须释放全部资源

覆盖：
- client disconnect；
- GeneratorExit；
- CancelledError；
- serialization exception；
- first batch exception。

释放：
query slot / DuckDB slot / remote slot / execution lease / source block / temp file。


## R32-P0-096 — HTTP stream query_slots exactly-once release

审计 `read_arrow_stream` 的：
- first=StopIteration；
- first抛异常；
- StreamingResponse构造失败；
- body finally；
- 外层 except

确保没有 double-release 或漏 release。用 resource lease替代手工多个 release 更好。


## R32-P0-097 — HTTP QueryBudget v2 字段全量传播

API budget 必须保留：
- scan objects/files；
- scan bytes；
- remote requests；
- result bytes；
- rows；
- memory；
- elapsed/deadline；
- required columns/time range；
- spill/temp limits。

不要重新构造时丢字段。


## R32-P0-098 — factor read HTTP 路径同样必须走 QueryBudget/ExecutionContext/Trace

`/v1/factors/read` 不能成为特殊旁路。

与 dataset read 共用：
- authorization；
- query budget；
- resource governor；
- execution context；
- trace；
- lineage；
- error mapping。


## R32-P0-099 — write/delete/publish/metadata mutation 必须有逻辑 action authorization

统一动作：
- dataset:read
- dataset:write
- dataset:delete
- dataset:publish
- metadata:read
- metadata:write
- factor:read
- factor:write
- factor:catalog:write

Path sandbox 不能替代逻辑权限。


## R32-P0-100 — dataset-specific DatasetPathBoundary

全局 registered roots allowlist 不足。

任何：
- physical_scope；
- write_dir；
- write_root；
- staging root；
- publish target

都必须证明属于“当前已授权 dataset”的 boundary。


## R32-P0-101 — public API 不接受裸 physical_scope path

使用：
```python
VerifiedPhysicalScope(
 dataset,
 snapshot_id,
 exact_objects,
 contract_digest,
)
```

只能由 PreparedRead/Resolver 创建。
用户不能 `dataset=A + paths=B`。


## R32-P0-102 — append/upsert/delete/publish dataset-level generation atomicity

单文件 rename 不够。

Reader 必须只看到：
```text
old generation
OR
complete new generation
```

不能看到 mixed partitions。

推荐：
immutable generation + atomic pointer/manifest swap。


## R32-P0-103 — publish 两次 rename 的 missing-target window 消除

不要：
```text
target → archive
candidate → target
```
中间暴露 target absent。

使用 immutable generation + one visibility pointer commit。


## R32-P0-104 — COS/object-store write 使用 distributed fencing

本地 POSIX mutation lock 无法协调多 server 同一 COS prefix。

未来/当前 remote writes：
- generation id；
- ETag/If-Match；
- conditional pointer update；
- fencing epoch；
- idempotency key。


## R32-P0-105 — Metadata mutator 纳入同一 write transaction/security/audit

包括：
- factor catalog refresh/save；
- manifest build/publish；
- schema metadata；
- DQ certification；
- coverage metadata。

security metadata 不能用普通 helper 静默写。


## R32-P0-106 — SQL sandbox 只使用 AST security boundary

regex 不作为 table source authority。

通过 sqlglot/DuckDB AST 枚举：
- comma join；
- CTE；
- subquery；
- table function；
- schema/catalog；
- lateral；
- cross/natural join；
- system tables。

所有 undeclared source reject。


## R32-P0-107 — CanonicalIdentityEncoder 全仓唯一

统一：
- list/tuple 保序；
- set无序；
- map key sort；
- timezone-aware datetime；
- enum type+value；
- bytes deterministic；
- dataclass canonical fields。

production 禁止 repr fallback。


## R32-P0-108 — 关键 identity 至少 128-bit，不用 64-bit short digest

cache display id可短；
source/experiment/security/policy/artifact identity 应 ≥128 bit，最好 full SHA256 internal。


## R32-P0-109 — 版本改成 SCM/tag 自动驱动

`pyproject.toml` 不再人工改 base version。

使用 setuptools-scm 或等价：
- tag: `dataaccess-v0.11.0`；
- non-tag: dev+sha；
- build-time `_build_info.py`。

major/minor/patch 由 release policy/tag决定，不自动猜。


## R32-P0-110 — R30/R40 production package 必须进入 wheel 或迁移出 r30

当前显式 packages 未含 r30。

短期：
- wheel inventory包含实际生产模块。

长期：
- r30能力迁回 canonical packages；
- r30只compat；
- 删除第二套 runtime。


## R32-P0-111 — build SHA 在 build-time 固化，不依赖运行时 .git

wheel/container 中无 `.git` 很正常。

构建生成 `_build_info.py`：
- version；
- sha；
- build id；
- build time；
- dirty flag。

production缺 build info → startup fail。


## R32-P0-112 — 当前 HEAD CI evidence 闭环

当前审计没看到 current HEAD GitHub Actions evidence。

最终：
- commit final code；
- clean clone；
- wheel install；
- tests；
- benchmarks；
- push；
- GitHub Actions；
- evidence 绑定最终 SHA。

没有 current-head workflow/status 时只能写 `NOT PROVEN`。


---

# 5. P1 — 强烈建议本轮一次性补齐的成熟度、性能与长期维护项

下面项目原则上也应一次性处理。若因为工程量必须延期，Closure Report 中必须写明确技术原因、风险与后续 owner，不得直接忽略。



## R32-P1-001 — R30 cache hierarchy 从文档模型迁入真实 cache manager

`CACHE_HIERARCHY` 不能长期只是声明列表；实际 L0-L4 cache 必须使用同一 security/snapshot identity 与 invalidation provider。


## R32-P1-002 — Cache identity 分 logical 与 representation

同一 logical source 的 ArrowBlock、DuckDBRelation、Polars lazy object 是不同 representation；使用 LogicalSourceIdentity + RepresentationIdentity 两层。


## R32-P1-003 — 跨 session cache 明确 immutable/read-only

共享 Arrow/source block必须 immutable；需要修改的 consumer copy-on-write。


## R32-P1-004 — Cache admission 考虑复用次数和 materialization 成本

第一次 scan 前由 ReadWavePlanner 提供 expected consumers，不要等多次 miss 后才决定缓存。


## R32-P1-005 — CostModel 真实支持 MATERIALIZE/RELATION/RESCAN

relation 模式必须真正保留 backend relation，不先 Arrow materialize。


## R32-P1-006 — 缓存 spill 进入统一 Temp/Spill budget

spill路径、容量、回收、checksum、security permissions全部受 ExecutionLease。


## R32-P1-007 — Cache GC exactly-once + crash recovery

持久 cache 有 journal/index；进程 crash 后能清 orphan temp/partial blocks。


## R32-P1-008 — MetadataPlane 一次绑定同一 source generation

Manifest/Partition/Coverage/DQ/ChangeSet不各自重新解析来源。


## R32-P1-009 — MetadataPlane 只读路径禁止隐式 rebuild

reader miss不应顺便 build/write manifest；由 publisher/maintenance明确刷新。


## R32-P1-010 — Metadata cache key 包含 source generation

dataset+params+contract+source generation，不能只靠本地 built epoch。


## R32-P1-011 — PartitionIndex byte_size unknown 保持 None

未知不是0；planner走 conservative。


## R32-P1-012 — Partition pruning 使用 typed bounds

time/instrument min/max 要标 EXACT/RANGE/UNKNOWN，不能把统计摘要当精确 row filter。


## R32-P1-013 — Row-group statistics 进入 planner

Parquet row-group min/max用于大表剪枝，避免只按文件级。


## R32-P1-014 — Manifest statistics 加 statistics authority

区分 publisher exact / parquet footer / sampled / approximate。


## R32-P1-015 — CoverageDescription authority 使用 enum

EXACT / MANIFEST_EXACT / APPROXIMATE / UNKNOWN，不使用任意字符串。


## R32-P1-016 — Coverage concept 参数必须执行字段存在性

传 concept 时真正解析 SemanticField 并计算该字段 coverage，不只是标签。


## R32-P1-017 — Coverage by_instrument 没数据时不要伪造

明确 unavailable/approximate，严格 mining gate 可拒绝。


## R32-P1-018 — Coverage 分位数基于 expected cell ratios

不要对文件权重归一化值求分位数冒充 coverage。


## R32-P1-019 — DQ 分 publish/query/periodic 三层

publish-time full，query-time light certification check，periodic deep。


## R32-P1-020 — DQ certification 绑定 source snapshot+ruleset

source_snapshot_id + contract_digest + DQ_RULESET_VERSION。


## R32-P1-021 — DQ 对 suspension/limit-up/down 做市场语义检查

A股成交量0、价格停牌、涨跌停不能被通用 OHLC 规则误判。


## R32-P1-022 — DQ timezone/session consistency

分钟数据 timestamp 必须落在正确 market session，处理午休、DST、early close。


## R32-P1-023 — DQ duplicate key/grain certification

日频 `(date,instrument)`、财务 `(period,instrument,vintage)` 等唯一键检查。


## R32-P1-024 — DQ numeric finite/nonfinite policy

NaN/Inf/overflow按字段 contract判断，不统一粗暴清洗。


## R32-P1-025 — DQ currency 与 unit/market contract联动

USD/CNY不是简单枚举，字段必须符合 provider/market semantic definition。


## R32-P1-026 — LineageStore DuckDB mode 不维护全量 `_rows` 镜像

DuckDB模式 count/search/export直接SQL，避免长期内存无界。


## R32-P1-027 — Lineage durable mode required/best_effort/disabled

production audit需要 required_durable，打开失败不能静默 memory fallback。


## R32-P1-028 — Lineage connection线程安全

共享 DuckDB connection加队列/锁/单writer或connection-per-thread策略。


## R32-P1-029 — Lineage retention/partition/compaction

按日期/job分区，retention、archive、索引，避免单表无限append。


## R32-P1-030 — Lineage source_snapshot 不得 fallback build_sha/dataset

unknown source显式unknown，不伪造 provenance。


## R32-P1-031 — Audit 与 Lineage 职责分离

Audit=security/compliance；Lineage=data/result provenance；共享typed identities。


## R32-P1-032 — Metrics 区分 Counter/Gauge/Histogram

不要全部 `TYPE counter`。


## R32-P1-033 — Metrics 使用真实 histogram buckets/OTel histogram

需要P50/P95/P99，而不是count/sum/min/max伪histogram。


## R32-P1-034 — scan_bytes 与 result_bytes 分开

backend actual bytes scanned、remote bytes、cache bytes、returned bytes分别记录。


## R32-P1-035 — Metrics error_type 来自 typed exception

不要通过字符串 `split(':')` 猜异常类型。


## R32-P1-036 — Metrics label cardinality policy

禁止 request_id/factor_id/raw path/instrument/user邮箱进Prometheus label。


## R32-P1-037 — QueryTrace 进入 canonical ReadPipeline hooks

auth/contract/resolution/snapshot/resource/backend/normalize/join/serialize/postverify。


## R32-P1-038 — Trace context 跨 stream 生命周期保持

stream每个batch/close/cancel都关联同一trace root。


## R32-P1-039 — Trace/evidence 错误先统一 redaction

异常消息进入metrics/trace/audit前sanitize。


## R32-P1-040 — Read stats 分 planned 与 actual

字段命名必须 `planned_* / estimated_* / actual_*`，不能混。


## R32-P1-041 — Actual object count 来自 backend/open events

不从planner group数量推导。


## R32-P1-042 — Actual scan bytes 尽量从 backend/profile采集

DuckDB profiling/Parquet filesystem wrapper/remote counters。


## R32-P1-043 — ServerPerformanceProfile

CPU、RAM、NUMA、disk、network、COS region、DuckDB threads、FE workers绑定benchmark。


## R32-P1-044 — Cold/Warm 分层报告

OS cache、DuckDB metadata、DA session、source block、mirror、remote client分别标记。


## R32-P1-045 — Synthetic fixture 固定seed和hash

benchmark evidence记录seed/schema/rows/fixture digest。


## R32-P1-046 — 真实COS production-like benchmark

CI synthetic之外定期在真实服务器跑cold/warm/profile。


## R32-P1-047 — Benchmark只用public execution path

private API只做fixture setup，不作为被测路径。


## R32-P1-048 — Benchmark从FE expression compile开始

完整链：compile→source plan→wave→DA→operator。


## R32-P1-049 — FactorArtifact绑定完整实验身份

SourceSnapshots + Calendar + Universe + Plan + ExecutionIdentity。


## R32-P1-050 — DataDemandIdentity与FactorIdentity分离

不同factor可以共享同一数据需求CSE，但factor identity仍不同。


## R32-P1-051 — CSE compatibility typed化

Security/PIT/Universe/Provider/PriceBasis/Revision/Join/Grain不同绝不合并。


## R32-P1-052 — Aggregation recipe digest完整且保序

不能只取第一个aggregation；window/session/missing/halt policy全部进identity。


## R32-P1-053 — Join contract进入source block identity

ASOF tolerance、keys、grain、revision policy、availability axis。


## R32-P1-054 — SourceScopeId加入provider/source version

Wind/Massive/其他provider同名字段不能误复用。


## R32-P1-055 — MultiMarket batch显式MarketScope

A股+美股不能取第一个market；必须CalendarAlignment/Currency/CrossSection policy。


## R32-P1-056 — Grain与Timeframe分开

grain是键结构，timeframe是季度/TTM/daily等；禁止互相fallback。


## R32-P1-057 — PIT用 REQUIRED/NOT_REQUIRED/UNKNOWN

UNKNOWN生产不可执行；None不能同时表示不需要和没证明。


## R32-P1-058 — PriceBasis由SemanticField权威推导

adjusted_close不能planner缺值就默认raw。


## R32-P1-059 — ResultGrain certification

join后检查1:1/N:1/duplicates/row explosion，FE只消费certified panel。


## R32-P1-060 — Backend switching不改变SourceSnapshot

backend/version进入ExecutionIdentity；logical source不变。


## R32-P1-061 — SemanticExecutionVersion

README等无关提交不清结果cache；PIT/compiler/operator语义改动才失效。


## R32-P1-062 — Metadata cache和result cache分层失效

metadata由source generation；result由source + semantic execution。


## R32-P1-063 — PolicyManifest deep immutable

外部 supplied digest 必须重新计算验证，不能caller随便传。


## R32-P1-064 — security digest失败不返回共享None

production fail；research生成unshareable unknown scope。


## R32-P1-065 — Security policy/version进入缓存和artifact identity

权限变更后旧cache/artifact不能自动当当前可访问。


## R32-P1-066 — Credential client pool

key=endpoint+region+scope+credential generation；避免每HEAD建boto client。


## R32-P1-067 — Credential rotation generation-safe

旧query旧lease，新query新generation；不修改正在执行连接。


## R32-P1-068 — Remote retry受统一deadline/budget

指数backoff+jitter但不突破absolute deadline和remote request budget。


## R32-P1-069 — Remote retry不重试永久安全错误

403/invalid token等按typed classifier处理。


## R32-P1-070 — Remote LIST分页完整

大prefix必须处理continuation token；实际object count进入budget。


## R32-P1-071 — Publisher manifest签名/完整性扩展点

如果将来数据由独立publisher生成，预留签名/key id/hash chain，先定义接口。


## R32-P1-072 — Manifest generation monotonicity

同dataset generation不能倒退；rollback必须显式标记。


## R32-P1-073 — Manifest object排序/Unicode canonicalization

digest跨平台/语言稳定，不受dict/路径编码差异影响。


## R32-P1-074 — Local file checksum策略

小文件全hash，大文件可publisher checksum/immutable generation；不能永远只mtime。


## R32-P1-075 — Snapshot post-verify按实际扫描对象

只验证reader实际打开的object set，不再次glob猜测。


## R32-P1-076 — fail_if_changed从plan到collect全过程保持

stream/lazy collect时仍校验绑定generation。


## R32-P1-077 — Pinned snapshot pin失败必须fail

任何required object stat/HEAD失败不能降为空pin。


## R32-P1-078 — Pinned snapshot只pin relevant scope但必须证明pruning

时间/股票剪枝使用authoritative manifest；approximate不能减少pin scope。


## R32-P1-079 — Calendar freeze后禁止live set_calendar

production session active期间修改calendar应拒绝或创建新execution generation。


## R32-P1-080 — CalendarSnapshot覆盖minute session细节

A股午休；US DST/early close；timezone。


## R32-P1-081 — UniverseSnapshot包含membership effective intervals

不能只hash当前股票列表。


## R32-P1-082 — Universe PIT规则

上市/退市/ST/指数成分等变化按decision time可见。


## R32-P1-083 — Financial RevisionIdentity一等化

instrument+period_end+knowledge_time+vintage+provider revision。


## R32-P1-084 — Financial YTD/quarterly/TTM semantic compiler

避免不同provider flow口径混用。


## R32-P1-085 — Corporate action与price adjustment source identity

复权因子版本进入PriceBasis identity。


## R32-P1-086 — A股limit/suspension calendar语义进入minute→daily

停牌、涨跌停、集合竞价、午休明确policy。


## R32-P1-087 — US corporate action/filing timezone

filing timestamp按US market decision clock转换，不能只取date。


## R32-P1-088 — Cross-market currency policy

跨市场因子必须显式FX source/PIT/转换时间，不隐式比较CNY和USD。


## R32-P1-089 — SourceAdapter unknown format fail closed

不默认为Parquet capability。


## R32-P1-090 — Format/Storage/Backend capability三层拆分

Parquet、S3、DuckDB分别建capability，组合而非单bool。


## R32-P1-091 — CSV production显式schema

禁止sample inference；声明delimiter/encoding/null/timestamp。


## R32-P1-092 — Arrow/CSV同样有SourceSnapshot

不是只有Parquet才可复现。


## R32-P1-093 — Backend capability需要versioned certification

DuckDB/Polars/PyArrow feature matrix绑定版本和测试。


## R32-P1-094 — Polars native与delegate明确标记

FE/DA不要把Pandas delegate写成Polars native。


## R32-P1-095 — DuckDB connection fork/PID guard

fork后旧connection不可复用；worker初始化重新建。


## R32-P1-096 — DuckDB per-query settings恢复

threads/memory_limit/temp_directory等请求级设置结束后恢复，不污染下一请求。


## R32-P1-097 — DuckDB temp spill cleanup

异常/取消后temp files回收；spill bytes计入lease。


## R32-P1-098 — Polars collect resource gate

LazyFrame collect前再次检查deadline/memory/snapshot。


## R32-P1-099 — Arrow conversion避免无意义copy

能zero-copy就zero-copy；转换次数进trace。


## R32-P1-100 — JSON HTTP响应限制

to_pandas/to_json可能复制巨量内存；JSON使用更严格row/byte上限或stream encoder。


## R32-P1-101 — HTTP response serialization也计入deadline/resource

读完数据不代表请求结束；Arrow/Parquet/JSON编码也受预算。


## R32-P1-102 — HTTP disconnect向backend取消

ASGI disconnect触发CancellationToken，停止DuckDB/remote/serialization。


## R32-P1-103 — Rate limit/per-principal fairness

服务层并发槽之外，按principal和cost class做公平排队。


## R32-P1-104 — Health/Liveness/Readiness分层

/health只进程活；/ready可服务；/deep-health做远端依赖。


## R32-P1-105 — Service schema version header

API version、contract version、build id在响应元数据明确。


## R32-P1-106 — HTTP error taxonomy

400 validation / 401 authn / 403 authz / 404 visible-not-found / 409 snapshot conflict / 413 budget / 429 throttle / 503 dependency。


## R32-P1-107 — 避免metadata existence leak

未授权dataset/factor的404/403策略一致，避免枚举敏感资源。


## R32-P1-108 — read_uri privileged API默认production禁用

即便有权限也最好deployment feature gate，日志完整审计。


## R32-P1-109 — write namespace生产显式

namespaced/staging write必须run_id/principal namespace，不用warning fallback。


## R32-P1-110 — Write idempotency key

重试append/publish不能生成重复数据。


## R32-P1-111 — Write precondition绑定source generation

upsert/delete基于哪个generation执行必须明确，防lost update。


## R32-P1-112 — Schema evolution在真实write path执行

不只测试helper；兼容性检查发生在commit前。


## R32-P1-113 — Write DQ gate

publish前exact candidate snapshot通过required DQ certification。


## R32-P1-114 — Write lineage

记录parent generation、candidate generation、writer build/principal/job。


## R32-P1-115 — Delete tombstone语义

对象存储删除用generation/tombstone，reader不会复活旧对象。


## R32-P1-116 — Factor catalog security metadata版本化

access tags/classification变更形成metadata generation。


## R32-P1-117 — ClickHouse数据也统一Snapshot/PIT/QueryBudget

不能成为独立弱治理路径。


## R32-P1-118 — ClickHouse factor version identity

查询/物化结果绑定表版本、schema、revision/source。


## R32-P1-119 — SQL relation/CTE预算

AST解析后对每个declared dataset成本求和，再admit。


## R32-P1-120 — SQL output grain/row explosion gate

复杂join在执行前估算，执行后验证actual rows。


## R32-P1-121 — SQL UDF/table function allowlist

禁止任意filesystem/http/extension function逃逸。


## R32-P1-122 — SQL extension install/load限制

production禁止SQL动态INSTALL/LOAD未批准extension。


## R32-P1-123 — Runtime registry deep immutable

Dataset dataclass nested dict全部freeze，不能get_dataset后原地改schema。


## R32-P1-124 — Registry reload用generation swap

构建新registry→validate→atomic swap；旧session绑定旧generation完成。


## R32-P1-125 — Config/environment解析统一typed parser

bool/int/enum不散落各模块手写。


## R32-P1-126 — RuntimeModeIdentity单一权威

production/strict/research所有模块不各自重读env。


## R32-P1-127 — Build/Contract/Schema版本职责清晰

package version、API_VERSION、CONTRACT_SCHEMA_VERSION、STORAGE_FORMAT_VERSION分开。


## R32-P1-128 — Compatibility matrix

reader/writer最小版本、forward/backward compatibility、migration tests。


## R32-P1-129 — Deprecation policy

旧API alias有warning/version removal plan；不要永久维护双轨。


## R32-P1-130 — Public API surface inventory

自动列出public methods并标 GOVERNED/INTERNAL/UNSAFE/DEPRECATED。


## R32-P1-131 — Raw I/O bypass static audit

搜索read_parquet/scan_parquet/open/boto3/sql_relation等，FactorEngine生产路径必须经过DA。


## R32-P1-132 — Fail-open static audit

搜索except Exception/return None/pass/PASS，按security/PIT/identity/resource/DQ分类。


## R32-P1-133 — Identity static audit

搜索hash/repr/json.dumps/fingerprint/cache_key/snapshot_id，全部归CanonicalEncoder。


## R32-P1-134 — Mutable singleton static audit

registry/cache/calendar/credential/governor全局单例检查request isolation。


## R32-P1-135 — Thread/fork/process safety audit

所有global mutable state标thread-safe/fork-safe/process-local。


## R32-P1-136 — Cancellation audit

所有blocking IO/lock/semaphore/remote/backends都能被deadline/cancel终止。


## R32-P1-137 — Memory ownership audit

Arrow/Polars/DuckDB buffers、cache、stream batch谁持有谁释放明确。


## R32-P1-138 — Large-object backpressure

stream producer不能无限领先consumer；bounded queue/iterator。


## R32-P1-139 — Adaptive batch size

结合row width/memory headroom调batch，不固定一套大小。


## R32-P1-140 — Remote object coalescing

相邻小对象/partition可按ReadWave批量读取，但不跨snapshot/security scope。


## R32-P1-141 — Small-file problem监控

对象数/平均size/metadata latency进入指标，发布端可compact。


## R32-P1-142 — Parquet compression/rowgroup基准

对A股daily/minute测试zstd/snappy、rowgroup大小、列裁剪收益。


## R32-P1-143 — Partition strategy基准

daily/yearly/monthly按真实常见query pattern衡量，而不是固定理念。


## R32-P1-144 — Minute→daily aggregation pushdown

尽量在source/backend侧只读所需列并聚合，避免分钟全量出backend。


## R32-P1-145 — Fundamental PIT join pushdown

在保持availability正确前提下减少先全量materialize再join。


## R32-P1-146 — Universe filter pushdown

已知exact PIT universe时尽早pushdown instrument list，避免全市场扫描。


## R32-P1-147 — Column projection completeness test

FE factor leaf fields→DA projection exact，无少列无多读。


## R32-P1-148 — PreparedRead explainability

输出为什么选这些objects/backend/pit policy/cache scope/resource estimate。


## R32-P1-149 — Query plan hash稳定性

语义相同请求hash相同；顺序有语义的字段不排序。


## R32-P1-150 — Production evidence truth gate

任何报告“green/actual/physical/prod-safe”必须有机器可核验证据来源。


---

# 6. P2 — 中期结构升级与未来实盘准备

这些不一定阻塞本轮核心 Freeze，但接口应该现在就设计对，避免两个月后重构。



## R32-P2-001 — 迁移到标准 `src/data_access` package layout

完成当前 correctness 后单独迁移，避免显式package-dir长期漏包。


## R32-P2-002 — MetadataStore Provider接口

manifest/index/coverage/DQ/lineage未来可从local切object store/DB。


## R32-P2-003 — Distributed MaterializationClaimProvider

多server同一cold source block避免重复materialize，当前默认local singleflight。


## R32-P2-004 — Distributed ResourceLimiter接口

如果未来多worker/多server，不再依赖process-local governor。


## R32-P2-005 — HybridSourceSnapshot

Historical immutable generation + live sequence checkpoint。


## R32-P2-006 — LiveData sequence/watermark

实盘分钟/逐笔数据需要offset/watermark/late data/replay。


## R32-P2-007 — Late-arrival revision replay

晚到数据形成ChangeSet，FE stateful/rolling局部重算。


## R32-P2-008 — Exactly-once materialization protocol

Factor artifact写入采用run id/generation/commit marker。


## R32-P2-009 — Cross-region source policy

COS region与计算节点region进入cost/latency planner。


## R32-P2-010 — Multi-provider failover不自动混源

provider failover必须生成新SourceScope/Experiment identity并满足语义parity。


## R32-P2-011 — Provider semantic certification

同一Concept不同vendor字段定义做unit/PIT/revision parity认证。


## R32-P2-012 — Source SLA registry

freshness/expected publish delay/availability window进入数据契约。


## R32-P2-013 — Data freshness monitor

交易日数据过期自动标not-ready，不靠用户发现。


## R32-P2-014 — Backfill orchestration interface

历史补数据后自动生成typed DataChangeSet并触发最小重算。


## R32-P2-015 — Data contract migration tool

schema/semantic/storage升级有dry-run、diff、rollback。


## R32-P2-016 — Dataset provenance graph

原始vendor→normalized→aggregated→factor lake lineage图。


## R32-P2-017 — Semantic catalog codegen

从registry/contract生成FE field catalog和docs，减少双维护。


## R32-P2-018 — Policy-as-code测试

security policy变更有fixture和negative tests。


## R32-P2-019 — Chaos/fault injection

模拟COS timeout/403/partial LIST、disk full、DuckDB cancel、manifest corrupt。


## R32-P2-020 — Soak test

长时间1000+factor job验证cache/lineage/client/lease无泄漏。


## R32-P2-021 — Memory fragmentation监控

Arrow/Polars长期服务RSS不回落时有指标和worker recycle policy。


## R32-P2-022 — Hot reload generation isolation

新registry/calendar/policy只影响新request，旧request继续旧generation。


## R32-P2-023 — Blue/green data generation

大规模数据发布可并存验证后切pointer。


## R32-P2-024 — Replayable execution manifest

一次factor run可导出所有source/plan/calendar/universe/backend identities。


## R32-P2-025 — Artifact garbage collection

根据lineage引用关系清理不再使用generation/cache/artifact。


## R32-P2-026 — Performance regression CI

代表性micro/macro benchmark超阈值自动告警，不与correctness gate混为一个。


## R32-P2-027 — Cross-backend differential testing

DuckDB/Polars/PyArrow同一PreparedRead输出semantic parity。


## R32-P2-028 — Property-based testing

identity canonicalization、partition pruning、PIT join、manifest parser用Hypothesis/fuzz。


## R32-P2-029 — Security fuzzing

SQL AST、URI/path boundary、manifest prefix、API参数做恶意输入fuzz。


## R32-P2-030 — Documentation generation

current-state docs从canonical contracts/code生成，旧Rxx报告归档为history。


---

# 7. R31 中仍必须继续验证的残留项

下面这些即使 R31 已经写过，当前最新 HEAD 仍应重新验证，不允许因为“以前文档提过”就自动标 CLOSED：

```text
1. static base version 0.10.2
2. data_access.r30 wheel/package inventory
3. CanonicalIdentityEncoder
4. SourceSnapshot vs ExecutionIdentity
5. source-block exact cache identity
6. source-block release/singleflight
7. DataReadSession canonical convergence
8. CostModel真实三态
9. typed Concept→Source binding
10. FE batch projection correctness
11. dependency extraction fail-closed
12. BatchDataRequest / ReadWavePlanner单一authority
13. actual scan evidence
14. QueryTrace进入ReadPipeline
15. MetadataPlane read-only boundary
16. Concept/Unit→SemanticField
17. STS token / credential lease
18. Coverage/DQ correctness
19. ExecutionLease child return
20. Lineage durability
21. Metrics真实语义
22. Adapter unknown format fail-closed
23. final SHA / wheel / CI evidence
```

---

# 8. Canonical Migration Ledger — 必须创建

生成：

```text
dataaccess/docs/R32_CANONICAL_MIGRATION_LEDGER.md
```

表：

| Capability | Current Implementations | Canonical Target | Action | Tests | Status |
|---|---|---|---|---|---|

至少列：

```text
ReadSession
ResolutionCache
SourceBlockCache
QueryCache
MetadataPlane
PartitionIndex
Coverage
DataQuality
CalendarSnapshot
UniverseSnapshot
SourceSnapshot
DataSnapshot
ExperimentSnapshot
ExecutionLease
ResourceGovernor
ResolutionLease
QueryTrace
Metrics
Lineage
Policy
SemanticField/Concept/Unit
FactorSourcePlan
BatchDataRequest
ReadWavePlanner
ChangeImpact
```

Action 只能：

```text
KEEP_CANONICAL
MIGRATE
DELEGATE
COMPAT_ALIAS
DELETE_DUPLICATE
```

---

# 9. Public Surface Audit — 必须创建

生成：

```text
dataaccess/docs/R32_PUBLIC_API_SURFACE.csv
```

字段：

```text
module
symbol
public
governed
production_allowed
security_checked
snapshot_bound
budget_bound
deadline_bound
trace_bound
replacement
status
```

重点审：

```text
read
read_result
read_arrow
read_arrow_stream
read_cached
read_uri
read_joined
read_factors
scan
scan_polars
sql
sql_relation
write_arrow
upsert
delete
publish
build_dataset_manifest
refresh_factor_catalog
set_calendar
physical_scope
```

凡是：
```text
public=True
production_allowed=True
governed=False
```
必须修或降级 internal/unsafe。

---

# 10. `except Exception` 全仓分级审计

运行 repo-wide 静态搜索。

每一个 broad exception 必须分类：

## A. 严禁吞

```text
security/auth
credential
PIT
calendar
snapshot
identity
contract
dependency extraction
resource admission/release
write commit
change impact
DQ certification
durable lineage
policy/version compatibility
```

## B. 可 best-effort

```text
debug metric
optional trace exporter
non-critical docs
```

但也必须：
- typed metric；
- 不改变主语义；
- 不把失败伪装PASS。

输出：

```text
dataaccess/docs/R32_FAIL_OPEN_AUDIT.csv
```

---

# 11. Identity 全仓审计

搜索：

```text
hash(
repr(
json.dumps(
sha256(
stable_digest
fingerprint
snapshot_id
cache_key
plan_hash
```

每个 identity 标：

```text
SOURCE
EXECUTION
SECURITY
PLAN
CACHE
ARTIFACT
METADATA
```

禁止一个 build SHA/registry hash无脑塞进所有identity。

生成：

```text
R32_IDENTITY_LEDGER.csv
```

---

# 12. Resource / Deadline 全仓审计

搜索所有：

```text
Semaphore
Lock
Condition
Queue
timeout
deadline
requests
head_object
list_objects
duckdb.execute
collect
to_arrow
```

检查：

```text
deadline?
cancellation?
resource lease?
wait metrics?
finally release?
external callback under lock?
```

目标：

```text
一个 absolute deadline
一棵 lease tree
一个 cancellation tree
```

---

# 13. Cache 全仓审计

每种 cache 必须回答：

```text
What is cached?
Logical identity?
Representation identity?
Security scope?
Snapshot scope?
TTL?
Generation invalidation?
Max entries/bytes?
Eviction?
Singleflight?
Ownership/release?
Cross-process?
Negative cache?
Audit on hit?
```

任何答不出来：
```text
not production-ready
```

---

# 14. Snapshot/Manifest destructive tests

必须新增：

## T-R32-SNAP-001 — local same path same size replacement

只改：
```text
mtime/content
```
snapshot identity 必须变。

## T-R32-SNAP-002 — empty authoritative generation

明确：
```json
objects=[]
object_count=0
complete=true
```
能得到 canonical non-empty digest。

缺 `objects`：
```text
strict fail
```

## T-R32-SNAP-003 — manifest complete missing

strict fail。

## T-R32-SNAP-004 — zero-byte object

size=0 保留0，不变None。

## T-R32-SNAP-005 — future schema version

unknown manifest_version：
```text
fail
```

## T-R32-SNAP-006 — published_at invalid/future beyond skew

fail。

## T-R32-SNAP-007 — local publisher manifest无object identity

不能自动满足production minimum evidence。

## T-R32-SNAP-008 — boundary collision

```text
.../abc
.../abcd
```
不得越界。

## T-R32-SNAP-009 — FileSelector helper path

split resolver helper 不能读到 `shares_*`。

---

# 15. Resource destructive tests

## T-R32-LEASE-001

已有1 discovery slot；
本轮取3，第2个governor失败；
原slot仍在，本轮slot全回滚。

## T-R32-LEASE-002

child release后父budget恢复。

## T-R32-LEASE-003

多线程 child allocate/release 总量不超过parent。

## T-R32-LEASE-004

deadline已过：
- resolution fail；
- duckdb wait fail；
- remote wait fail。

## T-R32-GOV-001

host-backed模式 duplicate query id仍reject。

## T-R32-GOV-002

host coordinator配置后抛异常：
```text
production fail
```
不能local fallback。

## T-R32-GOV-003

host lease成功，本地bookkeeping失败：
host lease必须rollback。

---

# 16. Cache / Session destructive tests

## T-R32-CACHE-001

Principal A premium read填cache；
Principal B无权：
同请求必须AccessDenied，不能hit数据。

## T-R32-CACHE-002

相同dataset：
Close vs Volume key不同。

## T-R32-CACHE-003

相同dataset：
不同instrument/timeframe/price basis/provider/revision key不同。

## T-R32-CACHE-004

`clear_resolution_cache()` 后 mutation_generation重新读取。

## T-R32-SESSION-001

double enter reject。

## T-R32-SESSION-002

`set_runtime_mode_identity` 成功后、set cache抛异常：
execution context/mode全部恢复。

## T-R32-SESSION-003

critical cleanup故障：
worker health/metric能看到，不静默。

## T-R32-SINGLEFLIGHT-001

100并发同PreparedRead：
真实resolution执行一次。

---

# 17. DQ / Coverage destructive tests

## T-R32-DQ-001
每个 checker 人为抛异常：
```text
CHECK_FAILED
```
绝不是PASS。

## T-R32-DQ-002
A股停牌：
volume=0不自动判坏。

## T-R32-DQ-003
高低开收别名通过SemanticField解析。

## T-R32-COVERAGE-001
完整A股交易日：
daily coverage≈1。

## T-R32-COVERAGE-002
周末/节假日不降低coverage。

## T-R32-COVERAGE-003
文件存在但目标字段全null：
field coverage=0，不是1。

## T-R32-COVERAGE-004
multi-day file rows不能复制给每天。

---

# 18. ChangeImpact destructive tests

## T-R32-IMPACT-001
`x[t]`变，MA20影响`t..t+19`。

## T-R32-IMPACT-002
跨春节/国庆用20 trading bars，不是20自然日。

## T-R32-IMPACT-003
一只股票ROE变：
`rank(ROE)`同日整个universe affected。

## T-R32-IMPACT-004
industry neutralize：
同组扩散。

## T-R32-IMPACT-005
split adjustment：
允许backward impact。

## T-R32-IMPACT-006
min/max instrument统计：
不能生成ExactSet。

## T-R32-IMPACT-007
多个changed columns：
全部进入FE matcher。

## T-R32-IMPACT-008
universe membership change：
对应日期cross-section全部重算。

---

# 19. HTTP/Streaming destructive tests

```text
client disconnect
first batch empty
first batch error
serialization error
body cancellation
backend timeout
auth failure
budget failure
```

每种验证：

```text
query_slots=returned once
resource leases=0
duckdb slots=0
remote slots=0
temp/spill cleaned
trace closed
audit emitted
```

另外：

- production `--skip-startup-gate` 拒绝；
- 直接 `uvicorn app:app` 仍触发 startup gate；
- `/ready` 不每次做远程LIST/HEAD；
- `/ready` 不泄内部path；
- factor read拥有完整QueryBudget。

---

# 20. Write transaction destructive tests

## Append
暂停写入 M 个partition中间：
并发reader只能看到 old 或 complete new。

## Upsert/Delete
多个partition提交中暂停：
不得mixed generation。

## Publish
旧target与新candidate切换中暂停：
不得出现target missing。

## Concurrent Writers
两个server相同base generation：
只有一个conditional commit成功。

## Unauthorized Write
read-only principal：
write/upsert/delete/publish/catalog/manifest全部拒绝。

## Boundary
dataset A写向dataset B root：
拒绝。

---

# 21. SQL sandbox destructive tests

必须覆盖：

```sql
FROM a,b
CROSS JOIN
NATURAL JOIN
LATERAL
CTE
nested subquery
information_schema
read_parquet(...)
read_csv(...)
httpfs/URL
ATTACH
COPY
INSTALL
LOAD
PRAGMA
```

Production只允许明确allowlist。

---

# 22. FE × DA 100 / 1000 / 10000 factors 真集成测试

构造：

```text
price-only
price+volume
fundamental PIT
minute-derived daily
cross-sectional
mixed A/US（单独测试）
```

完整链：

```text
Expression compile
→ dependency extraction
→ typed SourceBinding
→ BatchDataRequest
→ ReadWavePlanner
→ PreparedRead
→ DataReadSession
→ SourceBlock CSE
→ Backend
→ FE operator DAG
```

必须采集实际：

```text
factor_count
unique_concepts
unique_source_scopes
planned_read_waves
planned_scan_count
actual_backend_scan_count
actual_object_open_count
actual_remote_requests
actual_scan_bytes
source_block_producers
source_block_consumers
cache hits
resolution calls
metadata calls
DA wall time
FE compute time
serialization time
peak RSS
spill bytes
```

核心：

```text
actual_backend_scan_count << factor_count
```

且不能靠关闭PIT/snapshot/security实现。

---

# 23. 多数据读取能力验收矩阵

DataAccess必须明确测试：

| 数据组合 | 预期 |
|---|---|
| A股日行情 + 财务 | PIT join |
| A股日行情 + 股票池 | time-varying universe |
| A股分钟 + 日行情 | minute→daily / mixed grain |
| A股行情 + 行业 | as-of/group |
| A股行情 + 复权因子 | price basis |
| US daily + financial | filing-date PIT |
| A股 + US | explicit multi-market alignment |
| 多provider同concept | provider scope isolation |
| premium + public source | security-scope isolation |
| historical + live | future HybridSnapshot |

对每类记录：
- binding；
- join；
- calendar；
- PIT；
- unit；
- snapshot；
- actual reads；
- result grain。

---

# 24. 性能专项：怎样做到“尽可能快”

这轮不要单纯追求单条SQL毫秒，而要优化 FactorEngine 的整体数据准备成本。

## 第一优先级

```text
同数据不重复resolve
同对象不重复HEAD
同source不重复scan
同block不重复materialize
同batch不重复join
```

## 推荐热路径

```text
1 job
→ 1 DataReadSession
→ compile all factor demands
→ group by SourceScope
→ one resolution per scope
→ one physical scan/read wave per compatible scope
→ immutable source blocks
→ N factor consumers
```

## 必须 profile 的热点

```text
glob
stat
manifest parse
LIST
HEAD
boto client create
DuckDB connection
Parquet open
row-group scan
Arrow materialization
Polars collect
join
unit normalize
PIT availability join
serialization
```

判断调用频率：

```text
per factor   ← 尽量消灭
per source   ← 可以
per session  ← 更好
per generation ← metadata最好
```

---

# 25. 性能优化具体任务

1. metadata provider session cache；
2. S3 client pool；
3. resolution singleflight；
4. prepared-read singleflight；
5. source-block singleflight；
6. row-group pruning；
7. early projection；
8. early exact universe filter；
9. minute aggregation pushdown；
10. PIT join pushdown；
11. backend relation reuse；
12. Arrow zero-copy；
13. adaptive record batch；
14. bounded stream backpressure；
15. remote metadata batch/list；
16. object-count-aware ReadWave；
17. cold/warm calibration；
18. server-specific calibration；
19. small-file compaction recommendations；
20. publish-time metadata precompute。

---

# 26. 不允许的“伪性能优化”

禁止：

```text
关闭 snapshot
减少PIT检查
不做security scope
用mtime伪vintage
把unknown bytes当0
把planned scans写成actual
把full materialize叫stream
用全局cache跨principal
为了快而裸scan_polars
为了快让FE直接read_parquet
```

---

# 27. A股 production 专项验收

必须覆盖：

```text
交易日历
春节/国庆
午休
停牌
涨跌停
新股上市
退市
ST/风险警示（若universe policy需要）
复权
分红送转
财报period_end
PubDate
revision
YTD flow
TTM
行业分类as-of
指数成分PIT
分钟→日频
```

---

# 28. US production 专项验收

必须覆盖：

```text
NYSE/Nasdaq calendar
DST
early close
filing timestamp
period_end
revision/vintage
USD unit
split/shares
corporate action
date-label availability
provider source version
```

---

# 29. Version / Release 最终方案

使用 SCM/tag。

示例：

```text
tag: dataaccess-v0.11.0
→ package: 0.11.0

后续 commit:
→ 0.11.1.devN+g<sha>
```

build-time：

```text
_build_info.py
```

暴露：

```text
PACKAGE_VERSION
BUILD_SHA
BUILD_ID
BUILD_TIME
GIT_DIRTY
```

`/version` 返回这些，不从运行时git读取。

---

# 30. Wheel 验收

必须：

```bash
python -m build
python dataaccess/scripts/check_wheel_inventory.py
python -m venv /tmp/da-r32-venv
pip install dist/data_access-*.whl
cd /tmp
python -c "import data_access"
```

导入全部生产包。

如果 r30 已迁出：
- 不要求保留 r30 public；
- compatibility alias按迁移计划测试。

---

# 31. Clean Checkout 验收

不能只在开发目录跑。

```bash
git clone <repo> clean
git checkout <FINAL_SHA>
pip install wheel
pytest
```

确保：
- 没有未提交本地文件；
- 没有依赖 `.gitignored` credential/module；
- 没有依赖 repo root import side effect。

---

# 32. GitHub Actions 必须建立/确认

至少：

```text
lint
unit
integration
wheel-install
security-negative
PIT-negative
cross-backend
FE×DA integration
benchmark-smoke
```

当前 HEAD 如果没有workflow evidence：
```text
不得写 CI GREEN
```

---

# 33. Evidence Truth 格式

每份最终 evidence：

```text
code_sha
package_version
build_id
fixture_hash
machine_profile
dataset/source snapshot ids
calendar snapshot ids
universe snapshot id
semantic execution version
planner digest
actual counter source
timestamp
```

估算字段：
```text
planned_*
estimated_*
```

实际字段：
```text
actual_*
```

---

# 34. R32 Closure Report

生成：

```text
DATAACCESS_R32_FINAL_ACCEPTANCE_REPORT.md
```

必须包含：

## Baseline
- execution start SHA
- final SHA
- clean status

## Closure
- P0逐条
- P1逐条
- P2 deferred

## Canonical convergence
- 删除了哪些parallel implementation
- 哪些compat alias仍保留
- public canonical path

## Correctness
- PIT
- revision
- snapshot
- security
- DQ
- coverage
- ChangeImpact
- write atomicity

## FE integration
- actual scan evidence
- 100/1000/10000 factors

## Performance
- cold/warm
- machine profile
- peak RSS
- remote metadata
- source reuse

## Release
- SCM version
- wheel
- clean install
- current-head CI

---

# 35. AI 不得提前结束的条件

Coding AI 不得因为：
```text
pytest passed
```
就结束。

它必须继续执行：

```text
repo-wide static audits
→ destructive tests
→ integration tests
→ performance truth
→ canonical migration
→ final clean checkout
→ CI evidence
```

只有没有新的 concrete blocker，才可以结束。

---

# 36. Freeze 标准

只有同时满足：

1. 所有 R32 P0 CLOSED；
2. R31残留已重新验证；
3. r30/r40不再形成第二套production runtime；
4. source/execution/security/plan identity分层清楚；
5. snapshot/manifest destructive tests通过；
6. STS/credential生命周期通过；
7. DQ无fail-open；
8. Coverage基于正确expected grain；
9. ChangeImpact支持forward/bar/cross-section/revision；
10. write generation atomic；
11. HTTP stream无资源泄漏；
12. FE×DA真实1000/10000 factors CSE成立；
13. actual counters不再冒充planner估算；
14. wheel clean install；
15. SCM version自动；
16. final SHA GitHub CI有证据；

才允许：

```text
DATAACCESS CORE FREEZE = YES
```

---

# 37. 最后要求：修完本文后继续自己找问题

完成所有已知条目后，Coding AI 必须再做最后一次全仓主动审计，至少从以下维度各扫描一遍：

```text
security
PIT
revision
calendar
universe
snapshot
manifest
identity
cache
resource
deadline
concurrency
thread/fork
remote/COS
write/publish
SQL
HTTP/stream
metadata
coverage
DQ
lineage/audit
metrics
FE integration
incremental recompute
cross-market
units/currency
packaging
versioning
CI
performance
```

发现新的 concrete issue：
- 直接修；
- 补测试；
- 写入 Closure Report；
- 不需要再等待用户发下一份提示词。

目标不是“把 R32 做完”。

目标是：

> **让当前 DataAccess 在不牺牲 PIT / 安全 / 可复现性的前提下，成为 FactorEngine 大规模多数据、多因子批量读取的唯一高性能数据底座，并把重复 runtime、模糊 identity、fail-open 和伪 evidence 一次性收干净。**
