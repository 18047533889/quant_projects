# FactorEngine × DataAccess 最新 HEAD 全量整改提示词
## 260 项全量缺陷与优化：跨路径一致性、生产安全、资源治理、数值语义、市场时钟、身份/缓存/证据闭环

> 本文件可直接发送给 coding AI / Codex / Cursor / Claude Code。
>
> 审计基线：本轮审计时 `main = 8e9893b562f80e083c4baed2e0a15eb4e040cdc9`。**开始整改前必须重新读取仓库最新 HEAD**；若代码已变化，以最新 HEAD 为准逐项复核。
>
> 仓库：`https://github.com/18047533889/quant_projects`
>
> 主范围：`factor_engine/**`、`dataaccess/**` 及其 CI、packaging、evidence、service/runtime、tests。

## 总目标

面向 A 股日频横截面、分钟→日频、财务/PIT、多源事件/关系数据以及未来美股扩展，把 FactorEngine × DataAccess 修成企业级生产系统。服务 AlphaProbe、AlphaMiner、FactorMiner、CogAlpha、AlphaSage、EvoAlpha、AlphaCFG、QuantaAlpha 等自动因子挖掘算法。

目标生产链：

```text
Factor batch
→ Compile / Typed IR
→ canonicalize
→ unified dependency graph
→ immutable ExecutionSemanticContext
→ DataAccess PreparedBatchReadSession
→ pinned snapshot
→ history/source clustered ReadWave
→ DuckDB / Polars / Numpy-Numba cost-driven physical plan
→ real physical-stage execution
→ FactorBlock
→ block DQ
→ streaming merge/materialize
→ streaming partition writer
→ COW Generation
→ atomic commit
→ current-SHA evidence
```

统一资源树：

```text
HostResourceCoordinator
├─ JobLease
│  ├─ ComputeLease
│  ├─ DataAccessScanLease
│  ├─ DuckDBLease
│  ├─ PolarsLease
│  ├─ NumbaLease
│  ├─ CacheLease
│  ├─ SpillLease
│  └─ WriterLease
```

## 绝对禁止的“假修复”

以下均不算完成：

1. 只改注释、README、报告、metadata、catalog。
2. 只增加 gate，但真实生产主链不经过。
3. 测试里手工设置 `shardable=True`、fake contract 绕过真实 lowerer/planner。
4. 只测 helper，不测 production entry point。
5. canonical whitelist 代替 parameter/backend/variant 证据。
6. stale evidence 继续复用。
7. broad `except Exception: pass/return/default` 吞错。
8. test-only helper 让测试过、正式代码不用。
9. 强制 Pandas fallback 掩盖 Polars/DuckDB/Numba 问题。
10. 关闭 PIT/DQ/snapshot/resource gate 换速度。
11. 人工 `production_safe=True` 代替可执行证据。
12. 固定 252、`BDay`、`bdate_range` 代替真实市场日历。
13. OOM 后同 shape 原样重试。
14. failed/unknown/unresolved 当 empty/zero/false。
15. 为绿测试删除 safety gate。
16. 同一逻辑复制第二套 authority。
17. 用 process-global `os.environ` 表达单 job 语义。
18. 只修源码不验证 wheel clean install。
19. 只修 A 股而破坏 US/multi-market。
20. P0 以“暂时 defer”结束。

---

# 第一批：AutoShard / OOM / CSE / Spill / Resource / Physical Execution（1–25）

### 1. P0 — Production ROOT AutoShard 实际不可达
`physical_lowerer` 的生产 ROOT contract 可能仍落为 `shardable=False, shard_dimension=None`。必须从完整 plan 推导 `ShardSemantics`，ROOT contract 带上最终严格合成的 shardable、dimension、warmup、state/checkpoint、CS/group/full-panel barriers、merge contract。真实测试必须走 `compile_many → lower_batch_dag → scheduler → shard → merge`。

### 2. P0 — Shard semantics 只看顶层 op
如 `add(cs_rank(x), y)` 顶层是 elementwise，但内部有截面屏障。递归扫描全 plan，建立 `UNSHARDABLE / FULL_CROSS_SECTION / TIME_ONLY / ASSET_ONLY / TIME_OR_ASSET / STATEFUL_*` lattice，最严格者胜出。

### 3. P0 — `split_instrument_universe` 丢 remainder
使用 quotient/remainder 或 `np.array_split`。必测 5003/5007、shard_count>universe、空池、单票；所有 shard union 必须与原 universe 完全一致且无重复。

### 4. P0 — Time shard 使用 `pd.bdate_range` / calendar-day warmup
全部改为 pinned `MarketCalendar / CalendarSnapshot`，按真实交易 session 算 warmup；覆盖 SSE/SZSE、US holiday/DST/early close。

### 5. P0 — OOM shape-signature key 不一致
统一 `TaskAttemptIdentity(original_task_id, attempt_id, shard_generation, shard_id, merge_generation)`；同 shape OOM 后不得原样重试。

### 6. P0 — OOM replan 与旧 late futures 竞态
task/shard/result 全带 attempt/generation；scheduler 只消费 current generation，旧 future 结果 discard。

### 7. P0 — Merge task 资源合同可能自锁死
新增 `MergeResourceModel + StreamingMerge + StreamingMaterializer`。最终输出大于内存 envelope 时不得要求全量驻内存。

### 8. P0 — shard slicing 只覆盖部分接口
统一 `ShardableDataView / BufferRef.slice()`；覆盖 pandas、NumPy、Polars、Arrow、DataAccess ReadHandle、panel/long-table/cache。

### 9. P0 — Fusion calibration 有 `_res` 赋值前使用风险
检查 scheduler 真实顺序；先拿 result 再 calibration；覆盖 fast/cache-hit/fused/error/no-result。

### 10. P0 — L0 CSE 双 authority
`ExpressionCache` 与 `GovernedBufferStore` 只能留一个 authoritative L0 owner。

### 11. P0 — LRU 可驱逐 live shared CSE
refcount>0 不可直接物理驱逐；entry 状态至少含 `PINNED/RECLAIMABLE/SPILLED/RECOMPUTABLE/EVICTED`。

### 12. P0 — Spill reload 绕过内存预算
reload 前申请 memory lease；不够则 streaming reload / smaller slice / recompute / wait。

### 13. P0 — Spill 生命周期泄文件
execution-scoped spill root；覆盖 success/failure/cancel/timeout/crash scavenger。

### 14. P0 — spill-vs-recompute helper 未进入真实 eviction
真实 eviction 依据 recompute_cost、reload_cost、write_cost、bytes、future consumers、disk/memory pressure。

### 15. P0 — ResourceAutopilot 丢 per-job/sink/DA 实时信号
统一输入 JobLease、writer queue、DA backlog、result queue、spill、DuckDB/Polars inflight、cache reclaimable、PSI、MemAvailable、PSS/RSS、disk pressure。

### 16. P0 — stale resource decision 继续使用旧 `last_decision`
stale 必须 fresh recompute 或 conservative safe envelope。

### 17. P0 — `target_concurrency` 与 CPU token 未独立执行
分别做 task-count gate 与 CPU-token gate。

### 18. P0 — JobLease 未传播到 FE/DA child execution
JobLease 下真正创建 Compute/Scan/Writer/Cache/Spill child leases；DA 不得再独立 host authority。

### 19. P1 — HostResourceCoordinator release 残留 child lease
释放时从 registry 移除完整 subtree，只留 bounded summary。

### 20. P0 — Physical DAG stages 大量 planning-only
不能 ROOT 最后再由 backend 整树执行。scan/join/shared/native region/transition/stateful/merge/writer 必须真实 stage-by-stage execution。

### 21. P1 — task start time 在 submit 后记录
改为 submit 前 monotonic 记录。

### 22. P1 — FastLinear sliding Gram 缺 periodic exact rebase
按 N 次、condition number、symmetry/residual drift 触发 exact rebase。

### 23. P0 — Writer fatal 未立即传播 scheduler
worker fatal → close queue → fatal token → stop new admission → cancel materialization → release leases → job fail。

### 24. P0 — DataAccess standalone safe memory 仍 opt-in
production 默认绑定 host/cgroup/SLURM/RLIMIT safe memory，仅允许 controlled opt-out。

### 25. P0 — production canonical evidence 严重不完整
最终必须覆盖 `canonical × parameter domain × backend × execution variant × grain/source context`，且 current SHA 可执行。

---

# 第二批：参数域 / PIT / DataAccess / ChangeImpact / ReadHandle / QueryCache / Evidence（26–75）

### 26. P0 — Parameter-domain production mode 未显式绑定 `ExecutionContext`
认证必须用真实 `ctx.run_mode`，不依赖 env/global 猜。

### 27. P0 — 默认参数调用可跳过认证
用 signature binding 将 explicit args + defaults 绑定成完整 `BoundParameterPoint`，每次生产调用均认证。

### 28. P0 — Parameter certification identity 轴不完整
至少绑定 canonical、semantic_version、backend、execution_variant、source_context、parameter_point、dtype、grain、market、numeric semantics。

### 29. P0 — ParameterDomainStore startup/freshness 可绕
production 必须 `load → current HEAD/build hash validate → negative controls → freeze`；empty/stale store 禁止生产。

### 30. P0 — DataAccess TemporalPlan availability 调用签名错误
复核 `_build_temporal_plan()` 与 `compile_available_from_result(knowledge, availability, ...)` 的真实调用；不能靠 except 变 None。

### 31. P0 — TemporalPlan availability 错误被吞成 fail-open
production 抛 typed PIT/TemporalContractError；research 也要 non-authoritative + degradation reason。

### 32. P0 — `prepare_read()` deadline 起点太晚
deadline 从最外层 API entry 起，包括 path/glob/LIST/HEAD/snapshot/schema/contract/query。

### 33. P0 — composed read deadline 同样起晚
共享同一 request deadline context。

### 34. P0 — PreparedRead memory reservation 是 0
必须估 scan/join/aggregate/sort/result/conversion P99 memory。

### 35. P0 — `QueryBudget.max_estimated_memory` 未进入真实 admission
plan-time 拒绝或缩小 shape。

### 36. P0 — Polars governed lazy 先 full collect 后检查预算
改 preflight estimate、streaming/admitted collect、deadline/cancel。

### 37. P0 — Polars `max_elapsed` 只是事后检查
必须能中断 engine execution。

### 38. P0 — composed stream 未消费会泄 reservation/anchor
ReadHandle 自己拥有 cleanup callback。

### 39. P0 — `ReadHandle.close()` 不关闭 underlying stream
必须 source.close + reservation/anchor release + stats finalize。

### 40. P1 — ReadHandle 缺生命周期状态机
`OPEN → CONSUMING → MATERIALIZED / FAILED / CLOSED`；terminal 后拒绝读。

### 41. P0 — native stream terminal consumption 不统一释放 reservation
stream/to_arrow/collect 全走统一 finalize。

### 42. P1 — composed stream stats 可一直为 0
真实 rows/bytes/elapsed/batches/source files/spill/retries 写 terminal stats。

### 43. P0 — `register_anchor_relation` broad exception 触发 temp parquet fallback
只 capability unavailable 可 fallback；DB/schema/pool/permission/PIT/deadline 必须 propagate。

### 44. P0 — `_prepare_dataset_read` 失败被当 empty source paths
区分 `EmptyPhysicalScope` 与 `SourceResolutionError`。

### 45. P0 — ChangeImpact shared DAG node-ID 不稳定
维护 object-id→canonical-node-id。

### 46. P0 — ChangeImpact 只按 field name 匹配
改 dataset+table+field+market+grain+revision+provider+source scope。

### 47. P0 — ChangeImpact 使用 `BDay`
改 pinned exchange calendar。

### 48. P0 — ChangeImpact resolver 异常后用 name heuristic
production resolver error → UNKNOWN/full recompute，不能猜短 impact。

### 49. P0 — HistoryTransform unknown kind 可能返回 0
unknown → UNKNOWN_HISTORY / FULL_HISTORY。

### 50. P0 — FE/DA 存在多个 run-mode authority
收敛成 request-scoped `RuntimeModeIdentity`。

### 51. P0 — QueryBudget production floor 与 strict semantics 不一致
`DATA_ACCESS_RUN_MODE=production` 单独就必须开启 production floor。

### 52. P0 — SourceSnapshot fidelity 不是一等公民
定义 `PUBLISHER_MANIFEST/REMOTE_VERSION_ID/CONTENT_HASH/LOCAL_STAT/FALLBACK/UNKNOWN`；authoritative production 拒绝低 fidelity。

### 53. P0 — ScanCost error 变 `cost=None` 后继续
unknown cost 用 `COST_UNKNOWN_CONSERVATIVE`。

### 54. P0 — snapshot pin 枚举整个 dataset
只 pin 实际 time range/instruments/partition/provider scope。

### 55. P1 — manifest/token 获取失败缺 typed degradation
explain 能区分 absent/lookup-failed/permission/deadline。

### 56. P0 — resolution cache identity 太窄
加入 namespace、contract digest、source root/profile、mutation generation，或 session 冻结 `PhysicalResolutionContext`。

### 57. P0 — expensive source discovery 在 governor admission 前发生
两阶段 `ResolutionLease → ExecutionLease`。

### 58. P0 — remote concurrency 未覆盖 LIST/HEAD/snapshot discovery
全部远端 metadata 调用走中央 remote lease。

### 59. P1 — production Startup Gate 可绕
生产 store 必须带 `StartupCertificate`。

### 60. P1 — startup calendar smoke 用 `date.today()`
改验证 calendar coverage/digest/authoritativeness/version/requested-window support。

### 61. P0 — SQL production-safe 只看 canonical compile capability
必须并入 parameter-domain backend-call certificate。

### 62. P1 — Pandas backend status 先信 catalog flag
live evidence validation 才是最终 truth。

### 63. P1 — evidence validity `lru_cache` 无 version key
绑定 HEAD/build manifest/TCB/artifact hash。

### 64. P1 — SQL emitter capability cache 无界
改 byte/entry bounded LRU。

### 65. P1 — continuous params 导致 emitter cache 爆炸
缓存 compiled template，不缓存每个 literal point。

### 66. P1 — DuckDB inflight telemetry 读 private semaphore `_value`
用自有 atomic counter/lease registry。

### 67. P0 — DataAccess DerivedFieldCompiler 未真正实现
lower derived fields 到 QueryGraph。

### 68. P1 — DataRequest transforms/field_params/frequency 是接口但执行拒绝
真正 lower 到 PhysicalPlan，或从 public API 移除。

### 69. P0 — automated mining ambiguous field 在 research 可拿 first match
机器 API 所有 mode 均 hard-fail ambiguity。

### 70. P1 — explicit factor matrix consistency 在 research 不能放宽
用户显式要求 same snapshot/version/universe，无法证明就拒。

### 71. P0 — DataAccess QueryResultCache byte cap 可为 None
绑定 host cache budget。

### 72. P0 — QueryCache shrink/grow 未接 ResourceController
接统一 `CostAwareCacheBudgetAllocator`。

### 73. P1 — QueryCache hit 未充分重检当前 result budget
重新检查 rows/nbytes/representation。

### 74. P1 — QueryCache 是另一个独立 memory owner
中央 controller 必须统一 FE L0/panel/DA cache/DuckDB/Polars/writer/spill memory inventory。

### 75. P1 — runtime evidence 依赖 live `.git`
构建时生成 `BuildManifest/SCMManifest/source-tree digest`；runtime 不依赖 `.git`。



---

# 第三批：Config / Packaging / HTTP / Identity / Job Lifecycle / Unit / Security / Mining（76–150）

### 76. P0 — `canonical_config_hash()` 直接 `NameError`
`runtime/config.py` 若调用 `json.dumps()` 却未 import `json`，真实调用直接报错。补 import，并增加真正执行 `canonical_config_hash()` 的测试；只 import module 不算。

### 77. P0 — profile merge 可绕过 config schema-version gate
不能先 validate raw config 再 merge profile。必须 `load raw → load profile → merge final effective payload → validate final schema/version/keys/types → bind config`。

### 78. P0 — config schema-version 合同不严格
明确处理 missing、0、negative、bool、older、current、future。bool/negative/future reject；older 走 migration chain；missing 走明确 legacy policy，不能自动当 current。

### 79. P1 — 存在允许但不消费的 config 字段
类似 `data_access`、`label` 若 loader 不消费，就不得静默接受。要么真正执行，要么从 schema 移除并 fail-fast。

### 80. P0 — wheel package discovery 漏 `market*` / `security*`
检查 `pyproject.toml` setuptools discovery。必须构建 wheel 并在 clean venv 验证 `api/backend/fields/runtime/service/storage/market/security` 全可 import，且最小 compile/run/DataAccess-backed run 可执行。

### 81. P0 — `security/__init__.py` import 风格与真实 package layout 冲突
若 distribution 采用 top-level `security`，不能 `from factor_engine.security.access import ...`，应统一相对 import，并全仓审计 `factor_engine.xxx` 与实际 wheel layout。

### 82. P1 — lineage package version 采集方式可能错误
统一使用 `importlib.metadata.version("factor-engine")` 或 BuildManifest，不能依赖不存在的 import package name。

### 83. P0 SECURITY — 普通 password/token 字符串没有真正 redact
`{"password": "abc123"}` 必须 hash 前变 `<redacted>`。URI credential 只保留 scheme/host/port/db/path/non-secret query。必测 plain password/token、DSN、URL query token、credential rotation identity stable、host/db change identity changes。

### 84. P0 SECURITY — production config-path source-policy validator 可能 fail-open
任何 security/source-policy gate 中的 `except Exception: return/pass` 都不允许。修正确 import path；production validator 初始化/导入失败必须 hard-fail。

### 85. P0 — HTTP `validate_spec()` production 校验没传 market
必须使用真实 canonical market 调 `validate_production_dsl(formula, market=...)`。

### 86. P0 — `validate_production_fastpath_dsl()` 没有 market 参数
API 增加 market，并贯穿 fastpath capability/evidence。

### 87. P1 — `validate_us_dsl()` 名称与实际能力不符
要么改 syntax-only 名称，要么真正绑定 US MarketContext + US field/provider/operator contracts。

### 88. P0 — HTTP inline production build DataSource 未传 `DataSourceBuildContext`
从 validated execution context 注入 run_mode、market、calendar、timezone、PIT、mining gate、snapshot policy、coverage policy。

### 89. P0 — HTTP 验证的 market/calendar/decision-time 没完整进入 actual execution
构建 immutable `ExecutionSemanticContext`，validation/compile/source/runtime/PIT/cache/materialization/lineage 全部消费同一对象。

### 90. P0 — ValidatedRequest digest 没绑定 backend
至少绑定 requested backend、resolved physical backend policy、backend capability generation。

### 91. P0 — ValidatedRequest digest 没绑定 resolved source contract
不能只绑 profile ID；应绑定 source profile version、source contract hash、dataset contract、snapshot policy、provider identity。

### 92. P1 — HTTP inline factor name 被丢弃
确保 `ComputeRequest.name` 进入 execution payload 与 Factor，不能全部退化为 `inline_factor`。

### 93. P1 — `timeout_seconds` 是死接口
若 `_submit_job()` 使用，就必须在 typed request model 定义与验证；否则删除。

### 94. P1 — service 模块 import 即启动 worker
把 JobStore/ThreadPool/Queue.start 等移到 FastAPI lifespan/startup。import module 不应产生线程/DB/目录副作用。

### 95. P1 — FEATURE_POLICY / SOURCE_POLICY import 时冻结
每个 job 绑定 immutable `policy_id/policy_version/policy_digest`，支持受控 reload。

### 96. P0 — cancel/deadline 未进入真正 FactorEngine execution
增加 request-scoped `CancellationToken + Deadline`，贯穿 scheduler、DataAccess、DuckDB、Polars、Numba、writer、materializer。

### 97. P1 — `drain()` 实际立即 cancel running jobs
正确 graceful drain：stop admission → 允许 queued/running 自然完成 → grace timeout → cancel stragglers → force interrupt。

### 98. P1 — queue `get()` 后缺 `task_done()`
worker finally 统一调用。

### 99. P1 — drain 中状态变化未立即 durable update
CANCELLING/INTERRUPTED 必须写 durable authority。

### 100. P1 — RUNNING + no heartbeat 被永久跳过
超过 grace 后，missing heartbeat 应视为 unhealthy。

### 101. P0 — heartbeat 标 INTERRUPTED 但 worker 仍继续计算
状态 fencing 必须触发真实 cancellation；旧 generation result 禁止 publish/materialize。

### 102. P1 — resource rejection durable update 错误被吞
不能 `except Exception: pass`；至少 fallback WAL/emergency journal/fatal signal。

### 103. P1 — current job manifest 缺 checksum 仍接受
current schema checksum mandatory；legacy 无 checksum 走 migration。

### 104. P1 — manifest 缺 schema_version 被当 current
必须明确 legacy version/migration。

### 105. P0 — CatalogJsonField 不拒绝 future schema
`schema_version > supported` 直接 hard-fail。

### 106. P0 — Catalog migration checksum 顺序错误
正确顺序：parse envelope → verify OLD value/OLD checksum → migrate → validate new schema → compute new checksum。

### 107. P1 — current Catalog envelope checksum optional
生产 current schema 必须 mandatory。

### 108. P1 — Catalog JSON 未先强制 mapping/object
scalar/list 不可当 envelope。

### 109. P0 — `scoped_operator_contract_hash()` 未绑定完整 implementation contract
必须统一包含 canonical、semantic_version、policy hash、ParamSpec/signature hash、implementation hash、backend implementation hashes。

### 110. P0 — FactorSemanticIdentity 与 PlanHash operator contract 严格程度不一致
抽出唯一 `OperatorSemanticContractDigest`，factor identity、plan cache、CSE、evidence、checkpoint、materialization 都投影自它。

### 111. P0 — partition fingerprint 用 `astype(str)` 丢 dtype
改 typed hash：schema/dtypes、index identity、null mask、canonical Arrow/native bytes。

### 112. P1 — partition fingerprint 对物理 row order 敏感
若语义 key 是 `(time,instrument)`，先 canonical sort 再 hash。

### 113. P1 — Identity Mapping mixed key 类型存在排序/碰撞风险
identity payload mapping 只允许 string key；非字符串 key fail-closed。

### 114. P1 — `ast_hash` 同时代表多个 identity layer
禁止裸 hash string；使用 `TypedHash(kind=SOURCE_EXPRESSION|TYPED_IR|OPTIMIZED_PLAN|EXECUTION_SEMANTIC, value=...)`。

### 115. P0 — NumPy scalar identity 丢 dtype
`np.int32(1)` 与 `np.int64(1)` 不得同 semantic hash；hash dtype + canonical bytes/value。

### 116. P0 — DataFrame semantic hash 转 float 会让大 int64 碰撞
不得 `to_numpy(dtype=float)` 统一数值。按原 dtype canonical bytes hash；必测 `2^53` 与 `2^53+1`、int64/uint64/nullable integer。

### 117. P1 — FieldRegistry.catalog_hash 仍 `default=str`
改 typed JSON；未知对象不可 `str()` 进入身份。

### 118. P1 — CSE column structural key 仍绑定全 Field Catalog
改 dependency-scoped field contract。

### 119. P1 — Persistent Plan Cache namespace 仍绑定全 operator/field catalog
用 global compiler semantics + dependency-scoped contracts，避免无关 operator/field 改动导致全 cache 冷启动。

### 120. P0 CONCURRENCY — bounded per-key cache write lock 可淘汰 active lock
绝不能淘汰 currently-held/refcount>0 lock。实现 refcounted key-lock entry。必测 > `_SAVE_LOCK_MAX` unique keys 时，早期 key 的 active writer 仍保持唯一锁。

### 121. P1 — optimizer/lowering cache invalidation 依赖人工 bump
实际 compiler code/build digest 纳入 semantic identity。

### 122. P1 — `wrap_context()` 覆盖已有 runtime_stats
必须 merge，而不是重建只剩 cache stats。

### 123. P1 — cache release accounting reconciliation 顺序不合理
先 snapshot resident/ref accounting → unregister/account → clear → verify post-clear。

### 124. P1 — `execution_id="session"` 默认值并发不安全
production 必须唯一 execution ID。

### 125. P1 — PlanNode Python equality 与 structural identity 不一致
建议 `eq=False`，禁止业务逻辑依赖 dataclass equality，统一 typed structural identity。

### 126. P1 — IRNode 文档/身份规则与 Plan hash 实现冲突
明确 `SourceExpressionIdentity / TypedIRStructuralIdentity / TypedIRSemanticIdentity / ExecutionSemanticIdentity` 四层。

### 127. P1 — operator manifest 硬编码 `frequency="any"`
从真实 contract 导出 daily/minute/event/financial/minute→daily 等。

### 128. P1 — operator manifest 硬编码 `output_type="series"`
从真实 OutputContract 导出。

### 129. P1 — `build_operator_spec()` 对所有 operator 写 `supports_panel=True`
改真实 capability，避免对 mining overclaim。

### 130. P1 — `dual_backend_target` 主要按 execution_kind 猜
改由 backend evidence/capability 决定。

### 131. P0 — annual flow 已有 SemanticType，但 fiscal grain 缺 annual mapping
支持 `annual_flow → annual`，并审计 annual financial fields/operators。

### 132. P1 — FieldSpec 自动 flow semantics 只覆盖 ytd/balance
完善 annual、quarter、single-period、YTD、TTM、balance；最好拆开 period duration 与 flow semantics。

### 133. P0 — `CNY/share` 被当成与 `CNY` 同维度
拆分 `currency/currency_per_share/share_count/share_ratio/ratio/percentage/basis_point`。禁止 `CNY/share → CNY` 直接 normalization。

### 134. P0 — nested DataSource options 使用 Python `bool(value)`
统一 typed binder。`"false"` 不能因 `bool("false")==True` 改变 PIT/security 行为。

### 135. P1 — Composite production/options 也需 typed coercion
所有 nested source config 同一 schema。

### 136. P1 — FactorExecutionScopeHint 声称完整但字段少于 runtime identity
要么明确只是 hint，要么由完整 ExecutionSemanticContext 投影；不要继续扩第二套半完整 identity。

### 137. P1 — MarketContext 默认 annualization 仍固定 252
production 默认从 pinned calendar/session/frequency 解析 annualization。

### 138. P1 — MarketContext.extra 不参与 identity
限制为 diagnostic-only，semantic keys 禁止放 extra。

### 139. P1 — 独立 SemanticIdentityDigest 覆盖维度不足
退役不完整 semantic identity，或改名为明确 partial projection。

### 140. P0 — `pandas_modin` backend 修改 process-global env
禁止 job path 修改 `os.environ`；backend choice 必须 execution-context scoped。

### 141. P1 — `clickhouse_sql` 最终构建 DuckDBPushdownBackend 的语义需显式
execution certificate 记录 requested backend、resolved engine、dialect、datasource、actual physical backend。

### 142. P1 — ResolvedRunKwargs 有 calendar_id，但 `to_run_kwargs()` 不传
确认 calendar authority，禁止 batch grouping 用 calendar 而 actual run 不用。

### 143. P0 SECURITY — DeclassificationApproval `approved=True` 时 reviewer/policy 可为空
生产降密必须有 reviewer、policy_version、reason/ticket、timestamp、approval identity。

### 144. P1 SECURITY — empty source access tags 被当敏感度 0
区分 KNOWN_PUBLIC 与 UNKNOWN_CLASSIFICATION；production unknown fail-closed。

### 145. P2 SECURITY — FactorId 只 NFC 不能解决 homoglyph
机器 ID 建议 production 收紧 `[A-Za-z0-9_.:-]+`，Unicode 放 display_name。

### 146. P1 — `fin_mad` 与 `fin_mean_abs_deviation` 两个 canonical 指同 kernel
对 mining 是重复搜索空间。保留一个 canonical，另一个 alias/deprecation。

### 147. P1 — operator alias/canonical rename 缺版本化 migration table
至少记录 old canonical、new canonical、effective DSL version、semantic-equivalent、migration rule、deprecated since/remove-after。

### 148. P1 — universe request validator 使用 import-time DEFAULT_SIZE_BUDGET
每 request admission 生成 immutable `RequestBudgetSnapshot`，validation/cost/admission 共用。

### 149. P1 — ValidatedFactorRequest catalog_generations 主要只绑定 operator generation
一起绑定 operator、field、market、calendar、source profile、backend evidence、parameter-domain、compiler/build generation。

### 150. P1 SECURITY — result preview artifact 未继承 derived access classification
result preview、logs、artifact、debug dump、temp parquet、spill、checkpoint、factor lake、factor matrix 都必须继承最高 source access classification。


---

# 四、不要逐项打补丁：必须收敛成 5 个单一事实源

## 4.1 Execution Semantic Truth

建立 immutable `ExecutionSemanticContext`，至少包含：

```text
run_mode
market
calendar_id
calendar_version
timezone
decision_timestamp
decision_time_policy
frequency
universe_id
universe_membership_hash
price_basis
pit_policy
source_scope
source_snapshot
source_contract
field_contract
operator-contract generation
backend request
resolved physical backend
numeric semantics
mathematical semantics
compiler/lowering/optimizer identity
storage precision
access classification
request budget
deadline
cancellation token
```

以下全部从它投影，不再独立拼：

```text
FactorExecutionScope
CSE scope
cache namespace
checkpoint identity
factor_version
lineage
ValidatedRequest digest
materialization generation
evidence identity
```

## 4.2 Operator Semantic Truth

建立唯一 `OperatorSemanticContract`：

```text
canonical
aliases
semantic_version
kernel implementation hash
per-backend implementation hashes
ParamSpec/defaults
relational constraints
parameter roles
history contract
PIT/availability contract
grain/frequency contract
unit contract
statefulness
sharding contract
broadcast contract
output contract
backend capability
evidence generation
```

plan hash、factor identity、manifest、operator spec、production gate 不得各自维护不一致字段集。

## 4.3 Data Source Truth

建立 `ResolvedSourceContract`：

```text
dataset
provider
physical source
market
field mapping
time/instrument keys
knowledge time
revision/vintage
availability
PIT
snapshot fidelity
snapshot token
calendar
requested physical scope
access classification
source profile version
```

## 4.4 Resource Truth

唯一 `HostResourceCoordinator`。service job、FE task、DA resolution/scan、DuckDB、Polars、Numba、cache、spill、writer、merge 全挂同一 Lease tree。

## 4.5 Build / Evidence Truth

构建时生成 immutable BuildManifest：

```text
package version
source commit
source-tree digest
operator source digest
compiler digest
dependency versions
Python ABI
platform
build timestamp
```

runtime evidence 不再依赖 `.git`。

---

# 五、推荐整改顺序

## Phase A — 直接 correctness / security / packaging P0

优先处理：

```text
#76
#80 #81
#83 #84
#85 #86
#88 #89
#90 #91
#105 #106
#109 #110
#115 #116
#120
#131
#133 #134
#143
```

## Phase B — 参数域、PIT、source/time truth

处理：

```text
#26–33
#46–61
#85–91
#131–139
#149
```

## Phase C — AutoShard / OOM / streaming merge

处理：

```text
#1–9
#20
```

## Phase D — Resource / CSE / Spill / Cache

处理：

```text
#10–19
#34–44
#64–75
#118–124
```

## Phase E — Service lifecycle / cancellation / durable state

处理：

```text
#93–104
#140–150 中 service/security 相关
```

## Phase F — Full Physical Execution

把 planning-only QueryGraph/PhysicalStageGraph 改成真实 stage runtime，不允许 root 后端整树兜底掩盖 planner。

## Phase G — Operator/Field contract 机器可用化

重点：

```text
#109–110
#117
#125–147
```

保证 mining algorithm 看到的频率、输出、参数、backend、grain、unit、history 都是真实 contract。

## Phase H — Full Canonical Certification

生成 current HEAD / current BuildManifest / current operator-field-source-compiler digests / current test / benchmark / evidence。

---

# 六、必须新增或补齐的测试矩阵

## 6.1 Packaging / clean install

必须真正执行：

```bash
python -m build
python -m venv /tmp/fe_clean
# 激活后安装 dist wheel
pip install dist/*.whl
python -c "import api, backend, fields, runtime, service, storage, market, security"
```

随后在 clean env：

1. 最小 DSL compile
2. 最小 factor run
3. DataAccess-backed run
4. production validator
5. service import（确认不自动启动 worker）

## 6.2 Multi-market HTTP

覆盖：

```text
US formula validated as US
A-share formula validated as A-share
US field 不得走 A registry
A field 不得走 US registry
market typo hard fail
validation market == execution market
calendar/decision-time 一致
```

## 6.3 Config

覆盖：

```text
profile future schema
profile old schema
bool schema version
negative version
missing version
unknown effective key
nested bool "false"
nested bool "true"
unused config field
```

## 6.4 AutoShard

覆盖：

```text
5003 / 5007 instruments
nested cs op under elementwise root
nested group op
stateful
PCA/full-panel
minute→daily
exact SSE/SZSE warmup
US holiday/DST/early-close
```

## 6.5 OOM/replan

构造：

```text
multiple shards running
one shard OOM
late old future arrives
new generation already running
```

旧 generation 结果绝不能进入 merge。

## 6.6 Streaming merge/materialize

最终输出超过 safe memory envelope，仍应：

```text
stream merge
→ stream DQ
→ stream writer
```

成功，不要求完整结果驻内存。

## 6.7 CSE / Spill

覆盖：

```text
same subtree across roots
same subtree twice within root
live refcount
eviction pressure
spill
reload
recompute
cancel
failure
```

## 6.8 Persistent cache active-lock

制造超过 `_SAVE_LOCK_MAX` unique keys；保持早期 key writer 长时间持锁；触发 registry 淘汰后第二个同 key writer 不得拿到新锁并行写。

## 6.9 Typed hash negative controls

必须：

```text
np.int32(1) != np.int64(1)
int64(2^53) != int64(2^53+1)
int 1 != str "1"
same values different dtype distinguish
same semantic partition reordered → canonical fingerprint policy正确
```

## 6.10 Unit

至少：

```text
CNY -> CNY
CNY_10K -> CNY
CNY/share -> CNY : reject
share -> share_10K
percent -> ratio
basis_point -> ratio
```

并做 mixed-unit operator compile-time validation。

## 6.11 Catalog migration

覆盖：

```text
legacy raw JSON
old envelope + valid checksum
old envelope migration changes payload
future schema
current schema missing checksum
corrupt checksum
non-object JSON
```

## 6.12 Cancellation / deadline

真实长任务：

```text
RUNNING
→ cancel
→ scheduler stop admission
→ DA read stop
→ writer stop
→ leases release
→ no publish
```

deadline 也做相同验证。不能只断言 job.status。

## 6.13 Access classification

```text
public → public
premium → derived premium
restricted → derived restricted
unknown → production reject
approved=True but no reviewer → reject
preview/log/spill/checkpoint inherit classification
```

## 6.14 Parameter domain

每个 production call 对：

```text
explicit params
default params
active_when
relational constraint
backend variant
dtype/grain
```

形成完整 BoundParameterPoint。

## 6.15 DataAccess deadline

让 LIST / HEAD / snapshot / schema / query 各自超时，证明同一个 absolute deadline 覆盖全链。

## 6.16 ReadHandle lifecycle

覆盖：

```text
create but never consume
partial consume then close
stream complete
to_arrow
exception
cancel
GC fallback
```

资源全部释放。

---

# 七、静态审计 Hard Checks

整改后增加 static audit，扫描但逐项人工分类：

```text
except Exception: pass
except Exception: return
except Exception: return None
bool(config_value)
pd.BDay
pd.bdate_range
os.environ[...] in backend/job path
default=str in semantic hash
astype(str) in identity/checkpoint hash
production_safe=True hardcode
shardable=True hardcode outside authoritative contract
estimated_memory=0
factor_engine.xxx imports incompatible with wheel layout
whole field/operator catalog hash in dependency-scoped identity
```

---

# 八、CI Hard Gates

## 8.1 Current-SHA binding

所有 evidence：

```text
evidence.head_sha == actual HEAD/build SHA
```

不一致直接 fail。

## 8.2 Clean-install gate

CI 必须 build wheel + clean environment test，不能只依赖 repo PYTHONPATH。

## 8.3 Production-path gate

真实：

```text
DSL/Factor
→ Analyzer
→ Lowerer
→ batch DAG
→ physical planner
→ scheduler
→ backend
→ DataAccess
→ result
```

不得 test-only fake plan。

## 8.4 Parameter gate

默认参数也认证。

## 8.5 PIT gate

negative controls：

```text
future availability
unknown availability
revision/vintage
calendar
decision time
current-snapshot-only
```

必须 fail。

## 8.6 Resource gate

多 job 并发验证：

```text
no OOM
不超过 host safe envelope
pressure 自动收缩
pressure 解除 slow recovery
background lane 降级
writer backpressure 生效
```

## 8.7 Cache/spill gate

验证：

```text
no live-entry corruption
no spill leak
no reload over budget
no active-lock eviction
no stale cache after semantic changes
```

## 8.8 Service gate

验证：

```text
import 不启动 worker
lifespan 正常
cancel 真停 compute
deadline 真停 compute
durable terminal state
source policy fail closed
result preview permission correct
```

---

# 九、性能验收

不要只测单因子。

规模：

```text
1
10
100
500
1000
5000
```

类型：

```text
daily PV
cross-sectional
rolling
fundamental PIT
minute→daily
mixed multi-source
expensive state/model
```

至少记录：

```text
TTFC
TTDC
peak RSS/PSS
read bytes
scan count
source-open count
DuckDB region ratio
Polars region ratio
Pandas fallback ratio
Numba time
cache hit
spill bytes
writer throughput
CPU utilization
OOM/replan count
```

比较：

```text
cold / warm
single job
2 jobs
4 jobs
contention
memory pressure
```

---

# 十、最终 Definition of Done

## 10.1 Correctness

- production ROOT AutoShard 真可达
- 无 shard remainder loss
- nested semantic barrier 正确
- exact market calendar
- no old-attempt contamination
- large output streaming merge
- default params certified
- PIT availability 真编译
- ChangeImpact full source identity
- typed unit semantics
- no dtype/int hash collision
- no CNY/share dimension conflation

## 10.2 Security

- no source-policy fail-open
- secrets truly redacted
- derived access classification inherited
- unknown classification fail-closed
- declassification 有真实 approval metadata
- preview/log/spill/checkpoint 同权限

## 10.3 Runtime

- real cancellation/deadline
- graceful drain
- durable job state
- no import-time worker side effects
- one resource authority
- task-count/CPU-token 两道 gate
- writer fatal 即时停止 scheduler

## 10.4 Cache / Memory

- one L0 CSE authority
- live refs safe
- spill lifecycle safe
- reload admitted
- no active key-lock eviction
- QueryCache governed
- central cache budget

## 10.5 DataAccess

- deadline from API entry
- real non-zero memory estimate
- discovery governed
- snapshot exact scope
- fidelity explicit
- ReadHandle always closes
- derived fields/transforms 真执行或删除

## 10.6 Packaging

- wheel 含全部运行包
- clean install passes
- runtime version 来自 BuildManifest/package metadata
- no live-git requirement

## 10.7 Evidence

每个 production canonical 至少有：

```text
canonical identity
parameter-domain evidence
backend-call evidence
grain/source context
PIT evidence where applicable
numeric parity
prefix causality
shape/index contract
current BuildManifest
current HEAD
```

不得因为 metadata/whitelist 就 READY。

---

# 十一、最终交付物

coding AI 完成后必须输出：

1. 完整代码修改
2. 变更文件清单
3. **260 项逐项状态表**
4. 当前 HEAD SHA
5. BuildManifest digest
6. 完整 test commands + pass counts
7. wheel clean-install test
8. production hard gates
9. benchmark
10. 仍存在风险
11. current-SHA evidence 路径
12. 删除所有临时 workflow/fake fixture/stale evidence

逐项状态只允许：

```text
FIXED
PARTIAL
NOT_APPLICABLE_WITH_PROOF
BLOCKED_BY_EXTERNAL_DEPENDENCY
```

P0 不得无证明地 PARTIAL。

格式示例：

```text
#1 FIXED
code: ...
tests: ...
evidence: ...

#2 FIXED
...
```

---

# 十二、执行指令

不要只输出整改建议，直接开始改。

```text
1. read latest HEAD
2. map 260 findings to actual current files/functions
3. already-fixed item 也必须给真实 proof
4. implement all still-open items
5. add/repair tests
6. continuous targeted tests
7. full factor_engine + dataaccess regression
8. build wheel + clean-install test
9. production-path integration
10. multi-market/PIT/security/resource negative controls
11. generate current-SHA evidence
12. run benchmark
13. final re-audit all 260
14. only then stop
```

如果最新 HEAD 已修好某项，不要重复重构；以真实代码、测试和 current-SHA evidence 标 FIXED。

如果整改过程中发现新问题，追加为：

```text
NEW-261+
```

不要因为本文止于 260 就停止审计。

---

# 十三、最高优先级复核列表

最先检查：

```text
#83  secret plaintext redaction
#84  production source-policy fail-open
#85/#86 HTTP market validation
#88/#89 inline source/execution context
#90/#91 validated request fencing
#80/#81 wheel packaging
#120 persistent-cache active-lock eviction
#133 CNY/share dimensional bug
#109/#110 operator identity split-brain
#115/#116 semantic hash collision
#105/#106 catalog schema/checksum migration
#134 nested source bool coercion
#96/#101 real cancellation/fencing
#131 annual-flow grain
#76 canonical_config_hash direct crash
```

这些优先级高于新功能开发。

---

# 十四、最终原则

最终目标不是“测试变绿”，而是：

> **同一条因子，在明确的市场、日历、股票池、PIT、数据快照、参数、算子实现、backend 数值语义、资源环境和物化精度下，得到唯一、可复现、可证明、可扩展、不会未来函数、不会内存失控、不会缓存污染、不会跨市场串义、不会因为部署方式改变而失效的生产结果。**

任何不能满足这句话的实现，都继续整改，不要停止。


---

# 十五、补充深度审计：最新 HEAD 新增整改项（151–220）

> **重要：本节新增审计基线已经更新为**
>
> `main@5f44df633ee4d1763a5d2d1cdeb55eb579f1e4e5`
>
> 前文 1–150 项仍然必须在这个最新 HEAD 上重新逐项核验；本节只追加此前没有系统列出的新问题。若 coding AI 开始执行时 HEAD 再次变化，仍以执行时最新 HEAD 为准。
>
> 本节每条都要求：**定位真实代码 → 修改真实生产路径 → 单测 → 集成测试 → negative control → current-SHA evidence**。

## A. Operator Registry 初始化、线程安全与 production/research 表面隔离

### 151. P0 — Registry bootstrap 存在线程竞态：第二线程可直接拿到半初始化 Registry

#### 当前风险

`cleaned_operators.load_all()` 当前使用全局：

```python
_LOADED = False
_INITIALIZING = False
```

并存在：

```python
if _INITIALIZING:
    return
```

这不是并发 barrier。

典型竞态：

```text
Thread A
→ _INITIALIZING=True
→ 已加载 40% operator modules

Thread B
→ 调 ensure_cleaned_loaded()
→ 看到 _INITIALIZING=True
→ 直接 return
→ 开始 OperatorRegistry.get()/resolve/build allowlist

此时 B 看到的是半加载 registry
```

这在 service、多 job、测试并行、import 并发场景都有可能发生。

#### 必须怎么改

实现单一 bootstrap authority：

```python
class RegistryBootstrap:
    state: NEW | INITIALIZING | READY | FAILED
    owner_thread_id: int | None
    error: BaseException | None
    lock: threading.RLock
    condition: threading.Condition
```

规则：

```text
NEW:
  一个线程获得 owner，进入 INITIALIZING

INITIALIZING:
  owner 线程的合法 reentrant call 可以直接返回内部 staging handle
  其他线程必须 WAIT，不得 return

READY:
  直接返回 immutable RegistrySnapshot

FAILED:
  所有线程收到同一个 initialization error
  不得自动在半初始化状态上 retry
```

最终：

```python
snapshot = bootstrap.ensure_ready(surface=...)
```

不要再让 `_LOADED / _INITIALIZING / backend._CLEANED_LOADED` 成为三套状态。

#### 必测

- 32/64 threads 同时第一次调用。
- 在 module import 中人工 sleep。
- 所有调用要么等待 READY，要么拿到同一个 FAILED。
- 任意线程绝不能观察到 canonical_count 的中间值。

---

### 152. P0 — `include_research` 被“第一次调用者”永久决定，production/research 会互相污染

#### 当前风险

Registry 最终只有一个：

```text
_LOADED=True
Registry=frozen
```

但入口允许：

```python
load_all(include_research=True/False)
```

这意味着：

```text
research 先启动
→ registry 包含 research modules
→ production 后启动
→ 因 _LOADED 直接 return
→ production 实际继承 research-loaded global registry
```

反过来：

```text
production 先 include_research=False
→ registry freeze
→ research 后续无法加载 research family
```

#### 必须怎么改

推荐不要创建两套 implementation registry。

采用：

```text
FullImmutableImplementationRegistry
+
OperatorSurfaceSnapshot
    ├─ PRODUCTION
    ├─ RESEARCH
    ├─ COMPAT
    └─ INTERNAL
```

即：

1. implementation registry 可一次完整加载；
2. production 是否可见/可执行由 immutable surface/capability/evidence 决定；
3. **不能靠“不 import research 模块”表达 production safety**；
4. first caller 不再改变全局行为。

例如：

```python
registry = OperatorRegistrySnapshot(...)
surface = registry.surface(RuntimeSurface.PRODUCTION)
surface.require_allowed(canonical)
```

#### 必测

四种顺序：

```text
production → research
research → production
production ∥ research
research ∥ production
```

production exposed canonical set / digest 必须完全一致。

---

### 153. P0 — Registry 初始化失败后的 retry 不是事务性的，可能在“残缺 Registry”上继续

#### 当前风险

Python import 具有副作用。

若：

```text
module 1..100 已注册
module 101 import error
```

则：

```text
_INITIALIZING reset False
_LOADED 仍 False
```

但：

- 前 100 个 module 已存在 `sys.modules`；
- Registry 已被前 100 个 module 修改。

再次 `load_all()` 时并不是“干净重试”。

#### 必须怎么改

使用 transactional staging registry：

```text
build fresh RegistryBuilder
→ modules 注册到 staging
→ dedupe/governance/evidence/final audit
→ compute digest
→ freeze
→ atomic publish RegistrySnapshot
```

失败：

```text
discard staging completely
global active snapshot unchanged
```

禁止直接修改 live singleton。

#### 必测

- 在第 N 个 module 注入异常。
- 失败后 active registry 不得有新增 canonical。
- 再次正常初始化，其 digest 必须等于 clean-process 初始化 digest。

---

### 154. P1 — `PRODUCTION_LOAD_MODULES` 的注释语义与实际加载路径不一致

当前代码虽然定义：

```text
PRODUCTION_LOAD_MODULES
```

但初始化循环仍主要从 `_LOAD_MODULES` 出发，只条件排除 research modules，internal kernel modules 的真实加载策略与注释存在差异。

#### 必须怎么改

不要再靠多个 module tuple 隐含治理。

新增：

```python
BootstrapModuleSpec(
    module,
    role=IMPLEMENTATION | INTERNAL_KERNEL | RESEARCH_EXTENSION | GOVERNANCE,
    required=True/False,
)
```

初始化结束后验证：

```text
INTERNAL_KERNEL module 不得产生 user-facing DSL canonical
RESEARCH_EXTENSION 不得自动进入 PRODUCTION surface
required module 不得被 optional skip
```

---

### 155. P0 — `backend.cleaned_bridge.ensure_cleaned_loaded()` 自己又维护 `_CLEANED_LOADED`，并默认加载 research-inclusive Registry

#### 当前风险

当前 bridge：

```python
_CLEANED_LOADED = False

def ensure_cleaned_loaded():
    if _CLEANED_LOADED:
        return
    load_all()
    _CLEANED_LOADED = True
```

问题：

1. 又是一套全局状态；
2. 无锁；
3. `load_all()` 默认 include_research=True；
4. production Analyzer/Backend 也会调用这个入口。

#### 必须怎么改

彻底删除 `_CLEANED_LOADED` authority。

改：

```python
def ensure_operator_registry(ctx_or_surface):
    return REGISTRY_BOOTSTRAP.ensure_ready().surface(ctx.surface)
```

生产代码必须显式知道执行 surface，不得默认 research。

---

## B. Parameter-domain 认证真实生产路径仍存在 fail-open

### 156. P0 — production 下“该 backend 没有任何 certified region”当前仍允许执行

#### 当前风险

cleaned bridge 当前逻辑大意：

```python
if store.operator_has_any_certified_region_by_backend(...):
    assert_parameter_point_certified(...)
else:
    telemetry("coverage_skip")
    # 继续执行 kernel
```

这等价于：

```text
有认证 → 检查
完全没认证 → 反而放行
```

生产语义不成立。

#### 必须怎么改

production：

```python
if parameterless_formally_certified:
    pass
elif not has_certified_region(canonical, backend, variant):
    raise NoCertifiedParameterRegionError(...)
else:
    assert point in certified region
```

允许豁免的 operator 必须有 typed exemption：

```python
ParameterCertificationExemption(
    reason="parameterless",
    build_digest=...,
    evidence_digest=...,
)
```

不能 telemetry skip。

research 才允许：

```text
UNCERTIFIED_RESEARCH execution + degradation record
```

---

### 157. P0 — production 参数认证基础设施异常会被 broad except 吞掉后继续执行

#### 当前风险

如果：

- store load error
- bound-call error
- semantic version resolver error
- identity serialization error

不是特定 `ParameterDomainError`，当前路径可能只记 telemetry 后继续 kernel。

#### 必须怎么改

将认证流程拆成 typed phases：

```text
BindCall
ResolveCertificationIdentity
LoadCertificate
CheckMembership
```

production 任一步：

```text
ERROR → ParameterCertificationInfrastructureError → hard fail
```

research：

```text
ERROR → mark UNCERTIFIED + reason
```

禁止整个认证块外包一个 broad `except Exception`。

---

### 158. P0 — `_bound_scalar_parameters()` bind 失败时退化成“只取显式 kwargs”，会生成不完整认证点

#### 当前风险

默认参数、位置参数、alias-normalized 参数可能丢失。

#### 必须怎么改

执行时只绑定一次：

```python
bound_call: BoundOperatorCall = bind_operator_call(...)
```

然后：

```text
认证使用 bound_call
kernel 调用也使用同一个 bound_call
```

禁止：

```text
认证 bind 一次
失败后猜参数
kernel 再 bind 第二次
```

production 只接受：

```text
bound_call.complete == True
```

---

### 159. P0 — operator semantic version 解析失败时返回空字符串

任何 production certificate key 中：

```text
semantic_version=""
```

都应视为无效身份，不应继续匹配证据。

#### 必须怎么改

semantic version 从统一 `OperatorSemanticContractDigest` 读取。

production：

```text
missing/unknown version → hard fail
```

---

### 160. P0 — 参数认证 dtype 目前只看“第一个合适输入”的 dtype

#### 当前风险

例如：

```text
x=float32
y=float64
mask=bool
group=category
```

可能只得到：

```text
dtype=float32
```

认证证据没有表达完整 dispatch shape。

#### 必须怎么改

定义：

```python
InputDTypeSignature(
    inputs=(
      InputDType(name="x", dtype="float32", nullable=True),
      InputDType(name="y", dtype="float64", nullable=True),
      InputDType(name="mask", dtype="bool", nullable=False),
    )
)
```

认证 key 绑定**有序参数角色 + dtype + nullable + representation**。

---

### 161. P0 — actual execution variant 不能继续硬编码 `"reference"`

真实执行可能是：

```text
pandas reference
Numba
Polars native
Polars bridge
DuckDB SQL
fused SQL
fast recursive kernel
specialized model kernel
```

#### 必须怎么改

由实际 selected implementation 产生：

```python
ExecutionVariantIdentity(
    backend="polars",
    implementation_id="...",
    kernel_variant="native_expr",
    fused=False,
    code_hash="...",
)
```

认证、lineage、runtime event 使用同一个对象。

---

### 162. P1 — runtime backend-route telemetry 使用普通 dict read-modify-write，可能在并发 context 中丢计数

移到 thread-safe：

```text
ExecutionPerfCounters
AtomicCounterMap
```

telemetry 不能修改 semantic context。

---

## C. Panel / Polars 表示转换存在“轴重排被重新贴标签”的正确性风险

### 163. P0 — shape-changing/downsampled result 的轴校验过弱

当前类似只验证：

```text
DatetimeIndex
unique
same columns
len(output) <= len(input)
```

这不能证明 minute→daily 结果正确。

#### 必须怎么改

新增：

```python
GrainTransformCertificate(
    input_grain,
    output_grain,
    calendar_id,
    calendar_version,
    timezone,
    session_id,
    mapping_policy,
)
```

minute→daily 输出必须证明：

1. 每个输出 date 对应一个真实 market session；
2. 输出 date 在 input session coverage 中；
3. 输出 monotonic；
4. 无 future date；
5. 无重复；
6. market calendar/version 一致；
7. instrument axis exact preserved；
8. availability = declared session-close/next-open policy。

---

### 164. P0 — Polars bulk → pandas 时直接把 template.index 贴回结果，可能隐藏 Polars 行重排

#### 当前风险

若 Polars kernel：

```text
输出行顺序被 reverse/sort/shuffle
```

但 row count 没变，转换层如果：

```python
pdf.index = template.index
```

就把错误值重新贴到了“看似正确”的日期。

#### 必须怎么改

axis-preserving Polars result 必须保留：

```text
__fe_time__
```

转换回 pandas 前：

```python
assert result.__fe_time__ == template.index  # exact ordered
```

然后才 drop metadata / set index。

缺时间元数据：

```text
production hard fail
```

---

### 165. P0 — Polars column-loop fallback 与 bulk path 同样存在 row-restamping 风险

bulk/fallback 必须共同走一个：

```python
verify_and_restore_axis(...)
```

不能两套实现。

---

### 166. P0 — Pandas bridge 写回 Polars 时只按位置取 `out[c].to_numpy()`，可隐藏 pandas inner kernel 的 index 重排

#### 必须怎么改

`from_pandas_panel(base, out)` 先：

```text
expected = frame_time_index(base)
assert out.index exact equals expected
```

axis-preserving operator index 不同：

```text
hard fail
```

shape-changing operator 必须走独立 OutputAxisContract，不能复用该函数。

---

### 167. P0 — 将 instrument/value 列名 `str(c)` 化时可能发生非单射碰撞

示例：

```python
columns=[1, "1"]
```

fallback 构造：

```python
{str(c): ...}
```

会覆盖。

#### 必须怎么改

在任何 representation boundary 前：

```python
PhysicalColumnNameMap.validate_injective()
```

逻辑 InstrumentKey 与物理 Polars/Arrow 列名分开。

不能因为 bulk 不安全就 fallback 到一个会碰撞的 column-loop。

---

### 168. P1 — value column 与保留元数据列名冲突会被 name-based `SKIP` 静默当 metadata

例如真实 instrument/value column 恰好叫：

```text
date
timestamp
instrument
symbol
__fe_time__
```

#### 必须怎么改

Panel schema 不再通过“列名猜角色”。

定义：

```text
AxisColumnRole.TIME
AxisColumnRole.INSTRUMENT_ID
AxisColumnRole.VALUE
AxisColumnRole.METADATA
```

物理命名只是 serialization。

---

### 169. P0 — production `PanelIdentity.__eq__` 当前将 UNKNOWN grain/frequency 视为与 known compatible

#### 当前风险

```text
daily panel
vs
unknown panel
```

只要 axes 一样，就可能判相同。

对于 research 可以降级兼容，但 production 不应该。

#### 必须怎么改

`PanelCompatibilityPolicy`：

```text
production:
  time axis PROVEN
  instrument axis PROVEN
  grain PROVEN + equal
  frequency PROVEN + equal
  calendar PROVEN where time semantics matter

research:
  UNKNOWN can be compatible but emits degradation
```

不要让 `__eq__` 自己暗含 mode。

---

### 170. P1 — 通过 Pandas `index.freq` 推断交易频率不可靠

真实 SSE/SZSE trading dates 的 DatetimeIndex 通常：

```text
freq=None
```

因此 daily panel 常被推成 unknown。

#### 必须怎么改

grain/frequency 来自：

```text
DataContract / SourceContract / ExecutionSemanticContext
```

index inference 只能做 diagnostic consistency check。

---

### 171. P1 — MultiIndex `instrument_count` 当前更接近“非时间 level 数量”，不是标的实际 cardinality

若这个数进入：

- memory model
- scheduler cost
- telemetry
- axis certificate

会误导。

#### 必须怎么改

从明确 instrument level 统计 unique instrument count；不能把 level 数当 cardinality。

---

### 172. P1 — axis hash 使用 `repr()` 对任意对象做 canonical serialization，不足以作为生产身份

自定义对象 repr 可能含地址/环境信息。

#### 必须怎么改

production instrument/time labels 只能使用 typed canonical encoder：

```text
str
int
datetime normalized
InstrumentKey
```

未知对象 reject。

---

### 173. P1 — `PanelIdentity.__hash__` 不包含 known grain/frequency，容易被误用成完整 semantic cache key

Python equality/hash 规则虽允许不等对象 hash 相同，但工程层不能把它当完整 semantic digest。

新增：

```text
AxisIdentityDigest
```

显式包含所有 production axis semantics。

---

## D. Analyzer / Typed IR 仍残留 name/category heuristic

### 174. P0 — `Analyzer(production=True, market=None)` 仍会回退 legacy A-share resolver

production compile 不得允许市场未知。

#### 必须怎么改

构造 production Analyzer 必须：

```python
Analyzer(
    production=True,
    market_context=MarketContext(...)
)
```

没有 market：

```text
ProductionMarketContextRequiredError
```

legacy A-share fallback 仅 research/compat。

---

### 175. P0 — Analyzer 仍调用 mode-agnostic `ensure_cleaned_loaded()`

与 #151–155 一起整改。

Analyzer 应依赖已冻结的 `OperatorSurfaceSnapshot`，不能自行触发默认 research bootstrap。

---

### 176. P0 — `has_ts` / `has_cs` 仍大量靠 category、prefix、canonical-name set 猜

#### 当前危险

新 operator 一旦名字/category 不在旧列表：

```text
planner 可能不知道它是 TS/CS/global/stateful
→ warmup/shard/fusion 都可能错
```

#### 必须怎么改

OperatorContract 增加：

```python
AxisEffectContract(
    kind=
      ELEMENTWISE |
      TIME_SERIES |
      CROSS_SECTION |
      GROUP_CROSS_SECTION |
      GLOBAL_PANEL |
      GRAIN_CHANGE |
      RELATION |
      STATEFUL,
)
```

Analyzer：

```text
只读 contract
```

production 禁止 name/category fallback。

静态 gate：

```text
PRODUCTION_AXIS_EFFECT_UNDECLARED == 0
```

---

### 177. P0 — full-history operator contract 加载 ImportError 被吞掉后可能低估 history

任何 full-history contract authority 读取失败都不能：

```python
except ImportError:
    pass
```

production planning hard-fail。

更根本的做法：

```text
history/statefulness 都进入 OperatorSemanticContract
```

---

### 178. P1 — Analyzer 仍维护大量 `_WINDOW_PARAM_NAMES` / `_FIXED_LAGS` / pattern-name sets

这些应成为 compatibility fallback，而不是 production history authority。

#### 必须怎么改

所有 production time/state operator 必须显式声明：

```text
HistoryContract
```

可表达：

```text
rows = window - 1
rows = window - 1 + lag
full_history
report_periods=N
session_slots=N
event_count=N
checkpoint
```

---

### 179. P1 — FieldRef identity 绑定整个 market Field Catalog hash，会因无关字段变化导致大量历史公式失效

#### 必须怎么改

每个 FieldRef 存：

```text
field_id
FieldContractDigest
catalog_schema_version
```

global catalog hash 留给整个 run audit，不作为单字段语义 identity。

---

### 180. P0 — `available_at` 与 `knowledge_time_column` 目前存在“字符串同槽”问题

当前语义容易变成：

```text
available_at = "session_close"
```

与：

```text
available_at = "PubDate"
```

混在同一 string 字段。

#### 必须怎么改

类型拆分：

```python
AvailabilityExpr
├─ SessionOpen
├─ SessionClose
├─ NextSessionOpen
├─ TimestampColumn(name="PubDate")
├─ MaxAvailability(...)
└─ UnknownAvailability
```

`knowledge_time_column` 永远是列引用，不是 policy label。

---

### 181. P1 — `PeriodDuration` 已在 IR type 层存在，但 FieldSpec/Schema 不是完整一等字段

US annual 与 quarterly 都可能属于 single-period flow；仅靠 flow_semantics 不够。

#### 必须怎么改

贯穿：

```text
FieldSpec.period_duration
Schema.period_duration
SemanticTypeBundle.period_duration
FactorSemanticIdentity
Operator input contract
```

可选：

```text
QUARTER
HALF_YEAR
ANNUAL
TTM
YTD
IRREGULAR
UNKNOWN
```

---

### 182. P1 — 当前五维 SemanticLattice 不足以覆盖所有真正会改变经济含义的维度

目前核心 lattice 主要覆盖：

```text
domain
frequency
source vintage
universe
semantic kind
```

但实际还存在：

```text
unit
price basis
period duration
calendar
timezone
missing policy
availability
access classification
```

#### 必须怎么改

推荐升级为：

```python
SemanticTypeBundle(
    value_semantics,
    unit_expr,
    axis_type,
    temporal_type,
    knowledge_type,
    source_vintage,
    universe_type,
    missing_policy,
)
```

Analyzer/operator legality 只消费这一对象，避免散落 attrs。

---

### 183. P1 — `SemanticIdentityDigest` 当前仍有 `default=str`

生产 identity 禁止 unsupported object 通过 repr/string 化。

统一 typed canonical serializer。

---

## E. Unit system 除了 CNY/share 维度错误之外还有进一步问题

### 184. P0 — 未知单位拼写在 `source == target` 时会被直接视为合法

例如：

```text
CYN
```

若 source/canonical 都是 `"CYN"`，当前 normalization 可能直接 multiplier=1。

#### 必须怎么改

`canonical_unit()` 生产模式必须返回：

```text
KnownUnitId
```

未知：

```text
UnknownUnitError
```

research 若要保留未知单位，必须显式：

```text
OpaqueUnit("...")
```

OpaqueUnit 不能自动参与单位代数。

---

### 185. P1 — `UNIT_CNY_PER_SHARE` / `UNIT_SHARE_RATIO` 公共导出不完整

统一 `__all__`、catalog export、docs/API schema，并增加：

```text
所有注册 UnitId 都必须 public-export 或显式 internal
```

的静态测试。

---

### 186. P0 — Output unit 不能继续只靠静态字符串，缺完整单位代数

例如：

```text
price / price → ratio
currency/share × share → currency
revenue / assets → ratio
log(x) → 要求 dimensionless
sqrt(x) → unit exponent 1/2
power(x,2) → unit²
```

#### 必须怎么改

实现：

```python
UnitExpr
```

支持 base dimension + exponent。

OperatorContract 声明：

```text
ADD: unit(x)==unit(y), output=unit(x)
DIV: output=unit(x)/unit(y)
MUL: output=unit(x)*unit(y)
LOG: require dimensionless
RANK: output=ratio/dimensionless
```

自动挖掘时 compile 阶段直接剪掉经济上不合法公式。

---

## F. Missing / NaN / Inf / support policy 还没有完全统一

### 187. P0 — 截面统计 finite-sample policy 不统一

当前部分算子使用：

```text
np.isfinite
```

但其他例如 demean/neutralize 路径仍可能只用：

```text
notna
mean(skipna)
```

Inf 会进入样本。

#### 必须怎么改

定义：

```python
StatisticalSamplePolicy(
    valid=FINITE_ONLY,
    min_count=...,
)
```

所有：

- rank
- mean/std/sum
- zscore
- neutralize
- regression
- group statistics

都通过统一 mask provider。

---

### 188. P0 — group global fallback 中 `np.nanmean` 仍可能纳入 ±Inf

`nanmean` 只忽略 NaN，不忽略 Inf。

所有统计 fallback 必须显式 finite mask。

---

### 189. P0 — expanding/window family 的 missing/nonfinite 语义不一致

有的：

```text
pd.isna → Inf 当合法
```

有的：

```text
np.isfinite
```

有的由 Pandas rolling 默认决定。

#### 必须怎么改

每个统计 operator 明确：

```text
SampleValidity
MinSupport
ContiguityPolicy
MissingOutputPolicy
```

并进入 semantic version/evidence。

---

### 190. P0 — `ffill` 许可当前默认 True，而且本质由用户参数控制，不是 Field semantic contract 控制

这对：

```text
return
event flag
revision
financial release
trade status
```

非常危险。

#### 必须怎么改

FieldSpec 增加：

```python
CarryForwardPolicy(
    allowed: bool,
    max_gap_sessions: int,
    reset_on_session_boundary: bool,
    reset_on_event: bool,
)
```

编译时从 input semantic contract 自动绑定。

用户只能进一步收紧，不能把 provider FORBID 改成 ALLOW。

production unknown policy → reject。

---

### 191. P0 — `fillna(method=ffill)` / aliases / SQL / Polars 必须证明全部经过同一 CarryForwardContract

做全仓 static audit：

```text
ffill
forward_fill
pad
fill_null(strategy=forward)
last_value ignore nulls
```

任何旁路 production reject。

---

### 192. P1 — `nan_to_num` 实际同时将 NaN 和 ±Inf 转成同一个 num，名称会误导 mining/用户

二选一：

1. canonical 语义改为只处理 NaN；
2. 诚实 rename `nonfinite_to_num`，旧名 alias/deprecate。

NaN 与 Inf 是不同 DQ 状态，不能默认混在一个“填零”操作里。

---

### 193. P0 — `min_periods` / estimator support 还不是完整 factor semantic identity

同一个：

```text
ts_std(x,20)
```

如果：

```text
min_periods=1
vs
min_periods=20
```

是不同因子定义。

#### 必须怎么改

OperatorContract：

```python
SupportPolicy(
    min_observations,
    require_contiguous,
    partial_window,
    ddof,
)
```

不能把 Pandas 默认值藏在 kernel 内。

---

### 194. P0 — EWM family 缺完整 recurrence semantics contract

必须显式：

```text
adjust
ignore_na
min_periods
alpha/span/halflife mapping
bias
ddof/correction
seed policy
```

否则 Pandas/Polars/SQL/recursive checkpoint 很容易“看起来接近但不完全一致”。

新增：

```text
EWMContract
```

并进入 operator semantic version + evidence。

---

### 195. P1 — WMA partial-window policy 虽已有内部常量，但应进入正式合同/身份

例如：

```text
RenormalizedPartialWMA_age_slot_anchored
```

必须进入：

```text
OperatorContract
Factor identity
backend parity evidence
```

改变该策略必须自动失效旧 cache/materialization。

---

### 196. P0 — Top-N routing 对 Inf 与 tie 的处理缺统一确定性合同

当前常见模式：

```python
valid = notna()
sort_values()
head(top)
```

问题：

- Inf 可能作为合法极值；
- 等值时结果可能依赖 instrument column order。

#### 必须怎么改

引入：

```python
TopKContract(
    sample_policy=FINITE_ONLY,
    tie_policy=STABLE_INSTRUMENT_KEY | INCLUDE_ALL_TIES | AVERAGE_WEIGHT,
)
```

禁止未声明 tie policy 的 production topk/bottomk/nlargest/nsmallest。

---

### 197. P1 — rank `method="first"` 会把物理列顺序变成 alpha

如果允许：

```text
rank(method="first")
```

同值股票的输出依赖 instrument order。

production：

- 默认禁止；或
- 明确先按 canonical InstrumentKey 排序。

该 policy 不应作为 mining 连续搜索维度。

---

## G. Elementwise 跨后端 truth-table 仍有直接语义漂移

### 198. P0 — compare 的 ±Inf 语义 Pandas / Polars / SQL 不一致

需要统一以下 truth table：

```text
finite vs finite
NaN
NULL
+Inf
-Inf
scalar Inf
```

推荐 factor numeric comparison：

```text
non-finite input → NULL/NaN
```

Pandas/Polars/DuckDB/ClickHouse 全部一致。

---

### 199. P0 — `protected_div` 对 Inf / overflow 的处理跨后端不一致

定义完整 truth table：

```text
numerator NULL
denominator NULL
NaN
+Inf/-Inf input
denom=0
abs(denom)<=eps
finite/finite normal
finite/finite overflow to Inf
```

必须有一个 `ProtectedDivisionSemantics`，各 backend emitter 从它实现。

---

### 200. P0 — `div_or_default` 的 Inf/overflow 语义同样不一致

不要让：

```text
Pandas → default
Polars/SQL → Inf
```

存在。

---

### 201. P1 — max/min 是否允许 Inf 需要正式定义，不要隐含使用不同 MissingNumeric 规则

元素数学可选择：

```text
Inf = mathematical value
```

统计样本则：

```text
Inf = invalid sample
```

两者可以不同，但必须进入 OperatorContract，optimizer 不能假设统一。

---

### 202. P0 — SQL capability 不能只“编译成功”，必须执行 edge-value semantic parity

建立共享 edge corpus：

```text
NULL
NaN
+Inf
-Inf
+0
-0
subnormal
1e308
1e-308
zero denominator
near-zero denominator
```

对 Pandas / Polars / DuckDB / ClickHouse 做可执行 parity。

---

### 203. P1 — representation transition counters 仍是 module-global ints

改 request/job scoped `ExecutionPerfCounters`，否则并发 benchmark 混在一起。

---

### 204. P1 — 所有 grain/frequency inference 必须从 contract 来，不能从 observed index 作为 authority

该规则扩展到：

```text
Pandas bridge
Polars bridge
ReadWave
checkpoint
minute agg
materializer
```

observed data 只用来验证声明，不用于自我认证。

---

### 205. P0 — typed BroadcastSpec 最终不能退化成 `allow_broadcast=True → 完全跳过 identity check`

#### 必须怎么改

实现：

```python
verify_broadcast_mapping(
    spec,
    source_axis,
    target_axis,
    calendar,
)
```

分别证明：

```text
daily→minute session mapping
same-trading-date mapping
scalar→cross-section
session-boundary mapping
instrument subset/exact policy
timezone
availability
```

不能 boolean bypass。

---

### 206. P0 — production BroadcastSpec 中 unknown availability/calendar/timezone 必须被拒绝

如果映射需要 session/date，但 contract 是：

```text
availability_policy="unknown"
calendar=None
timezone=None
```

不能因为“有 BroadcastSpec 对象”就算安全。

---

### 207. P1 — Polars→Pandas bridge 对 unexpected extra output columns 当前可能静默忽略

axis-preserving single-output operator：

```text
unexpected output column → hard fail
```

multi-output operator 需要显式：

```text
OutputColumnContract
```

---

## H. Registry / Capability 的 machine identity 仍需进一步硬化

### 208. P0 — Registry mutation 自身无单一线程安全事务约束

即使 bootstrap 加 barrier，也要保证：

```text
register
replace
unregister
alias
finalize
freeze
```

只在合法 mutation token 下发生。

运行期 public read 使用 immutable snapshot。

---

### 209. P0 — override auto-pin 到“当前看到的 old source”可能掩盖 import-order 漂移

生产 override 必须来自显式 manifest：

```python
BackendOverrideSpec(
    canonical,
    backend,
    expected_old_source,
    expected_old_contract_hash,
    new_source,
)
```

不能启动时自动把“碰巧先注册的实现”当成 expected old。

#### 必测

随机 shuffle module import order 100 次：

```text
最终 registry digest 必须一致
or
明确 hard-fail
```

绝不能静默得到不同 canonical ownership。

---

### 210. P1 — Registry 初始化流程中直接操作 `_operators` pop backend，绕过 registry audit

所有 implementation replacement 使用 typed：

```text
replace_backend()
```

并记录：

```text
old implementation hash
new implementation hash
reason
migration id
```

---

### 211. P0 — canonical logical contract 不能继续由“第一个注册的 backend implementation”决定

#### 必须怎么改

逻辑合同独立：

```text
CanonicalOperatorManifest
```

先注册 contract，再让每个 backend implementation：

```text
attest_conforms_to(contract)
```

implementation import order 不再影响 param_names/defaults/history/output semantics。

---

### 212. P1 — production canonical semantic version 必须显式 authored

禁止：

```text
missing → "1.0"
missing → ""
```

production registration gate：

```text
semantic_version explicitly declared
```

---

### 213. P0 — operator capability payload hash 失败后 fallback `repr(payload)`

production identity 不允许 repr fallback。

typed serializer error → capability infrastructure error。

---

### 214. P1 — emitter source hash 获取失败后缓存 `"unknown"`

production SQL capability：

```text
emitter identity unknown → hard fail
```

source hash应来自 BuildManifest，不依赖 runtime inspect/file open。

---

### 215. P1 — production signature authority import 失败被吞后继续生成 allowed param names

production capability check 无 signature authority → hard fail。

---

### 216. P1 — SQL emitter compile exception 被压成 `False/unsupported`，把 infrastructure failure 与“不支持”混为一谈

改返回：

```text
SUPPORTED
SEMANTICALLY_UNSUPPORTED
COMPILER_ERROR
```

生产 startup/certification 对 COMPILER_ERROR 必须失败。

---

### 217. P1 — emitter cache 容量治理仍按前文 #64/#65 执行，同时增加“模板级缓存”而非 literal 参数爆炸

不要创建新的第三套 SQL capability cache。

---

### 218. P1 — FieldSpec/TableSpec loose `metadata` 可能藏入真正 semantic key，但 dataclass compare/hash 不感知

定义扩展 namespace：

```text
semantic_extensions
descriptive_metadata
```

semantic extensions 进入 identity；description/tags 不进入。

禁止 known semantic keys 塞 loose metadata。

---

### 219. P0 — FieldSpec 的 `allowed_operator_families` / applicability / null_policy 必须审计“是否真的被 compiler 消费”

如果只是 catalog 字段但编译器没用，就是“合同写了但运行不执行”。

#### 必须怎么改

建立：

```text
FieldLegalityPass
```

检查：

- operator family allowed
- applicability
- role/value_kind
- missing/null policy
- mining_allowed
- PIT
- unit/semantic type

并做 production-path negative test。

---

### 220. P1 — production Field semantic kind 不再允许 exact-name fallback

name-based：

```text
volume → activity
event → EventBool
```

只保留 research raw-column compatibility。

production catalog field：
- explicit semantic_kind；
- 或从显式 price_basis/flow_semantics/frequency/role 可证明推导。

无法证明 → unknown → fail legality where needed。


---

# 十六、市场日历 / 股票池 / 价格口径 / 分钟会话 / Stateful / 数值确定性补充（221–260）

## I. Universe / Listing / Index Membership：股票池的 PIT 语义还不完整

### 221. P0 — `UniverseMembership` 只有 valid_time 起点，没有 valid_to/exit_time，无法表达“退出后失效”

#### 当前问题

现有 membership 主要表达：

```text
valid_time
knowledge_time
```

`effective_at(decision_time)` 只检查：

```text
decision_time >= valid_time
decision_time >= knowledge_time
```

这无法表示：

```text
2020-01-01 加入 CSI300
2021-06-15 被调出
```

如果只保留 valid_from，它在 2022 年仍会被认为 effective。

这与“退出后不能用今天成员表重写过去”的目标不完整。

#### 必须怎么改

升级为 bitemporal interval：

```python
UniverseMembershipInterval(
    universe_id,
    instrument,
    valid_from,
    valid_to_exclusive,
    knowledge_from,
    revision_time,
    source_snapshot,
)
```

查询：

```python
visible_and_effective(decision_time, trade_time)
```

必须同时满足：

```text
valid_from <= trade_time < valid_to
knowledge_from <= decision_time
revision visible
```

对没有 valid_to 的成员显式：

```text
OPEN_ENDED
```

不能把“字段缺失”和“仍有效”混在一起。

#### 必测

- 加入前
- 加入当天
- 调出前一天
- 调出当天
- 调出后
- 后来修订 membership
- decision time 在公告前/后
- today-members 回测历史不能污染

---

### 222. P0 — `UniverseMembership.effective_at()` 把“valid_time 与 knowledge_time 都缺失”直接当 True，production 属于 fail-open

#### 必须怎么改

production：

```text
missing temporal membership contract → UniverseKnowledgeUnknownError
```

research 若允许 static universe：

必须显式：

```python
StaticUniverseSnapshot(
    as_of=...,
    intended_use="research_only",
)
```

并写 lineage：

```text
NON_PIT_STATIC_UNIVERSE
```

---

### 223. P0 — Universe membership identity 仅 hash 当前 member tuple/as_of，不能表达成员区间与 announcement/revision history

#### 必须怎么改

`UniverseMembershipDigest` 必须基于：

```text
universe definition version
all effective intervals intersecting requested range
knowledge/revision coordinates
provider snapshot
calendar
```

如果 requested range 是 2018–2026，不能只 hash 2026 一张成员表。

---

### 224. P0 — 上市/退市/ST/停牌/可交易状态应拆成不同 mask 维度，不能只合成一个 opaque universe mask

最终推荐：

```python
CrossSectionEligibility(
    listed_mask,
    universe_membership_mask,
    instrument_type_mask,
    st_policy_mask,
    suspension_mask,
    valid_price_mask,
    corporate_action_state_mask,
)
```

这样：

- “策略股票池”
- “能否交易”
- “能否参与截面统计”
- “是否有有效行情”

不再混成一个 bool。

每个 operator/campaign 可声明：

```text
ranking eligibility policy
trading eligibility policy
```

---

### 225. P1 — Universe coverage ratio 不能只看“finite cells 保留比例”，还要区分缺失原因

至少记录：

```text
OUT_NOT_LISTED
OUT_DELISTED
OUT_NOT_MEMBER
OUT_SUSPENDED
OUT_ST_POLICY
OUT_NO_VALID_BAR
OUT_UNKNOWN_STATUS
OUT_UNKNOWN_MEMBERSHIP
```

否则 30% coverage loss 无法判断是正常过滤还是数据断档。

---

## J. Instrument Identity：ticker 不是永久资产身份

### 226. P1 — `InstrumentKey(market, instrument)` 只做字符串 strip，缺 market-specific canonicalization

例如 US ticker：

```text
aapl
AAPL
```

以及数据源大小写不同可能成为两个 InstrumentKey。

A 股也要统一：

```text
000001.SZ
SZ000001
000001
```

如果 provider 有不同表达。

#### 必须怎么改

建立：

```python
InstrumentNormalizer.for_market(market)
```

输出稳定：

```text
canonical_security_id
display_ticker
provider_symbol
market
exchange
```

cache/factor lake identity 用 canonical_security_id，而不是 provider ticker。

---

### 227. P1 — ticker rename / symbol reuse / corporate identity change 需要 permanent security ID

长期历史中 ticker 可能：

- 改名
- 退市后代码重用
- provider symbol change

因此应支持：

```text
SecurityMasterId
+
SymbolValidityInterval
```

不要把 ticker string 当永久主键。

---

## K. Price Basis / Corporate Action：价格口径必须成为真正的编译类型

### 228. P0 — 涨跌停算子没有机器可读地强制 OHLC 与 official limit price 同为 RAW 官方口径

#### 风险

如果：

```text
close = 后复权 close
upper_limit = 原始官方涨停价
```

公式仍能数值运行：

```text
close >= upper_limit
```

但完全错误。

#### 必须怎么改

OperatorContract 对涨跌停系列声明：

```text
open/high/low/close:
  PriceBasis = RAW

upper_limit/lower_limit:
  PriceBasis = RAW_OFFICIAL_LIMIT

comparison compatibility:
  RAW price ↔ RAW_OFFICIAL_LIMIT allowed
  CONTINUOUS/ADJUSTED ↔ official limit forbidden
```

compile 时 fail。

---

### 229. P0 — Return 的价格基准必须显式区分 raw return / adjusted total return / close-to-close / open-close

不能把所有：

```text
ret
return
pct_change(close)
```

都视为同一种 `RETURN_DECIMAL`。

建议：

```python
ReturnSemantic(
    source_price_basis,
    interval=CLOSE_TO_CLOSE | OPEN_TO_CLOSE | OVERNIGHT,
    corporate_action_adjustment=RAW | SPLIT_DIV_ADJUSTED | TOTAL_RETURN,
)
```

进入 semantic identity。

---

### 230. P0 — Corporate-action adjustment factor 必须 PIT/versioned，不能用“今天最终复权因子”回写历史而无身份

对于前/后复权序列：

```text
同一历史 TradeDate 的 adjusted price
```

会随着后续除权事件变化。

必须明确：

```text
AdjustmentPolicy
AdjustmentKnowledgeTime
AdjustmentVintage
AsOfPolicy
```

研究允许 retrospective continuous series，但必须标：

```text
RETROSPECTIVE_ADJUSTED
```

生产实时因子不能把未来 corporate action adjustment 反映到过去输入。

---

### 231. P1 — tick tolerance 应从 instrument/tick-size contract 得到，不能默认统一绝对 `0.005`

A 股大多数股票 tick 是 0.01，但不同资产类别/未来扩展不应写死。

Operator 参数允许用户 override 时也应：

```text
tick_tolerance <= declared tick policy bound
```

最好：

```python
PriceGridContract.tick_size(instrument, trade_date)
```

---

## L. SessionCalendar / Intraday Clock：当前还有两套 session 模型，需要收敛

### 232. P0 — `market.SessionSpec` 与 `runtime.SessionCalendar` 是两套会话事实源

目前存在：

```text
market/session.py → SessionSpec
runtime/session_calendar.py → SessionCalendar
```

两者都描述：

- segment
- timestamp convention
- slot count
- early close/session

长期一定会漂移。

#### 必须怎么改

建立唯一：

```python
ExchangeSessionCalendar
```

包含：

```text
market
exchange
timezone
calendar_version
trade_date
segments
auctions
lunch break
early close
timestamp convention
bar grids
holidays
DST
```

`SessionSpec` 只可作为 immutable projection，不维护第二份业务规则。

---

### 233. P0 — `SessionCalendar` 自己说明 holiday selection 由上层负责，但 offset/warmup 又能自己用“周一到周五近似”

production 任何跨日 bar offset 不得在 calendar 缺真实 holiday set 时继续。

#### 必须怎么改

```text
calendar.authoritativeness == EXCHANGE_CERTIFIED
```

才允许 production `offset_bars/warmup_load_start`。

没有：

```text
hard fail
```

research 才可 `WEEKDAY_APPROXIMATION` 并记录 degradation。

---

### 234. P0 — `SessionCalendar` 没有一等 timezone，`session_key()` 直接 normalize 输入时间，可能在 UTC 数据上切错 TradeDate

timezone 必须进入 calendar 本身。

所有：

```text
session_key
minute_ordinal
segment_id
bar_slots
```

先统一到 session-local timezone。

---

### 235. P0 — US early-close / DST 不能由 generic `SessionCalendar(market="US")` + 固定 09:30–16:00 完整表达

production US minute path 必须按具体 trade_date 取 authoritative session：

```text
regular close
early close
DST timezone conversion
holiday
```

不能 generic fixed session。

---

### 236. P1 — `SessionSpec.to_dict()` 没包含完整 early-close date/time table，若用于 identity 会漏重要语义

Session identity 应至少绑定：

```text
calendar_version
early_close source digest
bar convention
segment definitions
timezone
```

不要把 huge dates 直接塞每个 factor identity，可绑定 authoritative CalendarDigest。

---

### 237. P1 — `early_close_policy="down_weight"` 本身不够确定，缺具体数学定义

down_weight 到底是：

```text
按 slot count?
按 session duration?
按 regular-day ratio?
按 volatility scaling?
```

必须用：

```python
EarlyCloseNormalizationPolicy
```

明确公式、适用 operator family、semantic version。

---

## M. Minute→Daily SessionPanel：DQ fail 与“因子自然 NaN”没有完全分开

### 238. P0 — duplicate official minute slot 虽被标记 DQ failed，但聚合层可能只把结果变 NaN，而不是让生产 run DQ 失败

`SessionPanel` 目前：

```text
is_duplicate=True
is_valid_bar=False
```

但只要 coverage 仍高于 90%，部分统计可能继续输出。

#### 必须怎么改

定义：

```python
SessionDQReport(
    duplicate_slots,
    off_grid_rows,
    missing_slots,
    timezone_errors,
    multi_trade_date_input,
)
```

production policy：

```text
duplicate official slot > 0 → hard DQ fail
```

除非 provider contract 明确有合法 dedupe policy，并在 DataAccess 层完成 deterministic dedupe。

---

### 239. P0 — minute aggregation `_daily_agg/_pair_agg/_triple_agg` broad catch `ValueError` 后写 NaN，会掩盖结构性 DQ/合同错误

#### 必须怎么改

异常分层：

```text
InvalidNumericSample → output NaN 可以
InsufficientCoverage → output NaN/typed missing
DuplicateSlotError → DQ fail
CalendarContractError → run fail
TimezoneContractError → run fail
AxisMismatchError → run fail
```

不能所有 ValueError 都变因子缺失。

---

### 240. P0 — `build_session_panel()` 本身没有强制输入 timestamps 属于单一 session trade date

如果调用方错误地传入两天数据：

```text
minute-of-day 会映射到同一 grid
→ 不同日期的同一分钟可能变 duplicate
```

应显式：

```python
unique(session_trade_date(timestamps)) == 1
```

否则 `MultipleSessionDatesError`。

---

### 241. P1 — off-grid bars 当前主要被 drop，必须可观测并有 policy

例如：

- 午休
- 集合竞价
- 错时区
- provider timestamp offset
- 非法 09:30/13:00 标签

都可能变 off-grid。

必须记录：

```text
off_grid_count
off_grid_examples
off_grid_ratio
```

production 超阈值 hard fail；不能静默丢。

---

### 242. P0 — minute aggregator 默认 `market="ashare"` 属于生产危险默认

production minute operator 必须从 ExecutionSemanticContext 获取 market。

缺 market：

```text
hard fail
```

不能：

```python
market or "ashare"
```

---

### 243. P0 — non-A-share `_declared_calendar()` 不能通过 generic bar-end calendar 猜 session

US bar-start、HK session、期货夜盘等都不同。

生产：

```text
known market → authoritative calendar adapter
unknown market → unsupported
```

---

## N. Stateful / Checkpoint：这部分还存在几个真正会污染增量结果的问题

### 244. P0 — `_source_snapshot_scope()` 计算失败返回固定 `"ephemeral"`，不同未知数据源可能共享 checkpoint identity

#### 当前风险

两个 source 都因为 scope resolver 故障：

```text
source A → ephemeral
source B → ephemeral
```

可能被视作同数据身份。

#### 必须怎么改

production：

```text
source snapshot identity unresolved → checkpoint resume disabled + hard/full replay
```

不能生成可复用 checkpoint identity `"ephemeral"`。

research 如允许：

```text
unique run-scoped ephemeral ID
```

绝不能跨 run resume。

---

### 245. P0 — `_boundary_timeline()` probe 失败返回 `[]`，continuity auditor 随后可能“无法证明有 gap”却允许 resume

这属于明显 fail-open。

#### 必须怎么改

返回 typed：

```text
BoundaryProbeResult.PROVEN_CONTIGUOUS
BoundaryProbeResult.GAP
BoundaryProbeResult.UNKNOWN
```

production：

```text
UNKNOWN → 不允许 checkpoint resume → full replay
```

---

### 246. P0 — boundary timeline 只按第一个 instrument 的 checkpoint probe 一次，却用于所有 instruments

不同股票的 latest checkpoint.as_of 可能不同。

#### 必须怎么改

按：

```text
distinct checkpoint.as_of
```

分组 probe；或每 instrument 证明 continuity。

不能把第一个 instrument 的 boundary 当全 universe authority。

---

### 247. P0 — stateful multi-input source panels 没有在 segmented path 前严格验证 exact axis identity

当前多个 input column 分别 unstack 后，应明确验证：

```text
same timestamp index
same instrument columns
same ordering
same source snapshot
```

不能只把 first input 当 reference。

#### 必须怎么改

复用 `AxisIdentityCertificate / verify_frames_share_identity`，并要求 production exact。

---

### 248. P0 — checkpoint input identity 没有完整绑定 ExecutionSemanticContext

目前应进一步绑定：

```text
market
calendar/version
timezone
universe/membership hash
price basis
field contract digests
operator semantic contract digest
actual kernel variant
numeric semantics
missing/support policy
```

不能只 factor_id+canonical+columns+params+source scope。

---

### 249. P0 — stateful segmented path 的 source load / execution / checkpoint commit 错误多数返回 `None` 再 full replay，必须区分“可安全 fallback”与“系统损坏”

可 fallback：

```text
no checkpoint
checkpoint incompatible
unsupported segmented expression
```

必须 hard fail：

```text
checkpoint corruption
identity serializer corruption
source PIT violation
calendar error
permission
state schema corruption
commit durability failure where publish already advanced
```

引入 typed `SegmentedFallbackReason`。

---

### 250. P0 — stateful checkpoint path 必须有完整 chunk-invariance property，而不是只少数 known kernels example parity

对每个 checkpoint-capable canonical：

```text
full_history(x)
==
concat(
  segment1,
  resume(segment2),
  resume(segment3)
)
```

对随机 chunk boundaries、NaN/Inf、gaps、single-bar chunks、market holidays 做 property test。

Hard Gate：

```text
CHECKPOINT_CHUNK_INVARIANCE_FAILURE == 0
```

---

## O. 数值稳定性、确定性和跨机器可复现性

### 251. P0 — regression/neutralization 对 near-zero variance 不能只判断 `var == 0`

浮点数下：

```text
var = 1e-30
```

虽然非零，但 beta 会爆炸。

#### 必须怎么改

定义机器可读：

```python
DegeneracyPolicy(
    absolute_floor,
    relative_floor,
    condition_number_limit,
)
```

OLS/neutralize/beta/tstat 共享。

阈值进入 semantic/numeric contract，不允许各 kernel 手写不同 epsilon。

---

### 252. P0 — 高动态范围 sum/mean/variance 需要数值稳定算法合同

对于：

```text
1e16 + small values
大市值/成交额
长期 expanding
```

普通累计 sum 会丢精度。

对关键算子评估：

```text
pairwise summation
Kahan/Neumaier
Welford
stable centered moments
```

并规定 canonical numeric algorithm。

不要为了“更稳定”随意更换而不 bump semantic/numeric version。

---

### 253. P0 — ddof/bias/correction 必须成为统一统计合同

涉及：

```text
std
var
cov
corr
skew
kurt
EWM var/std/cov
regression variance/tstat
```

禁止依赖 backend 默认。

定义：

```python
MomentConvention(
    ddof,
    bias,
    fisher,
    nan_policy,
    finite_policy,
)
```

进入 backend parity evidence。

---

### 254. P0 — float32 / float64 / Decimal 输入的 canonical compute precision 要明确

建议 production factor numeric canonical：

```text
compute_float64
```

除非 operator contract 显式支持其它精度。

如果 source float32：

```text
输入 dtype identity保留
计算是否 promote 到 float64明确
输出 storage precision明确
```

不能 backend 各自自动 cast。

---

### 255. P1 — signed zero、subnormal、overflow policy 需要 edge-contract

尤其：

```text
log
sign
reciprocal
protected_div
power
rank around ±0
```

对：

```text
+0.0
-0.0
tiny subnormal
overflow
```

明确是否保留/归一。

通常 factor identity不应因 NaN payload/sign-bit 随机器漂移。

---

### 256. P0 — 并行 reduction / BLAS 线程数不同不得导致不可控因子值漂移

对：

- PCA
- regression
- covariance
- matrix ops
- group reductions

定义数值容差等级：

```text
BITWISE_DETERMINISTIC
DETERMINISTIC_WITHIN_TOLERANCE
NONDETERMINISTIC_RESEARCH_ONLY
```

production evidence 必须标级别。

如果依赖 MKL/OpenBLAS 多线程，记录：

```text
BLAS vendor/version/thread config
```

---

### 257. P0 — tie-sensitive operator 必须具有 permutation-equivariance 测试

对于：

```text
rank
quantile bucket
topk
group rank
winsorize
neutralize
cross-sectional regression
```

随机 permutation instrument columns：

```text
permute input
→ compute
→ unpermute output
```

应与原结果相同（除非 contract 明确依赖 InstrumentKey tie-break，仍应按 key 得到相同结果）。

---

### 258. P0 — 所有 causal TS operator 必须做 prefix-invariance，而不是只测试最终窗口

测试：

```text
F(x[0:t]) == F(x[0:T])[0:t]
```

对随机 t、随机 NaN/gap、不同 backend。

任何未来数据改变历史输出：

```text
production hard fail
```

---

### 259. P0 — 所有可分块/流式 operator 必须做 chunk-boundary invariance

不仅 checkpoint operator。

例如：

```text
rolling
EWM
minute aggregation
streaming group
writer block DQ
```

随机 chunk partition 后结果与 monolithic reference 相同。

---

### 260. P1 — 生产证据需要跨进程/跨重启确定性测试

至少同一个 build/source snapshot：

```text
process A
process B
fresh restart
different worker count
```

比较：

```text
factor identity
plan hash
axis hash
cache key
result values
materialization checksum
```

不能只在同一 Python process 证明稳定。

---

# 十七、151–260 新增项的专项 Hard Gates

完成新增整改后，再加以下 Gate。

## Gate-N1 Registry bootstrap

```text
REGISTRY_PARTIAL_VISIBILITY == 0
REGISTRY_FIRST_CALLER_SURFACE_CONTAMINATION == 0
REGISTRY_FAILED_BOOTSTRAP_PARTIAL_STATE == 0
```

## Gate-N2 Parameter certification

```text
PRODUCTION_PARAMETER_CERTIFICATION_SKIP == 0
PRODUCTION_PARAMETER_CERT_INFRA_ERROR_SWALLOWED == 0
INCOMPLETE_BOUND_PARAMETER_POINT == 0
```

## Gate-N3 Axis truth

```text
AXIS_PRESERVING_RESULT_RESTAMP_WITHOUT_VERIFY == 0
PRODUCTION_UNKNOWN_GRAIN_COMPATIBILITY == 0
PHYSICAL_COLUMN_NAME_COLLISION == 0
```

## Gate-N4 Market/time truth

```text
PRODUCTION_MARKET_DEFAULT_GUESS == 0
PRODUCTION_WEEKDAY_ONLY_CALENDAR == 0
PRODUCTION_GENERIC_SESSION_GUESS == 0
```

## Gate-N5 Universe truth

```text
STATIC_TODAY_UNIVERSE_USED_FOR_HISTORY == 0
MEMBERSHIP_WITHOUT_VALIDITY_INTERVAL_PRODUCTION == 0
UNKNOWN_MEMBERSHIP_TREATED_IN_POOL == 0
```

## Gate-N6 Price-basis truth

```text
ADJUSTED_PRICE_COMPARED_TO_RAW_LIMIT == 0
PRICE_BASIS_UNKNOWN_IN_PRODUCTION_PRICE_OPERATOR == 0
```

## Gate-N7 Minute DQ

```text
DUPLICATE_OFFICIAL_SLOT_SILENT_NAN == 0
MULTI_SESSION_DATE_PANEL == 0
OFF_GRID_BAR_UNOBSERVED == 0
```

## Gate-N8 Stateful

```text
EPHEMERAL_CHECKPOINT_IDENTITY_REUSED == 0
UNKNOWN_BOUNDARY_CONTINUITY_RESUMED == 0
MULTI_INPUT_AXIS_UNVERIFIED_CHECKPOINT == 0
CHECKPOINT_CHUNK_INVARIANCE_FAILURE == 0
```

## Gate-N9 Numeric semantics

```text
UNDECLARED_DDOF_BIAS == 0
UNDECLARED_NONFINITE_POLICY == 0
UNDECLARED_TIE_POLICY == 0
PRODUCTION_NONDETERMINISTIC_NUMERIC_UNCLASSIFIED == 0
```

---

# 十八、新增测试套件建议

新增测试目录建议：

```text
tests/registry/test_bootstrap_concurrency.py
tests/registry/test_surface_isolation.py
tests/registry/test_transactional_bootstrap.py

tests/parameters/test_production_certification_fail_closed.py
tests/parameters/test_dtype_signature.py
tests/parameters/test_actual_execution_variant.py

tests/axis/test_polars_row_reorder_detection.py
tests/axis/test_pandas_bridge_index_reorder.py
tests/axis/test_column_name_injective.py
tests/axis/test_unknown_grain_production.py

tests/universe/test_membership_interval_pit.py
tests/universe/test_membership_revision.py
tests/universe/test_listing_delisting_masks.py

tests/market/test_price_basis_legality.py
tests/market/test_corporate_action_adjustment_vintage.py
tests/market/test_session_calendar_authoritative.py

tests/minute/test_duplicate_slot_hard_dq.py
tests/minute/test_multiple_trade_dates_rejected.py
tests/minute/test_offgrid_observability.py
tests/minute/test_us_early_close_dst.py

tests/stateful/test_boundary_probe_fail_closed.py
tests/stateful/test_per_instrument_checkpoint_boundary.py
tests/stateful/test_multi_input_axis_identity.py
tests/stateful/test_chunk_invariance_property.py

tests/numerics/test_nonfinite_truth_table.py
tests/numerics/test_ddof_bias_contract.py
tests/numerics/test_regression_near_singular.py
tests/numerics/test_permutation_equivariance.py
tests/numerics/test_prefix_invariance_property.py
tests/numerics/test_chunk_invariance_property.py
tests/numerics/test_cross_process_determinism.py
```

property-based test 推荐使用 Hypothesis，但如果项目不希望增加 runtime dependency，可作为 test extra：

```toml
[project.optional-dependencies]
test = ["hypothesis>=..."]
```

---

# 十九、对 coding AI 的新增执行要求

不要把 151–260 当作“建议性优化”。

执行时建立：

```text
R40_ISSUE_CLOSURE_LEDGER.md
```

每条必须包含：

```text
Issue
Severity
Current code location
Root cause
Implementation
Why this fixes production path
Unit tests
Integration tests
Negative controls
Current-SHA evidence
Status
```

尤其对 P0，禁止只写：

```text
“已有保护”
“测试里没复现”
“暂时不影响”
```

如果认为某项在执行时最新 HEAD 已经修复：

```text
必须给代码路径 + 测试 + current SHA proof
```

才能 `FIXED_ALREADY`。

如果继续发现新问题：

```text
NEW-261+
```

继续追加并修，不要停止。
