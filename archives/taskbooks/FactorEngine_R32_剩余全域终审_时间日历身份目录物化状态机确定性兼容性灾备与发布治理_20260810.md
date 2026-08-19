# FactorEngine R32：剩余全域终审——时间日历、身份、目录、物化、状态机、确定性、兼容性、灾备与发布治理

- 仓库：`18047533889/quant_projects`
- 审计对象：`factor_engine`，必要时联动 `dataaccess`
- 当前基线 HEAD：`58490a634eef36efc92ecaec945fc4a26016661c`
- 日期：2026-08-10
- 当前前提：本轮继续以该 HEAD 为唯一代码基线。
- R32 定位：在 R30（逐算子/PIT/身份安全）和 R31（系统架构/三后端/性能/调度）之外，继续补齐此前没有系统收口的生产工程问题。
- 本文既包含当前 HEAD 已经确认的代码问题，也包含 FactorEngine 要达到高成熟度必须建立的契约、测试、迁移和灾备能力。两者在条目中明确区分。

## 总结

R30/R31 之后，FactorEngine 仍然不能只看“算子正确、后端够快”。生产级因子引擎还必须保证：

1. 交易日历、session、时区、分钟 bar offset 正确；
2. ticker、instrument、universe、calendar 都有稳定且版本化的身份；
3. Catalog/JobStore/Materializer 的事务和状态机不会在并发、重启、失败时分叉；
4. float64 不会在增量更新时被悄悄降成 float32；
5. 输出 grain 不会在物化时被静默丢维；
6. factor_id 不会因为截断、路径穿越、大小写或 Unicode 造成碰撞；
7. factor identity 不会因为无关算子/字段变化而全库失效，也不会因为 secret redaction 丢失真实数据源身份；
8. schema、cache、checkpoint、factor lake、catalog 都能版本化迁移；
9. 同一版本在不同线程数、硬件、BLAS/库版本下有明确的确定性等级；
10. 服务重启、任务取消、磁盘满、worker crash、catalog 损坏时可以恢复；
11. 删除/改名算子后，cold-start、recipe、mining grammar、文档、evidence 全同步；
12. release 能 canary、shadow、rollback、DR restore，而不是只看 pytest green。

下面按“当前已确认问题 → 体系整改 → 验收门禁”一次性列完。


## 1. 当前 HEAD 新确认问题总表

### R32-P0-001｜交易日历加载失败会静默退化为普通工作日

`storage/trading_calendar.py` 的 DataAccess 日历读取使用宽泛 `except Exception: return None`，之后可退回 `pd.bdate_range`。生产环境不得把“真实交易日历不可用”伪装成“普通周一到周五”。

整改：
- production：`CalendarUnavailableError`，禁止 fallback；
- research：只有显式 `allow_approximate_calendar=True` 才允许 bdate；
- 记录 calendar source、snapshot、version。

### R32-P0-002｜TradingCalendar.offset 超出 coverage 会静默 clamp

当前超出左/右边界时返回第一天/最后一天。缺历史日历会变成一个看似合法的 warmup 日期。

整改：
- 默认抛 `CalendarCoverageError`；
- `clamp=True` 仅用于显式 UI/研究用途；
- production incremental 禁止 clamp。

### R32-P0-003｜n=0 不验证 base 是否交易日

`offset(base,0)` 直接返回 base，周末/节假日也返回原日期。

整改：
定义明确 anchor policy：
- `exact_trade_day`
- `previous_trade_day`
- `next_trade_day`

生产默认 exact/fail-closed。

### R32-P0-004｜日内 incremental 的 tick_precise 实际不是 session_precise

当前逻辑本质上是：
`交易日零点 + bar_duration × bars_per_day`，
再减 lookback bar。

这不认识：
- 09:30 开盘；
- 11:30–13:00 午休；
- 集合竞价；
- 半日市；
- DST；
- early close；
- 临时休市。

整改：
引入 `SessionCalendar + BarClock + SessionSlotIndex`，所有分钟 lookback/tail/forward-impact 在真实 session slot 上偏移。

### R32-P0-005｜timezone-aware bound 到 naive index 时直接 strip tz

不能把 `tz_localize(None)` 当成跨时区转换。必须先按 source/index timezone 做 `tz_convert`，再按明确 timestamp convention 处理。

### R32-P0-006｜Composite source 被统一裁同一个 start/end

对 price source 可能没问题，但 PIT fundamental/consensus/shareholder/event/as-of source 往往需要 output_start 之前的最后一条历史记录。

整改：
每个 source dependency 自己声明 `SourceHistoryRequirement`，不能统一剪裁。

### R32-P0-007｜Structural CSE 会生成 orphan shared nodes

当前所有重复 subtree 都登记 shared；父 subtree 被 plan_ref 替换后，其内部重复 child 不再被 rewrite，child shared 可能无人引用。运行时完整性校验又会拒绝 orphan。

整改二选一：
A. 只提取 maximal profitable repeated subtree；
B. shared definition 自己也 rewrite 成 nested shared DAG。

推荐 B + cost-based CSE。

### R32-P0-008｜CSE 对 literal/column/便宜节点也一视同仁

出现两次不等于值得 materialize。引入：
`benefit = recompute_saved - materialize_cost - memory_cost - conversion_cost`

### R32-P0-009｜Rolling CSE 又实现了一套 repr-based semantic hash

`rolling_cse` 对未知 semantic attr 使用 `repr()`，与主 `plan_hash` 的 typed fail-closed 语义冲突。

整改：
删除重复 canonicalizer，所有 semantic digest 只走一个权威实现。

### R32-P0-010｜FactorCatalog 未看到 foreign_keys=ON

SQLite foreign key 默认关闭。Schema 虽写了 FK，如果没有 `PRAGMA foreign_keys=ON`，约束可能实际不执行。

启动必须验证：
`PRAGMA foreign_keys == 1`

### R32-P0-011｜Catalog execute→commit 不是一个锁住的 transaction

`execute()` 和 `commit()` 各自独立锁。线程 A execute 后释放锁，线程 B 可以插入 statement，再被 A 的 commit 一起提交。

整改：
`with catalog.transaction():`
必须持同一 RLock + BEGIN/COMMIT 完整区间。

### R32-P0-012｜Catalog migration 缺 schema-versioned 独占事务

需要：
- catalog_schema_version；
- `BEGIN IMMEDIATE/EXCLUSIVE`；
- from_version → to_version；
- migration checksum；
- backup；
- migration report。

### R32-P0-013｜checkpoint partition_key 应 NOT NULL

作为逻辑主键的一部分，必须 `TEXT NOT NULL`，并迁移 legacy NULL。

### R32-P0-014｜Catalog production mode 解析异常会 fail-open 到 research

Production authority 解析失败不能默认 `False`。应直接抛 `ProductionModeResolutionError`。

### R32-P0-015｜JobStore 内存 idempotency key 与 SQLite 语义不一致

内存索引只按 `idempotency_key`；SQLite 唯一键按 `(idempotency_key, owner_principal, job_type)`。

统一成：
`(owner_principal, job_type, idempotency_key)`

### R32-P0-016｜SQLite JobStore 使用 INSERT OR REPLACE，不是真 idempotency

两个进程同时提交相同 idempotency tuple，可能互相 replace。应：
`INSERT ... ON CONFLICT DO NOTHING`
随后 SELECT existing run_id。

同 key 不同 request_digest 应 409，而不是静默复用。

### R32-P0-017｜startup reconciliation 只改内存，不 durable persist

启动把 running/queued 标成 interrupted 后，不能 `write_manifest=False`。必须写回 durable authority。

### R32-P0-018｜Job manifest rename 后缺 directory fsync

若宣称 durable，`os.replace` 后需要 fsync manifest directory。

### R32-P0-019｜Queue drain 状态机与“drain”语义冲突

当前一开始 `_stopping=True`，worker loop 又以 `while not _stopping` 运行，因此 queued job 可能不再被消费。

拆状态：
- ACCEPTING
- DRAINING
- STOPPING
- STOPPED

DRAINING：拒绝新任务，但继续清空已有队列。

### R32-P0-020｜Queue admission 检查与 put 不是原子

qsize/running 检查后再 `put()` 有竞态，并可能阻塞。

整改：
- `put_nowait`
- 失败 rollback per-principal counters
- 不允许请求线程无限阻塞。

### R32-P0-021｜unexpected run_fn exception 可能杀掉 worker thread

worker loop 必须兜底：
- job → FAILED；
- error durable；
- worker继续服务下一任务；
- unexpected exception 不等于 worker pool容量永久减少。

### R32-P0-022｜heartbeat monitor 标 interrupted 后，真实线程还可能随后写 succeeded

需要 CAS 状态迁移：
`UPDATE ... WHERE status = RUNNING`
terminal state 不能被后来的旧执行覆盖。

### R32-P0-023｜SQLite JobStore 并不等于全局多进程 Queue

多个 Uvicorn worker各自有本地 queue、max_running、per-user counter。4×4 worker可能实际跑16个 job。

生产选择：
A. 单 service process + FactorEngine内部并行；
B. 真正外部/global lease queue。

### R32-P0-024｜增量 upsert 会把旧 float64 历史数据固定 downcast 为 float32

当前已有 Parquet 时会执行：
`existing_df["value"].astype("float32")`

这是严重 P0。Production 默认 float64 的历史分区只要被更新一次，旧值就被永久降精度。

整改：
- 读取 existing parquet dtype；
- 读取 partition precision contract；
- 与 current precision policy 对齐；
- 禁止隐式 downcast；
- precision change 必须显式 migration/rebuild。

### R32-P0-025｜Materializer 对 MultiIndex nlevels>=2 静默丢额外 grain

只保留前两层 datetime/asset，第三维及以上被丢掉。

整改：
引入 `OutputGrainContract`。
Daily factor必须 exact `timestamp × instrument`；
minute/session/event/relation 走专门 schema，禁止静默降维。

### R32-P0-026｜factor_schema.py 与 materializer 实际 metadata 已漂移

Schema authority缺：
- `resolved_snapshot_id`
- `storage_precision_policy`

需要 FactorLakeSchemaVersion，Parquet/DataAccess/ClickHouse/readers/writers单一事实源。

### R32-P0-027｜write_mode 没严格 enum gate

除了 `"append"`，其它拼错字符串会默认为 keep-last upsert。

入口严格只允许：
- upsert
- append
- replace_window
- recompute_window

### R32-P0-028｜wide storage path 没完整遵守 write_mode

wide path固定 keep-last，未完整处理 append/replace_window/recompute_window。

如果 wide保留 production能力，必须与 long语义一致；否则降为 export/research-only。

### R32-P0-029｜wide row_count/metadata 与 long 语义不一致

wide Parquet 一行是日期，long 一行是 cell。不能共用一个不加解释的 row_count。

统一记录：
- physical_row_count
- date_count
- asset_count
- cell_count
- non_null_cell_count

### R32-P0-030｜precision parity 的 rank_changes 不是逐日横截面

当前把所有日期/股票 flatten 后统一 rank。因子精度认证必须逐 timestamp 做横截面 rank/top-k/quantile overlap，再聚合。

### R32-P0-031｜precision parity equal 可忽略 NaN mask mismatch

必须满足：
- exact key/index match；
- `nan_mismatch == 0`；
- `+inf/-inf mask` 一致；
- finite值 tolerance通过；
才能 equal。

### R32-P0-032｜checkpoint identity sidecar 写失败 production仍吞掉

Production不能把“数据写成功但 identity sidecar失败”标成完整成功。Checkpoint success必须在 sidecar durable 后提交。

### R32-P0-033｜identity sidecar tmp只用 PID，线程内会碰撞

改用 uuid/run_id/task_id，且写操作放在 partition lock 内。

### R32-P0-034｜deleted_keys-only tombstone 可能缺完整 run metadata

如果 df没有正常行，不能用 `df.iloc[0]` 取 metadata。删除 tombstone必须显式从当前 MaterializeMetadata生成。

### R32-P0-035｜factor name/id 采用截断而不是 reject

HTTP validator `[:128]` 会让两个不同 ID 静默碰撞。超长必须报 validation error。

### R32-P0-036｜factor_id 缺统一 filesystem-safe domain gate

Materializer/Delete直接把 factor_id拼到 path。必须统一 `FactorId`：
- 无 `../`
- 无 `/`、反斜杠；
- 无控制字符；
- 长度有限；
- Unicode/case策略明确；
- resolve后仍在 allowed root下。

### R32-P0-037｜DSN secret redaction 可能把不同数据源折成同一 source hash

不能把整个 DSN替换为 `<redacted>`。应 parse URI，只去：
- password/token/credential
保留：
- scheme
- host
- port
- database
- dataset/path
- semantic options

### R32-P0-038｜credential 藏在普通 url 值里可能没被 key-name redaction发现

需要 value-aware URI sanitizer。

### R32-P0-039｜relative Path source identity 可能跨 working directory碰撞

优先以：
`logical dataset id + manifest snapshot`
作为 source identity，而不是依赖 CWD下的 `./data` 字符串。

### R32-P0-040｜每个因子绑定整个 operator/field catalog，导致无关改动触发全库失效

添加一个完全无关 operator，不应让所有 factor_version/cache/checkpoint失效。

改：
- `PlanOperatorDependencyDigest`
- `PlanFieldDependencyDigest`
- `PlanSourceDependencyDigest`

真正全局语义（numeric/compiler/calendar rule）单独 global digest。

### R32-P0-041｜lineage field catalog hash失败会写空字符串

Production lineage必须 fail closed。

### R32-P0-042｜RunLineage engine_version 不是实际可复现版本

记录：
- package_version
- build/git SHA
- DataAccess version
- Python
- NumPy/Pandas/Polars/DuckDB/PyArrow/SciPy
- container/build digest

### R32-P0-043｜Direct Python API 的 Factor.name 可绕过 service validator

FactorId验证必须下沉 domain层，不只 HTTP。

### R32-P1-044｜`FactorSemanticIdentity = FactorExecutionScopeHint` 兼容别名应退役

runtime已有真正 FactorSemanticIdentity，继续保留重名会误导开发者。做 deprecation周期后删除。

### R32-P1-045｜Persistent cache per-key lock表可能无限增长

模块级 `_SAVE_LOCKS` 对每个唯一 key保留 Lock。改：
- striped locks；
- WeakValueDictionary；
- bounded lock registry。

### R32-P1-046｜persistent cache 的 `"unknown_ops"` namespace production不得使用

语义 namespace unresolved：
production禁写/禁读；
research只当 cache miss。

### R32-P1-047｜大量 planner/hash/CSE 递归，深表达式可能 RecursionError

必须有：
- max AST depth
- max plan depth
- iterative traversal
- cycle detection

### R32-P1-048｜Physical DAG critical path递归无 memo

改 reverse-topological DP，O(V+E)。

### R32-P1-049｜topological_order 用 pop(0)+反复 sort

大 DAG改 heapq/deque。

### R32-P1-050｜FactorBatchGraph 因子依赖推断 O(F²)

未来 FactorRef应显式 edge；不要 10k/100k候选 pairwise扫描。

### R32-P1-051｜dependency layers 反复扫描 remaining

标准 Kahn indegree算法。

### R32-P1-052｜Factor lake缺正式 schema migration authority

需要：
- current schema version
- min readable version
- migration functions
- backward compatibility tests。

### R32-P1-053｜fcntl不可用时 partition lock静默 no-op

需要 strict mode或 single-writer退化，不能假装有并发安全。

### R32-P1-054｜partition lock无 timeout/owner诊断

记录 owner pid/run_id/task_id/started_at，支持 lock timeout。

### R32-P1-055｜retired factor rebuild需要完整 generation reset

不仅清 tombstone，还要处理：
- old partitions
- watermark
- checkpoint
- identity
- generation
- rollback。

### R32-P1-056｜row_count/non_null_count 在不同结果形态含义不统一

统一拆成 row/cell/date/asset metrics。

### R32-P1-057｜precision parity需独立比较 Inf mask

+Inf、-Inf、NaN、finite 四类分开。

### R32-P1-058｜float32 certificate需加入组合层影响

至少：
- CS rank corr
- top-k overlap
- quantile membership
- sign flip
- threshold flip
- IC delta
- turnover delta

### R32-P1-059｜live/materialized parity必须 exact比较 key/grain/timezone

不能简单 union index后只看数值。

### R32-P1-060｜所有写/删路径做 resolve-under-root检查

不仅 factor lake，还包括：
- service root
- cache root
- staging path
- artifact path
- checkpoint path。


## 2. 时间体系：从 TradingCalendar 升级为 Temporal Authority

建议最终拆成四个权威对象：

```text
MarketCalendar
  哪些日期交易

SessionCalendar
  每个交易日有哪些 session、开闭市、午休、early close

BarClock
  合法 bar slot、bar_start/bar_end、slot offset

MarketRuleVersion
  当日适用的涨跌停/板块/ST/IPO/竞价等市场规则
```

必须 versioned：
- market
- calendar_id
- calendar_version
- source_snapshot
- timezone
- valid_from/valid_to

A股历史规则不能“用今天规则套历史”。至少覆盖：
- 主板/创业板/科创板/北交所（若支持）；
- ST/*ST；
- IPO特殊期；
- 涨跌停制度历史变化；
- 集合竞价/连续竞价；
- tick size/lot size；
- 停复牌。

美股至少覆盖：
- exchange calendar；
- DST；
- half day/early close；
- split/ticker change；
- 盘前盘后是否纳入数据语义。

每个 dataset声明 `TimestampConvention`：
- bar_start
- bar_end
- event_time
- exchange-local
- UTC/local timezone

交易日历 coverage不足必须 hard fail，不能 clamp。


## 3. Security Master：不能把 ticker 当永久证券身份

这是长期生产必备契约。

内部统一 `InstrumentId`，ticker/code只是 effective-dated属性：

```text
instrument_id
symbol
exchange/MIC
share_class
currency
valid_from
valid_to
source_snapshot
```

必须处理：
- ticker rename；
- delist/relist；
- merger；
- spin-off；
- share-class change；
- cross-market同 symbol；
- vendor symbol normalization。

Universe membership、factor lake asset key、corporate action关系都应以 canonical instrument_id 为主。

DataAccess adapter负责 vendor id → canonical id，并记录 mapping snapshot。
未来 security-master修订不能回写污染历史 as-of结果。


## 4. Identity 分层：避免过度失效与绑定不足

建议最终六层：

```text
FactorDefinitionIdentity
DependencySemanticIdentity
ExecutionSemanticIdentity
DataSnapshotIdentity
PhysicalPlanIdentity
MaterializationIdentity
```

关键规则：

1. 业务因子身份只绑定真正影响数值语义的内容。
2. `worker_count/lazy_scan/read_auto` 等物理参数若已 parity-certified，不应改变业务 factor identity。
3. 未显式配置的 default，必须把“解析后的 effective value”写入 identity。
4. Operator/field hash只计算本 plan依赖项。
5. secret轮换不改变 factor identity。
6. 数据库 host/database/dataset不能被 secret redaction一起删掉。
7. 内部存完整 SHA-256；16位前缀只做展示。
8. source snapshot、universe membership、calendar version必须独立可追踪。


## 5. Catalog：从 SQLite 文件升级为严格元数据事务层

必须完成：

- `PRAGMA foreign_keys=ON` 并验证；
- `journal_mode=WAL`；
- 明确 `synchronous` policy；
- `catalog.transaction()`；
- schema_version表；
- migration lock；
- destructive migration前 backup；
- `PRAGMA quick_check`；
- `PRAGMA foreign_key_check`；
- WAL checkpoint策略；
- migration丢弃/IGNORE记录必须有 report；
- factor retired/rebuild使用 generation语义；
- catalog snapshot/export用于灾备。

不要把 SQLite method-level lock误当 transaction isolation。


## 6. Service Job State Machine

建立唯一合法 transition graph：

```text
SUBMITTED → QUEUED
QUEUED → RUNNING | CANCELLED | REJECTED | TIMED_OUT
RUNNING → CANCELLING | SUCCEEDED | FAILED | TIMED_OUT | INTERRUPTED
CANCELLING → CANCELLED | FAILED | INTERRUPTED
```

所有 terminal状态不可被旧 worker覆盖。

必须：
- CAS update；
- idempotency tuple；
- request_digest conflict；
- durable worker lease；
- heartbeat + lease expiry；
- retry attempt独立记录；
- DRAINING状态；
- global concurrency authority；
- cancellation真正传播至 scheduler/backend；
- manifest与SQLite只留一个 state authority。


## 7. FactorId 与文件系统安全

建议区分：
- factor_id：稳定机器ID
- display_name：人类名称
- description：说明

FactorId严格验证：
- 长度超限 reject，不截断；
- 禁止 path separator、`..`、控制字符；
- Unicode NFC策略；
- case sensitivity策略；
- reserved ids；
- path `resolve()` 后必须在 allowed root；
- symlink escape防护；
- delete_files同样验证。

直接 Python API、HTTP、materializer、catalog、delete都用同一个 FactorId domain validator。


## 8. Factor Lake Schema 与物化协议

建立 `FACTOR_LAKE_SCHEMA_VERSION`。

统一定义：
- datetime dtype/timezone
- instrument_id
- value dtype
- factor_version_full/short
- data_snapshot_id
- resolved_snapshot_id
- storage_precision_policy
- is_valid
- invalid_reason
- generation_id
- schema_version

规则：
- writer只写 current schema；
- reader支持受控旧schema迁移；
- union_by_name不能替代语义 migration；
- partition schema digest；
- production generation可记录 checksum/etag；
- long/wide若都保留，语义必须对齐；
- Null/NaN/Tombstone由显式 validity contract定义。


## 9. 数值确定性：同 backend、不同机器/线程也要定义

为 operator定义 `DeterminismGrade`：
- BITWISE
- NUMERIC_TOLERANCE
- ORDER_STABLE
- RESEARCH_NONDETERMINISTIC

重点：
- parallel reduction顺序；
- BLAS/OpenMP thread count；
- SVD/eigen sign ambiguity；
- degenerate eigenvalue排序；
- PCA/ICA component canonicalization；
- HMM/cluster label permutation；
- PYTHONHASHSEED；
- stable sort/tie break；
- x86_64 vs arm64；
- FMA/SIMD；
- NumPy/SciPy版本变化。

测试：
`workers=1 vs N`
`backend threads=1 vs N`
`hardware family A vs B`
结果按 operator determinism contract验收。


## 10. Vendor Revision：不仅财报会改历史

所有 source都可能修订：
- OHLC
- volume/amount
- minute bar
- calendar
- industry
- index membership/weight
- security master
- benchmark
- corporate action

DataAccess manifest最好给：
- changed fields
- changed partitions
- changed time range
- old/new snapshot
- source_version

供 R31 change-impact DAG只重算受影响区域。

Production run必须能 pin snapshot重放。


## 11. Arrow / Parquet / Schema Round-trip

必须覆盖：
- duplicate columns；
- nullable int；
- categorical/dictionary；
- all-null column；
- empty panel；
- timezone；
- timestamp precision；
- row ordering；
- string/bytes；
- large integer；
- Arrow→Pandas→Polars→DuckDB；
- old artifact → new reader。

不能把 Parquet文件“碰巧按某顺序写出”当 ordering contract。


## 12. API / Compiler Robustness

自动因子挖掘意味着输入可能非常深/大。

Direct Python API和HTTP都要有：
- max AST nodes
- max plan nodes
- max depth
- max literal bytes
- max source refs
- max parameters
- max window
- max group cardinality estimate

并做：
- iterative traversal；
- cycle detection；
- shared graph validation；
- parser fuzz；
- typed IR fuzz；
- invalid IR fuzz；
- no hang/no process crash；
- precise error path定位。


## 13. CSE / DAG 最终形态

目标不是 shared_nodes bag，而是真正 shared DAG：

```text
shared_parent
  └─ plan_ref(shared_child)
```

要求：
- only profitable；
- nested allowed；
- no orphan；
- no dangling；
- no cycle；
- refcount来自 physical consumer graph；
- shared result immutable；
- shared task成功后才 publish cache；
- shared失败后所有 dependents fail/cancel；
- structural CSE + rolling CSE组合 golden tests。


## 14. Persistent Cache 最终收口

需要：
- bounded lock registry；
- cross-process writer安全；
- disk GC；
- namespace retention；
- corrupt pair cleanup；
- unknown namespace production禁止；
- cache metrics；
- schema/output contract validation；
- optional read-only serving mode；
- cache loss永远只影响性能，不影响 correctness。

如果多个进程共享 persistent cache，`threading.Lock` 不足。


## 15. Materializer 并发、Crash Recovery 与 Generation

建议 transaction顺序：

```text
compute
→ DQ
→ write staging/generation
→ durable data
→ durable identity/checksum
→ checkpoint
→ lineage/catalog
→ publish manifest
→ atomically switch current generation
→ watermark last
```

做 crash-point injection：
- temp write后；
- parquet replace后；
- sidecar前后；
- checkpoint前后；
- publish前后；
- watermark前后。

重启后必须明确：
- old generation仍可读；
- incomplete generation不成为 current；
- watermark不前移；
-可恢复/可重算。

推荐 factor path：
`factor_id/generation=<full_semantic_digest>/...`
并有 `_current_generation` 原子指针。

支持 rollback/retention/GC。


## 16. 环境与依赖可复现

当前 `factor_engine` 子目录未发现权威 `uv.lock/requirements.txt`；如果 repo root也没有 production lock，应建立。

区分：
- pyproject兼容范围；
- production tested lock。

Release manifest记录：
- Python
- NumPy
- Pandas
- Polars
- DuckDB
- PyArrow
- SciPy
- DataAccess
- BLAS/LAPACK
- OS/arch
- container image digest
- package git/build SHA

依赖升级必须重新跑：
- backend parity
- determinism
- performance
- artifact roundtrip。


## 17. Release / Migration / Rollback

分别版本化：
- package version
- DSL version
- operator contract version
- catalog schema version
- factor lake schema version
- DataAccess compatibility version

Release流程：
1. migration dry-run；
2. fixed canary factors old/new双跑；
3. shadow compare values/NaNs/rank/runtime/memory；
4. evidence通过；
5. production promotion；
6. current generation切换；
7. rollback可用。

Removed operator必须同步：
- parser
- aliases
- recipes
- cold-start
- mining grammar
- docs
- examples
- evidence
- benchmark corpus。


## 18. Cold-start / Mining / Recipe 同步

新增 `MiningOperatorManifest` 后，还要维护生命周期同步：

- removed operator引用 = 0；
- research-only不进入 production grammar；
- parameter role变化同步 generator；
- alias自动迁移；
- source field不可用时 recipe预检拒绝；
- recipe semantic digest；
- candidate reject reason记录；
- generator/grammar version可追踪；
- cold-start按 semantic hash去重，不只字符串去重。


## 19. Property / Fuzz / Metamorphic Tests

除了传统 unit/mutation，加入：

Property：
- group_demean group mean≈0；
- zscore满足前提时 mean≈0/std≈1；
- rank对合法正单调变换保持顺序。

Metamorphic：
- append future data → past不变；
- asset reorder → symbol-indexed结果不变；
- workers=1 vs N；
- chunk vs full；
- backend parity；
- live vs parquet reload；
- checkpoint restart vs full。

Fuzz：
- DSL parser；
- typed IR；
- planner/CSE；
- service state transitions；
- catalog corruption/lock；
- filesystem temp/disk-full。


## 20. A股 / 美股专门时间与身份测试

A股 fixture至少覆盖：
- 普通主板；
- ST/*ST；
- 创业板；
- 科创板；
- 北交所（如支持）；
- 节前最后交易日；
- 节后首日；
- 停复牌；
- 涨跌停；
- 除权除息；
- IPO特殊期；
- 11:30午休边界；
- 15:00收盘边界。

美股：
- DST前后；
- Thanksgiving half-day；
- early close；
- split；
- ticker change；
- delist；
- market holiday。

这些测试针对 calendar/session/window/universe/identity，不是策略收益。


## 21. Security / Supply Chain

基本治理：
- `pip-audit` / OSV；
- SBOM；
- secret不进 job manifest/lineage/error；
- filesystem permission；
- path traversal/symlink；
- SQL只由 typed AST/emitter生成；
- production禁 debug backend；
- environment overrides白名单；
- build provenance；
- package/container digest。


## 22. Disaster Recovery

必须定义：
- 备份：catalog、generation manifest、watermark、checkpoint metadata、semantic manifests；
- RPO/RTO；
- catalog corruption处理；
- factor partition checksum/recompute；
- cache丢失可全重建；
- checkpoint丢失安全回退重算；
- watermark可从 published manifest恢复；
- orphan staging reconcile；
- interrupted publish不切 current；
- 定期在空机器做真实 restore test；
- 时钟/NTP异常监控。


## 23. Docs / Developer Experience

建议增加：
- generated architecture truth；
- generated backend/operator counts；
- operator development contract guide；
- backend porting guide；
- data source integration guide；
- deprecation/migration guide；
- release checklist；
- incident playbook。

增加命令：
`factor-engine explain`
输出：semantic/source/physical/backend/cost/identity。

`factor-engine doctor`
检查：dependency、DataAccess、calendar、catalog、cache、disk、backend capabilities、schema compatibility。


## 24. 建议增加的 Domain Types

减少 stringly-typed bug：

```text
FactorId
InstrumentId
MarketId
CalendarId
CalendarVersion
SessionId
BarFrequency
TimestampConvention
DatasetId
SourceSnapshotId
UniverseId
UniverseSnapshotId
OperatorCanonical
FactorGenerationId
RunId
JobId
PartitionKey
StorageSchemaVersion
CatalogSchemaVersion
DSLVersion
```

不必都做复杂 class；NewType/Enum/frozen dataclass即可。重点是关键身份不再到处裸传 str。


## 25. 实施优先级

### Phase 1｜当前 P0 correctness
1. float64→float32历史降精度；
2. output grain静默丢维；
3. write_mode严格验证；
4. factor schema drift；
5. precision parity修复；
6. factor_id/path安全；
7. calendar fail-open/clamp；
8. intraday session-clock；
9. CSE orphan/nested shared；
10. rolling semantic repr移除。

### Phase 2｜Catalog + Job State
11. foreign_keys；
12. transaction；
13. migration；
14. idempotency compound key；
15. INSERT OR REPLACE移除；
16. reconciliation durable；
17. queue drain；
18. worker exception；
19. state CAS；
20. global concurrency policy。

### Phase 3｜Identity
21. dependency-scoped operator/field digest；
22. DSN/URI sanitizer；
23. source location identity；
24. actual engine/build version。

### Phase 4｜时间/确定性
25. calendar/session/version；
26. timezone/DST；
27. thread/hardware determinism。

### Phase 5｜Persistence/Generation
28. schema version；
29. generation publish；
30. rollback；
31. crash recovery；
32. DR。

### Phase 6｜Scale/Robustness
33. iterative traversal；
34. CSE/DAG线性算法；
35. persistent lock GC；
36. fuzz/property/chaos。

### Phase 7｜Ecosystem Sync
37. cold-start；
38. recipe；
39. mining grammar；
40. docs/release manifest。


## 26. R32 Evidence Artifacts

输出到：

```text
factor_engine/docs/evidence/r32/
```

至少包含：

```text
R32_HEAD.json
R32_CONFIRMED_ISSUES.csv
R32_FINAL_DISPOSITION.csv

R32_CALENDAR_COVERAGE_AUDIT.json
R32_CALENDAR_FAILURE_POLICY.json
R32_SESSION_BAR_CLOCK_TESTS.json
R32_TIMEZONE_DST_TESTS.json

R32_CSE_REACHABILITY_AUDIT.json
R32_NESTED_CSE_TESTS.json
R32_CSE_COMPLEXITY_REPORT.json

R32_CATALOG_PRAGMA_AUDIT.json
R32_CATALOG_TRANSACTION_TESTS.json
R32_CATALOG_MIGRATION_TESTS.json
R32_CATALOG_INTEGRITY_CHECK.json

R32_JOB_IDEMPOTENCY_TESTS.json
R32_JOB_STATE_MACHINE_TESTS.json
R32_QUEUE_SHUTDOWN_TESTS.json
R32_MULTIWORKER_SERVICE_AUDIT.json

R32_FACTOR_ID_SAFETY_TESTS.json
R32_PATH_CONFINEMENT_TESTS.json

R32_MATERIALIZER_PRECISION_TESTS.json
R32_MATERIALIZER_GRAIN_TESTS.json
R32_WRITE_MODE_PARITY_TESTS.json
R32_WIDE_LONG_SEMANTIC_PARITY.json
R32_FACTOR_LAKE_SCHEMA_AUDIT.json
R32_TOMBSTONE_METADATA_TESTS.json
R32_PRECISION_CERTIFICATE_VALIDATION.json

R32_FACTOR_IDENTITY_DEPENDENCY_SCOPE.json
R32_SOURCE_IDENTITY_SANITIZATION.json
R32_LINEAGE_REPRODUCIBILITY.json

R32_DETERMINISM_MATRIX.csv
R32_THREAD_COUNT_PARITY.json
R32_HARDWARE_FAMILY_PARITY.json

R32_ARROW_PARQUET_ROUNDTRIP.json
R32_SCHEMA_MIGRATION_TESTS.json

R32_PERSISTENT_CACHE_LOCK_AUDIT.json
R32_CACHE_NAMESPACE_AUDIT.json

R32_PROPERTY_TEST_RESULTS.json
R32_FUZZ_TEST_RESULTS.json
R32_CHAOS_RECOVERY_RESULTS.json

R32_RELEASE_COMPATIBILITY_MATRIX.json
R32_DR_RESTORE_TEST.json

R32_COLD_START_OPERATOR_SYNC.json
R32_RECIPE_MIGRATION_AUDIT.json
R32_DOC_TRUTH_AUDIT.json

R32_FINAL_ACCEPTANCE_REPORT.md
```

所有 artifact绑定：
- current git SHA
- package version
- operator dependency digest
- field dependency digest
- DataAccess version
- runtime dependency versions。


## 27. R32 Hard Gates

```text
R32_CALENDAR_PRODUCTION_FALLBACK_ZERO
R32_CALENDAR_OUT_OF_COVERAGE_FAILS
R32_INTRADAY_WINDOW_IS_SESSION_CLOCKED
R32_TIMEZONE_NAIVE_STRIP_ZERO

R32_CSE_ORPHAN_SHARED_ZERO
R32_CSE_DANGLING_REF_ZERO
R32_CSE_NESTED_DAG_VALID
R32_CSE_TYPED_SEMANTIC_HASH_ONLY

R32_SQLITE_FOREIGN_KEYS_ON
R32_CATALOG_TRANSACTION_INTERLEAVE_ZERO
R32_CATALOG_SCHEMA_VERSIONED
R32_CATALOG_MIGRATION_RACE_PASS
R32_PARTITION_KEY_NULL_ZERO
R32_PRODUCTION_MODE_FAIL_OPEN_ZERO

R32_JOB_IDEMPOTENCY_CROSS_USER_COLLISION_ZERO
R32_JOB_IDEMPOTENCY_CROSS_TYPE_COLLISION_ZERO
R32_IDEMPOTENCY_REPLACE_ZERO
R32_RECONCILIATION_DURABLE
R32_QUEUE_DRAIN_PASS
R32_QUEUE_ADMISSION_NONBLOCKING
R32_UNEXPECTED_JOB_EXCEPTION_WORKER_SURVIVES
R32_TERMINAL_STATE_OVERWRITE_ZERO
R32_SERVICE_GLOBAL_CONCURRENCY_POLICY_EXPLICIT

R32_MATERIALIZER_FLOAT64_HISTORY_DOWNCAST_ZERO
R32_OUTPUT_GRAIN_SILENT_DROP_ZERO
R32_WRITE_MODE_TYPO_ACCEPTANCE_ZERO
R32_WIDE_LONG_WRITE_MODE_PARITY
R32_FACTOR_LAKE_SCHEMA_SINGLE_TRUTH
R32_CHECKPOINT_SUCCESS_REQUIRES_IDENTITY_DURABLE
R32_DELETION_ONLY_METADATA_COMPLETE

R32_FACTOR_ID_TRUNCATION_ZERO
R32_FACTOR_ID_PATH_TRAVERSAL_ZERO
R32_DELETE_PATH_ESCAPE_ZERO

R32_PRECISION_NAN_MASK_MISMATCH_ZERO
R32_PRECISION_INF_MASK_MISMATCH_ZERO
R32_PRECISION_RANK_IS_CROSS_SECTIONAL

R32_SOURCE_DSN_IDENTITY_COLLISION_ZERO
R32_SOURCE_URL_SECRET_LEAK_ZERO
R32_SOURCE_PATH_IDENTITY_CANONICAL

R32_FACTOR_IDENTITY_UNRELATED_OPERATOR_INVALIDATION_ZERO
R32_FACTOR_IDENTITY_UNRELATED_FIELD_INVALIDATION_ZERO
R32_PRODUCTION_LINEAGE_FIELD_HASH_EMPTY_ZERO
R32_LINEAGE_ACTUAL_ENGINE_VERSION_PRESENT

R32_PERSISTENT_CACHE_LOCK_TABLE_BOUNDED
R32_PRODUCTION_UNKNOWN_CACHE_NAMESPACE_ZERO

R32_MAX_PLAN_DEPTH_ENFORCED
R32_GRAPH_TRAVERSAL_STACK_SAFE
R32_CRITICAL_PATH_LINEAR_COMPLEXITY
R32_DEPENDENCY_LAYER_LINEAR_COMPLEXITY

R32_FACTOR_LAKE_SCHEMA_VERSIONED
R32_OLD_ARTIFACT_READER_COMPAT_PASS
R32_GENERATION_ROLLBACK_PASS

R32_THREAD_COUNT_DETERMINISM_PASS
R32_RELEASE_ENVIRONMENT_LOCKED

R32_COLD_START_REMOVED_OPERATOR_REF_ZERO
R32_RECIPE_REMOVED_OPERATOR_REF_ZERO
R32_DOC_GENERATED_TRUTH_PASS

R32_DR_RESTORE_PASS
R32_HARD_BLOCKERS_ZERO
```


## 28. 最终验收问题

整改 AI 最后必须逐项回答：

1. 当前 HEAD 是什么？
2. R30/R31/R32 的 P0 哪些已真正进主路径？
3. 哪些只是文档或未接线模块？
4. 每个新 hard gate 的实现位置？
5. 每个 gate 的测试位置？
6. 哪些 production path仍可绕过？
7. 哪些 fallback仍在吞异常？
8. 是否还有 whole-catalog hash污染单因子 identity？
9. 是否还有任何 materialize path把 float64隐式降为float32？
10. 所有分钟 incremental 是否都走 session bar clock？
11. Job terminal state是否全部 CAS？
12. 所有 factor/cache/service路径是否做 root confinement？
13. 删除的 canonical 是否还残留在 recipe/cold-start/docs？
14. 是否能从空机器按 release manifest复现同一生产 run？
15. 是否能一键 rollback到上一 published generation？
16. DR restore是否真实演练过？
17. R32_HARD_BLOCKERS_ZERO 是否由 evidence证明，而非人工声明？


## 29. Definition of Done

R28–R32 全部落实后，FactorEngine 的“完善”应定义为：

- 算子语义正确；
- PIT/交易时钟正确；
- universe/corporate-action/security-master历史正确；
- 三后端/最佳后端执行可靠；
- DAG scheduler和资源治理可靠；
- 时间日历/session/bar clock版本化；
- Catalog/JobStore/Materializer是严格事务和状态机；
- 因子ID、source identity、semantic identity稳定；
- factor lake schema可迁移；
- incremental/cache/checkpoint可恢复；
- 同 backend不同线程/硬件有确定性合同；
- dependency环境可复现；
- release可 canary/shadow/rollback；
- 数据、catalog、generation可以 disaster restore；
- cold-start/mining/recipe/docs与真实 runtime永远同步。

最终标准不是“代码很多、测试绿了”，而是：

> 一个因子在历史回放、每日增量、多进程并发、服务重启、数据修订、依赖升级、schema迁移、机器更换和版本回滚之后，系统仍然可以证明它算的是同一个东西；如果不是同一个东西，身份和 generation 会明确变化，而不是悄悄变化。
