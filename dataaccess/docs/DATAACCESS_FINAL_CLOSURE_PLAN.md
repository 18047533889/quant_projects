# DataAccess 最终收官整改计划

## 执行状态（2026-08-08 完成）

### 第四轮最终 P0 收口（0.9.2，HEAD=`6c1453f` 之上）

14 个最终 P0 全部落地（10 项为已修复项补回归锁住，4 项为真实代码改动）：
- ✅ **Publish 契约门**：`_validate_candidate_contract`（candidate 实际数据 vs
  target 声明 schema：缺列 / 类型不符 / mixed schema → fail-closed）；manifest
  `base_dir="."`（相对路径恒有效）；symlink 复核（copytree 前拒绝）。
- ✅ **asof 统一 availability**：`read_cos_events_asof` 与 read_joined 共用
  `availability_uses_calendar/availability_strict_next`——有日历用
  `available_from <= decision`，无日历按 strict-next `<` 回退；financial
  next_trading_day 真正生效。
- ✅ **ContractIR 时间轴硬化**：event_time 只来自契约时钟列 / registry
  time_column，不再从值字段 time_role 推导（`event_time="Close"` bug 消除）；
  修 `roe` 缺失 availability → audit 重新 71 一致。
- ✅ **Manifest rowgroup 持久化**：`_save_row_groups` 改列数组字典构造（旧
  `pa.table(list_of_dicts)` 在 pyarrow 25 直接崩，sidecar 从未成功写过）；
  save/load 双侧 generation 校验。
- ✅ 回归：`tests/unit/test_final_closure_round4.py` 22 条 + 全量 `tests/` 626
  passed + ContractIR audit 71 一致。
- 并发协调：并发会话把 `us_stock_dividend` 升级为 strict-PIT；3 处旧语义测试
  改指 `us_stock_capital_split`，验证目标不变。

### 第三轮增量收口（0.9.1，HEAD=`6d66e9c` 之上）

第三层审计清单（P0 1-23 / P1 24-40）全部落地：
- ✅ **计划=执行**：`CompiledDataRequest` 冻结请求语义（plan 后改 req 不影响
  execute；plan 不再写回 request.anchor）；`snapshot_policy` latest/
  fail_if_changed/pin；组合执行器 honor `result="stream"` + 静态 universe；
  execute 把 plan 编译的 `join_specs_effective` 原样交 read_joined。
- ✅ **Snapshot 真实**：`_expand_glob_paths` 冻结精确文件清单（TOCTOU）；
  ScanHandle.collect 前 revalidate + strict fail-closed；`FileVersion` 加
  etag/version_id/content_length/last_modified（s3 对象头 best-effort）。
- ✅ **日历**：available_from 先转交易所本地时区；非交易日不生成当天开盘；
  缓存 key 含 source token（更新自动失效 / store-local）；next_day O(N) zip。
- ✅ **契约门**：required_filters/allowed_filter_values 升级为 PredicateConstraint
  （Ne/IsNotNull/OR-部分分支不再算已约束）；strict_read 完全共享 production
  fail-closed。
- ✅ **read_uri**：精确 URI physical scope + 对象 URI 不再用 pathlib + 格式匹配修正。
- ✅ **Serving**：`materialize_daily_aggregate` 一次扫描 + 行级谓词；
  Polars 按 FormatSpec 选 scan；CBO arrow/feather → pyarrow。
- ✅ **写事务**：`_dataset_mutation` 显式 PREPARED/COMMITTED/ABORTED（mutate
  前置 mark dirty；ABORTED 绝不 build fresh）。
- ✅ **Spec/IR/Storage 硬化**：TemporalJoinSpec bool/类型 fail-closed；ContractIR
  external + required_filters 完整合并 + 冲突检测；StorageSpec 非法 type 构造即抛；
  COS 解析失败 fail-closed；能力矩阵。
- ✅ **Query cache**：无权威 manifest 不缓存；plan 按 dataset 传 source_params。
- ✅ 回归：`tests/unit/test_final_closure_round3.py` 18 条 + 全量 420 passed +
  ContractIR audit 71 数据集一致。

### 第二轮收官（0.9.0，HEAD=`5073edf`）

**P0 全部落地**：
- ✅ P0-1/P0-2 统一 JoinCompiler（删除 `_join_aggregated_anchor`，聚合锚点走同一
  `_read_joined_sql`；`_effective_join_specs` 全链路共享）
- ✅ P0-3 financial seed PIT 状态（Q3 不被晚到 Q2 修订回滚）
- ✅ P0-4/5/6/7/8 Calendar：IsTradeDay 过滤 / fallback 不污染权威缓存 /
  session 前瞻 / 细粒度 availability / US early close
- ✅ P0-9/10 多时钟 TemporalAxes + pruning clock 安全 / dataset 级契约门
- ✅ P0-11..16 `is_strict_semantics()` / read_uri / event cutoff / fanout 单值 /
  PIT index（文件级剪枝、文件计数、截断 incomplete、真 schema hash）
- ✅ P0-17/23/24/25 factor matrix（gate 前置、精确投影、DISTINCT 版本检查）
- ✅ P0-18..22/35/36 执行治理（stream 直路由 + deadline 前置、governed lazy、
  Scanner pushdown、sql_relation FROM 绑定、cache 列序/默认列）
- ✅ P0-26..31 写事务（DatasetTransaction、upsert prepare-then-promote、
  publish unique_key gate + content_hash、mutation lock host 感知）
- ✅ P0-32/33/34 namespace contextvars / ContractIR issues 字段 / Metadata Plane
  epoch 感知

**P1 部分落地**：P1-14 CBO 不重复计 selectivity；P1-17 row-group 去重；
P1-21 storage backend/layout/format 分离；P1-23 PIT schema hash；P1-28 order_by。
其余 P1（CoverageMatrix 内部孔洞、CBO 校准粒度、mirror registry 并入 datasets.yaml、
QueryProvenance、InstrumentKey、incremental manifest 等）列为后续批次。

**P2 / DoD / CI**：17 条 DoD 回归（`tests/unit/test_final_closure_dod.py`）+ 永久
CI（`.github/workflows/dataaccess-final-closure.yml`）。API 收敛到
`read(DataRequest)/scan/sql/write` 的统一 compiler 列为下一批次。

**回归**：`tests/` 380 passed。

> 基线：`main` @ `5073edf`（dataaccess 本轮整改在 HEAD=`6d66e9c` 之上）。
> 目标：消灭所有"第二套语义实现"。同一个 DataRequest 无论走 minute aggregation、
> read_joined、Polars、PyArrow、stream、factor matrix 还是 remote/mirror，都必须由
> 同一个 **ContractIR → PreparedReadRequest → MetadataSnapshot → PhysicalPlan →
> 统一 Semantic Compiler → executor** 决定语义。backend 只能改变执行方式，不能改变
> 数据含义。
>
> 并发安全：本计划**只改 `dataaccess/` 树**。`factor_engine/` 全部改动归并发会话。
> `read/scan_handle.py` 已被并发会话修改，需要编辑时先 `git status` + 重读。

---

## 收口链路（唯一事实源）

```
读:
DataRequest
  → ContractIR
  → PreparedReadRequest
  → MetadataSnapshot
  → PhysicalPlan
  → 统一 Semantic Compiler (compile_temporal_join)
  → DuckDB / Polars / PyArrow Executor
  → ReadHandle
  → QueryProvenance

写:
WriteRequest
  → DatasetTransaction (PREPARED/COMMITTED/ABORTED)
  → immutable candidate generation
  → schema/key/data validation
  → manifest + metadata commit
  → atomic generation switch
```

---

## P0 — 必须修完才能宣布正确性收官

### 1. 统一 JoinCompiler（P0-1 / P0-2）
- 删除 `PhysicalPlanExecutor._join_aggregated_anchor` 的第二套 join 语义。
- aggregation 结果 → `DerivedRelationSource` → **同一个** `compile_temporal_join`。
- `compile_temporal_join` 支持 `PhysicalSource / DerivedRelationSource / MaterializedSource`。
- 生成 `EffectiveJoinSpec`：`SemanticField 默认 → 同 dataset 字段语义合并 → COS Contract
  默认 → request.joins/join_specs 覆盖`。
- PIT validator / PhysicalPlan / read_joined / cache key / lineage / explain 全部消费
  EffectiveJoinSpec。
- 验收：`minute aggregation + financial PIT + industry` 结果 == 手工 aggregate 再走完整
  read_joined，逐行逐列一致。

### 2. Financial seed / latest_period（P0-3）
- seed 必须计算"截至 start 的 PIT financial state"（同 instrument+period 取最新 revision，
  再在所有可见 period 中取 max(period)），而不是"start 前最后一条 event"。
- 修复 Q3 已发布后被晚到 Q2 revision 回滚的问题。专项回归测试。

### 3. Calendar 正确性（P0-4 / P0-5 / P0-6 / P0-7 / P0-8）
- Calendar loader 过滤 `IsTradeDay`/`trade_date` flag（不再把自然日当交易日）。
- Calendar 用 `mode="dimension"` 读（STATIC 契约 production/strict 下不被拒）。
- production 真实 calendar 不可用 → **fail closed**，不静默退回 business day。
- 全局 `_calendars[market]` 缓存 key 至少含 `market + calendar_source + snapshot/version`，
  fallback calendar 不得污染 authoritative calendar。
- `_session_avail_sql` lookahead ≥1 个交易日（映射窗口延伸到 query end 之后的下一交易日）。
- availability 拆分：`same_instant / next_bar / next_session_open / next_trading_day /
  after_close_next_open / effective_date_only` + `availability_latency`。
- US early close（如 13:00）作为 date-specific session override 进入 engine。

### 4. 多时钟时间轴 + dataset 级契约门（P0-9 / P0-10）
- ContractIR 增加 `TemporalAxes`（partition_time / event_time / knowledge_time /
  effective_time / period_time / decision_time / storage_timezone / semantic_timezone）。
- PreparedRead 明确 `predicate_clock / pruning_clock / join_clock`；manifest 无对应 clock
  统计 → 禁止基于另一根时间轴 prune（宁可多扫，不能 false negative）。
- Dataset 级 required filters（timeframe / IndustrySource / IndexSymbol）**无论用户选哪个
  字段都强制执行** —— 两层门禁 `DatasetContractGate` + `FieldContractGate`。

### 5. Strict 语义统一 + read_uri + cutoff + fanout + PIT index（P0-11..16）
- 单一 `is_strict_semantics() = production OR strict_read`，全链路统一。
- `read_uri` production/strict 必须唯一反查已登记 dataset（不能用 prefix 根绕过）。
- effective-event 硬 cutoff：无 end 的 Dividend/split 查询不得读未来生效事件。
- fanout 单值证明：`IndustrySource IN ('sw_l1','sw_l2')` 仍判 fanout。
- PITEventIndex：文件级 min/max prune（不 O(records)）、真 schema hash、`limit` 构建
  `complete=False`、indexed_file_count 必须等于源文件数。

### 6. Factor matrix 正确性（P0-17 / P0-23 / P0-24 / P0-25）
- 一致性 gate 移到 matrix route **之前**（版本 / data_snapshot / universe）。
- matrix 精确投影请求的 fids，缺一个 → `MatrixCoverageMiss`。
- `FactorArtifactManifest`（factor_version / data_snapshot_id / universe_id / date 范围 /
  row_count / schema_hash / source_lineage_hash）取代 `limit=1` 行探测。

### 7. 执行治理（P0-18..22 / P0-35 / P0-36）
- `result="stream"` 在 execution 前直接路由到真正 reader，禁止"先 materialize 再 stream"。
- stream deadline 作用在第一批之前；generator `finally` 里 interrupt/close cursor、
  归还连接、结束 watchdog。
- `GovernedLazyHandle`：production 不暴露 raw LazyFrame，只有 `collect()/stream()` 受控
  终点；research 用 `unsafe_scan_polars()`。
- PyArrow executor 改 `Scanner`：projection / filter / partition pruning pushdown +
  真实 time_range。
- sql_relation production 走 `SafeSQLRequest → read_datasets → scoped temp views →
  rewrite query`（token sandbox 只是辅助，不当作数据边界）。
- QueryCache key：columns 不排序（[A,B] != [B,A]）、default columns 显式展开、含 ordering。

### 8. 写事务原子性（P0-26..31）
- overwrite → `.generation_uuid/` immutable candidate → validate → manifest → atomic
  generation switch（禁止 clear-dir-then-write）。
- `DatasetTransaction` PREPARED/COMMITTED/ABORTED：metadata 只在 COMMIT 前进，
  ABORTED 不进位（修掉 finally 无条件 bump）。
- 多分区 upsert → transaction manifest 全成功才 commit。
- publish 只发布 immutable staging generation（capture source snapshot + lock + copy +
  verify unchanged）。
- publish 强校验：schema hash / file inventory / partition inventory / row count /
  primary-key count / manifest hash / source generation；关键 dataset 加 column checksum。
- mutation lock：host-aware stale reclaim（跨机器不得用本地 os.kill 回收活锁）。

### 9. Namespace / ContractIR / Metadata Plane（P0-32..34）
- Namespace → `contextvars.ContextVar`，enter 存 token / exit restore；registry 保留
  namespace token，path resolve 时才绑定。
- **ContractIR `issues: list[str]` 字段缺失是直接 bug**（`audit()` 引用 `d.issues` 会
  AttributeError）→ 补字段；`store.contract_ir().audit()` 必须在完整 registry 上通过。
- Metadata Plane 改 immutable `MetadataSnapshot(dataset, params, source_epoch)`，accessor
  检测 epoch 变化即清 derived metadata。

---

## P1 — 元数据 / CBO / COS 收口

- CoverageMatrix（expected trading days、internal hole → partial）。
- 物理源感知 CBO：`logical_source=COS` vs `physical_source=local_mirror/s3_httpfs/cli_cache`；
  不重复计 selectivity；`calibrated_score` 换成真实 elapsed/memory/network/spill 预测；
  校准粒度 dataset+backend+engine+query shape+projection bucket+selectivity bucket+remote/local。
- mirror registry 并入 datasets.yaml storage 声明（删除硬编码 DATASET_MIRROR_REGISTRY）。
- mirror freshness 在 deep verify / publish / suspected corruption 时重算 checksum；
  full cos sync 后为每个文件生成 inventory sidecar。
- PITEventIndex 文件级 serving 结构（file_path/min_filing/max_filing/timeframe set/bloom）。
- FactorArtifactCatalog（factor_id/version/coverage/snapshot/universe/frequency/schema/
  storage generation/lineage）。
- QueryProvenance：request_hash / filter_ast_hash / effective_join_spec_hash /
  physical_plan_hash / contract_ir_hash / semantic_catalog_hash / calendar_hash /
  universe_snapshot / normalization_version / engine / engine_version / query_budget /
  code commit。
- remote object 版本（etag / version_id / content_length / last_modified）。
- 审计升级：query_id / request_id / transaction_id / operator / namespace / host / pid /
  snapshot / plan hash / 成功失败 / runtime / bytes / rows；redaction allowlist。
- 输出顺序契约：unordered raw read 文档化；`order_by=[time,instrument]` 显式请求，
  进 cache key / lineage / plan。
- InstrumentKey（market, security_id）PIT TickerMap，Symbol/Ticker 只作显示 key。
- schema evolution：required/nullable/safe_cast/enum/domain/timezone/primary_key/unique_key/
  partition schema/schema_version/compatible_from/deprecated/renamed/migration；key
  uniqueness 进 publish gate。
- upsert → DuckDB COW merge（不整 partition→pandas→merge）。
- incremental manifest mutation（ADD/REMOVE/REPLACE partition，commit 时 O(changed files)）。

---

## P2 — 平台整理

- 统一公开 API：`store.read(DataRequest)` / `store.scan(DataRequest)` /
  `store.sql(SafeSQLRequest)` / `store.write(WriteRequest)`；其余入口保留 compatibility
  wrapper 全部 lower 到同一 compiler。
- empty policy 统一、deterministic ordering 契约。

---

## 落地顺序（依赖优先）

1. **JoinCompiler 统一**（P0-1/P0-2）— 依赖面最大，先做。
2. **ContractIR 修复**（P0-33 issues 字段 + TemporalAxes + EffectiveJoinSpec 承载）。
3. **Financial seed**（P0-3）。
4. **Calendar 正确性**（P0-4..8）。
5. **Temporal axes + dataset gates**（P0-9/P0-10）。
6. **Strict/PIT/cutoff/fanout**（P0-11..16）。
7. **Factor matrix**（P0-17/23/24/25）。
8. **执行治理**（P0-18..22/35/36）。
9. **写事务**（P0-26..31）。
10. **Namespace/Metadata Plane**（P0-32/34）。
11. **P1**（元数据/CBO/COS）。
12. **P2 + DoD 回归 + CI**。

## 并发安全
- 只改 `dataaccess/`；`factor_engine/` 不碰。
- `read/scan_handle.py`（并发会话已改）需要编辑时先重读。
- 测试：`tests/` 属本仓库，可安全更新。
